import logging
import sys
import threading
import time
from pathlib import Path
from threading import Lock
from typing import Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from config import (
    find_config_path,
    get_ingestion_dir,
    get_storage_dir,
    load_config,
)
from pipelines import run_ingestion
from utils.status import now, write_status

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEBOUNCE_SECONDS = 5.0
SUPPORTED_EXTENSIONS = {".pdf"}


class IngestionWatcher(FileSystemEventHandler):
    """File watcher with debouncing to batch multiple file events."""

    def __init__(
        self,
        watch_dir: Path,
        storage_dir: Path,
        config_path: Path,
        debounce_seconds: float = DEBOUNCE_SECONDS,
        supported_extensions: set[str] | None = None,
    ):
        self.watch_dir = watch_dir
        self.storage_dir = storage_dir
        self.config_path = config_path
        self.debounce_seconds = debounce_seconds
        self.supported_extensions = supported_extensions or SUPPORTED_EXTENSIONS.copy()

        self.lock = Lock()
        self.is_processing = False
        self.pending_files: set[str] = set()
        self.debounce_timer: Optional[threading.Timer] = None

    def _schedule_ingestion(self, file_path: str) -> None:
        """Schedule ingestion with debouncing.

        Adds the file to the pending set, cancels any existing timer, and starts
        a new timer.  Only after no new events occur for ``debounce_seconds``
        will ingestion be triggered.

        Args:
            file_path: Path to the file that triggered the event.
        """
        with self.lock:
            self.pending_files.add(file_path)
            pending_count = len(self.pending_files)

            if self.debounce_timer is not None:
                self.debounce_timer.cancel()
                self.debounce_timer = None

            self.debounce_timer = threading.Timer(
                self.debounce_seconds, self._trigger_ingestion
            )
            self.debounce_timer.start()

        logger.info(
            "File event: %s. Waiting %.1fs for more files (%d pending)",
            Path(file_path).name,
            self.debounce_seconds,
            pending_count,
        )

    def _trigger_ingestion(self) -> None:
        """Trigger ingestion after debounce period completes."""
        with self.lock:
            if not self.pending_files:
                return
            files_count = len(self.pending_files)
            self.pending_files.clear()
            self.debounce_timer = None

        logger.info("Debounce complete. Processing %d file(s)", files_count)
        self._run_ingestion_with_lock()

    def _run_ingestion_with_lock(self) -> None:
        """Run ingestion with proper locking and status tracking."""
        with self.lock:
            if self.is_processing:
                logger.info("Already processing, skipping trigger")
                return
            self.is_processing = True

        try:
            started_at = now()
            write_status(self.storage_dir, "processing", started_at=started_at)
            logger.info("Starting incremental ingestion at %s", started_at)

            results = run_ingestion(self.config_path, incremental=True)

            completed_at = now()
            docs_processed = results.get("documents", 0)

            if docs_processed == 0:
                logger.info("No new or changed files at %s", completed_at)
            else:
                logger.info("Ingestion complete at %s", completed_at)

            write_status(
                self.storage_dir,
                "complete",
                started_at=started_at,
                completed_at=completed_at,
                files_processed=docs_processed,
            )
            logger.info(
                "Documents: %d, Chunks: %d",
                docs_processed,
                results.get("chunks", 0),
            )

        except Exception as e:
            completed_at = now()
            logger.error("Ingestion failed: %s", e)
            write_status(
                self.storage_dir,
                "error",
                completed_at=completed_at,
                error_message=str(e),
            )

        finally:
            with self.lock:
                self.is_processing = False

    def on_created(self, event) -> None:  # type: ignore[override]
        """Handle file creation events with debouncing."""
        if event.is_directory:
            return
        src_path = str(event.src_path)
        if Path(src_path).suffix.lower() in self.supported_extensions:
            self._schedule_ingestion(src_path)

    def on_modified(self, event) -> None:  # type: ignore[override]
        """Handle file modification events with debouncing."""
        if event.is_directory:
            return
        src_path = str(event.src_path)
        if Path(src_path).suffix.lower() in self.supported_extensions:
            self._schedule_ingestion(src_path)

    def start(self) -> None:
        """Start watching the directory for file changes."""
        logger.info("Starting ingestion watcher on %s", self.watch_dir)
        logger.info("Debounce period: %.1f seconds", self.debounce_seconds)
        self.watch_dir.mkdir(parents=True, exist_ok=True)

        observer = Observer()
        observer.schedule(self, str(self.watch_dir), recursive=False)
        observer.start()

        logger.info("Ingestion watcher started. Waiting for files...")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down watcher...")
            if self.debounce_timer is not None:
                self.debounce_timer.cancel()
            observer.stop()
        observer.join()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Watch folder for changes and run ingestion"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to configuration file (default: config.toml in project root)",
    )

    args = parser.parse_args()

    config_path = find_config_path(args.config)
    config = load_config(config_path)

    watch_dir = get_ingestion_dir(config, config_path)
    storage_dir = get_storage_dir(config, config_path)
    storage_dir.mkdir(parents=True, exist_ok=True)

    watcher = IngestionWatcher(
        watch_dir=watch_dir,
        storage_dir=storage_dir,
        config_path=config_path,
    )
    watcher.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
