import argparse
import json
import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import load_config, resolve_path
from pipelines import run_ingestion

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


DEBOUNCE_SECONDS = 5.0


class IngestionWatcher(FileSystemEventHandler):
    """File watcher with debouncing to batch multiple file events.

    Instead of triggering ingestion on every file event, this watcher
    collects events and waits for a quiet period (debounce_seconds)
    before running ingestion. This prevents redundant processing when
    multiple files are copied to the watch directory simultaneously.

    Attributes:
        watch_dir: Directory to watch for file changes.
        status_file: Path to the status JSON file.
        config_path: Path to the configuration file.
        debounce_seconds: Seconds to wait after last event before processing.
    """

    def __init__(
        self,
        watch_dir: Path,
        status_file: Path,
        config_path: Path,
        debounce_seconds: float = DEBOUNCE_SECONDS,
    ):
        self.watch_dir = watch_dir
        self.status_file = status_file
        self.config_path = config_path
        self.debounce_seconds = debounce_seconds

        self.lock = Lock()
        self.is_processing = False

        self.pending_files: set[str] = set()
        self.debounce_timer: Optional[threading.Timer] = None

    def write_status(
        self,
        status: str,
        started_at: Optional[str] = None,
        completed_at: Optional[str] = None,
        files_processed: int = 0,
        error_message: Optional[str] = None,
    ):
        """Write status to JSON file for UI consumption."""
        status_data: dict = {
            "status": status,
            "started_at": started_at,
            "completed_at": completed_at,
            "files_processed": files_processed,
            "error_message": error_message,
        }
        with open(self.status_file, "w") as f:
            json.dump(status_data, f, indent=2)

    def get_status(self) -> dict:
        """Read status from JSON file."""
        if not self.status_file.exists():
            return {"status": "idle"}
        try:
            with open(self.status_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"status": "idle"}

    def _schedule_ingestion(self, file_path: str):
        """Schedule ingestion with debouncing.

        When a file event occurs, this method:
        1. Adds the file to the pending set
        2. Cancels any existing timer
        3. Starts a new timer for debounce_seconds

        Only after no new events occur for debounce_seconds will
        the actual ingestion be triggered.

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
            f"File event: {Path(file_path).name}. "
            f"Waiting {self.debounce_seconds}s for more files "
            f"({pending_count} pending)"
        )

    def _trigger_ingestion(self):
        """Trigger ingestion after debounce period completes."""
        with self.lock:
            if not self.pending_files:
                return
            files_count = len(self.pending_files)
            self.pending_files.clear()
            self.debounce_timer = None

        logger.info(f"Debounce complete. Processing {files_count} file(s)")
        self.run_ingestion_safe()

    def run_ingestion_safe(self):
        """Run ingestion with proper locking and status tracking."""
        with self.lock:
            if self.is_processing:
                logger.info("Already processing, skipping trigger")
                return
            self.is_processing = True

        try:
            started_at = datetime.now().isoformat()
            self.write_status(
                "processing",
                started_at=started_at,
            )
            logger.info(f"Starting ingestion at {started_at}")

            results = run_ingestion(self.config_path, force=False)

            completed_at = datetime.now().isoformat()
            self.write_status(
                "complete",
                started_at=started_at,
                completed_at=completed_at,
                files_processed=results.get("documents", 0),
            )
            logger.info(f"Ingestion complete at {completed_at}")
            logger.info(
                f"Documents: {results['documents']}, Chunks: {results['chunks']}"
            )

        except Exception as e:
            completed_at = datetime.now().isoformat()
            logger.error(f"Ingestion failed: {e}")
            self.write_status(
                "error",
                completed_at=completed_at,
                error_message=str(e),
            )

        finally:
            with self.lock:
                self.is_processing = False

    def on_created(self, event):
        """Handle file creation events with debouncing."""
        if event.is_directory:
            return
        if event.src_path.endswith(".pdf"):
            self._schedule_ingestion(event.src_path)

    def on_modified(self, event):
        """Handle file modification events with debouncing."""
        if event.is_directory:
            return
        if event.src_path.endswith(".pdf"):
            self._schedule_ingestion(event.src_path)

    def start(self):
        """Start watching the directory for file changes."""
        logger.info(f"Starting ingestion watcher on {self.watch_dir}")
        logger.info(f"Debounce period: {self.debounce_seconds} seconds")
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


def main():
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

    if args.config:
        config_path = args.config
    else:
        config_path = Path(__file__).parent.parent / "config.toml"
        if not config_path.exists():
            config_path = Path("config.toml")

    config = load_config(config_path)

    watch_dir = resolve_path(
        config.get("ingestion", {}).get("directory", "data/pdfs"), config_path
    )
    storage_dir = resolve_path(
        config.get("storage", {}).get("directory", "storage"), config_path
    )
    status_file = storage_dir / "ingestion_status.json"

    status_file.parent.mkdir(parents=True, exist_ok=True)

    watcher = IngestionWatcher(
        watch_dir=watch_dir,
        status_file=status_file,
        config_path=config_path,
    )
    watcher.start()


if __name__ == "__main__":
    sys.exit(main())
