import argparse
import sys
from pathlib import Path

from config import find_config_path
from pipelines import run_ingestion


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into the vector store"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to configuration file (default: config.toml in project root)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing of all documents",
    )

    args = parser.parse_args()

    try:
        config_path = find_config_path(args.config)
        results = run_ingestion(config_path, force=args.force)
        print("\n=== Ingestion Complete ===")
        print(f"Documents processed: {results['documents']}")
        print(f"Chunks created: {results['chunks']}")
        print(f"Embeddings generated: {results['embeddings']}")
        print(f"Total vectors in store: {results['total_vectors']}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
