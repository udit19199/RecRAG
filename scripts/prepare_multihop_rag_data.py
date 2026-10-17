"""Download the pinned MultiHop-RAG questions and full corpus locally."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

from dataset import MULTIHOP_RAG_REVISION

DATASET_DIR = Path(__file__).resolve().parents[1] / "datasets/multihop_rag"
FILES = ("MultiHopRAG.json", "corpus.json")


def main() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        path = hf_hub_download(
            repo_id="yixuantt/MultiHopRAG",
            repo_type="dataset",
            filename=filename,
            revision=MULTIHOP_RAG_REVISION,
            local_dir=DATASET_DIR,
        )
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
