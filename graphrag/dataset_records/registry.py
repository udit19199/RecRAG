from __future__ import annotations

from .base import DatasetAdapter
from .hotpotqa import adapter as hotpotqa
from .natural_questions import adapter as natural_questions
from .two_wiki_multihopqa import adapter as two_wiki_multihopqa

DATASET_ADAPTERS: list[DatasetAdapter] = [
    hotpotqa,
    two_wiki_multihopqa,
    natural_questions,
]


def get_adapter(name: str) -> DatasetAdapter:
    for adapter in DATASET_ADAPTERS:
        if adapter.name == name:
            return adapter
    raise ValueError(f"Unknown dataset: {name}")
