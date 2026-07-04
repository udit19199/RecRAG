"""FiNER-139 graph-construction entity-recognition benchmark.

Compares the five graph-construction methods from the GraphRAG analysis doc on
how well they recognise entities in the FiNER-139 dataset, scored as
type-agnostic span detection (precision/recall/F1).

Heavy dependencies (``datasets``, ``spacy``) are imported lazily so importing
this package never pulls them; they are only needed when a benchmark runs.
Install them via the optional extra: ``pip install -e ".[experiments]"`` and
``python -m spacy download en_core_web_sm``.
"""
