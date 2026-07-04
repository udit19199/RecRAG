"""Decoupled experiments router(s) mounted in the orchestrator service.

Kept separate from the recommendation flow. Heavy engine imports
(``experiments.finer139``) happen lazily inside the background task so importing
this package never pulls ``datasets``/``spacy``.
"""
