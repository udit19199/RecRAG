"""Use the pipeline's default schema inference and extraction prompt."""

from neo4j_graphrag.generation.prompts import ERExtractionTemplate

SCHEMA = None
EXTRACTION_PROMPT = ERExtractionTemplate.DEFAULT_TEMPLATE
