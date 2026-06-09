"""Input validation utilities for RecRAG APIs."""

# Constants
MAX_QUERY_LENGTH = 4096


def validate_query(query: str) -> str:
    """Validate and sanitize a query string.

    Args:
        query: The query string to validate.

    Returns:
        The validated query string.

    Raises:
        ValueError: If the query is invalid.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"Query too long: {len(query)} chars (max {MAX_QUERY_LENGTH})")

    return query.strip()
