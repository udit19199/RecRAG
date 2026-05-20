"""Input validation utilities for RecRAG APIs."""

import re
from pathlib import Path

# Constants
MAX_QUERY_LENGTH = 4096
MAX_FILENAME_LENGTH = 255
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB
ALLOWED_FILENAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.\s]+$")


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


def validate_filename(filename: str) -> str:
    """Validate and sanitize a filename.

    Args:
        filename: The filename to validate.

    Returns:
        The validated filename.

    Raises:
        ValueError: If the filename is invalid.
    """
    if not filename:
        raise ValueError("Filename is required")

    if len(filename) > MAX_FILENAME_LENGTH:
        raise ValueError(
            f"Filename too long: {len(filename)} chars (max {MAX_FILENAME_LENGTH})"
        )

    # Check for path traversal
    safe_name = Path(filename).name
    if safe_name != filename:
        raise ValueError("Filename cannot contain path separators")

    # Check for allowed characters
    if not ALLOWED_FILENAME_PATTERN.match(safe_name):
        raise ValueError(
            "Filename can only contain alphanumeric characters, hyphens, "
            "underscores, spaces, and dots"
        )

    return safe_name


def validate_file_size(content: bytes) -> None:
    """Validate file size does not exceed maximum.

    Args:
        content: The file content in bytes.

    Raises:
        ValueError: If the file is too large.
    """
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File too large: {len(content) / (1024 * 1024):.1f} MB "
            f"(max {MAX_FILE_SIZE_BYTES / (1024 * 1024):.0f} MB)"
        )
