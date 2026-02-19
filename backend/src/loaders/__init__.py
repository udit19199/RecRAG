from pathlib import Path
from typing import Any

from core import BaseDocumentLoader


def create_loader(
    provider: str,
    directory: Path | str,
    **kwargs: Any,
) -> BaseDocumentLoader:
    """Create a document loader based on provider.

    Args:
        provider: Provider name ("pdf" or "auto")
        directory: Directory to load documents from
        **kwargs: Additional provider-specific parameters

    Returns:
        BaseDocumentLoader instance
    """
    if provider in ("pdf", "auto"):
        from core import DocumentLoader

        return DocumentLoader(directory, **kwargs)
    else:
        raise ValueError(f"Unknown loader provider: {provider}")


def get_loader_for_file(file_path: Path | str) -> BaseDocumentLoader:
    """Get the appropriate loader for a file based on extension.

    Args:
        file_path: Path to the file

    Returns:
        BaseDocumentLoader instance appropriate for the file type
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        from core import DocumentLoader

        return DocumentLoader(file_path.parent)

    raise ValueError(f"No loader available for file type: {suffix}")
