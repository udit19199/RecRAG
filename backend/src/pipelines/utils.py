import hashlib
from pathlib import Path

def _compute_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of a file for change detection."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
