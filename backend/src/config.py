import os
import re
from pathlib import Path
from typing import Any

import toml


def resolve_path(path: str | Path, config_path: Path) -> Path:
    """Resolve a path relative to the config file's parent directory.

    If the path is absolute, return it as-is.
    If the path is relative, resolve it relative to the config file's parent.

    Args:
        path: The path to resolve (absolute or relative).
        config_path: Path to the configuration file.

    Returns:
        Resolved absolute path.
    """
    path = Path(path)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def find_config_path(explicit_path: Path | None = None) -> Path:
    """Centralized config path resolution."""
    if explicit_path:
        return explicit_path
    # Try common locations
    candidates = [
        Path("config.toml"),
        Path(__file__).parent.parent.parent / "config.toml",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("config.toml not found")


def load_config(config_path: Path = Path("config.toml")) -> dict[str, Any]:
    """Load configuration from TOML file with environment variable substitution.

    Supports ${ENV_VAR} and ${ENV_VAR:-default} syntax.

    Args:
        config_path: Path to the TOML configuration file.

    Returns:
        Dictionary with configuration values.
    """
    config = toml.load(config_path)
    return _substitute_env_vars(config)


def _substitute_env_vars(value: Any) -> Any:
    """Recursively substitute environment variables in config values."""
    if isinstance(value, str):
        return _substitute_string(value)
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    return value


def _substitute_string(value: str) -> str:
    """Substitute environment variables in a string.

    Supports ${VAR} and ${VAR:-default} syntax.
    """
    pattern = r"\$\{([^}:]+)(?::-([^}]*))?\}"

    def replacer(match):
        var_name = match.group(1)
        default = match.group(2) or ""
        return os.environ.get(var_name, default)

    return re.sub(pattern, replacer, value)


def get_config_value(config: dict, key_path: str, default: Any = None) -> Any:
    """Get a nested config value using dot notation.

    Args:
        config: Configuration dictionary.
        key_path: Dot-separated path (e.g., "embedding.openai.model").
        default: Default value if key not found.

    Returns:
        The config value or default.
    """
    keys = key_path.split(".")
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def get_storage_dir(config: dict, config_path: Path) -> Path:
    """Get the storage directory path from configuration.

    Args:
        config: Configuration dictionary.
        config_path: Path to the configuration file.

    Returns:
        Resolved absolute path to storage directory.
    """
    storage_dir = config.get("storage", {}).get("directory", "storage")
    return resolve_path(storage_dir, config_path)


def get_ingestion_dir(config: dict, config_path: Path) -> Path:
    """Get the ingestion directory path from configuration.

    Args:
        config: Configuration dictionary.
        config_path: Path to the configuration file.

    Returns:
        Resolved absolute path to ingestion directory.
    """
    ingestion_dir = config.get("ingestion", {}).get("directory", "data/pdfs")
    return resolve_path(ingestion_dir, config_path)
