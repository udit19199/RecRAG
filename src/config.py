import os
import re
from pathlib import Path
from typing import Any

import toml


def resolve_path(path: str | Path, config_path: Path) -> Path:
    """Resolve path relative to config file; return absolute paths unchanged."""
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
    """Load TOML config with ${VAR:-default} environment variable substitution."""
    config = toml.load(config_path)
    if not isinstance(config, dict):
        raise ValueError(
            f"Config file must contain a table at top level: {config_path}"
        )
    return _substitute_env_vars(config)


def _substitute_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return _substitute_string(value)
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    return value


def _substitute_string(value: str) -> str:
    pattern = r"\$\{([^}:]+)(?::-([^}]*))?\}"

    def replacer(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default = match.group(2) or ""
        return os.environ.get(var_name, default)

    return re.sub(pattern, replacer, value)


def get_config_value(config: dict, key_path: str, default: Any = None) -> Any:
    """Get a nested config value using dot notation (e.g. "embedding.model")."""
    keys = key_path.split(".")
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def get_storage_dir(config: dict, config_path: Path) -> Path:
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


def get_frontend_origins(config: dict[str, Any]) -> list[str]:
    """Return the configured frontend origins for CORS."""
    origins = config.get("frontend", {}).get("origins", ["http://localhost:3000"])
    if isinstance(origins, str):
        return [origin.strip() for origin in origins.split(",") if origin.strip()]
    return [str(origin).strip() for origin in origins if str(origin).strip()]
