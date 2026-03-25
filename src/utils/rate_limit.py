"""Rate limit utilities for external API calls."""

import json
import time
from pathlib import Path

QUOTA_FILE = Path(".api_quota.json")
WINDOW_SECONDS = 60
MAX_REQUESTS = 35


def _load_quota() -> dict[str, list[float]]:
    if QUOTA_FILE.exists():
        try:
            with open(QUOTA_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_quota(data: dict[str, list[float]]) -> None:
    with open(QUOTA_FILE, "w") as f:
        json.dump(data, f)


def check_rate_limit(service: str) -> None:
    """Check if the service has exceeded the rate limit in the current window."""
    data = _load_quota()
    now = time.time()

    timestamps = data.get(service, [])
    if not isinstance(timestamps, list):
        timestamps = []

    valid_timestamps = [ts for ts in timestamps if now - ts < WINDOW_SECONDS]

    if len(valid_timestamps) >= MAX_REQUESTS:
        oldest = min(valid_timestamps)
        time_left = WINDOW_SECONDS - (now - oldest)
        raise RuntimeError(
            f"Rate limit reached for '{service}' ({MAX_REQUESTS} req/{WINDOW_SECONDS}s). "
            f"Try again in {time_left:.1f} seconds."
        )


def record_success(service: str) -> None:
    """Record a successful request for the service."""
    data = _load_quota()
    now = time.time()

    timestamps = data.get(service, [])
    if not isinstance(timestamps, list):
        timestamps = []

    valid_timestamps = [ts for ts in timestamps if now - ts < WINDOW_SECONDS]
    valid_timestamps.append(now)

    data[service] = valid_timestamps
    _save_quota(data)
