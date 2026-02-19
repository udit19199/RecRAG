"""Shared utilities for adapter implementations."""

import requests


def create_session_with_pooling(
    pool_connections: int = 10,
    pool_maxsize: int = 20,
    max_retries: int = 3,
) -> requests.Session:
    """Create a requests Session with connection pooling.

    Args:
        pool_connections: Number of connection pools to cache.
        pool_maxsize: Maximum number of connections to save per pool.
        max_retries: Maximum number of retries per connection.

    Returns:
        Configured requests Session.
    """
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
        max_retries=max_retries,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
