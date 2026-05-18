import requests
from urllib3.util import Retry


def create_session_with_pooling(
    pool_connections: int = 10,
    pool_maxsize: int = 20,
    max_retries: int = 3,
) -> requests.Session:
    """Create a requests.Session with connection pooling and idempotent-method-only retries.

    Retries are limited to HTTP methods that are idempotent by nature
    (GET, HEAD, PUT, DELETE, OPTIONS, TRACE) to avoid unintentionally
    duplicating non-idempotent requests (POST, PATCH).
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        status=max_retries,
        other=0,
        allowed_methods=frozenset({"GET", "HEAD", "PUT", "DELETE", "OPTIONS", "TRACE"}),
        backoff_factor=0.5,
        status_forcelist=frozenset({429, 500, 502, 503, 504}),
        raise_on_status=True,
    )
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
        max_retries=retry_strategy,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
