"""RecRAG API package.

This package contains the FastAPI services for the RecRAG system:
- ingestion: API for uploading documents and checking status
- retrieval: API for querying documents with RAG
"""

from fastapi import APIRouter

# Create routers for each service
# These can be imported and mounted in a unified API if needed
