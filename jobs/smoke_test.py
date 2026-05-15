#!/usr/bin/env python3
"""Smoke tests for RecRAG post-deployment verification.

Usage:
    python jobs/smoke_test.py --base-url http://localhost

Tests:
    1. Health endpoints for both APIs
    2. Upload a test PDF
    3. Wait for ingestion to complete
    4. Query the uploaded document
    5. Verify response contains expected content
"""

import argparse
import sys
import time
from pathlib import Path

import requests

TIMEOUT = 300  # 5 minutes max for ingestion
POLL_INTERVAL = 5  # seconds between status checks


def check_health(base_url: str, port: int, service_name: str) -> bool:
    """Check if a service health endpoint responds."""
    url = f"{base_url}:{port}/health"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        print(f"[OK] {service_name} health check passed")
        return True
    except requests.RequestException as e:
        print(f"[FAIL] {service_name} health check failed: {e}")
        return False


def upload_pdf(base_url: str, pdf_path: Path) -> str | None:
    """Upload a PDF and return the task ID, or None on failure."""
    url = f"{base_url}:8001/upload"
    try:
        with open(pdf_path, "rb") as f:
            resp = requests.post(url, files={"file": (pdf_path.name, f, "application/pdf")}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        task_id = data.get("task_id")
        print(f"[OK] PDF uploaded, task_id={task_id}")
        return task_id
    except requests.RequestException as e:
        print(f"[FAIL] PDF upload failed: {e}")
        return None


def wait_for_ingestion(base_url: str, task_id: str) -> bool:
    """Poll ingestion status until complete or timeout."""
    url = f"{base_url}:8001/status/{task_id}"
    start = time.time()

    while time.time() - start < TIMEOUT:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            status = resp.json()
            state = status.get("state", "unknown")

            if state == "completed":
                print(f"[OK] Ingestion completed in {time.time() - start:.1f}s")
                return True
            if state == "failed":
                print(f"[FAIL] Ingestion failed: {status.get('error', 'unknown error')}")
                return False

            print(f"  Ingestion state: {state} ({time.time() - start:.0f}s elapsed)")
            time.sleep(POLL_INTERVAL)
        except requests.RequestException as e:
            print(f"[WARN] Status check failed: {e}")
            time.sleep(POLL_INTERVAL)

    print(f"[FAIL] Ingestion timed out after {TIMEOUT}s")
    return False


def query_rag(base_url: str, question: str) -> bool:
    """Send a query to the retrieval API and verify response."""
    url = f"{base_url}:8000/query"
    try:
        resp = requests.post(url, json={"query": question}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer", "")

        if answer:
            print(f"[OK] Query returned answer ({len(answer)} chars)")
            return True
        print(f"[FAIL] Query returned empty answer")
        return False
    except requests.RequestException as e:
        print(f"[FAIL] Query failed: {e}")
        return False


def run_smoke_tests(base_url: str, test_pdf: Path) -> int:
    """Run all smoke tests. Returns 0 on success, 1 on failure."""
    failures = 0

    print("=" * 60)
    print("RecRAG Smoke Tests")
    print("=" * 60)

    # Step 1: Health checks
    print("\n--- Health Checks ---")
    if not check_health(base_url, 8000, "Retrieval API"):
        failures += 1
    if not check_health(base_url, 8001, "Ingestion API"):
        failures += 1

    # Step 2: Upload PDF
    print("\n--- Upload Test ---")
    if not test_pdf.exists():
        print(f"[FAIL] Test PDF not found: {test_pdf}")
        return 1

    task_id = upload_pdf(base_url, test_pdf)
    if not task_id:
        return 1

    # Step 3: Wait for ingestion
    print("\n--- Ingestion Wait ---")
    if not wait_for_ingestion(base_url, task_id):
        return 1

    # Step 4: Query
    print("\n--- Query Test ---")
    if not query_rag(base_url, "What is RecRAG?"):
        failures += 1

    # Summary
    print("\n" + "=" * 60)
    if failures == 0:
        print("All smoke tests passed!")
        return 0
    print(f"Smoke tests completed with {failures} failure(s)")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="RecRAG post-deployment smoke tests")
    parser.add_argument("--base-url", default="http://localhost", help="Base URL for the APIs")
    parser.add_argument("--test-pdf", type=Path, default=Path("data/sample.pdf"), help="Path to a test PDF")
    args = parser.parse_args()

    return run_smoke_tests(args.base_url, args.test_pdf)


if __name__ == "__main__":
    sys.exit(main())
