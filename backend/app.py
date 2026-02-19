import json
import re
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import resolve_path, find_config_path
from pipelines import get_retrieval_pipeline

MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
PDF_MAGIC_BYTES = b"%PDF-"

st.set_page_config(page_title="RecRAG", page_icon="📚")

st.title("📚 RecRAG - RAG Pipeline")

CONFIG_PATH = find_config_path()


def get_status_file(config_path: Path) -> Path:
    from config import load_config

    config = load_config(config_path)
    storage_dir = resolve_path(
        config.get("storage", {}).get("directory", "storage"), config_path
    )
    return storage_dir / "ingestion_status.json"


def get_ingestion_status(config_path: Path) -> dict:
    status_file = get_status_file(config_path)
    if not status_file.exists():
        return {"status": "idle"}
    try:
        with open(status_file, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {"status": "idle"}


def get_upload_dir(config_path: Path) -> Path:
    from config import load_config

    config = load_config(config_path)
    return resolve_path(
        config.get("ingestion", {}).get("directory", "data/pdfs"), config_path
    )


def sanitize_filename(filename: str) -> str:
    safe_name = Path(filename).name
    safe_name = re.sub(r"[^\w\-_.]", "_", safe_name)
    return safe_name


def validate_file(uploaded_file) -> tuple[bool, str]:
    if uploaded_file.size > MAX_FILE_SIZE_BYTES:
        return False, f"File exceeds maximum size of {MAX_FILE_SIZE_MB}MB"

    content = uploaded_file.getvalue()
    if not content.startswith(PDF_MAGIC_BYTES):
        return False, "Invalid PDF file (missing PDF header)"

    return True, ""


CONFIG_PATH = find_config_path()

tab_ingest, tab_query = st.tabs(["Ingest Documents", "Query"])

with tab_ingest:
    st.header("Ingest Documents")

    current_status = get_ingestion_status(CONFIG_PATH)

    if current_status.get("status") == "processing":
        st.info(
            f"⏳ Processing started at {current_status.get('started_at', 'unknown')}"
        )
        st.write("Waiting for ingestion to complete...")
    elif current_status.get("status") == "complete":
        st.success(
            f"✅ Ingestion complete! Processed {current_status.get('files_processed', 0)} files."
        )
    elif current_status.get("status") == "error":
        st.error(
            f"❌ Ingestion failed: {current_status.get('error_message', 'Unknown error')}"
        )

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        upload_dir = get_upload_dir(CONFIG_PATH)
        upload_dir.mkdir(parents=True, exist_ok=True)

        saved_count = 0
        for uploaded_file in uploaded_files:
            is_valid, error_msg = validate_file(uploaded_file)
            if not is_valid:
                st.error(f"❌ {uploaded_file.name}: {error_msg}")
                continue

            safe_name = sanitize_filename(uploaded_file.name)
            file_path = upload_dir / safe_name

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"✅ Saved {safe_name}")
            saved_count += 1

        if saved_count == 0:
            st.warning("No valid files were uploaded")
        else:
            st.info(
                "Files uploaded. The ingestion container will automatically process them."
            )

            placeholder = st.empty()
            placeholder.info("🔄 Waiting for ingestion to complete...")

            for _ in range(60):
                time.sleep(2)
                status = get_ingestion_status(CONFIG_PATH)
                if status.get("status") == "complete":
                    placeholder.success(
                        f"✅ Ingestion complete! Processed {status.get('files_processed', 0)} files."
                    )
                    break
                elif status.get("status") == "error":
                    placeholder.error(
                        f"❌ Error: {status.get('error_message', 'Unknown error')}"
                    )
                    break
            else:
                placeholder.warning("⏱️ Still processing... (this may take a while)")

with tab_query:
    st.header("Query Documents")

    if "pipeline" not in st.session_state:
        with st.spinner("Loading retrieval pipeline..."):
            try:
                st.session_state.pipeline = get_retrieval_pipeline(CONFIG_PATH)
            except Exception as e:
                st.error(f"Failed to load pipeline: {e}")
                st.session_state.pipeline = None

    if st.session_state.pipeline is None:
        st.warning(
            "⚠️ Pipeline not loaded. Check configuration and ensure ingestion has completed."
        )
    else:
        query = st.text_input("Ask a question about your documents:")

        if query:
            with st.spinner("Searching and generating..."):
                try:
                    result = st.session_state.pipeline.query(query)

                    st.subheader("Answer:")
                    st.write(result["response"])

                    with st.expander("View retrieved context"):
                        for i, doc in enumerate(result["context"]):
                            st.write(
                                f"**Source {i + 1}** (distance: {doc.get('distance', 'N/A'):.4f})"
                            )
                            st.write(doc.get("text", "")[:500] + "...")
                            st.divider()

                except Exception as e:
                    st.error(f"Error: {e}")
