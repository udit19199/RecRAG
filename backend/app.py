import json
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from config import resolve_path
from pipelines import get_retrieval_pipeline

st.set_page_config(page_title="RecRAG", page_icon="📚")

st.title("📚 RecRAG - RAG Pipeline")

DEFAULT_CONFIG_PATH = Path("config.toml")


def get_config_path() -> Path:
    config_path = Path(__file__).parent.parent / "config.toml"
    if config_path.exists():
        return config_path
    return DEFAULT_CONFIG_PATH


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


CONFIG_PATH = get_config_path()

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

        for uploaded_file in uploaded_files:
            file_path = upload_dir / uploaded_file.name

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"✅ Saved {uploaded_file.name}")

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
