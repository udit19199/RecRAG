"""Classical RAG on Milvus with LangChain."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from dotenv import load_dotenv
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_milvus import Milvus
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4j_graphrag.generation.types import RagResultModel, RetrieverResult
from neo4j_graphrag.types import RetrieverResultItem

from graphrag.construction.construction import SourcePage
from graphrag.graph_rag import DEFAULT_LLM_MODEL, DEFAULT_REASONING_EFFORT

DEFAULT_MILVUS_URI = "http://localhost:19530"
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the question using only the retrieved context."),
        ("human", "Context:\n{context}\n\nQuestion:\n{input}"),
    ]
)


class ClassicalRAG:
    def __init__(
        self,
        *,
        embeddings: OpenAIEmbeddings,
        answer_llm: ChatOpenAI,
        milvus_uri: str,
        collection_name: str,
    ) -> None:
        self._embeddings = embeddings
        self._answer_llm = answer_llm
        self._milvus_uri = milvus_uri
        self._collection_name = collection_name

    @classmethod
    def from_config(
        cls, config_path: Path | None = None, *, collection_name: str
    ) -> ClassicalRAG:
        project_root = Path(__file__).resolve().parents[1]
        load_dotenv(dotenv_path=project_root / ".env", override=True)
        path = config_path or project_root / "config.toml"
        config = tomllib.loads(path.read_text())
        embeddings = OpenAIEmbeddings(
            model=config["embedding"]["model"],
            timeout=config["embedding"]["timeout"],
        )
        answer_llm = ChatOpenAI(
            model=config["llm"].get("model", DEFAULT_LLM_MODEL),
            timeout=config["llm"]["timeout"],
            use_responses_api=True,
            reasoning={
                "effort": config["llm"].get(
                    "reasoning_effort", DEFAULT_REASONING_EFFORT
                )
            },
        )
        return cls(
            embeddings=embeddings,
            answer_llm=answer_llm,
            milvus_uri=os.environ.get("MILVUS_URI", DEFAULT_MILVUS_URI),
            collection_name=collection_name,
        )

    @staticmethod
    def _documents(pages: list[SourcePage]) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        documents = []
        for page in pages:
            text = f"Page: {page.title}\n" + "\n\n".join(page.passages)
            for chunk in splitter.split_text(text):
                documents.append(
                    Document(page_content=chunk, metadata={"source": page.title})
                )
        return documents

    def index(self, pages: list[SourcePage]) -> None:
        Milvus.from_documents(
            self._documents(pages),
            self._embeddings,
            collection_name=self._collection_name,
            connection_args={"uri": self._milvus_uri},
            drop_old=True,
        )

    def _chain(self):
        store = Milvus(
            self._embeddings,
            collection_name=self._collection_name,
            connection_args={"uri": self._milvus_uri},
        )
        combine = create_stuff_documents_chain(self._answer_llm, ANSWER_PROMPT)
        return create_retrieval_chain(store.as_retriever(), combine)

    def answer(self, question: str) -> RagResultModel:
        chain = self._chain()
        output = chain.invoke({"input": question})
        contexts = [doc.page_content for doc in output.get("context", [])]
        return RagResultModel(
            answer=str(output.get("answer", "")),
            retriever_result=None
            if not contexts
            else RetrieverResult(
                items=[RetrieverResultItem(content=text) for text in contexts]
            ),
        )


__all__ = ["ClassicalRAG", "DEFAULT_MILVUS_URI"]
