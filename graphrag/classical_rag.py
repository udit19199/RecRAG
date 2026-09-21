"""Classical RAG on Milvus with LangChain."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.callbacks import UsageMetadataCallbackHandler
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
        usage: UsageMetadataCallbackHandler,
    ) -> None:
        self._embeddings = embeddings
        self._answer_llm = answer_llm
        self._milvus_uri = milvus_uri
        self._collection_name = collection_name
        self._usage = usage

    @classmethod
    def from_config(
        cls,
        config_path: Path | None = None,
        *,
        collection_name: str,
        usage: UsageMetadataCallbackHandler | None = None,
    ) -> ClassicalRAG:
        project_root = Path(__file__).resolve().parents[1]
        load_dotenv(dotenv_path=project_root / ".env", override=True)
        path = config_path or project_root / "config.toml"
        config = tomllib.loads(path.read_text())
        embedding_model = config["embedding"]["model"]
        answer_model = config["llm"].get("model", DEFAULT_LLM_MODEL)
        usage = usage or UsageMetadataCallbackHandler()
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            timeout=config["embedding"]["timeout"],
        )
        answer_llm = ChatOpenAI(
            model=answer_model,
            timeout=config["llm"]["timeout"],
            use_responses_api=True,
            reasoning={
                "effort": config["llm"].get(
                    "reasoning_effort", DEFAULT_REASONING_EFFORT
                )
            },
            callbacks=[usage],
        )
        return cls(
            embeddings=embeddings,
            answer_llm=answer_llm,
            milvus_uri=os.environ.get("MILVUS_URI", DEFAULT_MILVUS_URI),
            collection_name=collection_name,
            usage=usage,
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

    def index(
        self,
        pages: list[SourcePage],
    ) -> None:
        Milvus.from_documents(
            self._documents(pages),
            self._embeddings,
            collection_name=self._collection_name,
            connection_args={"uri": self._milvus_uri},
            drop_old=True,
        )

    def _store(self):
        return Milvus(
            self._embeddings,
            collection_name=self._collection_name,
            connection_args={"uri": self._milvus_uri},
        )

    def answer(
        self,
        question: str,
    ) -> RagResultModel:
        documents = self._store().similarity_search(question)
        contexts = [document.page_content for document in documents]
        prompt = ANSWER_PROMPT.invoke(
            {"context": "\n\n".join(contexts), "input": question}
        )
        output = self._answer_llm.invoke(prompt.to_messages())
        return RagResultModel(
            answer=str(output.content),
            retriever_result=None
            if not contexts
            else RetrieverResult(
                items=[RetrieverResultItem(content=text) for text in contexts]
            ),
        )

    @property
    def usage(self) -> UsageMetadataCallbackHandler:
        return self._usage


__all__ = ["ClassicalRAG", "DEFAULT_MILVUS_URI"]
