from __future__ import annotations

import asyncio
import json
import os
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from neo4j import Driver, GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm import LLMBase
from neo4j_graphrag.llm.types import LLMResponse, ToolCall, ToolCallResponse
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.tool import Tool
from neo4j_graphrag.types import LLMMessage
from pydantic import BaseModel

from .construction.construction import ConstructionMethod, SourcePage, rebuild_graph
from .cost import TokenLedger
from .retrieval.answering import RetrievalMethod, answer_question

DEFAULT_LLM_MODEL = "gpt-5.6-luna"
DEFAULT_REASONING_EFFORT = "medium"
LLMInput = str | list[LLMMessage]
LLMHistory = list[LLMMessage] | MessageHistory | None


class GraphRAGChatLLM(LLMBase):
    supports_structured_output = True

    def __init__(self, model: ChatOpenAI) -> None:
        super().__init__(model_name=model.model_name)
        self._model = model

    @staticmethod
    def _messages(
        input: LLMInput,
        history: LLMHistory,
        system_instruction: str | None,
    ) -> list[LLMMessage]:
        if isinstance(input, list):
            messages = list(input)
        else:
            previous = (
                history.messages
                if isinstance(history, MessageHistory)
                else history or []
            )
            messages = [*previous, LLMMessage(role="user", content=input)]
        if system_instruction:
            messages.insert(0, LLMMessage(role="system", content=system_instruction))
        return messages

    @staticmethod
    def _content(response: Any) -> str:
        if isinstance(response, BaseMessage):
            return response.text
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, BaseModel):
            return content.model_dump_json()
        return json.dumps(content)

    @staticmethod
    def _tool_schema(tool: Tool) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.get_name(),
                "description": tool.get_description(),
                "parameters": tool.get_parameters(),
            },
        }

    def _invoke(
        self,
        input: LLMInput,
        history: LLMHistory,
        system_instruction,
        response_format,
        **kwargs,
    ):
        model = (
            self._model.with_structured_output(
                response_format, method="function_calling"
            )
            if response_format
            else self._model
        )
        return model.invoke(
            self._messages(input, history, system_instruction), **kwargs
        )

    def invoke(
        self,
        input: LLMInput,
        message_history: LLMHistory = None,
        system_instruction=None,
        response_format=None,
        **kwargs,
    ):
        try:
            return LLMResponse(
                content=self._content(
                    self._invoke(
                        input,
                        message_history,
                        system_instruction,
                        response_format,
                        **kwargs,
                    )
                )
            )
        except Exception as exc:
            raise LLMGenerationError(exc) from exc

    async def ainvoke(
        self,
        input: LLMInput,
        message_history: LLMHistory = None,
        system_instruction=None,
        response_format=None,
        **kwargs,
    ):
        return await asyncio.to_thread(
            self.invoke,
            input,
            message_history,
            system_instruction,
            response_format,
            **kwargs,
        )

    def invoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: LLMHistory = None,
        system_instruction=None,
    ):
        try:
            response = self._model.bind_tools(
                [self._tool_schema(tool) for tool in tools]
            ).invoke(self._messages(input, message_history, system_instruction))
            return ToolCallResponse(
                tool_calls=[
                    ToolCall(name=call["name"], arguments=call["args"])
                    for call in response.tool_calls
                ],
                content=self._content(response) or None,
            )
        except Exception as exc:
            raise LLMGenerationError(exc) from exc

    async def ainvoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: LLMHistory = None,
        system_instruction=None,
    ):
        return await asyncio.to_thread(
            self.invoke_with_tools, input, tools, message_history, system_instruction
        )


class GraphRAG:
    def __init__(
        self,
        *,
        driver: Driver,
        llm: LLMBase,
        embedder: Embedder,
        answer_llm,
        embedding_dimensions: int,
        usage: TokenLedger | None = None,
    ) -> None:
        self._driver = driver
        self._llm = llm
        self._embedder = embedder
        self._answer_llm = answer_llm
        self._embedding_dimensions = embedding_dimensions
        self._usage = usage or TokenLedger()

    @classmethod
    def from_config(cls, config_path: Path | None = None) -> GraphRAG:
        project_root = Path(__file__).resolve().parents[1]
        load_dotenv(dotenv_path=project_root / ".env", override=True)
        path = config_path or project_root / "config.toml"
        config = tomllib.loads(path.read_text())
        username, password = os.environ["NEO4J_AUTH"].split("/", 1)
        driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(username, password),
        )
        try:
            usage = TokenLedger(
                input_price_per_million=config.get("cost", {}).get(
                    "input_price_per_million", 0.20
                ),
                output_price_per_million=config.get("cost", {}).get(
                    "output_price_per_million", 1.20
                ),
            )
            chat_model = ChatOpenAI(
                model=config["llm"].get("model", DEFAULT_LLM_MODEL),
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
                driver=driver,
                llm=GraphRAGChatLLM(chat_model),
                embedder=OpenAIEmbeddings(
                    model=config["embedding"]["model"],
                    timeout=config["embedding"]["timeout"],
                ),
                answer_llm=chat_model,
                embedding_dimensions=config["embedding"]["dimensions"],
                usage=usage,
            )
        except Exception:
            driver.close()
            raise

    def construct(
        self,
        pages: Sequence[SourcePage],
        *,
        method: ConstructionMethod,
        database: str,
    ) -> None:
        rebuild_graph(
            pages,
            driver=self._driver,
            method=method,
            llm=self._llm,
            embedder=self._embedder,
            database=database,
            embedding_dimensions=self._embedding_dimensions,
        )

    def answer(
        self,
        question: str,
        *,
        database: str,
        retrieval_methods: list[RetrievalMethod],
    ):
        return answer_question(
            question,
            driver=self._driver,
            database=database,
            llm=self._llm,
            embedder=self._embedder,
            answer_llm=self._answer_llm,
            retrieval_methods=retrieval_methods,
        )

    def close(self) -> None:
        self._driver.close()

    @property
    def usage(self) -> TokenLedger:
        return self._usage


__all__ = [
    "DEFAULT_LLM_MODEL",
    "DEFAULT_REASONING_EFFORT",
    "GraphRAGChatLLM",
    "GraphRAG",
]
