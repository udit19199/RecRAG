    def get_search_results(
        self,
        query_text: str,
        message_history: list[LLMMessage] | None = None,
        top_k: int = RETRIEVAL_TOP_K,
        **kwargs: Any,
    ) -> RawSearchResult:
        """Run the LangChain agent and pass on the evidence it picked.

        The agent ends by listing the IDs of the results the answer needs,
        from either tool; those come first, in its order. Remaining slots up
        to top_k are filled as before: pooled vector_search passages by best
        similarity score, then cypher_search rows in arrival order. If the
        agent names no valid ID, that old rule picks all top_k. The metadata
        records every tool call and, as a final "select_evidence" step, the
        agent's reply and how many of its picks were used.
        """
        messages = [*(message_history or []), {"role": "user", "content": query_text}]
        key = sha256(f"{self.neo4j_database}\0{query_text}".encode()).hexdigest()
        result = self._agent.invoke(
            {"messages": messages},
            context=_AgentContext(cache_key=f"recrag-agent-{key[:32]}"),
        )
        queries = {
            call["id"]: call["args"]
            for message in result["messages"]
            if isinstance(message, AIMessage)
            for call in message.tool_calls
        }
        # A passage found by several searches keeps its best score.
        passages: dict[str, tuple[float, int, neo4j.Record]] = {}
        seen_rows: set[str] = set()
        rows: list[neo4j.Record] = []
        trace: list[dict[str, Any]] = []
        for message in result["messages"]:
            if not isinstance(message, ToolMessage):
                continue
            artifact: list[neo4j.Record] = message.artifact or []
            step = len(trace)
            trace.append(
                {
                    "step": step,
                    "tool": message.name,
                    "args": queries.get(message.tool_call_id, {}),
                    "status": message.status,
                    "output": str(message.content),
                    "results": len(artifact),
                }
            )
            for record in artifact:
                key = _evidence_key(str(record["content"]))
                score = (record["metadata"] or {}).get("score")
                if record["tool_name"] == "vector_search" and score is not None:
                    found = passages.get(key)
                    if found is None:
                        passages[key] = (float(score), len(passages), record)
                    elif float(score) > found[0]:
                        passages[key] = (float(score), found[1], record)
                elif key not in seen_rows:
                    seen_rows.add(key)
                    rows.append(record)
        # Highest score first; ties keep the order the agent found them in.
        ranked = sorted(passages.values(), key=lambda entry: (-entry[0], entry[1]))
        fallback = [record for _, _, record in ranked] + rows
        by_id = {evidence_id(str(record["content"])): record for record in fallback}
        final = result["messages"][-1]
        reply = final.text if isinstance(final, AIMessage) else ""
        picked: list[neo4j.Record] = []
        for found_id in EVIDENCE_ID.findall(reply):
            record = by_id.get(found_id)
            if record is not None and all(record is not p for p in picked):
                picked.append(record)
        kept = picked[:top_k]
        kept += [r for r in fallback if all(r is not k for k in kept)][
            : top_k - len(kept)
        ]
        trace.append(
            {
                "step": len(trace),
                "tool": "select_evidence",
                "args": {},
                "status": "success" if picked else "no_valid_ids",
                "output": reply,
                "results": min(len(picked), top_k),
            }
        )
        return RawSearchResult(
            records=kept,
            metadata={"trace": trace, "agent_messages": len(result["messages"])},
        )


def evidence_id(content: str) -> str:
    """Short ID the agent uses to pick an item; the same text gets the same ID."""
    return "E" + sha256(_evidence_key(content).encode()).hexdigest()[:6]

# EVIDENCE_ID = \bE[0-9a-f]{6}\b
