"""Metered sample of RAPTOR summary calls; input tokens for all calls are exact from the tree."""
import json, random, statistics as st
from pathlib import Path
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
from benchmark import Settings
from cost import MeteredClient, Pricing, measure
from rag_retrieval import ResponsesLLM

load_dotenv(".env", override=True)
run = sorted(Path("runs/raptor").glob("reanswer-20261101-*"))[0]
settings = Settings.model_validate_json((run / "settings.json").read_text())
nodes = json.load(open("runs/raptor/indexes/be25c875a58cbf2a5d8f880fb4214f5b0ec8e357c6a1a0c9c78713d48442c6a4/raptor_tree.json"))["nodes"]
enc = tiktoken.get_encoding("cl100k_base")
def msgs(node):
    context = "".join(" ".join(nodes[i]["text"].splitlines()) + "\n\n" for i in node["children"])
    return [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a summary of the following, including as many key details as possible. "
             f"Use at most 100 tokens:\n{context}"}]
parents = [n for n in nodes if n["layer"] > 0]
exact_input = sum(len(enc.encode(m["content"])) + 4 for n in parents for m in msgs(n))
sample = random.Random(11).sample(parents, 40)
with OpenAI(timeout=settings.llm.timeout, max_retries=0, http_client=MeteredClient(timeout=settings.llm.timeout)) as client:
    llm = ResponsesLLM(client, settings)
    costs = []
    for n in sample:
        with measure(Pricing()) as c:
            llm.infer(msgs(n), max_output_tokens=8192)
        costs.append(c)
calls = [c for q in costs for c in q.calls]
out_tok = st.mean(c.output_tokens for c in calls)
in_ratio = st.mean(c.input_tokens for c in calls) / st.mean(sum(len(enc.encode(m["content"])) + 4 for m in msgs(n)) for n in sample)
inp = exact_input * in_ratio
out = {"summary_calls": len(parents), "input_tokens_total": inp, "output_per_call": out_tok,
       "reasoning_per_call": st.mean(c.reasoning_tokens for c in calls),
       "list_usd_full_build": inp * 0.10 / 1e6 + out_tok * len(parents) * 0.50 / 1e6}
json.dump(out, open("runs/heldout400/build_calibration/raptor_build_calibration.json", "w"), indent=1); print(out)
