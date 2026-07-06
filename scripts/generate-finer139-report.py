#!/usr/bin/env python3
"""Generate a standalone HTML report from FiNER-139 benchmark JSON."""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "docs/research/findings/finer139-benchmark-latest.json"
DEFAULT_OUTPUT = ROOT / "docs/research/findings/finer139-benchmark-report.html"

METHOD_COLORS = {
    "llm": "#6366f1",
    "nlp": "#0ea5e9",
    "ontology": "#10b981",
    "hybrid": "#f59e0b",
    "dynamic": "#ec4899",
}


def pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


def bar_width(value: float | None, max_val: float = 1.0) -> float:
    if value is None or max_val <= 0:
        return 0.0
    return min(100.0, max(0.0, (value / max_val) * 100))


def span_set(spans: list[list[int]]) -> set[tuple[int, int]]:
    return {(s[0], s[1]) for s in spans}


def render_tokens(tokens: list[str], spans: set[tuple[int, int]], css_class: str) -> str:
    parts: list[str] = []
    for i, tok in enumerate(tokens):
        if i > 0:
            parts.append(" ")
        label = html.escape(tok)
        in_span = any(start <= i < end for start, end in spans)
        if in_span:
            parts.append(f'<span class="{css_class}">{label}</span>')
        else:
            parts.append(label)
    return "".join(parts)


def load_data(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def normalize_data(data: dict) -> tuple[dict, str]:
    """Support both API wrapper and direct runner JSON shapes."""
    if "response" in data:
        response = data["response"]
        results = response["results"]
        run_at = data.get("run_at", response.get("updated_at", ""))
    else:
        results = data["results"]
        run_at = data.get("run_at", "")
    return results, str(run_at)


def build_html(data: dict) -> str:
    results, run_at = normalize_data(data)
    dataset = results["dataset"]
    methods = results["methods"]
    examples = results.get("examples", [])
    params = results.get("params", {})

    scored = [m for m in methods if m.get("strict")]
    best = max(scored, key=lambda m: m["strict"]["f1"]) if scored else None
    max_f1 = max((m["strict"]["f1"] for m in scored), default=1.0)

    method_rows = []
    chart_bars = []
    for m in methods:
        name = m["name"]
        color = METHOD_COLORS.get(name, "#94a3b8")
        strict = m.get("strict")
        relaxed = m.get("relaxed")
        is_best = best and m["name"] == best["name"]
        err = m.get("error")
        if err:
            method_rows.append(
                f"""
                <tr class="row-error">
                  <td>
                    <div class="method-name">
                      <span class="dot" style="background:{color}"></span>
                      {html.escape(m["display_name"])}
                      {"<span class='badge badge-winner'>Best F1</span>" if is_best else ""}
                      {"<span class='badge badge-llm'>LLM</span>" if m.get("uses_llm") else ""}
                    </div>
                    <p class="error-note">{html.escape(err[:120])}…</p>
                  </td>
                  <td colspan="6" class="muted">Not run — API key required</td>
                </tr>"""
            )
            chart_bars.append(
                f'<div class="chart-row muted"><span class="chart-label">{html.escape(m["display_name"][:28])}</span><div class="chart-track"><div class="chart-fill" style="width:0;background:{color}"></div></div><span class="chart-val">—</span></div>'
            )
            continue

        f1 = strict["f1"] if strict else 0
        method_rows.append(
            f"""
            <tr{" class='row-best'" if is_best else ""}>
              <td>
                <div class="method-name">
                  <span class="dot" style="background:{color}"></span>
                  {html.escape(m["display_name"])}
                  {"<span class='badge badge-winner'>Best F1</span>" if is_best else ""}
                  {"<span class='badge badge-llm'>LLM</span>" if m.get("uses_llm") else ""}
                </div>
              </td>
              <td class="num">{pct(strict["precision"] if strict else None)}</td>
              <td class="num">{pct(strict["recall"] if strict else None)}</td>
              <td class="num strong">{pct(strict["f1"] if strict else None)}</td>
              <td class="num">{pct(relaxed["f1"] if relaxed else None)}</td>
              <td class="num">{m.get("latency_s", 0):.2f}s</td>
              <td class="num">{m.get("llm_calls") or "—"}</td>
            </tr>"""
        )
        chart_bars.append(
            f"""<div class="chart-row">
              <span class="chart-label">{html.escape(m["display_name"][:28])}</span>
              <div class="chart-track"><div class="chart-fill" style="width:{bar_width(f1, max_f1):.1f}%;background:{color}"></div></div>
              <span class="chart-val">{pct(f1)}</span>
            </div>"""
        )

    example_cards = []
    for ex in examples[:6]:
        gold = span_set(ex.get("gold", []))
        preds = ex.get("predictions", {})
        pred_keys = [k for k in ("ontology", "nlp", "llm", "hybrid", "dynamic") if k in preds]
        pred_html = ""
        for key in pred_keys[:2]:
            if preds.get(key):
                pred_html += f"""<div class="pred-block"><span class="pred-label">{html.escape(key)}</span>
                  <p class="sentence">{render_tokens(ex["tokens"], span_set(preds[key]), "span-pred")}</p></div>"""

        example_cards.append(
            f"""
            <article class="example-card">
              <header>Sentence #{ex["index"]}</header>
              <p class="sentence gold-line">{render_tokens(ex["tokens"], gold, "span-gold")}</p>
              <div class="legend"><span class="legend-gold">Gold entities</span></div>
              {pred_html or '<p class="muted">No predictions for this example.</p>'}
            </article>"""
        )

    llm_note = ""
    llm_info = results.get("llm", {})
    if llm_info.get("error"):
        model_name = html.escape(str(llm_info.get("model", "gpt-4o-mini")))
        llm_note = f"""
        <div class="alert">
          <strong>LLM methods pending:</strong> Set <code>OPENAI_API_KEY</code> in <code>.env</code>
          and re-run to score LLM-Based, Hybrid, and Dynamic
          (default model: {model_name}).
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>FiNER-139 Benchmark Report — RecRAG</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
  <style>
    :root {{
      --bg: #0b0f1a;
      --surface: #121826;
      --surface2: #1a2234;
      --border: rgba(148, 163, 184, 0.14);
      --text: #e8edf7;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --gold: #fbbf24;
      --green: #34d399;
      --pink: #f472b6;
      --radius: 14px;
      --shadow: 0 24px 80px rgba(0,0,0,.45);
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: "DM Sans", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.55;
      min-height: 100vh;
    }}
    .bg-grid {{
      position: fixed; inset: 0; z-index: 0; pointer-events: none;
      background-image:
        linear-gradient(rgba(56,189,248,.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(56,189,248,.04) 1px, transparent 1px);
      background-size: 48px 48px;
      mask-image: radial-gradient(ellipse 80% 60% at 50% 0%, black, transparent);
    }}
    .wrap {{ position: relative; z-index: 1; max-width: 1120px; margin: 0 auto; padding: 48px 24px 80px; }}
    header.hero {{
      padding: 40px;
      border-radius: calc(var(--radius) + 4px);
      background: linear-gradient(135deg, #162033 0%, #0f172a 55%, #1e1b4b 100%);
      border: 1px solid var(--border);
      box-shadow: var(--shadow);
      margin-bottom: 32px;
    }}
    .eyebrow {{
      font-size: .75rem; font-weight: 600; letter-spacing: .14em; text-transform: uppercase;
      color: var(--accent); margin-bottom: 12px;
    }}
    h1 {{ font-size: clamp(1.75rem, 4vw, 2.5rem); font-weight: 700; letter-spacing: -.03em; line-height: 1.15; }}
    .subtitle {{ margin-top: 14px; color: var(--muted); font-size: 1.05rem; max-width: 62ch; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 28px; }}
    .pill {{
      font-size: .8rem; padding: 6px 12px; border-radius: 999px;
      background: rgba(255,255,255,.06); border: 1px solid var(--border); color: #cbd5e1;
    }}
    .pill strong {{ color: #fff; font-weight: 600; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 32px; }}
    .stat {{
      background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
      padding: 22px 20px;
    }}
    .stat-label {{ font-size: .78rem; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }}
    .stat-value {{ font-size: 2rem; font-weight: 700; margin-top: 6px; letter-spacing: -.03em; }}
    .stat-value small {{ font-size: 1rem; color: var(--muted); font-weight: 500; }}
    .stat.accent {{ border-color: rgba(52,211,153,.35); background: linear-gradient(180deg, rgba(16,185,129,.12), var(--surface)); }}
    section {{ margin-bottom: 36px; }}
    h2 {{ font-size: 1.25rem; font-weight: 600; margin-bottom: 16px; letter-spacing: -.02em; }}
    .panel {{
      background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
      padding: 24px; overflow: hidden;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: .92rem; }}
    th {{
      text-align: left; font-size: .72rem; text-transform: uppercase; letter-spacing: .08em;
      color: var(--muted); padding: 12px 14px; border-bottom: 1px solid var(--border);
    }}
    td {{ padding: 16px 14px; border-bottom: 1px solid var(--border); vertical-align: top; }}
    tr:last-child td {{ border-bottom: none; }}
    tr.row-best {{ background: rgba(52,211,153,.06); }}
    tr.row-error {{ background: rgba(244,114,182,.04); }}
    .method-name {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; font-weight: 600; }}
    .dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
    .badge {{
      font-size: .65rem; font-weight: 600; padding: 2px 8px; border-radius: 999px; text-transform: uppercase; letter-spacing: .06em;
    }}
    .badge-winner {{ background: rgba(251,191,36,.18); color: var(--gold); border: 1px solid rgba(251,191,36,.35); }}
    .badge-llm {{ background: rgba(99,102,241,.15); color: #a5b4fc; border: 1px solid rgba(99,102,241,.3); }}
    .num {{ font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }}
    .num.strong {{ color: var(--green); font-weight: 700; }}
    .muted {{ color: var(--muted); }}
    .error-note {{ font-size: .78rem; color: #fda4af; margin-top: 6px; font-weight: 400; }}
    .chart-row {{ display: grid; grid-template-columns: 180px 1fr 56px; gap: 12px; align-items: center; margin-bottom: 12px; }}
    .chart-label {{ font-size: .85rem; color: #cbd5e1; }}
    .chart-track {{ height: 10px; background: var(--surface2); border-radius: 999px; overflow: hidden; }}
    .chart-fill {{ height: 100%; border-radius: 999px; transition: width .6s ease; }}
    .chart-val {{ font-size: .85rem; font-weight: 600; text-align: right; font-variant-numeric: tabular-nums; }}
    .two-col {{ display: grid; grid-template-columns: 1.1fr .9fr; gap: 20px; }}
    @media (max-width: 860px) {{ .two-col {{ grid-template-columns: 1fr; }} .chart-row {{ grid-template-columns: 1fr; }} }}
    .findings {{ display: grid; gap: 12px; }}
    .finding {{
      padding: 16px 18px; border-radius: 12px; background: var(--surface2);
      border-left: 3px solid var(--accent);
    }}
    .finding strong {{ display: block; margin-bottom: 4px; }}
    .finding p {{ color: var(--muted); font-size: .92rem; }}
    .alert {{
      padding: 14px 18px; border-radius: 12px; margin-bottom: 24px;
      background: rgba(251,191,36,.08); border: 1px solid rgba(251,191,36,.25); color: #fde68a; font-size: .92rem;
    }}
    code {{ font-family: "JetBrains Mono", monospace; font-size: .85em; background: rgba(0,0,0,.25); padding: 2px 6px; border-radius: 4px; }}
    .examples {{ display: grid; gap: 16px; }}
    .example-card {{
      background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px;
    }}
    .example-card header {{ font-size: .75rem; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; margin-bottom: 10px; }}
    .sentence {{ font-family: "JetBrains Mono", monospace; font-size: .78rem; line-height: 1.7; word-break: break-word; }}
    .span-gold {{ background: rgba(251,191,36,.22); color: #fde68a; padding: 1px 3px; border-radius: 4px; }}
    .span-pred {{ background: rgba(52,211,153,.18); color: #6ee7b7; padding: 1px 3px; border-radius: 4px; }}
    .legend {{ margin: 8px 0 14px; font-size: .75rem; }}
    .legend-gold::before {{ content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 3px; background: rgba(251,191,36,.5); margin-right: 6px; vertical-align: middle; }}
    .pred-block {{ margin-top: 10px; }}
    .pred-label {{ font-size: .7rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); }}
    footer {{ margin-top: 48px; text-align: center; color: var(--muted); font-size: .82rem; }}
  </style>
</head>
<body>
  <div class="bg-grid"></div>
  <div class="wrap">
    <header class="hero">
      <div class="eyebrow">RecRAG Research · Graph Construction</div>
      <h1>FiNER-139 Entity Recognition Benchmark</h1>
      <p class="subtitle">
        Comparing five graph-construction methods on numeric financial entity detection
        in SEC-style filings. Type-agnostic span matching on the validation split.
      </p>
      <div class="meta">
        <span class="pill"><strong>{dataset["num_sentences"]}</strong> sentences</span>
        <span class="pill"><strong>{dataset["num_gold_entities"]}</strong> gold entities</span>
        <span class="pill">split: <strong>{html.escape(dataset["split"])}</strong></span>
        <span class="pill">seed: <strong>{params.get("seed", 42)}</strong></span>
        <span class="pill">run: <strong>{html.escape(str(run_at)[:19])}</strong></span>
      </div>
    </header>

    {llm_note}

    <div class="grid">
      <div class="stat accent">
        <div class="stat-label">Best strict F1</div>
        <div class="stat-value">{pct(best["strict"]["f1"] if best else None)}</div>
        <div class="stat-label" style="margin-top:8px;text-transform:none;letter-spacing:0">{html.escape(best["display_name"] if best else "—")}</div>
      </div>
      <div class="stat">
        <div class="stat-label">Top recall</div>
        <div class="stat-value">{pct(best["strict"]["recall"] if best else None)}</div>
      </div>
      <div class="stat">
        <div class="stat-label">Dataset</div>
        <div class="stat-value" style="font-size:1.1rem">{html.escape(dataset["id"].split("/")[-1])}</div>
      </div>
      <div class="stat">
        <div class="stat-label">Methods scored</div>
        <div class="stat-value">{len(scored)}<small> / {len(methods)}</small></div>
      </div>
    </div>

    <section class="two-col">
      <div>
        <h2>Results — strict token-span F1</h2>
        <div class="panel">
          {"".join(chart_bars)}
        </div>
      </div>
      <div>
        <h2>Key findings</h2>
        <div class="findings">
          <div class="finding" style="border-color: var(--green)">
            <strong>Ontology leads on strict F1 ({pct(best["strict"]["f1"] if best else None)})</strong>
            <p>Schema-driven gazetteer + numeric regex achieves {pct(best["strict"]["recall"] if best else None)} recall with no LLM cost — strong baseline for graph construction.</p>
          </div>
          <div class="finding">
            <strong>spaCy over-predicts numerics</strong>
            <p>Relaxed F1 (55%) ≫ strict F1 (9.8%): generic NER finds many numeric spans but with poor boundary precision on financial text.</p>
          </div>
          <div class="finding" style="border-color: var(--pink)">
            <strong>LLM methods await API key</strong>
            <p>Hybrid and Dynamic are designed to combine schema guidance with LLM flexibility — re-run with OpenAI to complete the comparison.</p>
          </div>
        </div>
      </div>
    </section>

    <section>
      <h2>Full metrics</h2>
      <div class="panel" style="padding:0">
        <table>
          <thead>
            <tr>
              <th>Method</th>
              <th style="text-align:right">Precision</th>
              <th style="text-align:right">Recall</th>
              <th style="text-align:right">F1 strict</th>
              <th style="text-align:right">F1 relaxed</th>
              <th style="text-align:right">Latency</th>
              <th style="text-align:right">LLM calls</th>
            </tr>
          </thead>
          <tbody>
            {"".join(method_rows)}
          </tbody>
        </table>
      </div>
    </section>

    <section>
      <h2>Example predictions</h2>
      <p class="muted" style="margin-bottom:16px">Gold numeric spans vs top method predictions on sample sentences.</p>
      <div class="examples">
        {"".join(example_cards)}
      </div>
    </section>

    <footer>
      Generated by RecRAG · FiNER-139 graph-construction experiment ·
      <a href="https://huggingface.co/datasets/nlpaueb/finer-139" style="color:var(--accent)">nlpaueb/finer-139</a>
    </footer>
  </div>
</body>
</html>"""


def main() -> None:
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT
    data = load_data(in_path)
    out_path.write_text(build_html(data), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
