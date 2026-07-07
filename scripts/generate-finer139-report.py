#!/usr/bin/env python3
"""Generate a meeting-ready HTML report from FiNER-139 benchmark JSON."""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "docs/research/graphrag/finer139/latest.json"
DEFAULT_OUTPUT = ROOT / "docs/research/graphrag/finer139/report.html"

METHOD_COLORS = {
    "llm": "#6366f1",
    "nlp": "#0284c7",
    "ontology": "#059669",
    "hybrid": "#d97706",
    "dynamic": "#db2777",
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
            parts.append(f'<mark class="{css_class}">{label}</mark>')
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


def _method_by_name(methods: list[dict], name: str) -> dict | None:
    return next((m for m in methods if m.get("name") == name), None)


def build_takeaways(
    methods: list[dict],
    best: dict | None,
    comparison: dict,
    llm_info: dict,
) -> list[tuple[str, str]]:
    """Data-driven talking points for the meeting summary."""
    points: list[tuple[str, str]] = []

    if best and best.get("strict"):
        strict = best["strict"]
        points.append(
            (
                f"{best['display_name']} leads on strict F1 ({pct(strict['f1'])})",
                f"Recall {pct(strict['recall'])}, precision {pct(strict['precision'])}. "
                f"{'Uses an LLM.' if best.get('uses_llm') else 'Runs offline with no LLM cost.'}",
            )
        )

    nlp = _method_by_name(methods, "nlp")
    if nlp and nlp.get("strict") and nlp.get("relaxed"):
        points.append(
            (
                "Generic spaCy NER misses exact span boundaries",
                f"Relaxed F1 {pct(nlp['relaxed']['f1'])} vs strict F1 {pct(nlp['strict']['f1'])}. "
                "Many numeric tokens are found, but token boundaries rarely match FiNER gold.",
            )
        )

    wins = comparison.get("sentence_wins") or {}
    if wins:
        leader = max(wins.items(), key=lambda x: x[1])
        total = comparison.get("sentences_with_gold") or 0
        points.append(
            (
                f"{leader[0]} wins {leader[1]} of {total} sentences head-to-head",
                f"{comparison.get('ties', 0)} sentences tied on strict F1.",
            )
        )

    pending = [m for m in methods if m.get("error")]
    if pending:
        model = llm_info.get("model", "gpt-4o-mini")
        names = ", ".join(m["display_name"] for m in pending[:3])
        suffix = f" and {len(pending) - 3} more" if len(pending) > 3 else ""
        points.append(
            (
                f"{len(pending)} method(s) still need an API key",
                f"{names}{suffix} were not scored. Re-run with OPENAI_API_KEY to compare "
                f"schema-guided LLM methods (default model: {model}).",
            )
        )

    if not points:
        points.append(
            (
                "No scored methods in this run",
                "Select at least one extraction method and re-run the benchmark.",
            )
        )

    return points[:4]


def build_html(data: dict) -> str:
    results, run_at = normalize_data(data)
    dataset = results["dataset"]
    methods = results["methods"]
    examples = results.get("examples", [])
    params = results.get("params", {})
    llm_info = results.get("llm", {})
    comparison = results.get("comparison") or {}

    scored = [m for m in methods if m.get("strict")]
    best = max(scored, key=lambda m: m["strict"]["f1"]) if scored else None
    max_f1 = max((m["strict"]["f1"] for m in scored), default=1.0)
    takeaways = build_takeaways(methods, best, comparison, llm_info)

    run_date = html.escape(str(run_at)[:10] if run_at else "—")
    split = html.escape(str(dataset.get("split", "validation")))
    seed = params.get("seed", 42)
    num_sentences = dataset.get("num_sentences", 0)
    num_gold = dataset.get("num_gold_entities", 0)

    # Horizontal F1 comparison (primary meeting visual)
    chart_rows: list[str] = []
    for m in methods:
        color = METHOD_COLORS.get(m["name"], "#64748b")
        strict = m.get("strict")
        err = m.get("error")
        if err:
            chart_rows.append(
                f"""<div class="compare-row compare-row--pending">
                  <div class="compare-meta">
                    <span class="swatch" style="background:{color}"></span>
                    <span class="compare-name">{html.escape(m["display_name"])}</span>
                  </div>
                  <div class="compare-track"><div class="compare-fill" style="width:0"></div></div>
                  <span class="compare-value muted">Not scored</span>
                </div>"""
            )
            continue
        f1 = strict["f1"] if strict else 0
        is_best = best and m["name"] == best["name"]
        chart_rows.append(
            f"""<div class="compare-row{" compare-row--best" if is_best else ""}">
              <div class="compare-meta">
                <span class="swatch" style="background:{color}"></span>
                <span class="compare-name">{html.escape(m["display_name"])}</span>
                {"<span class='tag tag-best'>Leader</span>" if is_best else ""}
              </div>
              <div class="compare-track">
                <div class="compare-fill" style="width:{bar_width(f1, max_f1):.1f}%;background:{color}"></div>
              </div>
              <span class="compare-value">{pct(f1)}</span>
            </div>"""
        )

    method_rows: list[str] = []
    for m in methods:
        color = METHOD_COLORS.get(m["name"], "#64748b")
        strict = m.get("strict")
        relaxed = m.get("relaxed")
        is_best = best and m["name"] == best["name"]
        err = m.get("error")
        if err:
            method_rows.append(
                f"""<tr class="row-pending">
                  <td>
                    <div class="method-cell">
                      <span class="swatch" style="background:{color}"></span>
                      <div>
                        <div class="method-title">{html.escape(m["display_name"])}</div>
                        <div class="method-note">{html.escape(err[:100])}</div>
                      </div>
                    </div>
                  </td>
                  <td colspan="6" class="muted">Pending</td>
                </tr>"""
            )
            continue
        method_rows.append(
            f"""<tr{" class='row-best'" if is_best else ""}>
              <td>
                <div class="method-cell">
                  <span class="swatch" style="background:{color}"></span>
                  <div>
                    <div class="method-title">{html.escape(m["display_name"])}</div>
                    {"<span class='tag tag-best'>Leader</span>" if is_best else ""}
                    {"<span class='tag tag-llm'>LLM</span>" if m.get("uses_llm") else ""}
                  </div>
                </div>
              </td>
              <td class="num">{pct(strict["precision"] if strict else None)}</td>
              <td class="num">{pct(strict["recall"] if strict else None)}</td>
              <td class="num num-highlight">{pct(strict["f1"] if strict else None)}</td>
              <td class="num">{pct(relaxed["f1"] if relaxed else None)}</td>
              <td class="num">{m.get("latency_s", 0):.2f}s</td>
              <td class="num">{m.get("llm_calls") if m.get("llm_calls") else "—"}</td>
            </tr>"""
        )

    takeaway_html = "".join(
        f"""<li class="takeaway">
          <p class="takeaway-title">{html.escape(title)}</p>
          <p class="takeaway-body">{html.escape(body)}</p>
        </li>"""
        for title, body in takeaways
    )

    wins = comparison.get("sentence_wins") or {}
    h2h_bars = ""
    if wins:
        max_wins = max(wins.values()) if wins else 1
        h2h_bars = "".join(
            f"""<div class="h2h-row">
              <span class="h2h-label">{html.escape(k)}</span>
              <div class="h2h-track">
                <div class="h2h-fill" style="width:{bar_width(v, max_wins):.1f}%"></div>
              </div>
              <span class="h2h-value">{v}</span>
            </div>"""
            for k, v in sorted(wins.items(), key=lambda x: -x[1])
        )

    diag_rows: list[str] = []
    for m in methods:
        d = m.get("diagnostics")
        if not d or m.get("error"):
            continue
        ci = d.get("bootstrap_strict_f1_ci", {})
        err = d.get("errors", {})
        diag_rows.append(
            f"""<tr>
              <td>{html.escape(m["display_name"])}</td>
              <td class="num">{pct(m.get("partial", {}).get("f1"))}</td>
              <td class="num">{pct(m.get("macro_strict", {}).get("f1"))}</td>
              <td class="num">{pct(ci.get("low"))} – {pct(ci.get("high"))}</td>
              <td class="num">{pct(d.get("sentence_hit_rate"))}</td>
              <td class="num">{err.get("boundary_fp", "—")}</td>
              <td class="num">{err.get("spurious_fp", "—")}</td>
            </tr>"""
        )

    example_cards: list[str] = []
    for ex in examples[:3]:
        gold = span_set(ex.get("gold", []))
        preds = ex.get("predictions", {})
        pred_keys = [k for k in ("ontology", "nlp", "llm", "hybrid", "dynamic") if k in preds]
        pred_html = ""
        for key in pred_keys[:2]:
            if preds.get(key):
                pred_html += f"""<div class="example-pred">
                  <p class="example-pred-label">{html.escape(key)}</p>
                  <p class="example-text">{render_tokens(ex["tokens"], span_set(preds[key]), "mark-pred")}</p>
                </div>"""

        example_cards.append(
            f"""<article class="example">
              <p class="example-id">Sentence {ex["index"]}</p>
              <p class="example-label">Gold numeric spans</p>
              <p class="example-text">{render_tokens(ex["tokens"], gold, "mark-gold")}</p>
              {pred_html or '<p class="muted">No predictions recorded.</p>'}
            </article>"""
        )

    best_name = html.escape(best["display_name"]) if best else "—"
    best_f1 = pct(best["strict"]["f1"]) if best and best.get("strict") else "—"
    best_recall = pct(best["strict"]["recall"]) if best and best.get("strict") else "—"

    llm_alert = ""
    pending_count = sum(1 for m in methods if m.get("error"))
    if pending_count:
        model_name = html.escape(str(llm_info.get("model", "gpt-4o-mini")))
        llm_alert = f"""
        <aside class="notice" role="note">
          <p class="notice-title">{pending_count} method(s) not scored in this run</p>
          <p class="notice-body">
            Configure <code>OPENAI_API_KEY</code> in <code>.env</code> and re-run the benchmark
            to score LLM-based extractors (default model: {model_name}).
          </p>
        </aside>"""

    diag_section = ""
    if diag_rows:
        diag_section = f"""
        <details class="details-block">
          <summary>Extended metrics (protocol v2)</summary>
          <div class="table-wrap">
            <table>
              <thead><tr>
                <th>Method</th><th>Partial F1</th><th>Macro strict F1</th><th>95% CI</th>
                <th>Sentence hit rate</th><th>Boundary FP</th><th>Spurious FP</th>
              </tr></thead>
              <tbody>{"".join(diag_rows)}</tbody>
            </table>
          </div>
          <p class="details-note">
            Partial F1 uses IoU ≥ 0.5. Macro F1 averages per-sentence strict F1.
            Bootstrap CI resamples sentences.
          </p>
        </details>"""

    h2h_section = ""
    if h2h_bars:
        h2h_section = f"""
        <section class="section" id="head-to-head">
          <div class="section-head">
            <h2>Sentence-level wins</h2>
            <p class="section-lead">
              {comparison.get("sentences_with_gold", 0)} sentences with gold entities;
              {comparison.get("ties", 0)} ties on strict F1.
            </p>
          </div>
          <div class="panel h2h-panel">{h2h_bars}</div>
        </section>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>FiNER-139 Benchmark — RecRAG Research</title>
  <style>
    :root {{
      --canvas: #f8fafc;
      --surface: #ffffff;
      --ink: #0f172a;
      --muted: #475569;
      --border: #e2e8f0;
      --accent: #2563eb;
      --accent-soft: #eff6ff;
      --success: #059669;
      --success-soft: #ecfdf5;
      --warning-soft: #fffbeb;
      --warning-ink: #92400e;
      --radius: 10px;
      --font: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
      --mono: ui-monospace, "JetBrains Mono", "SF Mono", Consolas, monospace;
    }}

    *, *::before, *::after {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      font-family: var(--font);
      font-size: 1rem;
      line-height: 1.6;
      color: var(--ink);
      background: var(--canvas);
    }}

    @media (prefers-reduced-motion: reduce) {{
      html {{ scroll-behavior: auto; }}
      .compare-fill, .h2h-fill {{ transition: none !important; }}
    }}

    .page {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 2rem 1.5rem 4rem;
    }}

    .doc-header {{
      display: flex;
      flex-wrap: wrap;
      align-items: flex-start;
      justify-content: space-between;
      gap: 1.5rem;
      padding-bottom: 1.75rem;
      border-bottom: 1px solid var(--border);
      margin-bottom: 2rem;
    }}

    .doc-kicker {{
      font-size: 0.875rem;
      font-weight: 500;
      color: var(--muted);
      margin: 0 0 0.35rem;
    }}

    h1 {{
      font-size: clamp(1.75rem, 3vw, 2.25rem);
      font-weight: 700;
      line-height: 1.2;
      letter-spacing: -0.02em;
      text-wrap: balance;
      margin: 0;
    }}

    .doc-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem 1rem;
      margin-top: 0.85rem;
      font-size: 0.875rem;
      color: var(--muted);
    }}

    .doc-meta span {{ white-space: nowrap; }}

    .nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      font-size: 0.875rem;
    }}

    .nav a {{
      color: var(--muted);
      text-decoration: none;
      padding: 0.35rem 0.75rem;
      border: 1px solid var(--border);
      border-radius: 999px;
      background: var(--surface);
    }}

    .nav a:hover {{ color: var(--accent); border-color: #bfdbfe; }}

    .notice {{
      background: var(--warning-soft);
      border: 1px solid #fde68a;
      border-radius: var(--radius);
      padding: 1rem 1.15rem;
      margin-bottom: 1.5rem;
    }}

    .notice-title {{ font-weight: 600; color: var(--warning-ink); margin: 0 0 0.25rem; }}
    .notice-body {{ margin: 0; color: #78350f; font-size: 0.9375rem; }}

    .summary-grid {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 1.25rem;
      margin-bottom: 2rem;
    }}

    @media (max-width: 820px) {{
      .summary-grid {{ grid-template-columns: 1fr; }}
    }}

    .panel {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.25rem 1.35rem;
    }}

    .recommendation {{
      background: var(--success-soft);
      border-color: #a7f3d0;
    }}

    .recommendation-label {{
      font-size: 0.8125rem;
      font-weight: 600;
      color: var(--success);
      margin: 0 0 0.35rem;
    }}

    .recommendation-title {{
      font-size: 1.125rem;
      font-weight: 700;
      margin: 0 0 0.5rem;
      text-wrap: balance;
    }}

    .recommendation-stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 1.25rem;
      margin-top: 0.75rem;
      font-size: 0.9375rem;
    }}

    .recommendation-stats dt {{
      font-size: 0.75rem;
      font-weight: 600;
      color: var(--muted);
      margin: 0;
    }}

    .recommendation-stats dd {{
      margin: 0.1rem 0 0;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
    }}

    .takeaway-list {{
      list-style: none;
      margin: 0;
      padding: 0;
      display: grid;
      gap: 1rem;
    }}

    .takeaway-title {{
      font-weight: 600;
      margin: 0 0 0.2rem;
      font-size: 0.9375rem;
    }}

    .takeaway-body {{
      margin: 0;
      color: var(--muted);
      font-size: 0.875rem;
      text-wrap: pretty;
    }}

    .section {{ margin-bottom: 2.25rem; }}

    .section-head {{ margin-bottom: 0.85rem; }}

    h2 {{
      font-size: 1.125rem;
      font-weight: 700;
      margin: 0 0 0.25rem;
      letter-spacing: -0.01em;
    }}

    .section-lead {{
      margin: 0;
      color: var(--muted);
      font-size: 0.9375rem;
      max-width: 65ch;
      text-wrap: pretty;
    }}

    .compare-row {{
      display: grid;
      grid-template-columns: minmax(180px, 1fr) 2fr auto;
      gap: 0.75rem 1rem;
      align-items: center;
      padding: 0.65rem 0;
      border-bottom: 1px solid var(--border);
    }}

    .compare-row:last-child {{ border-bottom: none; }}
    .compare-row--best {{ background: var(--success-soft); margin: 0 -1.35rem; padding-left: 1.35rem; padding-right: 1.35rem; }}
    .compare-row--pending {{ opacity: 0.72; }}

    @media (max-width: 640px) {{
      .compare-row {{ grid-template-columns: 1fr; }}
    }}

    .compare-meta {{ display: flex; align-items: center; gap: 0.5rem; min-width: 0; }}
    .compare-name {{ font-size: 0.9375rem; font-weight: 500; }}
    .compare-track {{
      height: 0.65rem;
      background: #f1f5f9;
      border-radius: 999px;
      overflow: hidden;
    }}
    .compare-fill {{
      height: 100%;
      border-radius: 999px;
      transition: width 0.35s cubic-bezier(0.22, 1, 0.36, 1);
    }}
    .compare-value {{
      font-weight: 700;
      font-variant-numeric: tabular-nums;
      font-size: 0.9375rem;
      min-width: 3.5rem;
      text-align: right;
    }}

    .swatch {{
      width: 0.55rem;
      height: 0.55rem;
      border-radius: 50%;
      flex-shrink: 0;
    }}

    .tag {{
      display: inline-block;
      font-size: 0.6875rem;
      font-weight: 600;
      padding: 0.1rem 0.45rem;
      border-radius: 999px;
      margin-left: 0.35rem;
    }}

    .tag-best {{ background: #fef3c7; color: #92400e; }}
    .tag-llm {{ background: var(--accent-soft); color: var(--accent); }}

    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      background: var(--surface);
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.875rem;
    }}

    th {{
      text-align: left;
      font-size: 0.75rem;
      font-weight: 600;
      color: var(--muted);
      padding: 0.75rem 1rem;
      border-bottom: 1px solid var(--border);
      background: #f8fafc;
      white-space: nowrap;
    }}

    th:not(:first-child), td.num {{ text-align: right; }}

    td {{
      padding: 0.85rem 1rem;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }}

    tr:last-child td {{ border-bottom: none; }}
    tr.row-best {{ background: var(--success-soft); }}
    tr.row-pending {{ background: #fafafa; }}

    .method-cell {{ display: flex; gap: 0.65rem; align-items: flex-start; }}
    .method-title {{ font-weight: 600; }}
    .method-note {{ font-size: 0.8125rem; color: #b45309; margin-top: 0.15rem; }}
    .num {{ font-variant-numeric: tabular-nums; }}
    .num-highlight {{ font-weight: 700; color: var(--success); }}
    .muted {{ color: var(--muted); }}

    .h2h-panel {{ padding-top: 0.5rem; padding-bottom: 0.5rem; }}
    .h2h-row {{
      display: grid;
      grid-template-columns: 7rem 1fr 2.5rem;
      gap: 0.75rem;
      align-items: center;
      padding: 0.45rem 0;
    }}
    .h2h-label {{ font-size: 0.875rem; font-weight: 500; text-transform: capitalize; }}
    .h2h-track {{ height: 0.5rem; background: #f1f5f9; border-radius: 999px; overflow: hidden; }}
    .h2h-fill {{ height: 100%; background: var(--accent); border-radius: 999px; }}
    .h2h-value {{ font-weight: 700; font-variant-numeric: tabular-nums; text-align: right; }}

    .examples {{ display: grid; gap: 1rem; }}

    .example {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1rem 1.15rem;
    }}

    .example-id {{ font-size: 0.8125rem; font-weight: 600; color: var(--muted); margin: 0 0 0.65rem; }}
    .example-label, .example-pred-label {{
      font-size: 0.75rem;
      font-weight: 600;
      color: var(--muted);
      margin: 0 0 0.35rem;
    }}
    .example-pred {{ margin-top: 0.85rem; padding-top: 0.85rem; border-top: 1px dashed var(--border); }}
    .example-text {{
      font-family: var(--mono);
      font-size: 0.8125rem;
      line-height: 1.65;
      margin: 0;
      word-break: break-word;
      text-wrap: pretty;
    }}

    mark {{
      border-radius: 3px;
      padding: 0 0.15em;
      font: inherit;
      color: inherit;
    }}
    .mark-gold {{ background: #fef3c7; }}
    .mark-pred {{ background: #d1fae5; }}

    .details-block {{
      border: 1px solid var(--border);
      border-radius: var(--radius);
      background: var(--surface);
      padding: 0.75rem 1rem 1rem;
    }}

    .details-block summary {{
      cursor: pointer;
      font-weight: 600;
      font-size: 0.9375rem;
      padding: 0.35rem 0;
    }}

    .details-note {{
      margin: 0.75rem 0 0;
      font-size: 0.8125rem;
      color: var(--muted);
    }}

    .context-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 1rem;
    }}

    .context-item h3 {{
      font-size: 0.875rem;
      font-weight: 600;
      margin: 0 0 0.35rem;
    }}

    .context-item p {{
      margin: 0;
      font-size: 0.875rem;
      color: var(--muted);
      text-wrap: pretty;
    }}

    code {{
      font-family: var(--mono);
      font-size: 0.85em;
      background: #f1f5f9;
      padding: 0.1em 0.35em;
      border-radius: 4px;
    }}

    .doc-footer {{
      margin-top: 3rem;
      padding-top: 1.25rem;
      border-top: 1px solid var(--border);
      font-size: 0.8125rem;
      color: var(--muted);
      text-align: center;
    }}

    .doc-footer a {{ color: var(--accent); }}

    @media print {{
      body {{ background: #fff; }}
      .nav, .notice {{ display: none; }}
      .page {{ max-width: none; padding: 0; }}
      .section {{ break-inside: avoid; }}
      .compare-fill, .h2h-fill {{ print-color-adjust: exact; -webkit-print-color-adjust: exact; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <header class="doc-header">
      <div>
        <p class="doc-kicker">RecRAG research · Graph construction benchmark</p>
        <h1>Which extraction method finds financial numerics best?</h1>
        <div class="doc-meta">
          <span>Run date: <strong>{run_date}</strong></span>
          <span>Dataset: <strong>FiNER-139 ({split})</strong></span>
          <span>Sample: <strong>{num_sentences} sentences</strong></span>
          <span>Gold entities: <strong>{num_gold}</strong></span>
          <span>Seed: <strong>{seed}</strong></span>
        </div>
      </div>
      <nav class="nav" aria-label="Report sections">
        <a href="#summary">Summary</a>
        <a href="#results">Results</a>
        <a href="#metrics">Metrics</a>
        <a href="#examples">Examples</a>
        <a href="#context">Context</a>
      </nav>
    </header>

    {llm_alert}

    <section class="summary-grid" id="summary">
      <div class="panel recommendation">
        <p class="recommendation-label">Current leader (strict F1)</p>
        <p class="recommendation-title">{best_name}</p>
        <p class="section-lead">
          Five graph-construction extractors were compared on numeric span detection
          in SEC-style filing sentences. Strict F1 requires an exact token-boundary match.
        </p>
        <dl class="recommendation-stats">
          <div><dt>Strict F1</dt><dd>{best_f1}</dd></div>
          <div><dt>Recall</dt><dd>{best_recall}</dd></div>
          <div><dt>Methods scored</dt><dd>{len(scored)} / {len(methods)}</dd></div>
        </dl>
      </div>
      <div class="panel">
        <h2 style="margin-bottom:0.75rem">Talking points</h2>
        <ul class="takeaway-list">{takeaway_html}</ul>
      </div>
    </section>

    <section class="section" id="results">
      <div class="section-head">
        <h2>Strict F1 by method</h2>
        <p class="section-lead">Primary ranking metric. Higher is better; exact span match required.</p>
      </div>
      <div class="panel">{"".join(chart_rows)}</div>
    </section>

    <section class="section" id="metrics">
      <div class="section-head">
        <h2>Full comparison</h2>
        <p class="section-lead">Precision, recall, relaxed F1 (any token overlap), latency, and LLM usage.</p>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Method</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1 strict</th>
              <th>F1 relaxed</th>
              <th>Latency</th>
              <th>LLM calls</th>
            </tr>
          </thead>
          <tbody>{"".join(method_rows)}</tbody>
        </table>
      </div>
      {diag_section}
    </section>

    {h2h_section}

    <section class="section" id="examples">
      <div class="section-head">
        <h2>Sample predictions</h2>
        <p class="section-lead">
          Gold numeric spans (amber) vs method predictions (green). Useful for explaining
          boundary errors in the meeting.
        </p>
      </div>
      <div class="examples">{"".join(example_cards)}</div>
    </section>

    <section class="section" id="context">
      <div class="section-head">
        <h2>What we measured</h2>
      </div>
      <div class="context-grid">
        <div class="context-item panel">
          <h3>Task</h3>
          <p>Detect numeric entity spans in financial filing sentences. We score span location only, not the 139 XBRL type labels.</p>
        </div>
        <div class="context-item panel">
          <h3>Fair comparison</h3>
          <p>All methods are filtered to numeric predictions before scoring, so open-ended extractors are not penalized for non-numeric entities FiNER never annotates.</p>
        </div>
        <div class="context-item panel">
          <h3>Next step</h3>
          <p>Run all five methods with an LLM API key to see whether schema-guided Hybrid beats the ontology baseline on precision while keeping recall.</p>
        </div>
      </div>
    </section>

    <footer class="doc-footer">
      RecRAG FiNER-139 experiment ·
      <a href="https://huggingface.co/datasets/nlpaueb/finer-139">nlpaueb/finer-139</a>
    </footer>
  </main>
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
