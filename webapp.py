#!/usr/bin/env python3
"""
Web app for interactive LLM evaluation + hallucination checks.

Run:
  uvicorn webapp:app --reload
Open:
  http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

# Allow running from repo root without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from llm_eval.evaluation.evaluator import evaluate_response
from llm_eval.hallucination.detector import hallucination_result

app = FastAPI(title="LLM Evaluation & Hallucination Web")


def _pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _page(
    *,
    prompt: str = "",
    response: str = "",
    reference: str = "",
    claim: str = "",
    source: str = "",
    enable_hallucination: bool = False,
    use_nli: bool = False,
    result: Optional[dict[str, Any]] = None,
) -> str:
    result_html = ""
    if result is not None:
        result_html = f"""
        <section class="result">
          <div class="result-head">
            <h2>Result</h2>
          </div>
          <pre>{_pretty(result)}</pre>
        </section>
        """

    checked_h = "checked" if enable_hallucination else ""
    checked_nli = "checked" if use_nli else ""

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>LLM Evaluation</title>
    <style>
      :root {{
        --bg: #020617;
        --card: rgba(15, 23, 42, 0.92);
        --muted: #94a3b8;
        --text: #e5e7eb;
        --border: rgba(148, 163, 184, 0.25);
        --accent: #22c55e;
        --accent2: #60a5fa;
      }}
      * {{ box-sizing: border-box; font-family: system-ui, -apple-system, Segoe UI, sans-serif; }}
      body {{
        margin: 0;
        min-height: 100vh;
        padding: 32px 16px;
        background: radial-gradient(circle at top, #0b1224, var(--bg) 55%);
        color: var(--text);
      }}
      .wrap {{ max-width: 980px; margin: 0 auto; }}
      header {{
        display: flex;
        flex-direction: column;
        gap: 6px;
        margin-bottom: 18px;
      }}
      header h1 {{ margin: 0; font-size: 22px; font-weight: 650; }}
      header p {{ margin: 0; color: var(--muted); font-size: 13px; }}
      .grid {{
        display: grid;
        grid-template-columns: 1fr;
        gap: 16px;
      }}
      @media (min-width: 960px) {{
        .grid {{ grid-template-columns: 1.15fr 0.85fr; align-items: start; }}
      }}
      .card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 18px 60px rgba(0,0,0,0.5);
      }}
      form {{ display: flex; flex-direction: column; gap: 12px; }}
      label {{ display: block; font-size: 12px; color: #cbd5e1; margin-bottom: 6px; }}
      textarea, input[type="text"] {{
        width: 100%;
        padding: 10px 11px;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.35);
        background: rgba(2, 6, 23, 0.55);
        color: var(--text);
        outline: none;
        font-size: 13px;
        line-height: 1.35;
      }}
      textarea:focus, input[type="text"]:focus {{
        border-color: rgba(96, 165, 250, 0.9);
        box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.22);
      }}
      textarea {{ min-height: 110px; resize: vertical; }}
      .row {{
        display: grid;
        grid-template-columns: 1fr;
        gap: 12px;
      }}
      @media (min-width: 700px) {{
        .row {{ grid-template-columns: 1fr 1fr; }}
      }}
      .toggles {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px 14px;
        align-items: center;
        margin-top: 6px;
      }}
      .toggle {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 12px;
        color: var(--muted);
        user-select: none;
      }}
      .toggle input {{ width: 14px; height: 14px; }}
      .actions {{
        display: flex;
        gap: 10px;
        align-items: center;
        margin-top: 6px;
      }}
      button {{
        border: none;
        border-radius: 999px;
        padding: 10px 14px;
        font-weight: 650;
        cursor: pointer;
        color: #052e1f;
        background: linear-gradient(135deg, var(--accent), #16a34a);
      }}
      a.link {{
        color: var(--accent2);
        text-decoration: none;
        font-size: 12px;
      }}
      a.link:hover {{ text-decoration: underline; }}
      .hint {{
        color: var(--muted);
        font-size: 12px;
        line-height: 1.4;
        margin-top: 10px;
      }}
      .result h2 {{ margin: 0; font-size: 14px; }}
      pre {{
        margin: 10px 0 0;
        padding: 12px;
        background: rgba(2, 6, 23, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 14px;
        overflow: auto;
        font-size: 12px;
        line-height: 1.45;
      }}
      .pill {{
        display: inline-flex;
        gap: 8px;
        align-items: center;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(2, 6, 23, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.22);
        color: var(--muted);
        font-size: 11px;
        width: fit-content;
      }}
      .dot {{
        width: 8px; height: 8px; border-radius: 999px; background: var(--accent2);
        box-shadow: 0 0 0 4px rgba(96, 165, 250, 0.15);
      }}
    </style>
  </head>
  <body>
    <div class="wrap">
      <header>
        <div class="pill"><span class="dot"></span>Interactive evaluation & hallucination checks</div>
        <h1>LLM Evaluation & Hallucination Detection</h1>
        <p>Fill in fields and submit. Evaluation uses simple heuristic metrics; hallucination checks are rule-based unless you install transformers/torch and enable NLI.</p>
      </header>

      <div class="grid">
        <section class="card">
          <form method="post" action="/query">
            <div>
              <label for="prompt">Prompt</label>
              <textarea id="prompt" name="prompt" placeholder="Enter the prompt...">{prompt}</textarea>
            </div>
            <div>
              <label for="response">Response</label>
              <textarea id="response" name="response" placeholder="Paste the model response...">{response}</textarea>
            </div>
            <div>
              <label for="reference">Reference (optional)</label>
              <textarea id="reference" name="reference" placeholder="Ground truth / expected answer (optional)...">{reference}</textarea>
            </div>

            <div class="row">
              <div>
                <label for="claim">Claim (optional, for hallucination detection)</label>
                <input id="claim" name="claim" type="text" value="{claim}" placeholder="e.g. The Eiffel Tower is in London." />
              </div>
              <div>
                <label for="source">Source (optional, for hallucination detection)</label>
                <input id="source" name="source" type="text" value="{source}" placeholder="e.g. The Eiffel Tower is in Paris, France." />
              </div>
            </div>

            <div class="toggles">
              <label class="toggle"><input type="checkbox" name="enable_hallucination" {checked_h} />Enable hallucination check</label>
              <label class="toggle"><input type="checkbox" name="use_nli" {checked_nli} />Use NLI (if installed)</label>
            </div>

            <div class="actions">
              <button type="submit">Run</button>
              <a class="link" href="/api-docs">API docs</a>
            </div>

            <div class="hint">
              Tip: For hallucination checks, set <strong>Claim</strong> and <strong>Source</strong>. For reference-based scoring, set <strong>Reference</strong>.
            </div>
          </form>
        </section>

        <aside>
          {result_html}
        </aside>
      </div>
    </div>
  </body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse(_page())


@app.get("/api-docs", response_class=HTMLResponse)
def api_docs_redirect() -> HTMLResponse:
    return HTMLResponse(
        '<meta http-equiv="refresh" content="0; url=/docs" />',
    )


@app.post("/query", response_class=HTMLResponse)
def query(
    prompt: str = Form(""),
    response: str = Form(""),
    reference: str = Form(""),
    claim: str = Form(""),
    source: str = Form(""),
    enable_hallucination: Optional[str] = Form(None),
    use_nli: Optional[str] = Form(None),
) -> HTMLResponse:
    enable_h = enable_hallucination is not None
    use_n = use_nli is not None

    result: dict[str, Any] = {}
    if prompt.strip() or response.strip():
        result["evaluation"] = evaluate_response(
            prompt=prompt,
            response=response,
            reference=reference if reference.strip() else None,
        )

    if enable_h and claim.strip() and source.strip():
        result["hallucination"] = hallucination_result(
            claim=claim,
            source=source,
            use_nli=use_n,
        )

    return HTMLResponse(
        _page(
            prompt=prompt,
            response=response,
            reference=reference,
            claim=claim,
            source=source,
            enable_hallucination=enable_h,
            use_nli=use_n,
            result=result,
        )
    )


@app.post("/api/query")
def api_query(payload: dict[str, Any]) -> dict[str, Any]:
    prompt = str(payload.get("prompt", "") or "")
    response = str(payload.get("response", "") or "")
    reference = payload.get("reference")
    claim = str(payload.get("claim", "") or "")
    source = str(payload.get("source", "") or "")
    enable_h = bool(payload.get("enable_hallucination", False))
    use_n = bool(payload.get("use_nli", False))

    out: dict[str, Any] = {}
    if prompt.strip() or response.strip():
        out["evaluation"] = evaluate_response(
            prompt=prompt,
            response=response,
            reference=str(reference) if reference is not None and str(reference).strip() else None,
        )
    if enable_h and claim.strip() and source.strip():
        out["hallucination"] = hallucination_result(claim=claim, source=source, use_nli=use_n)
    return out

