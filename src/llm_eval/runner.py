"""Batch evaluation runner: read JSONL, run eval and/or hallucination detection, write results."""
import json
from pathlib import Path
from typing import Any, Optional

from llm_eval.evaluation.evaluator import evaluate_response
from llm_eval.hallucination.detector import hallucination_result


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config; return empty dict if missing or invalid."""
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def run_eval(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    config_path: Optional[str | Path] = None,
    *,
    config: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """
    Read JSONL from input_path; for each line run evaluation and/or hallucination detection.
    If output_path is set, write JSONL results there. Returns list of result dicts.
    """
    cfg = config or load_config(config_path or "")
    metrics = cfg.get("metrics") or ["factuality", "relevance", "exact_match", "coherence", "length"]
    run_hallucination = cfg.get("hallucination", {}).get("enabled", False)
    use_nli = cfg.get("hallucination", {}).get("use_nli", False)

    results: list[dict[str, Any]] = []
    input_path = Path(input_path)

    with open(input_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                results.append({"index": i, "error": "Invalid JSON", "raw": line[:200]})
                continue

            out: dict[str, Any] = {"index": i, **row}

            if "prompt" in row and "response" in row:
                ev = evaluate_response(
                    prompt=row["prompt"],
                    response=row["response"],
                    reference=row.get("reference"),
                    metrics=metrics,
                )
                # Store only metric fields under evaluation (avoid duplicating prompt/response)
                out["evaluation"] = {k: v for k, v in ev.items() if k not in ("prompt", "response", "reference")}

            if run_hallucination and "claim" in row and "source" in row:
                out["hallucination"] = hallucination_result(
                    claim=row["claim"],
                    source=row["source"],
                    use_nli=use_nli,
                )

            results.append(out)

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute simple aggregate stats over evaluation results."""
    evals = [r.get("evaluation") for r in results if isinstance(r.get("evaluation"), dict)]
    if not evals:
        return {"count": len(results), "evaluations": 0}

    agg: dict[str, list[float]] = {}
    for e in evals:
        for k, v in e.items():
            if k in ("prompt", "response", "reference"):
                continue
            if isinstance(v, (int, float)):
                agg.setdefault(k, []).append(float(v))
            elif isinstance(v, dict) and "score" in v:
                agg.setdefault(k, []).append(float(v["score"]))

    summary: dict[str, Any] = {"count": len(results), "evaluations": len(evals)}
    for k, vals in agg.items():
        if vals:
            summary[f"{k}_mean"] = round(sum(vals) / len(vals), 4)
            summary[f"{k}_min"] = round(min(vals), 4)
            summary[f"{k}_max"] = round(max(vals), 4)
    return summary
