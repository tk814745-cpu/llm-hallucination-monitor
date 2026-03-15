"""Response evaluator: runs enabled metrics and returns a result dict."""
from typing import Any, Optional

from llm_eval.metrics.factuality import factuality_score
from llm_eval.metrics.relevance import relevance_score
from llm_eval.metrics.exact_match import exact_match_score
from llm_eval.metrics.coherence import coherence_score
from llm_eval.metrics.length import length_stats


def evaluate_response(
    prompt: str,
    response: str,
    reference: Optional[str] = None,
    *,
    metrics: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Evaluate a single (prompt, response, reference) triple.
    metrics: list of metric names to run; default all.
    """
    if metrics is None:
        metrics = ["factuality", "relevance", "exact_match", "coherence", "length"]
    ref = reference or ""
    result: dict[str, Any] = {"prompt": prompt, "response": response, "reference": ref}

    if "factuality" in metrics and ref:
        result["factuality"] = factuality_score(response, ref)
    if "relevance" in metrics:
        result["relevance"] = relevance_score(prompt, response)
    if "exact_match" in metrics and ref:
        result["exact_match"] = exact_match_score(response, ref)
    if "coherence" in metrics:
        result["coherence"] = coherence_score(response)
    if "length" in metrics:
        result["length"] = length_stats(response)

    return result
