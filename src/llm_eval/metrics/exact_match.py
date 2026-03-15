"""Exact match: strict string equality after normalization."""
from llm_eval.metrics.factuality import _normalize


def exact_match_score(response: str, reference: str) -> float:
    """
    Returns 1.0 if normalized response equals normalized reference, else 0.0.
    """
    if not reference and not response:
        return 1.0
    return 1.0 if _normalize(response) == _normalize(reference) else 0.0
