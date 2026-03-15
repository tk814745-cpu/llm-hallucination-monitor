"""Coherence: simple proxy via sentence count and length (no semantic model)."""
import re
from typing import Any


def coherence_score(response: str) -> dict[str, Any]:
    """
    Heuristic coherence stats: sentence count and avg length.
    No semantic model; useful as a basic sanity check.
    """
    if not response or not response.strip():
        return {"sentence_count": 0, "avg_sentence_length": 0.0, "score": 0.0}
    sentences = [s.strip() for s in re.split(r"[.!?]+", response) if s.strip()]
    n = len(sentences)
    if n == 0:
        return {"sentence_count": 0, "avg_sentence_length": 0.0, "score": 0.0}
    total_chars = sum(len(s) for s in sentences)
    avg = total_chars / n
    # Simple score: cap at 1.0 for reasonable sentence length (e.g. 20–200 chars)
    score = min(1.0, avg / 100.0) if avg > 0 else 0.0
    return {
        "sentence_count": n,
        "avg_sentence_length": round(avg, 2),
        "score": round(score, 4),
    }
