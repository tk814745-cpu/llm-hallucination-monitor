"""Factuality: overlap and containment of key information vs reference."""
import re
from typing import Optional


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _token_set(s: str) -> set[str]:
    return set(_normalize(s).split())


def factuality_score(response: str, reference: str) -> float:
    """
    Simple lexical factuality: fraction of reference tokens that appear in response.
    Returns 0.0 to 1.0. Use with reference as ground truth.
    """
    if not reference or not reference.strip():
        return 1.0
    ref_tokens = _token_set(reference)
    resp_tokens = _token_set(response)
    if not ref_tokens:
        return 1.0
    overlap = len(ref_tokens & resp_tokens) / len(ref_tokens)
    return round(overlap, 4)
