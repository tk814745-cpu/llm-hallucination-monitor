"""Relevance: how much the response relates to the prompt (lexical overlap)."""
from llm_eval.metrics.factuality import _token_set


def relevance_score(prompt: str, response: str) -> float:
    """
    Simple relevance: Jaccard similarity between prompt and response token sets.
    Returns 0.0 to 1.0.
    """
    if not prompt.strip() or not response.strip():
        return 0.0
    p_tokens = _token_set(prompt)
    r_tokens = _token_set(response)
    if not p_tokens and not r_tokens:
        return 1.0
    if not p_tokens or not r_tokens:
        return 0.0
    inter = len(p_tokens & r_tokens)
    union = len(p_tokens | r_tokens)
    return round(inter / union, 4) if union else 0.0
