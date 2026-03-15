"""Hallucination detection: claim vs source (NLI or rule-based fallback)."""
from typing import Any, Optional

# Optional NLI model; fallback to rule-based if not available
_NLI_PIPELINE = None


def _load_nli():
    """Optional: load NLI model for claim vs source. Requires: pip install transformers torch."""
    global _NLI_PIPELINE
    if _NLI_PIPELINE is not None:
        return _NLI_PIPELINE
    try:
        from transformers import pipeline
        _NLI_PIPELINE = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,
        )
        return _NLI_PIPELINE
    except Exception:
        return None


def _nli_predict(claim: str, source: str) -> tuple[float, str]:
    """Run NLI: does source entail claim? Falls back to rule-based if NLI unavailable."""
    pipe = _load_nli()
    if pipe is None:
        return _rule_based_score(claim, source)
    try:
        # Format as single sequence for zero-shot; template ties claim to source
        sequence = f"Source: {source[:400]}. Claim: {claim[:200]}."
        out = pipe(
            sequence,
            candidate_labels=["entailment", "neutral", "contradiction"],
            multi_label=False,
        )
        labels = out.get("labels", [])
        scores = out.get("scores", [])
        for lab, sc in zip(labels, scores):
            lab = (lab or "").lower()
            if "entail" in lab:
                return round(float(sc), 4), "support"
            if "contradict" in lab:
                return round(1 - float(sc), 4), "contradiction"
        return round(float(scores[0]) if scores else 0.5, 4), "neutral"
    except Exception:
        return _rule_based_score(claim, source)


def _rule_based_score(claim: str, source: str) -> tuple[float, str]:
    """Fallback: simple token overlap; low overlap => possible hallucination."""
    claim_tokens = set(claim.lower().split())
    source_tokens = set(source.lower().split())
    if not claim_tokens:
        return 1.0, "neutral"
    overlap = len(claim_tokens & source_tokens) / len(claim_tokens)
    if overlap >= 0.6:
        return round(overlap, 4), "support"
    if overlap <= 0.2:
        return round(1 - overlap, 4), "contradiction"
    return round(overlap, 4), "neutral"


def detect_hallucination(
    claim: str,
    source: str,
    *,
    use_nli: bool = True,
    nli_model: Optional[str] = None,
) -> tuple[float, str]:
    """
    Detect if `claim` is supported by `source`.
    Returns (score, label) where label is one of: support, contradiction, neutral.
    score: 0–1; higher = more support (less hallucination).
    """
    claim = (claim or "").strip()
    source = (source or "").strip()
    if not claim:
        return 1.0, "neutral"
    if not source:
        return 0.0, "neutral"

    if use_nli:
        return _nli_predict(claim, source)
    return _rule_based_score(claim, source)


def hallucination_result(
    claim: str,
    source: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience: return dict with score, label, and is_hallucination."""
    score, label = detect_hallucination(claim, source, **kwargs)
    return {
        "claim": claim,
        "source": source,
        "score": score,
        "label": label,
        "is_hallucination": label == "contradiction" or (label == "neutral" and score < 0.5),
    }
