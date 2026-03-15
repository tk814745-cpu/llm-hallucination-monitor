"""Length statistics for response."""
from typing import Any


def length_stats(response: str) -> dict[str, Any]:
    """Word and character counts."""
    text = (response or "").strip()
    words = text.split()
    return {
        "char_count": len(text),
        "word_count": len(words),
    }
