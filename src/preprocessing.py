"""
Text preprocessing for Shakespeare RAG pipeline.
Cleans raw scene and utterance text before chunking.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List

Record = Dict[str, Any]


def clean_text(text: str) -> str:
    # Remove content inside square brackets [stage directions]
    text = re.sub(r'\[.*?\]', '', text)

    # Remove ALL-CAPS location headers at start of line (e.g. "ELSINORE. A platform...")
    # Only remove lines where the whole line is a location, not speaker lines
    text = re.sub(r'^[A-Z][A-Z ]+\.\s+[A-Z][a-z].*$', '', text, flags=re.MULTILINE)

    # Normalise multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Normalise multiple spaces (but NOT the double-space speaker format)
    text = re.sub(r'[ \t]{3,}', ' ', text)

    return text.strip()


def is_meaningful_utterance(text: str, min_words: int = 5) -> bool:
    """
    For utterance-level chunks:
    - Rejects lines that are too short
    - Rejects non-alphabetic fragments
    - Rejects leftover stage directions
    """
    if re.match(r'^\[.*\]$', text.strip()):
        return False
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.5:
        return False
    if len(text.split()) < min_words:
        return False
    return True


def preprocess_records(records: List[Record], mode: str = "scene") -> List[Record]:
    """
    Apply cleaning to records.
    mode="scene"     → clean_text() on 'text' field (for scene chunks)
    mode="utterance" → clean_text() + is_meaningful_utterance() filter
    """
    cleaned = []
    for record in records:
        r = dict(record)
        if "text" in r and isinstance(r["text"], str):
            r["text"] = clean_text(r["text"])
            if mode == "utterance" and not is_meaningful_utterance(r["text"]):
                continue  # drop noise utterances
        cleaned.append(r)
    return cleaned