"""
Chunking utilities — four strategies for comparison.
"""

from __future__ import annotations
from typing import Any, Dict, List
from preprocessing import is_meaningful_utterance

Record = Dict[str, Any]
Chunk  = Dict[str, Any]


def _get_text(record: Record) -> str:
    """Extract text from a record using common field names."""
    for key in ["text", "utterance", "excerpt", "content", "passage"]:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    parts = []
    for key in ["speaker", "summary", "modern_summary"]:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return " ".join(parts).strip()


def create_chunks(records: List[Record]) -> List[Chunk]:
    """
    Strategy 1 - Scene-level (baseline chunking).
    One chunk per scene, plain text only, no summary prepended.
    """
    chunks: List[Chunk] = []
    for i, record in enumerate(records):
        text = _get_text(record)
        if not text:
            continue
        chunks.append({
            "chunk_id":      record.get("source_id") or record.get("scene_id") or f"chunk_{i:06d}",
            "play":          record.get("play", record.get("play_key", "unknown")),
            "act":           record.get("act", None),
            "scene":         record.get("scene", None),
            "speaker":       record.get("speaker", None),
            "text":          text,
            "scene_summary": record.get("scene_summary", ""),
            "keywords":      record.get("keywords", []),
            "metadata":      record,
        })
    return chunks


def create_summary_enhanced_chunks(records: List[Record]) -> List[Chunk]:
    """
    Strategy 2 - Summary-enhanced scene-level.
    Prepends modern English scene summary to scene text.
    Improves retrieval for beginner queries using plain English.
    """
    chunks: List[Chunk] = []
    for i, record in enumerate(records):
        text = _get_text(record)
        if not text:
            continue
        summary = record.get("scene_summary", "")
        enhanced_text = (
            f"[Scene Summary: {summary}]\n\n{text}"
            if summary else text
        )
        chunks.append({
            "chunk_id":      record.get("source_id") or record.get("scene_id") or f"chunk_{i:06d}",
            "play":          record.get("play", record.get("play_key", "unknown")),
            "act":           record.get("act", None),
            "scene":         record.get("scene", None),
            "speaker":       record.get("speaker", None),
            "text":          enhanced_text,
            "scene_summary": summary,
            "keywords":      record.get("keywords", []),
            "metadata":      record,
        })
    return chunks


def create_utterance_chunks(utterances: List[Record]) -> List[Chunk]:
    """
    Strategy 3 - Utterance-level (fine-grained).
    One chunk per speaker turn. Skips stage directions and short lines.
    """
    SKIP_SPEAKERS = {"STAGE_DIRECTION", "ELSINORE"}
    chunks: List[Chunk] = []
    for utt in utterances:
        speaker = utt.get("speaker", "UNKNOWN")
        text    = utt.get("text", "").strip()
        if not text or speaker in SKIP_SPEAKERS or not is_meaningful_utterance(text):
            continue
        chunks.append({
            "chunk_id":      utt.get("source_id", utt.get("utterance_id", f"utt_{len(chunks):06d}")),
            "play":          utt.get("play", "unknown"),
            "act":           utt.get("act", None),
            "scene":         utt.get("scene", None),
            "speaker":       speaker,
            "text":          f"{speaker}: {text}",
            "scene_summary": utt.get("scene_summary", ""),
            "keywords":      utt.get("keywords", []),
            "metadata":      utt,
        })
    return chunks


def create_summary_enhanced_utterance_chunks(utterances: List[Record]) -> List[Chunk]:
    """
    Strategy 4 - Summary-enhanced utterance-level (combined).
    One chunk per speaker turn with the scene summary prepended.
    Combines utterance-level retrieval precision with summary semantic anchoring.
    """
    SKIP_SPEAKERS = {"STAGE_DIRECTION", "ELSINORE"}
    chunks: List[Chunk] = []
    for utt in utterances:
        speaker = utt.get("speaker", "UNKNOWN")
        text    = utt.get("text", "").strip()
        summary = utt.get("scene_summary", "").strip()
        if not text or speaker in SKIP_SPEAKERS or not is_meaningful_utterance(text):
            continue
        summary_prefix = f"[Scene: {summary}] " if summary else ""
        chunks.append({
            "chunk_id":      utt.get("source_id", utt.get("utterance_id", f"sutt_{len(chunks):06d}")),
            "play":          utt.get("play", "unknown"),
            "act":           utt.get("act", None),
            "scene":         utt.get("scene", None),
            "speaker":       speaker,
            "text":          summary_prefix + f"{speaker}: {text}",
            "scene_summary": summary,
            "keywords":      utt.get("keywords", []),
            "metadata":      utt,
        })
    return chunks


def format_chunk_for_display(chunk: Chunk) -> str:
    """Format a retrieved chunk for display to the user."""
    play    = chunk.get("play", "Unknown play")
    act     = chunk.get("act", "?")
    scene   = chunk.get("scene", "?")
    speaker = chunk.get("speaker", "")
    header  = f"{play}, Act {act}, Scene {scene}"
    if speaker:
        header += f", Speaker: {speaker}"
    return f"[{header}]\n{chunk.get('text', '')}"