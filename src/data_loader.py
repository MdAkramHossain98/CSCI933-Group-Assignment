"""
Data loading utilities.

This file assumes that the processed Shakespeare dataset is available in JSON format.

Expected examples:
1. A file containing a list of records:
   [
     {"play": "Macbeth", "act": 1, "scene": 3, "speaker": "MACBETH", "text": "..."}
   ]

2. A file containing a dictionary with a "records" or "scenes" key:
   {"records": [...]} or {"scenes": [...]}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from config import PLAY_FILES, RAW_SCENE_FILES, RAW_UTTERANCE_FILES


Record = Dict[str, Any]


def _extract_records(obj: Any) -> List[Record]:
    """
    Extract a list of records from a JSON object.

    """
    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for key in ["records", "utterances", "scenes", "chunks", "data"]:
            if key in obj and isinstance(obj[key], list):
                return obj[key]

    raise ValueError(
        "Could not extract records. Expected a list or a dictionary containing "
        "one of: records, utterances, scenes, chunks, data."
    )


def load_json_records(path: Path) -> List[Record]:
    """
    Load one processed Shakespeare JSON file.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find dataset file: {path}\n"
            "Place the provided dataset files in data/processed/."
        )

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    records = _extract_records(obj)
    return records

def load_jsonl(path: Path) -> List[Record]:
    """
    Load a .jsonl file - one JSON object per line.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find dataset file: {path}\n"
            "Place the provided dataset files in data/processed/."
        )
    
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_all_plays() -> List[Record]:
    """
    Load scene records from the main processed JSON files of all three compulsory plays.
    """
    all_records: List[Record] = []

    for play_key, path in PLAY_FILES.items():
        records = load_json_records(path)
        for r in records:
            r.setdefault("play_key", play_key)
        all_records.extend(records)

    return all_records

def load_all_scene_chunks() -> List[Record]:
    """
    Load pre-built scene-level chunks from raw JSONL files.
    """

    all_records: List[Record] = []
    for path in RAW_SCENE_FILES:
        all_records.extend(load_jsonl(path))
    return all_records


def load_all_utterances() -> List[Record]:
    """
    Load pre-built utterance-level records from raw JSONL files.
    """

    all_records: List[Record] = []
    for path in RAW_UTTERANCE_FILES:
        all_records.extend(load_jsonl(path))
    return all_records


if __name__ == "__main__":
    scenes = load_all_scene_chunks()
    utts   = load_all_utterances()
    print(f"Scene chunks loaded:     {len(scenes)}")
    print(f"Utterance records loaded: {len(utts)}")
    print("First scene keys:")
    print(list(scenes[0].keys()))
    print("First record text sample:", scenes[0].get("text", "")[:300])
    
    print("First utterance keys:")
    print(list(utts[0].keys()))
