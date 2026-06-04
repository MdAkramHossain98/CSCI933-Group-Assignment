"""
Configuration for the Assignment 2 code.

"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data" / "processed"
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
PROMPT_DIR   = PROJECT_ROOT / "prompts"
RESULTS_DIR  = PROJECT_ROOT / "results"

PLAY_FILES = {
    "hamlet": DATA_DIR / "hamlet.json",
    "macbeth": DATA_DIR / "macbeth.json",
    "romeo_and_juliet": DATA_DIR / "romeo_and_juliet.json",
}

RAW_SCENE_FILES = [
    RAW_DIR / "hamlet_scene_chunks.jsonl",
    RAW_DIR / "macbeth_scene_chunks.jsonl",
    RAW_DIR / "romeo_and_juliet_scene_chunks.jsonl",
]

RAW_UTTERANCE_FILES = [
    RAW_DIR / "hamlet_utterances.jsonl",
    RAW_DIR / "macbeth_utterances.jsonl",
    RAW_DIR / "romeo_and_juliet_utterances.jsonl",
]

INSTRUCTOR_QUESTIONS_PATH = RESULTS_DIR / "instructor_questions.json"

DEFAULT_TOP_K = 5

GROQ_API_KEY = "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"

# Lightweight embedding model.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
