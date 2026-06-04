from __future__ import annotations

import csv
import json
import time
import requests
from pathlib import Path
from typing import Dict, List

from config import RESULTS_DIR, GROQ_API_KEY, DEFAULT_TOP_K, EMBEDDING_MODEL_NAME, INSTRUCTOR_QUESTIONS_PATH
from data_loader import load_all_scene_chunks, load_all_utterances
from chunking import create_summary_enhanced_utterance_chunks, format_chunk_for_display
#from chunking import create_summary_enhanced_chunks, format_chunk_for_display
from preprocessing import preprocess_records
from retrieval import EmbeddingRetriever
from baseline import baseline_answer
from rag_chatbot import build_rag_prompt, build_stylised_prompt, generate_answer


OUTPUT_PATH = RESULTS_DIR / "evaluation_results.csv"

"""
# Group-designed questions - sample questions. need change
"""

GROUP_QUESTIONS = [
    {
        "id": "G1", "play": "Hamlet",
        "question": "Who is Ophelia and what happens to her?",
        "expected_focus": "Ophelia's role, her madness caused by Hamlet and her father's death, and her drowning.",
        "type": "concept_explanation"
    },
    {
        "id": "G2", "play": "Macbeth",
        "question": "What role do the witches play in Macbeth?",
        "expected_focus": "The witches provide prophecies that ignite Macbeth's ambition and contribute to his downfall.",
        "type": "concept_explanation"
    },
    {
        "id": "G3", "play": "Romeo and Juliet",
        "question": "How does the story of Romeo and Juliet end?",
        "expected_focus": "Both Romeo and Juliet die in the tomb; their deaths reconcile the feuding families.",
        "type": "contextual_qa"
    },
    {
        "id": "G4", "play": "Hamlet",
        "question": "What is the significance of the play within a play in Hamlet?",
        "expected_focus": "Hamlet stages the play to confirm Claudius's guilt as his father's murderer.",
        "type": "contextual_qa"
    },
    {
        "id": "G5", "play": "Macbeth",
        "question": "Generate a short Shakespearean-style response from Macbeth before killing Duncan.",
        "expected_focus": "Creative stylised output showing Macbeth's internal conflict between ambition and conscience.",
        "type": "stylised_generation"
    },
    {
        "id": "G6", "play": "Romeo and Juliet",
        "question": "Why do Romeo and Juliet keep their relationship a secret?",
        "expected_focus": "Their families are sworn enemies; revealing the relationship would end it and endanger them both.",
        "type": "contextual_qa"
    },
    {
        "id": "G7", "play": "Macbeth",
        "question": "How does Lady Macbeth influence Macbeth's decision to kill Duncan?",
        "expected_focus": "Lady Macbeth manipulates Macbeth by questioning his courage and driving his ambition.",
        "type": "concept_explanation"
    },
    {
        "id": "G8", "play": "Hamlet",
        "question": "What is Hamlet's relationship with his mother Gertrude?",
        "expected_focus": "Hamlet is hurt and angry that Gertrude remarried Claudius so quickly after his father's death.",
        "type": "concept_explanation"
    },
    {
        "id": "G9", "play": "Romeo and Juliet",
        "question": "Generate a short Shakespearean-style speech from Romeo when he first sees Juliet.",
        "expected_focus": "Creative stylised output capturing Romeo's wonder and immediate infatuation with Juliet.",
        "type": "stylised_generation"
    },
    {
        "id": "G10", "play": "Macbeth",
        "question": "What happens to Macbeth at the end of the play?",
        "expected_focus": "Macbeth is killed by Macduff in battle; Malcolm is restored as the rightful king of Scotland.",
        "type": "contextual_qa"
    },
]


def load_instructor_questions(path: Path = INSTRUCTOR_QUESTIONS_PATH) -> List[Dict]:

    if not path.exists():
        raise FileNotFoundError(f"Question file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data.get("questions", [])
    return data


def rag_answer(query: str, retriever: EmbeddingRetriever, question_type: str = "") -> tuple:
    """
    Run one question through the RAG pipeline.

    Uses build_stylised_prompt for stylised_generation questions and
    build_rag_prompt for all others — identical to rag_chatbot.py behaviour.

    Returns (answer_text, passages_summary).
    """
    retrieved = retriever.retrieve(query, top_k=DEFAULT_TOP_K)

    passages = " | ".join(
        f"[{c['play']} Act{c['act']} Sc{c['scene']}]: {c['text'][:100]}"
        for c, _ in retrieved
    )

    if question_type == "stylised_generation":
        prompt = build_stylised_prompt(query, retrieved)
    else:
        prompt = build_rag_prompt(query, retrieved)

    answer = generate_answer(prompt)
    return answer, passages


def run_evaluation() -> None:
    print("Loading corpus and building index...")
    utt_records = load_all_utterances()
    utt_records = preprocess_records(utt_records, mode="utterance")
    chunks = create_summary_enhanced_utterance_chunks(utt_records)

    retriever = EmbeddingRetriever(EMBEDDING_MODEL_NAME)
    retriever.build_index(chunks)
    print(f"Index ready: {len(chunks)} chunks.\n")

    instructor_qs = load_instructor_questions()
    all_questions = instructor_qs + GROUP_QUESTIONS
    print(
        f"Total questions: {len(all_questions)} "
        f"({len(instructor_qs)} instructor + {len(GROUP_QUESTIONS)} group)\n"
    )

    fieldnames = [
        "question_id", "question", "play", "question_type", "expected_focus",
        "system", "retrieved_passages", "generated_response",
        "correctness_score", "grounding_score", "retrieval_relevance_score",
        "usefulness_score", "style_quality_score", "comments",
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for q in all_questions:
        qid      = q.get("id", q.get("question_id", "?"))
        question = q.get("question", "")
        play     = q.get("play", "")
        qtype    = q.get("type", q.get("question_type", ""))
        focus    = q.get("expected_focus", "")

        print(f"[{qid}] {question[:65]}...")

        # ── Baseline (prompt-only, no retrieval) ──────────────────────────
        print("  → Baseline...", end=" ", flush=True)
        try:
            b_answer = baseline_answer(question)
        except Exception as exc:
            b_answer = f"[Error: {exc}]"
        print("done")

        rows.append({
            "question_id": qid, "question": question, "play": play,
            "question_type": qtype, "expected_focus": focus,
            "system": "baseline",
            "retrieved_passages": "N/A - no retrieval",
            "generated_response": b_answer,
            "correctness_score": "", "grounding_score": "",
            "retrieval_relevance_score": "", "usefulness_score": "",
            "style_quality_score": "", "comments": "",
        })

        time.sleep(5)

        # ── RAG ───────────────────────────────────────────────────────────
        print("  → RAG...", end=" ", flush=True)
        try:
            r_answer, r_passages = rag_answer(question, retriever, qtype)
        except Exception as exc:
            r_answer, r_passages = f"[Error: {exc}]", ""
        print("done\n")

        rows.append({
            "question_id": qid, "question": question, "play": play,
            "question_type": qtype, "expected_focus": focus,
            "system": "rag",
            "retrieved_passages": r_passages,
            "generated_response": r_answer,
            "correctness_score": "", "grounding_score": "",
            "retrieval_relevance_score": "", "usefulness_score": "",
            "style_quality_score": "", "comments": "",
        })

        time.sleep(5)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved to: {OUTPUT_PATH}")
    print("Open the CSV and fill in the scores (1-5) for each row.")


if __name__ == "__main__":
    run_evaluation()
