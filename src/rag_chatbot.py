"""
RAG chatbot scaffold.
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple

import requests

from config import DEFAULT_TOP_K, EMBEDDING_MODEL_NAME, PROMPT_DIR, GROQ_API_KEY
from data_loader import load_all_scene_chunks, load_all_utterances
from chunking import create_summary_enhanced_utterance_chunks, format_chunk_for_display
#from chunking import create_summary_enhanced_chunks, format_chunk_for_display
from preprocessing import preprocess_records
from retrieval import EmbeddingRetriever


Chunk = Dict[str, Any]


def load_system_prompt() -> str:
    prompt_path = PROMPT_DIR / "system_prompt.txt"
    return prompt_path.read_text(encoding="utf-8")


def build_rag_prompt(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    system_prompt = load_system_prompt()

    context_blocks = []
    for rank, (chunk, score) in enumerate(retrieved, start=1):
        context_blocks.append(
            f"[Context {rank} | {chunk['play']}, Act {chunk['act']}, Scene {chunk['scene']}]\n"
            f"{format_chunk_for_display(chunk)}"
        )
    context = "\n\n".join(context_blocks)

    prompt = f"""{system_prompt}

You are a Shakespeare-aware assistant helping a beginner understand the plays.

Answer the question below using ONLY the retrieved context as your evidence.

Rules:
1. Directly answer "{query}" in your first sentence in plain English.
2. Write 4-6 sentences total.
3. Do NOT include direct quotes from the text.
4. Do NOT use words like "suggests", "implies", "evident", "as seen in".
5. Treat the retrieved context as background knowledge — extract the meaning, not the words.
6. Do NOT summarise scenes or reference act/scene numbers.

Retrieved context:
{context}

Answer:
"""
    return prompt


def build_stylised_prompt(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    context = "\n\n".join([
        f"[{c['play']}, Act {c['act']}, Scene {c['scene']}]\n{c['text'][:300]}"
        for c, _ in retrieved
    ])
    return f"""You are Shakespeare-aware assistant. Write a SHORT creative response (4-6 lines only) in Elizabethan verse.
Rules:
- Use thee, thou, dost, hath, wherefore, doth, art, thy style
- Maximum 6 lines, no more
- Do NOT explain or summarise - just write the verse
- Do NOT add labels like "Creative response:" - just write the verse immediately

Context from the plays:
{context}

Request: {query}

Respond in verse NOW (4-6 lines only):"""

def generate_answer(prompt: str) -> str:
    if len(prompt) > 5000:
            prompt = prompt[:5000] + "\n\n[Context truncated]\n\nAnswer:"
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.7
        }
    )

    return response.json()["choices"][0]["message"]["content"]


def main() -> None:
    print("Loading Shakespeare corpus...")
    utt_records = load_all_utterances()
    utt_records = preprocess_records(utt_records, mode="utterance")
    chunks      = create_summary_enhanced_utterance_chunks(utt_records)

    #records = load_all_scene_chunks()
    #records = preprocess_records(records, mode="scene")
    #chunks  = create_summary_enhanced_chunks(records)

    retriever = EmbeddingRetriever(EMBEDDING_MODEL_NAME)
    retriever.build_index(chunks)

    print(f"Index built: {len(chunks)} scene chunks.")
    print("Commands: type 'style: <your question>' for Shakespearean style | 'quit' to exit\n")

    while True:
        query = input("Question: ").strip()
        if not query:
            continue
        if query.lower() in {"quit", "exit"}:
            break
        
        stylised_mode = query.lower().startswith("style:")
        if stylised_mode:
            query = query[6:].strip()

        retrieved = retriever.retrieve(query, top_k=DEFAULT_TOP_K)

        print("\nRETRIEVED EVIDENCE:")
        for rank, (chunk, score) in enumerate(retrieved, start=1):
            print(f"  {'─'*80}")
            print(f"  Rank {rank} | Score: {score:.4f}")
            print(f"  {chunk['play']} | Act {chunk['act']} | Scene {chunk['scene']}")
            if chunk.get("scene_summary"):
                print(f"  Summary: {chunk['scene_summary']}")
            print(f"  Text: {chunk['text'][:200]}...")

        if stylised_mode:
            prompt = build_stylised_prompt(query, retrieved)
            print("\nSTYLISED RESPONSE:")
        else:
            prompt = build_rag_prompt(query, retrieved)
            print("\nANSWER:")

        print(generate_answer(prompt))
        print()


if __name__ == "__main__":
    main()


