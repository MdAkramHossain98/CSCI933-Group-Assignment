"""
Build and compare all four chunking strategies.
"""

from config import DEFAULT_TOP_K, EMBEDDING_MODEL_NAME
from data_loader import load_all_scene_chunks, load_all_utterances
from preprocessing import preprocess_records
from chunking import (
    create_chunks,
    create_summary_enhanced_chunks,
    create_utterance_chunks,
    create_summary_enhanced_utterance_chunks,
    format_chunk_for_display,
)
from retrieval import EmbeddingRetriever


def test_strategy(name, chunks, query, top_k=DEFAULT_TOP_K):
    print(f"\n{'='*80}")
    print(f"  STRATEGY: {name}  |  Total chunks: {len(chunks)}")
    print(f"{'='*80}")
    retriever = EmbeddingRetriever(EMBEDDING_MODEL_NAME)
    retriever.build_index(chunks)
    results = retriever.retrieve(query, top_k=top_k)
    print(f"\nQuery: {query}\n")
    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"--- Rank {rank} | Score: {score:.4f} ---")
        print(f"  {chunk['play']} | Act {chunk['act']} | Scene {chunk['scene']}")
        if chunk.get("scene_summary"):
            print(f"  Summary: {chunk['scene_summary']}")
        if chunk.get("speaker"):
            print(f"  Speaker: {chunk['speaker']}")
        print(f"  Text preview: {chunk['text'][:200]}\n")


def main():
    scene_records = load_all_scene_chunks()
    scene_records = preprocess_records(scene_records, mode="scene")
    utt_records   = load_all_utterances()
    utt_records   = preprocess_records(utt_records, mode="utterance")
    query         = "Why does Macbeth kill Duncan?"

    scene_chunks    = create_chunks(scene_records)
    enhanced_chunks = create_summary_enhanced_chunks(scene_records)
    utt_chunks      = create_utterance_chunks(utt_records)
    combined_chunks = create_summary_enhanced_utterance_chunks(utt_records)

    test_strategy("1. Scene-level (baseline)",             scene_chunks,    query)
    test_strategy("2. Summary-enhanced (improved)",        enhanced_chunks, query)
    test_strategy("3. Utterance-level (fine-grained)",     utt_chunks,      query)
    test_strategy("4. Summary+Utterance (combined)",       combined_chunks, query)

    print(f"\n{'='*80}")
    print(f"  Scene-level chunks:              {len(scene_chunks)}")
    print(f"  Summary-enhanced chunks:         {len(enhanced_chunks)}")
    print(f"  Utterance-level chunks:          {len(utt_chunks)}")
    print(f"  Summary+Utterance chunks:        {len(combined_chunks)}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()