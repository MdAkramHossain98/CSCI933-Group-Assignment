import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR      = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from config      import DEFAULT_TOP_K, EMBEDDING_MODEL_NAME
from data_loader import load_all_utterances
from chunking    import create_summary_enhanced_utterance_chunks
from preprocessing import preprocess_records
from retrieval   import EmbeddingRetriever
from rag_chatbot import build_rag_prompt, build_stylised_prompt, generate_answer
from baseline    import baseline_answer

st.set_page_config(
    page_title="Shakespeare RAG Chatbot",
    layout="wide",
)

st.title("Shakespeare-Aware RAG Chatbot")
st.write("Ask questions about **Hamlet**, **Macbeth**, and **Romeo and Juliet**.")


@st.cache_resource
def load_system():
    """Load corpus, build chunks, and build embedding index (cached across reruns)."""
    utt_records = load_all_utterances()
    utt_records = preprocess_records(utt_records, mode="utterance")
    chunks      = create_summary_enhanced_utterance_chunks(utt_records)
    retriever   = EmbeddingRetriever(EMBEDDING_MODEL_NAME)
    retriever.build_index(chunks)
    return chunks, retriever


chunks, retriever = load_system()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("🎛️ Options")

mode = st.sidebar.radio(
    "Choose system",
    ["RAG System", "Baseline System"],
    help="RAG retrieves Shakespeare passages before answering. "
         "Baseline uses LLM knowledge only.",
)

response_style = None
if mode == "RAG System":
    response_style = st.sidebar.radio(
        "Response style",
        ["Plain English", "Shakespearean Style"],
        help="Shakespearean Style produces creative verse — not factual evidence.",
    )

# ── Main input ────────────────────────────────────────────────────────────────
# key= binds the widget to st.session_state["question"] automatically.
# Do NOT manually pre-set st.session_state["question"] — that causes
# StreamlitAPIException in Streamlit >= 1.28.
question = st.text_area(
    "Enter your question:",
    height=100,
    placeholder="For instance — Why does Macbeth kill Duncan?",
    key="question",
)

if st.button("Generate Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):

            # ── Baseline (prompt-only) ────────────────────────────────────
            if mode == "Baseline System":
                st.subheader("Baseline Answer")
                st.caption("No retrieval — answer based on LLM knowledge only.")
                try:
                    answer = baseline_answer(question)
                    st.write(answer)
                except Exception as exc:
                    st.error(f"API Error: {exc}")
                st.info("Baseline does not use retrieved Shakespeare evidence.")

            # ── RAG ───────────────────────────────────────────────────────
            else:
                retrieved = retriever.retrieve(question, top_k=DEFAULT_TOP_K)
                stylised  = response_style == "Shakespearean Style"

                if stylised:
                    prompt = build_stylised_prompt(question, retrieved)
                    label  = "Stylised Response"
                else:
                    prompt = build_rag_prompt(question, retrieved)
                    label  = "RAG Answer"

                try:
                    answer = generate_answer(prompt)
                    st.subheader(label)

                    if stylised:
                        st.warning(
                            "Creative Shakespearean-style verse — not factual evidence."
                        )
                        # Render each verse line on its own line in italics
                        raw_lines = [ln.strip() for ln in answer.split("\n") if ln.strip()]
                        poem = "  \n".join(raw_lines)
                        st.markdown(f"*{poem}*")
                    else:
                        # Plain-English RAG answer
                        st.write(answer)

                except Exception as exc:
                    st.error(f"API Error: {exc}")

                # Retrieved evidence expanders (always shown for RAG)
                st.subheader("Retrieved Evidence")
                for rank, (chunk, score) in enumerate(retrieved, start=1):
                    with st.expander(
                        f"Evidence {rank} | {chunk['play']} "
                        f"Act {chunk['act']} Scene {chunk['scene']} "
                        f"| Similarity: {score:.4f}"
                    ):
                        if chunk.get("scene_summary"):
                            st.caption(f"Summary: {chunk['scene_summary']}")
                        st.text(chunk["text"][:600])
