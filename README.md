# Shakespeare-Aware RAG System

## Overview

This project implements a Shakespeare-aware Retrieval-Augmented Generation (RAG) system designed to help users with little or no prior Shakespeare knowledge understand and interact with three plays:

- Hamlet
- Macbeth
- Romeo and Juliet

The system combines semantic retrieval and language-model generation to provide grounded, beginner-friendly answers supported by evidence from the source texts. In addition to factual question answering, the system can generate short Shakespearean-style responses when requested.

The project also includes a baseline system without retrieval to evaluate the effectiveness of the RAG approach.

---

## Features

### Concept Explanation

Examples:

- Who is Hamlet?
- What is the role of Lady Macbeth?
- What is the conflict between the Montagues and the Capulets?

### Contextual Question Answering

Examples:

- Why does Macbeth kill Duncan?
- Why does Hamlet delay taking revenge?
- Why is Juliet conflicted after Romeo kills Tybalt?

### Evidence-Based Retrieval

For each answer, the system retrieves relevant passages from the Shakespeare corpus and displays supporting evidence with metadata.

### Shakespearean Style Generation

The system can generate short creative responses in a Shakespearean style while clearly separating creative output from factual answers.

### Evaluation Framework

The project includes:

- Instructor-provided evaluation questions
- Group-designed evaluation questions
- Baseline vs RAG comparison
- Structured evaluation results

---

## System Architecture

```text
User Question
      │
      ▼
Query Embedding
      │
      ▼
Semantic Retrieval
      │
      ▼
Top-k Relevant Chunks
      │
      ▼
Prompt Construction
      │
      ▼
Language Model
      │
      ▼
Answer + Retrieved Evidence
```

### Retrieval Component

- Embedding Model:
  - sentence-transformers/all-MiniLM-L6-v2

- Retrieval Method:
  - Cosine Similarity

- Retrieval Strategy:
  - Summary-enhanced chunk retrieval

### Generation Component

- Language Model:
  - Llama 3.1 8B Instant

- API Provider:
  - Groq API

---

## Dataset

The project uses the instructor-provided Shakespeare dataset containing:

- Hamlet
- Macbeth
- Romeo and Juliet

The dataset includes:

- Scene metadata
- Speaker information
- Scene summaries
- Keywords
- Source identifiers
- Original Shakespeare text

Both scene-level and utterance-level formats are provided.

---

## Project Structure

```text
assignment2/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── prompts/
│   └── system_prompt.txt
│
├── results/
│   └── evaluation_results.csv
│
├── src/
│   ├── app.py
│   ├── baseline.py
│   ├── build_index.py
│   ├── chunking.py
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── rag_chatbot.py
│   └── retrieval.py
│
├── report/
│
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone <repository_url>
cd assignment2
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Recommended packages include:

```bash
sentence-transformers
scikit-learn
numpy
requests
pandas
streamlit
```

---

## Configuration

Open:

```text
src/config.py
```

Set your API key:

```python
GROQ_API_KEY = "YOUR_API_KEY"
```

---

## Running the System

### Launch Interactive Chatbot

#### For Terminal Interface
```bash
python src/rag_chatbot.py
```

#### For Web Application Interface
```bash
streamlit run app.py
```

#### Live Demo

You can access the deployed Streamlit application here:

[Open the Shakespeare RAG Chatbot](https://csci933-group-assignment.streamlit.app/)

Example:

```text
Question: Why does Macbeth kill Duncan?
```

The system will:

1. Retrieve relevant evidence.
2. Display retrieved passages.
3. Generate a grounded answer.

### Shakespeare Style Mode

```text
style: Write a warning from Macbeth
```

---

## Baseline System

Run:

```bash
python src/baseline.py
```

The baseline sends questions directly to the language model without retrieval.

Purpose:

- Demonstrate the impact of retrieval augmentation.
- Provide comparison against the RAG system.

---

## Evaluation

Run:

```bash
python src/evaluate.py
```

This will:

- Load instructor questions
- Load group-designed questions
- Run baseline responses
- Run RAG responses
- Save outputs to:

```text
results/evaluation_results.csv
```

Evaluation criteria:

- Correctness
- Grounding
- Retrieval Relevance
- Usefulness
- Style Quality

---

## Design Decisions

### Why Retrieval-Augmented Generation?

Shakespearean language differs significantly from modern English. A retrieval-based approach allows the language model to access relevant evidence directly from the source material rather than relying solely on pretrained knowledge.

### Why Summary-Enhanced Retrieval?

Scene summaries improve retrieval quality for beginner-oriented questions by providing modern-English context while preserving the original Shakespearean text as evidence.

### Why MiniLM?

The selected embedding model is lightweight, efficient, and suitable for resource-constrained environments, aligning with the assignment’s Small Language Model objectives.

---

## Limitations

- Retrieval quality depends on chunking strategy.
- Questions spanning multiple scenes may require broader context.
- Creative generation may occasionally overemphasize style.
- The system remains limited to the three assigned plays.

---
