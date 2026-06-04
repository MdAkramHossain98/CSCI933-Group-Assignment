"""
Baseline system.

Students must implement a baseline for comparison with the RAG system.
A baseline may be:
- prompt-only generation without retrieval;
- simple keyword search;
- retrieval-only response without generation;
- another justified minimal approach.

The baseline must be described and compared against the improved RAG system.
"""

from __future__ import annotations
import requests
from config import GROQ_API_KEY

def baseline_answer(query: str) -> str:
    """
    Prompt-only baseline - no retrieval.
    Sends the question directly to the LLM with no Shakespeare context.
    Used to demonstrate whether RAG improves grounding over pure LLM memory.
    """
    
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": f"Answer this Shakespeare question briefly: {query}"}],
            "max_tokens": 300,
            "temperature": 0.7
        }
    )

    result = response.json()
    if "choices" not in result:
        return f"[API Error] {result.get('error', {}).get('message', str(result))}"
    return result["choices"][0]["message"]["content"]



if __name__ == "__main__":
    question = "Who is Hamlet?"
    print("Question:", question)
    print("Answer:", baseline_answer(question))
