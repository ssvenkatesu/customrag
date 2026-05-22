"""Quick verification: Ollama embed, in-memory vector DB, and chat."""
import os
import sys

import ollama

EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
LANGUAGE_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"
FALLBACK_CHAT_MODEL = "llama3.1:latest"
DATASET_PATH = "IPL_2026_Stats_and_Tournament_Stars.txt"


def load_dataset():
    path = DATASET_PATH
    if not os.path.exists(path) and os.path.exists("facts.txt"):
        path = "facts.txt"
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    return path, lines


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x**2 for x in a) ** 0.5
    norm_b = sum(x**2 for x in b) ** 0.5
    if norm_a * norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def main():
    print("=== Custom RAG verification ===\n")

    # 1. Dataset
    path, chunks = load_dataset()
    print(f"[OK] Dataset: {path} ({len(chunks)} chunks)")

    # 2. Ollama embedding + vector DB
    vector_db = []
    for i, chunk in enumerate(chunks[:5]):  # sample first 5 for speed
        emb = ollama.embed(model=EMBEDDING_MODEL, input=chunk)["embeddings"][0]
        vector_db.append((chunk, emb))
    print(f"[OK] Ollama embeddings via {EMBEDDING_MODEL}")
    print(f"[OK] In-memory vector DB: {len(vector_db)} vectors (sample)")

    # 3. Retrieval
    query = "Who was a top run scorer in IPL 2026?"
    q_emb = ollama.embed(model=EMBEDDING_MODEL, input=query)["embeddings"][0]
    ranked = sorted(
        ((c, cosine_similarity(q_emb, e)) for c, e in vector_db),
        key=lambda x: x[1],
        reverse=True,
    )
    top = ranked[0]
    print(f"[OK] Retrieval top match (score {top[1]:.3f}): {top[0][:80]}...")

    # 4. Chat
    context = "\n".join(f"- {c}" for c, _ in ranked[:3])
    chat_model = LANGUAGE_MODEL
    try:
        resp = ollama.chat(
            model=chat_model,
            messages=[
                {
                    "role": "system",
                    "content": f"Answer using only this context:\n{context}",
                },
                {"role": "user", "content": query},
            ],
        )
    except Exception:
        chat_model = FALLBACK_CHAT_MODEL
        resp = ollama.chat(
            model=chat_model,
            messages=[
                {
                    "role": "system",
                    "content": f"Answer using only this context:\n{context}",
                },
                {"role": "user", "content": query},
            ],
        )
    answer = resp["message"]["content"]
    print(f"[OK] Ollama chat via {chat_model}")
    print(f"\nSample answer:\n{answer[:300]}...\n")
    print("=== All checks passed ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)
