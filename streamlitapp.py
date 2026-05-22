import streamlit as st
import ollama
import os

# Set page configuration
st.set_page_config(page_title="IPL 2026 RAG Chatbot", layout="centered")

st.title("🏏 IPL 2026 RAG Chatbot")
st.write("Ask questions based on IPL 2026 stats, match memos, top scorers, and tournament stars!")

# Model Configurations
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'
FALLBACK_CHAT_MODEL = 'llama3.1:latest'


def chat_model_name():
    try:
        names = {m['name'] for m in ollama.list()['models']}
        if any(LANGUAGE_MODEL in n for n in names):
            return LANGUAGE_MODEL
    except Exception:
        pass
    return FALLBACK_CHAT_MODEL

# -----------------------------
# 1. Vector Database Setup
# -----------------------------
@st.cache_resource
def initialize_vector_db():

    dataset_path = 'IPL_2026_Stats_and_Tournament_Stars.txt'
    if not os.path.exists(dataset_path) and os.path.exists('facts.txt'):
        dataset_path = 'facts.txt'

    # Fallback dataset
    if not os.path.exists(dataset_path):
        with open(dataset_path, 'w', encoding='utf-8') as f:
            f.write("Virat Kohli was among the top run scorers in IPL 2026.\n")
            f.write("Jasprit Bumrah was one of the leading wicket takers.\n")
            f.write("RCB vs CSK was one of the biggest rivalry matches.\n")
            f.write("IPL 2026 started on March 28, 2026.\n")
            f.write("Mumbai Indians and Chennai Super Kings continued their historic rivalry.\n")

    with open(dataset_path, 'r', encoding='utf-8') as file:
        dataset = file.readlines()

    vector_db = []

    status_text = st.empty()
    progress_bar = st.progress(0)

    for i, chunk in enumerate(dataset):

        chunk = chunk.strip()

        if not chunk:
            continue

        status_text.text(f"Embedding chunk {i+1}/{len(dataset)}...")

        try:
            embedding = ollama.embed(
                model=EMBEDDING_MODEL,
                input=chunk
            )['embeddings'][0]

            vector_db.append((chunk, embedding))

        except Exception as e:
            st.error(f"Error connecting to Ollama: {e}")
            return []

        progress_bar.progress((i + 1) / len(dataset))

    status_text.empty()
    progress_bar.empty()

    return vector_db


# Initialize Vector Database
with st.spinner("Initializing IPL 2026 Vector Database..."):
    VECTOR_DB = initialize_vector_db()


# -----------------------------
# 2. Cosine Similarity
# -----------------------------
def cosine_similarity(a, b):

    dot_product = sum([x * y for x, y in zip(a, b)])

    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5

    if int(norm_a * norm_b) == 0:
        return 0

    return dot_product / (norm_a * norm_b)


# -----------------------------
# 3. Retrieval Function
# -----------------------------
def retrieve(query, top_n=5):

    try:
        query_embedding = ollama.embed(
            model=EMBEDDING_MODEL,
            input=query
        )['embeddings'][0]

    except Exception as e:
        st.error(f"Failed to generate query embedding: {e}")
        return []

    similarities = []

    for chunk, embedding in VECTOR_DB:

        similarity = cosine_similarity(query_embedding, embedding)

        similarities.append((chunk, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_n]


# -----------------------------
# 4. Sidebar
# -----------------------------
with st.sidebar:

    st.header("📊 IPL Database Info")

    st.success(f"Loaded {len(VECTOR_DB)} IPL 2026 data chunks.")

    st.markdown("---")

    st.subheader("Retrieved Context")

    context_placeholder = st.empty()

    context_placeholder.info(
        "Ask any IPL 2026 related question to view retrieved context!"
    )


# -----------------------------
# 5. User Query
# -----------------------------
input_query = st.text_input(
    "Ask me about IPL 2026:",
    placeholder="e.g., Who was the top run scorer?"
)

if input_query:

    # Retrieve relevant chunks
    retrieved_knowledge = retrieve(input_query)

    # Display retrieved chunks
    with context_placeholder.container():

        for chunk, similarity in retrieved_knowledge:

            st.markdown(
                f"**Similarity Score:** `{similarity:.2f}`\n\n{chunk}"
            )

            st.markdown("---")

    # Prompt
    instruction_prompt = f"""
You are an IPL 2026 assistant chatbot.

Answer ONLY using the provided context.

Context:
{'\n'.join([f'- {chunk}' for chunk, similarity in retrieved_knowledge])}
"""

    st.subheader("🏏 Response")

    response_placeholder = st.empty()

    full_response = ""

    try:

        stream = ollama.chat(
            model=chat_model_name(),
            messages=[
                {
                    'role': 'system',
                    'content': instruction_prompt
                },
                {
                    'role': 'user',
                    'content': input_query
                },
            ],
            stream=True,
        )

        for chunk in stream:

            token = chunk['message']['content']

            full_response += token

            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

    except Exception as e:
        st.error(f"Error generating response: {e}")