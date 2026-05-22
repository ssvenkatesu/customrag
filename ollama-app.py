import os
import ollama

# -----------------------------------
# Load IPL 2026 Dataset
# -----------------------------------

DATASET_PATH = 'IPL_2026_Stats_and_Tournament_Stars.txt'
FALLBACK_PATH = 'facts.txt'

if not os.path.exists(DATASET_PATH) and os.path.exists(FALLBACK_PATH):
    DATASET_PATH = FALLBACK_PATH

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f'Dataset not found. Add {DATASET_PATH} or {FALLBACK_PATH} to the project folder.'
    )

with open(DATASET_PATH, 'r', encoding='utf-8') as file:
    dataset = file.readlines()

print(f'Loaded {len(dataset)} entries from {DATASET_PATH}')


# -----------------------------------
# Models
# -----------------------------------

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


# -----------------------------------
# Vector Database
# -----------------------------------

VECTOR_DB = []


def add_chunk_to_database(chunk):

    embedding = ollama.embed(
        model=EMBEDDING_MODEL,
        input=chunk
    )['embeddings'][0]

    VECTOR_DB.append((chunk, embedding))


# Add dataset chunks into vector database
for i, chunk in enumerate(dataset):

    chunk = chunk.strip()

    if not chunk:
        continue

    add_chunk_to_database(chunk)

    print(f'Added chunk {i+1}/{len(dataset)} to the database')


# -----------------------------------
# Cosine Similarity Function
# -----------------------------------

def cosine_similarity(a, b):

    dot_product = sum([x * y for x, y in zip(a, b)])

    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5

    if int(norm_a * norm_b) == 0:
        return 0

    return dot_product / (norm_a * norm_b)


# -----------------------------------
# Retrieval Function
# -----------------------------------

def retrieve(query, top_n=5):

    query_embedding = ollama.embed(
        model=EMBEDDING_MODEL,
        input=query
    )['embeddings'][0]

    similarities = []

    for chunk, embedding in VECTOR_DB:

        similarity = cosine_similarity(query_embedding, embedding)

        similarities.append((chunk, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_n]


# -----------------------------------
# User Query
# -----------------------------------

input_query = input('Ask me a question about IPL 2026: ')

retrieved_knowledge = retrieve(input_query)

print('\nRetrieved Knowledge:\n')

for chunk, similarity in retrieved_knowledge:

    print(f' - (Similarity: {similarity:.2f}) {chunk}')


# -----------------------------------
# Prompt Construction
# -----------------------------------

instruction_prompt = f'''
You are an IPL 2026 assistant chatbot.

Use ONLY the following context to answer the question.
Do not make up any information.

Context:
{'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
'''


# -----------------------------------
# Chat with LLM
# -----------------------------------

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


# -----------------------------------
# Streaming Response
# -----------------------------------

print('\nChatbot Response:\n')

for chunk in stream:

    print(chunk['message']['content'], end='', flush=True)