import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_index():
    index = faiss.read_index("faiss_index.bin")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def retrieve(query, top_k=3):
    index, chunks = load_index()
    query_vector = model.encode([query])
    query_vector = np.array(query_vector, dtype=np.float32)
    distances, indices = index.search(query_vector, top_k)
    results = []
    for i in indices[0]:
        if i != -1:
            results.append(chunks[i])
    return results