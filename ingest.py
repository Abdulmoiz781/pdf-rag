import sys
import pdfplumber
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def extract_text(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append({
                    "text": text,
                    "page": i + 1,
                    "source": pdf_path
                })
    return pages

def chunk_text(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    for page in pages:
        text = page["text"]
        page_num = page["page"]
        source = page["source"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append({
                "text": chunk,
                "page": page_num,
                "source": source
            })
            start = end - overlap
    return chunks

def embed_chunks(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def save_to_faiss(chunks, embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    faiss.write_index(index, "faiss_index.bin")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved {len(chunks)} chunks to FAISS index")

if len(sys.argv) < 2:
    print("Usage: python ingest.py <path_to_pdf>")
    print("Example: python ingest.py Data/NIPS.pdf")
    sys.exit(1)

PDF_PATH = sys.argv[1]

print(f"Processing: {PDF_PATH}")
print("Step 1: Extracting text from PDF...")
pages = extract_text(PDF_PATH)
print(f"Extracted {len(pages)} pages")

print("Step 2: Chunking text...")
chunks = chunk_text(pages)
print(f"Created {len(chunks)} chunks")

print("Step 3: Generating embeddings...")
embeddings = embed_chunks(chunks)

print("Step 4: Saving to FAISS...")
save_to_faiss(chunks, embeddings)

print(f"Done! {PDF_PATH} has been ingested successfully.")



