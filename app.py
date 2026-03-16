import os
import re
import pickle
import tempfile
import streamlit as st
import faiss
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import pdfplumber

load_dotenv(override=True)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = SentenceTransformer("all-MiniLM-L6-v2")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def extract_text(pdf_path, source_name):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append({
                    "text": text,
                    "page": i + 1,
                    "source": source_name
                })
    return pages

def chunk_text(pages):
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunks.append({
                "text": text[start:end],
                "page": page["page"],
                "source": page["source"]
            })
            start = end - CHUNK_OVERLAP
    return chunks

def build_index(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype=np.float32)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks

def retrieve(query, index, chunks, top_k=3):
    query_vector = np.array(
        model.encode([query]), dtype=np.float32
    )
    distances, indices = index.search(query_vector, top_k)
    results = []
    for i in indices[0]:
        if i != -1:
            results.append(chunks[i])
    return results

def get_chunks_by_page(page_num, source_name, chunks):
    return [
        c for c in chunks
        if c["page"] == page_num and c["source"] == source_name
    ]

def detect_page_query(question, total_pages):
    q = question.lower()
    if "last page" in q:
        return total_pages
    if "first page" in q:
        return 1
    match = re.search(r'page\s*(\d+)', q)
    if match:
        return int(match.group(1))
    return None

st.title("PDF-RAG: Ask Your PDFs")
st.markdown("Upload one or more PDFs and ask questions about them!")
st.markdown("---")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    if "index" not in st.session_state or \
       st.session_state.get("uploaded_names") != [f.name for f in uploaded_files]:

        with st.spinner("Processing PDFs..."):
            all_chunks = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                pages = extract_text(tmp_path, uploaded_file.name)
                chunks = chunk_text(pages)
                all_chunks.extend(chunks)
                os.unlink(tmp_path)

            index, chunks = build_index(all_chunks)
            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.uploaded_names = [f.name for f in uploaded_files]

        st.success(f"Processed {len(uploaded_files)} PDF(s) — {len(all_chunks)} chunks created!")

    st.markdown("---")
    question = st.text_input("Enter your question:")

    if st.button("Ask"):
        if not question:
            st.warning("Please enter a question!")
        else:
            with st.spinner("Searching and generating answer..."):
                index = st.session_state.index
                chunks = st.session_state.chunks
                total_pages = max(c["page"] for c in chunks)
                page_num = detect_page_query(question, total_pages)

                if page_num:
                    source = st.session_state.uploaded_names[0]
                    result_chunks = get_chunks_by_page(page_num, source, chunks)
                    if not result_chunks:
                        st.error(f"Page {page_num} not found!")
                        st.stop()
                    source_type = f"page {page_num}"
                else:
                    result_chunks = retrieve(question, index, chunks)
                    source_type = "semantic search"

                context = "\n\n".join([
                    f"[Page {c['page']} - {c['source']}]: {c['text']}"
                    for c in result_chunks
                ])

                prompt = f"""You are a helpful assistant answering questions about research papers.

Use the context below to answer the question clearly and concisely.
If the context is messy or partially readable, still try your best to answer.
Only say you could not find the answer if there is truly no relevant information.

Context:
{context}

Question: {question}

Answer:"""

                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.3
                )

                answer = response.choices[0].message.content

                st.markdown("### Answer")
                st.write(answer)

                st.markdown("### Sources")
                st.caption(f"Retrieved via: {source_type}")
                for c in result_chunks:
                    st.info(f"**{c['source']}** — Page {c['page']}: {c['text'][:150]}...")

else:
    st.info("Please upload one or more PDF files to get started!")