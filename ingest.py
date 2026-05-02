"""
ingest.py
Handles everything related to PDF ingestion:
  - Text extraction (PyMuPDF + pdfplumber fallback + OCR fallback)
  - Caching (MD5 hash-based)
  - Sentence-aware chunking with overlap (fixes missed concepts)
  - Embedding (sentence-transformers, normalised for cosine similarity)
  - Building FAISS + BM25 hybrid index
"""

import os
import io
import re
import pickle
import hashlib
import concurrent.futures

import fitz                          # PyMuPDF
import pdfplumber
import pytesseract
import numpy as np
import faiss
from PIL import Image
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ── Constants ──────────────────────────────────────────────────────────────────
# Larger chunks = more context per chunk = fewer split concepts
CHUNK_SIZE    = 1200   # chars (was 700 — too small, split concepts mid-sentence)
CHUNK_OVERLAP = 200    # chars (was 70 — too small, context lost at boundaries)
CACHE_DIR     = "pdf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  1. HASHING
# ─────────────────────────────────────────────────────────────────────────────
def compute_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
#  2. OCR FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
def ocr_page(page, idx: int) -> tuple:
    pix = page.get_pixmap(dpi=150)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    text = pytesseract.image_to_string(img)
    return idx, text.strip()


# ─────────────────────────────────────────────────────────────────────────────
#  3. TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_text(pdf_path: str, source_name: str, progress_callback=None) -> list:
    """
    Three-layer extraction: PyMuPDF → pdfplumber → Tesseract OCR.
    Results cached by MD5 so re-uploads are instant.
    """
    with open(pdf_path, "rb") as f:
        raw = f.read()

    file_hash  = compute_hash(raw)
    cache_path = os.path.join(CACHE_DIR, f"{file_hash}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as cf:
            return pickle.load(cf)

    doc   = fitz.open(pdf_path)
    total = len(doc)
    pages = [None] * total

    def process_page(i):
        page = doc[i]
        # Layer 1: PyMuPDF (fast)
        text = page.get_text().strip()
        if text:
            return i, text
        # Layer 2: pdfplumber (better on complex layouts)
        try:
            with pdfplumber.open(pdf_path) as plumb:
                pl_text = plumb.pages[i].extract_text()
                if pl_text and pl_text.strip():
                    return i, pl_text.strip()
        except Exception:
            pass
        # Layer 3: OCR (scanned pages)
        return ocr_page(page, i)

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        futures = [ex.submit(process_page, i) for i in range(total)]
        for done, future in enumerate(concurrent.futures.as_completed(futures)):
            idx, text = future.result()
            if text:
                pages[idx] = {"text": text, "page": idx + 1, "source": source_name}
            if progress_callback:
                progress_callback((done + 1) / total)

    doc.close()
    pages = [p for p in pages if p]

    with open(cache_path, "wb") as cf:
        pickle.dump(pages, cf)

    return pages


# ─────────────────────────────────────────────────────────────────────────────
#  4. SENTENCE-AWARE CHUNKING
#  Key fix for Issue 1: respect sentence boundaries so concepts are never
#  split mid-sentence across chunk edges.
# ─────────────────────────────────────────────────────────────────────────────
def _split_sentences(text: str) -> list:
    """Split text into sentences using punctuation boundaries."""
    # Split on . ! ? followed by whitespace or end-of-string
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_pages(pages: list,
                chunk_size: int  = CHUNK_SIZE,
                overlap: int     = CHUNK_OVERLAP) -> list:
    """
    Sentence-aware chunking with sliding window overlap.

    Instead of blindly cutting every 700 chars (which splits sentences),
    we accumulate sentences until we hit chunk_size, then slide back
    by `overlap` chars worth of sentences for the next chunk.

    This ensures:
    - No concept is cut mid-sentence
    - Overlapping context catches terms near chunk boundaries
    - Each chunk has its source PDF and page number tracked
    """
    chunks = []

    for p in pages:
        sentences = _split_sentences(p["text"])
        if not sentences:
            continue

        current_chunk   = []
        current_length  = 0

        for sentence in sentences:
            sentence_len = len(sentence) + 1  # +1 for space

            # If adding this sentence exceeds chunk_size AND we already
            # have content, save current chunk and start a new one
            if current_length + sentence_len > chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text":   chunk_text,
                    "page":   p["page"],
                    "source": p["source"],
                })

                # Slide back: keep sentences from the end that fit in overlap
                overlap_chunk  = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    s_len = len(s) + 1
                    if overlap_length + s_len <= overlap:
                        overlap_chunk.insert(0, s)
                        overlap_length += s_len
                    else:
                        break

                current_chunk  = overlap_chunk
                current_length = overlap_length

            current_chunk.append(sentence)
            current_length += sentence_len

        # Don't forget the last chunk
        if current_chunk:
            chunks.append({
                "text":   " ".join(current_chunk),
                "page":   p["page"],
                "source": p["source"],
            })

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
#  5. EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────
def embed_chunks(chunks: list, model: SentenceTransformer) -> np.ndarray:
    """
    Cosine-similarity embeddings.
    normalize_embeddings=True → unit vectors → IndexFlatIP = cosine similarity.
    """
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True,
        batch_size=64,
    )
    return np.array(embeddings, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  6. BUILD HYBRID INDEX
# ─────────────────────────────────────────────────────────────────────────────
def _tokenize(text: str) -> list:
    return re.findall(r'\w+', text.lower())


def build_hybrid_index(chunks: list, model: SentenceTransformer) -> tuple:
    """
    Build FAISS (cosine) + BM25 indexes over all chunks from all PDFs.
    Both indexes share the same chunk list so indices align perfectly.
    """
    # Dense index
    embeddings  = embed_chunks(chunks, model)
    dim         = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)

    # Sparse index
    tokenized  = [_tokenize(c["text"]) for c in chunks]
    bm25_index = BM25Okapi(tokenized)

    return faiss_index, bm25_index, chunks


# ─────────────────────────────────────────────────────────────────────────────
#  7. PIPELINE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def ingest_pdf(pdf_path: str, source_name: str, model: SentenceTransformer,
               progress_callback=None) -> tuple:
    """
    Full pipeline for a single PDF: extract → chunk.
    build_hybrid_index() is called once after ALL PDFs are ingested.
    """
    pages  = extract_text(pdf_path, source_name, progress_callback)
    chunks = chunk_pages(pages)
    return pages, chunks