"""
ingest.py  —  upgraded pipeline
Handles everything related to PDF ingestion:
  - Text extraction (column-aware block sorting)
  - Table extraction via pdfplumber → markdown (preserves rows/cols)
  - Image captioning via Groq vision API (llama-3.2-11b-vision-preview)
  - Caching (MD5 hash-based)
  - WORD-based chunking with overlap (fixes char-boundary splits)
  - Embedding (sentence-transformers, normalised for cosine similarity)
  - Building FAISS + BM25 hybrid index

CHANGES over previous version:
  - BLIP removed entirely. BLIP was a ~900 MB model that downloaded on first
    run and then ran slowly on CPU — causing 10+ minute hangs (Issue 2).
  - Image captioning now uses Groq's llama-3.2-11b-vision-preview via API.
    This is the same approach used by Claude, GPT-4o, and Gemini for vision:
    send a base64 image to a vision-capable LLM → get a caption back in < 2s.
  - Captioning is done one image at a time with a short timeout (10s) so a
    single bad image never blocks the whole pipeline.
  - If the GROQ_API_KEY is not set, or if captioning fails for any reason,
    the image is silently skipped — the rest of the document still loads fine.
"""

import os
import io
import re
import base64
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
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CHUNK_WORDS         = 200    # target words per chunk
CHUNK_OVERLAP_WORDS = 30     # overlap words
MIN_IMAGE_SIZE      = 100    # pixels — skip tiny icons/bullets
CACHE_DIR           = "pdf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  0. GROQ VISION CLIENT  (replaces BLIP — Issue 2 fix)
#
#  Why Groq instead of BLIP:
#  • BLIP downloads a ~900 MB model on first run and runs on CPU → 10+ min hang
#  • Groq vision (llama-3.2-11b-vision-preview) is a remote API call:
#    send base64 image → receive caption in ~1-2 seconds
#  • Same pattern used by Claude (claude-3-haiku), GPT-4o-mini, Gemini Flash
#    for production RAG pipelines
#
#  Graceful degradation: if GROQ_API_KEY is missing or the call fails,
#  we return None and the image is silently skipped. The rest of the
#  document still ingests normally.
# ─────────────────────────────────────────────────────────────────────────────

def _caption_image_with_groq(img: Image.Image) -> str | None:
    """
    Captions a PIL image using Groq's vision LLM.

    Steps:
    1. Convert PIL image → JPEG bytes → base64 string
    2. POST to Groq /v1/chat/completions with image_url content block
    3. Return the text caption, or None on any failure

    FIX: timeout must be passed to the Groq client constructor, not to
    create() — passing it to create() raises a TypeError that was being
    silently swallowed, causing ALL captions to return None.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[ingest] WARNING: GROQ_API_KEY not set — skipping image captioning")
        return None

    try:
        from groq import Groq
        # FIX: timeout goes on the client, not on create()
        groq_client = Groq(api_key=api_key, timeout=10.0)

        # Convert to JPEG bytes
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Describe this image in 1-2 sentences for a document "
                                "search index. Focus on what information it conveys "
                                "(e.g. chart type, subject of photo, diagram topic). "
                                "Be specific and factual."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=120,
            temperature=0.1,
        )
        caption = response.choices[0].message.content.strip()
        print(f"[ingest] Image caption: {caption[:80]}...")
        return caption if caption else None

    except Exception as e:
        print(f"[ingest] Image captioning failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  1. HASHING
# ─────────────────────────────────────────────────────────────────────────────
def compute_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
#  2. IMAGE CAPTIONING  (updated — no longer uses BLIP)
# ─────────────────────────────────────────────────────────────────────────────
def caption_images_on_page(fitz_page) -> list[str]:
    """
    Extracts embedded images from a fitz page and returns caption strings
    using the Groq vision API.

    Returns a list like:
      ["[Image: a bar chart showing quarterly revenue by product line]",
       "[Image: photograph of a server rack in a data centre]"]

    Images smaller than MIN_IMAGE_SIZE on either dimension are skipped
    (they are almost always decorative bullets or horizontal rules).

    If Groq captioning is unavailable or fails, returns an empty list
    so the rest of the page still ingests normally.
    """
    captions    = []
    image_list  = fitz_page.get_images(full=True)

    if not image_list:
        return captions

    doc = fitz_page.parent  # fitz.Document the page belongs to

    for img_info in image_list:
        xref = img_info[0]
        try:
            base_image = doc.extract_image(xref)
            img_bytes  = base_image["image"]
            img        = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # Skip tiny decorative images
            if img.width < MIN_IMAGE_SIZE or img.height < MIN_IMAGE_SIZE:
                continue

            caption = _caption_image_with_groq(img)
            if caption:
                captions.append(f"[Image: {caption}]")

        except Exception:
            continue  # corrupt / unsupported image → skip silently

    return captions


# ─────────────────────────────────────────────────────────────────────────────
#  3. TABLE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def _table_to_markdown(table: list) -> str:
    if not table or not table[0]:
        return ""

    def clean(cell):
        if cell is None:
            return ""
        return str(cell).replace("\n", " ").strip()

    rows = [[clean(cell) for cell in row] for row in table]
    col_widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]

    def fmt_row(row):
        return "| " + " | ".join(
            cell.ljust(col_widths[i]) for i, cell in enumerate(row)
        ) + " |"

    header    = fmt_row(rows[0])
    separator = "| " + " | ".join("-" * w for w in col_widths) + " |"
    body      = "\n".join(fmt_row(r) for r in rows[1:])
    return f"{header}\n{separator}\n{body}"


def extract_tables_from_page(plumb_page) -> list[str]:
    tables = plumb_page.extract_tables()
    return [_table_to_markdown(t) for t in tables if t]


# ─────────────────────────────────────────────────────────────────────────────
#  4. COLUMN-AWARE TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_column_aware_text(fitz_page) -> str:
    blocks = fitz_page.get_text("blocks")
    if not blocks:
        return ""

    blocks = [(b[0], b[1], b[4].strip()) for b in blocks if b[4].strip()]
    if not blocks:
        return ""

    page_width = fitz_page.rect.width
    right_half_blocks = [b for b in blocks if b[0] > page_width * 0.45]

    if right_half_blocks:
        midpoint     = page_width / 2
        left_blocks  = sorted([b for b in blocks if b[0] < midpoint],  key=lambda b: b[1])
        right_blocks = sorted([b for b in blocks if b[0] >= midpoint], key=lambda b: b[1])
        ordered = left_blocks + right_blocks
    else:
        ordered = sorted(blocks, key=lambda b: b[1])

    return "\n".join(b[2] for b in ordered)


# ─────────────────────────────────────────────────────────────────────────────
#  5. OCR FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
def ocr_page(fitz_page, idx: int) -> tuple:
    pix  = fitz_page.get_pixmap(dpi=150)
    img  = Image.open(io.BytesIO(pix.tobytes("png")))
    text = pytesseract.image_to_string(img)
    return idx, text.strip()


# ─────────────────────────────────────────────────────────────────────────────
#  6. FULL TEXT EXTRACTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def extract_text(pdf_path: str, source_name: str, progress_callback=None) -> list:
    with open(pdf_path, "rb") as f:
        raw = f.read()

    file_hash  = compute_hash(raw)
    cache_path = os.path.join(CACHE_DIR, f"{file_hash}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as cf:
            return pickle.load(cf)

    # Get total page count by opening once, then close before threading.
    # FIX (Bug 1): PyMuPDF fitz.Document is NOT thread-safe. Sharing a single
    # fitz_doc across threads causes silent crashes — captions never run.
    # Solution: each worker opens its own fitz.Document independently.
    with fitz.open(pdf_path) as _tmp:
        total = len(_tmp)
    pages = [None] * total

    def process_page(i):
        # FIX: open a fresh fitz.Document per thread — no shared state
        fitz_doc  = fitz.open(pdf_path)
        fitz_page = fitz_doc[i]
        parts     = []

        # Layer 1: Column-aware text
        text = extract_column_aware_text(fitz_page)

        if not text:
            # Layer 2: pdfplumber
            try:
                with pdfplumber.open(pdf_path) as plumb:
                    pl_text = plumb.pages[i].extract_text()
                    if pl_text and pl_text.strip():
                        text = pl_text.strip()
            except Exception:
                pass

        if not text:
            # Layer 3: OCR
            _, text = ocr_page(fitz_page, i)

        if text:
            parts.append(text)

        # Table extraction
        try:
            with pdfplumber.open(pdf_path) as plumb:
                table_mds = extract_tables_from_page(plumb.pages[i])
                for md in table_mds:
                    if md:
                        parts.append("\n[TABLE]\n" + md + "\n[/TABLE]")
        except Exception:
            pass

        # Image captioning via Groq vision API
        try:
            captions = caption_images_on_page(fitz_page)
            parts.extend(captions)
        except Exception as e:
            print(f"[ingest] Caption error page {i+1}: {e}")

        fitz_doc.close()
        return i, "\n\n".join(parts).strip()

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        futures = [ex.submit(process_page, i) for i in range(total)]
        for done, future in enumerate(concurrent.futures.as_completed(futures)):
            idx, text = future.result()
            if text:
                pages[idx] = {
                    "text":   text,
                    "page":   idx + 1,
                    "source": source_name,
                }
            if progress_callback:
                progress_callback((done + 1) / total)

    pages = [p for p in pages if p]

    with open(cache_path, "wb") as cf:
        pickle.dump(pages, cf)

    return pages


# ─────────────────────────────────────────────────────────────────────────────
#  7. WORD-BASED CHUNKING
# ─────────────────────────────────────────────────────────────────────────────
def _word_count(text: str) -> int:
    return len(text.split())


def _split_sentences(text: str) -> list[str]:
    parts  = re.split(r'(\[TABLE\].*?\[/TABLE\])', text, flags=re.DOTALL)
    result = []
    for part in parts:
        if part.startswith("[TABLE]"):
            result.append(part.strip())
        else:
            sentences = re.split(r'(?<=[.!?])\s+', part.strip())
            result.extend(s.strip() for s in sentences if s.strip())
    return result


def chunk_pages(
    pages:         list,
    chunk_words:   int = CHUNK_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list:
    chunks = []

    for p in pages:
        sentences = _split_sentences(p["text"])
        if not sentences:
            continue

        current_sentences = []
        current_words     = 0

        for sentence in sentences:
            s_words = _word_count(sentence)

            if current_words + s_words > chunk_words and current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append({
                    "text":   chunk_text,
                    "page":   p["page"],
                    "source": p["source"],
                })

                overlap_sentences = []
                overlap_count     = 0
                for s in reversed(current_sentences):
                    w = _word_count(s)
                    if overlap_count + w <= overlap_words:
                        overlap_sentences.insert(0, s)
                        overlap_count += w
                    else:
                        break

                current_sentences = overlap_sentences
                current_words     = overlap_count

            current_sentences.append(sentence)
            current_words += s_words

        if current_sentences:
            chunks.append({
                "text":   " ".join(current_sentences),
                "page":   p["page"],
                "source": p["source"],
            })

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
#  8. EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────
def embed_chunks(chunks: list, model: SentenceTransformer) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True,
        batch_size=64,
    )
    return np.array(embeddings, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  9. BUILD HYBRID INDEX
# ─────────────────────────────────────────────────────────────────────────────
def _tokenize(text: str) -> list:
    return re.findall(r'\w+', text.lower())


def build_hybrid_index(chunks: list, model: SentenceTransformer) -> tuple:
    embeddings  = embed_chunks(chunks, model)
    dim         = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)

    tokenized  = [_tokenize(c["text"]) for c in chunks]
    bm25_index = BM25Okapi(tokenized)

    return faiss_index, bm25_index, chunks


# ─────────────────────────────────────────────────────────────────────────────
#  10. PIPELINE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def ingest_pdf(
    pdf_path:          str,
    source_name:       str,
    model:             SentenceTransformer,
    progress_callback = None,
) -> tuple:
    pages  = extract_text(pdf_path, source_name, progress_callback)
    chunks = chunk_pages(pages)
    return pages, chunks