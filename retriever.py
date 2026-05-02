"""
retriever.py
Hybrid Retrieval: BM25 + FAISS cosine + safe domain-aware spell correction
+ Query intent detection + Context-aware diversification (multi-PDF fix)

FIXES:
  1. Single-page CV bug: diversify_by_page now falls back to score-only ranking
     when the document has only 1 unique page — all chunks are returned by score.
  2. top_k boosted inside hybrid_retrieve for broad single-doc queries.
  3. detect_page_query now receives per-source page counts, not a global max.
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from ingest import build_hybrid_index


# ─────────────────────────────────────────────
#  Tokenizer
# ─────────────────────────────────────────────
def tokenize(text: str) -> list:
    return re.findall(r'\w+', text.lower())


# ─────────────────────────────────────────────
#  Safe spell correction
# ─────────────────────────────────────────────
TYPO_MAP = {
    # Retrieval / RAG
    "retreival":    "retrieval",    "retreive":     "retrieve",
    "retreived":    "retrieved",    "retreiver":    "retriever",
    # Embeddings
    "embeding":     "embedding",    "embedings":    "embeddings",
    "embedd":       "embed",
    # Document
    "documnet":     "document",     "docuemnt":     "document",
    "docuemnts":    "documents",    "documentss":   "documents",
    # Hallucination
    "halucination": "hallucination","hallucinaion": "hallucination",
    "halluciantion":"hallucination",
    # Knowledge
    "knwoledge":    "knowledge",    "kwowledge":    "knowledge",
    "konwledge":    "knowledge",
    # Summarize
    "sumamrize":    "summarize",    "sumarize":     "summarize",
    "summairze":    "summarize",
    # Semantic
    "seamntic":     "semantic",     "semtantic":    "semantic",
    "semnatic":     "semantic",
    # Language
    "langauge":     "language",     "laguage":      "language",
    "languge":      "language",
    # Model
    "modle":        "model",        "mdoel":        "model",
    # Attention
    "attnetion":    "attention",    "atention":     "attention",
    # Generation
    "genreation":   "generation",   "generaiton":   "generation",
    # Projects (CV)
    "porject":      "project",      "proejct":      "project",
    "porjects":     "projects",     "proejcts":     "projects",
    # Certifications (CV)
    "certifcate":   "certificate",  "certificaiton":"certification",
    "certifcaiton": "certification","certifications":"certifications",
    # Achievements (CV)
    "acheivement":  "achievement",  "acheivements": "achievements",
    "achivement":   "achievement",  "achivements":  "achievements",
    # Experience (CV)
    "experince":    "experience",   "expereince":   "experience",
    # Education
    "educaiton":    "education",    "eudcation":    "education",
    # Skills
    "sklls":        "skills",       "skils":        "skills",
}

def correct_query(query: str) -> str:
    words = query.split()
    corrected = []
    changed = False

    for word in words:
        stripped = word.strip(".,!?;:'\"()")
        if not stripped:
            corrected.append(word)
            continue
        lead  = word[:len(word) - len(word.lstrip(".,!?;:'\"()"))]
        trail = word[len(stripped) + len(lead):]
        core  = stripped

        if (
            "-" in core
            or any(c.isupper() for c in core[1:])
            or core.isupper()
            or any(c.isdigit() for c in core)
        ):
            corrected.append(word)
            continue

        lower = core.lower()
        if lower in TYPO_MAP:
            corrected.append(lead + TYPO_MAP[lower] + trail)
            changed = True
        else:
            corrected.append(word)

    return " ".join(corrected) if changed else query


# ─────────────────────────────────────────────
#  Query intent detection
# ─────────────────────────────────────────────
BROAD_ONLY_TRIGGERS = {
    "summarize everything", "summarise everything",
    "summarize the document", "summarise the document",
    "summarize the whole", "summarise the whole",
    "summarize all", "summarise all",
    "summary of everything", "summary of the document",
    "summary of the whole", "full summary", "complete summary",
    "overview of the document", "overview of everything",
    "overview of all", "full overview",
    "key points of the document", "main points of the document",
    "all sections", "whole document", "entire document",
    "both sections", "all topics", "both documents",
    "tell me everything", "cover everything", "all documents",
    # CV-specific broad triggers
    "list all projects", "list my projects", "all projects",
    "list all skills", "all skills", "list all certifications",
    "all certifications", "list everything", "show all",
    "full cv", "full resume", "entire cv", "entire resume",
    "all experience", "all achievements", "list achievements",
}

BARE_BROAD_TRIGGERS = {
    "summarize", "summarise", "summary", "overview",
    "key points", "main points",
}

STOP_WORDS = {
    "it", "this", "the", "document", "pdf", "file", "text",
    "everything", "all", "whole", "entire", "please", "me", "us",
    "of", "a", "an", "that", "these", "those",
}

def detect_query_intent(query: str) -> str:
    q = query.lower().strip()
    if any(phrase in q for phrase in BROAD_ONLY_TRIGGERS):
        return "broad"
    for trigger in BARE_BROAD_TRIGGERS:
        if q.startswith(trigger) or q == trigger:
            remainder = q[len(trigger):].strip(" .,?!")
            remaining_words = [w for w in remainder.split() if w not in STOP_WORDS]
            if not remaining_words:
                return "broad"
    return "specific"


# ─────────────────────────────────────────────
#  Diversification strategies
# ─────────────────────────────────────────────
def diversify_by_source(chunks, scored_indices, top_k):
    """
    For BROAD multi-doc queries.
    Guarantees at least 1 chunk per source PDF,
    then fills remaining slots by score.
    """
    seen_sources = {}
    first_pass, second_pass = [], []
    for idx in scored_indices:
        src = chunks[idx]["source"]
        if src not in seen_sources:
            seen_sources[src] = True
            first_pass.append(idx)
        else:
            second_pass.append(idx)
    return (first_pass + second_pass)[:top_k]


def diversify_by_page(chunks, scored_indices, top_k):
    """
    For SPECIFIC queries and single-doc queries.

    FIX: If the entire document is a single page (e.g. a 1-page CV),
    page-level deduplication collapses ALL chunks to 1 result.
    In that case we skip deduplication entirely and return by score order.

    For multi-page docs: picks best chunk per (source, page) pair first,
    then fills by score — so every section gets representation.
    """
    # Count unique (source, page) pairs in the candidate set
    unique_pages = {(chunks[i]["source"], chunks[i]["page"]) for i in scored_indices}

    # FIX: single-page doc (or all chunks on same page) — skip page dedup
    if len(unique_pages) == 1:
        return list(scored_indices)[:top_k]

    seen_pages = {}
    first_pass, second_pass = [], []
    for idx in scored_indices:
        key = (chunks[idx]["source"], chunks[idx]["page"])
        if key not in seen_pages:
            seen_pages[key] = True
            first_pass.append(idx)
        else:
            second_pass.append(idx)
    return (first_pass + second_pass)[:top_k]


# ─────────────────────────────────────────────
#  Hybrid retrieval
# ─────────────────────────────────────────────
def hybrid_retrieve(
    query: str,
    faiss_index,
    bm25_index,
    chunks: list,
    model: SentenceTransformer,
    top_k: int = 6,
    alpha: float = 0.6,
    use_spell_correction: bool = True,
) -> tuple:
    """
    Hybrid retrieval: cosine FAISS (dense) + BM25 (sparse).

    Diversification:
    - BROAD + multiple docs  → diversify_by_source (every PDF gets a slot)
    - SPECIFIC or single doc → diversify_by_page (every section gets a slot)
    - Single-page doc        → score-order only (no dedup, returns all top chunks)

    Returns: (results, intent)
    """
    original_query = query
    n = len(chunks)
    num_sources = len({c["source"] for c in chunks})

    # FIX: detect if any source is a single-page document
    # (used to boost top_k so CV doesn't get starved)
    pages_per_source = {}
    for c in chunks:
        pages_per_source.setdefault(c["source"], set()).add(c["page"])
    has_single_page_source = any(len(pages) == 1 for pages in pages_per_source.values())

    # 1. Safe spell correction
    if use_spell_correction:
        query = correct_query(query)

    # 2. Intent
    intent = detect_query_intent(query)

    # 3. Fetch sizes
    # FIX: For single-page docs (CVs), boost fetch_k and final_k so we get
    # ALL chunks from that page, not just 1 after deduplication.
    if has_single_page_source and intent in ("broad", "specific"):
        # Retrieve all chunks for single-page docs
        fetch_k = n
        final_k = n  # no dedup will happen, so return all scored chunks
    elif intent == "broad":
        fetch_k = min(n, top_k * 15)
        final_k = min(n, top_k * num_sources)
    else:
        fetch_k = min(n, top_k * 10)
        final_k = top_k

    # 4. Dense cosine scores
    q_vec = np.array(
        model.encode([query], normalize_embeddings=True), dtype=np.float32
    )
    raw_scores, raw_indices = faiss_index.search(q_vec, fetch_k)

    dense_scores = np.zeros(n, dtype=np.float32)
    for score, idx in zip(raw_scores[0], raw_indices[0]):
        if idx != -1:
            dense_scores[idx] = float(score)

    d_min, d_max = dense_scores.min(), dense_scores.max()
    if d_max - d_min > 1e-9:
        dense_scores = (dense_scores - d_min) / (d_max - d_min)

    # 5. BM25 scores
    bm25_raw = np.array(
        bm25_index.get_scores(tokenize(query)), dtype=np.float32
    )
    b_min, b_max = bm25_raw.min(), bm25_raw.max()
    if b_max - b_min > 1e-9:
        bm25_scores = (bm25_raw - b_min) / (b_max - b_min)
    else:
        bm25_scores = bm25_raw

    # 6. Combined score
    final_scores = alpha * dense_scores + (1.0 - alpha) * bm25_scores

    # 7. Rank
    sorted_indices = np.argsort(final_scores)[::-1]

    # 8. Diversify by correct strategy
    if intent == "broad" and num_sources > 1:
        top_indices = diversify_by_source(chunks, sorted_indices.tolist(), final_k)
    else:
        top_indices = diversify_by_page(chunks, sorted_indices.tolist(), final_k)

    # 9. Build results
    results = []
    for idx in top_indices:
        chunk = dict(chunks[idx])
        chunk["vector_score"]    = round(float(dense_scores[idx]), 4)
        chunk["bm25_score"]      = round(float(bm25_scores[idx]),  4)
        chunk["final_score"]     = round(float(final_scores[idx]), 4)
        chunk["corrected_query"] = query if query != original_query else None
        results.append(chunk)

    return results, intent