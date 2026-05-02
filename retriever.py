"""
retriever.py
Hybrid Retrieval: BM25 + FAISS cosine + safe domain-aware spell correction
+ Query intent detection + Context-aware diversification (multi-PDF fix)
+ Query Expansion (fixes "dumb" keyword mismatch — Issue 1)

FIXES:
  1. Single-page CV bug: diversify_by_page now falls back to score-only ranking
     when the document has only 1 unique page — all chunks are returned by score.
  2. top_k boosted inside hybrid_retrieve for broad single-doc queries.
  3. detect_page_query now receives per-source page counts, not a global max.

NEW — Query Expansion (Issue 1 fix):
  - expand_query() generates semantic synonyms/rewrites for the user query.
  - hybrid_retrieve now scores each expansion and takes element-wise MAX,
    so "what is the heading" finds "DocMind RAG" even without the word "heading".
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from ingest import build_hybrid_index


# ─────────────────────────────────────────────────────────────────────────────
#  Tokenizer
# ─────────────────────────────────────────────────────────────────────────────
def tokenize(text: str) -> list:
    return re.findall(r'\w+', text.lower())


# ─────────────────────────────────────────────────────────────────────────────
#  Query Expansion  ← NEW (Issue 1 fix)
#
#  Why: BM25 is purely lexical — "heading" never matches "DocMind RAG"
#  because those words don't appear in the query. Dense cosine is better
#  but still anchored to the original query vector.
#
#  Fix: expand the query into several semantic rewrites and score all of them.
#  The final score for each chunk = MAX score across all expansions.
#  This means if ANY expansion matches, the chunk is surfaced.
#
#  The expansion map is domain-aware (document structure + CV + NLP terms).
# ─────────────────────────────────────────────────────────────────────────────

# Structural / document-intent synonyms
EXPANSION_MAP: dict[str, list[str]] = {
    # Document structure
    "heading":        ["title", "header", "name", "document name", "topic"],
    "title":          ["heading", "name", "header", "document title"],
    "header":         ["heading", "title", "name", "top of document"],
    "topic":          ["subject", "title", "heading", "theme", "focus"],
    "subject":        ["topic", "title", "heading", "theme"],
    "name":           ["title", "heading", "label", "identifier"],
    "introduction":   ["overview", "abstract", "background", "intro", "beginning", "start"],
    "abstract":       ["summary", "overview", "introduction", "brief"],
    "conclusion":     ["summary", "result", "outcome", "findings", "end", "final"],
    "result":         ["outcome", "finding", "conclusion", "output", "answer"],
    "section":        ["part", "chapter", "segment", "portion", "area"],
    "chapter":        ["section", "part", "unit", "module"],
    "content":        ["information", "text", "material", "data", "details"],
    "definition":     ["meaning", "description", "explanation", "what is"],
    "purpose":        ["goal", "objective", "aim", "intent", "reason"],
    "description":    ["explanation", "detail", "overview", "about"],
    "overview":       ["summary", "introduction", "description", "abstract"],
    "summary":        ["overview", "abstract", "conclusion", "recap"],

    # CV / resume
    "experience":     ["work history", "employment", "jobs", "career", "professional background"],
    "education":      ["degree", "qualification", "university", "academic background", "school"],
    "skills":         ["abilities", "competencies", "expertise", "technologies", "tools"],
    "projects":       ["work", "portfolio", "assignments", "developments", "builds"],
    "certifications": ["certificates", "credentials", "qualifications", "courses"],
    "achievements":   ["accomplishments", "awards", "recognition", "milestones"],
    "contact":        ["email", "phone", "address", "location", "linkedin"],
    "objective":      ["goal", "aim", "career summary", "profile", "about me"],
    "profile":        ["about me", "objective", "bio", "summary", "introduction"],

    # NLP / ML
    "model":          ["architecture", "network", "system", "algorithm", "approach"],
    "training":       ["learning", "fine-tuning", "optimization", "fitting"],
    "performance":    ["accuracy", "results", "metrics", "evaluation", "score"],
    "dataset":        ["data", "corpus", "training data", "benchmark"],
    "approach":       ["method", "technique", "strategy", "algorithm"],
    "method":         ["approach", "technique", "algorithm", "procedure"],
    "evaluation":     ["assessment", "testing", "metrics", "results", "benchmark"],
    "architecture":   ["design", "structure", "model", "framework", "layout"],
}

def expand_query(query: str) -> list[str]:
    """
    Returns a list of query variants: [original] + expanded forms.

    Strategy:
    1. For each word in the query that has an entry in EXPANSION_MAP,
       generate a new query where that word is replaced by each synonym.
    2. Also add a query that appends all synonyms as extra context words
       (helps BM25 surface results containing synonyms).
    3. Deduplicate and return.

    Example:
      "what is the heading" →
        ["what is the heading",
         "what is the title",
         "what is the header",
         "what is the name",
         "what is the document name",
         "what is the topic",
         "what is the heading title header name document name topic"]
    """
    words = query.lower().split()
    variants = [query]  # always include original

    # Track which expansions are added so we can also build an enriched query
    all_synonyms: list[str] = []

    for i, word in enumerate(words):
        # Strip punctuation for lookup
        clean = word.strip(".,!?;:'\"()")
        if clean in EXPANSION_MAP:
            syns = EXPANSION_MAP[clean]
            all_synonyms.extend(syns)
            for syn in syns:
                new_words = words[:i] + [syn] + words[i + 1:]
                variant = " ".join(new_words)
                if variant not in variants:
                    variants.append(variant)

    # Also add a "kitchen sink" variant: original query + all synonyms appended
    # This boosts BM25 recall for any synonym appearing verbatim in text
    if all_synonyms:
        enriched = query + " " + " ".join(dict.fromkeys(all_synonyms))
        if enriched not in variants:
            variants.append(enriched)

    return variants


# ─────────────────────────────────────────────────────────────────────────────
#  Safe spell correction
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
#  Query intent detection
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
#  Diversification strategies
# ─────────────────────────────────────────────────────────────────────────────
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
    """
    unique_pages = {(chunks[i]["source"], chunks[i]["page"]) for i in scored_indices}

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


# ─────────────────────────────────────────────────────────────────────────────
#  Core scoring helper  ← NEW
#
#  Scores ALL chunks against a single query string.
#  Returns (dense_scores, bm25_scores, combined_scores) as np arrays of shape (n,).
# ─────────────────────────────────────────────────────────────────────────────
def _score_single_query(
    query: str,
    faiss_index,
    bm25_index,
    n: int,
    model: SentenceTransformer,
    alpha: float,
    fetch_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (dense_norm, bm25_norm, combined) for one query string."""

    # Dense
    q_vec = np.array(
        model.encode([query], normalize_embeddings=True), dtype=np.float32
    )
    raw_scores, raw_indices = faiss_index.search(q_vec, fetch_k)

    dense = np.zeros(n, dtype=np.float32)
    for score, idx in zip(raw_scores[0], raw_indices[0]):
        if idx != -1:
            dense[idx] = float(score)

    d_min, d_max = dense.min(), dense.max()
    if d_max - d_min > 1e-9:
        dense = (dense - d_min) / (d_max - d_min)

    # BM25
    bm25_raw = np.array(bm25_index.get_scores(tokenize(query)), dtype=np.float32)
    b_min, b_max = bm25_raw.min(), bm25_raw.max()
    bm25 = (bm25_raw - b_min) / (b_max - b_min) if b_max - b_min > 1e-9 else bm25_raw

    combined = alpha * dense + (1.0 - alpha) * bm25
    return dense, bm25, combined


# ─────────────────────────────────────────────────────────────────────────────
#  Hybrid retrieval  (updated to use query expansion)
# ─────────────────────────────────────────────────────────────────────────────
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

    NEW — Query Expansion:
    The query is expanded into multiple semantic variants. Each variant is
    scored independently and the final score for each chunk is the MAX across
    all variants. This means if ANY expansion matches a chunk, it gets surfaced
    — fixing the "heading → DocMind RAG" type of keyword mismatch.

    Diversification:
    - BROAD + multiple docs  → diversify_by_source
    - SPECIFIC or single doc → diversify_by_page
    - Single-page doc        → score-order only

    Returns: (results, intent)
    """
    original_query = query
    n = len(chunks)
    num_sources = len({c["source"] for c in chunks})

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
    if has_single_page_source and intent in ("broad", "specific"):
        fetch_k = n
        final_k = n
    elif intent == "broad":
        fetch_k = min(n, top_k * 15)
        final_k = min(n, top_k * num_sources)
    else:
        fetch_k = min(n, top_k * 10)
        final_k = top_k

    fetch_k = max(fetch_k, 1)  # guard against empty index

    # 4. Query expansion  ← NEW
    #    Score every expansion variant, take element-wise MAX so the best
    #    matching variant "wins" for each chunk.
    expansions = expand_query(query)

    # Accumulators: we keep track of dense/bm25/final per-expansion
    # and reduce with MAX so any strong match is preserved.
    max_dense   = np.zeros(n, dtype=np.float32)
    max_bm25    = np.zeros(n, dtype=np.float32)
    max_combined = np.zeros(n, dtype=np.float32)

    for exp_query in expansions:
        dense, bm25, combined = _score_single_query(
            exp_query, faiss_index, bm25_index, n, model, alpha, fetch_k
        )
        # Element-wise max: keep the best score each chunk gets from any expansion
        np.maximum(max_dense,    dense,    out=max_dense)
        np.maximum(max_bm25,     bm25,     out=max_bm25)
        np.maximum(max_combined, combined, out=max_combined)

    # 5. Rank by max combined score
    sorted_indices = np.argsort(max_combined)[::-1]

    # 6. Diversify
    if intent == "broad" and num_sources > 1:
        top_indices = diversify_by_source(chunks, sorted_indices.tolist(), final_k)
    else:
        top_indices = diversify_by_page(chunks, sorted_indices.tolist(), final_k)

    # 7. Build results
    results = []
    for idx in top_indices:
        chunk = dict(chunks[idx])
        chunk["vector_score"]    = round(float(max_dense[idx]),    4)
        chunk["bm25_score"]      = round(float(max_bm25[idx]),     4)
        chunk["final_score"]     = round(float(max_combined[idx]), 4)
        chunk["corrected_query"] = query if query != original_query else None
        results.append(chunk)

    return results, intent