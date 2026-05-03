"""
app.py — DocMind UI
Editorial dark-navy hero + light document workspace + Claude-style chat
"""

import os
import re
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from styles import apply_styles
from ingest import ingest_pdf, build_hybrid_index
from retriever import hybrid_retrieve

apply_styles()

load_dotenv(override=True)
#client = Groq(api_key=os.getenv("GROQ_API_KEY"))
client = Groq(api_key=st.secrets["GROQ_API_KEY"])


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


def detect_page_query(question: str, chunks: list):
    q = question.lower()
    sources = list({c["source"] for c in chunks})
    if "last page" in q and len(sources) == 1:
        return max(c["page"] for c in chunks)
    if "first page" in q:
        return 1
    m = re.search(r'page\s*(\d+)', q)
    return int(m.group(1)) if m else None


# ── Navbar + Hero ────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
  <div class="navbar-brand">
    <div class="navbar-logo">✦</div>
    <span class="navbar-name">Doc<span class="navbar-accent">Mind</span></span>
    <span class="navbar-tag">V 1.0</span>
  </div>
  <div class="navbar-right">
    <span class="navbar-status"><span class="navbar-online"></span>System Ready</span>
  </div>
</div>

<div class="hero-wrapper">
  <div class="hero-eyebrow">
    <span class="hero-dot"></span>
    AI Document Intelligence
  </div>
  <h1 class="hero-title">
    Ask anything about<br><em>your documents.</em>
  </h1>
  <p class="hero-sub">
    Upload PDFs and get instant, accurate answers powered by hybrid
    semantic search, BM25 retrieval, and large language model reasoning.
  </p>
  <div class="hero-pills">
    <span class="hero-pill"><span class="hero-pill-icon">⚡</span> Hybrid BM25 + Cosine Search</span>
    <span class="hero-pill"><span class="hero-pill-icon">🔒</span> Multi-PDF Support</span>
    <span class="hero-pill"><span class="hero-pill-icon">✦</span> Sentence-Transformers Embeddings</span>
    <span class="hero-pill"><span class="hero-pill-icon">◈</span> FAISS Vector Index</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_docs, tab_chat = st.tabs(["📄  Documents", "💬  Chat"])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — DOCUMENTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_docs:

    # ── Section intro ─────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:28px;">
      <div style="font-family:'Playfair Display',serif;font-size:22px;
                  font-weight:600;color:#1a1a2e;margin-bottom:6px;">
        Document Workspace
      </div>
      <div style="font-size:14px;color:#8a8aaa;font-weight:300;line-height:1.6;">
        Upload one or more PDF files below. Each document is automatically
        extracted, chunked into semantic units, and indexed for hybrid retrieval.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── How it works cards ────────────────────────────────────────────────
    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:32px;">
      <div style="background:#ffffff;border:1px solid rgba(26,26,46,0.10);
                  border-radius:14px;padding:20px;position:relative;overflow:hidden;">
        <div style="width:36px;height:36px;background:rgba(79,70,229,0.08);
                    border-radius:10px;display:flex;align-items:center;
                    justify-content:center;font-size:18px;margin-bottom:12px;">📤</div>
        <div style="font-size:11px;font-weight:700;letter-spacing:0.8px;
                    text-transform:uppercase;color:#8a8aaa;margin-bottom:5px;">Step 1</div>
        <div style="font-size:14px;font-weight:600;color:#1a1a2e;margin-bottom:4px;">Upload</div>
        <div style="font-size:12px;color:#8a8aaa;line-height:1.5;">
          Drop your PDFs — text, scanned, or mixed layouts.
        </div>
      </div>
      <div style="background:#ffffff;border:1px solid rgba(26,26,46,0.10);
                  border-radius:14px;padding:20px;">
        <div style="width:36px;height:36px;background:rgba(79,70,229,0.08);
                    border-radius:10px;display:flex;align-items:center;
                    justify-content:center;font-size:18px;margin-bottom:12px;">✂️</div>
        <div style="font-size:11px;font-weight:700;letter-spacing:0.8px;
                    text-transform:uppercase;color:#8a8aaa;margin-bottom:5px;">Step 2</div>
        <div style="font-size:14px;font-weight:600;color:#1a1a2e;margin-bottom:4px;">Chunk</div>
        <div style="font-size:12px;color:#8a8aaa;line-height:1.5;">
          Sentence-aware splitting with 200-char overlap.
        </div>
      </div>
      <div style="background:#ffffff;border:1px solid rgba(26,26,46,0.10);
                  border-radius:14px;padding:20px;">
        <div style="width:36px;height:36px;background:rgba(79,70,229,0.08);
                    border-radius:10px;display:flex;align-items:center;
                    justify-content:center;font-size:18px;margin-bottom:12px;">🧠</div>
        <div style="font-size:11px;font-weight:700;letter-spacing:0.8px;
                    text-transform:uppercase;color:#8a8aaa;margin-bottom:5px;">Step 3</div>
        <div style="font-size:14px;font-weight:600;color:#1a1a2e;margin-bottom:4px;">Embed</div>
        <div style="font-size:12px;color:#8a8aaa;line-height:1.5;">
          all-MiniLM-L6-v2 cosine embeddings via FAISS.
        </div>
      </div>
      <div style="background:#ffffff;border:1px solid rgba(26,26,46,0.10);
                  border-radius:14px;padding:20px;">
        <div style="width:36px;height:36px;background:rgba(79,70,229,0.08);
                    border-radius:10px;display:flex;align-items:center;
                    justify-content:center;font-size:18px;margin-bottom:12px;">💬</div>
        <div style="font-size:11px;font-weight:700;letter-spacing:0.8px;
                    text-transform:uppercase;color:#8a8aaa;margin-bottom:5px;">Step 4</div>
        <div style="font-size:14px;font-weight:600;color:#1a1a2e;margin-bottom:4px;">Ask</div>
        <div style="font-size:12px;color:#8a8aaa;line-height:1.5;">
          Hybrid BM25 + semantic retrieval feeds your LLM.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── File uploader ─────────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Drag and drop your PDF files here, or click to browse",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="visible",
    )

    if uploaded_files:
        current_names = [f.name for f in uploaded_files]

        if "faiss_index" not in st.session_state or \
           st.session_state.get("files") != current_names:

            with st.status("⚡ Processing documents...", expanded=True) as status:
                all_chunks = []

                for f in uploaded_files:
                    status.write(f"📄 Extracting: **{f.name}**")
                    prog = st.progress(0)

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.read())
                        tmp_path = tmp.name

                    pages, chunks = ingest_pdf(
                        pdf_path=tmp_path,
                        source_name=f.name,
                        model=model,
                        progress_callback=lambda v: prog.progress(v),
                    )

                    os.unlink(tmp_path)
                    prog.empty()

                    if chunks:
                        all_chunks.extend(chunks)
                        status.write(f"✅ {len(pages)} pages → {len(chunks)} chunks")
                    else:
                        status.write(f"⚠️ No text found in {f.name}")

                if not all_chunks:
                    st.error("No text could be extracted. Please check your PDF files.")
                    st.stop()

                with st.spinner("🔗 Building hybrid index (FAISS cosine + BM25)..."):
                    faiss_index, bm25_index, indexed_chunks = build_hybrid_index(
                        all_chunks, model
                    )

                st.session_state.faiss_index = faiss_index
                st.session_state.bm25_index  = bm25_index
                st.session_state.chunks      = indexed_chunks
                st.session_state.files       = current_names
                st.session_state.file_meta   = [
                    {"name": f.name, "chunks": len([
                        c for c in indexed_chunks if c["source"] == f.name
                    ])}
                    for f in uploaded_files
                ]

                status.update(label="✅ Documents indexed and ready!", state="complete")

        # ── Stats strip ───────────────────────────────────────────────────
        meta  = st.session_state.get("file_meta", [])
        total_chunks = sum(m["chunks"] for m in meta)

        st.markdown(f"""
        <div class="stats-strip">
          <div class="stat-card" style="--accent:#4f46e5;">
            <div class="stat-label">Documents Loaded</div>
            <div class="stat-value">{len(meta)}</div>
            <div class="stat-sub">PDF files indexed</div>
          </div>
          <div class="stat-card" style="--accent:#7c3aed;">
            <div class="stat-label">Chunks Indexed</div>
            <div class="stat-value">{total_chunks:,}</div>
            <div class="stat-sub">Semantic text units</div>
          </div>
          <div class="stat-card" style="--accent:#059669;">
            <div class="stat-label">Search Mode</div>
            <div class="stat-value" style="font-size:20px;padding-top:4px;">Hybrid</div>
            <div class="stat-sub">BM25 + Cosine similarity</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Documents table ───────────────────────────────────────────────
        st.markdown("""
        <div style="font-size:13px;font-weight:600;letter-spacing:0.3px;
                    color:#1a1a2e;margin-bottom:12px;">
          Indexed Documents
        </div>
        """, unsafe_allow_html=True)

        rows = "".join([
            f"""<tr>
              <td>
                <div style="display:flex;align-items:center;gap:10px;">
                  <div style="width:32px;height:32px;background:rgba(79,70,229,0.08);
                              border-radius:8px;display:flex;align-items:center;
                              justify-content:center;font-size:14px;flex-shrink:0;">📄</div>
                  <span style="font-weight:500;">{m['name']}</span>
                </div>
              </td>
              <td>
                <span class="status-badge">
                  <span class="status-dot"></span> Indexed
                </span>
              </td>
              <td style="color:#8a8aaa;font-size:13px;">{m['chunks']} chunks</td>
              <td style="color:#8a8aaa;font-size:13px;">BM25 + FAISS cosine</td>
            </tr>"""
            for m in meta
        ])

        st.markdown(f"""
        <table class="doc-table">
          <thead>
            <tr>
              <th style="width:45%;">File Name</th>
              <th>Status</th>
              <th>Chunks</th>
              <th>Index Type</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:16px;padding:14px 20px;background:rgba(79,70,229,0.05);
                    border:1px solid rgba(79,70,229,0.15);border-radius:12px;
                    font-size:13px;color:#4f46e5;display:flex;align-items:center;gap:8px;">
          ✦ &nbsp;Documents are ready. Switch to the <strong>Chat</strong> tab to start asking questions.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="margin-top:8px;padding:40px;background:#ffffff;
                    border:1px solid rgba(26,26,46,0.10);border-radius:16px;
                    text-align:center;">
          <div style="font-size:36px;margin-bottom:14px;">📭</div>
          <div style="font-family:'Playfair Display',serif;font-size:18px;
                      font-weight:600;color:#1a1a2e;margin-bottom:8px;">
            No documents yet
          </div>
          <div style="font-size:14px;color:#8a8aaa;max-width:340px;
                      margin:0 auto;line-height:1.6;">
            Upload a PDF above to get started. Supports text-based,
            scanned, and mixed-layout documents.
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:

    if "faiss_index" not in st.session_state:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;padding:80px 0;text-align:center;">
          <div style="width:64px;height:64px;background:rgba(79,70,229,0.08);
                      border-radius:16px;display:flex;align-items:center;
                      justify-content:center;font-size:28px;margin-bottom:20px;">📄</div>
          <div style="font-family:'Playfair Display',serif;font-size:22px;
                      font-weight:600;color:#1a1a2e;margin-bottom:10px;">
            No documents indexed yet
          </div>
          <div style="color:#8a8aaa;font-size:14px;max-width:380px;
                      line-height:1.7;font-weight:300;">
            Head to the <strong style="color:#4f46e5;">Documents</strong> tab,
            upload your PDFs, and come back here to start your conversation.
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # ── Welcome message if no history ─────────────────────────────────
        if not st.session_state.messages:
            loaded = st.session_state.get("files", [])
            names  = " · ".join(loaded) if loaded else "your documents"
            st.markdown(f"""
            <div style="background:#ffffff;border:1px solid rgba(26,26,46,0.10);
                        border-radius:16px;padding:28px 32px;margin-bottom:24px;">
              <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">
                <div style="width:40px;height:40px;background:#1a1a2e;border-radius:10px;
                            display:flex;align-items:center;justify-content:center;
                            font-size:18px;">✦</div>
                <div>
                  <div style="font-family:'Playfair Display',serif;font-size:16px;
                              font-weight:600;color:#1a1a2e;">DocMind is ready</div>
                  <div style="font-size:12px;color:#8a8aaa;margin-top:1px;">
                    {len(loaded)} document(s) indexed · Hybrid retrieval active
                  </div>
                </div>
              </div>
              <div style="font-size:13px;color:#4a4a6a;line-height:1.7;
                          border-top:1px solid rgba(26,26,46,0.07);padding-top:14px;">
                I've indexed <strong>{names}</strong>. You can ask me anything — 
                specific questions, summaries, comparisons across documents, 
                or page-level lookups. Try:
              </div>
              <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:14px;">
                <span style="background:rgba(79,70,229,0.07);border:1px solid rgba(79,70,229,0.15);
                             border-radius:8px;padding:6px 14px;font-size:12px;color:#4f46e5;
                             font-weight:500;">Summarize the document</span>
                <span style="background:rgba(79,70,229,0.07);border:1px solid rgba(79,70,229,0.15);
                             border-radius:8px;padding:6px 14px;font-size:12px;color:#4f46e5;
                             font-weight:500;">List all projects</span>
                <span style="background:rgba(79,70,229,0.07);border:1px solid rgba(79,70,229,0.15);
                             border-radius:8px;padding:6px 14px;font-size:12px;color:#4f46e5;
                             font-weight:500;">What are the key findings?</span>
                <span style="background:rgba(79,70,229,0.07);border:1px solid rgba(79,70,229,0.15);
                             border-radius:8px;padding:6px 14px;font-size:12px;color:#4f46e5;
                             font-weight:500;">Explain self-RAG</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Render chat history ───────────────────────────────────────────
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(
                    f'<div style="color:#2d2d44;font-size:15px;line-height:1.75;'
                    f'font-family:DM Sans,sans-serif;">{msg["content"]}</div>',
                    unsafe_allow_html=True
                )

        question = st.chat_input("Ask anything about your documents…")

        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(
                    f'<div style="color:#2d2d44;font-size:15px;">{question}</div>',
                    unsafe_allow_html=True
                )

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):

                    chunks      = st.session_state.chunks
                    faiss_index = st.session_state.faiss_index
                    bm25_index  = st.session_state.bm25_index

                    page_num = detect_page_query(question, chunks)
                    loaded_sources = list({c["source"] for c in chunks})
                    sources_str    = ", ".join(loaded_sources)

                    if page_num:
                        relevant = [c for c in chunks if c["page"] == page_num][:8]
                        for c in relevant:
                            c.setdefault("vector_score", "—")
                            c.setdefault("bm25_score",   "—")
                            c.setdefault("final_score",  "—")
                        intent = "specific"
                        corrected_query = None
                    else:
                        relevant, intent = hybrid_retrieve(
                            query=question,
                            faiss_index=faiss_index,
                            bm25_index=bm25_index,
                            chunks=chunks,
                            model=model,
                            top_k=8,
                            alpha=0.6,
                            use_spell_correction=True,
                        )
                        corrected_query = (
                            relevant[0].get("corrected_query") if relevant else None
                        )

                    if not relevant:
                        answer = "I couldn't find anything relevant in the indexed documents."
                    else:
                        if corrected_query:
                            st.caption(f"🔤 Query corrected: *\"{corrected_query}\"*")
                        if intent == "broad":
                            st.caption("🗂️ Broad query — scanning all sections & documents")

                        if intent == "broad":
                            relevant_sorted = sorted(
                                relevant,
                                key=lambda c: (c["source"], c["page"])
                            )
                            MAX_CONTEXT = 7000
                            context_parts, total_chars = [], 0
                            for c in relevant_sorted:
                                txt = c["text"]
                                if total_chars + len(txt) > MAX_CONTEXT:
                                    rem = MAX_CONTEXT - total_chars
                                    if rem > 100:
                                        context_parts.append(
                                            f"[SOURCE: {c['source']} | Page {c['page']}]\n{txt[:rem]}"
                                        )
                                    break
                                context_parts.append(
                                    f"[SOURCE: {c['source']} | Page {c['page']}]\n{txt}"
                                )
                                total_chars += len(txt)
                            context = "\n\n".join(context_parts)
                        else:
                            context = "\n\n".join(
                                f"[SOURCE: {c['source']} | Page {c['page']}]\n{c['text'][:1500]}"
                                for c in relevant
                            )

                        if intent == "broad":
                            prompt = f"""You are an expert assistant summarizing documents.

Available documents: {sources_str}

CRITICAL RULES — follow every one:
  1. Read EVERY chunk in the context — do not skip any section.
  2. List ALL items in list-type sections (projects, certifications, skills, etc.)
     — enumerate every single item, do NOT say "among others" or "etc."
  3. Write a structured summary with a heading for each section found.
  4. If you see 3 projects, mention all 3. If you see 6 certifications, list all 6.
  5. Be exhaustive and complete — the user wants everything.

Context (read every word):
{context}

Question: {question}

Complete structured summary:"""
                        else:
                            prompt = f"""You are an expert assistant answering questions about documents.

Available documents: {sources_str}

Rules:
- Answer using ONLY the context provided below.
- Give a clean, direct answer without mentioning source filenames or page numbers
  (a Sources panel already shows that).
- Use reasoning — if the user asks for something that's implied but not explicit,
  infer it from context.
- Only say "not found" if the context genuinely contains nothing relevant.
- When listing items (projects, skills, certifications), list ALL of them — be exhaustive.

Context:
{context}

Question: {question}
Answer:"""

                        resp = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.2,
                            max_tokens=1500 if intent == "broad" else 1000,
                        )
                        raw    = resp.choices[0].message.content
                        answer = re.sub(r'`([^`]+)`', r'\1', raw)

                    # Stream word by word using st.write_stream
                    def _stream_answer():
                        for word in answer.split(" "):
                            yield word + " "
                            time.sleep(0.04)

                    st.write_stream(_stream_answer())

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                    })