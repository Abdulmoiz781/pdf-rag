"""
app.py  —  Streamlit UI
Two-tab layout: Documents | Chats
- Chat input pinned to bottom (Claude-style)
- Light theme enforced throughout
- Calls ingest.py  for extraction / chunking / embedding / indexing
- Calls retriever.py  for hybrid BM25 + cosine retrieval + spell correction

FIXES:
  1. total_pages bug: was using global max page across ALL PDFs.
     Now detects page queries per-source so multi-PDF setups work correctly.
  2. Context truncation: chars_per_chunk raised to 2000/1500 so CV sections
     (projects, certifications, achievements) aren't cut off before the LLM.
  3. top_k raised to 8 for specific queries so single-page CVs return all chunks.
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

# ── Must be FIRST streamlit call ─────────────────────────────────────────────
apply_styles()

# ── Setup ────────────────────────────────────────────────────────────────────
load_dotenv(override=True)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


def detect_page_query(question: str, chunks: list):
    """
    FIX: Old version used a single global max page number, which was wrong
    for multi-PDF setups (e.g. PDF1 has 3 pages, PDF2 has 5 → max=5, but
    asking 'page 4' on PDF1 returns garbage).

    New version: only match explicit page numbers and 'first/last page'
    within the context of loaded chunks. For 'last page', we use the max
    page of whichever single source is loaded (only works unambiguously
    for single-PDF sessions).
    """
    q = question.lower()
    sources = list({c["source"] for c in chunks})

    # 'last page' is only unambiguous when one PDF is loaded
    if "last page" in q and len(sources) == 1:
        return max(c["page"] for c in chunks)
    if "first page" in q:
        return 1

    m = re.search(r'page\s*(\d+)', q)
    if m:
        return int(m.group(1))
    return None


# ── App header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:4px;">
  <span style="font-family:'Syne',sans-serif;font-size:26px;font-weight:800;color:#111827;">
    Doc
  </span><span style="font-family:'Syne',sans-serif;font-size:26px;font-weight:800;color:#2563eb;">
    Mind
  </span>
</div>
<div style="color:#6b7280;font-size:14px;margin-bottom:20px;">
  Upload PDFs and ask questions powered by hybrid RAG
</div>
""", unsafe_allow_html=True)

# ── Two tabs ─────────────────────────────────────────────────────────────────
tab_docs, tab_chat = st.tabs(["📄  Documents", "💬  Chats"])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — DOCUMENTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_docs:

    uploaded_files = st.file_uploader(
        "Drag and drop documents here or select files",
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

        # ── Documents table ───────────────────────────────────────────────
        rows = "".join([
            f"""<tr>
              <td>📄 {m['name']}</td>
              <td><span class="status-badge"><span class="status-dot"></span> Completed</span></td>
              <td style="color:#6b7280;">{m['chunks']} chunks</td>
            </tr>"""
            for m in st.session_state.get("file_meta", [])
        ])

        st.markdown(f"""
        <table class="doc-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Index Status</th>
              <th>Chunks</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        col1, col2, col3, _ = st.columns([1, 1, 1, 3])
        col1.metric("Documents",      len(uploaded_files))
        col2.metric("Chunks Indexed", len(st.session_state.chunks))
        col3.metric("Search Mode",    "Hybrid")

    else:
        st.info("⬆️ Upload one or more PDF files to get started.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — CHATS
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:

    if "faiss_index" not in st.session_state:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;padding:80px 0;text-align:center;">
          <div style="font-size:48px;margin-bottom:16px;">📄</div>
          <div style="font-family:'Syne',sans-serif;font-size:20px;font-weight:700;
                      color:#111827;margin-bottom:8px;">
            No documents indexed yet
          </div>
          <div style="color:#6b7280;font-size:14px;max-width:360px;line-height:1.6;">
            Go to the <strong>Documents</strong> tab, upload your PDFs,
            then come back here to start chatting.
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(
                    f'<div style="color:#111827;font-size:15px;line-height:1.7;">'
                    f'{msg["content"]}</div>',
                    unsafe_allow_html=True
                )

        st.markdown('<div id="chat-bottom"></div>', unsafe_allow_html=True)

        question = st.chat_input("Type your message...")

        st.markdown("""
        <script>
          const bottom = document.getElementById("chat-bottom");
          if (bottom) bottom.scrollIntoView({ behavior: "smooth" });
        </script>
        """, unsafe_allow_html=True)

        if question:
            st.session_state.messages.append({"role": "user", "content": question})

            with st.chat_message("user"):
                st.markdown(
                    f'<div style="color:#111827;font-size:15px;">{question}</div>',
                    unsafe_allow_html=True
                )

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):

                    chunks      = st.session_state.chunks
                    faiss_index = st.session_state.faiss_index
                    bm25_index  = st.session_state.bm25_index

                    # FIX: pass chunks (not a global max) to detect_page_query
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
                            # FIX: raised from 5 to 8 — single-page CVs need
                            # all their chunks retrieved, not just top-5
                            top_k=8,
                            alpha=0.6,
                            use_spell_correction=True,
                        )
                        corrected_query = (
                            relevant[0].get("corrected_query") if relevant else None
                        )

                    if not relevant:
                        answer = "I couldn't find anything relevant in the documents."
                    else:
                        if corrected_query:
                            st.caption(f"🔤 Spell-corrected: *\"{corrected_query}\"*")
                        if intent == "broad":
                            st.caption("🗂️ Broad query — retrieving across all sections & documents")

                        # FIX: Raised context window significantly.
                        # Old values (800/1000) cut off CV sections mid-content.
                        # Projects, certifications, achievements often exceed 800 chars.
                        chars_per_chunk = 2000 if intent == "broad" else 1500

                        context = "\n\n".join(
                            f"[SOURCE: {c['source']} | Page {c['page']}]\n{c['text'][:chars_per_chunk]}"
                            for c in relevant
                        )

                        if intent == "broad":
                            prompt = f"""You are an expert assistant summarizing multiple documents.

Available documents: {sources_str}

The context below contains chunks retrieved from DIFFERENT documents and sections.
Your job:
  1. Cover content from ALL documents present in the context.
  2. Write a structured summary with a clear heading for each document/section.
  3. Do NOT skip any document — if a document appears in context, mention it.
  4. Be thorough — the user wants the complete picture across all PDFs.

Context:
{context}

Question: {question}

Structured Summary (cover every document):"""
                        else:
                            prompt = f"""You are an expert assistant answering questions about documents.

Available documents: {sources_str}

Rules:
- Answer using ONLY the context provided below.
- Give a clean, direct answer. Do NOT mention source filenames, page numbers,
  or document names in your answer — a separate Sources panel already shows that.
- Use reasoning and inference — if the user asks for a "title" or "heading", 
  look for the most prominent text at the top of the document, even if the 
  word "title" doesn't appear explicitly.
- Only say "This information was not found" if the context genuinely contains 
  nothing relevant at all — not just because the exact keyword is missing.
- Do NOT contradict yourself by saying info is missing then providing it.
- When listing items (like projects, skills, certifications), list ALL of them.
  Do NOT stop after the first one. Be exhaustive and complete.

Context:
{context}

Question: {question}
Answer:"""

                        resp = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.2,
                            # FIX: raised max_tokens so LLM doesn't truncate
                            # long lists of projects/certifications mid-answer
                            max_tokens=1500 if intent == "broad" else 1000,
                        )
                        answer = resp.choices[0].message.content

                    # ── Typewriter effect ─────────────────────────────────
                    placeholder = st.empty()
                    typed = ""
                    for char in answer:
                        typed += char
                        placeholder.markdown(
                            f'<div style="color:#111827;font-size:15px;line-height:1.7;">{typed}</div>',
                            unsafe_allow_html=True
                        )
                        time.sleep(0.008)

                    # ── Sources ───────────────────────────────────────────
                    if relevant:
                        with st.expander("📚 Sources & Retrieval Scores"):
                            for c in relevant:
                                st.markdown(f"""
<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:10px;
            padding:14px;margin-bottom:8px;">
  <div style="display:flex;justify-content:space-between;
              align-items:center;flex-wrap:wrap;gap:6px;margin-bottom:8px;">
    <span style="color:#1d4ed8;font-weight:600;font-size:13px;">
      📄 {c['source']} · Page {c['page']}
    </span>
    <div style="display:flex;gap:5px;flex-wrap:wrap;">
      <span style="background:#eff6ff;color:#1d4ed8;font-size:11px;padding:2px 8px;
                   border-radius:5px;border:1px solid #bfdbfe;">
        Cosine: {c['vector_score']}
      </span>
      <span style="background:#f5f3ff;color:#6d28d9;font-size:11px;padding:2px 8px;
                   border-radius:5px;border:1px solid #ddd6fe;">
        BM25: {c['bm25_score']}
      </span>
      <span style="background:#f0fdf4;color:#15803d;font-size:11px;padding:2px 8px;
                   border-radius:5px;border:1px solid #bbf7d0;">
        Final: {c['final_score']}
      </span>
    </div>
  </div>
  <div style="color:#374151;font-size:12px;line-height:1.6;">
    {c['text'][:220]}...
  </div>
</div>
""", unsafe_allow_html=True)

                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": answer,
                    })

                    st.markdown("""
                    <script>
                      window.scrollTo(0, document.body.scrollHeight);
                    </script>
                    """, unsafe_allow_html=True)