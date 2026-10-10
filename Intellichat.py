import os
import time
from itertools import chain

import chromadb
import pandas as pd
import streamlit as st
from groq import Groq
from pypdf import PdfReader

# ============================================================
# INTELLICHAT: DOCUMENT-AWARE AI ASSISTANT
# STREAMLIT + GROQ + CHROMADB
# ============================================================

MODEL = "openai/gpt-oss-20b"

MODES = [
    "Auto (documents first, then general)",
    "Documents only",
    "General chat only",
]

GENERAL_PROMPT = """You are IntelliChat, a helpful, accurate and friendly AI assistant.
Explain clearly, use examples when useful, never make up information,
and be concise unless the user asks for detail."""

RAG_PROMPT = """You are IntelliChat, an assistant that answers from the user's documents.

Rules:
- Answer ONLY from the context below. Do not use outside knowledge.
- If the context does not contain the answer, reply with exactly: NOT_IN_DOCUMENTS
- Cite sources after facts like [file.pdf, p.3].
- Be concise, well structured and accurate.

Context:
"""

SUMMARY_PROMPT = """You analyse documents. Using ONLY the text provided, write clean Markdown:

**📌 What it is** - one sentence on the document type and purpose.
**📝 Overview** - 3 to 4 sentences.
**🔑 Key points** - 5 short bullets.
**👥 Who it's for** - one line.

If there are several documents, add a short heading with each file name.
Finish with a line exactly `SUGGESTED_QUESTIONS:` followed by 3 useful questions
a reader might ask, one per line, each starting with "- "."""

# ============================================================
# PAGE + THEME
# ============================================================

st.set_page_config(page_title="IntelliChat", page_icon="✨", layout="wide")

st.markdown("""
<style>
:root { --acc:#6366f1; --acc2:#22d3ee; --bd:#26304f; --mut:#94a3b8; }
.stApp {
    background:
      radial-gradient(1100px 560px at 8% -10%, #1b2250 0%, transparent 60%),
      radial-gradient(900px 480px at 100% 0%, #0f3a4a 0%, transparent 55%),
      #0b1020;
}
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; max-width: 1100px; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1530, #0b1020);
    border-right: 1px solid var(--bd);
}
.brand {
    font-size: 36px; font-weight: 800; letter-spacing: -.5px;
    background: linear-gradient(90deg, #818cf8, #22d3ee);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.tag { color: var(--mut); margin: -4px 0 18px 0; font-size: 15px; }
.stat {
    display: flex; gap: 12px; align-items: center;
    background: rgba(18,26,51,.8); border: 1px solid var(--bd);
    border-radius: 14px; padding: 14px 16px;
}
.stat-ic { font-size: 24px; }
.stat-v { font-size: 20px; font-weight: 700; color: #fff; line-height: 1.1; }
.stat-l { font-size: 12px; color: var(--mut); }
.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 14px; }
.step {
    background: rgba(18,26,51,.75); border: 1px solid var(--bd);
    border-radius: 16px; padding: 20px;
}
.step b { color: #fff; font-size: 16px; }
.step p { color: var(--mut); margin: 6px 0 0 0; font-size: 14px; }
.step .n {
    display: inline-block; width: 28px; height: 28px; line-height: 28px;
    text-align: center; border-radius: 50%; margin-bottom: 10px; font-weight: 700;
    background: linear-gradient(135deg, #6366f1, #22d3ee); color: #fff;
}
[data-testid="stChatMessage"] {
    background: rgba(18,26,51,.65); border: 1px solid var(--bd);
    border-radius: 16px; padding: 14px 16px;
}
.stButton > button, .stDownloadButton > button {
    border-radius: 12px; border: 1px solid var(--bd); transition: .15s;
}
.stButton > button:hover { border-color: var(--acc); transform: translateY(-1px); }
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(90deg, #6366f1, #22d3ee);
    border: none; color: #fff; font-weight: 600;
}
@media (max-width: 800px) { .steps { grid-template-columns: 1fr; } }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "messages": [],
    "chunk_count": 0,
    "docs_info": [],
    "suggestions": [],
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)

# ============================================================
# DOCUMENT PIPELINE: LOAD -> CLEAN -> CHUNK -> EMBED -> STORE
# ============================================================


@st.cache_resource
def get_chroma_client():
    return chromadb.EphemeralClient()


def extract_pages(f):
    """Return a list of (page_number, text) for pdf, docx, txt or md files."""
    name = f.name.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(f)
        return [(i, p.extract_text() or "") for i, p in enumerate(reader.pages, 1)]

    if name.endswith(".docx"):
        from docx import Document
        doc = Document(f)
        return [(1, "\n".join(p.text for p in doc.paragraphs))]

    return [(1, f.read().decode("utf-8", errors="ignore"))]


def chunk_text(text, size, overlap):
    text = " ".join(text.split())
    chunks, start = [], 0

    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start, end)
            if cut > start + size // 2:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks


def build_index(files, chunk_size, overlap):
    """Index every file. Returns (info, pages_by_file, chunk_count)."""
    client = get_chroma_client()
    try:
        client.delete_collection("docs")
    except Exception:
        pass
    collection = client.create_collection("docs")

    ids, docs, metas, info, pages_by_file = [], [], [], [], {}

    for n, f in enumerate(files):
        try:
            pages = extract_pages(f)
        except Exception as e:
            st.warning(f"Could not read {f.name}: {e}")
            continue

        pages_by_file[f.name] = pages
        words = sum(len(t.split()) for _, t in pages)
        info.append({"name": f.name, "pages": len(pages), "words": words})

        for page_no, text in pages:
            for c, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
                ids.append(f"{n}-{f.name}-p{page_no}-c{c}")
                docs.append(chunk)
                metas.append({"source": f.name, "page": page_no})

    for i in range(0, len(docs), 100):
        collection.add(ids=ids[i:i + 100], documents=docs[i:i + 100],
                       metadatas=metas[i:i + 100])

    return info, pages_by_file, len(docs)


def retrieve(query, k):
    collection = get_chroma_client().get_collection("docs")
    k = max(1, min(k, collection.count()))
    res = collection.query(query_texts=[query], n_results=k)
    return [
        {"text": d, "source": m["source"], "page": m["page"]}
        for d, m in zip(res["documents"][0], res["metadatas"][0])
    ]


# ============================================================
# LLM HELPERS (STREAMING FOR SPEED)
# ============================================================


def stream_llm(client, messages, temperature, reasoning):
    stream = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=float(temperature),
        reasoning_effort=reasoning, max_completion_tokens=4096, stream=True,
    )
    for event in stream:
        if event.choices and event.choices[0].delta.content:
            yield event.choices[0].delta.content


def route(client, question, history, mode, k, temperature, reasoning):
    """Hybrid router. Returns (text_stream, sources, label)."""

    if mode != "General chat only":
        if st.session_state.chunk_count == 0:
            if mode == "Documents only":
                return iter(["Please upload and analyse a document first."]), [], "⚠️ No documents"
        else:
            chunks = retrieve(question, k)
            context = "\n\n".join(
                f"[{c['source']}, p.{c['page']}]\n{c['text']}" for c in chunks
            )
            messages = ([{"role": "system", "content": RAG_PROMPT + context}]
                        + history + [{"role": "user", "content": question}])

            stream = stream_llm(client, messages, temperature, reasoning)
            head, buffered = "", []
            for piece in stream:  # peek at the start to detect "not in documents"
                buffered.append(piece)
                head += piece
                if len(head) >= 16:
                    break

            if not head.strip().upper().startswith("NOT_IN_DOC"):
                return chain(buffered, stream), chunks, "📄 From your documents"

            if mode == "Documents only":
                return (iter(["I couldn't find this in the uploaded documents."]),
                        chunks, "🚫 Not in documents")

    messages = ([{"role": "system", "content": GENERAL_PROMPT}]
                + history + [{"role": "user", "content": question}])
    return (stream_llm(client, messages, temperature, reasoning), [],
            "🌐 General knowledge (not verified against documents)")


def sample_text(pages, budget):
    """Fast summary input: whole text if short, else evenly spaced slices."""
    full = " ".join(" ".join(t.split()) for _, t in pages)
    if len(full) <= budget:
        return full
    n = 6
    piece = budget // n
    step = (len(full) - piece) // (n - 1)
    return " ... ".join(full[i * step: i * step + piece] for i in range(n))


def summarise(client, pages_by_file, budget=12000):
    per_file = budget // max(1, len(pages_by_file))
    body = "\n\n".join(
        f"=== {name} ===\n{sample_text(pages, per_file)}"
        for name, pages in pages_by_file.items()
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SUMMARY_PROMPT},
                  {"role": "user", "content": body}],
        temperature=0.2, reasoning_effort="low", max_completion_tokens=1800,
    )
    text = (response.choices[0].message.content or "").strip()

    summary, _, tail = text.partition("SUGGESTED_QUESTIONS:")
    questions = [ln.lstrip("-• ").strip() for ln in tail.splitlines()
                 if ln.strip().startswith(("-", "•"))][:3]
    return summary.strip(), questions


def render_message(msg):
    with st.chat_message(msg["role"], avatar="✨" if msg["role"] == "assistant" else None):
        if msg.get("label"):
            took = f" · ⏱ {msg['time']:.1f}s" if msg.get("time") else ""
            st.caption(msg["label"] + took)
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']} · page {s['page']}**")
                    st.caption(s["text"])


def stat_card(icon, value, label):
    return (f'<div class="stat"><div class="stat-ic">{icon}</div>'
            f'<div><div class="stat-v">{value}</div>'
            f'<div class="stat-l">{label}</div></div></div>')


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown('<div class="brand" style="font-size:28px">✨ IntelliChat</div>',
                unsafe_allow_html=True)
    page = st.radio("View", ["💬 Chat", "📊 Evaluation"],
                    horizontal=True, label_visibility="collapsed")

    api_key = st.text_input("🔑 Groq API key", type="password",
                            placeholder="gsk_...", value=os.getenv("GROQ_API_KEY", ""))

    st.markdown("##### 📄 Your documents")
    uploaded = st.file_uploader("Drop files", type=["pdf", "txt", "md", "docx"],
                                accept_multiple_files=True,
                                label_visibility="collapsed")

    mode = st.selectbox("🧭 Answer mode", MODES)

    with st.expander("⚙️ Advanced settings"):
        temperature = st.slider("Temperature", 0.0, 1.5, 0.3, 0.1)
        reasoning = st.selectbox("Reasoning effort", ["low", "medium", "high"], index=0)
        chunk_size = st.slider("Chunk size (characters)", 200, 2000, 800, 100)
        overlap = st.slider("Chunk overlap", 0, 400, 100, 50)
        top_k = st.slider("Top-K chunks", 1, 10, 4)

    if st.button("🚀 Analyse documents", type="primary",
                 use_container_width=True, disabled=not uploaded):
        if not api_key.strip():
            st.error("Enter your Groq API key first.")
        else:
            started = time.time()
            with st.spinner("Reading, indexing and summarising..."):
                info, pages_by_file, count = build_index(uploaded, chunk_size, overlap)

                if count == 0:
                    st.warning("No text found. Scanned files need OCR; try a digital file.")
                else:
                    try:
                        summary, questions = summarise(Groq(api_key=api_key.strip()),
                                                       pages_by_file)
                    except Exception as e:
                        summary, questions = f"Documents indexed, but the summary failed: {e}", []

                    st.session_state.chunk_count = count
                    st.session_state.docs_info = info
                    st.session_state.suggestions = questions
                    st.session_state.messages = [{
                        "role": "assistant", "content": summary, "kind": "summary",
                        "label": "📋 Document summary", "time": time.time() - started,
                    }]

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.suggestions = []
        st.rerun()

# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="brand">IntelliChat</div>', unsafe_allow_html=True)
st.markdown('<div class="tag">Upload a document. Get an instant summary. '
            'Ask anything, with sources.</div>', unsafe_allow_html=True)

# ============================================================
# PAGE 1: CHAT
# ============================================================

if page == "💬 Chat":

    info = st.session_state.docs_info
    if info:
        words = sum(d["words"] for d in info)
        cols = st.columns(4)
        cols[0].markdown(stat_card("📄", len(info), "Documents"), unsafe_allow_html=True)
        cols[1].markdown(stat_card("📑", sum(d["pages"] for d in info), "Pages"), unsafe_allow_html=True)
        cols[2].markdown(stat_card("🧩", st.session_state.chunk_count, "Searchable chunks"), unsafe_allow_html=True)
        cols[3].markdown(stat_card("⏱", f"{max(1, round(words / 200))} min", "Reading time"), unsafe_allow_html=True)
        st.write("")
    elif not st.session_state.messages:
        st.markdown("""
        <div class="steps">
          <div class="step"><div class="n">1</div><b>Add your key</b>
            <p>Paste your Groq API key in the sidebar.</p></div>
          <div class="step"><div class="n">2</div><b>Upload documents</b>
            <p>PDF, Word, TXT or Markdown, then click Analyse.</p></div>
          <div class="step"><div class="n">3</div><b>Chat with them</b>
            <p>Get a summary at once, then ask questions with cited sources.</p></div>
        </div>""", unsafe_allow_html=True)

    for message in st.session_state.messages:
        render_message(message)

    # Suggested questions from the summary
    if st.session_state.suggestions and len(st.session_state.messages) <= 1:
        st.caption("💡 Try asking:")
        for i, q in enumerate(st.session_state.suggestions):
            if st.button(q, key=f"sugg{i}", use_container_width=True):
                st.session_state.pending_q = q
                st.session_state.suggestions = []
                st.rerun()

    typed = st.chat_input("Ask anything about your documents...")
    user_message = typed or st.session_state.pop("pending_q", None)

    if user_message:
        if not api_key.strip():
            st.error("⚠️ Please enter your Groq API key in the sidebar.")
            st.stop()

        user_msg = {"role": "user", "content": user_message}
        render_message(user_msg)

        history = [{"role": m["role"], "content": m["content"]}
                   for m in st.session_state.messages
                   if m.get("kind") != "summary"][-6:]

        try:
            started = time.time()
            client = Groq(api_key=api_key.strip())
            stream, sources, label = route(client, user_message, history,
                                           mode, top_k, temperature, reasoning)

            with st.chat_message("assistant", avatar="✨"):
                st.caption(label)
                answer = st.write_stream(stream)
                if sources:
                    with st.expander(f"📚 Sources ({len(sources)})"):
                        for s in sources:
                            st.markdown(f"**{s['source']} · page {s['page']}**")
                            st.caption(s["text"])

            st.session_state.messages.append(user_msg)
            st.session_state.messages.append({
                "role": "assistant", "content": answer, "sources": sources,
                "label": label, "time": time.time() - started,
            })
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error: {e}")

# ============================================================
# PAGE 2: EVALUATION
# ============================================================

else:
    st.subheader("📊 Evaluation")
    st.caption("Upload a CSV or Excel file with the columns: question, expected_keyword. "
               "The keyword is a short phrase or number found in the correct passage.")

    if st.session_state.chunk_count == 0:
        st.warning("Analyse your documents first (sidebar), then run the evaluation.")

    test_file = st.file_uploader("Test set", type=["csv", "xlsx"], key="eval_file")
    check_answers = st.checkbox("Also check generated answers (one API call per question)", True)

    if test_file is not None and st.button("▶️ Run evaluation", type="primary"):

        if st.session_state.chunk_count == 0:
            st.error("Analyse your documents in the sidebar first.")
            st.stop()
        if check_answers and not api_key.strip():
            st.error("Enter your Groq API key to check answers.")
            st.stop()

        df = (pd.read_excel(test_file) if test_file.name.lower().endswith(".xlsx")
              else pd.read_csv(test_file))

        if not {"question", "expected_keyword"}.issubset(df.columns):
            st.error("The file needs the columns: question, expected_keyword")
            st.stop()

        client = Groq(api_key=api_key.strip()) if check_answers else None
        rows, bar = [], st.progress(0.0)

        for _, row in df.iterrows():
            question, keyword = str(row["question"]), str(row["expected_keyword"]).lower()

            hits = [keyword in c["text"].lower() for c in retrieve(question, top_k)]
            first = hits.index(True) + 1 if any(hits) else None

            result = {
                "question": question,
                "expected_keyword": row["expected_keyword"],
                "hit@k": int(any(hits)),
                "precision@k": round(sum(hits) / len(hits), 2),
                "reciprocal_rank": round(1 / first, 2) if first else 0.0,
            }

            if check_answers:
                stream, _, _ = route(client, question, [], "Documents only",
                                     top_k, temperature, reasoning)
                answer = "".join(stream)
                result["answer_correct"] = int(keyword in answer.lower())
                result["answer"] = answer

            rows.append(result)
            bar.progress(len(rows) / len(df))

        results = pd.DataFrame(rows)

        cols = st.columns(4 if check_answers else 3)
        cols[0].metric(f"Hit@{top_k}", f"{results['hit@k'].mean():.0%}")
        cols[1].metric(f"Precision@{top_k}", f"{results['precision@k'].mean():.2f}")
        cols[2].metric("MRR", f"{results['reciprocal_rank'].mean():.2f}")
        if check_answers:
            cols[3].metric("Answer accuracy", f"{results['answer_correct'].mean():.0%}")

        st.dataframe(results, use_container_width=True)
        st.download_button("⬇️ Download results (CSV)", results.to_csv(index=False),
                           file_name="intellichat_evaluation.csv", mime="text/csv")
