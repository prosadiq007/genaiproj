import os

import chromadb
import pandas as pd
import streamlit as st
from groq import Groq
from pypdf import PdfReader

# ============================================================
# INTELLICHAT: RAG + GENERAL CHATBOT
# STREAMLIT + GROQ API + CHROMADB
# ============================================================

MODEL = "openai/gpt-oss-20b"

GENERAL_PROMPT = """
You are a helpful, intelligent and friendly AI assistant.

Rules:
- Answer questions accurately.
- Explain difficult topics clearly.
- Use examples when useful.
- Do not make up information.
- Remember the conversation context.
- Be concise unless the user asks for detailed information.
"""

RAG_PROMPT = """
You are IntelliChat, an assistant that answers questions from the
user's uploaded documents.

Rules:
- Answer ONLY using the context below. Do not use outside knowledge.
- If the context does not contain the answer, reply with exactly:
  NOT_IN_DOCUMENTS
- Cite the source after each fact, like [file.pdf, p.3].
- Be concise and accurate.

Context:
"""

MODES = [
    "Auto (documents first, then general)",
    "Documents only",
    "General chat only",
]

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SADIQ's Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
.title { font-size: 40px; font-weight: 700; margin-bottom: 0px; }
.subtitle { font-size: 16px; color: #9ca3af; margin-bottom: 25px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "doc_names" not in st.session_state:
    st.session_state.doc_names = []

# ============================================================
# RAG PIPELINE: LOAD -> CLEAN -> CHUNK -> EMBED -> STORE
# ============================================================


@st.cache_resource
def get_chroma_client():
    # In-memory vector database. ChromaDB embeds the text automatically
    # with its default sentence-embedding model.
    return chromadb.EphemeralClient()


def chunk_text(text, size, overlap):
    """Split text into overlapping chunks, preferring to cut at spaces."""
    text = " ".join(text.split())  # cleaning: collapse whitespace and newlines
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))

        if end < len(text):
            cut = text.rfind(" ", start, end)
            if cut > start + size // 2:
                end = cut

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def build_index(files, chunk_size, overlap):
    """Read the PDFs, chunk them and store them in ChromaDB."""
    client = get_chroma_client()

    try:
        client.delete_collection("docs")
    except Exception:
        pass

    collection = client.create_collection("docs")

    ids, documents, metadatas = [], [], []

    for file_no, f in enumerate(files):
        try:
            reader = PdfReader(f)
            pages = list(reader.pages)
        except Exception as e:
            st.warning(f"Could not read {f.name}: {e}")
            continue

        for page_no, page in enumerate(pages, start=1):
            text = page.extract_text() or ""

            for chunk_no, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
                ids.append(f"{file_no}-{f.name}-p{page_no}-c{chunk_no}")
                documents.append(chunk)
                metadatas.append({"source": f.name, "page": page_no})

    for i in range(0, len(documents), 100):
        collection.add(
            ids=ids[i:i + 100],
            documents=documents[i:i + 100],
            metadatas=metadatas[i:i + 100],
        )

    return len(documents)


def retrieve(query, k):
    """Return the top-k most similar chunks for a query."""
    collection = get_chroma_client().get_collection("docs")
    k = max(1, min(k, collection.count()))
    result = collection.query(query_texts=[query], n_results=k)

    chunks = []
    for doc, meta, dist in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "page": meta["page"],
            "distance": dist,
        })
    return chunks


# ============================================================
# LLM HELPERS
# ============================================================


def call_llm(client, messages, temperature, reasoning):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=float(temperature),
        reasoning_effort=reasoning,
        max_completion_tokens=4096,
    )
    return (response.choices[0].message.content or "").strip()


def answer_question(client, question, history, mode, k, temperature, reasoning):
    """
    Hybrid router.
    1. Try RAG (answer from the PDFs only).
    2. If the model says NOT_IN_DOCUMENTS, either refuse (Documents only)
       or fall back to general knowledge (Auto).
    Returns (answer, sources, label).
    """
    has_docs = st.session_state.chunk_count > 0

    if mode != "General chat only":

        if not has_docs:
            if mode == "Documents only":
                return "Please upload and process PDFs first.", [], "⚠️ No documents"
        else:
            chunks = retrieve(question, k)

            context = "\n\n".join(
                f"[{c['source']}, p.{c['page']}]\n{c['text']}" for c in chunks
            )

            messages = (
                [{"role": "system", "content": RAG_PROMPT + context}]
                + history
                + [{"role": "user", "content": question}]
            )

            answer = call_llm(client, messages, temperature, reasoning)

            if "NOT_IN_DOCUMENTS" not in answer:
                return answer, chunks, "📄 Answered from your documents"

            if mode == "Documents only":
                return (
                    "I couldn't find this in the uploaded documents.",
                    chunks,
                    "🚫 Not in documents",
                )

    # General knowledge fallback
    messages = (
        [{"role": "system", "content": GENERAL_PROMPT}]
        + history
        + [{"role": "user", "content": question}]
    )
    answer = call_llm(client, messages, temperature, reasoning)
    return answer, [], "🌐 General knowledge (not verified against documents)"


def render_message(msg):
    with st.chat_message(msg["role"]):
        if msg.get("label"):
            st.caption(msg["label"])

        st.markdown(msg["content"])

        if msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])} chunks)"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']}, page {s['page']}**")
                    st.write(s["text"])


# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="title">🤖 IntelliChat</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">⚡ Powered by Groq · RAG over your PDFs</div>',
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    page = st.radio("View", ["💬 Chat", "📊 Evaluation"], horizontal=True)

    st.divider()
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        value=os.getenv("GROQ_API_KEY", ""),
    )
    st.caption("Your API key is used only for this session.")

    temperature = st.slider("🌡️ Temperature", 0.0, 1.5, 0.3, 0.1)

    reasoning = st.selectbox(
        "🧠 Reasoning Effort", ["low", "medium", "high"], index=1
    )

    mode = st.selectbox("🧭 Answer mode", MODES)

    st.divider()
    st.header("📄 Documents")

    uploaded = st.file_uploader(
        "Upload PDFs", type="pdf", accept_multiple_files=True
    )

    chunk_size = st.slider("Chunk size (characters)", 200, 2000, 800, 100)
    overlap = st.slider("Chunk overlap", 0, 400, 100, 50)
    top_k = st.slider("Top-K chunks to retrieve", 1, 10, 4)

    if st.button(
        "⚙️ Process documents",
        use_container_width=True,
        disabled=not uploaded,
    ):
        with st.spinner("Reading and indexing PDFs..."):
            count = build_index(uploaded, chunk_size, overlap)

        st.session_state.chunk_count = count
        st.session_state.doc_names = [f.name for f in uploaded]

        if count == 0:
            st.warning(
                "No text found. Scanned PDFs need OCR, so try a digital PDF."
            )

    if st.session_state.chunk_count > 0:
        st.success(
            f"Indexed {st.session_state.chunk_count} chunks from "
            f"{len(st.session_state.doc_names)} file(s)."
        )

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ============================================================
# PAGE 1: CHAT
# ============================================================

if page == "💬 Chat":

    for message in st.session_state.messages:
        render_message(message)

    user_message = st.chat_input("Ask a question...")

    if user_message:

        if not api_key.strip():
            st.error("⚠️ Please enter your Groq API key in the sidebar.")
            st.stop()

        user_msg = {"role": "user", "content": user_message}
        render_message(user_msg)

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[-6:]
        ]

        try:
            with st.spinner("Thinking..."):
                client = Groq(api_key=api_key.strip())

                answer, sources, label = answer_question(
                    client, user_message, history,
                    mode, top_k, temperature, reasoning,
                )

            if not answer:
                answer = "I couldn't generate a response."

            assistant_msg = {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "label": label,
            }

            render_message(assistant_msg)

            st.session_state.messages.append(user_msg)
            st.session_state.messages.append(assistant_msg)

        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"❌ Error\n\n{e}")

# ============================================================
# PAGE 2: EVALUATION
# ============================================================

else:

    st.subheader("📊 Evaluation")
    st.caption(
        "Upload a CSV with two columns: question, expected_keyword. "
        "The keyword should be a short, unique phrase or number that "
        "appears in the correct passage of your PDFs."
    )

    if st.session_state.chunk_count == 0:
        st.info("Upload and process your PDFs first.")
        st.stop()

    test_file = st.file_uploader("Test set (CSV)", type="csv", key="eval_csv")
    check_answers = st.checkbox(
        "Also check generated answers (uses one API call per question)",
        value=True,
    )

    if test_file is not None and st.button("▶️ Run evaluation"):

        df = pd.read_csv(test_file)

        if not {"question", "expected_keyword"}.issubset(df.columns):
            st.error("CSV must have the columns: question, expected_keyword")
            st.stop()

        if check_answers and not api_key.strip():
            st.error("Enter your Groq API key to check answers.")
            st.stop()

        client = Groq(api_key=api_key.strip()) if check_answers else None

        rows = []
        progress = st.progress(0.0)

        for i, row in df.iterrows():

            question = str(row["question"])
            keyword = str(row["expected_keyword"]).lower()

            chunks = retrieve(question, top_k)
            hits = [keyword in c["text"].lower() for c in chunks]

            hit = any(hits)
            precision = sum(hits) / len(hits)
            rr = 1 / (hits.index(True) + 1) if hit else 0.0

            result = {
                "question": question,
                "expected_keyword": row["expected_keyword"],
                "hit@k": int(hit),
                "precision@k": round(precision, 2),
                "reciprocal_rank": round(rr, 2),
            }

            if check_answers:
                answer, _, _ = answer_question(
                    client, question, [], "Documents only",
                    top_k, temperature, reasoning,
                )
                result["answer_correct"] = int(keyword in answer.lower())
                result["answer"] = answer

            rows.append(result)
            progress.progress((len(rows)) / len(df))

        results = pd.DataFrame(rows)

        cols = st.columns(4 if check_answers else 3)
        cols[0].metric(f"Hit@{top_k}", f"{results['hit@k'].mean():.0%}")
        cols[1].metric(f"Precision@{top_k}", f"{results['precision@k'].mean():.2f}")
        cols[2].metric("MRR", f"{results['reciprocal_rank'].mean():.2f}")
        if check_answers:
            cols[3].metric(
                "Answer accuracy", f"{results['answer_correct'].mean():.0%}"
            )

        st.dataframe(results, use_container_width=True)

        st.download_button(
            "⬇️ Download results (CSV)",
            results.to_csv(index=False),
            file_name="intellichat_evaluation.csv",
            mime="text/csv",
        )
