"""Streamlit UI for RAG — upload PDFs and chat with your documents."""

import os
import tempfile
from pathlib import Path

# Fix SSL certificate issue on macOS
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chat",
    page_icon="📄",
    layout="wide",
)

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("GEMINI_API_KEY not found in .env file.")
    st.stop()


# ── Helpers ───────────────────────────────────────────────────────────────

@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        request_timeout=30,
        max_retries=3,
    )


def get_llm():
    return GoogleGenerativeAI(model="gemini-2.5-flash")


def load_vectorstore(embeddings):
    index_path = Path("faiss_index")
    if not index_path.exists():
        return None
    return FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def ingest_pdf(file_bytes: bytes, filename: str) -> str:
    """Save uploaded PDF to data/, chunk it, and rebuild the FAISS index."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    file_path = data_dir / filename
    with open(file_path, "wb") as f:
        f.write(file_bytes)

    # Load
    loader = PyPDFLoader(str(file_path))
    documents = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    texts = splitter.split_documents(documents)

    # Embed & store
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local("faiss_index")

    return f"✅ Ingested **{filename}** — {len(texts)} chunks added."


def ask_question(question: str, vectorstore) -> str:
    """Retrieve context and answer using the LLM."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])

    llm = get_llm()
    prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{question}

Answer:
"""
    return llm.invoke(prompt)


# ── Session state ─────────────────────────────────────────────────────────

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_vectorstore(get_embeddings())

if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Sidebar — PDF upload ──────────────────────────────────────────────────

with st.sidebar:
    st.header("📂 Document Management")

    uploaded_file = st.file_uploader(
        "Upload a PDF", type=["pdf"], accept_multiple_files=False
    )

    if uploaded_file and st.button("Ingest", type="primary", use_container_width=True):
        with st.spinner("Ingesting PDF…"):
            msg = ingest_pdf(uploaded_file.getvalue(), uploaded_file.name)
            st.success(msg)
            # Reload the vectorstore
            st.session_state.vectorstore = load_vectorstore(get_embeddings())
            st.session_state.messages = []  # clear chat history
            st.rerun()

    st.divider()
    st.caption(f"Index status: **{'✅ Ready' if st.session_state.vectorstore else '❌ Empty'}**")
    st.caption("PDFs in `data/`:")
    data_dir = Path("data")
    if data_dir.exists():
        for p in sorted(data_dir.glob("*.pdf")):
            st.caption(f"  • {p.name}")
    else:
        st.caption("  (no data directory)")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main — Chat ───────────────────────────────────────────────────────────

st.title("💬 Chat with your documents")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents…"):
    if not st.session_state.vectorstore:
        st.error("No FAISS index found. Please upload and ingest a PDF first.")
        st.stop()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer = ask_question(prompt, st.session_state.vectorstore)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})