"""Ingest PDFs, generate Gemini embeddings, and store them in a FAISS index.

This script follows LangChain 1.3.9 best practices:
* Explicit timeout / retry settings for the Gemini client.
* Loads all PDFs in the `data/` directory.
* Persists the FAISS index to `faiss_index`.
* Provides clear error messages when required env vars are missing.
"""

import os
from pathlib import Path

# Fix SSL certificate issue on macOS
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


def main() -> None:
    # Load env vars
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in .env")

    # Initialise Gemini embeddings client
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        request_timeout=30,
        max_retries=3,
    )
    

    # Load PDFs
    data_dir = Path("data")
    pdf_paths = list(data_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError("No PDF files found in ./data")

    documents = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load())

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    texts = splitter.split_documents(documents)

    # Build and persist FAISS index
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local("faiss_index")
    print(f"FAISS index created with {len(texts)} chunks and saved to 'faiss_index'.")


if __name__ == "__main__":
    main()
