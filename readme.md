# 📚 Retrieval-Augmented Generation (RAG) System using Gemini, FAISS, and LangChain

## 🚀 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that allows users to ask questions about PDF documents and receive accurate, context-aware answers.

The system combines:

* Google Gemini Embeddings
* Google Gemini LLM
* LangChain
* FAISS Vector Database
* Python
* Streamlit

Instead of relying solely on an LLM's pre-trained knowledge, the application retrieves relevant information from uploaded documents and uses that context to generate grounded responses.

---

## 🎯 Problem Statement

Large Language Models (LLMs) are powerful but cannot directly access the content of custom documents such as:

* DSA Notes
* Java Notes
* Company Reports
* Research Papers
* Technical Documentation

Without document retrieval, the model may:

* Hallucinate information
* Provide inaccurate answers
* Miss document-specific knowledge

**Retrieval-Augmented Generation (RAG)** solves this problem by retrieving relevant document chunks before generating a response.

---

## 🏗️ System Architecture

```text
PDF Documents
      │
      ▼
Document Loader
      │
      ▼
Text Chunking
      │
      ▼
Gemini Embeddings
      │
      ▼
FAISS Vector Store
      │
      ▼
User Question
      │
      ▼
Similarity Search
      │
      ▼
Relevant Chunks
      │
      ▼
Gemini LLM
      │
      ▼
Final Answer
```

---

## ⚙️ Technologies Used

| Technology    | Purpose             |
| ------------- | ------------------- |
| Python        | Backend Development |
| LangChain     | RAG Pipeline        |
| Google Gemini | Embeddings + LLM    |
| FAISS         | Vector Database     |
| Streamlit     | User Interface      |
| PyPDF         | PDF Processing      |

---

# 📂 Project Structure

```text
RAG-Gemini-FAISS/
│
├── app.py
├── requirements.txt
├── .env
├── data/
│   ├── dsa_notes.pdf
│   └── java_notes.pdf
│
├── vector_store/
│   └── faiss_index
│
├── utils/
│   ├── pdf_loader.py
│   ├── embeddings.py
│   └── retrieval.py
│
└── README.md
```

---

# 🔄 Workflow

## Step 1: Load PDF Documents

PDF files are loaded using LangChain.

```python
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("notes.pdf")
documents = loader.load()
```

### Purpose

* Extract text
* Preserve metadata
* Process multiple PDFs

---

## Step 2: Text Chunking

Large documents are split into smaller chunks.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

texts = splitter.split_documents(documents)
```

### Why Chunking?

* Embedding models have token limits
* Improves retrieval accuracy
* Preserves context with overlap

---

## Step 3: Generate Embeddings

Each chunk is converted into a vector representation.

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)
```

### Purpose

Convert text into numerical vectors that capture semantic meaning.

Example:

```text
"What is an algorithm?"

and

"Define algorithm"
```

Both produce similar embeddings.

---

## Step 4: Create FAISS Vector Store

Store embeddings efficiently.

```python
from langchain.vectorstores import FAISS

vectorstore = FAISS.from_documents(
    texts,
    embeddings
)
```

Save index:

```python
vectorstore.save_local("faiss_index")
```

Load index:

```python
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
```

### Benefits

* Fast similarity search
* Local storage
* Scalable retrieval

---

## Step 5: Retrieval

When a user asks:

```text
What is an Algorithm?
```

The query is embedded and compared with stored vectors.

```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k":4}
)

docs = retriever.get_relevant_documents(question)
```

### Output

Top 4 relevant chunks.

---

## Step 6: Context Construction

Combine retrieved chunks.

```python
context = "\n\n".join(
    doc.page_content
    for doc in docs
)
```

Example:

```text
Chunk 1
Chunk 2
Chunk 3
Chunk 4
```

This context becomes the knowledge source for Gemini.

---

## Step 7: Answer Generation

Prompt Gemini with context and question.

```python
prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{question}
"""
```

Generate response:

```python
response = model.invoke(prompt)
```

---

# 💻 Streamlit User Interface

Example UI:

```python
import streamlit as st

st.title("📚 Chat with PDF")

question = st.text_input(
    "Ask a Question"
)

if question:
    answer = qa_chain.invoke(question)
    st.write(answer)
```

---

# 📝 Example

### User Question

```text
What is an Abstract Data Type?
```

### Retrieved Chunk

```text
An Abstract Data Type (ADT) defines
the behavior of a data structure
without specifying implementation.
```

### Generated Answer

```text
An Abstract Data Type (ADT) defines a
set of operations and behavior without
specifying how the data structure is
implemented internally.
```

---

# 🌟 Features

✅ PDF Question Answering

✅ Semantic Search

✅ Google Gemini Integration

✅ FAISS Vector Database

✅ Multiple PDF Support

✅ Context-Aware Responses

✅ Streamlit UI

✅ Reduced Hallucinations

---

# 📦 Installation

## Clone Repository

```bash
git clone https://github.com/YerraRahul23/Simple-RAG.git

cd Simple-RAG
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
```

---

# ▶️ Run Application

```bash
streamlit run app.py
```

Application opens at:

```text
http://localhost:8501
```

---

# 📈 Strengths

* Reduces hallucinations
* Uses document-specific knowledge
* Fast semantic retrieval
* Scalable architecture
* Supports multiple PDFs
* Easy deployment

---

# ⚠️ Current Limitations

* Retrieval may miss information for broad queries.
* No citation display for retrieved chunks.
* Performance depends on chunking strategy.
* Evaluation dataset is limited.

---

# 🔮 Future Improvements

* Source Citations
* Chat History Memory
* Hybrid Search (BM25 + Vector Search)
* Multi-modal PDF Support
* Conversation Summarization
* Reranking Models
* Deployment on Cloud

---

# 📊 Results

The system successfully:

* Ingests PDF documents
* Generates embeddings
* Stores vectors in FAISS
* Retrieves relevant chunks
* Produces context-grounded answers

This significantly reduces hallucinations compared to standalone LLM responses.

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new branch

```bash
git checkout -b feature-name
```

3. Commit changes

```bash
git commit -m "Added feature"
```

4. Push changes

```bash
git push origin feature-name
```

5. Open a Pull Request

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Yerra Rahul**

GitHub: https://github.com/YerraRahul23

LinkedIn: https://www.linkedin.com/in/yerrarahul/

---

⭐ If you found this project useful, don't forget to star the repository.
