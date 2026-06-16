import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import (
    GoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_community.vectorstores import FAISS


def main():
    load_dotenv()

    index_path = Path(__file__).parent / "faiss_index"

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2"
    )

    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    question = input("Enter your question: ")

    docs = retriever.invoke(question)

    context = "\n\n".join([d.page_content for d in docs])

    llm = GoogleGenerativeAI(
        model="gemini-2.5-flash"
    )

    prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    print("\nAnswer:")
    print(response)


if __name__ == "__main__":
    while True:
        main()
    