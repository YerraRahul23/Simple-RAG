from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import (
    GoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_community.vectorstores import FAISS

load_dotenv()

# Load index
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2"
)

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = GoogleGenerativeAI(
    model="gemini-flash-lite-latest"
)

# Get chunks directly from FAISS
all_docs = list(vectorstore.docstore._dict.values())

# Use first N chunks for evaluation
evaluation_docs = all_docs[:10]

total_score = 0

for idx, doc in enumerate(evaluation_docs, start=1):

    source_text = doc.page_content[:1500]

    # Step 1: Generate a question from the chunk
    question_prompt = f"""
    Create ONE factual question whose answer
    can be found in this text.

    Text:
    {source_text}

    Return only the question.
    """

    question = llm.invoke(question_prompt).strip()

    # Step 2: Ask RAG
    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join(
        d.page_content for d in retrieved_docs
    )

    rag_prompt = f"""
    Answer only from the context.

    Context:
    {context}

    Question:
    {question}
    """

    answer = llm.invoke(rag_prompt)

    # Step 3: Judge answer against source chunk
    judge_prompt = f"""
    Source Text:
    {source_text}

    Question:
    {question}

    Answer:
    {answer}

    Score from 0 to 10.

    Return only a number.
    """

    score_text = llm.invoke(judge_prompt).strip()

    try:
        score = float(score_text)
    except:
        score = 0

    total_score += score

    print("=" * 80)
    print("Question:", question)
    print("Score:", score)

average = total_score / len(evaluation_docs)

print("\n" + "=" * 80)
print(f"Average RAG Score: {average:.2f}/10")