import os 
from dotenv import load_dotenv
from openai import OpenAI
from retriver import Retriever
from loader import PDFLoader
from chunking import *


# using dotenv to load environment variables from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API"))

def build_prompt(query: str, context: list[str]) -> str:
    context_block = "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(context))
    prompt = f"""
    You are an AI assistant that helps users find information in documents. 
    Use the following context to answer the question. 
    If the answer is not contained within the context, say "I don't know".

    Context: {context_block}

    Question: {query}
    Answer:
    """
    return prompt  

def generate_answer(query: str, retriever: Retriever, top_k: int = 6) -> dict:
    results = retriever.retrieve(query, top_k=top_k)
    context = results["text"].tolist()
    prompt = build_prompt(query, context)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )
    return {
        "query": query,
        "context": context,
        "answer": response.choices[0].message.content.strip()
    }
    
if __name__ == "__main__":
    # File Path to your PDF file
    file_path = r"C:\Ask My Paper\Data\Copy of AI Engineering-25-125.pdf"
    
    loader = PDFLoader(file_path)
    document = loader.load_pdf()
    chunks = chunk_document(file_path)
    
    retriever = Retriever(chunks)
    
    query = "What is prompt engineering?"
    answer_data = generate_answer(query, retriever, top_k=6)
    
    print("Query:", answer_data["query"])
    print("Context:", answer_data["context"])
    print("Answer:", answer_data["answer"])