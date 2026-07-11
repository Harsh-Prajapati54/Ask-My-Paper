import os 
from dotenv import load_dotenv
from openai import OpenAI
from src.retriever import Retriever
from loader import PDFLoader
from chunking import *

# using dotenv to load environment variables from .env file
env_path = os.path.join(os.path.dirname(__file__), "..",'.env')
load_dotenv(dotenv_path=env_path)

# using OpenAI's API client to interact with the Groq API for generating answers based on the retrieved context
client = OpenAI(api_key=os.getenv("GROQ_API"),
                base_url="https://api.groq.com/openai/v1")



# building a prompt for the AI assistant to answer the user's query based on the retrieved context from the document chunks
def build_prompt(query: str, context: list[str]) -> str:
    context_block = "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(context))
    # creating a prompt that instructs the AI assistant to use the provided context to answer the user's question, and to respond with "I don't know" if the answer is not contained within the context
    prompt = f"""
    You are an AI assistant that helps users find information in documents. 
    Use the following context to answer the question. 
    If the answer is not contained within the context, say "I don't know".

    Context: {context_block}

    Question: {query}
    Answer:
    """
    return prompt  
# generating an answer to the user's query by retrieving relevant context from the document chunks and using the AI assistant to provide a response
def generate_answer(query: str, retriever: Retriever, top_k: int = 6) -> dict:
    results = retriever.retrieve(query, top_k=top_k)
    context = results["text"].tolist()
    prompt = build_prompt(query, context)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,

    )
    # returning the query, context, and the AI assistant's answer as a dictionary
    return {
        "query": query,
        "context": context,
        "answer": response.choices[0].message.content.strip()
    }
    
    
# main function to demonstrate the retrieval and answer generation process
if __name__ == "__main__":
    
    from time import time
    start_time = time.time()

    retriever = Retriever()
    
    query = "What is prompt engineering?"
    answer_data = generate_answer(query, retriever, top_k=6)
    
    print("Context:", answer_data["context"])
    print("="*50) 
    print("Query:", answer_data["query"])
    print("="*50)
    print("Answer:", answer_data["answer"])
    print("="*50)
    
    print(f"total time: {time.time() - start_time:.2f} seconds")