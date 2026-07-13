import os 
from dotenv import load_dotenv
from openai import OpenAI
from src.retriever import Retriever
from src.loader import PDFLoader
from src.chunking import *
from Generation.Prompt import build_messages
from src.vectordb import *

# using dotenv to load environment variables from .env file
env_path = os.path.join(os.path.dirname(__file__), "..",'.env')
load_dotenv(dotenv_path=env_path)

# using OpenAI's API client to interact with the Groq API for generating answers based on the retrieved context
client = OpenAI(api_key=os.getenv("GROQ_API"),
                base_url="https://api.groq.com/openai/v1")


# generating an answer to the user's query by retrieving relevant context from the document chunks and using the AI assistant to provide a response
def generate_answer(query: str, retriever: Retriever, top_k: int = 6) -> dict:
    results = retriever.retrieve(query, top_k=top_k)
    context = results["text"].tolist()
    messages = build_messages(query, context)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.2,
        max_tokens=350

    )
    # returning the query, context, and the AI assistant's answer as a dictionary
    return {
        "query": query,
        "context": context,
        "answer": response.choices[0].message.content.strip()
    }
      
# main function to demonstrate the retrieval and answer generation process
if __name__ == "__main__":
    
    retriever = Retriever()
    
    query = "What is prompt engineering?"
    answer_data = generate_answer(query, retriever, top_k=6)
    
    print("Context:", answer_data["context"])
    print("="*50) 
    print("Query:", answer_data["query"])
    print("="*50)
    print("Answer:", answer_data["answer"])
    print("="*50)
    print("*" * 50)