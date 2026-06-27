import ollama
from chunking import *
from loader import *


""" 
    IN this Module we will be embedding the document chunks into a vector database for semantic search and retrieval.
    I am using Ollma's nomic-embed-text-embedding-3-large model for embedding the document chunks into a vector database.
    The embedding model is a large language model that is trained to generate embeddings for text data.
    
    Embedding Dimension: 768
    parameters: 137M
    context window: 2048 tokens
"""
embedding = []

input_text = chunk_document(file_path)
for chunk in input_text:
    response = ollama.embed(
        model='nomic-embed-text',
        input=chunk,
    )
    embedding.append(response.embeddings[0])
    

if __name__ == "__main__":
    print(f"Embedding vector for first chunk: {embedding[0]}") # prints the embedding vector for the first chunk    
    print(f"Total number of chunks embedded: {len(embedding)}")
    print(type(embedding))
    print(type(embedding[0]))
    print(type(embedding[0][0]))
    print("Embedding dimension:" ,len(embedding[0]))# prints the dimension of the embedding vector
    