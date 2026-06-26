from loader import *
from langchain_text_splitters import RecursiveCharacterTextSplitter

# chunking strategy 
"""
    using Lang-chain's text splitter to split the document into smaller chunks for processing.
    using recursive character text splitter to split the document into smaller chunks for processing.
    
"""
file_path = r"Data\llm.pdf"

def chunk_document(file_path):
    loader = PDFLoader(file_path)
    documents = loader.load_pdf()

    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # maximum size of each chunk
            chunk_overlap=200,  # overlap between chunks
            length_function=len,
             separators=[
                        "\n\n",
                        "\n",
                        " ",
                        ".",
                        ",",
                        "\u200b",  # Zero-width space
                        "\uff0c",  # Fullwidth comma
                        "\u3001",  # Ideographic comma
                        "\uff0e",  # Fullwidth full stop
                        "\u3002",  # Ideographic full stop
                        "",
                            ]# function to calculate the length of each chunk
             
    )
    
    all_chunks = []
    
    for page in documents:
        chunks = text_splitter.split_text(page['text'])
        all_chunks.extend(chunks)

    return all_chunks



if __name__ == "__main__":
     print(chunk_document(file_path))
     print(f"Total chunks: {len(chunk_document(file_path))}") # Total no of chunks generated from the document

    # Load specific chunk 
     print(f"Chunk 1: {chunk_document(file_path)[-1]},\n Length of chunk : {len(chunk_document(file_path)[-1])}") # Load the first chunk
    
    
    
    # prints the length of each chunk 
     """ for chunk in chunk_document(file_path):
        print(f"Chunk length: {len(chunk)}") """
        
# print(type(chunk_document(file_path))) # prints the type of the chunks generated from the document
# print(len(chunk_document(file_path))) # prints the total number of chunks generated from the document
# print(type(chunk_document(file_path)[0])) # prints the type of the first chunk generated from the document
