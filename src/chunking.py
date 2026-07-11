from .loader import *
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
    chunks = chunk_document(file_path)
    print(f"Total chunks: {len(chunks)}") # Total no of chunks generated
    
    CHECK = "chunk"  # options: "total", "chunk", "lengths, "all"
    
   
    if CHECK == "total":  # prints the total number of chunks generated
        print(f"Total chunks: {len(chunks)}")
        
    elif CHECK == "chunk":
        idx = 397  # index of the chunk you want to load
        print(f"Chunk {idx}: {chunks[idx]}")
        
    elif CHECK == "lengths":
        for c in chunks:  # length of each chunk
            print(f"Chunk length: {len(c)}")
            
    elif CHECK == "all":  # LOAD ALL CHUNKS
        for idx, c in enumerate(chunks):
            print(f"Chunk {idx}: {c}")
   
        
    print(f"data type of chunks: {type(chunks)}") # prints the type of the chunks generated from the document