from loader import PDFLoader, loader
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
            length_function=len  # function to calculate the length of each chunk
             
    )
    
    all_chunks = []
    
    for page in documents:
        chunks = text_splitter.split_text(page['text'])
        all_chunks.extend(chunks)

    return all_chunks

print(chunk_document(file_path))
