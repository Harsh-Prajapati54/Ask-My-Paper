from langchain_pymupdf4llm import PyMuPDF4LLMLoader

class DocumentLoader:
    
    def load(self, file_path):
        loader = PyMuPDF4LLMLoader(file_path)
        docs = loader.load()
        return docs


# Actually call it
if __name__ == "__main__":
    file_path = r"C:\Ask My Paper\Data\LAB3_IOT _2026.pdf"
    
    dl = DocumentLoader()
    docs = dl.load(file_path)
    
    print(f"Loaded {len(docs)} pages")
    print(docs[0].page_content[:500])  # preview first 500 chars