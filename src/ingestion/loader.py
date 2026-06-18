from langchain_pymupdf4llm import PyMuPDF4LLMLoader
import pprint
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

class DocumentLoader:
    def load(self, file_path):
        loader = PyMuPDF4LLMLoader(file_path,
                                   mode = "page",
                                   extract_images=True,
                                   image_parser = LLMImageBlobParser(api_key=api_key, model="gpt-4o-mini", max_image_size=1024)
                                   )
        docs = loader.load()
        return docs

if __name__ == "__main__":
    file_path = r"C:\Ask My Paper\Data\Copy of AI Engineering-25-125.pdf"
    
    dl = DocumentLoader()
    docs = dl.load(file_path)
    
    print(f"Loaded {len(docs)} pages")
    pprint.pp(docs[2].metadata)