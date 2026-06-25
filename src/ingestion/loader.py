import fitz

class PDFLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        doc = fitz.open(self.file_path)
        
        # extract document metadeta
        metadata = doc.metadata
        pages = []
        
        for page_number,page in enumerate(doc,start=1):
            pages.append({
            "pages":page_number,
            "text": page.get_text(),
        
            # metadata extraction
            "source": self.file_path,
            "title": metadata.get("title"),
            "author": metadata.get("author"),
            "subject": metadata.get("subject"),
            "keywords": metadata.get("keywords"),
        })
        
        doc.close()
        
        return pages
    

# File Path to your PDF file
file_path = "path/to/your/file.pdf"

loader = PDFLoader(file_path)
document = loader.load()

#
for doc in document:
    print(f"Page {doc['pages']}:")
    print(doc['text'])
    print(f"Source: {doc['source']}")
    print(f"Title: {doc['title']}")
    print(f"Author: {doc['author']}")
    print(f"Subject: {doc['subject']}")
    print(f"Keywords: {doc['keywords']}")
    print("\n---\n")  