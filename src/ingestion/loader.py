# document loader for multimodel file formats, such as pdf, docx, pptx, etc.

import os
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

doc = pymupdf.Document(r"CData\LAB3_IOT _2026.pdf")
    
# cretating a text output
out = open("extracted_text.txt","wb")
# iterating over pages
for page in doc:
  tp = page.get_textpage_ocr()
  text = page.get_text(textpage = tp).encode("utf8") # get plain text (is in UTF-8)
  out.write(text) # write text of pages
  out.write(f"\n--- Page {page.number + 1} ---\n".encode("utf8"))

out.close()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 512, chunk_overlap = 100)
chunks = text_splitter.split_text(doc)


print(f"Total chunks: {len(chunks)}")
print(chunks[0])

print(type(doc))
print(len(text))