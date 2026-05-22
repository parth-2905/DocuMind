import fitz
import re
from typing import Generator
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_pages(pdf_path: str) -> Generator[dict, None, None]:
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            yield {"text": text, "page": page_num + 1}

def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", "", text)
    return text.strip()

def chunk_document(pdf_path: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = []
    for page_data in extract_pages(pdf_path):
        cleaned = clean_text(page_data["text"])
        page_chunks = splitter.split_text(cleaned)
        for i, chunk in enumerate(page_chunks):
            chunks.append({
                "text": chunk,
                "page": page_data["page"],
                "chunk_id": f"page{page_data['page']}_chunk{i}"
            })
    return chunks
