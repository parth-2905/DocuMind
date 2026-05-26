import fitz
import re
import csv
import io
from pathlib import Path
from typing import Generator
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── OCR helpers ──────────────────────────────────────────────────────────────

def _ocr_page(page) -> str:
    """Render a PDF page to image and extract text via tesseract."""
    import pytesseract
    mat = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return pytesseract.image_to_string(img)

def _ocr_image_bytes(blob: bytes) -> str:
    """Run OCR on raw image bytes (for embedded images in DOCX/PPTX)."""
    import pytesseract
    img = Image.open(io.BytesIO(blob))
    return pytesseract.image_to_string(img)

# ── Format handlers ──────────────────────────────────────────────────────────

OCR_THRESHOLD = 50

def _extract_pdf(path: str) -> Generator[dict, None, None]:
    doc = fitz.open(path)
    for page_num, page in enumerate(doc):
        text = page.get_text()
        is_image_page = len(text.strip()) < OCR_THRESHOLD
        if is_image_page:
            text = _ocr_page(page)
        if text.strip():
            yield {
                "text": text,
                "page": page_num + 1,
                "ocr": is_image_page
            }

def _extract_docx(path: str) -> Generator[dict, None, None]:
    from docx import Document
    doc = Document(path)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    image_texts = []
    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            ocr_text = _ocr_image_bytes(rel.target_part.blob)
            if ocr_text.strip():
                image_texts.append(ocr_text)
    full_text = text + ("\n" + "\n".join(image_texts) if image_texts else "")
    if full_text.strip():
        yield {"text": full_text, "page": 1, "ocr": bool(image_texts)}

def _extract_txt(path: str) -> Generator[dict, None, None]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if text.strip():
        yield {"text": text, "page": 1, "ocr": False}

def _extract_pptx(path: str) -> Generator[dict, None, None]:
    from pptx import Presentation
    prs = Presentation(path)
    for slide_num, slide in enumerate(prs.slides):
        texts = [shape.text for shape in slide.shapes
                 if hasattr(shape, "text") and shape.text.strip()]
        for shape in slide.shapes:
            if shape.shape_type == 13:
                ocr_text = _ocr_image_bytes(shape.image.blob)
                if ocr_text.strip():
                    texts.append(ocr_text)
        text = "\n".join(texts)
        if text.strip():
            yield {"text": text, "page": slide_num + 1, "ocr": False}

def _extract_csv(path: str) -> Generator[dict, None, None]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        rows = [", ".join(f"{k}: {v}" for k, v in row.items()) for row in reader]
    text = "\n".join(rows)
    if text.strip():
        yield {"text": text, "page": 1, "ocr": False}

# ── Router ───────────────────────────────────────────────────────────────────

_HANDLERS = {
    ".pdf":  _extract_pdf,
    ".docx": _extract_docx,
    ".txt":  _extract_txt,
    ".pptx": _extract_pptx,
    ".csv":  _extract_csv,
}

def extract_pages(file_path: str) -> Generator[dict, None, None]:
    ext = Path(file_path).suffix.lower()
    handler = _HANDLERS.get(ext)
    if handler is None:
        raise ValueError(
            f"Unsupported file type: '{ext}'. Supported: {list(_HANDLERS)}"
        )
    yield from handler(file_path)

# ── Text cleaning & chunking ─────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", "", text)
    return text.strip()

def chunk_document(file_path: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = []
    for page_data in extract_pages(file_path):
        cleaned = clean_text(page_data["text"])
        page_chunks = splitter.split_text(cleaned)
        for i, chunk in enumerate(page_chunks):
            chunks.append({
                "text": chunk,
                "page": page_data["page"],
                "chunk_id": f"page{page_data['page']}_chunk{i}",
                "ocr": page_data.get("ocr", False)
            })
    return chunks
