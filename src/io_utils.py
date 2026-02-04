import os
from typing import List
import pdfplumber
from docx import Document

SUPPORTED_EXTS = {".pdf", ".docx", ".txt"}

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_docx(path: str) -> str:
    doc = Document(path)
    parts = []
    for p in doc.paragraphs:
        if p.text:
            parts.append(p.text)
    return "\n".join(parts)

def read_pdf(path: str) -> str:
    """
    Extract text from PDF using pdfplumber.
    Note: Some PDFs are image-only scans; those will return minimal text
    unless you add OCR later.
    """
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                parts.append(page_text)
    return "\n".join(parts)

def load_resume_file(path: str) -> str:
    ext = os.path.splitext(path.lower())[1]
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTS}")

    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".docx":
        return read_docx(path)
    return read_txt(path)

def safe_filename(name: str) -> str:
    """
    Safer filenames when saving uploads.
    """
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("_", "-", ".", " "):
            keep.append(ch)
    return "".join(keep).strip().replace(" ", "_")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def chunk_text(text: str, max_chars: int = 4000) -> List[str]:
    """
    Split long text to reduce token/embedding issues if you expand later.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+max_chars])
        i += max_chars
    return chunks
