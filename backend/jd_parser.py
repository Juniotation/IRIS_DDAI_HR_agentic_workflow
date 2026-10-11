from __future__ import annotations

from pathlib import Path


def extract_jd(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    raw = uploaded_file.getvalue()
    if suffix in {".txt", ".md"}:
        text = raw.decode("utf-8", errors="ignore").strip()
    elif suffix == ".pdf":
        from pypdf import PdfReader
        import io
        reader = PdfReader(io.BytesIO(raw))
        pages = [(page.extract_text() or "") for page in reader.pages]
        text = "\n\n".join(pages).strip()
    else:
        raise ValueError("Supported JD formats: .txt, .md, .pdf")
    if len(text) < 100:
        raise ValueError("The uploaded JD contains too little readable text. Please use a text-based PDF or TXT file.")
    return text
