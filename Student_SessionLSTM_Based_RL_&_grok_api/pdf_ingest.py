"""
pdf_ingest.py — Extract, clean, and chunk both mechatronics textbooks.

Run once (or re-run if PDFs change):
    python pdf_ingest.py

Output:
    pdf_chunks.json  — list of {book, page, chunk_index, text, source} dicts

Extraction strategy (per page):
    1. pdfplumber (fast, lossless for text-layer PDFs like Rajput)
    2. If < 30 words extracted → OCR fallback using pdfplumber's built-in
       page renderer (pypdfium2, already installed with pdfplumber) + pytesseract.
       NO poppler / pdf2image required.

Dependencies:
    pip install pdfplumber pytesseract --break-system-packages
    # also needs Tesseract OCR binary:
    #   macOS:  brew install tesseract
    #   Ubuntu: sudo apt install tesseract-ocr
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PDF_FILES = [
    ("bolton", "Mechatronics by W Bolton.pdf"),
    ("rajput", "Mechatronics-by-rk-rajput.pdf"),
]
CHUNK_TOKENS   = 500   # target words per chunk
OVERLAP_TOKENS = 50    # word overlap between consecutive chunks
OCR_MIN_WORDS  = 30    # pages with fewer words from pdfplumber trigger OCR
OCR_DPI        = 200   # render DPI for OCR (200 is good balance of speed/accuracy)
OUTPUT         = Path("pdf_chunks.json")


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def _word_count(text: str) -> int:
    return len(text.split())


def _clean(text: str) -> str:
    """Remove common PDF/OCR artefacts."""
    text = re.sub(r"-\n", "", text)            # de-hyphenate across lines
    text = re.sub(r"\n+", " ", text)           # flatten newlines
    text = re.sub(r"\s{2,}", " ", text)        # collapse whitespace
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")  # ligatures
    # OCR noise: lone single characters on a line → drop
    text = re.sub(r"(?<!\w)[^a-zA-Z0-9\s.,;:!?()\-]{2,}(?!\w)", " ", text)
    return text.strip()


def chunk_text(
    text: str,
    target: int = CHUNK_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> list[str]:
    """Split text into overlapping word-based chunks of ~target words."""
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + target, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += target - overlap
    return chunks


# ---------------------------------------------------------------------------
# OCR fallback
# ---------------------------------------------------------------------------
def _ocr_page(page) -> str:
    """
    Render a pdfplumber page to an image using pdfplumber's built-in pypdfium2
    renderer, then run Tesseract OCR.  No poppler / pdf2image required.

    Parameters
    ----------
    page : pdfplumber.Page  (open, inside a `with pdfplumber.open(...)` block)

    Returns extracted text, or "" on failure.
    """
    try:
        import pytesseract
    except ImportError:
        return ""

    try:
        pil_img = page.to_image(resolution=OCR_DPI).original  # PIL.Image via pypdfium2
        return pytesseract.image_to_string(pil_img, lang="eng")
    except Exception as e:
        print(f"    OCR error on page {page.page_number}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Per-book extraction
# ---------------------------------------------------------------------------
def extract_chunks(book_key: str, pdf_path: Path) -> list[dict]:
    """Extract text chunks from a PDF, using OCR when pdfplumber fails."""
    try:
        import pdfplumber
    except ImportError:
        sys.exit("pdfplumber not installed. Run: pip install pdfplumber --break-system-packages")

    if not pdf_path.exists():
        print(f"  WARNING: {pdf_path} not found, skipping.")
        return []

    # Check if OCR libraries are available (pytesseract + pdfplumber's pypdfium2 renderer)
    ocr_available = True
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        ocr_available = False
        print(f"  NOTE: pytesseract not installed — OCR disabled for {book_key}")

    records: list[dict] = []
    chunk_index = 0
    n_pdfplumber = 0
    n_ocr = 0
    n_skipped = 0

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages, start=1):
            if page_num % 50 == 0:
                print(f"    ... page {page_num}/{total_pages}")

            raw = page.extract_text() or ""
            cleaned = _clean(raw)
            extraction_source = "pdfplumber"

            # Fallback to OCR if pdfplumber returned too little text
            if _word_count(cleaned) < OCR_MIN_WORDS and ocr_available:
                ocr_raw = _ocr_page(page)
                ocr_cleaned = _clean(ocr_raw)
                if _word_count(ocr_cleaned) >= OCR_MIN_WORDS:
                    cleaned = ocr_cleaned
                    extraction_source = "ocr"

            if _word_count(cleaned) < OCR_MIN_WORDS:
                n_skipped += 1
                continue

            if extraction_source == "pdfplumber":
                n_pdfplumber += 1
            else:
                n_ocr += 1

            for chunk in chunk_text(cleaned):
                if _word_count(chunk) < 20:
                    continue
                records.append({
                    "book":           book_key,
                    "page":           page_num,
                    "chunk_index":    chunk_index,
                    "text":           chunk,
                    "source":         extraction_source,
                })
                chunk_index += 1

    print(f"    pdfplumber pages={n_pdfplumber}  ocr pages={n_ocr}  skipped={n_skipped}")
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    base = Path(__file__).parent
    all_chunks: list[dict] = []

    for book_key, filename in PDF_FILES:
        pdf_path = base / filename
        print(f"\nProcessing {filename} ({book_key}) ...")
        chunks = extract_chunks(book_key, pdf_path)
        print(f"  -> {len(chunks)} chunks extracted")
        all_chunks.extend(chunks)

    out = base / OUTPUT
    out.write_text(json.dumps(all_chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    from collections import Counter
    book_counts = Counter(c["book"] for c in all_chunks)
    src_counts  = Counter(c["source"] for c in all_chunks)
    print(f"\nSaved {len(all_chunks)} total chunks -> {out}")
    for book, n in book_counts.items():
        print(f"  {book}: {n} chunks")
    for src, n in src_counts.items():
        print(f"  extraction method '{src}': {n} chunks")


if __name__ == "__main__":
    main()
