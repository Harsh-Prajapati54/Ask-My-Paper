"""
document_loader.py
------------------
A robust PDF document loader that handles:
  - Regular text (multi-column, scanned, embedded)
  - Images  (raster + vector via page rasterization)
  - Complex equations (detected + rasterized as image blocks)
  - Tables  (extracted via pdfplumber with markdown output)

Output: a list of PageContent objects — one per page — each carrying
text, tables, image paths, and equation image paths.

Usage:
    from document_loader import load_pdf
    pages = load_pdf("paper.pdf", output_dir="output")
    for page in pages:
        print(page)          # readable summary
        print(page.to_dict()) # serialisable dict
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import fitz          # PyMuPDF  — text, images, rasterisation
import pdfplumber    # tables


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class TableData:
    """Represents a single extracted table."""
    page: int
    index: int          # table index on the page (0-based)
    headers: list[str]
    rows: list[list[str]]
    markdown: str = ""

    def __post_init__(self):
        if not self.markdown:
            self.markdown = self._to_markdown()

    def _to_markdown(self) -> str:
        if not self.headers and not self.rows:
            return ""
        col_count = len(self.headers) if self.headers else (len(self.rows[0]) if self.rows else 0)
        if col_count == 0:
            return ""

        def fmt_row(cells):
            cleaned = [(str(c).replace("|", "\\|").replace("\n", " ") if c is not None else "") for c in cells]
            # Pad if needed
            while len(cleaned) < col_count:
                cleaned.append("")
            return "| " + " | ".join(cleaned[:col_count]) + " |"

        lines = []
        if self.headers:
            lines.append(fmt_row(self.headers))
            lines.append("| " + " | ".join(["---"] * col_count) + " |")
        else:
            lines.append("| " + " | ".join(["---"] * col_count) + " |")

        for row in self.rows:
            lines.append(fmt_row(row))

        return "\n".join(lines)


@dataclass
class PageContent:
    """All extracted content for one PDF page."""
    page_number: int          # 1-based
    text: str = ""            # extracted plain text (may be OCR'd)
    tables: list[TableData] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)   # saved image files
    equation_paths: list[str] = field(default_factory=list) # cropped equation images
    is_scanned: bool = False
    page_image_path: Optional[str] = None  # full page raster (for scanned PDFs)

    def __str__(self):
        lines = [f"── Page {self.page_number} {'(scanned)' if self.is_scanned else ''} ──"]
        if self.text.strip():
            preview = self.text.strip()[:300].replace("\n", " ")
            lines.append(f"  Text ({len(self.text)} chars): {preview}{'…' if len(self.text)>300 else ''}")
        if self.tables:
            lines.append(f"  Tables: {len(self.tables)}")
            for t in self.tables:
                lines.append(f"    Table {t.index}: {len(t.headers)} cols × {len(t.rows)} rows")
        if self.image_paths:
            lines.append(f"  Images: {len(self.image_paths)} → {self.image_paths}")
        if self.equation_paths:
            lines.append(f"  Equations: {len(self.equation_paths)} → {self.equation_paths}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tables"] = [asdict(t) for t in self.tables]
        return d


# ─────────────────────────────────────────────
# Equation detection heuristics
# ─────────────────────────────────────────────

# Patterns that suggest mathematical / chemical equations in a text span
_EQ_PATTERNS = [
    re.compile(r"[∑∏∫∂∇√∞≈≠≤≥±×÷→←⇒⇔∈∉⊂⊃∪∩∀∃]"),   # unicode math symbols
    re.compile(r"\b(lim|sup|inf|max|min|argmax|argmin)\b.*[({]"),
    re.compile(r"[a-zA-Z]\s*[=<>]\s*[a-zA-Z0-9()\[\]{}\^_+\-*/]+"),  # simple assignment / relation
    re.compile(r"[\^_]\{"),          # LaTeX-style superscript/subscript
    re.compile(r"\\frac|\\sum|\\int|\\prod|\\sqrt|\\alpha|\\beta|\\gamma|\\delta|\\lambda|\\sigma|\\theta|\\mu|\\pi"),  # LaTeX macros
]

def _looks_like_equation(text: str) -> bool:
    """Return True if the text span looks like a mathematical expression."""
    if len(text.strip()) < 3:
        return False
    for pat in _EQ_PATTERNS:
        if pat.search(text):
            return True
    # Density of non-alphanumeric symbols
    sym = sum(1 for c in text if not c.isalnum() and c not in " \t\n.,;:!?\"'()-")
    return sym / max(len(text), 1) > 0.25


# ─────────────────────────────────────────────
# Core loader
# ─────────────────────────────────────────────

def load_pdf(
    pdf_path: str,
    output_dir: str = "output",
    dpi: int = 150,
    extract_images: bool = True,
    extract_equations: bool = True,
    extract_tables: bool = True,
    ocr_fallback: bool = True,
    pages: Optional[list[int]] = None,   # 1-based page numbers; None = all
    save_page_images: bool = False,      # save full-page rasters too
) -> list[PageContent]:
    """
    Load a PDF and return a list of PageContent objects.

    Parameters
    ----------
    pdf_path        : path to the PDF file
    output_dir      : directory where extracted images/equations are saved
    dpi             : resolution for rasterisation (150 is a good default)
    extract_images  : save embedded + vector images
    extract_equations: crop & save equation regions as images
    extract_tables  : extract tables via pdfplumber
    ocr_fallback    : if a page has no text layer, rasterise & store page image
    pages           : restrict to these 1-based page numbers (None = all)
    save_page_images: also save full-page rasters for every page
    """
    pdf_path = str(pdf_path)
    out = Path(output_dir)
    img_dir = out / "images"
    eq_dir  = out / "equations"
    page_dir = out / "pages"

    for d in (img_dir, eq_dir, page_dir):
        d.mkdir(parents=True, exist_ok=True)

    doc_fitz = fitz.open(pdf_path)
    total_pages = len(doc_fitz)
    target_pages = set(pages) if pages else set(range(1, total_pages + 1))

    results: list[PageContent] = []

    # ── pdfplumber for tables (open once) ──
    plumber_pdf = pdfplumber.open(pdf_path) if extract_tables else None

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        if page_num not in target_pages:
            continue

        fitz_page = doc_fitz[page_idx]
        pc = PageContent(page_number=page_num)

        # ── 1. Text extraction ──
        raw_text = fitz_page.get_text("text")

        # Detect scanned page (no text layer)
        if not raw_text.strip() and ocr_fallback:
            pc.is_scanned = True
            # Rasterise for visual context / downstream OCR
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = fitz_page.get_pixmap(matrix=mat, alpha=False)
            page_img_path = str(page_dir / f"page_{page_num:04d}.png")
            pix.save(page_img_path)
            pc.page_image_path = page_img_path
            # Basic OCR attempt via pytesseract if available
            try:
                from PIL import Image as PILImage
                import pytesseract
                pil_img = PILImage.open(page_img_path)
                raw_text = pytesseract.image_to_string(pil_img)
            except Exception:
                raw_text = f"[Scanned page — see {page_img_path} for visual content]"

        pc.text = raw_text

        # ── 2. Full-page raster (optional) ──
        if save_page_images and not pc.is_scanned:
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = fitz_page.get_pixmap(matrix=mat, alpha=False)
            page_img_path = str(page_dir / f"page_{page_num:04d}.png")
            pix.save(page_img_path)
            pc.page_image_path = page_img_path

        # ── 3. Embedded images ──
        if extract_images:
            img_list = fitz_page.get_images(full=True)
            for img_idx, img_info in enumerate(img_list):
                xref = img_info[0]
                try:
                    base_img = doc_fitz.extract_image(xref)
                    img_bytes = base_img["image"]
                    ext = base_img["ext"]
                    img_path = str(img_dir / f"page{page_num:04d}_img{img_idx:02d}.{ext}")
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)
                    pc.image_paths.append(img_path)
                except Exception:
                    pass

            # Vector images: rasterise page and crop bounding boxes of drawings
            # (matplotlib / Excel charts are vector — pdfimages misses them)
            drawings = fitz_page.get_drawings()
            if drawings:
                # Merge drawing rects into clusters
                clusters = _cluster_rects([d["rect"] for d in drawings], gap=5)
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                for ci, cluster_rect in enumerate(clusters):
                    if cluster_rect.width < 20 or cluster_rect.height < 20:
                        continue
                    clip = cluster_rect + fitz.Rect(-2, -2, 2, 2)
                    pix = fitz_page.get_pixmap(matrix=mat, clip=clip, alpha=False)
                    vimg_path = str(img_dir / f"page{page_num:04d}_vector{ci:02d}.png")
                    pix.save(vimg_path)
                    pc.image_paths.append(vimg_path)

        # ── 4. Equation detection & crop ──
        if extract_equations:
            # Strategy A: scan text spans for equation patterns, crop their bbox
            blocks = fitz_page.get_text("dict")["blocks"]
            eq_count = 0
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            for block in blocks:
                if block["type"] != 0:   # 0 = text block
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        span_text = span.get("text", "")
                        if _looks_like_equation(span_text):
                            bbox = fitz.Rect(span["bbox"])
                            # Pad generously
                            padded = bbox + fitz.Rect(-6, -6, 6, 6)
                            padded &= fitz_page.rect  # clamp to page
                            pix = fitz_page.get_pixmap(matrix=mat, clip=padded, alpha=False)
                            eq_path = str(eq_dir / f"page{page_num:04d}_eq{eq_count:02d}.png")
                            pix.save(eq_path)
                            pc.equation_paths.append(eq_path)
                            eq_count += 1

        # ── 5. Tables ──
        if extract_tables and plumber_pdf:
            plumber_page = plumber_pdf.pages[page_idx]
            raw_tables = plumber_page.extract_tables()
            for ti, raw_table in enumerate(raw_tables):
                if not raw_table:
                    continue
                # Treat first row as header if it has no None cells
                if raw_table[0] and all(c is not None for c in raw_table[0]):
                    headers = [str(c) for c in raw_table[0]]
                    rows = [[str(c) if c is not None else "" for c in r] for r in raw_table[1:]]
                else:
                    headers = []
                    rows = [[str(c) if c is not None else "" for c in r] for r in raw_table]
                td = TableData(page=page_num, index=ti, headers=headers, rows=rows)
                pc.tables.append(td)

        results.append(pc)

    if plumber_pdf:
        plumber_pdf.close()
    doc_fitz.close()
    return results


# ─────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────

def _cluster_rects(rects: list, gap: float = 5) -> list:
    """Merge nearby fitz.Rect objects into bounding boxes."""
    if not rects:
        return []
    clusters = []
    used = [False] * len(rects)
    for i, r in enumerate(rects):
        if used[i]:
            continue
        merged = fitz.Rect(r)
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j, s in enumerate(rects):
                if used[j]:
                    continue
                expanded = merged + fitz.Rect(-gap, -gap, gap, gap)
                if expanded.intersects(fitz.Rect(s)):
                    merged = merged | fitz.Rect(s)
                    used[j] = True
                    changed = True
        clusters.append(merged)
    return clusters


def save_results_json(pages: list[PageContent], path: str = "output/results.json"):
    """Serialise all extracted content to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([p.to_dict() for p in pages], f, indent=2, ensure_ascii=False)
    print(f"Results saved → {path}")


def encode_image_base64(image_path: str) -> str:
    """Return base64 string for an image file (for API embedding)."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    
    pages = load_pdf("Data\llm.pdf", output_dir="output")
    for page in pages:
        print(page.text)           # full extracted text
    for table in page.tables:
        print(table.markdown)  # markdown table string
    for img in page.image_paths:
        print(img)             # path to saved image
    for eq in page.equation_paths:
        print(eq) 
    import argparse

    parser = argparse.ArgumentParser(description="PDF Document Loader — text, tables, images, equations")
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("--output", "-o", default="output", help="Output directory (default: output/)")
    parser.add_argument("--dpi", type=int, default=150, help="Rasterisation DPI (default: 150)")
    parser.add_argument("--pages", nargs="*", type=int, help="Page numbers to process (default: all)")
    parser.add_argument("--no-images", action="store_true", help="Skip image extraction")
    parser.add_argument("--no-equations", action="store_true", help="Skip equation detection")
    parser.add_argument("--no-tables", action="store_true", help="Skip table extraction")
    parser.add_argument("--save-page-images", action="store_true", help="Save full-page rasters")
    parser.add_argument("--json", action="store_true", help="Save results to output/results.json")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress per-page output")
    args = parser.parse_args()

    print(f"\n📄 Loading: {args.pdf}")
    pages = load_pdf(
        pdf_path=args.pdf,
        output_dir=args.output,
        dpi=args.dpi,
        extract_images=not args.no_images,
        extract_equations=not args.no_equations,
        extract_tables=not args.no_tables,
        pages=args.pages,
        save_page_images=args.save_page_images,
    )

    if not args.quiet:
        for p in pages:
            print(p)
            if p.tables:
                for t in p.tables:
                    print(f"\n  [Table {t.index} — Page {t.page}]\n{t.markdown}\n")
            print()

    if args.json:
        save_results_json(pages, path=f"{args.output}/results.json")

    # Summary
    total_text  = sum(len(p.text) for p in pages)
    total_tables = sum(len(p.tables) for p in pages)
    total_images = sum(len(p.image_paths) for p in pages)
    total_eqs   = sum(len(p.equation_paths) for p in pages)
    scanned     = sum(1 for p in pages if p.is_scanned)

    print("─" * 50)
    print(f"✅ Done  |  Pages: {len(pages)}  |  Scanned: {scanned}")
    print(f"   Text chars : {total_text:,}")
    print(f"   Tables     : {total_tables}")
    print(f"   Images     : {total_images}")
    print(f"   Equations  : {total_eqs}")
    print(f"   Output dir : {args.output}/")
    print("─" * 50)