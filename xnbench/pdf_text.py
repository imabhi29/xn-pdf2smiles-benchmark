"""PDF handling with PyMuPDF: page rendering and text-layer label lookup."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF
import numpy as np

POINTS_PER_INCH = 72.0


@dataclass
class TextBlock:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


@dataclass
class RenderedPage:
    number: int                 # 1-based page number
    image: np.ndarray           # RGB array, shape (H, W, 3)
    blocks: list[TextBlock]     # text layer, in PDF points
    dpi: int


def iter_pages(pdf_path: Path, dpi: int) -> Iterator[RenderedPage]:
    """Render each page to an RGB image and read its text blocks."""
    with fitz.open(pdf_path) as doc:
        for index, page in enumerate(doc):
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8)
            img = img.reshape(pix.height, pix.width, pix.n)[..., :3].copy()
            blocks = [TextBlock(b[0], b[1], b[2], b[3], b[4])
                      for b in page.get_text("blocks") if b[4].strip()]
            yield RenderedPage(index + 1, img, blocks, dpi)


def find_label(page: RenderedPage, bbox_px, max_gap_pt: float = 40) -> str:
    """Return the short text block closest below a structure (its likely name).

    ``bbox_px`` is (y0, x0, y1, x1) in rendered-image pixels, the order used
    by DECIMER Segmentation. Returns "" when no box is available or no
    suitable caption is found. This is a heuristic: it suits layouts where a
    compound name or number sits under its drawing.
    """
    if bbox_px is None:
        return ""
    scale = POINTS_PER_INCH / page.dpi
    y0, x0, y1, x1 = (float(v) * scale for v in bbox_px)
    centre_x = (x0 + x1) / 2
    best, best_gap = "", max_gap_pt
    for block in page.blocks:
        first_line = block.text.strip().split("\n")[0].strip()
        if not first_line or len(first_line) > 40:   # body text and captions are long
            continue
        gap = block.y0 - y1
        horizontally_aligned = block.x0 - 20 <= centre_x <= block.x1 + 20
        if 0 <= gap < best_gap and horizontally_aligned:
            best, best_gap = first_line, gap
    return best
