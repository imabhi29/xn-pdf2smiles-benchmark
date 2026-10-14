"""Dataset loading.

Two inputs are supported:

* **Image benchmark** (BioVista OCSR task): structure images plus a label file
  mapping each image to its ground-truth SMILES. The label file format is
  detected automatically (CSV, TSV, JSON, JSONL or two-column TXT), and column
  names can be fixed in the config if auto-detection picks the wrong ones.
* **PDF benchmark**: a folder of PDFs plus an optional ground-truth CSV with
  one row per expected molecule (columns ``pdf`` and ``smiles``).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .config import resolve_path

log = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
LABEL_EXTS = {".csv", ".tsv", ".json", ".jsonl", ".txt"}
SMILES_CANDIDATES = ["smiles", "gt_smiles", "label_smiles", "canonical_smiles",
                     "target", "label", "smi"]
IMAGE_CANDIDATES = ["image", "image_path", "img_path", "file_path", "file_name",
                    "filename", "file", "img", "path", "image_id", "id", "name"]


# --------------------------------------------------------------------------- #
# Label-file parsing
# --------------------------------------------------------------------------- #
def read_label_file(path: Path) -> pd.DataFrame:
    """Read a label file of any supported format into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            values = list(data.values())
            if values and all(isinstance(v, str) for v in values):
                # {"img_001.png": "CCO", ...}
                return pd.DataFrame({"image": list(data.keys()), "smiles": values})
            if values and all(isinstance(v, dict) for v in values):
                # {"img_001.png": {"smiles": "CCO", ...}, ...}
                df = pd.DataFrame.from_dict(data, orient="index")
                return df.rename_axis("image").reset_index()
        return pd.DataFrame(data)
    if suffix == ".txt":
        # "image_name SMILES" per line
        rows = [ln.split(maxsplit=1) for ln in path.read_text(encoding="utf-8").splitlines()
                if ln.strip()]
        return pd.DataFrame([r for r in rows if len(r) == 2], columns=["image", "smiles"])
    raise ValueError(f"Unsupported label file type: {path}")


def _pick_column(df: pd.DataFrame, wanted: str | None, candidates: list[str]) -> str | None:
    if wanted:
        return wanted if wanted in df.columns else None
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


# --------------------------------------------------------------------------- #
# Image benchmark
# --------------------------------------------------------------------------- #
def load_image_dataset(cfg, seed: int, limit: int | None = None) -> pd.DataFrame:
    """Return a DataFrame with columns sample_id, image_path, true_smiles."""
    root = resolve_path(cfg.root)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found. Run: python scripts/download_data.py --what biovista")

    # Index every image once so labels can be matched by path, name or stem.
    images = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    by_name = {p.name: p for p in images}
    by_stem = {p.stem: p for p in images}
    if not images:
        raise FileNotFoundError(f"No images found under {root}")

    candidates = ([resolve_path(cfg.labels_file)] if cfg.labels_file else
                  sorted(p for p in root.rglob("*") if p.suffix.lower() in LABEL_EXTS))
    tried = []
    for label_path in candidates:
        try:
            df = read_label_file(label_path)
        except Exception as exc:  # unreadable or wrong format: try the next file
            tried.append(f"{label_path.name} (unreadable: {exc})")
            continue
        img_col = _pick_column(df, cfg.image_col, IMAGE_CANDIDATES)
        smi_col = _pick_column(df, cfg.smiles_col, SMILES_CANDIDATES)
        if not img_col or not smi_col:
            tried.append(f"{label_path.name} (columns: {list(df.columns)[:8]})")
            continue

        def locate(value) -> Path | None:
            value = str(value)
            for base in (label_path.parent, root):
                if (base / value).is_file():
                    return base / value
            return by_name.get(Path(value).name) or by_stem.get(Path(value).stem)

        out = pd.DataFrame({
            "sample_id": df[img_col].astype(str),
            "image_path": df[img_col].map(locate),
            "true_smiles": df[smi_col].astype(str).str.strip(),
        })
        missing = out["image_path"].isna().sum()
        out = out.dropna(subset=["image_path"])
        out = out[out["true_smiles"].ne("") & out["true_smiles"].ne("nan")]
        if out.empty:
            tried.append(f"{label_path.name} (no rows matched an image)")
            continue
        if missing:
            log.warning("%d labels had no matching image and were skipped", missing)
        log.info("Labels: %s  (image col '%s', SMILES col '%s', %d usable rows)",
                 label_path.name, img_col, smi_col, len(out))

        n = limit or cfg.sample_size
        if n and n < len(out):
            out = out.sample(n=n, random_state=seed)
        out["image_path"] = out["image_path"].astype(str)
        return out.reset_index(drop=True)

    raise ValueError(
        "Could not find a usable label file. Set datasets.images.labels_file, "
        "image_col and smiles_col in the config. Files tried:\n  " + "\n  ".join(tried))


# --------------------------------------------------------------------------- #
# PDF benchmark
# --------------------------------------------------------------------------- #
def list_pdfs(cfg, limit: int | None = None) -> list[Path]:
    root = resolve_path(cfg.root)
    pdfs = sorted(root.rglob("*.pdf")) if root.is_dir() else []
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {root}. Run: python scripts/download_data.py --what pdfs")
    return pdfs[:limit] if limit else pdfs


def load_pdf_truth(cfg) -> pd.DataFrame | None:
    """Ground truth for PDFs: columns pdf (file name) and smiles, optional name."""
    if not cfg.truth_csv:
        return None
    path = resolve_path(cfg.truth_csv)
    if not path.exists():
        log.warning("No ground truth at %s; PDF metrics will be skipped.", path)
        return None
    truth = pd.read_csv(path)
    missing = {"pdf", "smiles"} - set(truth.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    truth["pdf"] = truth["pdf"].map(lambda v: Path(str(v)).name)
    return truth
