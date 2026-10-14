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
import re
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
def _norm(value) -> str:
    """Normalise a label value for matching: 3.0 -> '3', strip spaces/extensions."""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].lstrip("-").isdigit():
        text = text[:-2]
    if text.lower().endswith(".pdf"):
        text = text[:-4]
    return text


def _build_image_index(root: Path, images: list[Path]) -> dict[str, Path]:
    """Map every way an image might be referred to (name, stem, relative path
    with or without extension) to its path."""
    index: dict[str, Path] = {}
    ambiguous: set[str] = set()
    for img in images:
        rel = img.relative_to(root).as_posix()
        rel_no_ext = rel[: -len(img.suffix)]
        for key in {img.name, img.stem, rel, rel_no_ext,
                    # also match the last two path parts, e.g. "6r8r/3"
                    "/".join(Path(rel_no_ext).parts[-2:])}:
            if key in index and index[key] != img:
                ambiguous.add(key)      # e.g. "0.png" exists in several folders
            index.setdefault(key, img)
    # Never match on a key that points to more than one image
    for key in ambiguous:
        del index[key]
    return index


_PAGE_BOX = re.compile(r"^[A-Za-z]*(\d+)-(\d+)$")


def _template_fields(row) -> dict[str, str]:
    """Row values as template fields, plus derived fields for page-box IDs.

    A value like "P3-1" (page 3, box 1) in column c also provides:
      {c_nop}   -> "3-1"   (prefix letters removed)
      {c_page}  -> "3",  {c_box}  -> "1"
      {c_page0} -> "2",  {c_box0} -> "0"   (counted from 0 instead of 1)
    """
    fields = {str(k): _norm(v) for k, v in row.items()}
    for key, value in list(fields.items()):
        m = _PAGE_BOX.match(value)
        if m:
            page, box = int(m.group(1)), int(m.group(2))
            fields.update({f"{key}_nop": f"{page}-{box}", f"{key}_page": str(page),
                           f"{key}_box": str(box), f"{key}_page0": str(page - 1),
                           f"{key}_box0": str(box - 1)})
    return fields


def _from_template(template: str):
    """Build an image key by filling a template such as
    "{pdf_name}_image_molminer_bbox_{molminer_id_nop}" from a label row."""
    def build(row):
        return template.format_map(_template_fields(row))
    return build


def _safe(build):
    """Wrap a key builder so a row it cannot handle gives None instead of crashing."""
    def wrapped(row):
        try:
            return build(row)
        except (KeyError, ValueError, TypeError, IndexError):
            return None
    return wrapped


def _key_builders(df: pd.DataFrame, smiles_col: str, wanted: str | None,
                  templates: list[str] | None = None):
    """Candidate ways to turn a label row into an image key.

    Yields (description, function(row) -> key): file-name templates from the
    config first, then single columns, then pairs of columns joined with
    common separators (e.g. pdf_name/index).
    """
    if wanted:
        yield wanted, lambda r, c=wanted: _norm(r[c])
        return
    for template in templates or []:
        yield template, _from_template(template)
    cols = [c for c in df.columns if c != smiles_col and not str(c).startswith("Unnamed")]
    preferred = [c for c in cols if str(c).lower() in IMAGE_CANDIDATES]
    for c in preferred + [c for c in cols if c not in preferred]:
        yield c, lambda r, c=c: _norm(r[c])
    for a in cols:
        for b in cols:
            if a == b:
                continue
            for sep in ("/", "_", "-", ""):
                yield f"{a}{sep}{b}", lambda r, a=a, b=b, sep=sep: f"{_norm(r[a])}{sep}{_norm(r[b])}"


def _best_key(df: pd.DataFrame, smiles_col: str, wanted: str | None,
              index: dict[str, Path], templates: list[str] | None = None, probe: int = 300):
    """Pick the key builder that matches the most images on a sample of rows."""
    sample = df.sample(n=min(probe, len(df)), random_state=0)
    best, strong = (None, None, 0.0), []
    for desc, build in _key_builders(df, smiles_col, wanted, templates):
        build = _safe(build)
        keys = [build(row) for _, row in sample.iterrows()]
        matched = [index[k] for k in keys if k is not None and k in index]
        # Each label row must point to a different image, otherwise labels would be mixed up
        if len(set(matched)) < len(matched):
            continue
        rate = len(matched) / max(len(sample), 1)
        if rate >= 0.5:
            strong.append(f"{desc} ({rate:.0%})")
        if rate > best[2]:          # ties keep the earlier (more likely) pattern
            best = (desc, build, rate)
    if len(strong) > 1:
        log.warning("Several image-name patterns match: %s. Using '%s'. "
                    "Check the label/image sanity cell before trusting results.",
                    "; ".join(strong), best[0])
    return best


def load_image_dataset(cfg, seed: int, limit: int | None = None) -> pd.DataFrame:
    """Return a DataFrame with columns sample_id, image_path, true_smiles.

    The label file and the column(s) identifying each image are detected
    automatically by testing which choice matches the most image files.
    Set labels_file / image_col / smiles_col in the config to override.
    """
    root = resolve_path(cfg.root)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found. Run: python scripts/download_data.py --what biovista")

    images = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    if not images:
        raise FileNotFoundError(f"No images found under {root}")
    index = _build_image_index(root, images)
    examples = ", ".join(sorted(p.relative_to(root).as_posix() for p in images)[:5])

    candidates = ([resolve_path(cfg.labels_file)] if cfg.labels_file else
                  sorted(p for p in root.rglob("*") if p.suffix.lower() in LABEL_EXTS))
    tried = []
    for label_path in candidates:
        try:
            df = read_label_file(label_path)
        except Exception as exc:  # unreadable or wrong format: try the next file
            tried.append(f"{label_path.name} (unreadable: {exc})")
            continue
        smi_col = _pick_column(df, cfg.smiles_col, SMILES_CANDIDATES)
        if not smi_col:
            tried.append(f"{label_path.name} (no SMILES column; columns: {list(df.columns)[:8]})")
            continue
        desc, build, rate = _best_key(df, smi_col, cfg.image_col, index,
                                      cfg.get("image_key_templates"))
        if build is None or rate < 0.5:
            tried.append(f"{label_path.name} (columns {list(df.columns)[:8]}; best image match "
                         f"'{desc}' only {rate:.0%})\n    first rows:\n"
                         + df.head(3).to_string(max_colwidth=40).replace("\n", "\n    "))
            continue

        keys = [build(row) for _, row in df.iterrows()]   # build is already _safe-wrapped
        out = pd.DataFrame({
            "sample_id": keys,
            "image_path": [index.get(k) if k is not None else None for k in keys],
            "true_smiles": df[smi_col].astype(str).str.strip(),
        })
        missing = int(out["image_path"].isna().sum())
        out = out.dropna(subset=["image_path"])
        out = out[out["true_smiles"].ne("") & out["true_smiles"].ne("nan")]
        if missing:
            log.warning("%d labels had no matching image and were skipped", missing)
        log.info("Labels: %s | image key: '%s' (%.0f%% matched) | SMILES column: '%s' | %d usable rows",
                 label_path.name, desc, rate * 100, smi_col, len(out))

        n = limit or cfg.sample_size
        if n and n < len(out):
            out = out.sample(n=n, random_state=seed)
        out["image_path"] = out["image_path"].astype(str)
        return out.reset_index(drop=True)

    raise ValueError(
        "Could not match the label file to the images. Add a matching pattern to "
        "datasets.images.image_key_templates in configs/default.yaml.\nExample image files: "
        f"{examples}\nFiles tried:\n  " + "\n  ".join(tried) +
        "\nRun `python scripts/inspect_dataset.py` and set datasets.images.labels_file, "
        "image_col and smiles_col in configs/default.yaml.")


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
