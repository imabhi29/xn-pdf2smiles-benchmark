"""Build the ground-truth file for the PDF benchmark automatically.

    python scripts/build_pdf_truth.py

Writes data/ground_truth/pdf_truth.csv (columns: pdf, name, smiles, source)
for every PDF in data/pdfs, from two sources:

1. **BioVista labels** - if a PDF's file name matches a paper in the
   downloaded BioVista label file (e.g. 68_6r8r.pdf), its expert-curated
   structures are used directly.
2. **SMILES files** - any file data/ground_truth/*_smiles.csv, such as a
   course answer key or a paper's Supporting Information "molecular formula
   strings" CSV. The SMILES column is found automatically. If the file has a
   ``pdf`` column it is used; otherwise the PDF is taken from the file name
   (``19_smiles.csv`` -> ``19.pdf``).
3. **Compound-name lists** - any file data/ground_truth/*_names.csv with
   columns ``pdf,name`` (e.g. drug names in a paper). Each name is resolved
   to SMILES with the PubChem web API. Names PubChem cannot find (in-house
   compound codes, for example) are reported and skipped, never guessed.

No SMILES are typed by hand, so the ground truth is reproducible.
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xnbench.config import load_config, resolve_path  # noqa: E402
from xnbench.datasets import LABEL_EXTS, read_label_file  # noqa: E402

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
# PubChem renamed its SMILES properties in 2025; try the new name, then the old ones
SMILES_PROPERTIES = ["SMILES", "IsomericSMILES", "CanonicalSMILES"]


# --------------------------------------------------------------------------- #
# Source 1: BioVista labels
# --------------------------------------------------------------------------- #
def biovista_truth(images_root: Path, pdf_stems: set[str]) -> pd.DataFrame:
    """Rows of the BioVista label file whose paper matches one of our PDFs."""
    if not images_root.exists():
        print(f"[biovista] {images_root} not found; skipping (run download_data.py first)")
        return pd.DataFrame()
    for label_path in sorted(p for p in images_root.rglob("*") if p.suffix.lower() in LABEL_EXTS):
        try:
            df = read_label_file(label_path)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}
        if "pdf_name" not in cols or "smiles" not in cols:
            continue
        paper = df[cols["pdf_name"]].astype(str).str.replace(r"\.pdf$", "", regex=True)
        hits = df[paper.isin(pdf_stems)]
        print(f"[biovista] {label_path.name}: {len(hits)} structures for "
              f"{paper[paper.isin(pdf_stems)].nunique()} of {len(pdf_stems)} PDFs")
        if hits.empty:
            return pd.DataFrame()
        name_col = cols.get("molminer_id") or cols.get("index")
        return pd.DataFrame({
            "pdf": paper[hits.index] + ".pdf",
            "name": hits[name_col].astype(str) if name_col else "",
            "smiles": hits[cols["smiles"]].astype(str),
            "source": "BioVista",
        })
    print("[biovista] no label file with pdf_name + smiles columns found")
    return pd.DataFrame()


# --------------------------------------------------------------------------- #
# Source 2: SMILES files (course answer key, Supporting Information CSVs)
# --------------------------------------------------------------------------- #
NAME_HINTS = ("name", "compound", "cmpd", "id", "label", "entry")


def smiles_file_truth(files: list[Path], pdf_names: set[str]) -> pd.DataFrame:
    """Read *_smiles.csv files; map each row to a PDF in data/pdfs."""
    parts = []
    by_stem = {Path(n).stem.lower(): n for n in pdf_names}
    for path in files:
        try:
            df = pd.read_csv(path, sep=None, engine="python")  # sniffs , ; or tab
        except Exception as exc:
            print(f"[smiles] {path.name}: unreadable ({exc}); skipped")
            continue
        lower = {str(c).lower(): c for c in df.columns}
        smi_col = next((lower[c] for c in lower if "smiles" in c), None)
        if smi_col is None:
            print(f"[smiles] {path.name}: no column containing 'smiles' "
                  f"(columns: {list(df.columns)[:8]}); skipped")
            continue
        name_col = next((lower[c] for c in lower if any(h in c for h in NAME_HINTS)
                         and lower[c] != smi_col), None)
        if "pdf" in lower:
            pdf = df[lower["pdf"]].astype(str).map(
                lambda v: by_stem.get(Path(v).stem.lower(), Path(v).name))
        else:
            stem = path.name[: -len("_smiles.csv")].lower()
            if stem not in by_stem:
                print(f"[smiles] {path.name}: no PDF named {stem}.pdf in data/pdfs; skipped")
                continue
            pdf = pd.Series(by_stem[stem], index=df.index)
        out = pd.DataFrame({
            "pdf": pdf,
            "name": df[name_col].astype(str) if name_col else "",
            "smiles": df[smi_col].astype(str).str.strip(),
            "source": f"file:{path.name}",
        })
        out = out[out["smiles"].ne("") & out["smiles"].ne("nan") & out["pdf"].isin(pdf_names)]
        print(f"[smiles] {path.name}: {len(out)} structures for {out['pdf'].nunique()} PDF(s)")
        parts.append(out)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Source 3: compound names resolved with PubChem
# --------------------------------------------------------------------------- #
def name_variants(name: str) -> list[str]:
    """'AMG925' -> also try 'AMG 925' and 'AMG-925' (common catalogue spellings)."""
    variants = [name]
    for i in range(1, len(name)):
        if name[i - 1].isalpha() and name[i].isdigit():
            variants += [f"{name[:i]} {name[i:]}", f"{name[:i]}-{name[i:]}"]
            break
    return variants


def pubchem_smiles(name: str, session: requests.Session) -> str | None:
    for variant in name_variants(name):
        quoted = urllib.parse.quote(variant, safe="")
        for prop in SMILES_PROPERTIES:
            time.sleep(0.25)  # PubChem allows at most 5 requests per second
            try:
                resp = session.get(f"{PUBCHEM}/{quoted}/property/{prop}/JSON", timeout=20)
            except requests.RequestException:
                continue
            if resp.status_code == 404:      # name unknown: no point trying other properties
                break
            if not resp.ok:                  # property name not supported: try the next one
                continue
            props = resp.json().get("PropertyTable", {}).get("Properties", [{}])[0]
            for key in SMILES_PROPERTIES:
                if props.get(key):
                    return props[key]
    return None


def names_truth(name_files: list[Path], pdf_names: set[str]) -> pd.DataFrame:
    rows, missing = [], []
    session = requests.Session()
    for path in name_files:
        names = pd.read_csv(path)
        for _, r in names.iterrows():
            pdf, name = Path(str(r["pdf"])).name, str(r["name"]).strip()
            if pdf not in pdf_names:
                continue
            smi = pubchem_smiles(name, session)
            if smi:
                rows.append({"pdf": pdf, "name": name, "smiles": smi, "source": "PubChem"})
            else:
                missing.append(f"{name} ({pdf})")
        print(f"[names] {path.name}: resolved {len(rows)} names so far")
    if missing:
        print("[names] not found in PubChem (skipped):", ", ".join(missing))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pdf-dir", help="folder of PDFs (default: datasets.pdfs.root)")
    ap.add_argument("--out", help="output CSV (default: datasets.pdfs.truth_csv)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pdf_dir = resolve_path(args.pdf_dir or cfg.datasets.pdfs.root)
    out = resolve_path(args.out or cfg.datasets.pdfs.truth_csv)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs in {pdf_dir}. Run scripts/download_data.py --what pdfs or add your own.")
    print(f"PDFs: {', '.join(p.name for p in pdfs)}")

    pdf_names = {p.name for p in pdfs}
    parts = [biovista_truth(resolve_path(cfg.datasets.images.root), {p.stem for p in pdfs}),
             smiles_file_truth(sorted(out.parent.glob("*_smiles.csv")), pdf_names),
             names_truth(sorted(out.parent.glob("*_names.csv")), pdf_names)]
    truth = pd.concat([p for p in parts if not p.empty], ignore_index=True) if any(
        not p.empty for p in parts) else pd.DataFrame(columns=["pdf", "name", "smiles", "source"])
    truth = truth.drop_duplicates(subset=["pdf", "smiles"])

    out.parent.mkdir(parents=True, exist_ok=True)
    truth.to_csv(out, index=False)
    print(f"\nWrote {len(truth)} ground-truth structures to {out}")
    if not truth.empty:
        print(truth.groupby(["pdf", "source"]).size().to_string())
    no_truth = sorted({p.name for p in pdfs} - set(truth["pdf"]))
    if no_truth:
        print("PDFs without ground truth (extraction still runs, but no precision/recall):",
              ", ".join(no_truth))


if __name__ == "__main__":
    main()
