"""Download the public benchmark data defined in the config.

    python scripts/download_data.py --what all
    python scripts/download_data.py --what biovista
    python scripts/download_data.py --what pdfs

* biovista - BioVista OCSR task (structure images + ground-truth SMILES),
             fetched from Google Drive with gdown and unzipped into
             datasets.images.root.
* pdfs     - open-access example papers listed under downloads.pdf_urls,
             saved into datasets.pdfs.root.

Existing files are skipped, so the script is safe to re-run.
"""
from __future__ import annotations

import argparse
import sys
import tarfile
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xnbench.config import load_config, resolve_path  # noqa: E402


def extract(archive: Path, dest: Path):
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(dest)
    else:
        raise ValueError(f"{archive} is not a zip or tar archive")


def download_biovista(cfg):
    import gdown

    dest = resolve_path(cfg.datasets.images.root)
    if dest.exists() and any(dest.rglob("*")):
        print(f"[biovista] already present in {dest}, skipping")
        return
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "biovista_ocsr_download"
    print("[biovista] downloading from Google Drive ...")
    gdown.download(id=cfg.downloads.biovista_ocsr_gdrive_id, output=str(archive), quiet=False)
    print("[biovista] extracting ...")
    extract(archive, dest)
    archive.unlink()
    print(f"[biovista] ready: {dest}")


def download_pdfs(cfg):
    dest = resolve_path(cfg.datasets.pdfs.root)
    dest.mkdir(parents=True, exist_ok=True)
    for url in cfg.downloads.pdf_urls or []:
        out = dest / url.rsplit("/", 1)[-1]
        if out.exists():
            print(f"[pdfs] {out.name} exists, skipping")
            continue
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        out.write_bytes(resp.content)
        print(f"[pdfs] saved {out.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--what", choices=["all", "biovista", "pdfs"], default="all")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.what in ("all", "biovista"):
        download_biovista(cfg)
    if args.what in ("all", "pdfs"):
        download_pdfs(cfg)


if __name__ == "__main__":
    main()
