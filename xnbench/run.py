"""Benchmark entry point.

    python -m xnbench.run --config configs/default.yaml --mode images
    python -m xnbench.run --mode pdfs --limit 2

Pipeline: data loading -> inference -> RDKit validation -> PubChem/ChEMBL
verification -> pandas evaluation -> structured outputs in results/<run>/.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

from . import __version__, datasets, evaluate
from .config import REPO_ROOT, load_config, pick_device, resolve_path

log = logging.getLogger("xnbench")


# --------------------------------------------------------------------------- #
# Run bookkeeping
# --------------------------------------------------------------------------- #
def make_run_dir(cfg, mode: str, run_name: str | None, output_dir: str | None = None) -> Path:
    name = run_name or cfg.run_name or f"{mode}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = resolve_path(output_dir or cfg.output_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logging(run_dir: Path):
    fmt = "%(asctime)s %(levelname)-7s %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(run_dir / "run.log", encoding="utf-8")])


def write_run_info(run_dir: Path, cfg, args):
    """Record everything needed to reproduce this run."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = "unknown"
    versions = {}
    for pkg in ["rdkit", "pandas", "numpy", "PyMuPDF", "decimer-segmentation",
                "MolScribe", "torch", "tensorflow", "requests"]:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    info = {"started": datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(sys.argv), "git_commit": commit,
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": versions}
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(dict(cfg), sort_keys=False),
                                              encoding="utf-8")


def _to_builtin(value):
    """Make numpy scalars JSON-friendly (so metrics stay numbers, not strings)."""
    return value.item() if hasattr(value, "item") else str(value)


def save_json(obj: dict, path: Path):
    path.write_text(json.dumps(obj, indent=2, default=_to_builtin), encoding="utf-8")


def build_verifier(cfg, enabled: bool):
    sources = list(cfg.verification.sources or [])
    if not enabled or not sources:
        return None
    from .verify import Verifier

    v = cfg.verification
    return Verifier(sources, resolve_path(v.cache_file), v.timeout_s, v.sleep_s)


def validation_columns(smiles_list, prefix: str) -> pd.DataFrame:
    from .chem import validate_smiles

    return pd.DataFrame([validate_smiles(s).as_dict(prefix) for s in smiles_list])


def similarity_fn(cfg):
    from .chem import tanimoto

    radius, bits = cfg.evaluation.fingerprint_radius, cfg.evaluation.fingerprint_bits
    return lambda a, b: tanimoto(a, b, radius, bits)


def add_verification(df: pd.DataFrame, verifier, check_labels: bool) -> pd.DataFrame:
    if verifier is None:
        return df
    log.info("Verifying %d valid predictions against %s ...",
             int(df["pred_valid"].sum()), ", ".join(verifier.sources))
    rows = []
    for _, r in df.iterrows():
        rows.append(verifier.verify(r["pred_inchikey"], r.get("label", ""), check_labels)
                    if r["pred_valid"] else {})
    verifier.save_cache()
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


# --------------------------------------------------------------------------- #
# Mode 1: BioVista OCSR image benchmark
# --------------------------------------------------------------------------- #
def run_images(cfg, args, run_dir: Path):
    from .chem import structure_category
    from .models import MolScribeRecognizer

    data = datasets.load_image_dataset(cfg.datasets.images, cfg.seed, args.limit)
    log.info("Benchmarking %d images", len(data))
    model = MolScribeRecognizer(cfg.recognition.hf_repo, cfg.recognition.checkpoint,
                                pick_device(args.device or cfg.device))

    preds, confs, secs, errors = [], [], [], []
    for i, path in enumerate(data["image_path"], 1):
        start = time.perf_counter()
        try:
            smi, conf = model.predict(path)
            err = ""
        except Exception as exc:  # a single bad image must not stop the benchmark
            smi, conf, err = "", None, f"{type(exc).__name__}: {exc}"
        secs.append(round(time.perf_counter() - start, 4))
        preds.append(smi); confs.append(conf); errors.append(err)
        if i % 50 == 0 or i == len(data):
            log.info("  %d / %d images", i, len(data))

    df = data.assign(pred_smiles=preds, confidence=confs, seconds=secs, inference_error=errors)
    df["category"] = df["true_smiles"].map(structure_category)
    df = pd.concat([df, validation_columns(df["true_smiles"], "true_"),
                    validation_columns(df["pred_smiles"], "pred_")], axis=1)
    df = evaluate.add_pairwise_scores(df, similarity_fn(cfg))
    df = add_verification(df, build_verifier(cfg, not args.no_verify), check_labels=False)

    overall, per_category = evaluate.summarize_images(df, cfg.evaluation.near_miss_threshold)
    overall |= evaluate.summarize_verification(df)

    df.to_csv(run_dir / "predictions.csv", index=False)
    per_category.to_csv(run_dir / "metrics_by_category.csv", index=False)
    save_json({"mode": "images", "tool": "MolScribe", **overall}, run_dir / "metrics.json")
    print("\n" + per_category.to_string(index=False))


# --------------------------------------------------------------------------- #
# Mode 2: full PDF pipeline
# --------------------------------------------------------------------------- #
def run_pdfs(cfg, args, run_dir: Path):
    from .models import DecimerDetector, MolScribeRecognizer, pad_crop
    from .pdf_text import find_label, iter_pages

    pdfs = datasets.list_pdfs(cfg.datasets.pdfs, args.limit)
    truth = datasets.load_pdf_truth(cfg.datasets.pdfs)
    detector = DecimerDetector(expand=True)
    model = MolScribeRecognizer(cfg.recognition.hf_repo, cfg.recognition.checkpoint,
                                pick_device(args.device or cfg.device))
    crops_dir = run_dir / "crops"
    crops_dir.mkdir(exist_ok=True)
    p = cfg.pdf

    rows = []
    for pdf in pdfs:
        log.info("PDF %s", pdf.name)
        for page in iter_pages(pdf, p.dpi):
            t0 = time.perf_counter()
            detections = detector(page.image)
            detect_secs = (time.perf_counter() - t0) / max(len(detections), 1)
            for k, (crop, bbox) in enumerate(detections):
                if min(crop.shape[:2]) < p.min_crop_px:
                    continue
                crop_id = f"{pdf.stem}_p{page.number:03d}_{k:02d}"
                crop_path = crops_dir / f"{crop_id}.png"
                Image.fromarray(pad_crop(crop, p.crop_padding)).save(crop_path)
                t1 = time.perf_counter()
                try:
                    smi, conf = model.predict(str(crop_path))
                    err = ""
                except Exception as exc:
                    smi, conf, err = "", None, f"{type(exc).__name__}: {exc}"
                rows.append({
                    "pdf": pdf.name, "page": page.number, "crop_id": crop_id,
                    "crop_path": str(crop_path.relative_to(run_dir)),
                    "bbox_px": list(map(int, bbox)) if bbox is not None else None,
                    "label": find_label(page, bbox, p.label_max_gap_pt),
                    "pred_smiles": smi, "confidence": conf, "inference_error": err,
                    "seconds": round(detect_secs + time.perf_counter() - t1, 4),
                })
        log.info("  %d structures so far", len(rows))

    if not rows:
        log.warning("No structures detected in any PDF.")
        return
    df = pd.DataFrame(rows)
    df = pd.concat([df, validation_columns(df["pred_smiles"], "pred_")], axis=1)
    df = add_verification(df, build_verifier(cfg, not args.no_verify),
                          cfg.verification.check_labels)

    metrics = {"mode": "pdfs", "detector": "DECIMER Segmentation", "recognizer": "MolScribe",
               "n_pdfs": len(pdfs), "n_crops": len(df),
               "validity_rate": round(float(df["pred_valid"].mean()), 4),
               "mean_seconds_per_structure": round(float(df["seconds"].mean()), 4),
               "labels_found_rate": round(float(df["label"].ne("").mean()), 4)}
    metrics |= evaluate.summarize_verification(df)

    if truth is not None:
        truth = pd.concat([truth.reset_index(drop=True),
                           validation_columns(truth["smiles"], "true_")], axis=1)
        overall, per_pdf, df, missed = evaluate.evaluate_pdfs(df, truth, similarity_fn(cfg))
        metrics |= overall
        per_pdf.to_csv(run_dir / "metrics_by_pdf.csv", index=False)
        missed.to_csv(run_dir / "missed_molecules.csv", index=False)
        print("\n" + per_pdf.to_string(index=False))

    df.to_csv(run_dir / "predictions.csv", index=False)
    save_json(metrics, run_dir / "metrics.json")


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="Benchmark PDF/image -> SMILES extraction")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mode", choices=["images", "pdfs"], help="overrides config 'mode'")
    ap.add_argument("--limit", type=int, help="process only N images / PDFs (smoke test)")
    ap.add_argument("--run-name", help="output folder name under results/")
    ap.add_argument("--output-dir", help="where run folders go (default: config output_dir), "
                                         "e.g. a mounted Google Drive folder")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--no-verify", action="store_true", help="skip PubChem/ChEMBL lookups")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    mode = args.mode or cfg.mode
    run_dir = make_run_dir(cfg, mode, args.run_name, args.output_dir)
    setup_logging(run_dir)
    write_run_info(run_dir, cfg, args)
    log.info("xnbench %s | run folder: %s", __version__, run_dir)

    (run_images if mode == "images" else run_pdfs)(cfg, args, run_dir)
    log.info("Done. Results in %s", run_dir)


if __name__ == "__main__":
    main()
