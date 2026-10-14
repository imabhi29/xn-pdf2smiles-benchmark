"""Evaluation with pandas.

Image mode (paired):  each image has exactly one ground-truth SMILES.
    exact match, connectivity match, validity, Tanimoto, near misses,
    confidence analysis, speed - overall and per structure category.

PDF mode (set-based): each PDF has a set of expected molecules.
    precision / recall / F1 over unique molecules (exact and connectivity
    level), best Tanimoto of each prediction to the paper's ground truth,
    and a list of missed molecules for error analysis.

The functions take a ``similarity(smiles_a, smiles_b)`` callable, so the
metric logic can be unit-tested without RDKit.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

Similarity = Callable[[str, str], "float | None"]


def _safe_div(a: float, b: float) -> float | None:
    return round(a / b, 4) if b else None


def _f1(p: float | None, r: float | None) -> float | None:
    if p is None or r is None:
        return None
    if p + r == 0:
        return 0.0
    return round(2 * p * r / (p + r), 4)


def _mean(series: pd.Series) -> float | None:
    series = pd.to_numeric(series, errors="coerce").dropna()
    return round(float(series.mean()), 4) if len(series) else None


# --------------------------------------------------------------------------- #
# Image mode
# --------------------------------------------------------------------------- #
def add_pairwise_scores(df: pd.DataFrame, similarity: Similarity) -> pd.DataFrame:
    """Add exact_match, connectivity_match and tanimoto columns (one row per image)."""
    df = df.copy()
    both = df["true_valid"].astype(bool) & df["pred_valid"].astype(bool)
    df["exact_match"] = both & (df["true_key"] == df["pred_key"])
    df["connectivity_match"] = both & (df["true_connectivity_key"] == df["pred_connectivity_key"])
    df["tanimoto"] = [
        similarity(t, p) if ok else None
        for t, p, ok in zip(df["true_canonical_smiles"], df["pred_canonical_smiles"], both)
    ]
    return df


def _summarize_group(g: pd.DataFrame, near_miss: float) -> dict:
    n = len(g)
    tan = pd.to_numeric(g["tanimoto"], errors="coerce")
    strict_tan = tan.fillna(0.0)          # invalid prediction counts as similarity 0
    near = (~g["exact_match"]) & (tan >= near_miss)
    conf = pd.to_numeric(g.get("confidence"), errors="coerce") if "confidence" in g else None
    row = {
        "n": n,
        "validity_rate": _safe_div(g["pred_valid"].sum(), n),
        "exact_match_rate": _safe_div(g["exact_match"].sum(), n),
        "connectivity_match_rate": _safe_div(g["connectivity_match"].sum(), n),
        "stereo_only_error_rate": _safe_div(
            (g["connectivity_match"] & ~g["exact_match"]).sum(), n),
        "mean_tanimoto": round(float(strict_tan.mean()), 4) if n else None,
        "mean_tanimoto_valid_only": _mean(tan),
        "median_tanimoto_valid_only": round(float(tan.median()), 4) if tan.notna().any() else None,
        "near_miss_rate": _safe_div(near.sum(), n),
        "mean_seconds_per_image": _mean(g["seconds"]) if "seconds" in g else None,
    }
    if conf is not None and conf.notna().any():
        row["mean_confidence_correct"] = _mean(conf[g["exact_match"]])
        row["mean_confidence_incorrect"] = _mean(conf[~g["exact_match"]])
    return row


def summarize_images(df: pd.DataFrame, near_miss: float = 0.85) -> tuple[dict, pd.DataFrame]:
    """Return (overall metrics dict, per-category DataFrame)."""
    scorable = df[df["true_valid"].astype(bool)]
    rows = [{"category": "all", **_summarize_group(scorable, near_miss)}]
    for cat, g in scorable.groupby("category"):
        rows.append({"category": cat, **_summarize_group(g, near_miss)})
    per_category = pd.DataFrame(rows)
    overall = rows[0] | {
        "n_images_total": len(df),
        "n_ground_truth_unparseable": int((~df["true_valid"].astype(bool)).sum()),
        "near_miss_threshold": near_miss,
    }
    return overall, per_category


# --------------------------------------------------------------------------- #
# PDF mode
# --------------------------------------------------------------------------- #
def evaluate_pdfs(preds: pd.DataFrame, truth: pd.DataFrame,
                  similarity: Similarity) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Set-based evaluation per PDF.

    ``preds`` needs: pdf, pred_valid, pred_key, pred_connectivity_key,
    pred_canonical_smiles. ``truth`` needs: pdf, true_valid, true_key,
    true_connectivity_key, true_canonical_smiles (and optionally name).

    Returns (overall metrics, per-PDF table, predictions with best_tanimoto
    and in_truth columns, table of missed ground-truth molecules).
    """
    preds = preds.copy()
    truth = truth[truth["true_valid"].astype(bool)].copy()
    preds["in_truth"] = False
    preds["best_tanimoto"] = np.nan

    per_pdf, missed = [], []
    for pdf in sorted(set(preds["pdf"]) | set(truth["pdf"])):
        p = preds[(preds["pdf"] == pdf) & preds["pred_valid"].astype(bool)]
        t = truth[truth["pdf"] == pdf]
        pred_keys, true_keys = set(p["pred_key"]), set(t["true_key"])
        pred_conn, true_conn = set(p["pred_connectivity_key"]), set(t["true_connectivity_key"])
        tp, tp_conn = len(pred_keys & true_keys), len(pred_conn & true_conn)

        # Per-prediction: is it in the paper's ground truth, and how close is it?
        for idx, row in p.iterrows():
            preds.at[idx, "in_truth"] = row["pred_key"] in true_keys
            sims = [similarity(row["pred_canonical_smiles"], s) for s in t["true_canonical_smiles"]]
            sims = [s for s in sims if s is not None]
            preds.at[idx, "best_tanimoto"] = max(sims) if sims else np.nan

        miss = t[~t["true_key"].isin(pred_keys)]
        missed.append(miss.assign(found_connectivity_only=miss["true_connectivity_key"].isin(pred_conn)))

        precision, recall = _safe_div(tp, len(pred_keys)), _safe_div(tp, len(true_keys))
        per_pdf.append({
            "pdf": pdf,
            "n_crops": int((preds["pdf"] == pdf).sum()),
            "n_valid_predictions": len(p),
            "n_unique_predicted": len(pred_keys),
            "n_truth": len(true_keys),
            "true_positives": tp,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
            "recall_connectivity": _safe_div(tp_conn, len(true_conn)),
            "mean_best_tanimoto": _mean(preds.loc[p.index, "best_tanimoto"]),
        })

    per_pdf_df = pd.DataFrame(per_pdf)
    has_truth = per_pdf_df[per_pdf_df["n_truth"] > 0]
    tp_sum = int(has_truth["true_positives"].sum())
    micro_p = _safe_div(tp_sum, has_truth["n_unique_predicted"].sum())
    micro_r = _safe_div(tp_sum, has_truth["n_truth"].sum())
    overall = {
        "n_pdfs": len(per_pdf_df),
        "n_pdfs_with_truth": len(has_truth),
        "n_crops": int(len(preds)),
        "validity_rate": _safe_div(preds["pred_valid"].sum(), len(preds)),
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": _f1(micro_p, micro_r),
        "macro_f1": _mean(has_truth["f1"]),
        "macro_recall_connectivity": _mean(has_truth["recall_connectivity"]),
        "mean_best_tanimoto": _mean(preds["best_tanimoto"]),
    }
    missed_df = pd.concat(missed, ignore_index=True) if missed else pd.DataFrame()
    return overall, per_pdf_df, preds, missed_df


def summarize_verification(df: pd.DataFrame) -> dict:
    """Share of valid predictions found in each database / matching their label."""
    out = {}
    for col in [c for c in df.columns if c.endswith("_found") or c.endswith("_label_match")]:
        known = df[col].dropna()
        out[f"{col}_rate"] = _safe_div(known.astype(bool).sum(), len(known))
        out[f"{col}_n_checked"] = int(len(known))
    return out
