"""Metric logic tests (pandas only; a stub similarity replaces RDKit)."""
import pandas as pd

from xnbench import evaluate


def stub_similarity(a, b):
    return 1.0 if a == b else 0.5


def _image_rows():
    # 3 images: exact hit, stereo-only error, invalid prediction
    return pd.DataFrame({
        "true_valid": [True, True, True],
        "pred_valid": [True, True, False],
        "true_key": ["AAA-X-N", "BBB-Y-N", "CCC-Z-N"],
        "pred_key": ["AAA-X-N", "BBB-Q-N", ""],
        "true_connectivity_key": ["AAA", "BBB", "CCC"],
        "pred_connectivity_key": ["AAA", "BBB", ""],
        "true_canonical_smiles": ["CCO", "C[C@H](N)O", "c1ccccc1"],
        "pred_canonical_smiles": ["CCO", "C[C@@H](N)O", ""],
        "confidence": [0.99, 0.8, 0.1],
        "seconds": [1.0, 1.0, 1.0],
        "category": ["full", "chiral", "full"],
    })


def test_pairwise_scores():
    df = evaluate.add_pairwise_scores(_image_rows(), stub_similarity)
    assert df["exact_match"].tolist() == [True, False, False]
    assert df["connectivity_match"].tolist() == [True, True, False]
    assert df["tanimoto"].tolist()[:2] == [1.0, 0.5]
    assert df["tanimoto"].isna().tolist()[2]


def test_summary_counts_invalid_as_wrong():
    df = evaluate.add_pairwise_scores(_image_rows(), stub_similarity)
    overall, per_cat = evaluate.summarize_images(df, near_miss=0.4)
    assert overall["n"] == 3
    assert overall["exact_match_rate"] == round(1 / 3, 4)
    assert overall["connectivity_match_rate"] == round(2 / 3, 4)
    assert overall["stereo_only_error_rate"] == round(1 / 3, 4)
    assert overall["validity_rate"] == round(2 / 3, 4)
    assert overall["mean_tanimoto"] == 0.5          # (1 + 0.5 + 0) / 3
    assert overall["mean_tanimoto_valid_only"] == 0.75
    assert overall["near_miss_rate"] == round(1 / 3, 4)
    assert set(per_cat["category"]) == {"all", "full", "chiral"}


def test_pdf_set_metrics():
    preds = pd.DataFrame({
        "pdf": ["a.pdf", "a.pdf", "a.pdf"],
        "pred_valid": [True, True, False],
        "pred_key": ["K1", "K9", ""],
        "pred_connectivity_key": ["C1", "C2", ""],
        "pred_canonical_smiles": ["s1", "s9", ""],
    })
    truth = pd.DataFrame({
        "pdf": ["a.pdf", "a.pdf"],
        "true_valid": [True, True],
        "true_key": ["K1", "K2"],
        "true_connectivity_key": ["C1", "C2"],
        "true_canonical_smiles": ["s1", "s2"],
    })
    overall, per_pdf, scored, missed = evaluate.evaluate_pdfs(preds, truth, stub_similarity)
    row = per_pdf.iloc[0]
    assert row["true_positives"] == 1
    assert row["precision"] == 0.5 and row["recall"] == 0.5 and row["f1"] == 0.5
    assert row["recall_connectivity"] == 1.0       # K9 has the right skeleton
    assert scored["in_truth"].tolist() == [True, False, False]
    assert missed["true_key"].tolist() == ["K2"]
    assert bool(missed["found_connectivity_only"].iloc[0]) is True
    assert overall["micro_f1"] == 0.5
