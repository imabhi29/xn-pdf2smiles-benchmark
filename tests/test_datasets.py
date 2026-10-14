"""Label-file auto-detection tests (no models needed)."""
import json

import pytest
from PIL import Image

from xnbench import datasets
from xnbench.config import Config


@pytest.fixture
def image_dir(tmp_path):
    for name in ["img_1.png", "img_2.png", "img_3.png"]:
        Image.new("RGB", (10, 10), "white").save(tmp_path / name)
    return tmp_path


def _cfg(root, **kw):
    base = {"root": str(root), "labels_file": None, "image_col": None,
            "smiles_col": None, "sample_size": None}
    return Config(base | kw)


def test_csv_labels(image_dir):
    (image_dir / "labels.csv").write_text("file_name,SMILES\nimg_1.png,CCO\nimg_2.png,CCN\n")
    df = datasets.load_image_dataset(_cfg(image_dir), seed=0)
    assert len(df) == 2 and set(df["true_smiles"]) == {"CCO", "CCN"}


def test_json_mapping_and_stem_match(image_dir):
    (image_dir / "gt.json").write_text(json.dumps({"img_1": "CCO", "img_3": "c1ccccc1"}))
    df = datasets.load_image_dataset(_cfg(image_dir), seed=0)
    assert len(df) == 2


def test_sampling_is_seeded(image_dir):
    rows = "\n".join(f"img_{i}.png,C{'C' * i}" for i in (1, 2, 3))
    (image_dir / "labels.csv").write_text("image,smiles\n" + rows)
    a = datasets.load_image_dataset(_cfg(image_dir, sample_size=2), seed=7)
    b = datasets.load_image_dataset(_cfg(image_dir, sample_size=2), seed=7)
    assert a["sample_id"].tolist() == b["sample_id"].tolist()


def test_helpful_error_when_columns_unknown(image_dir):
    (image_dir / "labels.csv").write_text("a,b\nimg_1.png,CCO\n")
    with pytest.raises(ValueError, match="labels_file"):
        datasets.load_image_dataset(_cfg(image_dir), seed=0)


def test_combined_key_when_names_repeat_across_folders(tmp_path):
    """BioVista-style labels: images live in per-paper folders as 0.png, 1.png ..."""
    for pdf in ["6r8r", "1abc"]:
        (tmp_path / pdf).mkdir()
        for i in range(2):
            Image.new("RGB", (10, 10), "white").save(tmp_path / pdf / f"{i}.png")
    (tmp_path / "gd_SMIELS.csv").write_text(
        ",pdf_name,index,molminer_id,smiles\n"
        "0,6r8r,0,m1,CCO\n1,6r8r,1,m2,CCN\n2,1abc,0,m3,CCC\n3,1abc,1,m4,CCCC\n")
    df = datasets.load_image_dataset(_cfg(tmp_path), seed=0)
    pairs = {(row.sample_id, row.true_smiles) for row in df.itertuples()}
    assert pairs == {("6r8r/0", "CCO"), ("6r8r/1", "CCN"), ("1abc/0", "CCC"), ("1abc/1", "CCCC")}
