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
