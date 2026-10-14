"""Show what the downloaded BioVista OCSR data looks like.

    python scripts/inspect_dataset.py

Prints the folder layout, example image names and the first rows of every
label file, which helps when setting image_col / smiles_col in the config.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xnbench.config import load_config, resolve_path  # noqa: E402
from xnbench.datasets import IMAGE_EXTS, LABEL_EXTS, read_label_file  # noqa: E402

cfg = load_config("configs/default.yaml")
root = resolve_path(cfg.datasets.images.root)
files = [p for p in root.rglob("*") if p.is_file()]
images = sorted(p for p in files if p.suffix.lower() in IMAGE_EXTS)

print(f"Root: {root}\nFiles: {len(files)}  Images: {len(images)}")
print("Images per folder:", dict(Counter(p.parent.relative_to(root).as_posix() for p in images).most_common(5)))
print("Example images:")
for p in images[:8]:
    print("  ", p.relative_to(root).as_posix())
for label in sorted(p for p in files if p.suffix.lower() in LABEL_EXTS):
    print(f"\nLabel file: {label.relative_to(root).as_posix()}")
    try:
        print(read_label_file(label).head(5).to_string())
    except Exception as exc:
        print("   unreadable:", exc)
