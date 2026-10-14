"""Replace the YOUR_GITHUB_USERNAME placeholder in the README and Colab notebook.

    python scripts/set_github_user.py <github-username>

Run once after creating your GitHub repo, then commit, so the "Open in Colab"
badge and the clone URL point at your repository.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "README.md", ROOT / "notebooks" / "run_benchmark_colab.ipynb"]

if len(sys.argv) != 2:
    sys.exit(__doc__)
user = sys.argv[1].strip()
for f in FILES:
    text = f.read_text(encoding="utf-8")
    f.write_text(text.replace("YOUR_GITHUB_USERNAME", user), encoding="utf-8")
    print(f"updated {f.relative_to(ROOT)}")
