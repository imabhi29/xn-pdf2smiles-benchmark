#!/usr/bin/env bash
# One-time setup for Google Colab (works on any Linux machine).
#
# Colab's system Python changes over time and is too new for MolScribe's
# pinned dependencies (numpy<2, OpenNMT-py/pyonmttok). So we create a private
# Python 3.10 environment in .venv with `uv` and run the benchmark there.
# The notebook kernel itself only needs pandas/matplotlib/rdkit for plots.
#
# Safe to re-run: skips everything once setup has succeeded.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/python"
MARKER=".venv/.setup_done"
if [[ -f "$MARKER" ]]; then
  echo "Environment already set up (.venv). Delete .venv to rebuild."
  exit 0
fi

echo "[1/4] Installing uv ..."
python -m pip install -q uv

echo "[2/4] Creating Python 3.10 environment in .venv ..."
uv venv .venv --python 3.10 --seed -q

echo "[3/4] Installing benchmark dependencies (about 5-8 minutes) ..."
uv pip install --python "$PY" -r requirements.txt

echo "[4/4] Checking imports ..."
"$PY" - << 'PY'
import importlib, sys
print("  python", sys.version.split()[0])
for mod in ["fitz", "numpy", "pandas", "rdkit", "torch", "decimer_segmentation", "molscribe"]:
    m = importlib.import_module(mod)
    print(f"  ok  {mod} {getattr(m, '__version__', '')}")
from molscribe import MolScribe  # noqa: F401  (fails early if MolScribe is broken)
PY

touch "$MARKER"
echo "Setup complete."
