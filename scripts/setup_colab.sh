#!/usr/bin/env bash
# One-time dependency install for Google Colab (also fine on any Linux VM).
# Safe to re-run: a marker file skips the install once it has succeeded.
set -euo pipefail
cd "$(dirname "$0")/.."

MARKER=".colab_setup_done"
if [[ -f "$MARKER" ]]; then
  echo "Dependencies already installed (delete $MARKER to force a reinstall)."
  exit 0
fi

echo "Installing dependencies (about 3-5 minutes on Colab)..."
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt

python - << 'PY'
import importlib
for mod in ["fitz", "rdkit", "decimer_segmentation", "molscribe", "torch", "pandas", "yaml"]:
    importlib.import_module(mod)
    print(f"  ok  {mod}")
PY
touch "$MARKER"
echo "Setup complete."
