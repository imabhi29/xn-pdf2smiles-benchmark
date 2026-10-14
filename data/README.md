# Data folder

Nothing in this folder is committed except this file and the ground-truth
template. Everything else is downloaded or created locally.

| Path | Contents | How to get it |
|---|---|---|
| `biovista_ocsr/` | BioVista OCSR images + ground-truth SMILES | `python scripts/download_data.py --what biovista` |
| `pdfs/` | PDFs for the full-pipeline benchmark | `python scripts/download_data.py --what pdfs`, or copy your own PDFs here |
| `ground_truth/pdf_truth.csv` | Expected molecules per PDF | Create from `pdf_truth_template.csv` |

## Ground truth for PDFs

`pdf_truth.csv` needs one row per expected molecule:

- `pdf`: the PDF file name (e.g. `68_6r8r.pdf`)
- `smiles`: the correct SMILES
- `name` (optional): the compound name or number used in the paper

Good sources are a paper's Supporting Information (many journals provide a
SMILES or "molecular formula strings" file) or PubChem records for named
compounds. Without this file the PDF mode still runs extraction, validation
and verification, but skips precision/recall.
