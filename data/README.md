# Data folder

Nothing in this folder is committed except this file and the ground-truth
template. Everything else is downloaded or created locally.

| Path | Contents | How to get it |
|---|---|---|
| `biovista_ocsr/` | BioVista OCSR images + ground-truth SMILES | `python scripts/download_data.py --what biovista` |
| `pdfs/` | PDFs for the full-pipeline benchmark | `python scripts/download_data.py --what pdfs`, or copy your own PDFs here |
| `ground_truth/pdf_truth.csv` | Expected molecules per PDF | Create from `pdf_truth_template.csv` |

## Ground truth for PDFs (automatic)

`python scripts/build_pdf_truth.py` creates `pdf_truth.csv` for every PDF in
`data/pdfs/` from two sources, without any hand-typed SMILES:

1. **BioVista labels**: PDFs named like BioVista papers (e.g. `68_6r8r.pdf`,
   the open-access BioMiner examples) get their expert-curated structures.
2. **Name lists**: `data/ground_truth/*_names.csv` files (columns `pdf,name`)
   are resolved to SMILES through PubChem. `binder2026_names.csv` covers the
   open-access kinase paper by Binder et al. (2026, *J. Med. Chem.*,
   CC BY-NC-ND 4.0; https://doi.org/10.1021/acs.jmedchem.5c02916). Save that
   PDF as `data/pdfs/binder2026_jmedchem.pdf` to include it.

To test your own paper: put the PDF in `data/pdfs/`, add a
`<something>_names.csv` listing the named compounds in it, and rerun the script.

## Ground truth format

`pdf_truth.csv` needs one row per expected molecule:

- `pdf`: the PDF file name (e.g. `68_6r8r.pdf`)
- `smiles`: the correct SMILES
- `name` (optional): the compound name or number used in the paper

Good sources are a paper's Supporting Information (many journals provide a
SMILES or "molecular formula strings" file) or PubChem records for named
compounds. Without this file the PDF mode still runs extraction, validation
and verification, but skips precision/recall.
