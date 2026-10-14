# XN Benchmark: PDF / Image → SMILES Extraction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imabhi29/xn-pdf2smiles-benchmark/blob/main/notebooks/run_benchmark_colab.ipynb)
[![tests](https://github.com/imabhi29/xn-pdf2smiles-benchmark/actions/workflows/tests.yml/badge.svg)](https://github.com/imabhi29/xn-pdf2smiles-benchmark/actions/workflows/tests.yml)

Reproducible benchmarking code for **XN Project Assignment 1.2** (Phase 1: Benchmarking & Project Setup).
Author: Abhishek Satish Singh

The code measures how accurately chemical structures can be extracted from scientific documents and converted to SMILES, using the BioVista benchmark's public OCSR data and the full PDF pipeline below. It runs end to end from one command: data loading → inference → validation → verification → metrics → structured output files.

## Tool stack

| Stage | Tool | What it does here |
|---|---|---|
| Text layer and rendering | **PyMuPDF** | Renders PDF pages to images; reads text blocks to find compound labels |
| Detection and cropping | **DECIMER Segmentation** | Locates each structure drawing on a page and crops it |
| Image → SMILES | **MolScribe** | Predicts the molecular graph and SMILES, with a confidence score |
| Validation | **RDKit** | Parses and sanitizes SMILES; computes canonical SMILES, InChIKey, formula |
| Verification | **PubChem / ChEMBL web APIs** | Checks that predicted structures exist; checks detected names against predictions |
| Evaluation | **RDKit + pandas** | Exact match, connectivity match, Tanimoto similarity, precision/recall/F1 |

## Two benchmark modes

**`images` (main benchmark).** Runs MolScribe on the **BioVista OCSR task** (structure images with expert-curated SMILES; Yan et al., 2026) and scores every prediction against its label. This is fully automatic: the dataset downloads with one command.

**`pdfs` (full pipeline).** Runs the complete PDF → SMILES pipeline on a folder of PDFs. If a ground-truth CSV is provided, it reports precision, recall and F1 per paper; otherwise it still extracts, validates and verifies.

## Repository map

```
xn-pdf2smiles-benchmark/
├── README.md                  # this file
├── requirements.txt           # pip dependencies (Python 3.10)
├── environment.yml            # conda environment wrapping requirements.txt
├── configs/default.yaml       # every setting: paths, sample size, seed, APIs, thresholds
├── notebooks/
│   └── run_benchmark_colab.ipynb  # one-click Colab run: install → test → download → run → results
├── scripts/
│   ├── download_data.py       # downloads BioVista OCSR data and example PDFs
│   ├── inspect_dataset.py     # prints the downloaded data layout and label columns
│   ├── setup_colab.sh         # builds the Python 3.10 environment (Colab or any Linux VM)
│   └── set_github_user.py     # fills your username into the Colab badge and clone URL
├── xnbench/                   # the benchmark package
│   ├── run.py                 #   entry point: orchestrates the whole pipeline
│   ├── datasets.py            #   loads BioVista images/labels and PDF folders
│   ├── pdf_text.py            #   PyMuPDF rendering + label lookup from the text layer
│   ├── models.py              #   DECIMER detector and MolScribe recognizer wrappers
│   ├── chem.py                #   RDKit validation, InChIKeys, Tanimoto similarity
│   ├── verify.py              #   PubChem / ChEMBL lookups with on-disk cache
│   ├── evaluate.py            #   pandas metrics for both modes
│   └── config.py              #   YAML loading; paths resolved from the repo root
├── tests/                     # pytest unit tests for metrics, chemistry, data loading
├── data/                      # downloaded data (git-ignored; see data/README.md)
└── results/                   # one folder per run (git-ignored)
```

## Quick start on Google Colab (recommended)

1. Click the **Open in Colab** badge at the top of this page.
2. Optional: *Runtime → Change runtime type → T4 GPU* for faster inference.
3. *Runtime → Run all*. The first run takes about 5–8 minutes to build the environment, then continues through to charts, an error gallery, and a downloadable zip of results.

MolScribe's dependencies only install on Python 3.10, while Colab ships a newer Python. The notebook therefore builds a private Python 3.10 environment (`.venv`, via `uv`) and runs every benchmark command with it, so the setup does not break when Colab upgrades.

Settings (mode, number of images, PubChem on/off, save to Google Drive) are in the notebook's first cell. Nothing needs to be uploaded: the notebook clones this repo and downloads the public data itself.

## Local setup

Use **Python 3.10** (MolScribe's dependencies do not install on newer versions). Ubuntu, macOS, or Windows via WSL2; a GPU is optional. Colab is the tested route; native Windows is untested.

```bash
git clone <this-repo-url>
cd xn-pdf2smiles-benchmark

# Option A: conda
conda env create -f environment.yml
conda activate xn-benchmark

# Option B: venv
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads the MolScribe checkpoint from Hugging Face and the DECIMER Segmentation weights automatically, so internet access is needed once. After a successful install, record exact versions with `pip freeze > requirements-lock.txt`.

## Running the benchmark

```bash
# 1. Check the install (no models or data needed)
pytest -q

# 2. Download public data
python scripts/download_data.py --what all

# 3. Quick smoke test on 10 images
python -m xnbench.run --mode images --limit 10 --run-name smoke_test

# 4. Main benchmark (500-image seeded sample by default; set sample_size: null for all)
python -m xnbench.run --mode images

# 5. Full PDF pipeline (add data/ground_truth/pdf_truth.csv for precision/recall)
python -m xnbench.run --mode pdfs
```

Useful options: `--device cuda`, `--no-verify` (skip API calls), `--output-dir <folder>` (e.g. Google Drive), `--config path/to/other.yaml`. Every other setting lives in `configs/default.yaml`.

## Outputs

Each run creates `results/<run-name>/`:

| File | Mode | Contents |
|---|---|---|
| `predictions.csv` | both | One row per image/crop: ground truth, predicted SMILES, confidence, RDKit validation fields, match flags, Tanimoto, verification results, timing |
| `metrics.json` | both | Overall metrics for the run |
| `metrics_by_category.csv` | images | Metrics split into full, chiral and Markush structures |
| `metrics_by_pdf.csv` | pdfs | Precision, recall, F1 and similarity per paper |
| `missed_molecules.csv` | pdfs | Ground-truth molecules the pipeline did not find |
| `crops/` | pdfs | The cropped image of every detected structure |
| `run_info.json`, `config_used.yaml`, `run.log` | both | Git commit, package versions, command and full config, for reproducibility |

## Metrics

| Metric | Definition |
|---|---|
| Validity rate | Share of predictions RDKit can parse and sanitize |
| Exact match | Predicted and true InChIKey identical (same molecule, including stereochemistry) |
| Connectivity match | First InChIKey block identical (same skeleton, stereo ignored) |
| Stereo-only error rate | Connectivity correct but stereochemistry wrong |
| Tanimoto similarity | Morgan fingerprints (radius 2, 2048 bits; Rogers & Hahn, 2010). `mean_tanimoto` counts invalid predictions as 0; `mean_tanimoto_valid_only` ignores them |
| Near-miss rate | Not an exact match, but Tanimoto ≥ 0.85 |
| Confidence (correct vs. incorrect) | Mean MolScribe confidence for right and wrong predictions; a gap means confidence can flag errors |
| Precision / recall / F1 (pdfs) | Over unique InChIKeys per paper, against the ground-truth CSV; micro and macro averages reported |
| Verification rates | Share of predictions found in PubChem/ChEMBL, and share of detected names whose database structure matches the prediction |
| Speed | Seconds per image or per detected structure |

Invalid predictions count as wrong in all accuracy rates. Structures that InChI cannot represent (Markush drawings with `*`) are compared by canonical SMILES instead.

## Results

MolScribe on a **500-image seeded sample** (seed 42) of the BioVista OCSR task (8,860 labelled images), run on a Google Colab T4 GPU. Full outputs: [`results/images_default/`](results/images_default/) (`predictions.csv` has every prediction, confidence score and per-image metric).

| Structure type | n | Valid SMILES | Exact match | Connectivity match | Stereo-only errors | Mean Tanimoto |
|---|---|---|---|---|---|---|
| Full | 175 | 93.7% | **73.7%** | 75.4% | 1.7% | 0.877 |
| Chiral | 106 | 94.3% | **51.9%** | 63.2% | 11.3% | 0.867 |
| Markush | 219 | 95.4% | **19.2%** | 20.1% | 0.9% | 0.587 |
| **All** | **500** | **94.6%** | **45.2%** | **48.6%** | **3.4%** | **0.748** |

**Comparison with published results.** The BioMiner authors report MolScribe accuracy on BioVista of 0.703 (full), 0.481 (chiral), 0.156 (Markush) and 0.455 overall (BioMiner repository). This benchmark's figures (0.737, 0.519, 0.192, 0.452) are within a few points on every category, which supports both the pipeline and the label-to-image matching.

**Key findings**

- **Ordinary structures are read well.** Nearly three in four full structures are reproduced exactly, and the median Tanimoto of valid predictions is 1.0.
- **Stereochemistry is the main error on chiral molecules.** 11.3% of chiral images have the right skeleton but wrong stereo, and a further 22.6% are near misses (Tanimoto ≥ 0.85).
- **Markush (R-group) drawings are the weak point.** Only 19.2% are exact; MolScribe often reads an "R" label as a carbon fragment instead of an attachment point. Since Markush structures make up 44% of this sample, they pull the overall score down sharply.
- **Confidence is a useful error signal.** Correct predictions average 0.87 confidence versus 0.73 for wrong ones, so low-confidence outputs can be routed to human review.
- **Most outputs are real compounds.** 74.7% of the 395 valid, InChI-representable predictions exist in PubChem (Markush outputs cannot be looked up by InChIKey).
- **Speed:** 0.64 s per image on a T4 GPU, about 1.6 hours for the full 8,860-image set.

**Reproduce:** open the Colab notebook, keep `LIMIT = None`, and *Run all* (about 10 minutes including PubChem checks).

## Reproducibility notes

- **No hardcoded paths or credentials.** All paths come from the config and resolve from the repo root; the public APIs need no keys.
- **Seeded sampling.** The image subset is drawn with `seed: 42`, so reruns score the same images.
- **Cached API calls.** PubChem/ChEMBL responses are cached in `results/cache/api_cache.json`, which keeps reruns consistent and respects rate limits.
- **Run records.** Each run saves its config, command, git commit and package versions.
- **Data label format.** The BioVista loader auto-detects the label file and columns. If detection fails, it prints the files and columns it found; set `labels_file`, `image_col` and `smiles_col` under `datasets.images` in the config.

## Sharing this repository

The repo is designed so that anyone with the link can run it:

- **One-time setup by the owner:** after creating the GitHub repo, run `python scripts/set_github_user.py <your-username>` and commit. This points the Colab badge and the notebook's clone URL at your repo. Keep the repository public (or add collaborators).
- **No local files required:** data downloads from public sources, model weights download from Hugging Face, and all paths are relative to the repo.
- **No credentials:** PubChem and ChEMBL need no API keys; Google Drive is used only if the user turns on `SAVE_TO_DRIVE`.
- **Every push is tested:** GitHub Actions runs the unit tests on each commit (see the tests badge).

## Known limitations

- Label linking in PDF mode is a heuristic (nearest short text block under the drawing) and needs a `decimer-segmentation` version that returns bounding boxes; otherwise labels are left empty and noted in the log.
- MolScribe handles Markush (R-group) structures poorly, so expect low scores in that category. This is reported separately rather than hidden in the average.
- Colab sessions are temporary. Results disappear when the session ends unless you download the zip (Step 7) or set `SAVE_TO_DRIVE = True`.
- The BioVista file is hosted on Google Drive, which occasionally rate-limits downloads; the notebook explains the manual fallback.
- Verification checks that a structure is a known compound. New compounds from a paper may be correct yet absent from PubChem/ChEMBL.

## References

- Kim, S., et al. (2023). PubChem 2023 update. *Nucleic Acids Research, 51*(D1), D1373–D1380.
- Mendez, D., et al. (2019). ChEMBL: Towards direct deposition of bioassay data. *Nucleic Acids Research, 47*(D1), D930–D940.
- Qian, Y., Guo, J., Tu, Z., Li, Z., Coley, C. W., & Barzilay, R. (2023). MolScribe: Robust molecular structure recognition with image-to-graph generation. *Journal of Chemical Information and Modeling, 63*(7), 1925–1934.
- Rajan, K., Brinkhaus, H. O., Sorokina, M., Zielesny, A., & Steinbeck, C. (2021). DECIMER-Segmentation: Automated extraction of chemical structure depictions from scientific literature. *Journal of Cheminformatics, 13*, 20.
- RDKit: Open-source cheminformatics. https://www.rdkit.org
- Rogers, D., & Hahn, M. (2010). Extended-connectivity fingerprints. *Journal of Chemical Information and Modeling, 50*(5), 742–754.
- Yan, J., et al. (2026). BioMiner: A multi-modal system for automated mining of protein-ligand bioactivity data from literature (arXiv:2604.21508). Benchmark data: https://github.com/jiaxianyan/BioMiner
