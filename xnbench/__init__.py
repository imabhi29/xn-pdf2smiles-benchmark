"""xnbench: reproducible benchmarking of PDF/image -> SMILES extraction.

Stack: PyMuPDF (text layer + rendering), DECIMER Segmentation (detection and
cropping), MolScribe (image -> SMILES), RDKit (validation), PubChem / ChEMBL
web APIs (verification), RDKit + pandas (evaluation).
"""
__version__ = "1.1.0"
