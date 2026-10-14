"""Chemistry utilities built on RDKit: validation, identity keys, similarity.

Identity is compared with InChIKeys:
* full key        -> same molecule including stereochemistry ("exact match")
* first 14 chars  -> same atom connectivity, stereo ignored ("connectivity match")
Molecules that InChI cannot represent (e.g. Markush structures with ``*``
attachment points) fall back to canonical SMILES: isomeric SMILES for the
exact key, non-isomeric SMILES for the connectivity key. Numbered R-group
placeholders are normalised first, so "[1*]c1ccccc1" equals "*c1ccccc1".
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

RDLogger.DisableLog("rdApp.*")  # keep logs readable; failures are recorded per row


@dataclass
class ValidationResult:
    valid: bool
    canonical_smiles: str = ""
    inchikey: str = ""
    key: str = ""               # inchikey, or canonical SMILES if no InChIKey
    connectivity_key: str = ""  # first InChIKey block, or canonical SMILES
    formula: str = ""
    mol_weight: float | None = None
    heavy_atoms: int | None = None
    error: str = ""

    def as_dict(self, prefix: str = "") -> dict:
        return {f"{prefix}{k}": v for k, v in asdict(self).items()}


@lru_cache(maxsize=100_000)
def _mol(smiles: str):
    return Chem.MolFromSmiles(smiles) if smiles else None


def validate_smiles(smiles: str | None) -> ValidationResult:
    """Parse and sanitize a SMILES string and compute identifiers."""
    if not smiles or not isinstance(smiles, str):
        return ValidationResult(valid=False, error="empty")
    mol = _mol(smiles.strip())
    if mol is None:
        return ValidationResult(valid=False, error="rdkit_parse_failed")
    canonical = Chem.MolToSmiles(mol)
    try:
        inchikey = Chem.MolToInchiKey(mol) or ""
    except Exception:
        inchikey = ""
    generic, generic_flat = _generic_smiles(mol)
    return ValidationResult(
        valid=True,
        canonical_smiles=canonical,
        inchikey=inchikey,
        key=inchikey or generic,
        connectivity_key=inchikey.split("-")[0] if inchikey else generic_flat,
        formula=CalcMolFormula(mol),
        mol_weight=round(Descriptors.MolWt(mol), 3),
        heavy_atoms=mol.GetNumHeavyAtoms(),
    )


def _generic_smiles(mol) -> tuple[str, str]:
    """Canonical (isomeric, non-isomeric) SMILES with R-group labels removed."""
    copy = Chem.Mol(mol)
    for atom in copy.GetAtoms():
        if atom.GetAtomicNum() == 0:      # dummy atom: *, [1*], [*:1], R-groups
            atom.SetIsotope(0)
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(copy), Chem.MolToSmiles(copy, isomericSmiles=False)


@lru_cache(maxsize=8)
def _generator(radius: int, bits: int):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)


def tanimoto(smiles_a: str, smiles_b: str, radius: int = 2, bits: int = 2048) -> float | None:
    """Morgan-fingerprint Tanimoto similarity; None if either SMILES is invalid."""
    mol_a, mol_b = _mol(smiles_a or ""), _mol(smiles_b or "")
    if mol_a is None or mol_b is None:
        return None
    gen = _generator(radius, bits)
    return float(DataStructs.TanimotoSimilarity(gen.GetFingerprint(mol_a),
                                                gen.GetFingerprint(mol_b)))


def structure_category(smiles: str) -> str:
    """Bucket a ground-truth structure: markush, chiral or full."""
    s = smiles or ""
    if "*" in s or "[R" in s:
        return "markush"
    if "@" in s or "/" in s or "\\" in s:
        return "chiral"
    return "full"
