"""RDKit-backed validation and similarity tests."""
import pytest

pytest.importorskip("rdkit")
from xnbench.chem import structure_category, tanimoto, validate_smiles  # noqa: E402


def test_valid_and_canonical():
    a, b = validate_smiles("OCC"), validate_smiles("CCO")
    assert a.valid and b.valid
    assert a.canonical_smiles == b.canonical_smiles
    assert a.inchikey == b.inchikey == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"


def test_invalid_smiles():
    assert not validate_smiles("C1CC").valid      # unclosed ring
    assert not validate_smiles("").valid


def test_stereo_differs_only_in_full_key():
    r, s = validate_smiles("C[C@H](N)O"), validate_smiles("C[C@@H](N)O")
    assert r.key != s.key
    assert r.connectivity_key == s.connectivity_key


def test_tanimoto():
    assert tanimoto("CCO", "OCC") == pytest.approx(1.0)
    assert 0 < tanimoto("CCO", "CCCO") < 1
    assert tanimoto("CCO", "not_smiles") is None


def test_categories():
    assert structure_category("C[C@H](N)O") == "chiral"
    assert structure_category("*c1ccccc1") == "markush"
    assert structure_category("CCO") == "full"


def test_markush_placeholders_are_normalised():
    numbered, plain = validate_smiles("[1*]c1ccccc1"), validate_smiles("*c1ccccc1")
    assert numbered.key == plain.key


def test_markush_connectivity_ignores_stereo():
    with_stereo = validate_smiles("*[C@H](N)C(=O)O")
    without = validate_smiles("*C(N)C(=O)O")
    assert with_stereo.key != without.key
    assert with_stereo.connectivity_key == without.connectivity_key
