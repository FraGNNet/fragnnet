"""Tests for Tier 1 metal atom support (Hg, Sn, Pb)."""

import pytest

METAL_SMILES = [
    ("[CH3][Hg]Cl", "methylmercury chloride"),
    ("CC[Sn](CC)(CC)Cl", "triethyltin chloride"),
    ("CC[Pb](CC)(CC)CC", "tetraethyllead"),
]


@pytest.mark.parametrize("smiles,name", METAL_SMILES)
def test_metal_featurization(smiles, name):
    """Metal atoms should featurize without error."""
    from fragnnet.utils.frag_utils import extract_mol_info

    mol_d = extract_mol_info(smiles)
    assert mol_d is not None, f"featurization failed for {name}"


@pytest.mark.parametrize("smiles,name", METAL_SMILES)
def test_metal_in_element_counts(smiles, name):
    """Metal element should appear in element_counts with count > 0."""
    from fragnnet.utils.frag_utils import extract_mol_info

    non_organic = {"C", "H", "O", "N", "P", "S", "F", "Cl", "Br", "I", "Se", "Si", "As", "B"}
    mol_d = extract_mol_info(smiles)
    metal_elems = [a for a in mol_d["elems"] if a not in non_organic]
    assert len(metal_elems) > 0, f"no metal atom found in element_counts for {name}"


def test_metal_not_disconnected():
    """Normalize+Reionize must not disconnect C–Hg bond (no MetalDisconnector)."""
    from fragnnet.utils.data_utils import mol_from_smiles

    mol = mol_from_smiles("[CH3][Hg]Cl")
    assert mol is not None
    assert mol.GetNumAtoms() == 3, "C, Hg, Cl should be intact"
    assert mol.GetNumBonds() == 2, "C-Hg and Hg-Cl bonds should be present"


def test_element_whitelist_contains_metals():
    """ELEMENT_TO_VE must include all three Tier-1 metals."""
    from fragnnet.utils.frag_utils import ELEMENT_TO_VE

    for metal in ("Hg", "Sn", "Pb"):
        assert metal in ELEMENT_TO_VE, f"{metal} missing from ELEMENT_TO_VE"


def test_num_heavy_elements_matches_whitelist():
    """NUM_HEAVY_ELEMENTS must equal len(ELEMENT_TO_VE)."""
    from fragnnet.utils.frag_utils import ELEMENT_TO_VE, NUM_HEAVY_ELEMENTS

    assert NUM_HEAVY_ELEMENTS == len(ELEMENT_TO_VE)
