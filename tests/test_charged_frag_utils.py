"""Tests for H-cap / ve_arr correctness with formally charged atoms.

Covers the fix in extract_mol_info that adjusts ve_value for cationic atoms
with no H (e.g. [N+] in nitro, N-oxide) to prevent the H-cap from being
inflated by the formal charge.

Tests:
1. ve_arr for [N+] in nitro is 3 (neutral N valence), not 4.
2. H-cap for an isolated NO2 fragment (N-C bond cut) is 2, not 3 (pre-fix).
3. Full DAG for nitrobenzene does not contain the phantom H3NO2 formula.
4. Protonated amine ([NH3+] in zwitterion) is unaffected: ve remains 4.
5. N-oxide ([N+][O-]) N+ ve is adjusted from 4 to 3.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from fragnnet.frag.compute_frags import update_bonds
from fragnnet.utils.frag_utils import compute_cc_h_cap, compute_dags, extract_mol_info

_NITROBENZENE = "c1ccc([N+](=O)[O-])cc1"


@pytest.fixture(scope="module")
def nitrobenzene_mol_d() -> dict:
    """Parsed mol_d for nitrobenzene, shared across tests 1–3."""
    return extract_mol_info(_NITROBENZENE)


# ---------------------------------------------------------------------------
# 1. ve_arr for [N+] in nitro is adjusted from 4 to 3
# ---------------------------------------------------------------------------


def test_nitro_nitrogen_ve_is_three(nitrobenzene_mol_d):
    """[N+] in nitro group must have ve=3 after formal-charge adjustment.

    Without fix: GetTotalValence()=4 → ve=4.
    With fix: FC=+1, Hs=0 → ve = 4 - 1 = 3.
    """
    mol_d = nitrobenzene_mol_d
    n_idx = next(i for i, e in enumerate(mol_d["elems"]) if e == "N")
    assert mol_d["ve_arr"][n_idx] == 3, (
        f"Expected ve=3 for [N+] in nitro, got {mol_d['ve_arr'][n_idx]}"
    )


# ---------------------------------------------------------------------------
# 2. H-cap for isolated NO2 fragment is 2 (reduced from pre-fix value of 3)
# ---------------------------------------------------------------------------


def test_h_cap_isolated_no2_fragment(nitrobenzene_mol_d):
    """After cutting the N-C(ring) bond, the NO2 fragment H-cap must be 2.

    Fragment atoms: {N+, =O, O-}.  With fix:
      N+:  ve=3, sbond=2 → diff=1
      =O:  ve=2, sbond=1 → diff=1  (residual pi-bond excess, same as any C=O)
      O-:  ve=1, sbond=1 → diff=0
      cap = 1 + 1 + 0 = 2

    Before fix N+ contributed diff=2 (ve=4), giving cap=3.
    The remaining +1 from =O is an inherent property of the loose-cap design
    (shared with all double-bonded atoms, not specific to formal charges).
    """
    mol_d = nitrobenzene_mol_d
    elems, ve = mol_d["elems"], mol_d["ve_arr"]

    # Identify nitro atoms from elems + ve (no second RDKit parse needed):
    #   N+ → unique N in nitrobenzene
    #   =O → O with ve=2 (neutral double-bond O)
    #   O- → O with ve=1 (anionic single-bond O; GetTotalValence()=1, FC unchanged)
    n_idx = next(i for i, e in enumerate(elems) if e == "N")
    o_double_idx = next(i for i, e in enumerate(elems) if e == "O" and ve[i] == 2)
    o_minus_idx = next(i for i, e in enumerate(elems) if e == "O" and ve[i] == 1)

    no2_atom_ids = np.array([n_idx, o_double_idx, o_minus_idx], dtype=np.int32)

    # Recompute sbond_arr for this fragment only (mirrors compute_dags internals)
    updated_sbond, _ = update_bonds(
        no2_atom_ids,
        mol_d["sbond_arr"],
        mol_d["bond_mask_arr"],
        mol_d["bonds"],
        mol_d["atoms_to_bonds"],
    )

    h_cap = compute_cc_h_cap(no2_atom_ids, mol_d["ve_arr"], updated_sbond, 0)
    assert h_cap == 2, (
        f"NO2 fragment H-cap should be 2 after fix (was 3 before); got {h_cap}"
    )


# ---------------------------------------------------------------------------
# 3. Full DAG for nitrobenzene has no phantom H3NO2 formula
# ---------------------------------------------------------------------------


def test_nitrobenzene_dag_no_h3no2(nitrobenzene_mol_d):
    """Nitrobenzene DAG must not contain the phantom H3NO2 fragment formula.

    H3NO2 arises when the N+ valence inflation is not corrected (cap=3 → H=3
    for isolated NO2). After the fix cap=2, so only H=0,1,2 are valid and
    H3NO2 is absent.
    """
    dag_d = compute_dags(
        nitrobenzene_mol_d,
        max_depth=3,
        h_prior=False,
        max_h_transfer=4,
        frag_max_time=30,
        isotopes=False,
        nb_isomorphic=True,
        wl_max_iterations=3,
    )

    all_formulas = set(dag_d["idx_to_formula"].values())
    assert "H3NO2" not in all_formulas, (
        "Phantom formula H3NO2 found in DAG — N+ ve adjustment not applied"
    )


# ---------------------------------------------------------------------------
# 4. Protonated amine in a zwitterion ([NH3+]) is unaffected: ve stays 4
# ---------------------------------------------------------------------------


def test_protonated_amine_ve_unaffected():
    """[NH3+] with 3 H atoms must NOT have its ve adjusted.

    Glycine zwitterion [NH3+]CC(=O)[O-] has net charge 0.
    The N+ has FC=+1 but Hs=3, so the guard (cur_num_hs == 0) blocks the
    adjustment → ve stays at GetTotalValence() = 4, correctly reflecting
    the 3 H bonds plus 1 heavy bond.

    The mol is passed directly to skip ml_standardize, which would neutralize
    the zwitterion to NCC(=O)O before we can test the charged form.
    """
    mol = Chem.MolFromSmiles("[NH3+]CC(=O)[O-]")
    mol_d = extract_mol_info(mol)

    n_idx = next(i for i, e in enumerate(mol_d["elems"]) if e == "N")
    assert mol_d["ve_arr"][n_idx] == 4, (
        f"[NH3+] ve must stay 4 (3H + 1 heavy bond = 4 total valence), got {mol_d['ve_arr'][n_idx]}"
    )


# ---------------------------------------------------------------------------
# 5. N-oxide ([N+][O-]) N+ ve is adjusted from 4 to 3
# ---------------------------------------------------------------------------


def test_n_oxide_nitrogen_ve_is_three():
    """Pyridine N-oxide: [n+]([O-]) nitrogen must have ve=3 after adjustment.

    The N+ has FC=+1 and no H.  GetTotalValence()=4 is reduced to 3 by the fix.
    The mol is passed directly to bypass the standardizer.
    """
    mol = Chem.MolFromSmiles("c1cc[n+]([O-])cc1")
    mol_d = extract_mol_info(mol)

    n_idx = next(i for i, e in enumerate(mol_d["elems"]) if e == "N")
    assert mol_d["ve_arr"][n_idx] == 3, (
        f"N-oxide [N+] ve must be 3, got {mol_d['ve_arr'][n_idx]}"
    )


# ---------------------------------------------------------------------------
# 6. [M]+ charged ions: net charge +1 molecules must featurize without error
# ---------------------------------------------------------------------------


_CATION_SMILES = [
    ("C[N+](C)(C)C", "tetramethylammonium"),  # quaternary N+, charge=1, no H on N
    ("C[N+](C)(C)CCO", "choline"),  # quaternary N+ with hydroxyl side-chain
    ("CC[S+](CC)CC", "triethylsulfonium"),  # sulfonium S+
]


@pytest.mark.parametrize("smiles,name", _CATION_SMILES)
def test_cation_featurizes_without_error(smiles, name):
    """Molecules with net charge +1 must pass extract_mol_info for [M]+ support."""
    mol = Chem.MolFromSmiles(smiles)
    mol_d = extract_mol_info(mol)
    assert mol_d is not None, f"featurization returned None for {name}"
    assert len(mol_d["elems"]) > 0, f"empty element list for {name}"


def test_cation_element_counts_correct():
    """Tetramethylammonium: 1 N and 4 C, no unexpected atoms."""
    mol = Chem.MolFromSmiles("C[N+](C)(C)C")
    mol_d = extract_mol_info(mol)
    assert mol_d["element_counts"]["N"] == 1
    assert mol_d["element_counts"]["C"] == 4


def test_cation_h_cap_is_nonnegative():
    """H-cap for any fragment of a charge-1 molecule must be >= 0."""
    from fragnnet.utils.frag_utils import compute_cc_h_cap

    mol = Chem.MolFromSmiles("C[N+](C)(C)C")
    mol_d = extract_mol_info(mol)
    all_atoms = list(range(len(mol_d["elems"])))
    import numpy as np

    atom_ids = np.array(all_atoms, dtype=np.int32)
    from fragnnet.frag.compute_frags import update_bonds

    updated_sbond, _ = update_bonds(
        atom_ids,
        mol_d["sbond_arr"],
        mol_d["bond_mask_arr"],
        mol_d["bonds"],
        mol_d["atoms_to_bonds"],
    )
    cap = compute_cc_h_cap(atom_ids, mol_d["ve_arr"], updated_sbond, 0)
    assert cap >= 0, f"H-cap is negative ({cap}) for tetramethylammonium"


def test_charge_2_raises():
    """Molecules with net charge > +1 must raise ValueError."""
    mol = Chem.MolFromSmiles("C[N+](C)(C)[N+](C)(C)C")
    import rdkit.Chem.rdmolops as rdmolops

    if rdmolops.GetFormalCharge(mol) > 1:
        with pytest.raises(ValueError, match="unsupported molecular charge"):
            extract_mol_info(mol)


def test_negative_charge_raises():
    """Molecules with net charge < 0 must raise ValueError."""
    mol = Chem.MolFromSmiles("CC(=O)[O-]")  # acetate, charge=-1
    import rdkit.Chem.rdmolops as rdmolops

    if rdmolops.GetFormalCharge(mol) < 0:
        with pytest.raises(ValueError, match="unsupported molecular charge"):
            extract_mol_info(mol)
