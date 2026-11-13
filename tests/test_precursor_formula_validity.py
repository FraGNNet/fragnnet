"""A precursor's neutral loss must not be forced onto fragments that never carried it."""

import pytest
import torch as th

from fragnnet.dataset.spec_mol_frag_dataset import SpecMolFragDataset
from fragnnet.utils.formula_utils import (
    PREC_DELTA_REMOVED_ELEMENTS,
    PREC_TYPE_TO_CMF_COMP_DIFF,
    PREC_TYPE_TO_COMP_DIFF,
    TOTAL_ATOM_COLUMN,
    prec_type_removed_counts,
    tracked_element_counts,
)
from fragnnet.utils.frag_utils import NODE_FEAT_TO_IDX, compute_dags, extract_mol_info

NODE_FEATS = ["cc", "base_formula", "depth", "h_formulae_idx", "cmf_h_formulae_idx"]
# Salicylic acid carries exactly one COOH, so only some fragments can lose CO2. It has no
# nitrogen at all, which makes it the wrong probe for an ammonia-loss precursor: every channel
# is rejected and a test that only walks surviving channels would pass without asserting
# anything. Phenylalanine supplies the nitrogen, so both losses are exercised for real.
SALICYLIC_ACID = "OC(=O)c1ccccc1O"
PHENYLALANINE = "NC(Cc1ccccc1)C(=O)O"
# Precursors that subtract a neutral loss from every enumerated fragment.
LOSS_PRECS = ("[M-H-CO2]-", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+H-NH3]+", "[M-H-H2O]-", "[M-H]-")
# Precursors that only add a carrier, so no fragment can be invalidated.
ADDITIVE_PRECS = ("[M+H]+", "[M+Na]+", "[M+Cl]-", "[M+K]+", "[M+NH4]+")


def _dag(smiles: str) -> dict:
    return compute_dags(extract_mol_info(smiles), 2, True, 2, 100)


@pytest.fixture(scope="module")
def frag_entry():
    return _dag(SALICYLIC_ACID)


@pytest.fixture(scope="module")
def nitrogen_frag_entry():
    return _dag(PHENYLALANINE)


def _make_dataset(include_cmf: bool) -> SpecMolFragDataset:
    dataset = SpecMolFragDataset.__new__(SpecMolFragDataset)
    dataset.frag_params = {
        "pyg": True,
        "pyg_node_feats": NODE_FEATS,
        "pyg_edge_feats": [],
        "pyg_edges": False,
        "pyg_bigraph": False,
        "formula_peak_probs": True,
        "formula_peak_mzs": True,
        "num_isotopes": 1,
        "formula_str": True,
        "include_cmf": include_cmf,
    }
    return dataset


def _referenced_rows(frag_data: dict, feature: str) -> set[int]:
    """Formula rows the node features actually point at, excluding the NULL slot."""
    node_feat_idxs = frag_data["frag_pyg"].node_feat_idxs[0]
    position = NODE_FEAT_TO_IDX[feature]
    block = frag_data["frag_pyg"].x[
        :, int(node_feat_idxs[position]) : int(node_feat_idxs[position + 1])
    ]
    return {int(v) for v in block.flatten().tolist() if v != 0}


def _apply(frag_entry, prec_type: str, include_cmf: bool = False) -> dict:
    dataset = _make_dataset(include_cmf)
    frag_data = dataset._process_neutral_frag(frag_entry)
    return dataset._apply_fragment_mz_change(
        frag_data, prec_type, frag_data["frag_formula_peak_mzs"], clone_pyg=True
    )


# ---------------------------------------------------------------- count helpers


def test_tracked_counts_include_total_atoms():
    counts = tracked_element_counts("C2H1O2")
    expected = {"C": 2, "H": 1, "O": 2}
    for position, element in enumerate(PREC_DELTA_REMOVED_ELEMENTS):
        assert counts[position] == expected.get(element, 0), element
    assert counts[TOTAL_ATOM_COLUMN] == 5


def test_tracked_counts_of_null_formula_are_zero():
    assert tracked_element_counts("") == (0,) * (TOTAL_ATOM_COLUMN + 1)


@pytest.mark.parametrize("prec", ADDITIVE_PRECS)
def test_additive_precursors_need_no_validity_check(prec):
    assert prec_type_removed_counts(prec) == {}


def test_removed_counts_state_the_minimum_a_formula_needs():
    removed = prec_type_removed_counts("[M-H-CO2]-")  # H-1 C-1 O-2
    by_element = {
        PREC_DELTA_REMOVED_ELEMENTS[p]: n for p, n in removed.items() if p != TOTAL_ATOM_COLUMN
    }
    assert by_element == {"C": 1, "H": 1, "O": 2}
    # Four atoms leave, so a survivor needs five to keep one.
    assert removed[TOTAL_ATOM_COLUMN] == 5


# ---------------------------------------------------------------- emitted support


@pytest.mark.parametrize("prec", LOSS_PRECS + ADDITIVE_PRECS)
@pytest.mark.parametrize("smiles", [SALICYLIC_ACID, PHENYLALANINE])
def test_no_referenced_formula_yields_an_impossible_ion(request, smiles, prec):
    """Every referenced channel must keep non-negative counts and at least one atom."""
    entry = request.getfixturevalue(
        "frag_entry" if smiles == SALICYLIC_ACID else "nitrogen_frag_entry"
    )
    frag_data = _apply(entry, prec)
    formulas = list(entry["idx_to_formula"].values())
    delta = PREC_TYPE_TO_COMP_DIFF[prec]

    rows = _referenced_rows(frag_data, "h_formulae_idx")
    # Only an unsuppliable loss may empty the support; otherwise the loop below is vacuous.
    if not rows:
        assert prec == "[M+H-NH3]+" and smiles == SALICYLIC_ACID, f"{prec} kept no channel at all"
        return
    for row in rows:
        counts = tracked_element_counts(formulas[row])
        by_element = dict(zip(PREC_DELTA_REMOVED_ELEMENTS, counts))
        for element, change in delta.items():
            assert by_element.get(element, 0) + change >= 0, (prec, formulas[row], element)
        assert counts[TOTAL_ATOM_COLUMN] + sum(delta.values()) >= 1, (prec, formulas[row])


def test_ammonia_loss_needs_a_nitrogen_bearing_fragment(frag_entry, nitrogen_frag_entry):
    """[M+H-NH3]+ can only apply where the fragment actually carries the nitrogen."""
    no_nitrogen = _referenced_rows(_apply(frag_entry, "[M+H-NH3]+"), "h_formulae_idx")
    assert no_nitrogen == set(), "a molecule without nitrogen cannot lose ammonia"

    with_nitrogen = _apply(nitrogen_frag_entry, "[M+H-NH3]+")
    kept = _referenced_rows(with_nitrogen, "h_formulae_idx")
    total = len(nitrogen_frag_entry["idx_to_formula"]) - 1
    assert 0 < len(kept) < total, "only the nitrogen-bearing fragments may survive"

    formulas = list(nitrogen_frag_entry["idx_to_formula"].values())
    nitrogen_position = PREC_DELTA_REMOVED_ELEMENTS.index("N")
    assert all(tracked_element_counts(formulas[row])[nitrogen_position] >= 1 for row in kept)


@pytest.mark.parametrize("prec", ADDITIVE_PRECS)
def test_additive_precursors_keep_every_channel(frag_entry, prec):
    """The adducts used in training must be completely unaffected by the validity guard."""
    kept = _referenced_rows(_apply(frag_entry, prec), "h_formulae_idx")
    assert len(kept) == len(frag_entry["idx_to_formula"]) - 1  # all but the NULL row


def test_co2_loss_is_rejected_for_fragments_without_the_carboxyl(frag_entry):
    """The CO2 left the precursor once; fragments that never held it cannot lose it."""
    all_rows = _referenced_rows(_apply(frag_entry, "[M-H]-"), "h_formulae_idx")
    co2_rows = _referenced_rows(_apply(frag_entry, "[M-H-CO2]-"), "h_formulae_idx")
    assert co2_rows < all_rows, "CO2 loss must not be applicable to every fragment"

    formulas = frag_entry["idx_to_formula"]
    rejected = all_rows - co2_rows
    assert any(tracked_element_counts(formulas[row])[TOTAL_ATOM_COLUMN] > 0 for row in rejected)


@pytest.mark.parametrize("prec", LOSS_PRECS)
def test_emitted_mz_values_stay_positive(nitrogen_frag_entry, prec):
    """Blanket neutral-loss subtraction used to push small fragments below zero m/z.

    Uses the nitrogen-bearing molecule so every loss precursor, ammonia included, retains
    channels to check.
    """
    frag_data = _apply(nitrogen_frag_entry, prec)
    rows = sorted(_referenced_rows(frag_data, "h_formulae_idx"))
    assert rows, prec
    mzs = frag_data["frag_formula_peak_mzs"][rows, 0]
    assert float(mzs.min()) > 0.0, prec


def test_cmf_channels_get_their_own_validity(frag_entry):
    """CMF carries a different delta, so its channels are judged separately from CRF."""
    frag_data = _apply(frag_entry, "[M+Cl]-", include_cmf=True)
    formulas = list(frag_entry["idx_to_formula"].values())
    num_formulae = len(formulas)
    delta = PREC_TYPE_TO_CMF_COMP_DIFF["[M+Cl]-"]  # H-1
    assert any(count < 0 for count in delta.values()), "expected a removing CMF delta"

    for row in _referenced_rows(frag_data, "cmf_h_formulae_idx"):
        # CMF indices address the appended half of the doubled formula arrays.
        neutral_row = row - num_formulae + 1
        counts = dict(
            zip(PREC_DELTA_REMOVED_ELEMENTS, tracked_element_counts(formulas[neutral_row]))
        )
        for element, change in delta.items():
            assert counts.get(element, 0) + change >= 0, (formulas[neutral_row], element)


def test_validity_guard_does_not_mutate_cached_graph(frag_entry):
    """Masking is precursor-specific, so it must not reach the cached per-molecule graph.

    Mirrors the preprocess path, which shallow-copies the cached frag_data per spectrum.
    """
    dataset = _make_dataset(include_cmf=False)
    cached = dataset._process_neutral_frag(frag_entry)
    cached_pyg = cached["frag_pyg"]
    before = cached_pyg.x.clone()

    per_spectrum = cached.copy()
    applied = dataset._apply_fragment_mz_change(
        per_spectrum, "[M-H-CO2]-", per_spectrum["frag_formula_peak_mzs"], clone_pyg=True
    )

    assert th.equal(cached_pyg.x, before), "cached graph was modified in place"
    assert not th.equal(applied["frag_pyg"].x, before), "expected channels to be masked"
