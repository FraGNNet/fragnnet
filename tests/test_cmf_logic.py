import pytest
import torch as th

from fragnnet.dataset.spec_mol_frag_dataset import SpecMolFragDataset
from fragnnet.utils.formula_utils import (
    ELECTRON_MASS,
    H_MASS,
    PREC_TYPE_TO_CMF_FORMULA_DIFF,
    PREC_TYPE_TO_CMF_MASS_DIFF,
    PREC_TYPE_TO_MASS_DIFF,
    cmf_duplicate_crf_h_shift,
)

# Adducts whose CMF ion is already enumerated as a CRF hydrogen shift, so CMF is skipped.
DEGENERATE_PRECS = ("[M+H]+", "[M-H]-")
# Adducts whose CMF ion is a distinct m/z, so the full H-shift range is written.
DISTINCT_PRECS = ("[M+Na]+", "[M+NH4]+", "[M+K]+", "[M+Cl]-")


class _FakePyG:
    def __init__(self, x: th.Tensor, node_feat_idxs: th.Tensor):
        # minimal attributes used by _apply_fragment_mz_change
        self.x = x
        # code expects node_feat_idxs[0] to be indexable
        self.node_feat_idxs = [node_feat_idxs]


def _make_obj(include_cmf: bool = True, with_cmf_feat: bool = True):
    obj = object.__new__(SpecMolFragDataset)
    feats = ["cmf_h_formulae_idx"] if with_cmf_feat else []
    obj.frag_params = {
        "include_cmf": include_cmf,
        "pyg_node_feats": feats,
        "formula_peak_mzs": True,
        "formula_peak_probs": False,
    }
    return obj


def _make_pyg(num_nodes: int, crf_values: th.Tensor):
    """Build a _FakePyG with crf_values in the h_formulae_idx slice (NODE_FEAT_TO_IDX=3)
    and zeros in cmf_h_formulae_idx (NODE_FEAT_TO_IDX=6).
    In production both CRF and CMF features have the same width (formula_node_feat_size),
    which is 1 + 2 * max_h_transfer, so widths here must be odd.
    """
    crf_width = crf_values.shape[1]
    assert crf_width % 2 == 1, "formula feature width must be 1 + 2 * max_h_transfer"
    cmf_width = crf_width  # always equal to crf_width in production DAGs
    # idxs: depth=1, cc=3, base=3, crf=crf_width, h_counts=crf_width, nb=1, cmf=cmf_width
    node_feat_idxs = th.tensor(
        [
            0,
            1,
            4,
            7,
            7 + crf_width,
            7 + 2 * crf_width,
            7 + 2 * crf_width + 1,
            7 + 2 * crf_width + 1 + cmf_width,
        ]
    )
    total_width = int(node_feat_idxs[-1].item())
    x = th.zeros((num_nodes, total_width), dtype=th.int64)
    crf_start = int(node_feat_idxs[3].item())
    x[:, crf_start : crf_start + crf_width] = crf_values
    return _FakePyG(x, node_feat_idxs)


def _cmf_slice(frag_pyg) -> th.Tensor:
    node_feat_idxs = frag_pyg.node_feat_idxs[0]
    cmf_start = int(node_feat_idxs[6].item())
    cmf_end = int(node_feat_idxs[7].item())
    return frag_pyg.x[:, cmf_start:cmf_end]


def _h_ladder(base_mz: float, max_h_transfer: int) -> th.Tensor:
    """Build a neutral peak column ordered like the DAG's h_delta columns.

    Row 0 is the NULL slot; the remaining rows follow the ``[0, -1, +1, -2, +2, ...]``
    ordering used by ``h_formulae_idx``, each offset from the base by whole hydrogens.
    """
    h_deltas = [0]
    for h in range(1, max_h_transfer + 1):
        h_deltas += [-h, h]
    rows = [0.0] + [base_mz + delta * H_MASS for delta in h_deltas]
    return th.tensor(rows, dtype=th.float64).unsqueeze(1)


# ---------------------------------------------------------------- degeneracy helper


@pytest.mark.parametrize(
    "prec, expected",
    [("[M+H]+", -1), ("[M-H]-", 1)],
)
def test_duplicate_shift_detects_direct_charge_adducts(prec, expected):
    """The direct-charge CMF state is the CRF state one hydrogen away."""
    assert cmf_duplicate_crf_h_shift(prec, max_h_transfer=4) == expected


@pytest.mark.parametrize("prec", DISTINCT_PRECS)
def test_duplicate_shift_none_for_distinct_adducts(prec):
    """Carrier swaps that are not a whole number of hydrogens stay distinct."""
    assert cmf_duplicate_crf_h_shift(prec, max_h_transfer=4) is None


def test_duplicate_shift_requires_reachable_shift():
    """Without hydrogen transfer the CRF ladder has no delta_h=-1 column to duplicate."""
    assert cmf_duplicate_crf_h_shift("[M+H]+", max_h_transfer=0) is None
    assert cmf_duplicate_crf_h_shift("[M+H]+", max_h_transfer=1) == -1


def test_duplicate_shift_none_for_adduct_without_cmf():
    assert cmf_duplicate_crf_h_shift("[M]+", max_h_transfer=4) is None


# ---------------------------------------------------------------- emitted m/z support


@pytest.mark.parametrize("prec", DEGENERATE_PRECS + DISTINCT_PRECS)
def test_no_duplicate_mz_in_emitted_support(prec):
    """No two emitted formula slots may land on the same m/z.

    Regression guard for the CMF/CRF degeneracy: for [M+H]+ the CMF state at delta_h=0 has
    the same formula and mass as the CRF state at delta_h=-1, so emitting both put two slots
    of the formula softmax on one peak.
    """
    max_h_transfer = 4
    neutral = _h_ladder(200.0, max_h_transfer)
    crf_width = 1 + 2 * max_h_transfer
    crf_values = th.arange(1, crf_width + 1, dtype=th.int64).unsqueeze(0)

    obj = _make_obj()
    frag_data = {
        "frag_formula_peak_mzs": neutral.clone(),
        "frag_pyg": _make_pyg(1, crf_values),
    }
    res = SpecMolFragDataset._apply_fragment_mz_change(obj, frag_data, prec, neutral)

    mzs = res["frag_formula_peak_mzs"][1:, 0]  # row 0 is the NULL slot
    gaps = th.abs(mzs.unsqueeze(0) - mzs.unsqueeze(1))
    gaps.fill_diagonal_(th.inf)
    assert gaps.min().item() > 1e-6, f"duplicate emitted m/z for {prec}"


@pytest.mark.parametrize("prec", DEGENERATE_PRECS)
def test_degenerate_adduct_emits_crf_only(prec):
    """A degenerate adduct falls back to the CRF-only support and writes no CMF columns."""
    obj = _make_obj()
    neutral = _h_ladder(150.0, max_h_transfer=1)
    crf_values = th.tensor([[1, 2, 3]], dtype=th.int64)
    frag_data = {
        "frag_formula_peak_mzs": neutral.clone(),
        "frag_pyg": _make_pyg(1, crf_values),
    }

    res = SpecMolFragDataset._apply_fragment_mz_change(obj, frag_data, prec, neutral)

    expected = neutral + PREC_TYPE_TO_MASS_DIFF[prec]
    assert th.allclose(res["frag_formula_peak_mzs"], expected)
    assert res["frag_formula_peak_mzs"].shape == neutral.shape
    cmf_matrix = _cmf_slice(res["frag_pyg"])
    assert th.equal(cmf_matrix, th.zeros_like(cmf_matrix))


def test_degenerate_adduct_keeps_cmf_without_h_transfer():
    """With max_h_transfer=0 the CMF ion is not enumerated by CRF, so it is kept."""
    obj = _make_obj()
    neutral = th.tensor([[100.0], [110.0]], dtype=th.float64)
    crf_values = th.tensor([[1]], dtype=th.int64)  # width 1 -> max_h_transfer = 0
    frag_data = {
        "frag_formula_peak_mzs": neutral.clone(),
        "frag_pyg": _make_pyg(1, crf_values),
    }

    res = SpecMolFragDataset._apply_fragment_mz_change(obj, frag_data, "[M+H]+", neutral)

    cmf_expected = neutral[1:] + PREC_TYPE_TO_CMF_MASS_DIFF["[M+H]+"]
    assert th.allclose(res["frag_formula_peak_mzs"][neutral.shape[0] :], cmf_expected)


def test_cmf_excluded_precursor_crf_only():
    """[M]+ has no CMF state; result should be CRF peaks only."""
    obj = _make_obj(include_cmf=True, with_cmf_feat=False)
    neutral = th.tensor([[10.0, 11.0], [20.0, 21.0]])
    prec = "[M]+"
    fake_pyg = _make_pyg(1, th.zeros((1, 3), dtype=th.int64))
    frag_data = {"frag_formula_peak_mzs": neutral.clone(), "frag_pyg": fake_pyg}

    res = SpecMolFragDataset._apply_fragment_mz_change(
        obj, frag_data, prec, neutral, clone_pyg=False
    )
    expected = neutral + PREC_TYPE_TO_MASS_DIFF[prec]

    assert th.allclose(res["frag_formula_peak_mzs"], expected)
    assert res["frag_formula_peak_mzs"].shape == expected.shape


def test_include_cmf_disabled_crf_only():
    obj = _make_obj(include_cmf=False)
    neutral = th.tensor([[10.0], [20.0], [30.0]])
    fake_pyg = _make_pyg(1, th.tensor([[1, 2, 3]], dtype=th.int64))
    frag_data = {"frag_formula_peak_mzs": neutral.clone(), "frag_pyg": fake_pyg}

    res = SpecMolFragDataset._apply_fragment_mz_change(obj, frag_data, "[M+Na]+", neutral)

    assert th.allclose(res["frag_formula_peak_mzs"], neutral + PREC_TYPE_TO_MASS_DIFF["[M+Na]+"])


def test_cmf_applied_updates_peaks_and_pyg():
    """[M+Na]+ (non-H adduct): CMF appends peaks with H mass_diff and fills all CRF cols."""
    obj = _make_obj()
    neutral = th.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    num_formulae = neutral.shape[0]
    probs = th.tensor([[0.5, 0.5], [0.6, 0.4], [0.4, 0.6]], dtype=th.float32)

    crf_values = th.tensor([[0, 1, 2], [2, 0, 1]], dtype=th.int64)
    fake_pyg = _make_pyg(2, crf_values)

    frag_data = {
        "frag_formula_peak_mzs": neutral.clone(),
        "frag_formula_peak_probs": probs.clone(),
        "frag_pyg": fake_pyg,
    }
    prec = "[M+Na]+"

    res = SpecMolFragDataset._apply_fragment_mz_change(
        obj, frag_data, prec, neutral, clone_pyg=False
    )

    crf_expected = neutral + PREC_TYPE_TO_MASS_DIFF[prec]
    assert th.allclose(res["frag_formula_peak_mzs"][:num_formulae], crf_expected)

    cmf_expected = neutral[1:] + PREC_TYPE_TO_CMF_MASS_DIFF[prec]
    assert th.allclose(res["frag_formula_peak_mzs"][num_formulae:], cmf_expected)

    # For [M+Na]+ all CMF columns should be filled (full H-shift range)
    cmf_matrix = _cmf_slice(res["frag_pyg"])
    expected_cmf = th.where(crf_values != 0, crf_values + num_formulae - 1, crf_values)
    assert th.equal(cmf_matrix, expected_cmf)


@pytest.mark.parametrize("prec", DISTINCT_PRECS)
def test_distinct_adducts_fill_all_cmf_columns(prec):
    """Every non-degenerate CMF adduct writes the full H-shift range."""
    obj = _make_obj()
    neutral = th.tensor([[10.0], [20.0], [30.0], [40.0]])
    num_formulae = neutral.shape[0]

    crf_values = th.tensor([[1, 2, 3]], dtype=th.int64)  # all non-zero
    fake_pyg = _make_pyg(1, crf_values)
    frag_data = {"frag_formula_peak_mzs": neutral.clone(), "frag_pyg": fake_pyg}

    SpecMolFragDataset._apply_fragment_mz_change(obj, frag_data, prec, neutral)

    cmf_matrix = _cmf_slice(frag_data["frag_pyg"])
    assert th.equal(cmf_matrix, crf_values + num_formulae - 1), prec


# ---------------------------------------------------------------- mass table guards


def test_cmf_mass_diffs_direct_charge():
    """[M+H]+ and [M-H]- CMF mass diffs are ±ELECTRON_MASS (direct charge, no H transfer)."""
    assert abs(PREC_TYPE_TO_CMF_MASS_DIFF["[M+H]+"] - (-ELECTRON_MASS)) < 1e-9
    assert abs(PREC_TYPE_TO_CMF_MASS_DIFF["[M-H]-"] - ELECTRON_MASS) < 1e-9


@pytest.mark.parametrize("prec", ["[M+H]+", "[M-H]-"])
def test_proton_carrier_precursors_migrate_to_a_direct_ion(prec):
    """A departing proton leaves nothing behind, so the fragment carries the charge itself.

    Handing these a proton instead would describe charge *retention*, not migration.
    """
    assert PREC_TYPE_TO_CMF_FORMULA_DIFF[prec] == ""
    charge = -1 if prec.endswith("-") else 1
    assert abs(PREC_TYPE_TO_CMF_MASS_DIFF[prec] + charge * ELECTRON_MASS) < 1e-9


@pytest.mark.parametrize(
    "prec", ["[M-H2O+H]+", "[M-2H2O+H]+", "[M+H-NH3]+", "[M-H-H2O]-", "[M-H-CO2]-"]
)
def test_in_source_loss_precursors_have_no_cmf_channel(prec):
    """Their charge is intrinsic to the loss ion, so no carrier remains to migrate away.

    Absence from the tables makes them emit CRF rows only, the same path as [M]+ and [M-].
    """
    assert prec in PREC_TYPE_TO_MASS_DIFF, f"{prec} must still be a usable precursor"
    assert prec not in PREC_TYPE_TO_CMF_MASS_DIFF
    assert prec not in PREC_TYPE_TO_CMF_FORMULA_DIFF


@pytest.mark.parametrize(
    "prec, expected",
    [
        ("[M+Na]+", "H1"),
        ("[M+K]+", "H1"),
        ("[M+NH4]+", "H1"),
        ("[M+Cl]-", "H-1"),
        ("[M+HCOOH-H]-", "H-1"),
        ("[M+CH3COO]-", "H-1"),
    ],
)
def test_discrete_adducts_migrate_to_the_conventional_carrier(prec, expected):
    """A detachable adduct leaves with the neutral and the fragment takes H / -H instead."""
    assert PREC_TYPE_TO_CMF_FORMULA_DIFF[prec] == expected


def test_no_adduct_has_degenerate_cmf_mass_diff():
    """No adduct may give CMF the same mass diff as CRF.

    CMF(F) == CRF(F) makes every CMF row an exact m/z duplicate of its own CRF row, so the
    CMF doubling adds no candidate m/z while still enlarging the formula softmax.
    """
    degenerate = [
        pt
        for pt, cmf in PREC_TYPE_TO_CMF_MASS_DIFF.items()
        if abs(cmf - PREC_TYPE_TO_MASS_DIFF[pt]) < 1e-9
    ]
    assert degenerate == [], f"CMF mass diff equals CRF mass diff for: {degenerate}"


def test_degenerate_adduct_set_is_exactly_the_direct_charge_pair():
    """Only the direct-charge adducts collide with a reachable CRF hydrogen shift."""
    degenerate = {
        pt
        for pt in PREC_TYPE_TO_CMF_MASS_DIFF
        if cmf_duplicate_crf_h_shift(pt, max_h_transfer=4) is not None
    }
    assert degenerate == set(DEGENERATE_PRECS)


# ---------------------------------------------------------------- misc behaviour


class _FakePyGClone(_FakePyG):
    def clone(self):
        return _FakePyGClone(self.x.clone(), self.node_feat_idxs[0])


def test_unknown_precursor_raises_keyerror():
    obj = _make_obj()
    neutral = th.tensor([[1.0, 2.0]])
    frag_data = {"frag_formula_peak_mzs": neutral.clone()}
    with pytest.raises(KeyError):
        SpecMolFragDataset._apply_fragment_mz_change(
            obj, frag_data, "UNKNOWN_PREC", neutral, clone_pyg=False
        )


def test_crf_matrix_zero_results_in_zero_cmf_slice():
    obj = _make_obj()
    neutral = th.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])

    crf_values = th.zeros((1, 3), dtype=th.int64)
    fake_pyg = _make_pyg(1, crf_values)

    frag_data = {"frag_formula_peak_mzs": neutral.clone(), "frag_pyg": fake_pyg}
    res = SpecMolFragDataset._apply_fragment_mz_change(
        obj, frag_data, "[M+Na]+", neutral, clone_pyg=False
    )

    cmf_matrix = _cmf_slice(res["frag_pyg"])
    assert th.equal(cmf_matrix, th.zeros_like(cmf_matrix))


def test_clone_pyg_returns_new_object_and_original_unchanged():
    obj = _make_obj()
    neutral = th.tensor([[10.0, 11.0], [20.0, 21.0]])

    crf_values = th.tensor([[1, 0, 2]], dtype=th.int64)
    fake_pyg = _FakePyGClone(
        _make_pyg(1, crf_values).x,
        _make_pyg(1, crf_values).node_feat_idxs[0],
    )
    original_x_before = fake_pyg.x.clone()
    frag_data = {"frag_formula_peak_mzs": neutral.clone(), "frag_pyg": fake_pyg}

    res = SpecMolFragDataset._apply_fragment_mz_change(
        obj, frag_data, "[M+Na]+", neutral, clone_pyg=True
    )

    assert res["frag_pyg"].x is not fake_pyg.x
    assert th.equal(fake_pyg.x, original_x_before)


def test_single_formula_no_cmf_appended():
    obj = _make_obj()
    neutral = th.tensor([[42.0, 43.0]])
    fake_pyg = _make_pyg(1, th.zeros((1, 3), dtype=th.int64))
    frag_data = {"frag_formula_peak_mzs": neutral.clone(), "frag_pyg": fake_pyg}
    prec = "[M+Na]+"

    res = SpecMolFragDataset._apply_fragment_mz_change(
        obj, frag_data, prec, neutral, clone_pyg=False
    )
    # neutral[1:] is empty, so CMF appends nothing
    expected = neutral + PREC_TYPE_TO_MASS_DIFF[prec]
    assert th.allclose(res["frag_formula_peak_mzs"], expected)


def test_multiple_precursors_shapes():
    precs = ["[M+H]+", "[M+Na]+", "[M+K]+", "[M-H]-", "[M+Cl]-", "[M]+"]
    neutral = th.tensor(
        [[10.0, 10.1, 10.2], [20.0, 20.1, 20.2], [30.0, 30.1, 30.2], [40.0, 40.1, 40.2]]
    )
    num_formulae = neutral.shape[0]

    for prec in precs:
        obj = _make_obj()
        crf_values = th.tensor([[0, 1, 2], [1, 0, 2], [2, 2, 1]], dtype=th.int64)
        fake_pyg = _make_pyg(3, crf_values)
        frag_data = {"frag_formula_peak_mzs": neutral.clone(), "frag_pyg": fake_pyg}

        res = SpecMolFragDataset._apply_fragment_mz_change(
            obj, frag_data, prec, neutral, clone_pyg=False
        )
        out = res["frag_formula_peak_mzs"]
        crf_expected = neutral + PREC_TYPE_TO_MASS_DIFF[prec]

        # [M]+ has no CMF state; [M+H]+/[M-H]- have one that CRF already enumerates.
        if prec == "[M]+" or prec in DEGENERATE_PRECS:
            assert th.allclose(out, crf_expected), prec
            assert out.shape == neutral.shape, prec
        else:
            cmf_expected = neutral[1:] + PREC_TYPE_TO_CMF_MASS_DIFF[prec]
            assert out.shape[0] == num_formulae + (num_formulae - 1), prec
            assert th.allclose(out[:num_formulae], crf_expected), prec
            assert th.allclose(out[num_formulae:], cmf_expected), prec
