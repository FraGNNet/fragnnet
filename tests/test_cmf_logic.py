import pytest
import torch as th

from fragnnet.dataset.spec_mol_frag_dataset import SpecMolFragDataset
from fragnnet.utils.formula_utils import PREC_TYPE_TO_CMF_MASS_DIFF, PREC_TYPE_TO_MASS_DIFF


class _FakePyG:
    def __init__(self, x: th.Tensor, node_feat_idxs: th.Tensor):
        # minimal attributes used by _apply_fragment_mz_change
        self.x = x
        # code expects node_feat_idxs[0] to be indexable
        self.node_feat_idxs = [node_feat_idxs]


def test_cmf_excluded_precursor_crf_only():
    obj = object.__new__(SpecMolFragDataset)
    obj.frag_params = {"include_cmf": True, "pyg_node_feats": []}

    neutral = th.tensor([[10.0, 11.0], [20.0, 21.0]])
    node_feat_idxs = th.tensor([0, 1, 2, 5, 7, 8, 11, 13])
    total_width = int(node_feat_idxs[-1].item())
    x = th.zeros((1, total_width), dtype=th.int64)
    fake_pyg = _FakePyG(x, node_feat_idxs)
    frag_data = {"frag_formula_peak_mzs": neutral.clone(), "frag_pyg": fake_pyg}
    prec = "[M-H]-"

    res = SpecMolFragDataset._apply_fragment_mz_change(
        obj, frag_data, prec, neutral, clone_pyg=False
    )
    expected = neutral + PREC_TYPE_TO_MASS_DIFF[prec]

    assert th.allclose(res["frag_formula_peak_mzs"], expected)
    assert res["frag_formula_peak_mzs"].shape == expected.shape


def test_cmf_applied_updates_peaks_and_pyg():
    obj = object.__new__(SpecMolFragDataset)
    # enable CMF and include cmf node feat name
    obj.frag_params = {
        "include_cmf": True,
        "pyg_node_feats": ["cmf_h_formulae_idx"],
        "formula_peak_mzs": True,
        "formula_peak_probs": True,
    }

    # neutral peaks: 3 formulas x 2 isotopes
    neutral = th.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    num_formulae = neutral.shape[0]

    # simple probs (normalized rows)
    probs = th.tensor([[0.5, 0.5], [0.6, 0.4], [0.4, 0.6]], dtype=th.float32)

    # build fake frag_pyg
    # construct node_feat_idxs such that h_formulae_idx at idx=3 has width 2 and cmf at idx=6 has width 2
    node_feat_idxs = th.tensor([0, 1, 2, 5, 7, 8, 11, 13])
    total_width = int(node_feat_idxs[-1].item())
    num_nodes = 2
    x = th.zeros((num_nodes, total_width), dtype=th.int64)

    # place crf formula matrix in the crf slice (node_feat_idxs[3]:node_feat_idxs[4]) width=2
    crf_start = int(node_feat_idxs[3].item())
    crf_end = int(node_feat_idxs[4].item())
    crf_width = crf_end - crf_start
    # set some non-zero indices to indicate possible formula matches
    x[:, crf_start:crf_end] = th.tensor([[0, 1], [2, 0]], dtype=th.int64)

    fake_pyg = _FakePyG(x, node_feat_idxs)

    frag_data = {
        "frag_formula_peak_mzs": neutral.clone(),
        "frag_formula_peak_probs": probs.clone(),
        "frag_pyg": fake_pyg,
    }

    prec = "[M+Na]+"  # this precursor should allow CMF (not in exclusion list)

    res = SpecMolFragDataset._apply_fragment_mz_change(
        obj, frag_data, prec, neutral, clone_pyg=False
    )

    # Check peaks: first block == neutral + CRF mass diff
    crf_expected = neutral + PREC_TYPE_TO_MASS_DIFF[prec]
    assert th.allclose(res["frag_formula_peak_mzs"][:num_formulae], crf_expected)

    # Check CMF peaks appended: neutral[1:] + CMF mass diff
    cmf_expected = neutral[1:] + PREC_TYPE_TO_CMF_MASS_DIFF[prec]
    assert th.allclose(res["frag_formula_peak_mzs"][num_formulae:], cmf_expected)

    # Check frag_pyg.x cmf slice updated as per logic
    cmf_start = int(node_feat_idxs[6].item())
    cmf_end = int(node_feat_idxs[7].item())
    crf_matrix = fake_pyg.x[:, crf_start:crf_end]
    cmf_matrix = res["frag_pyg"].x[:, cmf_start:cmf_end]

    # expected: where crf != 0 -> crf + num_formulae - 1, else crf
    expected_cmf = th.where(crf_matrix != 0, crf_matrix + num_formulae - 1, crf_matrix)
    assert th.equal(cmf_matrix, expected_cmf)


class _FakePyGClone(_FakePyG):
    def clone(self):
        # return a shallow-wrapper copy with cloned tensor for x
        # pass the raw tensor for node_feat_idxs (constructor will wrap it in a list)
        return _FakePyGClone(self.x.clone(), self.node_feat_idxs[0])


def test_cmf_exclusion_protonated_precursor_no_cmf():
    obj = object.__new__(SpecMolFragDataset)
    obj.frag_params = {"include_cmf": True, "pyg_node_feats": []}

    neutral = th.tensor([[5.0, 6.0], [15.0, 16.0]])
    frag_data = {"frag_formula_peak_mzs": neutral.clone()}
    prec = "[M+H]+"

    res = SpecMolFragDataset._apply_fragment_mz_change(
        obj, frag_data, prec, neutral, clone_pyg=False
    )
    expected = neutral + PREC_TYPE_TO_MASS_DIFF[prec]
    assert th.allclose(res["frag_formula_peak_mzs"], expected)


def test_unknown_precursor_raises_keyerror():
    obj = object.__new__(SpecMolFragDataset)
    obj.frag_params = {
        "include_cmf": True,
        "pyg_node_feats": ["cmf_h_formulae_idx"],
        "formula_peak_probs": False,
    }
    neutral = th.tensor([[1.0, 2.0]])
    frag_data = {"frag_formula_peak_mzs": neutral.clone()}
    with pytest.raises(KeyError):
        SpecMolFragDataset._apply_fragment_mz_change(
            obj, frag_data, "UNKNOWN_PREC", neutral, clone_pyg=False
        )


def test_crf_matrix_zero_results_in_zero_cmf_slice():
    obj = object.__new__(SpecMolFragDataset)
    obj.frag_params = {
        "include_cmf": True,
        "pyg_node_feats": ["cmf_h_formulae_idx"],
        "formula_peak_mzs": True,
        "formula_peak_probs": False,
    }
    neutral = th.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])

    node_feat_idxs = th.tensor([0, 1, 2, 5, 7, 8, 11, 13])
    total_width = int(node_feat_idxs[-1].item())
    x = th.zeros((1, total_width), dtype=th.int64)
    fake_pyg = _FakePyG(x, node_feat_idxs)

    frag_data = {"frag_formula_peak_mzs": neutral.clone(), "frag_pyg": fake_pyg}
    prec = "[M+Na]+"
    res = SpecMolFragDataset._apply_fragment_mz_change(
        obj, frag_data, prec, neutral, clone_pyg=False
    )

    cmf_start = int(node_feat_idxs[6].item())
    cmf_end = int(node_feat_idxs[7].item())
    cmf_matrix = res["frag_pyg"].x[:, cmf_start:cmf_end]
    assert th.equal(cmf_matrix, th.zeros_like(cmf_matrix))


def test_clone_pyg_returns_new_object_and_original_unchanged():
    obj = object.__new__(SpecMolFragDataset)
    obj.frag_params = {
        "include_cmf": True,
        "pyg_node_feats": ["cmf_h_formulae_idx"],
        "formula_peak_mzs": True,
        "formula_peak_probs": False,
    }
    neutral = th.tensor([[10.0, 11.0], [20.0, 21.0]])

    node_feat_idxs = th.tensor([0, 1, 2, 5, 7, 8, 11, 13])
    total_width = int(node_feat_idxs[-1].item())
    x = th.zeros((1, total_width), dtype=th.int64)
    # place a non-zero in crf slice so the cmf logic will modify the clone
    crf_start = int(node_feat_idxs[3].item())
    crf_end = int(node_feat_idxs[4].item())
    x[:, crf_start:crf_end] = th.tensor([[1, 0]], dtype=th.int64)

    fake_pyg = _FakePyGClone(x, node_feat_idxs)
    frag_data = {"frag_formula_peak_mzs": neutral.clone(), "frag_pyg": fake_pyg}
    prec = "[M+Na]+"

    original_x_before = fake_pyg.x.clone()
    res = SpecMolFragDataset._apply_fragment_mz_change(
        obj, frag_data, prec, neutral, clone_pyg=True
    )

    # Ensure returned pyg is not the same object as original
    assert res["frag_pyg"].x is not fake_pyg.x
    # original unchanged
    assert th.equal(fake_pyg.x, original_x_before)


def test_single_formula_no_cmf_appended():
    obj = object.__new__(SpecMolFragDataset)
    obj.frag_params = {
        "include_cmf": True,
        "pyg_node_feats": ["cmf_h_formulae_idx"],
        "formula_peak_mzs": True,
        "formula_peak_probs": False,
    }
    neutral = th.tensor([[42.0, 43.0]])
    node_feat_idxs = th.tensor([0, 1, 2, 5, 7, 8, 11, 13])
    total_width = int(node_feat_idxs[-1].item())
    x = th.zeros((1, total_width), dtype=th.int64)
    fake_pyg = _FakePyG(x, node_feat_idxs)
    frag_data = {"frag_formula_peak_mzs": neutral.clone(), "frag_pyg": fake_pyg}
    prec = "[M+Na]+"

    res = SpecMolFragDataset._apply_fragment_mz_change(
        obj, frag_data, prec, neutral, clone_pyg=False
    )
    # With a single formula, CMF peaks are neutral[1:] -> empty, so result should equal CRF only
    expected = neutral + PREC_TYPE_TO_MASS_DIFF[prec]
    assert th.allclose(res["frag_formula_peak_mzs"], expected)


def test_multiple_precursors_shapes():
    # test several precursor types and a multi-isotope/multi-formula neutral matrix
    precs = ["[M+H]+", "[M+Na]+", "[M+K]+", "[M-H]-", "[M+Cl]-", "[M]+"]
    # 4 formulas, 3 isotopes
    neutral = th.tensor(
        [[10.0, 10.1, 10.2], [20.0, 20.1, 20.2], [30.0, 30.1, 30.2], [40.0, 40.1, 40.2]]
    )
    num_formulae = neutral.shape[0]

    # common fake pyg layout
    # ensure CRF and CMF feature slices have the same width
    node_feat_idxs = th.tensor([0, 1, 2, 5, 7, 8, 10, 12])
    total_width = int(node_feat_idxs[-1].item())
    num_nodes = 3
    x = th.zeros((num_nodes, total_width), dtype=th.int64)

    # set some non-zero entries in the CRF slice (node_feat_idxs[3]:node_feat_idxs[4])
    crf_start = int(node_feat_idxs[3].item())
    crf_end = int(node_feat_idxs[4].item())
    # set some non-zero entries in CRF slice (width = crf_end-crf_start)
    width = crf_end - crf_start
    pattern = th.tensor([[0, 1], [1, 0], [2, 2]], dtype=th.int64)
    x[:, crf_start:crf_end] = pattern[:, :width]

    fake_pyg = _FakePyG(x, node_feat_idxs)

    for prec in precs:
        obj = object.__new__(SpecMolFragDataset)
        obj.frag_params = {
            "include_cmf": True,
            "pyg_node_feats": ["cmf_h_formulae_idx"],
            "formula_peak_mzs": True,
            "formula_peak_probs": False,
        }
        frag_data = {"frag_formula_peak_mzs": neutral.clone(), "frag_pyg": fake_pyg}

        res = SpecMolFragDataset._apply_fragment_mz_change(
            obj, frag_data, prec, neutral, clone_pyg=False
        )

        # determine if CMF should be applied according to code exclusion list
        excluded = prec in ["[M+H]+", "[M-H]-", "[M]+", "[M-]"]

        if excluded:
            # CRF only
            expected = neutral + PREC_TYPE_TO_MASS_DIFF[prec]
            assert th.allclose(res["frag_formula_peak_mzs"], expected)
            assert res["frag_formula_peak_mzs"].shape == neutral.shape
        else:
            # CMF appended
            crf_expected = neutral + PREC_TYPE_TO_MASS_DIFF[prec]
            cmf_expected = neutral[1:] + PREC_TYPE_TO_CMF_MASS_DIFF[prec]
            out = res["frag_formula_peak_mzs"]
            assert out.shape[0] == num_formulae + (num_formulae - 1)
            assert th.allclose(out[:num_formulae], crf_expected)
            assert th.allclose(out[num_formulae:], cmf_expected)
