"""Annotation columns must be reachable without asking for any probability.

`pred_formula_strs` is looked up with `pred_formula_formula_idxs` and `pred_node_ccs`
with `pred_node_node_idxs`. Those index arrays used to be requested only by the P(f) /
P(n) tokens, so `--return_formula_str --return_cc_hash` with a peaks-only
`--output_subset` silently produced NaN annotations. These tests pin the enabling
behaviour and the peak -> formula alignment that post-processing relies on.
"""

import warnings

import numpy as np
import pytest
import torch as th
import torch_geometric as pyg

from fragnnet.pl_model import FraGNNetPL
from fragnnet.utils import script_utils
from tests.test_export_schema import _StubInner

PEAKS_ONLY = {"pred_mzs", "pred_ints"}


class _StubFraGNNet(FraGNNetPL):
    """FraGNNetPL-typed stub replaying one canned model output, skipping __init__."""

    def __init__(self, batch_output: dict, hparams: dict, inner: "_StubInner | None" = None):
        self.training = False
        self._batch_output = batch_output
        self._hparams = hparams
        self.model = inner or _StubInner(("cc", "h_formulae_idx"), 4)
        # the exporter must never call this; `inference_step` already untransformed
        self.ints_untransform_func = _forbidden_untransform

    @property
    def hparams(self):
        return self._hparams

    def eval(self):
        return self

    def inference_step(self, batch_input, *, metrics=None):  # noqa: ARG002
        return self._batch_output


def _forbidden_untransform(ints, batch_idxs):  # noqa: ARG001
    raise AssertionError("the exporter must not re-apply ints_untransform_func")


def _log(vals):
    return th.log(th.tensor(vals, dtype=th.float32))


_CMF_INNER = _StubInner(("cc", "h_formulae_idx", "cmf_h_formulae_idx"), 4)


@pytest.fixture
def one_spectrum():
    """One spectrum: 3 peaks over 4 formulas (index 0 is NULL) and 3 DAG nodes.

    The joint grid is (f1,n0), (f1,n1), (f2,n1), (f3,n2) - so node 2 is reachable only
    through formula 3, which is the formula of the weakest peak.
    """
    batch_result = {
        "pred_mzs": th.tensor([100.0, 200.0, 300.0]),
        "pred_ints": th.tensor([0.5, 0.3, 0.2]),
        "pred_batch_idxs": th.tensor([0, 0, 0]),
        "pred_formula_logprobs": _log([0.1, 0.5, 0.3, 0.1]),
        "pred_formula_batch_idxs": th.tensor([0, 0, 0, 0]),
        "pred_formula_formula_idxs": th.arange(4),
        "pred_node_logprobs": _log([0.2, 0.5, 0.3]),
        "pred_node_batch_idxs": th.tensor([0, 0, 0]),
        "pred_node_node_idxs": th.arange(3),
        "pred_joint_batch_idxs": th.tensor([0, 0, 0, 0]),
        "pred_joint_formula_idxs": th.tensor([1, 1, 2, 3]),
        "pred_joint_node_idxs": th.tensor([0, 1, 1, 2]),
        "pred_formula_node_logprobs": _log([0.6, 0.4, 1.0, 1.0]),
        "pred_node_formula_logprobs": _log([1.0, 0.5, 0.5, 1.0]),
    }
    # column 0 is `depth`, column 1 is `cc` (NODE_FEAT_TO_IDX["cc"] == 1)
    frag_data = pyg.data.Data(
        x=th.tensor([[0.0, 11.0], [1.0, 22.0], [2.0, 33.0]]),
        node_feat_idxs=th.tensor([[0, 1, 2]]),
    )
    batch_input = {
        "spec_unique_id": th.tensor([7]),
        "mol_id": ["m0"],
        "batch_size": th.tensor(1),
        "frag_formula_cumsizes": th.tensor([0, 4]),
        "frag_num_nodes": th.tensor([0, 3]),
        "spec_prec_type_str": np.array(["[M+H]+"]),
        "frag_formula_str": np.array(["", "CH4", "C2H6", "C3H8"]),
        "frag_formula_peak_idxs": th.tensor([1, 2, 3]),
        "frag_formula_peak_mzs": th.tensor([100.0, 200.0, 300.0]),
        "frag_pyg": pyg.data.Batch.from_data_list([frag_data]),
    }
    return batch_input, batch_result


def _row(one_spectrum, output_set, *, min_peak_int=0.0, inner=None):
    batch_input, batch_result = one_spectrum
    hparams = {
        "auxiliary_scores": [],
        "eval_mz_bin_res": [],
        "eval_hun_sqrt": False,
        "eval_bin_sqrt": False,
    }
    model = _StubFraGNNet(batch_result, hparams, inner=inner)
    batch_idx_vars, node_idx_vars, formula_idx_vars, _ = script_utils.get_idx_vars(
        output_set, model, nb_iso=False, return_formula_str=True, return_cc_hash=True
    )
    columns = script_utils.get_tabular_columns(
        hparams, return_formula_str=True, return_cc_hash=True, output_subset=output_set
    )
    batch_rows = script_utils.process_batch_to_tabular_row(
        batch_input=batch_input,
        model=model,
        device=th.device("cpu"),
        evaluate=False,
        eval_score_types=[],
        BATCH_IDX_VARS=batch_idx_vars,
        NODE_IDX_VARS=node_idx_vars,
        FORMULA_IDX_VARS=formula_idx_vars,
        min_peak_int=min_peak_int,
        rebase_indices=True,
        return_formula_str=True,
        return_cc_hash=True,
        output_set=output_set,
    )
    assert len(batch_rows) == len(columns)
    return {k: v[0] for k, v in zip(columns, batch_rows)}


def test_annotation_requests_its_own_index_vars():
    """The idx arrays the annotation columns index with are pulled in on their own."""
    req_batch, req_node, req_formula, _ = script_utils.required_index_vars_for_outputs(
        PEAKS_ONLY, model=None, nb_iso=False, return_formula_str=True, return_cc_hash=True
    )
    assert "pred_formula_formula_idxs" in req_formula
    assert "pred_node_node_idxs" in req_node
    assert {"pred_formula_batch_idxs", "pred_node_batch_idxs"} <= req_batch
    # cc hashes additionally need the joint grid, which is what prunes them to the
    # nodes behind a surviving peak (and links a node back to a peak downstream)
    assert "pred_joint_batch_idxs" in req_batch
    assert "pred_joint_node_idxs" in req_node
    assert "pred_joint_formula_idxs" in req_formula


def test_no_annotation_requested_leaves_peaks_only():
    """Without the annotation flags a peaks-only subset stays peaks-only."""
    req_batch, req_node, req_formula, _ = script_utils.required_index_vars_for_outputs(
        PEAKS_ONLY, model=None, nb_iso=False
    )
    assert req_batch == {"pred_batch_idxs"}
    assert req_node == set()
    assert req_formula == set()


def test_formula_strings_populated_without_prob_tokens(one_spectrum):
    row = _row(one_spectrum, PEAKS_ONLY)
    assert row["pred_formula_strs"] == ["", "CH4", "C2H6", "C3H8"]


def test_cc_hashes_populated_without_prob_tokens(one_spectrum):
    row = _row(one_spectrum, PEAKS_ONLY)
    assert th.equal(row["pred_node_ccs"], th.tensor([[11.0], [22.0], [33.0]]))


def test_cc_hashes_pruned_to_surviving_peaks(one_spectrum):
    """`min_intensity` must prune the node arrays, not just the peaks and formulas.

    The threshold is on the base-peak-1000 scale the exporter emits, so 500 keeps the
    two strongest of [1000, 600, 400].
    """
    row = _row(one_spectrum, PEAKS_ONLY, min_peak_int=500.0)
    # peak 2 is dropped -> formula 3 goes -> node 2 is no longer reachable
    assert th.equal(row["pred_node_ccs"], th.tensor([[11.0], [22.0]]))
    assert row["pred_node_node_idxs"].tolist() == [0, 1]


def test_node_probs_pruned_with_their_nodes(one_spectrum):
    """P(n) stays aligned with the pruned node list."""
    row = _row(one_spectrum, PEAKS_ONLY | {"pred_node_probs"}, min_peak_int=500.0)
    assert th.allclose(row["pred_node_probs"], th.tensor([0.2, 0.5]), atol=1e-6)
    assert len(row["pred_node_ccs"]) == len(row["pred_node_probs"])


def test_joint_grid_links_peak_to_fragment(one_spectrum):
    """P(n|f) over the filtered joint is the peak -> fragment annotation."""
    row = _row(
        one_spectrum,
        PEAKS_ONLY | {"pred_formula_probs", "pred_formula_node_probs"},
        min_peak_int=500.0,
    )
    # joint entry (f3, n2) is filtered out with its peak
    assert th.allclose(row["pred_formula_node_probs"], th.tensor([0.6, 0.4, 1.0]), atol=1e-6)
    # every index is rebased into the surviving formula / node lists
    formulas = [row["pred_formula_strs"][i] for i in row["pred_joint_formula_idxs"].tolist()]
    assert formulas == ["CH4", "CH4", "C2H6"]
    ccs = row["pred_node_ccs"][row["pred_joint_node_idxs"].long()]
    assert th.equal(ccs, th.tensor([[11.0], [22.0], [22.0]]))


def test_probabilities_are_dropped_when_not_requested(one_spectrum):
    """Enabling annotation must not smuggle P(f)/P(n) back into the output."""
    row = _row(one_spectrum, PEAKS_ONLY)
    for prob_col in ("pred_formula_probs", "pred_node_probs", "pred_oos_prob"):
        assert isinstance(row[prob_col], float) and np.isnan(row[prob_col]), prob_col


def test_probabilities_kept_when_requested(one_spectrum):
    row = _row(one_spectrum, PEAKS_ONLY | {"pred_formula_probs", "pred_node_probs"})
    assert th.allclose(row["pred_formula_probs"], th.tensor([0.1, 0.5, 0.3, 0.1]), atol=1e-6)
    assert th.allclose(row["pred_node_probs"], th.tensor([0.2, 0.5, 0.3]), atol=1e-6)


def test_peak_to_formula_alignment_survives_intensity_filter(one_spectrum):
    """`pred_formula_strs[frag_formula_peak_idxs[k]]` is peak k's formula after filtering."""
    row = _row(one_spectrum, PEAKS_ONLY, min_peak_int=500.0)
    # peak 2 (400 on the base-peak-1000 scale) is dropped, so formula "C3H8" goes with it
    assert th.allclose(row["pred_mzs"], th.tensor([100.0, 200.0]))
    assert row["pred_formula_strs"] == ["CH4", "C2H6"]
    peak_formula = [row["pred_formula_strs"][i] for i in row["frag_formula_peak_idxs"].tolist()]
    assert peak_formula == ["CH4", "C2H6"]


@pytest.fixture
def two_spectra():
    """Two single-molecule spectra, so batch-global indices differ from local ones.

    Molecule 0 owns formulas/nodes 0-1, molecule 1 owns 2-3.
    """
    batch_result = {
        "pred_mzs": th.tensor([100.0, 200.0, 150.0, 250.0]),
        "pred_ints": th.tensor([0.6, 0.4, 0.7, 0.3]),
        "pred_batch_idxs": th.tensor([0, 0, 1, 1]),
        "pred_formula_logprobs": _log([0.6, 0.4, 0.7, 0.3]),
        "pred_formula_batch_idxs": th.tensor([0, 0, 1, 1]),
        "pred_formula_formula_idxs": th.arange(4),
        "pred_node_logprobs": _log([0.6, 0.4, 0.7, 0.3]),
        "pred_node_batch_idxs": th.tensor([0, 0, 1, 1]),
        "pred_node_node_idxs": th.arange(4),
        "pred_joint_batch_idxs": th.tensor([0, 0, 1, 1]),
        "pred_joint_formula_idxs": th.arange(4),
        "pred_joint_node_idxs": th.arange(4),
        "pred_formula_node_logprobs": _log([1.0, 1.0, 1.0, 1.0]),
        "pred_node_formula_logprobs": _log([1.0, 1.0, 1.0, 1.0]),
    }
    graphs = [
        pyg.data.Data(
            x=th.tensor([[0.0, 10.0], [1.0, 20.0]]), node_feat_idxs=th.tensor([[0, 1, 2]])
        ),
        pyg.data.Data(
            x=th.tensor([[0.0, 30.0], [1.0, 40.0]]), node_feat_idxs=th.tensor([[0, 1, 2]])
        ),
    ]
    batch_input = {
        "spec_unique_id": th.tensor([7, 8]),
        "mol_id": ["m0", "m1"],
        "batch_size": th.tensor(2),
        "frag_formula_cumsizes": th.tensor([0, 2, 4]),
        "frag_num_nodes": th.tensor([0, 2, 4]),
        "spec_prec_type_str": np.array(["[M+H]+", "[M-H]-"]),
        "frag_formula_str": np.array(["A0", "A1", "B0", "B1"]),
        "frag_formula_peak_idxs": th.tensor([0, 1, 0, 1]),
        "frag_formula_peak_mzs": th.tensor([100.0, 200.0, 150.0, 250.0]),
        "frag_pyg": pyg.data.Batch.from_data_list(graphs),
    }
    return batch_input, batch_result


def _rows(two_spectra, output_set, *, min_peak_int=0.0, inner=None):
    batch_input, batch_result = two_spectra
    hparams = {
        "auxiliary_scores": [],
        "eval_mz_bin_res": [],
        "eval_hun_sqrt": False,
        "eval_bin_sqrt": False,
    }
    model = _StubFraGNNet(batch_result, hparams, inner=inner)
    batch_idx_vars, node_idx_vars, formula_idx_vars, _ = script_utils.get_idx_vars(
        output_set, model, nb_iso=False, return_formula_str=True, return_cc_hash=True
    )
    columns = script_utils.get_tabular_columns(
        hparams, return_formula_str=True, return_cc_hash=True, output_subset=output_set
    )
    batch_rows = script_utils.process_batch_to_tabular_row(
        batch_input=batch_input,
        model=model,
        device=th.device("cpu"),
        evaluate=False,
        eval_score_types=[],
        BATCH_IDX_VARS=batch_idx_vars,
        NODE_IDX_VARS=node_idx_vars,
        FORMULA_IDX_VARS=formula_idx_vars,
        min_peak_int=min_peak_int,
        rebase_indices=True,
        return_formula_str=True,
        return_cc_hash=True,
        output_set=output_set,
    )
    return [dict(zip(columns, [col[i] for col in batch_rows])) for i in range(2)]


@pytest.mark.parametrize("min_peak_int", [0.0, 0.1])
def test_annotation_is_per_molecule_for_every_spectrum(two_spectra, min_peak_int):
    """Indices must be rebased to per-molecule whether or not the peak filter runs.

    With `min_peak_int <= 0` the filter used to return early and skip the rebase, so the
    second spectrum looked its annotation up with batch-global indices - out of bounds
    against a single molecule's node/formula table.
    """
    rows = _rows(two_spectra, PEAKS_ONLY, min_peak_int=min_peak_int)
    assert rows[0]["pred_formula_strs"] == ["A0", "A1"]
    assert rows[1]["pred_formula_strs"] == ["B0", "B1"]
    assert th.equal(rows[0]["pred_node_ccs"], th.tensor([[10.0], [20.0]]))
    assert th.equal(rows[1]["pred_node_ccs"], th.tensor([[30.0], [40.0]]))


# ---------------------------------------------------------------------------
# pred_formula_adducts: the CRF/CMF delta that makes "cc + adduct" resolvable
# ---------------------------------------------------------------------------


def test_adducts_populated_without_prob_tokens(one_spectrum):
    """The adduct column rides on the annotation flags, not on a probability token."""
    row = _row(one_spectrum, PEAKS_ONLY)
    # Nothing is pruned at threshold 0, so the NULL formula row (index 0) is still here;
    # it stands for no fragment, so it carries no adduct.
    assert row["pred_formula_adducts"] == ["", "+H", "+H", "+H"]
    # One entry per surviving formula row, aligned with the neutral formula strings.
    assert len(row["pred_formula_adducts"]) == len(row["pred_formula_strs"])


def test_adduct_is_per_spectrum_precursor(two_spectra):
    """[M+H]+ adds a proton, [M-H]- removes one; each spectrum gets its own delta."""
    rows = _rows(two_spectra, PEAKS_ONLY)
    # Each molecule owns 2 formula rows; row 0 of each is that molecule's NULL row.
    assert rows[0]["pred_formula_adducts"] == ["", "+H"]
    assert rows[1]["pred_formula_adducts"] == ["", "-H"]


def test_peak_reads_as_neutral_formula_plus_adduct(one_spectrum):
    """The documented annotation: peak -> neutral formula + its adduct delta."""
    row = _row(one_spectrum, PEAKS_ONLY)
    annots = [
        f"{row['pred_formula_strs'][i]} {row['pred_formula_adducts'][i]}"
        for i in row["frag_formula_peak_idxs"].tolist()
    ]
    assert annots == ["CH4 +H", "C2H6 +H", "C3H8 +H"]


def test_adduct_survives_the_intensity_filter_alignment(one_spectrum):
    """Pruning peaks must prune the adduct column with the formulas it labels."""
    row = _row(one_spectrum, PEAKS_ONLY, min_peak_int=500.0)
    # Peak 2 and its formula "C3H8" are pruned, so the adduct column must shrink with them.
    assert row["pred_formula_strs"] == ["CH4", "C2H6"]
    assert row["pred_formula_adducts"] == ["+H", "+H"]
    peak_idxs = row["frag_formula_peak_idxs"].tolist()
    assert max(peak_idxs) < len(row["pred_formula_adducts"])


@pytest.fixture
def cmf_spectrum():
    """One spectrum laid out as NULL + CRF[1:2] + CMF[1:2] - 5 formula rows.

    The CMF channel doubles the non-null formula rows, so a CMF batch always carries an
    odd number of them. The 4-row `one_spectrum` fixture is a CRF-only layout.
    """
    batch_result = {
        "pred_mzs": th.tensor([100.0, 200.0, 300.0, 400.0]),
        "pred_ints": th.tensor([0.4, 0.3, 0.2, 0.1]),
        "pred_batch_idxs": th.tensor([0, 0, 0, 0]),
        "pred_formula_logprobs": _log([0.1, 0.3, 0.3, 0.2, 0.1]),
        "pred_formula_batch_idxs": th.tensor([0, 0, 0, 0, 0]),
        "pred_formula_formula_idxs": th.arange(5),
        "pred_node_logprobs": _log([0.5, 0.5]),
        "pred_node_batch_idxs": th.tensor([0, 0]),
        "pred_node_node_idxs": th.arange(2),
        "pred_joint_batch_idxs": th.tensor([0, 0, 0, 0]),
        "pred_joint_formula_idxs": th.tensor([1, 2, 3, 4]),
        "pred_joint_node_idxs": th.tensor([0, 1, 0, 1]),
        "pred_formula_node_logprobs": _log([1.0, 1.0, 1.0, 1.0]),
        "pred_node_formula_logprobs": _log([1.0, 1.0, 1.0, 1.0]),
    }
    frag_data = pyg.data.Data(
        x=th.tensor([[0.0, 11.0], [1.0, 22.0]]),
        node_feat_idxs=th.tensor([[0, 1, 2]]),
    )
    batch_input = {
        "spec_unique_id": th.tensor([7]),
        "mol_id": ["m0"],
        "batch_size": th.tensor(1),
        "frag_formula_cumsizes": th.tensor([0, 5]),
        "frag_num_nodes": th.tensor([0, 2]),
        "spec_prec_type_str": np.array(["[M+Na]+"]),
        # CRF and CMF rows describe the same neutral fragments, so the strings repeat.
        "frag_formula_str": np.array(["", "CH4", "C2H6", "CH4", "C2H6"]),
        "frag_formula_peak_idxs": th.tensor([1, 2, 3, 4]),
        "frag_formula_peak_mzs": th.tensor([100.0, 200.0, 300.0, 400.0]),
        "frag_pyg": pyg.data.Batch.from_data_list([frag_data]),
    }
    return batch_input, batch_result


def test_cmf_half_carries_the_migration_delta(cmf_spectrum):
    """With CMF enumerated, the second half of the formula rows carries the CMF delta.

    [M+Na]+ keeps both channels: a CRF fragment holds the sodium, a CMF fragment is
    protonated, so the two halves must not share a delta. Without this the sodiated and
    protonated annotations of one neutral fragment would be indistinguishable.
    """
    row = _row(cmf_spectrum, PEAKS_ONLY, inner=_CMF_INNER)
    assert row["pred_formula_adducts"] == ["", "+Na", "+Na", "+H", "+H"]
    # The same neutral fragment therefore annotates as two different ions.
    annots = [
        f"{row['pred_formula_strs'][i]} {row['pred_formula_adducts'][i]}"
        for i in row["frag_formula_peak_idxs"].tolist()
    ]
    assert annots == ["CH4 +Na", "C2H6 +Na", "CH4 +H", "C2H6 +H"]


def test_cmf_suppressed_when_it_duplicates_a_crf_h_shift(one_spectrum):
    """[M+H]+ emits no CMF half: its CMF ion is the CRF state at delta_h = -1.

    So even with `cmf_h_formulae_idx` wired, every row keeps the CRF delta.
    """
    row = _row(one_spectrum, PEAKS_ONLY, inner=_CMF_INNER)
    assert row["pred_formula_adducts"] == ["", "+H", "+H", "+H"]


def test_missing_prec_type_str_degrades_to_nan(one_spectrum):
    """A batch without `spec_prec_type_str` yields NaN, not a KeyError mid-export.

    `spec_prec_type_str` exists only when `spec_params.prec_type_str` is on - False in
    `config/template.yml` and in nearly every experiment config - and it is
    `apply_inference_config` that forces it on. Any caller that does not go through it (a
    notebook, a future script, a unit-level call) must degrade the way every sibling field
    in the exporter does, and say so rather than degrading silently.
    """
    batch_input, batch_result = one_spectrum
    batch_input.pop("spec_prec_type_str")
    with pytest.warns(UserWarning, match="spec_params.prec_type_str"):
        row = _row((batch_input, batch_result), PEAKS_ONLY)
    assert len(row["pred_formula_adducts"]) == len(row["pred_formula_strs"])
    assert all(np.isnan(value) for value in row["pred_formula_adducts"])


def test_annotation_inputs_present_does_not_warn(one_spectrum):
    """The degradation warning must not fire on the normal, fully-configured path."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        row = _row(one_spectrum, PEAKS_ONLY)
    assert row["pred_formula_adducts"] == ["", "+H", "+H", "+H"]
