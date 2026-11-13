"""Tests for `run_inference_pkl` output-token translation.

`run_inference_pkl` receives user-facing `output_subset` tokens (e.g. `pred_ints`,
`pred_node_probs`) while the marginals are emitted in log space
(`pred_node_logprobs`). These tests pin the translation: probability tokens are exp()
of their logprob key, `pred_ints` is only base-peak rescaled because `inference_step`
already untransformed it, index variables are retained, and everything is rebatched
across batches so the flat columns stay regroupable.
"""

import pytest
import torch as th

from fragnnet.pl_model import FraGNNetPL
from fragnnet.utils import script_utils


class _StubFraGNNet(FraGNNetPL):
    """Minimal FraGNNetPL-typed stub that replays canned per-batch outputs.

    Subclasses FraGNNetPL so the `isinstance` checks in `run_inference_pkl` /
    `get_idx_vars` take the FraGNNet branch, but skips the heavy __init__.
    """

    def __init__(self, batch_outputs: list[dict]):
        self.training = False
        self._batch_outputs = batch_outputs
        # the exporter must never call this; a raising stub proves it
        self.ints_untransform_func = _forbidden_untransform

    def inference_step(self, batch_input, *, metrics=None):  # noqa: ARG002
        return self._batch_outputs[batch_input["_batch_i"]]


def _forbidden_untransform(ints, batch_idxs):  # noqa: ARG001
    raise AssertionError("the exporter must not re-apply ints_untransform_func")


def _log(vals):
    return th.log(th.tensor(vals, dtype=th.float64))


@pytest.fixture
def two_batches():
    """Two single-spectrum batches, 2 peaks and 2 nodes each."""
    b0 = {
        "pred_mzs": th.tensor([100.0, 200.0]),
        "pred_ints": th.tensor([0.6, 0.4], dtype=th.float64),
        "pred_batch_idxs": th.tensor([0, 0]),
        "pred_node_logprobs": _log([0.7, 0.3]),
        "pred_node_batch_idxs": th.tensor([0, 0]),
        "pred_node_node_idxs": th.tensor([0, 1]),
    }
    b1 = {
        "pred_mzs": th.tensor([150.0, 250.0]),
        "pred_ints": th.tensor([0.5, 0.5], dtype=th.float64),
        "pred_batch_idxs": th.tensor([0, 0]),
        "pred_node_logprobs": _log([0.9, 0.1]),
        "pred_node_batch_idxs": th.tensor([0, 0]),
        "pred_node_node_idxs": th.tensor([0, 1]),
    }
    return [b0, b1]


def _run(two_batches, output_set):
    model = _StubFraGNNet(two_batches)
    dataloader = [{"_batch_i": 0}, {"_batch_i": 1}]
    return script_utils.run_inference_pkl(
        model,
        dataloader,
        device=th.device("cpu"),
        output_set=output_set,
        evaluate=True,
    )


def test_node_probs_are_exp_of_logprobs(two_batches):
    out = _run(two_batches, {"pred_mzs", "pred_node_probs"})
    assert set(out.keys()) >= {"pred_mzs", "pred_node_probs"}
    # exp(log P(n)) recovers P(n), concatenated across both batches
    expected = th.tensor([0.7, 0.3, 0.9, 0.1], dtype=th.float64)
    assert th.allclose(out["pred_node_probs"], expected, atol=1e-6)


def test_pred_ints_does_not_reapply_untransform(two_batches):
    """`inference_step` already untransformed; the exporter must only rescale the base peak.

    The step runs `preproc_spec(train=False, pred=True)`, which applies
    `ints_untransform_func` and L1-normalizes. Re-applying the untransform here is what
    squared every exported relative intensity under `ints_transform: sqrt`.
    """
    out = _run(two_batches, {"pred_mzs", "pred_ints"})
    # base peak = 1000 per spectrum, relative intensities preserved LINEARLY:
    # batch 0 is 0.6/0.4 -> 1000/666.67 (squaring would have given 1000/444.4)
    expected = th.tensor([1000.0, 1000.0 * 0.4 / 0.6, 1000.0, 1000.0], dtype=th.float64)
    assert th.allclose(out["pred_ints"], expected, atol=1e-6)


@pytest.mark.parametrize("evaluate", [True, False])
def test_pred_ints_identical_in_both_modes(two_batches, evaluate):
    """`evaluate` only adds ground-truth scoring; the intensities must not move.

    Before the merge these were two different steps with two different postconditions,
    and `evaluate` had to be decoded by the exporter to pick an untransform.
    """
    model = _StubFraGNNet(two_batches)
    out = script_utils.run_inference_pkl(
        model,
        [{"_batch_i": 0}, {"_batch_i": 1}],
        device=th.device("cpu"),
        output_set={"pred_mzs", "pred_ints"},
        evaluate=evaluate,
    )
    expected = th.tensor([1000.0, 1000.0 * 0.4 / 0.6, 1000.0, 1000.0], dtype=th.float64)
    assert th.allclose(out["pred_ints"], expected, atol=1e-6)


def test_pred_ints_missing_raises(two_batches):
    """`pred_ints` is the only intensity contract, so its absence must fail loudly."""
    batches = [{k: v for k, v in b.items() if k != "pred_ints"} for b in two_batches]
    with pytest.raises(KeyError, match="pred_ints"):
        _run(batches, {"pred_mzs", "pred_ints"})


def test_index_vars_retained_and_rebatched(two_batches):
    # Node-only request: the node index vars are the required anchor and must
    # survive the token filter AND be rebatched across batches, even though
    # peak-level `pred_batch_idxs` is not required (and is correctly pruned).
    out = _run(two_batches, {"pred_node_probs"})
    assert "pred_node_batch_idxs" in out
    assert "pred_node_node_idxs" in out
    # node->spectrum idxs rebatched: batch 1's spectrum becomes global idx 1
    assert out["pred_node_batch_idxs"].tolist() == [0, 0, 1, 1]
    # node idxs rebatched by cumulative node count (2 nodes in batch 0)
    assert out["pred_node_node_idxs"].tolist() == [0, 1, 2, 3]


def test_pred_mzs_flat_concat(two_batches):
    out = _run(two_batches, {"pred_mzs"})
    assert out["pred_mzs"].tolist() == [100.0, 200.0, 150.0, 250.0]


def test_unproducible_token_raises(two_batches):
    # frag_formula_peak_mzs is not in the raw model output here -> must fail loud,
    # not silently drop (the original bug produced a one-column garbage file).
    with pytest.raises(KeyError, match="frag_formula_peak_mzs"):
        _run(two_batches, {"pred_mzs", "frag_formula_peak_mzs"})


def test_node_probs_normalized_per_spectrum(two_batches):
    # Sanity: within each spectrum the recovered P(n) should be a valid dist.
    out = _run(two_batches, {"pred_node_probs"})
    b0 = out["pred_node_probs"][out["pred_node_batch_idxs"] == 0].sum().item()
    b1 = out["pred_node_probs"][out["pred_node_batch_idxs"] == 1].sum().item()
    assert b0 == pytest.approx(1.0, abs=1e-6)
    assert b1 == pytest.approx(1.0, abs=1e-6)
