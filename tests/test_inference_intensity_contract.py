"""One inference step, one intensity space.

`inference_step` is the only entry point the exporters call, and it always returns
`pred_ints` linear and L1-normalized per spectrum - `metrics` adds ground-truth scoring
and changes nothing about the intensities. The exporters therefore owe exactly one
operation, `batched_mf1000_normalize`, with no branch to get wrong.

This replaces `resolve_pred_ints(..., step_untransformed=...)`, which reconciled three
steps that disagreed about what `pred_logprobs` meant. The bug that flag existed to fix:
both exporters unconditionally applied `ints_untransform_func` to `exp(pred_logprobs)`,
but the ground-truth step had already applied it inside `preproc_spec(train=False,
pred=True)`. Under `ints_transform: sqrt` the second application SQUARED every exported
relative intensity.
"""

import numpy as np
import pytest
import torch as th
import torch_geometric as pyg

from fragnnet.pl_model import FraGNNetPL
from fragnnet.utils import script_utils
from fragnnet.utils.spec_utils import batched_l1_normalize, get_ints_untransform_func

# what `ints_transform: sqrt` installs: rescale base peak to sqrt(1000), then square
SQRT_MAX_INTS = float(np.sqrt(1000.0))


def _sqrt_untransform(ints, batch_idxs):
    max_ints = th.zeros(int(batch_idxs.max()) + 1).scatter_reduce(
        0, batch_idxs, ints, reduce="amax", include_self=False
    )
    return (ints / max_ints[batch_idxs] * SQRT_MAX_INTS) ** 2


def _forbidden_untransform(ints, batch_idxs):  # noqa: ARG001
    raise AssertionError("the exporter must not re-apply ints_untransform_func")


@pytest.fixture
def batch_idxs():
    return th.tensor([0, 0, 0, 1, 1])


class TestSingleStepPostcondition:
    """`preproc_spec(train=False, pred=True)` is what the step applies, in both modes."""

    def test_untransform_then_l1_is_the_postcondition(self, batch_idxs):
        probs = th.tensor([0.5, 0.3, 0.2, 0.6, 0.4])
        untransformed = _sqrt_untransform(probs, batch_idxs)
        l1 = (
            untransformed
            / th.zeros(2).scatter_reduce(
                0, batch_idxs, untransformed, reduce="sum", include_self=False
            )[batch_idxs]
        )
        assert l1[:3].sum().item() == pytest.approx(1.0, abs=1e-6)
        assert l1[3:].sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_exporter_rescale_lands_on_base_peak_1000(self, batch_idxs):
        """The one operation the exporter owes, and it is idempotent."""
        l1 = th.tensor([0.5, 0.3, 0.2, 0.6, 0.4])
        out = script_utils.batched_mf1000_normalize(l1, batch_idxs)
        assert out[:3].max().item() == pytest.approx(1000.0, abs=1e-3)
        assert out[3:].max().item() == pytest.approx(1000.0, abs=1e-3)
        # relative intensities stay LINEAR - squaring 0.3/0.5 would give 360, not 600
        assert out[1].item() == pytest.approx(600.0, abs=1e-3)
        again = script_utils.batched_mf1000_normalize(out, batch_idxs)
        assert th.allclose(out, again, atol=1e-6)


# --------------------------------------------------------------------------------------
# End to end through the tabular exporter, which is what writes predictions_*.hdf5
# --------------------------------------------------------------------------------------


class _StubPL(FraGNNetPL):
    """FraGNNetPL-typed stub replaying one canned output, recording the `metrics` it got."""

    def __init__(self, batch_output: dict):
        self.training = False
        self._batch_output = batch_output
        self.seen_metrics = "unset"
        # the exporter must never call this; the step already untransformed
        self.ints_untransform_func = _forbidden_untransform
        self._hparams = {
            "auxiliary_scores": [],
            "eval_mz_bin_res": [],
            "eval_hun_sqrt": False,
            "eval_bin_sqrt": False,
        }

    @property
    def hparams(self):
        return self._hparams

    def eval(self):
        return self

    def inference_step(self, batch_input, *, metrics=None):  # noqa: ARG002
        self.seen_metrics = metrics
        return self._batch_output


def _tabular_ints(evaluate: bool, ints: th.Tensor, min_peak_int: float = 0.0, model=None):
    batch_result = {
        "pred_mzs": th.tensor([100.0, 200.0, 300.0]),
        "pred_ints": ints,
        "pred_batch_idxs": th.tensor([0, 0, 0]),
    }
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
        "frag_formula_peak_idxs": th.tensor([1, 2, 3]),
        "frag_formula_peak_mzs": th.tensor([100.0, 200.0, 300.0]),
        "frag_pyg": pyg.data.Batch.from_data_list([frag_data]),
    }
    output_set = {"pred_mzs", "pred_ints"}
    model = model if model is not None else _StubPL(batch_result)
    batch_idx_vars, node_idx_vars, formula_idx_vars, _ = script_utils.get_idx_vars(
        output_set, model, nb_iso=False
    )
    columns = script_utils.get_tabular_columns(
        model.hparams, return_formula_str=False, return_cc_hash=False, output_subset=output_set
    )
    batch_rows = script_utils.process_batch_to_tabular_row(
        batch_input=batch_input,
        model=model,
        device=th.device("cpu"),
        evaluate=evaluate,
        eval_score_types=[],
        BATCH_IDX_VARS=batch_idx_vars,
        NODE_IDX_VARS=node_idx_vars,
        FORMULA_IDX_VARS=formula_idx_vars,
        min_peak_int=min_peak_int,
        rebase_indices=True,
        return_formula_str=False,
        return_cc_hash=False,
        output_set=output_set,
    )
    row = {k: v[0] for k, v in zip(columns, batch_rows)}
    return th.as_tensor(np.asarray(row["pred_ints"], dtype=np.float64))


class TestTabularExporter:
    INTS = th.tensor([0.5, 0.3, 0.2])
    EXPECTED = th.tensor([1000.0, 600.0, 400.0], dtype=th.float64)

    @pytest.mark.parametrize("evaluate", [True, False])
    def test_both_modes_export_the_same_intensities(self, evaluate):
        """`evaluate` no longer selects an intensity space - it only adds scoring."""
        assert th.allclose(
            _tabular_ints(evaluate=evaluate, ints=self.INTS), self.EXPECTED, atol=1e-2
        )

    @pytest.mark.parametrize(("evaluate", "expected_metrics"), [(True, "test"), (False, None)])
    def test_evaluate_maps_to_metrics(self, evaluate, expected_metrics):
        batch_result = {
            "pred_mzs": th.tensor([100.0, 200.0, 300.0]),
            "pred_ints": self.INTS,
            "pred_batch_idxs": th.tensor([0, 0, 0]),
        }
        model = _StubPL(batch_result)
        _tabular_ints(evaluate=evaluate, ints=self.INTS, model=model)
        assert model.seen_metrics == expected_metrics

    def test_min_peak_int_sees_the_documented_scale(self):
        """`--min_intensity` is percent-of-base-peak x 10, so it must see 0-1000 ints.

        A peak at 10% of base must survive a 50 cut. It used to depend on which step ran:
        the eval branch squared it down to 10 and dropped it.
        """
        kept = _tabular_ints(evaluate=True, ints=th.tensor([1.0, 0.1, 0.01]), min_peak_int=50.0)
        assert len(kept) == 2
        assert kept[1].item() == pytest.approx(100.0, abs=1e-2)


# --------------------------------------------------------------------------------------
# The real step, with `preproc_spec` actually running
# --------------------------------------------------------------------------------------


class _RealStepStub(FraGNNetPL):
    """Only `forward` is canned; `preproc_spec` is the inherited production code."""

    def __init__(self, pred: dict):
        self.training = False
        self._pred = pred
        self.ints_untransform_func = get_ints_untransform_func("sqrt")
        self.ints_normalize_func = batched_l1_normalize

    def forward(self, **batch):  # noqa: ARG002
        return dict(self._pred)


class _NoGroundTruth(dict):
    """A batch that fails loudly if the step reaches for a ground-truth field."""

    _GT = frozenset({"spec_mzs", "spec_ints", "spec_batch_idxs", "spec_prec_mz"})

    def __getitem__(self, key):
        if key in self._GT:
            raise AssertionError(f"metrics=None must not read ground truth, got {key!r}")
        return super().__getitem__(key)


class TestRealStepPostcondition:
    @staticmethod
    def _pred():
        return {
            "pred_mzs": th.tensor([100.0, 200.0, 300.0]),
            # a model simplex in TRANSFORMED space, which is what forward emits
            "pred_logprobs": th.log(th.tensor([0.5, 0.3, 0.2])),
            "pred_batch_idxs": th.tensor([0, 0, 0]),
        }

    def test_prediction_only_never_reads_ground_truth(self):
        """A predict-only batch carries no reference spectrum, so the step must not look."""
        stub = _RealStepStub(self._pred())
        out = stub.inference_step(_NoGroundTruth(batch_size=th.tensor(1)), metrics=None)
        assert set(out) >= {"pred_mzs", "pred_ints", "pred_batch_idxs"}
        assert "true_mzs" not in out and "pred_logprobs" not in out

    def test_pred_ints_is_l1_normalized(self):
        stub = _RealStepStub(self._pred())
        out = stub.inference_step(_NoGroundTruth(batch_size=th.tensor(1)), metrics=None)
        assert out["pred_ints"].sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_exporter_rescale_recovers_the_untransformed_ratios(self):
        """L1 then mf1000 is the same as `ints_untransform_func` alone, as measured."""
        stub = _RealStepStub(self._pred())
        out = stub.inference_step(_NoGroundTruth(batch_size=th.tensor(1)), metrics=None)
        exported = script_utils.batched_mf1000_normalize(out["pred_ints"], out["pred_batch_idxs"])
        direct = get_ints_untransform_func("sqrt")(th.tensor([0.5, 0.3, 0.2]), th.tensor([0, 0, 0]))
        assert th.allclose(exported, direct, atol=1e-3)
        # and that is the squared-ratio shape the raw-forward contract asks for
        assert th.allclose(exported, th.tensor([1000.0, 360.0, 160.0]), atol=1e-2)

    def test_oos_prob_is_available_without_ground_truth(self):
        """P(out of spectrum) is a forward output, so prediction-only runs must get it.

        The exp() used to sit below the `metrics is None` early return, on the metrics side,
        so `run_inference.py --output_everything` emitted `pred_oos_prob` as NaN even though
        nothing about the quantity needs a reference spectrum.
        """
        pred = self._pred()
        pred["pred_oos_logprobs"] = th.log(th.tensor([0.25]))
        stub = _RealStepStub(pred)
        out = stub.inference_step(_NoGroundTruth(batch_size=th.tensor(1)), metrics=None)
        assert "oos_prob" in out
        assert out["oos_prob"].item() == pytest.approx(0.25, abs=1e-6)

    def test_no_oos_key_when_the_model_predicts_none(self):
        """`predict_oos: False` leaves the key absent rather than emitting a placeholder."""
        stub = _RealStepStub(self._pred())
        out = stub.inference_step(_NoGroundTruth(batch_size=th.tensor(1)), metrics=None)
        assert "oos_prob" not in out
