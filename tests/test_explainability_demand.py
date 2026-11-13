"""Who decides `return_explainability_tensors`, the single explainability switch.

The model defaults it True, so only an explicit False buys the skip. Two places produce
that False, and both are tested here:

- the inference exporters, which already compute `_need_expl` from the requested output
  tokens and now pass it either way rather than only when True;
- `inference_step(metrics=...)`, which ORs its own auxiliary-metric demand on top,
  because a metric can need a marginal the caller never asked to export. The OR is
  gated on `metrics`: a prediction-only call computes no metrics and so adds no demand.
"""

import contextlib

import torch as th

from fragnnet.model.fragnnet_model import EXPLAINABILITY_OUTPUT_KEYS
from fragnnet.pl_model import FraGNNetPL
from fragnnet.runner import load_config
from fragnnet.utils import script_utils

PEAKS_ONLY = {"pred_mzs", "pred_ints"}


class _Stub(FraGNNetPL):
    """Records what `return_explainability_tensors` the step passed to forward."""

    def __init__(self, extra_metric_args=()):
        self.training = False
        self.extra_metric_args = list(extra_metric_args)
        # Real models compute this in `_setup_metric_fns`; TestExplMetricArgs pins that.
        self.expl_metric_args = EXPLAINABILITY_OUTPUT_KEYS.intersection(self.extra_metric_args)
        self.seen = None

    def forward(self, **batch):
        self.seen = batch.get("return_explainability_tensors", "absent")
        raise _Stop


class _Stop(Exception):
    pass


def _run_gt_step(stub, batch, metrics="test"):
    with contextlib.suppress(_Stop, KeyError, TypeError):
        stub.inference_step(batch, metrics=metrics)
    assert stub.seen is not None, "forward was never reached"
    return stub.seen


def _batch(**kw):
    base = {
        "batch_size": th.tensor(1),
        "spec_mzs": th.tensor([100.0]),
        "spec_ints": th.tensor([1.0]),
        "spec_batch_idxs": th.tensor([0]),
        "spec_prec_mz": th.tensor([200.0]),
        "spec_per_group": th.tensor([1]),
        "spec_per_mol": th.tensor([1]),
        "group_per_mol": th.tensor([1]),
    }
    base.update(kw)
    return base


class TestExporterDemand:
    """`_need_expl` must reach the model as an explicit False, not just be omitted."""

    def test_peaks_only_request_declares_false(self):
        model = _Stub()
        batch_idx_vars, node_idx_vars, formula_idx_vars, _ = script_utils.get_idx_vars(
            PEAKS_ONLY, model, nb_iso=False
        )
        # peaks-only needs neither anchor index array
        assert "pred_formula_formula_idxs" not in formula_idx_vars
        assert "pred_node_node_idxs" not in node_idx_vars
        assert "pred_batch_idxs" in batch_idx_vars

    def test_annotation_request_declares_true(self):
        """--return_formula_str / --return_cc_hash pull in the anchors, so demand is True."""
        model = _Stub()
        _, node_idx_vars, formula_idx_vars, _ = script_utils.get_idx_vars(
            PEAKS_ONLY, model, nb_iso=False, return_formula_str=True, return_cc_hash=True
        )
        assert "pred_formula_formula_idxs" in formula_idx_vars
        assert "pred_node_node_idxs" in node_idx_vars

    def test_probability_tokens_declare_true(self):
        model = _Stub()
        _, node_idx_vars, formula_idx_vars, _ = script_utils.get_idx_vars(
            {"pred_mzs", "pred_node_probs"}, model, nb_iso=False
        )
        assert "pred_node_node_idxs" in node_idx_vars

    def test_output_everything_declares_true(self):
        """output_set=None means every token, so the anchors come along."""
        model = _Stub()
        _, node_idx_vars, formula_idx_vars, _ = script_utils.get_idx_vars(
            None, model, nb_iso=False
        )
        assert "pred_formula_formula_idxs" in formula_idx_vars
        assert "pred_node_node_idxs" in node_idx_vars


class TestMetricDemand:
    def test_no_metric_args_leaves_a_declared_false_alone(self):
        """The production path passes auxiliary_scores=[], so nothing overrides the False."""
        stub = _Stub(extra_metric_args=[])
        assert _run_gt_step(stub, _batch(return_explainability_tensors=False)) is False

    def test_gated_metric_arg_forces_true(self):
        """`pred_node_count` under nb_iso needs `pred_nb_node_batch_idxs`, which IS gated."""
        assert "pred_nb_node_batch_idxs" in EXPLAINABILITY_OUTPUT_KEYS
        stub = _Stub(extra_metric_args=["pred_nb_node_batch_idxs"])
        assert _run_gt_step(stub, _batch(return_explainability_tensors=False)) is True

    def test_ungated_metric_arg_does_not_force_true(self):
        """`pred_node_batch_idxs` comes off the frag graph, not the marginal block."""
        assert "pred_node_batch_idxs" not in EXPLAINABILITY_OUTPUT_KEYS
        stub = _Stub(extra_metric_args=["pred_node_batch_idxs", "pred_formula_batch_idxs"])
        assert _run_gt_step(stub, _batch(return_explainability_tensors=False)) is False

    def test_caller_true_is_passed_through(self):
        stub = _Stub(extra_metric_args=[])
        assert _run_gt_step(stub, _batch(return_explainability_tensors=True)) is True

    def test_absent_key_is_left_absent(self):
        """Non-FraGNNet models never get the key; the model's own default applies."""
        stub = _Stub(extra_metric_args=["pred_nb_node_batch_idxs"])
        assert _run_gt_step(stub, _batch()) == "absent"

    def test_metrics_none_does_not_force_true(self):
        """A prediction-only call scores nothing, so it must not override the caller."""
        stub = _Stub(extra_metric_args=["pred_nb_node_batch_idxs"])
        seen = _run_gt_step(stub, _batch(return_explainability_tensors=False), metrics=None)
        assert seen is False


class TestExplMetricArgs:
    """`expl_metric_args` is computed by real `_setup_metric_fns`, not by the stub above."""

    @staticmethod
    def _real_model(**overrides):
        config = load_config("config/template.yml", None, None)
        config.update(model_type="frag_gnn", compile=False, sparse_cosine_similarity=True)
        config.update(**overrides)
        return FraGNNetPL(**config)

    def test_set_by_init_and_is_the_intersection(self):
        """Every real model has it after __init__, and it holds only gated metric inputs."""
        model = self._real_model()
        assert isinstance(model.expl_metric_args, frozenset)
        assert model.expl_metric_args == EXPLAINABILITY_OUTPUT_KEYS.intersection(
            model.extra_metric_args
        )
        # whatever the template configures, an ungated arg can never land in the set
        assert "pred_node_batch_idxs" not in model.expl_metric_args

    def test_node_count_metric_under_nb_iso_is_gated(self):
        """`pred_node_count` + nb_iso is the one real config that demands a marginal."""
        model = self._real_model(nb_iso=True, auxiliary_scores=["pred_node_count"])
        assert "pred_nb_node_batch_idxs" in model.extra_metric_args
        assert model.expl_metric_args == frozenset({"pred_nb_node_batch_idxs"})

    def test_empty_without_gated_metrics(self):
        model = self._real_model(nb_iso=False, auxiliary_scores=["pred_formula_count"])
        assert "pred_formula_batch_idxs" in model.extra_metric_args
        assert model.expl_metric_args == frozenset()
