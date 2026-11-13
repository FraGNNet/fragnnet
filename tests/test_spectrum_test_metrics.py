"""Configured per-spectrum metrics must work in labeled inference and testing."""

from types import SimpleNamespace

import pytest
import torch as th

from fragnnet.pl_model.spectrum_pl import SpectrumPL
from fragnnet.runner import load_config


@pytest.mark.parametrize("score", ["ndcg", "true_oos_prob", "true_oos_e"])
@pytest.mark.parametrize("split", ["test", "predict"])
def test_indexed_metrics_match_validation(score, split):
    config = load_config("config/template.yml", None, None)
    config.update(auxiliary_scores=[score], sparse_cosine_similarity=True)
    pl = SimpleNamespace(
        hparams=SimpleNamespace(**config), metric_names=set(),
        tolerance=1e-5, relative=True, tolerance_min_mz=200.0,
    )
    SpectrumPL._setup_metric_fns(pl)
    inputs = {
        "true_mzs": th.tensor([100.0, 120.0, 100.0, 120.0]),
        "true_ints": th.tensor([0.8, 0.2, 0.3, 0.7]),
        "true_batch_idxs": th.tensor([0, 0, 1, 1]),
        "pred_mzs": th.tensor([100.0, 140.0, 100.0, 120.0]),
        "pred_ints": th.tensor([0.6, 0.4, 0.3, 0.7]),
        "pred_batch_idxs": th.tensor([0, 0, 1, 1]),
        "true_prec_mzs": th.tensor([200.0, 200.0]),
    }
    expected = pl.metric_fn(**inputs, split="val")
    actual = pl.metric_fn(**inputs, split=split)
    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        assert value.shape == (2,)
        assert th.isfinite(value).all()
        th.testing.assert_close(value, expected[key])
    if score == "true_oos_prob":
        th.testing.assert_close(actual[score], th.tensor([0.2, 0.0]))
