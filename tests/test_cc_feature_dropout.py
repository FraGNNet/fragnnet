from pathlib import Path

import torch as th
import yaml

from fragnnet.model.fragnnet_model import FraGNNetModel


def _make_model_stub(
    *,
    p: float,
    frag_node_feats: list[str] | None = None,
    training: bool = True,
) -> FraGNNetModel:
    model = FraGNNetModel.__new__(FraGNNetModel)
    th.nn.Module.__init__(model)
    model.cc_feature_dropout = p
    model.frag_node_feats = frag_node_feats or ["cc"]
    if training:
        model.train()
    else:
        model.eval()
    return model


def test_cc_feature_dropout_zero_never_changes_tensor():
    model = _make_model_stub(p=0.0, training=True)
    x = th.randn(4, 3)

    out = model._maybe_apply_cc_feature_dropout(x)

    assert out is x
    assert th.equal(out, x)


def test_cc_feature_dropout_one_zeros_cc_group_in_training():
    model = _make_model_stub(p=1.0, training=True)
    x = th.randn(4, 3)

    out = model._maybe_apply_cc_feature_dropout(x)

    assert out is not x
    assert th.equal(out, th.zeros_like(x))


def test_cc_feature_dropout_eval_mode_does_not_drop():
    model = _make_model_stub(p=1.0, training=False)
    x = th.randn(4, 3)

    out = model._maybe_apply_cc_feature_dropout(x)

    assert out is x
    assert th.equal(out, x)


def test_cc_feature_dropout_noop_when_cc_feature_absent():
    model = _make_model_stub(p=1.0, frag_node_feats=["boundary_cc"], training=True)
    x = th.randn(4, 3)

    out = model._maybe_apply_cc_feature_dropout(x)

    assert out is x
    assert th.equal(out, x)


def test_cc_feature_dropout_config_keys_exist():
    template = yaml.safe_load(Path("config/template.yml").read_text())
    exp = yaml.safe_load(
        Path(
            "config/exp_fraggnn_ma_mi/d3_new_features/"
            "fraggnn_d3_ma_mi_nist20_amolc_cc_dropout_02.yml"
        ).read_text()
    )

    assert template["cc_feature_dropout"] == 0.0
    assert exp["cc_feature_dropout"] == 0.2
