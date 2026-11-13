"""Inference must restore complete checkpoint weights, including compiled models."""

import copy

import pytest
import torch as th

from fragnnet.pl_model.fragnnet_pl import FraGNNetPL
from fragnnet.runner import load_config
from fragnnet.utils.script_utils import (
    apply_inference_config,
    get_config_from_ckpt,
    init_model_from_ckpt,
)


@pytest.fixture
def checkpoint_model():
    config = load_config("config/template.yml", None, None)
    config.update(model_type="frag_gnn", compile=False, sparse_cosine_similarity=True)
    th.manual_seed(7)
    return FraGNNetPL(**config).eval(), config


@pytest.mark.parametrize("compilation", ["legacy", "none", "whole", "partial"])
def test_inference_restores_exact_checkpoint_state(checkpoint_model, tmp_path, compilation):
    model, config = checkpoint_model
    expected = copy.deepcopy(model.state_dict())
    if compilation == "whole":
        model.model = th.compile(model.model, backend="eager")
    elif compilation == "partial":
        model.model.formula_module = th.compile(model.model.formula_module, backend="eager")
    if compilation == "legacy":
        config.pop("compile")
    else:
        config["compile"] = compilation != "none"
    path = tmp_path / "model.ckpt"
    th.save({"state_dict": model.state_dict(), "hyper_parameters": config}, path)
    inference_config = get_config_from_ckpt(str(path))
    apply_inference_config(inference_config)
    th.manual_seed(11)
    restored = init_model_from_ckpt(str(path), inference_config, th.device("cpu"))
    assert not restored.training
    assert inference_config["compile"] is False
    assert restored.state_dict().keys() == expected.keys()
    for key, value in restored.state_dict().items():
        th.testing.assert_close(value, expected[key], atol=0, rtol=0)


@pytest.mark.parametrize("defect", ["missing", "unexpected"])
def test_inference_rejects_incomplete_checkpoint(checkpoint_model, tmp_path, defect):
    model, config = checkpoint_model
    state = model.state_dict()
    if defect == "missing":
        state.pop(next(iter(dict(model.named_parameters()))))
    else:
        state["model.nonexistent.weight"] = th.ones(1)
    path = tmp_path / "invalid.ckpt"
    th.save({"state_dict": state, "hyper_parameters": config}, path)
    with pytest.raises(RuntimeError, match="key"):
        init_model_from_ckpt(str(path), config, th.device("cpu"))
