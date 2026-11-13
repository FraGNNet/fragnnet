"""Strict and partial checkpoint transfer must validate before copying weights."""

from collections import OrderedDict
from pathlib import Path

import lightning as L
import pytest
import torch as th

from fragnnet.pl_model.fragnnet_pl import FraGNNetPL


@pytest.fixture
def model() -> FraGNNetPL:
    th.manual_seed(0)
    pl = FraGNNetPL.__new__(FraGNNetPL)
    L.LightningModule.__init__(pl)
    pl.model = th.nn.Sequential(OrderedDict(encoder=th.nn.Linear(2, 2), head=th.nn.Linear(2, 1)))
    pl.pretrained_param_names = []
    return pl


def save_checkpoint(path: Path, state: dict) -> None:
    th.save(
        {
            "state_dict": {f"model.{k}": v for k, v in state.items()},
            "hyper_parameters": {"compile": False},
        },
        path,
    )


@pytest.mark.parametrize("problem", ["missing", "unexpected", "shape"])
def test_strict_failure_does_not_modify_weights(model: FraGNNetPL, tmp_path: Path, problem: str) -> None:
    before = {k: v.clone() for k, v in model.model.state_dict().items()}
    state = {k: th.ones_like(v) for k, v in before.items()}
    if problem == "missing":
        del state["encoder.bias"]
    elif problem == "unexpected":
        state["extra.weight"] = th.ones(1)
    else:
        state["head.weight"] = th.ones(1, 3)
    path = tmp_path / "checkpoint.ckpt"
    save_checkpoint(path, state)
    with pytest.raises(RuntimeError):
        model._load_pretrained_weights(str(path), strict=True)
    for key, value in model.model.state_dict().items():
        assert th.equal(value, before[key])
    assert model.pretrained_param_names == []


def test_strict_selected_component_transfer(model: FraGNNetPL, tmp_path: Path) -> None:
    head_before = model.model.head.weight.clone()
    state = {
        k: th.ones_like(v) for k, v in model.model.state_dict().items() if k.startswith("encoder")
    }
    path = tmp_path / "checkpoint.ckpt"
    save_checkpoint(path, state)
    model._load_pretrained_weights(str(path), strict=True, shared_components=["encoder"])
    assert th.equal(model.model.encoder.weight, th.ones(2, 2))
    assert th.equal(model.model.head.weight, head_before)
    assert set(model.pretrained_param_names) == set(state)


def test_strict_unknown_component_rejected(model: FraGNNetPL, tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.ckpt"
    save_checkpoint(path, model.model.state_dict())
    with pytest.raises(RuntimeError):
        model._load_pretrained_weights(str(path), strict=True, shared_components=["typo"])


def test_non_strict_loads_only_compatible_tensors(model: FraGNNetPL, tmp_path: Path) -> None:
    head_before = model.model.head.weight.clone()
    path = tmp_path / "checkpoint.ckpt"
    save_checkpoint(path, {"encoder.weight": th.ones(2, 2), "head.weight": th.ones(1, 3)})
    model._load_pretrained_weights(str(path), strict=False)
    assert th.equal(model.model.encoder.weight, th.ones(2, 2))
    assert th.equal(model.model.head.weight, head_before)
    assert model.pretrained_param_names == ["encoder.weight"]


def test_strict_accepts_selected_parameterless_module(model: FraGNNetPL, tmp_path: Path) -> None:
    model.model.add_module("pool", th.nn.Identity())
    path = tmp_path / "checkpoint.ckpt"
    save_checkpoint(path, model.model.state_dict())
    model._load_pretrained_weights(str(path), strict=True, shared_components=["encoder", "pool"])
    assert set(model.pretrained_param_names) == {"encoder.weight", "encoder.bias"}


@pytest.mark.parametrize("compiled", [False, True])
def test_strict_full_load_ignores_lightning_state(model: FraGNNetPL, tmp_path: Path, compiled: bool) -> None:
    state = {k: th.ones_like(v) for k, v in model.model.state_dict().items()}
    prefix = "model._orig_mod." if compiled else "model."
    path = tmp_path / "checkpoint.ckpt"
    th.save(
        {
            "state_dict": {**{prefix + k: v for k, v in state.items()}, "metric.total": th.ones(1)},
            "hyper_parameters": {"compile": compiled},
        },
        path,
    )
    model._load_pretrained_weights(str(path), strict=True)
    for key, value in model.model.state_dict().items():
        assert th.equal(value, state[key])
