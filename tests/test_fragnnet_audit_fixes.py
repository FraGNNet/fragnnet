"""Regression tests for FraGNNet probability and auxiliary-loss contracts."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as th
import yaml
from test_cc_atom_dropout import _forward_kwargs
from test_ce_multi_location import _build
from torch_geometric.data import Batch

from fragnnet.pl_model.fragnnet_pl import FraGNNetPL
from fragnnet.utils.feat_utils import get_frag_graph, get_node_feats
from fragnnet.utils.frag_utils import compute_dags, extract_mol_info


@pytest.fixture
def loss_config():
    with (Path(__file__).parents[1] / "config/template.yml").open() as stream:
        config = yaml.safe_load(stream)
    config.update(sparse_cosine_similarity=True, oos_tolerance_multiple=1, gaussian_renormalize=True)
    return config


@pytest.mark.parametrize("head", ["formula", "node_formula"])
@pytest.mark.parametrize("mask_null", [False, True])
def test_explanation_probabilities_match_spectrum(head, mask_null):
    th.manual_seed(0)
    model = _build(mlp_output_format=head, mask_null_formula=mask_null).eval()
    inputs = _forward_kwargs("CCCO", frag_node_feats=model.frag_node_feats)
    output = model(**inputs)
    spectrum = output["pred_logprobs"].exp()
    expected = spectrum / spectrum.sum()
    actual = output["pred_formula_logprobs"][inputs["frag_formula_peak_idxs"]].exp()
    th.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)


def test_conditional_logit_offsets_preserve_spectrum_and_explanations():
    th.manual_seed(0)
    model = _build(mlp_output_format="node_formula").eval()
    inputs = _forward_kwargs("CCCO", frag_node_feats=model.frag_node_feats)
    before = model(**inputs)

    def shift_logits(_module, _inputs, logits):
        return logits + th.arange(logits.shape[0], device=logits.device)[:, None] * 0.2

    handle = model.formula_module.register_forward_hook(shift_logits)
    after = model(**inputs)
    handle.remove()
    for key in ["pred_logprobs", "pred_formula_logprobs", "pred_node_logprobs", "pred_joint_logprobs"]:
        th.testing.assert_close(before[key].exp(), after[key].exp(), atol=1e-7, rtol=1e-5)
    (-after["pred_joint_logprobs"].mean()).backward()
    assert model.formula_module.input_layer.weight.grad.abs().sum() > 0
    assert model.node_module.input_layer.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("include_cmf", [False, True])
def test_edge_hydrogen_pairs_and_loss_match_explicit_enumeration(loss_config, include_cmf):
    th.manual_seed(0)
    features = ["cc", "base_formula", "depth", "h_formulae_idx", "h_counts"]
    if include_cmf:
        features.append("cmf_h_formulae_idx")
    model = _build(frag_node_feats=features, frag_edge_feats=["h_range"], skip_edge_loss=False).eval()
    inputs = _forward_kwargs("CCCO", frag_node_feats=features)
    dag = compute_dags(extract_mol_info("CCCO"), 3, True, 2, 100, multi_cut_bfs=False)
    graph = get_frag_graph(
        dag["dag"], frag_node_feats=features, frag_edge_feats=["h_range"], edges=True, bigraph=False
    )
    inputs["frag_pyg"] = Batch.from_data_list([graph])
    logits = []
    handle = model.formula_module.register_forward_hook(
        lambda _module, _inputs, output: logits.append(output.detach())
    )
    output = model(**inputs)
    handle.remove()
    counts = get_node_feats(graph.x, graph.node_feat_idxs.view(-1), "h_counts")
    # CRF channels alone describe edge H transfers, even when CMF output is enabled.
    if include_cmf:
        # Every unavailable CMF ion has zero mass, Δh=0 included: an adduct whose CMF ion
        # duplicates a CRF hydrogen shift leaves its whole CMF block NULL.
        cmf_indices = get_node_feats(graph.x, graph.node_feat_idxs.view(-1), "cmf_h_formulae_idx")
        logits[0][:, 5:] = logits[0][:, 5:].masked_fill(cmf_indices == 0, -th.inf)
    probabilities = logits[0].softmax(dim=1)[:, :5]
    expected_diffs = []
    expected_probs = []
    for source, target in graph.edge_index.T.tolist():
        expected_diffs.append([int(a - b) for a in counts[source] for b in counts[target]])
        expected_probs.append([float(a * b) for a in probabilities[source] for b in probabilities[target]])
    expected_diffs = th.tensor(expected_diffs)
    expected_probs = th.tensor(expected_probs)
    th.testing.assert_close(output["pred_edge_h_diffs"], expected_diffs, check_dtype=False)
    th.testing.assert_close(output["pred_edge_h_logprobs"].exp(), expected_probs)
    assert output["pred_edge_h_logprobs"].max() <= 0

    config = dict(loss_config, skip_edge_loss=False, edge_h_transfer_loss_weight=0.1)
    pl = SimpleNamespace(
        hparams=SimpleNamespace(**config), metric_names=set(),
        tolerance=1e-5, relative=True, tolerance_min_mz=200.0,
    )
    pl._setup_loss_names = lambda: FraGNNetPL._setup_loss_names(pl)
    FraGNNetPL._setup_loss_fn(pl)
    losses = pl.loss_fn(
        true_mzs=output["pred_mzs"][:1].detach(), true_logprobs=th.zeros(1),
        true_batch_idxs=th.zeros(1, dtype=th.long), batch_size=1, **output,
    )
    expected_cost = (
        output["pred_edge_logprobs"].exp()
        * (expected_probs * 4 * expected_diffs.abs()).sum(dim=1)
    ).sum()
    th.testing.assert_close(losses["edge_h_transfer_loss"].exp(), expected_cost.reshape(1))
    losses["loss"].sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(th.isfinite(grad).all() for grad in grads)
    assert any(grad.abs().sum() > 0 for grad in grads)


# NOTE: the two `skip_train_explainability` guard tests from feiw_v2 are omitted here.
# That hparam is part of the training-memory work that this branch does not carry,
# so there is no flag for the guard to reject.
