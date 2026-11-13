"""Tests for the per-graph global attention used by the GINE_GSA encoder."""

from __future__ import annotations

import math

import pytest
import torch as th

from fragnnet.model.nn_blocks import GINE_GSA, _graph_softmax_attn


def _reference_attention(
    queries: th.Tensor,
    keys: th.Tensor,
    values: th.Tensor,
    batch_index: th.Tensor,
    num_graphs: int,
) -> th.Tensor:
    """Per-graph softmax attention written straight from the definition."""
    head_dim = queries.shape[-1]
    out = th.zeros_like(values)
    for graph in range(num_graphs):
        members = (batch_index == graph).nonzero().flatten()
        for head in range(queries.shape[1]):
            scores = (queries[members, head] @ keys[members, head].T) / math.sqrt(head_dim)
            out[members, head] = th.softmax(scores, dim=-1) @ values[members, head]
    return out


def _random_qkv(
    num_nodes: int,
    num_heads: int,
    head_dim: int,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    return tuple(th.randn(num_nodes, num_heads, head_dim, dtype=th.float64) for _ in range(3))


def test_graph_softmax_attn_matches_definition() -> None:
    th.manual_seed(11)
    batch_index = th.tensor([0, 0, 0, 1, 1, 2])
    queries, keys, values = _random_qkv(6, 3, 8)

    attended = _graph_softmax_attn(queries, keys, values, batch_index, 3)
    expected = _reference_attention(queries, keys, values, batch_index, 3)

    assert attended.shape == values.shape
    assert th.allclose(attended, expected, atol=1e-12)


def test_graph_softmax_attn_isolates_graphs() -> None:
    """Padded keys must be masked, so one molecule cannot reach another."""
    th.manual_seed(12)
    batch_index = th.tensor([0, 0, 0, 1, 1, 2])
    queries, keys, values = _random_qkv(6, 3, 8)

    baseline = _graph_softmax_attn(queries, keys, values, batch_index, 3)
    perturbed_inputs = [tensor.clone() for tensor in (queries, keys, values)]
    for tensor in perturbed_inputs:
        tensor[3:] += 50.0
    perturbed = _graph_softmax_attn(*perturbed_inputs, batch_index, 3)

    assert th.allclose(baseline[:3], perturbed[:3], atol=1e-12)
    assert not th.allclose(baseline[3:], perturbed[3:], atol=1e-6)


def test_graph_softmax_attn_is_permutation_equivariant_within_a_graph() -> None:
    th.manual_seed(13)
    batch_index = th.tensor([0, 0, 0, 1, 1, 2])
    queries, keys, values = _random_qkv(6, 3, 8)
    permutation = th.tensor([2, 0, 1, 3, 4, 5])

    baseline = _graph_softmax_attn(queries, keys, values, batch_index, 3)
    permuted = _graph_softmax_attn(
        queries[permutation],
        keys[permutation],
        values[permutation],
        batch_index,
        3,
    )

    assert th.allclose(baseline[permutation], permuted, atol=1e-12)


def test_graph_softmax_attn_returns_own_value_for_a_singleton_graph() -> None:
    """A one-node graph attends only to itself, so softmax is the identity on V."""
    th.manual_seed(14)
    batch_index = th.tensor([0, 0, 1])
    queries, keys, values = _random_qkv(3, 2, 4)

    attended = _graph_softmax_attn(queries, keys, values, batch_index, 2)

    assert th.allclose(attended[2], values[2], atol=1e-12)


def test_graph_softmax_attn_ignores_dense_padding() -> None:
    """A small molecule's output must not depend on how large its batch mates are."""
    th.manual_seed(15)
    small_queries, small_keys, small_values = _random_qkv(2, 2, 4)
    large_queries, large_keys, large_values = _random_qkv(9, 2, 4)

    alone = _graph_softmax_attn(
        small_queries,
        small_keys,
        small_values,
        th.zeros(2, dtype=th.long),
        1,
    )
    padded = _graph_softmax_attn(
        th.cat([small_queries, large_queries]),
        th.cat([small_keys, large_keys]),
        th.cat([small_values, large_values]),
        th.tensor([0, 0] + [1] * 9),
        2,
    )

    assert th.allclose(alone, padded[:2], atol=1e-12)


def test_graph_softmax_attn_tolerates_an_empty_trailing_graph() -> None:
    """A graph with no nodes must not turn a fully masked softmax row into NaN."""
    th.manual_seed(18)
    batch_index = th.tensor([0, 0, 2])
    queries, keys, values = _random_qkv(3, 2, 4)

    attended = _graph_softmax_attn(queries, keys, values, batch_index, 3)

    assert th.isfinite(attended).all()
    assert th.allclose(attended[2], values[2], atol=1e-12)


def test_graph_softmax_attn_rejects_unsorted_batch_index() -> None:
    """Dense packing assumes graph-contiguous nodes; an unsorted batch must fail loudly."""
    queries, keys, values = _random_qkv(4, 2, 4)

    with pytest.raises(ValueError, match="batch_index must be sorted"):
        _graph_softmax_attn(queries, keys, values, th.tensor([0, 1, 0, 1]), 2)


@pytest.mark.parametrize("normalization", ["layer", "graph"])
def test_gine_gsa_forward_and_backward(normalization: str) -> None:
    th.manual_seed(16)
    hidden_size, num_heads = 16, 4
    encoder = GINE_GSA(
        hidden_size=hidden_size,
        node_feats_size=hidden_size,
        edge_feats_size=3,
        num_layers=2,
        dropout=0.0,
        normalization=normalization,
        num_heads=num_heads,
    ).eval()
    x = th.randn(5, hidden_size, requires_grad=True)
    batch_index = th.tensor([0, 0, 0, 1, 1])
    edge_index = th.tensor([[0, 1, 1, 2, 3], [1, 0, 2, 1, 4]])
    edge_attr = th.randn(edge_index.shape[1], 3)

    out = encoder(x, batch_index, edge_index, edge_attr)
    out.square().sum().backward()

    assert out.shape == (5, hidden_size)
    assert th.isfinite(out).all()
    assert x.grad is not None and th.isfinite(x.grad).all()
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None, name
        assert th.isfinite(parameter.grad).all(), name
    assert any(
        encoder.get_parameter(name).grad.abs().sum() > 0
        for name, _ in encoder.named_parameters()
        if name.startswith("gsa_")
    )


def test_gine_gsa_keeps_molecules_independent() -> None:
    """A second molecule's features must not change the first molecule's output."""
    th.manual_seed(17)
    hidden_size = 16
    encoder = GINE_GSA(
        hidden_size=hidden_size,
        node_feats_size=hidden_size,
        edge_feats_size=3,
        num_layers=2,
        dropout=0.0,
        normalization="layer",
        num_heads=4,
    ).eval()
    x = th.randn(5, hidden_size)
    batch_index = th.tensor([0, 0, 0, 1, 1])
    edge_index = th.tensor([[0, 1, 1, 2, 3], [1, 0, 2, 1, 4]])
    edge_attr = th.randn(edge_index.shape[1], 3)

    baseline = encoder(x, batch_index, edge_index, edge_attr)
    perturbed_x = x.clone()
    perturbed_x[3:] += 20.0
    perturbed = encoder(perturbed_x, batch_index, edge_index, edge_attr)

    assert th.allclose(baseline[:3], perturbed[:3], atol=1e-5)
