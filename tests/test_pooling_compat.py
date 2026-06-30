"""Regression tests for pooling module interface compatibility."""

import torch as th

from fragnnet.model.nn_blocks import build_pool_module


def test_pool_module_supports_none_pool_with_dim_size():
    pool = build_pool_module("none", node_dim=4)
    x = th.randn(2, 4)
    index = th.tensor([0, 2], dtype=th.long)

    pooled = pool(x, index, dim_size=4)

    assert pooled.shape == (4, 4)
    assert th.count_nonzero(pooled) == 0


def test_pool_module_pads_missing_groups():
    pool = build_pool_module("mean", node_dim=3)
    x = th.tensor([[1.0, 2.0, 3.0], [3.0, 6.0, 9.0]])
    index = th.tensor([0, 2], dtype=th.long)

    pooled = pool(x, index, dim_size=4)

    expected = th.tensor(
        [
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            [3.0, 6.0, 9.0],
            [0.0, 0.0, 0.0],
        ]
    )
    assert pooled.shape == (4, 3)
    assert th.allclose(pooled, expected)
