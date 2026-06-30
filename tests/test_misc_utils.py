"""Unit tests for fragnnet.utils.misc_utils module.

Tests for general utility functions and helpers.
"""

import math

import numpy as np
import pandas as pd
import pytest
import torch as th

from fragnnet.utils import misc_utils


def test_none_or_nan_variants() -> None:
    assert misc_utils.none_or_nan(None)
    assert misc_utils.none_or_nan("")
    assert misc_utils.none_or_nan(float("nan"))
    assert misc_utils.none_or_nan(np.array([np.nan]))
    assert misc_utils.none_or_nan(pd.Series([np.nan]))
    assert misc_utils.none_or_nan(pd.DataFrame({"a": [np.nan]}))
    assert misc_utils.none_or_nan([])
    assert not misc_utils.none_or_nan([1])
    assert not misc_utils.none_or_nan(0)
    assert not misc_utils.none_or_nan("value")


def test_flatten_lol() -> None:
    assert misc_utils.flatten_lol([[1, 2], [3]]) == [1, 2, 3]


def test_deep_update_merges_nested_dicts() -> None:
    base = {"a": 1, "b": {"c": 2}}
    updated = misc_utils.deep_update(base, {"b": {"d": 3}}, {"e": 4})
    assert updated == {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}


def test_scatter_logsumexp_matches_manual() -> None:
    logits = th.tensor([0.0, 1.0, 0.0])
    subset_idxs = th.tensor([0, 0, 1])
    out = misc_utils.scatter_logsumexp(logits, subset_idxs)
    expected_group0 = math.log(1.0 + math.e)  # logsumexp([0, 1])
    expected_group1 = 0.0  # logsumexp([0])
    assert th.isclose(out[0], th.tensor(expected_group0), atol=1e-6)
    assert th.isclose(out[1], th.tensor(expected_group1), atol=1e-6)


def test_to_device_and_to_cpu_roundtrip() -> None:
    data = {"t": th.tensor([1, 2, 3])}
    on_device = misc_utils.to_device(data.copy(), device="cpu")
    assert on_device["t"].device.type == "cpu"

    back_on_cpu = misc_utils.to_cpu(on_device.copy())
    assert back_on_cpu["t"].device.type == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
