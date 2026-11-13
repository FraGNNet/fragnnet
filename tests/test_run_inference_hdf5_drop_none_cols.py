"""`drop_none_cols` in `run_inference_hdf5`.

A column that the run never requested still reaches the writer, as one NaN per spectrum
(`process_batch_to_tabular_row` substitutes NaN for None). The eval script has always
exposed `--drop_none_cols`, but only the pkl path acted on it, so an HDF5 eval carried a
full float64 column of NaN for every unrequested output.

Dropping is decided on the FIRST flush and enforced afterwards: this writer appends to one
file, so a column skipped early and written late would be short by every earlier spectrum.
"""

import types

import h5py
import numpy as np
import pytest
import torch as th

from fragnnet.utils import script_utils

COLUMNS = [
    "spec_id",
    "mol_id",
    "pred_mzs",
    "pred_ints",
    "cos_sim",
    "pred_oos_prob",
    "index_rebased",
]


def _rows(specs, oos):
    """One batch worth of column-major rows; `oos` supplies pred_oos_prob per spectrum."""
    return [
        list(specs),
        [s * 10 for s in specs],
        [th.arange(2, dtype=th.float32) + s for s in specs],
        [th.ones(2, dtype=th.float32) * s for s in specs],
        [0.5 for _ in specs],
        list(oos),
        [True for _ in specs],
    ]


def _write(monkeypatch, tmp_path, batches, drop_none_cols=True, chunk_size=10_000):
    monkeypatch.setattr(script_utils, "get_tabular_columns", lambda *a, **k: COLUMNS)
    monkeypatch.setattr(script_utils, "run_inference_tabular_iter", lambda *a, **k: iter(batches))
    path = str(tmp_path / "pred.hdf5")
    script_utils.run_inference_hdf5(
        types.SimpleNamespace(hparams=None),
        dataloader=list(range(len(batches))),
        device=th.device("cpu"),
        chunk_size=chunk_size,
        save_path=path,
        drop_none_cols=drop_none_cols,
    )
    return path


def _keys(path):
    with h5py.File(path, "r") as f:
        return set(f.keys())


class TestPlaceholderColumns:
    def test_all_nan_column_is_not_written(self, monkeypatch, tmp_path):
        batches = [_rows([1, 2, 3], oos=[np.nan] * 3)]
        keys = _keys(_write(monkeypatch, tmp_path, batches))
        assert "pred_oos_prob" not in keys
        # the columns that carry values are untouched
        assert {"spec_id", "mol_id", "pred_mzs", "pred_ints", "cos_sim"} <= keys

    def test_opting_out_keeps_the_placeholder(self, monkeypatch, tmp_path):
        batches = [_rows([1, 2, 3], oos=[np.nan] * 3)]
        path = _write(monkeypatch, tmp_path, batches, drop_none_cols=False)
        with h5py.File(path, "r") as f:
            assert "pred_oos_prob" in f
            assert np.isnan(f["pred_oos_prob"][:]).all()

    def test_a_column_with_any_value_is_kept(self, monkeypatch, tmp_path):
        """One real value is enough: this is "no data at all", not "mostly missing"."""
        batches = [_rows([1, 2, 3], oos=[np.nan, 0.25, np.nan])]
        path = _write(monkeypatch, tmp_path, batches)
        with h5py.File(path, "r") as f:
            assert np.nanmax(f["pred_oos_prob"][:]) == pytest.approx(0.25)

    def test_non_float_columns_are_never_placeholders(self, monkeypatch, tmp_path):
        """index_rebased is all-False but bool, not NaN, so it must survive."""
        batches = [_rows([1, 2, 3], oos=[np.nan] * 3)]
        monkeypatch.setattr(script_utils, "get_tabular_columns", lambda *a, **k: COLUMNS)
        path = _write(monkeypatch, tmp_path, batches)
        with h5py.File(path, "r") as f:
            assert f["index_rebased"].dtype == np.bool_
            assert len(f["index_rebased"]) == 3


class TestDecisionIsMadeOnce:
    """A per-chunk decision would misalign the file, so the writer must not make one."""

    def _late_value_batches(self):
        # chunk_size=2 peaks per spectrum means each batch flushes on its own.
        return [_rows([1, 2], oos=[np.nan, np.nan]), _rows([3, 4], oos=[0.1, 0.2])]

    def test_late_values_in_a_dropped_column_raise(self, monkeypatch, tmp_path):
        with pytest.raises(RuntimeError, match="pred_oos_prob.*all-NaN placeholder"):
            _write(monkeypatch, tmp_path, self._late_value_batches(), chunk_size=1)

    def test_opting_out_writes_every_chunk_aligned(self, monkeypatch, tmp_path):
        path = _write(
            monkeypatch, tmp_path, self._late_value_batches(), drop_none_cols=False, chunk_size=1
        )
        with h5py.File(path, "r") as f:
            assert len(f["pred_oos_prob"]) == len(f["spec_id"]) == 4
            assert np.isnan(f["pred_oos_prob"][:2]).all()
            np.testing.assert_allclose(f["pred_oos_prob"][2:], [0.1, 0.2])

    def test_a_column_dropped_first_stays_dropped_when_still_empty(self, monkeypatch, tmp_path):
        batches = [_rows([1, 2], oos=[np.nan] * 2), _rows([3, 4], oos=[np.nan] * 2)]
        path = _write(monkeypatch, tmp_path, batches, chunk_size=1)
        with h5py.File(path, "r") as f:
            assert "pred_oos_prob" not in f
            assert len(f["spec_id"]) == 4
