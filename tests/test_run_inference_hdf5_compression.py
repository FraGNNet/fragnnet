"""HDF5 datasets are chunked and compressed.

The export is dominated by index columns: small integers stored in int64, which byte-shuffle
into long runs of zeros and compress ~15-26x. Downcasting them to int32 would have bought 2x
and risked silent overflow, so the writer compresses instead and leaves every dtype alone.

The large chunk and the compression are one decision, not two: HDF5 allocates a whole chunk
as soon as any of it is written, so a 65536-row chunk holding a short column is pure waste
uncompressed and nearly free once the unwritten tail is compressed.
"""

import types

import h5py
import numpy as np
import torch as th

from fragnnet.utils import script_utils

COLUMNS = [
    "spec_id",
    "pred_mzs",
    "pred_ints",
    "pred_formula_probs",
    "pred_formula_formula_idxs",
    "pred_formula_strs",
]


def _batch(specs, n_formula=3):
    return [
        list(specs),
        [th.arange(4, dtype=th.float32) + s for s in specs],
        [th.ones(4, dtype=th.float32) * s for s in specs],
        [th.full((n_formula,), 1.0 / n_formula, dtype=th.float32) for _ in specs],
        [th.arange(n_formula, dtype=th.int64) for _ in specs],
        [[f"C{s}H{i}" for i in range(n_formula)] for s in specs],
    ]


def _write(monkeypatch, tmp_path, name="pred.hdf5", **kw):
    batches = [_batch([1, 2, 3]), _batch([4, 5, 6])]
    monkeypatch.setattr(script_utils, "get_tabular_columns", lambda *a, **k: COLUMNS)
    monkeypatch.setattr(script_utils, "run_inference_tabular_iter", lambda *a, **k: iter(batches))
    path = str(tmp_path / name)
    script_utils.run_inference_hdf5(
        types.SimpleNamespace(hparams=None),
        dataloader=[0, 1],
        device=th.device("cpu"),
        chunk_size=1,
        save_path=path,
        **kw,
    )
    return path


class TestCompressionIsOnByDefault:
    def test_every_dataset_is_chunked_and_compressed(self, monkeypatch, tmp_path):
        with h5py.File(_write(monkeypatch, tmp_path), "r") as f:
            assert set(f.keys())
            for name, d in f.items():
                assert d.chunks is not None, name
                assert d.compression == "gzip", name

    def test_shuffle_is_on_for_numbers_and_off_for_strings(self, monkeypatch, tmp_path):
        with h5py.File(_write(monkeypatch, tmp_path), "r") as f:
            assert f["pred_mzs"].shuffle is True
            assert f["pred_formula_formula_idxs"].shuffle is True
            # variable-length strings have no fixed element width to shuffle
            assert f["pred_formula_strs"].shuffle is False

    def test_chunk_length_is_the_configured_one(self, monkeypatch, tmp_path):
        with h5py.File(_write(monkeypatch, tmp_path), "r") as f:
            assert f["pred_mzs"].chunks == (script_utils._HDF5_CHUNK_LEN,)


class TestValuesSurvive:
    """Compression is lossless; nothing about the stored values may change."""

    def _expected_peaks(self):
        return np.concatenate([np.arange(4, dtype=np.float32) + s for s in [1, 2, 3, 4, 5, 6]])

    def test_floats_round_trip_exactly(self, monkeypatch, tmp_path):
        with h5py.File(_write(monkeypatch, tmp_path), "r") as f:
            np.testing.assert_array_equal(f["pred_mzs"][:], self._expected_peaks())
            assert f["pred_mzs"].dtype == np.float32

    def test_indices_keep_their_dtype(self, monkeypatch, tmp_path):
        """Compression replaces the int32 downcast, so int64 must be preserved."""
        with h5py.File(_write(monkeypatch, tmp_path), "r") as f:
            assert f["pred_formula_formula_idxs"].dtype == np.int64
            np.testing.assert_array_equal(f["pred_formula_formula_idxs"][:], [0, 1, 2] * 6)

    def test_variable_length_strings_round_trip(self, monkeypatch, tmp_path):
        with h5py.File(_write(monkeypatch, tmp_path), "r") as f:
            got = [s.decode() if isinstance(s, bytes) else s for s in f["pred_formula_strs"][:]]
        assert got == [f"C{s}H{i}" for s in [1, 2, 3, 4, 5, 6] for i in range(3)]

    def test_index_pairs_still_slice_each_spectrum(self, monkeypatch, tmp_path):
        with h5py.File(_write(monkeypatch, tmp_path), "r") as f:
            sp, ep = f["start_points"][:], f["end_points"][:]
            mzs = f["pred_mzs"][:]
            np.testing.assert_array_equal(mzs[sp[2] : ep[2]], np.arange(4, dtype=np.float32) + 3)


class TestOptingOut:
    def test_compression_none_stores_raw_and_still_round_trips(self, monkeypatch, tmp_path):
        path = _write(monkeypatch, tmp_path, name="raw.hdf5", compression=None)
        with h5py.File(path, "r") as f:
            assert f["pred_mzs"].compression is None
            np.testing.assert_array_equal(
                f["pred_mzs"][:],
                np.concatenate([np.arange(4, dtype=np.float32) + s for s in [1, 2, 3, 4, 5, 6]]),
            )

    def test_opting_out_does_not_take_the_large_chunk(self, monkeypatch, tmp_path):
        """A 65536-row chunk is only affordable because the empty tail compresses away."""
        path = _write(monkeypatch, tmp_path, name="raw2.hdf5", compression=None)
        with h5py.File(path, "r") as f:
            assert f["pred_mzs"].chunks != (script_utils._HDF5_CHUNK_LEN,)
