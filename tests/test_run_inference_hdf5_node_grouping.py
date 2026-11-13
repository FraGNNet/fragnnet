"""Tests for node-level grouping in `run_inference_hdf5`.

Node-level per-spectrum arrays (`pred_node_probs`, `pred_node_node_idxs`) must be
saved with their own running offset and per-spectrum `node_start_points` /
`node_end_points`, so P(n) can be sliced per spectrum and aligned to each DAG.
Previously they were written at the per-spectrum offset like scalar columns, which
misaligns them across chunk flushes.

The writer's model/tabular machinery is bypassed by monkeypatching the row iterator
and column list, so these tests exercise only the flush/offset/chunking logic.
"""

import types

import numpy as np
import pytest
import torch as th

from fragnnet.utils import script_utils

COLUMNS = [
    "spec_id",
    "mol_id",
    "pred_mzs",
    "pred_ints",
    "pred_node_probs",
    "pred_node_node_idxs",
]

# Five spectra with deliberately varying node/peak block sizes so a per-spectrum
# vs per-node offset confusion cannot go unnoticed. Node probs sum to 1 per spectrum.
SPECTRA = [
    {"spec": 100, "mol": 10, "n_nodes": 3, "n_peaks": 2},
    {"spec": 101, "mol": 10, "n_nodes": 2, "n_peaks": 4},
    {"spec": 102, "mol": 11, "n_nodes": 5, "n_peaks": 1},
    {"spec": 103, "mol": 12, "n_nodes": 1, "n_peaks": 3},
    {"spec": 104, "mol": 13, "n_nodes": 4, "n_peaks": 2},
]


def _spectrum_row(s):
    probs = th.full((s["n_nodes"],), 1.0 / s["n_nodes"], dtype=th.float64)
    nidx = th.arange(s["n_nodes"], dtype=th.int64)  # per-spectrum DAG node ids
    mzs = th.arange(s["n_peaks"], dtype=th.float32) + 10.0 * s["spec"]
    ints = th.ones(s["n_peaks"], dtype=th.float32)
    return s["spec"], s["mol"], mzs, ints, probs, nidx


def _make_batches(batch_sizes):
    """Group SPECTRA into batches of the given sizes; yield column-major batch_rows."""
    batches = []
    cur = 0
    for bs in batch_sizes:
        rows = [_spectrum_row(s) for s in SPECTRA[cur : cur + bs]]
        # transpose rows -> columns, each column a list over spectra
        batches.append([list(col) for col in zip(*rows)])
        cur += bs
    return batches


@pytest.fixture
def patched_writer(monkeypatch):
    def _run(batch_sizes, chunk_size, tmp_path):
        batches = _make_batches(batch_sizes)
        monkeypatch.setattr(script_utils, "get_tabular_columns", lambda *a, **k: COLUMNS)
        monkeypatch.setattr(
            script_utils, "run_inference_tabular_iter", lambda *a, **k: iter(batches)
        )
        model = types.SimpleNamespace(hparams=None)
        save_path = str(tmp_path / "pred.hdf5")
        script_utils.run_inference_hdf5(
            model,
            dataloader=list(range(len(batches))),  # only len() is used
            device=th.device("cpu"),
            output_set={"pred_node_probs"},
            chunk_size=chunk_size,
            save_path=save_path,
        )
        return save_path

    return _run


def _read(save_path):
    import h5py

    with h5py.File(save_path, "r") as f:
        return {k: f[k][:] for k in f}


def test_node_grouping_single_chunk(patched_writer, tmp_path):
    # chunk_size huge -> one flush; still must produce correct node boundaries.
    out = _read(patched_writer([2, 2, 1], chunk_size=1000, tmp_path=tmp_path))
    _assert_node_alignment(out)


def test_node_grouping_across_chunks(patched_writer, tmp_path):
    # chunk_size=2 -> flush after each 2-spectrum batch (3 chunks): this is the case
    # the old code corrupted, because node arrays were placed at the spectrum offset.
    out = _read(patched_writer([2, 2, 1], chunk_size=2, tmp_path=tmp_path))
    _assert_node_alignment(out)


def _assert_node_alignment(out):
    assert "node_start_points" in out and "node_end_points" in out
    ns, ne = out["node_start_points"], out["node_end_points"]
    probs, nidx = out["pred_node_probs"], out["pred_node_node_idxs"]
    # one (start,end) per spectrum, contiguous, covering the whole flat node array
    assert len(ns) == len(SPECTRA)
    assert np.array_equal(ns[1:], ne[:-1])
    assert ns[0] == 0 and ne[-1] == len(probs) == sum(s["n_nodes"] for s in SPECTRA)
    # each spectrum's block: right size, P(n) sums to 1, node idxs are 0..k-1
    for i, s in enumerate(SPECTRA):
        block = probs[ns[i] : ne[i]]
        assert len(block) == s["n_nodes"]
        assert block.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.array_equal(nidx[ns[i] : ne[i]], np.arange(s["n_nodes"]))


def test_peak_and_node_offsets_are_independent(patched_writer, tmp_path):
    # Peak blocks and node blocks have different lengths; both must stay aligned.
    out = _read(patched_writer([2, 2, 1], chunk_size=2, tmp_path=tmp_path))
    sp, ep = out["start_points"], out["end_points"]
    assert ep[-1] == sum(s["n_peaks"] for s in SPECTRA)
    for i, s in enumerate(SPECTRA):
        assert ep[i] - sp[i] == s["n_peaks"]
    # spec_id/mol_id retained per spectrum
    assert np.array_equal(out["spec_id"], [s["spec"] for s in SPECTRA])
    assert np.array_equal(out["mol_id"], [s["mol"] for s in SPECTRA])


@pytest.mark.parametrize("cutoff", [0, 1, 2, 3, 10, -1])
@pytest.mark.parametrize("chunk_size", [1, 3, 10000])
def test_cutoff_flushes_all_processed_spectra(monkeypatch, tmp_path, cutoff, chunk_size):
    batches = _make_batches([2, 1, 2])
    count = len(batches) if cutoff < 0 else min(cutoff, len(batches))
    selected = batches[:count]
    monkeypatch.setattr(script_utils, "get_tabular_columns", lambda *a, **k: COLUMNS)
    monkeypatch.setattr(script_utils, "run_inference_tabular_iter", lambda *a, **k: iter(selected))
    path = str(tmp_path / "cutoff.h5")
    written = script_utils.run_inference_hdf5(
        types.SimpleNamespace(hparams=None),
        list(range(len(batches))),
        th.device("cpu"),
        batch_cutoff=cutoff,
        chunk_size=chunk_size,
        save_path=path,
    )
    expected = sum(len(batch[0]) for batch in selected)
    output = _read(path)
    assert written == expected
    assert len(output["start_points"]) == expected
    if expected:
        assert output["spec_id"].tolist() == [s["spec"] for s in SPECTRA[:expected]]
        assert output["end_points"][-1] == sum(s["n_peaks"] for s in SPECTRA[:expected])
        assert output["node_end_points"][-1] == sum(s["n_nodes"] for s in SPECTRA[:expected])


@pytest.mark.parametrize("cutoff, expected", [(0, 0), (1, 1), (10, 3), (-1, 3)])
def test_tabular_iterator_respects_cutoff(monkeypatch, cutoff, expected):
    monkeypatch.setattr(script_utils, "get_idx_vars", lambda *a, **kw: ([], [], [], []))
    monkeypatch.setattr(script_utils, "get_score_types", lambda *a: [])
    processed = []

    def process(**kwargs):
        processed.append(kwargs["batch_input"])
        return kwargs["batch_input"]

    monkeypatch.setattr(script_utils, "process_batch_to_tabular_row", process)
    model = types.SimpleNamespace(training=False, hparams={}, eval=lambda: None)
    result = list(
        script_utils.run_inference_tabular_iter(
            model,
            [0, 1, 2],
            th.device("cpu"),
            batch_cutoff=cutoff,
        )
    )
    assert result == processed == list(range(expected))


@pytest.mark.parametrize("reported_batches", [1, 5])
def test_writer_flushes_on_exhaustion_with_estimated_length(
    monkeypatch, tmp_path, reported_batches
):
    batches = _make_batches([2, 1, 2])
    monkeypatch.setattr(script_utils, "get_tabular_columns", lambda *a, **k: COLUMNS)
    monkeypatch.setattr(script_utils, "run_inference_tabular_iter", lambda *a, **k: iter(batches))
    path = str(tmp_path / "estimated_length.h5")
    written = script_utils.run_inference_hdf5(
        types.SimpleNamespace(hparams=None),
        list(range(reported_batches)),
        th.device("cpu"),
        chunk_size=3,
        save_path=path,
    )
    assert written == 5
    _assert_node_alignment(_read(path))
