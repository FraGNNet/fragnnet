"""Every ragged per-spectrum family in `run_inference_hdf5` needs its own running offset.

Peak, node, formula and joint arrays each contribute a variable-length block per spectrum,
and the four block sizes are unrelated. Writing any of them at the per-spectrum offset (the
one that is correct for scalar columns) makes each chunk flush clobber the blocks written by
the previous flush, so the defect only appears once a run exceeds chunk_size.

The sibling module test_run_inference_hdf5_node_grouping.py covers the peak and node
families; this one covers formula, joint, the 2-D `pred_node_ccs`, and the ragged-string
`pred_formula_strs` / `pred_formula_adducts`, and it checks every family round-trips per
spectrum.
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
    "pred_node_ccs",
    "pred_formula_probs",
    "pred_formula_formula_idxs",
    "pred_formula_strs",
    "pred_formula_adducts",
    "pred_joint_formula_idxs",
    "pred_joint_node_idxs",
]

# The four block sizes differ within every spectrum AND across spectra, so no family's
# offset can stand in for another's and still produce the right answer.
SPECTRA = [
    {"spec": 200, "mol": 20, "n_peaks": 2, "n_nodes": 3, "n_formulas": 5, "n_joint": 7},
    {"spec": 201, "mol": 20, "n_peaks": 4, "n_nodes": 1, "n_formulas": 2, "n_joint": 9},
    {"spec": 202, "mol": 21, "n_peaks": 1, "n_nodes": 5, "n_formulas": 4, "n_joint": 3},
    {"spec": 203, "mol": 22, "n_peaks": 3, "n_nodes": 2, "n_formulas": 6, "n_joint": 2},
    {"spec": 204, "mol": 23, "n_peaks": 5, "n_nodes": 4, "n_formulas": 1, "n_joint": 8},
    # n_formulas here keeps the four family totals distinct (17/21/22/34), so a mix-up
    # between two families cannot coincidentally produce the right total.
    {"spec": 205, "mol": 24, "n_peaks": 2, "n_nodes": 6, "n_formulas": 4, "n_joint": 5},
]


def _blocks(s):
    """Per-spectrum blocks whose values encode the spectrum id, so mix-ups are detectable."""
    tag = s["spec"] * 1000
    return {
        "spec_id": s["spec"],
        "mol_id": s["mol"],
        "pred_mzs": th.arange(s["n_peaks"], dtype=th.float64) + tag,
        "pred_ints": th.ones(s["n_peaks"], dtype=th.float64) * s["spec"],
        "pred_node_probs": th.full((s["n_nodes"],), 1.0 / s["n_nodes"], dtype=th.float64),
        "pred_node_node_idxs": th.arange(s["n_nodes"], dtype=th.int64) + tag,
        # 2-D: one (hash_hi, hash_lo) pair per DAG node.
        "pred_node_ccs": th.stack(
            [
                th.arange(s["n_nodes"], dtype=th.int64) + tag,
                th.arange(s["n_nodes"], dtype=th.int64) + tag + 500,
            ],
            dim=1,
        ),
        "pred_formula_probs": th.full((s["n_formulas"],), 1.0 / s["n_formulas"], dtype=th.float64),
        "pred_formula_formula_idxs": th.arange(s["n_formulas"], dtype=th.int64) + tag,
        "pred_formula_strs": [f"C{s['spec']}H{i}" for i in range(s["n_formulas"])],
        # CRF rows then a CMF row, so the family carries two distinct delta strings.
        "pred_formula_adducts": [
            "+H" if i < s["n_formulas"] - 1 else "+Na" for i in range(s["n_formulas"])
        ],
        "pred_joint_formula_idxs": th.arange(s["n_joint"], dtype=th.int64) + tag,
        "pred_joint_node_idxs": th.arange(s["n_joint"], dtype=th.int64) + tag + 100,
    }


def _make_batches(batch_sizes):
    batches, cur = [], 0
    for bs in batch_sizes:
        rows = [_blocks(s) for s in SPECTRA[cur : cur + bs]]
        batches.append([[r[c] for r in rows] for c in COLUMNS])
        cur += bs
    return batches


def _write(monkeypatch, tmp_path, batch_sizes, chunk_size):
    batches = _make_batches(batch_sizes)
    monkeypatch.setattr(script_utils, "get_tabular_columns", lambda *a, **k: COLUMNS)
    monkeypatch.setattr(script_utils, "run_inference_tabular_iter", lambda *a, **k: iter(batches))
    path = str(tmp_path / "pred.hdf5")
    script_utils.run_inference_hdf5(
        types.SimpleNamespace(hparams=None),
        dataloader=list(range(len(batches))),
        device=th.device("cpu"),
        chunk_size=chunk_size,
        save_path=path,
    )
    return path


def _read(path):
    import h5py

    with h5py.File(path, "r") as f:
        return {k: f[k][:] for k in f}


FAMILIES = [
    ("", "n_peaks", ["pred_mzs", "pred_ints"]),
    ("node_", "n_nodes", ["pred_node_probs", "pred_node_node_idxs", "pred_node_ccs"]),
    (
        "formula_",
        "n_formulas",
        [
            "pred_formula_probs",
            "pred_formula_formula_idxs",
            "pred_formula_strs",
            "pred_formula_adducts",
        ],
    ),
    ("joint_", "n_joint", ["pred_joint_formula_idxs", "pred_joint_node_idxs"]),
]

# chunk_size=2 and 3 both force several flushes over 6 spectra; 10 flushes once (control).
FLUSH_CASES = [([2, 2, 2], 2), ([3, 3], 2), ([1, 2, 3], 3), ([6], 2), ([2, 2, 2], 10)]


@pytest.mark.parametrize("batch_sizes, chunk_size", FLUSH_CASES)
def test_every_family_round_trips_per_spectrum(monkeypatch, tmp_path, batch_sizes, chunk_size):
    """Slicing by <prefix>start/end_points must return exactly what went in, for each family."""
    out = _read(_write(monkeypatch, tmp_path, batch_sizes, chunk_size))

    assert list(out["spec_id"]) == [s["spec"] for s in SPECTRA]

    for prefix, size_key, columns in FAMILIES:
        starts = out[f"{prefix}start_points"]
        ends = out[f"{prefix}end_points"]
        assert len(starts) == len(SPECTRA), f"{prefix}start_points has wrong length"
        # Blocks must tile the flat array contiguously, in spectrum order.
        assert starts[0] == 0
        assert list(starts[1:]) == list(ends[:-1]), f"{prefix} blocks are not contiguous"
        assert ends[-1] == sum(s[size_key] for s in SPECTRA), f"{prefix} total length wrong"

        for i, s in enumerate(SPECTRA):
            assert ends[i] - starts[i] == s[size_key], f"{prefix} block {i} wrong size"
            expected = _blocks(s)
            for col in columns:
                got = out[col][starts[i] : ends[i]]
                want = expected[col]
                if col in ("pred_formula_strs", "pred_formula_adducts"):
                    assert [b.decode() for b in got] == want, f"{col} spectrum {i}"
                else:
                    np.testing.assert_array_equal(got, want.numpy(), err_msg=f"{col} spec {i}")


def test_node_ccs_keeps_its_second_dimension(monkeypatch, tmp_path):
    """pred_node_ccs is (n_nodes, 2); a 1-D-only writer cannot store it."""
    out = _read(_write(monkeypatch, tmp_path, [2, 2, 2], 2))
    assert out["pred_node_ccs"].ndim == 2
    assert out["pred_node_ccs"].shape == (sum(s["n_nodes"] for s in SPECTRA), 2)


def test_missing_adducts_in_first_chunk_keep_formula_alignment(monkeypatch, tmp_path):
    batches = _make_batches([2, 2, 2])
    adduct_col = COLUMNS.index("pred_formula_adducts")
    for i in range(2):
        batches[0][adduct_col][i] = [np.nan] * SPECTRA[i]["n_formulas"]
    monkeypatch.setattr(script_utils, "get_tabular_columns", lambda *a, **k: COLUMNS)
    monkeypatch.setattr(script_utils, "run_inference_tabular_iter", lambda *a, **k: iter(batches))
    path = str(tmp_path / "missing_adducts.hdf5")
    script_utils.run_inference_hdf5(
        types.SimpleNamespace(hparams=None),
        dataloader=list(range(len(batches))),
        device=th.device("cpu"),
        chunk_size=2,
        save_path=path,
    )
    out = _read(path)
    starts, ends = out["formula_start_points"], out["formula_end_points"]
    for i, spectrum in enumerate(SPECTRA):
        got = [value.decode() for value in out["pred_formula_adducts"][starts[i] : ends[i]]]
        want = (
            ["NaN"] * spectrum["n_formulas"]
            if i < 2
            else _blocks(spectrum)["pred_formula_adducts"]
        )
        assert got == want


def test_families_do_not_share_an_offset(monkeypatch, tmp_path):
    """The four families have genuinely different totals, so the fixture can detect a mix-up."""
    out = _read(_write(monkeypatch, tmp_path, [2, 2, 2], 2))
    totals = {prefix: int(out[f"{prefix}end_points"][-1]) for prefix, _, _ in FAMILIES}
    assert len(set(totals.values())) == len(totals), f"fixture is too weak: {totals}"


def test_absent_family_writes_no_points(monkeypatch, tmp_path):
    """A family nobody requested must not leave empty start/end point datasets behind."""
    cols = ["spec_id", "mol_id", "pred_mzs", "pred_ints"]
    batches = []
    for start in range(0, len(SPECTRA), 2):
        rows = [_blocks(s) for s in SPECTRA[start : start + 2]]
        batches.append([[r[c] for r in rows] for c in cols])
    monkeypatch.setattr(script_utils, "get_tabular_columns", lambda *a, **k: cols)
    monkeypatch.setattr(script_utils, "run_inference_tabular_iter", lambda *a, **k: iter(batches))
    path = str(tmp_path / "sparse.hdf5")
    script_utils.run_inference_hdf5(
        types.SimpleNamespace(hparams=None),
        dataloader=list(range(len(batches))),
        device=th.device("cpu"),
        chunk_size=2,
        save_path=path,
    )
    out = _read(path)
    assert "start_points" in out
    for prefix in ("node_", "formula_", "joint_"):
        assert f"{prefix}start_points" not in out, f"unexpected {prefix}start_points"
