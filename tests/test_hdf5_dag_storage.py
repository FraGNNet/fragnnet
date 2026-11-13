"""Tests for HDF5 DAG serialization round-trip."""

import os
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import torch_geometric as pyg

from fragnnet.utils.frag_utils import (
    PACKED_DAG_BYTES_DATASET,
    PACKED_DAG_FORMAT,
    PACKED_DAG_FORMAT_ATTR,
    PACKED_DAG_INDEX_DATASET,
    append_packed_dag_bytes,
    dump_dag_hdf5,
    dump_dag_hdf5_packed,
    initialize_packed_dag_hdf5,
    load_dag_hdf5,
    load_dag_hdf5_from_group,
    merge_packed_dag_hdf5,
    pack_dag_payload,
    run_frag_gen_hdf5,
    run_frag_gen_hdf5_parts,
    timed_get_dags_packed,
)


def _packed(dag_d: dict):
    """What :func:`timed_get_dags_packed` returns for ``dag_d`` (None when it failed)."""
    if len(dag_d) == 0:
        return None
    return pack_dag_payload(dag_d), dag_d["dag"].num_nodes, dag_d["dag"].num_edges


def _make_dag_d(num_nodes: int, num_edges: int, num_formulae: int) -> dict:
    """Build a minimal dag_d dict matching the structure from compute_dags."""
    node_feat_dim = 10
    edge_feat_dim = 8
    num_isotopes = 5
    max_h_transfer = 4

    x = torch.randint(0, 100, (num_nodes, node_feat_dim), dtype=torch.int64)
    if num_edges > 0:
        edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.int64)
        edge_attr = torch.randint(0, 50, (num_edges, edge_feat_dim), dtype=torch.int64)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.int64)
        edge_attr = torch.zeros((0, edge_feat_dim), dtype=torch.int64)

    dag = pyg.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    dag.node_feat_idxs = torch.tensor([[0, 3, 6, 10]], dtype=torch.long)
    dag.edge_feat_idxs = torch.tensor([[0, 2, 5, 8]], dtype=torch.long)
    dag.boundary_pair_frag_idxs = torch.tensor([0, max(num_nodes - 1, 0)], dtype=torch.long)
    dag.boundary_pair_in_local = torch.tensor([0, 1 if num_nodes > 1 else 0], dtype=torch.long)
    dag.boundary_pair_out_local = torch.tensor([1 if num_nodes > 1 else 0, 0], dtype=torch.long)

    idx_to_formula = {i: f"C{i}H{i + 2}" for i in range(num_formulae)}
    idx_to_formula[0] = ""  # root node has empty formula

    num_h_delta_slots = 1 + 2 * max_h_transfer
    idx_by_h_delta = [
        set(np.random.choice(num_formulae, size=2, replace=False).tolist())
        for _ in range(num_h_delta_slots)
    ]

    edges_min_depth = (
        np.random.randint(0, 3, size=num_edges).astype(np.int32)
        if num_edges > 0
        else np.array([], dtype=np.int32)
    )
    nodes_min_depth = np.random.randint(0, 3, size=num_nodes).astype(np.int32)

    return {
        "dag": dag,
        "formula_peak_mzs": torch.randn(num_formulae, num_isotopes, dtype=torch.float32),
        "formula_peak_probs": torch.rand(num_formulae, num_isotopes, dtype=torch.float32),
        "idx_to_formula": idx_to_formula,
        "idx_by_h_delta": idx_by_h_delta,
        "edges_min_depth": edges_min_depth,
        "nodes_min_depth": nodes_min_depth,
        "dag_num_edges_by_depth": {0: num_edges // 2, 1: num_edges - num_edges // 2},
        "dag_num_nodes_by_depth": {0: 1, 1: num_nodes - 1},
        "max_depth": 4,
        "reached_depth": 3,
        "max_h_transfer": max_h_transfer,
        "dag_num_edges": num_edges,
        "dag_num_nodes": num_nodes,
        "dag_num_nodes_nb": num_nodes - 1,
        "dag_sparsity": 0.05,
        "formula_redundancy": 1.2,
        "node_feature_size": node_feat_dim,
        "edge_feature_size": edge_feat_dim,
        "is_directed": True,
        "force_stopped": False,
        "h_prior": True,
    }


def _assert_dag_d_equal(orig: dict, loaded: dict) -> None:
    # PyG tensors
    assert torch.equal(orig["dag"].x, loaded["dag"].x)
    assert torch.equal(orig["dag"].edge_index, loaded["dag"].edge_index)
    assert torch.equal(orig["dag"].edge_attr, loaded["dag"].edge_attr)
    assert torch.equal(orig["dag"].node_feat_idxs, loaded["dag"].node_feat_idxs)
    assert torch.equal(orig["dag"].edge_feat_idxs, loaded["dag"].edge_feat_idxs)
    assert torch.equal(orig["dag"].boundary_pair_frag_idxs, loaded["dag"].boundary_pair_frag_idxs)
    assert torch.equal(orig["dag"].boundary_pair_in_local, loaded["dag"].boundary_pair_in_local)
    assert torch.equal(orig["dag"].boundary_pair_out_local, loaded["dag"].boundary_pair_out_local)

    # Float tensors
    assert torch.allclose(orig["formula_peak_mzs"], loaded["formula_peak_mzs"])
    assert torch.allclose(orig["formula_peak_probs"], loaded["formula_peak_probs"])

    # Numpy arrays
    np.testing.assert_array_equal(orig["edges_min_depth"], loaded["edges_min_depth"])
    np.testing.assert_array_equal(orig["nodes_min_depth"], loaded["nodes_min_depth"])

    # Dicts
    assert orig["idx_to_formula"] == loaded["idx_to_formula"]
    assert [set(s) for s in orig["idx_by_h_delta"]] == loaded["idx_by_h_delta"]

    # Scalars
    for key in ["max_depth", "reached_depth", "max_h_transfer", "dag_num_edges", "dag_num_nodes"]:
        assert orig[key] == loaded[key], f"mismatch for {key}"
    assert orig["h_prior"] == loaded["h_prior"]
    assert orig["is_directed"] == loaded["is_directed"]
    assert orig["force_stopped"] == loaded["force_stopped"]
    assert abs(orig["dag_sparsity"] - loaded["dag_sparsity"]) < 1e-6


class TestDagHdf5RoundTrip:
    def test_normal_dag(self, tmp_path):
        """Round-trip a typical DAG with nodes and edges."""
        dag_d = _make_dag_d(num_nodes=10, num_edges=15, num_formulae=8)
        h5_fp = str(tmp_path / "dags.h5")
        mol_key = "00000001"

        with h5py.File(h5_fp, "w") as h5f:
            dump_dag_hdf5(h5f.create_group(mol_key), dag_d)

        with h5py.File(h5_fp, "r") as h5f:
            loaded = load_dag_hdf5_from_group(h5f[mol_key])

        _assert_dag_d_equal(dag_d, loaded)

    def test_single_node_no_edges(self, tmp_path):
        """Round-trip a DAG with a single node and zero edges (edge case)."""
        dag_d = _make_dag_d(num_nodes=1, num_edges=0, num_formulae=2)
        h5_fp = str(tmp_path / "dags.h5")
        mol_key = "0"

        with h5py.File(h5_fp, "w") as h5f:
            dump_dag_hdf5(h5f.create_group(mol_key), dag_d)

        with h5py.File(h5_fp, "r") as h5f:
            loaded = load_dag_hdf5_from_group(h5f[mol_key])

        assert loaded["dag"].num_edges == 0
        assert loaded["dag"].num_nodes == 1
        _assert_dag_d_equal(dag_d, loaded)

    def test_multiple_molecules(self, tmp_path):
        """Multiple molecules stored in one HDF5 file are independent."""
        dags = {1: _make_dag_d(5, 6, 4), 2: _make_dag_d(12, 20, 10)}
        h5_fp = str(tmp_path / "dags.h5")

        with h5py.File(h5_fp, "w") as h5f:
            for mol_id, dag_d in dags.items():
                dump_dag_hdf5(h5f.create_group(f"{int(mol_id)}"), dag_d)

        with h5py.File(h5_fp, "r") as h5f:
            for mol_id, orig in dags.items():
                loaded = load_dag_hdf5_from_group(h5f[f"{int(mol_id)}"])
                _assert_dag_d_equal(orig, loaded)

    def test_packed_round_trip_with_sparse_ids(self, tmp_path):
        """Packed storage provides direct lookup while allowing gaps in mol_ids."""
        dags = {0: _make_dag_d(5, 6, 4), 5: _make_dag_d(12, 20, 10)}
        h5_fp = str(tmp_path / "packed_dags.h5")

        with h5py.File(h5_fp, "w") as h5f:
            initialize_packed_dag_hdf5(h5f)
            for mol_id, dag_d in reversed(dags.items()):
                dump_dag_hdf5_packed(h5f, mol_id, dag_d)

        with h5py.File(h5_fp, "r") as h5f:
            assert h5f[PACKED_DAG_INDEX_DATASET].shape == (6,)
            assert int(h5f[PACKED_DAG_INDEX_DATASET][1]["length"]) == 0
            for mol_id, orig in dags.items():
                _assert_dag_d_equal(orig, load_dag_hdf5(h5f, mol_id))

    def test_packed_missing_and_duplicate_ids(self, tmp_path):
        """Zero-length index records mark missing IDs and duplicate writes fail."""
        dag_d = _make_dag_d(5, 6, 4)
        h5_fp = str(tmp_path / "packed_dags.h5")

        with h5py.File(h5_fp, "w") as h5f:
            initialize_packed_dag_hdf5(h5f)
            dump_dag_hdf5_packed(h5f, 3, dag_d)

            with pytest.raises(KeyError, match="mol_id 2"):
                load_dag_hdf5(h5f, 2)
            with pytest.raises(ValueError, match="Duplicate mol_id"):
                dump_dag_hdf5_packed(h5f, 3, dag_d)

    def test_legacy_file_lookup_by_mol_id(self, tmp_path):
        """The dataset-facing API continues to read existing group-based files."""
        dag_d = _make_dag_d(5, 6, 4)
        h5_fp = str(tmp_path / "legacy_dags.h5")

        with h5py.File(h5_fp, "w") as h5f:
            dump_dag_hdf5(h5f.create_group("7"), dag_d)

        with h5py.File(h5_fp, "r") as h5f:
            _assert_dag_d_equal(dag_d, load_dag_hdf5(h5f, 7))

    def test_frag_generation_consumes_results_unordered(self, tmp_path):
        """HDF5 keys make fragment-generation result order irrelevant."""
        mol_df = pd.DataFrame({"mol": ["first", "second"], "mol_id": [1, 2]})
        results_by_id = {
            1: _make_dag_d(4, 3, 3),
            2: _make_dag_d(6, 7, 5),
        }
        results = iter([(2, _packed(results_by_id[2])), (1, _packed(results_by_id[1]))])
        h5_fp = str(tmp_path / "dags.h5")

        with (
            patch("fragnnet.utils.frag_utils.par_apply", return_value=results) as mock_apply,
            h5py.File(h5_fp, "w") as h5f,
        ):
            meta_info = run_frag_gen_hdf5(
                mol_df=mol_df,
                h5_file=h5f,
                max_depth=3,
                max_h_transfer=4,
                max_time=150,
                isotopes=True,
                nb_isomorphic=False,
                wl_max_iterations=3,
                multi_cut_bfs=False,
                max_cut_size=2,
                smarts_prepass=False,
                min_frag_atoms=0,
                disable_tqdm=True,
            )

            assert set(h5f.keys()) == {
                PACKED_DAG_BYTES_DATASET,
                PACKED_DAG_INDEX_DATASET,
            }
            assert h5f.attrs[PACKED_DAG_FORMAT_ATTR] == PACKED_DAG_FORMAT
            _assert_dag_d_equal(results_by_id[1], load_dag_hdf5(h5f, 1))
            _assert_dag_d_equal(results_by_id[2], load_dag_hdf5(h5f, 2))

        assert set(meta_info) == {"1", "2"}
        assert mock_apply.call_args.kwargs == {"return_as_generator": True, "unordered": False}
        assert mock_apply.call_args.args[1] is timed_get_dags_packed

    def test_frag_generation_parts_resume_and_merge(self, tmp_path):
        """A killed part run resumes from its complete parts; the merge equals one file."""
        mol_df = pd.DataFrame({"mol": ["m"] * 5, "mol_id": [3, 5, 8, 11, 13]})
        by_id = {3: _make_dag_d(4, 3, 3), 5: _make_dag_d(6, 7, 5), 8: {},
                 11: _make_dag_d(5, 4, 4), 13: _make_dag_d(3, 2, 2)}  # fmt: skip
        kwargs = dict(
            part_size=2, max_depth=3, max_h_transfer=4, max_time=150, isotopes=False,
            nb_isomorphic=False, wl_max_iterations=-1, multi_cut_bfs=False, max_cut_size=2,
            smarts_prepass=False, min_frag_atoms=0,
        )  # fmt: skip
        calls = []

        def fake_par_apply(rows, func, need_unpack, return_as_generator, unordered, die_after=None):
            assert func is timed_get_dags_packed
            ids = [row[1] for row in rows]
            calls.append((ids, unordered))
            for n, m in enumerate(reversed(ids)):  # any order
                if n == die_after:
                    raise RuntimeError("killed")
                yield m, _packed(by_id[m])

        parts_dp = str(tmp_path / "parts")
        with patch("fragnnet.utils.frag_utils.par_apply",
                   side_effect=lambda *a, **k: fake_par_apply(*a, **k, die_after=3)):  # fmt: skip
            with pytest.raises(RuntimeError, match="killed"):
                run_frag_gen_hdf5_parts(mol_df, parts_dp, **kwargs)
        # Part 0 (13, 11) is complete; part 1 (8 failed, 5) was cut off before its marker.
        assert sorted(os.listdir(parts_dp)) == ["part_00000.h5", "part_00000.json",
                                                "part_00001.h5.tmp"]  # fmt: skip

        with patch("fragnnet.utils.frag_utils.par_apply", side_effect=fake_par_apply):
            meta_info, part_fps = run_frag_gen_hdf5_parts(mol_df, parts_dp, **kwargs)
        assert calls[1] == ([3, 5, 8], True)
        assert set(meta_info) == {"3", "5", "11", "13"}
        assert len(part_fps) == 3 and all(os.path.exists(fp) for fp in part_fps)

        # A third run has nothing left to do, failed molecule 8 included.
        with patch("fragnnet.utils.frag_utils.par_apply", side_effect=fake_par_apply):
            assert run_frag_gen_hdf5_parts(mol_df, parts_dp, **kwargs)[0] == meta_info
        assert len(calls) == 2

        with h5py.File(tmp_path / "dags.h5", "w") as h5f:
            assert merge_packed_dag_hdf5(h5f, part_fps) == 4
        with h5py.File(tmp_path / "dags.h5", "r") as h5f:
            for m in (3, 5, 11, 13):
                _assert_dag_d_equal(by_id[m], load_dag_hdf5(h5f, m))
            with pytest.raises(KeyError):
                load_dag_hdf5(h5f, 8)

    def test_append_packed_bytes_matches_dump(self, tmp_path):
        """Packing in a worker and appending bytes writes the file dump_dag_hdf5_packed writes."""
        dags = {3: _make_dag_d(4, 3, 3), 9: _make_dag_d(6, 7, 5)}
        for name, write in (
            ("dump", lambda h5, m, d: dump_dag_hdf5_packed(h5, m, d)),
            ("append", lambda h5, m, d: append_packed_dag_bytes(h5, m, pack_dag_payload(d))),
        ):
            with h5py.File(tmp_path / f"{name}.h5", "w") as h5f:
                initialize_packed_dag_hdf5(h5f)
                for m, d in dags.items():
                    write(h5f, m, d)
        with h5py.File(tmp_path / "dump.h5") as a, h5py.File(tmp_path / "append.h5") as b:
            for key in (PACKED_DAG_BYTES_DATASET, PACKED_DAG_INDEX_DATASET):
                assert a[key][:].tobytes() == b[key][:].tobytes()
            for m, d in dags.items():
                _assert_dag_d_equal(d, load_dag_hdf5(b, m))
        with h5py.File(tmp_path / "append.h5", "a") as h5f, pytest.raises(ValueError):
            append_packed_dag_bytes(h5f, 3, pack_dag_payload(dags[3]))

    def test_frag_generation_parts_real_workers(self, tmp_path):
        """Real molecules through loky workers: the packed results round-trip."""
        from rdkit import Chem

        from fragnnet.utils.frag_utils import timed_get_dags

        smiles = ["CCO", "CC(=O)O", "c1ccccc1O", "CC(N)C(=O)O", "[O-]C(=O)C"]
        mol_df = pd.DataFrame({"mol": [Chem.MolFromSmiles(x) for x in smiles],
                               "mol_id": [2, 4, 6, 8, 10]})  # fmt: skip
        meta_info, part_fps = run_frag_gen_hdf5_parts(
            mol_df, str(tmp_path / "parts"), part_size=2, max_depth=2, max_h_transfer=2,
            max_time=60, isotopes=False, nb_isomorphic=False, wl_max_iterations=-1,
            multi_cut_bfs=False, max_cut_size=2, smarts_prepass=False, min_frag_atoms=0,
        )  # fmt: skip
        assert set(meta_info) == {"2", "4", "6", "8"}  # the anion fails
        with h5py.File(tmp_path / "dags.h5", "w") as h5f:
            assert merge_packed_dag_hdf5(h5f, part_fps) == 4
        with h5py.File(tmp_path / "dags.h5", "r") as h5f:
            for m, row in mol_df.set_index("mol_id").iloc[:4].iterrows():
                _, ref = timed_get_dags(row["mol"], m, 2, True, 2, 60, False, False, -1,
                                        False, 2, False, 0)  # fmt: skip
                got = load_dag_hdf5(h5f, m)
                assert meta_info[str(m)] == [ref["dag"].num_nodes, ref["dag"].num_edges]
                assert torch.equal(got["dag"].x, ref["dag"].x)
                assert torch.equal(got["dag"].edge_index, ref["dag"].edge_index)

    def test_merge_packed_rejects_duplicate_ids(self, tmp_path):
        fps = []
        for i in range(2):
            fps.append(str(tmp_path / f"part_{i}.h5"))
            with h5py.File(fps[-1], "w") as h5f:
                initialize_packed_dag_hdf5(h5f)
                dump_dag_hdf5_packed(h5f, 7, _make_dag_d(4, 3, 3))
        with h5py.File(tmp_path / "dags.h5", "w") as h5f:
            with pytest.raises(ValueError, match="more than one part"):
                merge_packed_dag_hdf5(h5f, fps)

    def test_dtype_preservation(self, tmp_path):
        """node features (int64) and formula tensors (float32) dtypes survive round-trip."""
        dag_d = _make_dag_d(num_nodes=6, num_edges=8, num_formulae=5)
        h5_fp = str(tmp_path / "dags.h5")

        with h5py.File(h5_fp, "w") as h5f:
            dump_dag_hdf5(h5f.create_group("0"), dag_d)

        with h5py.File(h5_fp, "r") as h5f:
            loaded = load_dag_hdf5_from_group(h5f["0"])

        assert loaded["dag"].x.dtype == torch.int64
        assert loaded["dag"].edge_index.dtype == torch.int64
        assert loaded["dag"].edge_attr.dtype == torch.int64
        assert loaded["formula_peak_mzs"].dtype == torch.float32
        assert loaded["formula_peak_probs"].dtype == torch.float32

    def test_idx_to_formula_root_empty(self, tmp_path):
        """idx_to_formula[0] must survive as empty string (root node convention)."""
        dag_d = _make_dag_d(num_nodes=5, num_edges=4, num_formulae=5)
        dag_d["idx_to_formula"][0] = ""
        h5_fp = str(tmp_path / "dags.h5")

        with h5py.File(h5_fp, "w") as h5f:
            dump_dag_hdf5(h5f.create_group("0"), dag_d)

        with h5py.File(h5_fp, "r") as h5f:
            loaded = load_dag_hdf5_from_group(h5f["0"])

        assert loaded["idx_to_formula"][0] == ""

    def test_idx_by_h_delta_length(self, tmp_path):
        """idx_by_h_delta list length equals 1 + 2 * max_h_transfer after round-trip."""
        dag_d = _make_dag_d(num_nodes=8, num_edges=10, num_formulae=6)
        max_h_transfer = dag_d["max_h_transfer"]
        h5_fp = str(tmp_path / "dags.h5")

        with h5py.File(h5_fp, "w") as h5f:
            dump_dag_hdf5(h5f.create_group("0"), dag_d)

        with h5py.File(h5_fp, "r") as h5f:
            loaded = load_dag_hdf5_from_group(h5f["0"])

        assert len(loaded["idx_by_h_delta"]) == 1 + 2 * max_h_transfer
        for slot in loaded["idx_by_h_delta"]:
            assert isinstance(slot, set)

    def test_scalar_types(self, tmp_path):
        """Scalar attributes come back with correct Python types."""
        dag_d = _make_dag_d(num_nodes=4, num_edges=3, num_formulae=3)
        h5_fp = str(tmp_path / "dags.h5")

        with h5py.File(h5_fp, "w") as h5f:
            dump_dag_hdf5(h5f.create_group("0"), dag_d)

        with h5py.File(h5_fp, "r") as h5f:
            loaded = load_dag_hdf5_from_group(h5f["0"])

        for key in [
            "max_depth",
            "reached_depth",
            "max_h_transfer",
            "dag_num_edges",
            "dag_num_nodes",
            "dag_num_nodes_nb",
            "node_feature_size",
            "edge_feature_size",
        ]:
            assert isinstance(loaded[key], int), f"{key} should be int, got {type(loaded[key])}"
        for key in ["dag_sparsity", "formula_redundancy"]:
            assert isinstance(loaded[key], float), f"{key} should be float, got {type(loaded[key])}"
        for key in ["h_prior", "is_directed", "force_stopped"]:
            assert isinstance(loaded[key], bool), f"{key} should be bool, got {type(loaded[key])}"

    def test_depth_dicts_restored(self, tmp_path):
        """dag_num_edges_by_depth and dag_num_nodes_by_depth are restored with int keys."""
        dag_d = _make_dag_d(num_nodes=7, num_edges=9, num_formulae=5)
        dag_d["dag_num_edges_by_depth"] = {0: 4, 1: 3, 2: 2}
        dag_d["dag_num_nodes_by_depth"] = {0: 1, 1: 3, 2: 3}
        h5_fp = str(tmp_path / "dags.h5")

        with h5py.File(h5_fp, "w") as h5f:
            dump_dag_hdf5(h5f.create_group("0"), dag_d)

        with h5py.File(h5_fp, "r") as h5f:
            loaded = load_dag_hdf5_from_group(h5f["0"])

        assert loaded["dag_num_edges_by_depth"] == {0: 4, 1: 3, 2: 2}
        assert loaded["dag_num_nodes_by_depth"] == {0: 1, 1: 3, 2: 3}
        # Keys must be int, not str
        for key in loaded["dag_num_edges_by_depth"]:
            assert isinstance(key, int), f"depth key should be int, got {type(key)}"

    def test_formula_peak_shapes(self, tmp_path):
        """formula_peak_mzs and formula_peak_probs have correct shape after round-trip."""
        num_formulae, num_isotopes = 12, 5
        dag_d = _make_dag_d(num_nodes=8, num_edges=10, num_formulae=num_formulae)
        h5_fp = str(tmp_path / "dags.h5")

        with h5py.File(h5_fp, "w") as h5f:
            dump_dag_hdf5(h5f.create_group("0"), dag_d)

        with h5py.File(h5_fp, "r") as h5f:
            loaded = load_dag_hdf5_from_group(h5f["0"])

        assert loaded["formula_peak_mzs"].shape == (num_formulae, num_isotopes)
        assert loaded["formula_peak_probs"].shape == (num_formulae, num_isotopes)

    def test_nodes_min_depth_values(self, tmp_path):
        """nodes_min_depth array values survive exactly."""
        dag_d = _make_dag_d(num_nodes=5, num_edges=4, num_formulae=4)
        dag_d["nodes_min_depth"] = np.array([0, 1, 1, 2, 2], dtype=np.int32)
        h5_fp = str(tmp_path / "dags.h5")

        with h5py.File(h5_fp, "w") as h5f:
            dump_dag_hdf5(h5f.create_group("0"), dag_d)

        with h5py.File(h5_fp, "r") as h5f:
            loaded = load_dag_hdf5_from_group(h5f["0"])

        np.testing.assert_array_equal(loaded["nodes_min_depth"], [0, 1, 1, 2, 2])

    @pytest.mark.parametrize("force_stopped,h_prior", [(True, True), (False, False), (True, False)])
    def test_bool_flags(self, tmp_path, force_stopped, h_prior):
        """force_stopped and h_prior booleans round-trip correctly for all combinations."""
        dag_d = _make_dag_d(num_nodes=4, num_edges=3, num_formulae=3)
        dag_d["force_stopped"] = force_stopped
        dag_d["h_prior"] = h_prior
        h5_fp = str(tmp_path / "dags.h5")

        with h5py.File(h5_fp, "w") as h5f:
            dump_dag_hdf5(h5f.create_group("0"), dag_d)

        with h5py.File(h5_fp, "r") as h5f:
            loaded = load_dag_hdf5_from_group(h5f["0"])

        assert loaded["force_stopped"] == force_stopped
        assert loaded["h_prior"] == h_prior
