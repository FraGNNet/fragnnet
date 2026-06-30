"""Tests for HDF5 DAG serialization round-trip."""

import h5py
import numpy as np
import pytest
import torch
import torch_geometric as pyg

from fragnnet.utils.frag_utils import dump_dag_hdf5, load_dag_hdf5


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

    idx_to_formula = {i: f"C{i}H{i+2}" for i in range(num_formulae)}
    idx_to_formula[0] = ""  # root node has empty formula

    num_h_delta_slots = 1 + 2 * max_h_transfer
    idx_by_h_delta = [
        set(np.random.choice(num_formulae, size=2, replace=False).tolist())
        for _ in range(num_h_delta_slots)
    ]

    edges_min_depth = (
        np.random.randint(0, 3, size=num_edges).astype(np.int32) if num_edges > 0
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
            loaded = load_dag_hdf5(h5f[mol_key])

        _assert_dag_d_equal(dag_d, loaded)

    def test_single_node_no_edges(self, tmp_path):
        """Round-trip a DAG with a single node and zero edges (edge case)."""
        dag_d = _make_dag_d(num_nodes=1, num_edges=0, num_formulae=2)
        h5_fp = str(tmp_path / "dags.h5")
        mol_key = "0"

        with h5py.File(h5_fp, "w") as h5f:
            dump_dag_hdf5(h5f.create_group(mol_key), dag_d)

        with h5py.File(h5_fp, "r") as h5f:
            loaded = load_dag_hdf5(h5f[mol_key])

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
                loaded = load_dag_hdf5(h5f[f"{int(mol_id)}"])
                _assert_dag_d_equal(orig, loaded)

    def test_dtype_preservation(self, tmp_path):
        """node features (int64) and formula tensors (float32) dtypes survive round-trip."""
        dag_d = _make_dag_d(num_nodes=6, num_edges=8, num_formulae=5)
        h5_fp = str(tmp_path / "dags.h5")

        with h5py.File(h5_fp, "w") as h5f:
            dump_dag_hdf5(h5f.create_group("0"), dag_d)

        with h5py.File(h5_fp, "r") as h5f:
            loaded = load_dag_hdf5(h5f["0"])

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
            loaded = load_dag_hdf5(h5f["0"])

        assert loaded["idx_to_formula"][0] == ""

    def test_idx_by_h_delta_length(self, tmp_path):
        """idx_by_h_delta list length equals 1 + 2 * max_h_transfer after round-trip."""
        dag_d = _make_dag_d(num_nodes=8, num_edges=10, num_formulae=6)
        max_h_transfer = dag_d["max_h_transfer"]
        h5_fp = str(tmp_path / "dags.h5")

        with h5py.File(h5_fp, "w") as h5f:
            dump_dag_hdf5(h5f.create_group("0"), dag_d)

        with h5py.File(h5_fp, "r") as h5f:
            loaded = load_dag_hdf5(h5f["0"])

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
            loaded = load_dag_hdf5(h5f["0"])

        for key in ["max_depth", "reached_depth", "max_h_transfer", "dag_num_edges",
                    "dag_num_nodes", "dag_num_nodes_nb", "node_feature_size", "edge_feature_size"]:
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
            loaded = load_dag_hdf5(h5f["0"])

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
            loaded = load_dag_hdf5(h5f["0"])

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
            loaded = load_dag_hdf5(h5f["0"])

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
            loaded = load_dag_hdf5(h5f["0"])

        assert loaded["force_stopped"] == force_stopped
        assert loaded["h_prior"] == h_prior
