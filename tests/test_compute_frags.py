"""Tests for compute_frags BFS fragmentation (compute_frags.pyx).

Tests cover:
1. py_compute_node_to_edge_idx: correct neighbor-lookup structure.
2. mask_to_binary_hash: unique bytes per distinct mask.
3. compute_ccs — single-node molecule: trivial no-fragmentation case.
4. compute_ccs — max_depth=0: root only, no fragmentation attempted.
5. compute_ccs — linear chain (3 nodes, 2 undirected edges): known fragment
   count, DAG structure, and min-depth assignments.
6. compute_ccs — ethanol (real molecule via frag_utils): matches linear chain.
7. compute_ccs — cyclohexane ring: depth-1 fragments are all 6-node (ring opens,
   does not split into two components).
8. update_bonds — both atoms in CC: bond preserved.
9. update_bonds — only one atom in CC: bond removed.
10. compute_cc_h_floor — known H-floor for a 2-atom C–C fragment.
11. compute_cc_h_floor — single isolated atom.
"""

from __future__ import annotations

import numpy as np
import pytest

from fragnnet.frag.compute_frags import (
    compute_cc_h_floor,
    compute_ccs,
    mask_to_binary_hash,
    py_compute_node_to_edge_idx,
    update_bonds,
)
from fragnnet.utils.frag_utils import extract_mol_info, get_fraggen_input_arrays

# ---------------------------------------------------------------------------
# Constants matching compute_frags.pyx
# ---------------------------------------------------------------------------
_MAX_EDGES = 512  # 4 * MAX_NUM_NODES
_MASK_DTYPE = np.uint8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_edges(pairs: list[tuple[int, int]]) -> np.ndarray:
    """Build a padded edge array with _MAX_EDGES rows."""
    arr = np.zeros((_MAX_EDGES, 2), dtype=np.intc)
    for i, (u, v) in enumerate(pairs):
        arr[i] = [u, v]
    return arr


def _run_smiles(smiles: str, max_depth: int = 3, time_limit: int = 30, min_frag_atoms: int = 0) -> dict:
    """Run compute_ccs on a SMILES string and return result components."""
    mol_d = extract_mol_info(smiles)
    num_nodes, num_edges, node_mask, edges, edge_mask, n2e = get_fraggen_input_arrays(mol_d)
    nm, nd, de, meta = compute_ccs(
        num_nodes, num_edges, node_mask, edges, edge_mask, n2e,
        max_depth=max_depth, time_limit=time_limit, min_frag_atoms=min_frag_atoms,
    )
    return {
        "nodes_mask": nm,
        "nodes_depth": nd,
        "dag_edges": de,
        "meta": meta,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def linear_chain():
    """3-node undirected chain: 0 — 1 — 2."""
    num_nodes = 3
    num_edges = 2
    edges = _make_edges([(0, 1), (1, 2)])
    node_mask = np.ones(num_nodes, dtype=_MASK_DTYPE)
    edge_mask = np.ones(num_edges, dtype=_MASK_DTYPE)
    n2e = py_compute_node_to_edge_idx(num_nodes, num_edges, edges)
    return num_nodes, num_edges, node_mask, edges, edge_mask, n2e


@pytest.fixture
def chain_ccs(linear_chain):
    """compute_ccs result for the 3-node chain."""
    num_nodes, num_edges, node_mask, edges, edge_mask, n2e = linear_chain
    nm, nd, de, meta = compute_ccs(
        num_nodes, num_edges, node_mask, edges, edge_mask, n2e,
        max_depth=3, time_limit=30, min_frag_atoms=0,
    )
    return nm, nd, de, meta


# ---------------------------------------------------------------------------
# 1. py_compute_node_to_edge_idx
# ---------------------------------------------------------------------------


def test_node_to_edge_idx_interior_node(linear_chain):
    """Interior node 1 is incident to both edges (indices 0 and 1)."""
    _, _, _, _, _, n2e = linear_chain
    neighbors_1 = set(n2e[1].tolist()) - {-1}
    assert neighbors_1 == {0, 1}


def test_node_to_edge_idx_endpoint_nodes(linear_chain):
    """Endpoint nodes 0 and 2 each have exactly one incident edge."""
    _, _, _, _, _, n2e = linear_chain
    neighbors_0 = set(n2e[0].tolist()) - {-1}
    neighbors_2 = set(n2e[2].tolist()) - {-1}
    assert neighbors_0 == {0}
    assert neighbors_2 == {1}


def test_node_to_edge_idx_unused_slots_are_sentinel(linear_chain):
    """Unused neighbor slots must be -1."""
    _, _, _, _, _, n2e = linear_chain
    # Node 0 has 1 neighbor; slots [1:] should all be -1
    assert all(v == -1 for v in n2e[0, 1:].tolist())


# ---------------------------------------------------------------------------
# 2. mask_to_binary_hash
# ---------------------------------------------------------------------------


def test_mask_to_binary_hash_returns_bytes():
    mask = np.ones(4, dtype=_MASK_DTYPE)
    result = mask_to_binary_hash(mask)
    assert isinstance(result, bytes)


def test_mask_to_binary_hash_unique():
    """Different masks produce different hashes."""
    m1 = np.array([1, 0, 1], dtype=_MASK_DTYPE)
    m2 = np.array([1, 1, 0], dtype=_MASK_DTYPE)
    assert mask_to_binary_hash(m1) != mask_to_binary_hash(m2)


def test_mask_to_binary_hash_deterministic():
    """Same mask always produces the same hash."""
    m = np.array([1, 0, 0, 1], dtype=_MASK_DTYPE)
    assert mask_to_binary_hash(m) == mask_to_binary_hash(m.copy())


# ---------------------------------------------------------------------------
# 3. compute_ccs — single-node molecule
# ---------------------------------------------------------------------------


def test_single_node_one_fragment():
    """Single node with no edges yields exactly 1 fragment (the molecule itself)."""
    num_nodes = 1
    edges = _make_edges([])
    node_mask = np.ones(1, dtype=_MASK_DTYPE)
    edge_mask = np.zeros(0, dtype=_MASK_DTYPE)
    n2e = py_compute_node_to_edge_idx(num_nodes, 0, edges)
    nm, nd, de, meta = compute_ccs(
        num_nodes, 0, node_mask, edges, edge_mask, n2e,
        max_depth=3, time_limit=30,
    )
    assert nm.shape[0] == 1
    assert de.shape[0] == 0
    assert meta["reached_depth"] == 0
    assert not meta["force_stopped"]


# ---------------------------------------------------------------------------
# 4. compute_ccs — max_depth=0
# ---------------------------------------------------------------------------


def test_max_depth_zero_no_fragmentation():
    """max_depth=0 returns only the root fragment with no DAG edges."""
    result = _run_smiles("CCO", max_depth=0)
    assert result["nodes_mask"].shape[0] == 1
    assert result["dag_edges"].shape[0] == 0
    assert result["meta"]["reached_depth"] == 0


# ---------------------------------------------------------------------------
# 5. compute_ccs — linear chain known structure
# ---------------------------------------------------------------------------


def test_linear_chain_fragment_count(chain_ccs):
    """3-node chain with 2 bonds produces exactly 6 unique fragments."""
    nm, _, _, _ = chain_ccs
    assert nm.shape[0] == 6


def test_linear_chain_dag_edge_count(chain_ccs):
    """3-node chain DAG has exactly 8 directed edges."""
    _, _, de, _ = chain_ccs
    assert de.shape[0] == 8


def test_linear_chain_reached_depth(chain_ccs):
    """3-node chain reaches depth 2 (both bonds broken)."""
    _, _, _, meta = chain_ccs
    assert meta["reached_depth"] == 2


def test_linear_chain_root_is_full_molecule(chain_ccs):
    """First row of nodes_mask is the full molecule (all nodes active)."""
    nm, _, _, _ = chain_ccs
    assert nm[0].sum() == 3
    assert nm[0].tolist() == [1, 1, 1]


def test_linear_chain_singleton_fragments_exist(chain_ccs):
    """All three singleton node masks appear in the fragment set."""
    nm, _, _, _ = chain_ccs
    row_sums = nm.sum(axis=1).tolist()
    # 1 root (size 3) + 2 pairs (size 2) + 3 singletons (size 1)
    assert row_sums.count(1) == 3
    assert row_sums.count(2) == 2
    assert row_sums.count(3) == 1


def test_linear_chain_min_depths(chain_ccs):
    """Root has min_depth=0; all others ≥1; only node {1} appears first at depth 2."""
    _, _, _, meta = chain_ccs
    min_depths = meta["nodes_min_depth"].tolist()
    assert min_depths[0] == 0  # root
    assert max(min_depths) == 2  # node {1} first reachable at depth 2


def test_linear_chain_dag_endpoints_valid(chain_ccs):
    """All DAG edge endpoints reference valid fragment indices."""
    nm, _, de, _ = chain_ccs
    num_frags = nm.shape[0]
    assert de.min() >= 0
    assert de.max() < num_frags


def test_linear_chain_no_self_loops(chain_ccs):
    """DAG must not contain self-loop edges."""
    _, _, de, _ = chain_ccs
    assert not np.any(de[:, 0] == de[:, 1])


# ---------------------------------------------------------------------------
# 6. compute_ccs — ethanol via frag_utils (matches linear chain result)
# ---------------------------------------------------------------------------


def test_ethanol_fragment_count():
    """Ethanol (3 heavy atoms, 2 bonds) produces the same 6 fragments as the
    3-node linear chain fixture."""
    result = _run_smiles("CCO", max_depth=3)
    assert result["nodes_mask"].shape[0] == 6


def test_ethanol_dag_edge_count():
    result = _run_smiles("CCO", max_depth=3)
    assert result["dag_edges"].shape[0] == 8


def test_ethanol_not_force_stopped():
    result = _run_smiles("CCO", max_depth=3)
    assert not result["meta"]["force_stopped"]


# ---------------------------------------------------------------------------
# 7. compute_ccs — cyclohexane ring
# ---------------------------------------------------------------------------


def test_cyclohexane_depth1_all_six_nodes():
    """Cutting one bond in cyclohexane at depth 1 opens the ring but keeps all
    6 atoms in a single connected component (no split yet)."""
    result = _run_smiles("C1CCCCC1", max_depth=2)
    nd = result["nodes_depth"]
    nm = result["nodes_mask"]
    # fragments first seen at depth 1
    depth1_mask = nd[:, 1] == 1
    depth1_frags = nm[depth1_mask]
    node_counts = depth1_frags.sum(axis=1).tolist()
    # every depth-1 fragment should still have all 6 nodes
    assert all(c == 6 for c in node_counts), f"Expected all 6, got {node_counts}"


def test_cyclohexane_depth2_has_smaller_fragments():
    """At depth 2, two bonds are broken, producing smaller fragments."""
    result = _run_smiles("C1CCCCC1", max_depth=2)
    nd = result["nodes_depth"]
    nm = result["nodes_mask"]
    depth2_mask = nd[:, 2] == 1
    depth2_frags = nm[depth2_mask]
    node_counts = depth2_frags.sum(axis=1).tolist()
    assert any(c < 6 for c in node_counts)


# ---------------------------------------------------------------------------
# 8. update_bonds — both atoms in CC
# ---------------------------------------------------------------------------


def test_update_bonds_full_cc_preserves_bond():
    """When both endpoints are in the CC, the bond count and mask are preserved."""
    cc_atom_ids = np.array([0, 1], dtype=np.int32)
    sbond_arr = np.array([1, 1], dtype=np.int32)
    bond_mask_arr = np.array([1], dtype=np.uint8)
    bonds = np.array([[0, 1]], dtype=np.int32)
    atoms_to_bonds = {0: [0], 1: [0]}

    new_sbond, new_bond_mask = update_bonds(cc_atom_ids, sbond_arr, bond_mask_arr, bonds, atoms_to_bonds)

    assert new_sbond[0] == 1
    assert new_sbond[1] == 1
    assert new_bond_mask[0] == 1


# ---------------------------------------------------------------------------
# 9. update_bonds — only one atom in CC
# ---------------------------------------------------------------------------


def test_update_bonds_partial_cc_removes_bond():
    """When only one endpoint is in the CC, the cross bond is masked out."""
    cc_atom_ids = np.array([0], dtype=np.int32)
    sbond_arr = np.array([1, 1], dtype=np.int32)
    bond_mask_arr = np.array([1], dtype=np.uint8)
    bonds = np.array([[0, 1]], dtype=np.int32)
    atoms_to_bonds = {0: [0], 1: [0]}

    new_sbond, new_bond_mask = update_bonds(cc_atom_ids, sbond_arr, bond_mask_arr, bonds, atoms_to_bonds)

    assert new_sbond[0] == 0  # no intra-CC bonds for atom 0
    assert new_bond_mask[0] == 0  # bond removed


def test_update_bonds_output_shapes():
    """Output arrays have the same shape as inputs."""
    cc_atom_ids = np.array([0, 1], dtype=np.int32)
    sbond_arr = np.zeros(3, dtype=np.int32)
    bond_mask_arr = np.zeros(2, dtype=np.uint8)
    bonds = np.array([[0, 1], [1, 2]], dtype=np.int32)
    atoms_to_bonds = {0: [0], 1: [0, 1], 2: [1]}

    new_sbond, new_bond_mask = update_bonds(cc_atom_ids, sbond_arr, bond_mask_arr, bonds, atoms_to_bonds)

    assert new_sbond.shape == sbond_arr.shape
    assert new_bond_mask.shape == bond_mask_arr.shape


# ---------------------------------------------------------------------------
# 10. compute_cc_h_floor — 2-atom C–C fragment
# ---------------------------------------------------------------------------


def test_h_floor_two_atom_cc():
    """Two-carbon fragment sharing one bond: each C needs at least 2H (h_floor=4 total).

    diff_arr = max(valence - sbonds, 0) = max(4-1, 0) = 3 for each C.
    h_arr[0] = max(0, 3 - min(3, 2)) = 1
    h_arr[1] = max(0, 3 - min(3, 2)) = 1
    cc_floor = 1 + 1 = 2
    """
    cc_atom_ids = np.array([0, 1], dtype=np.int32)
    ve_arr = np.array([4, 4], dtype=np.int32)   # carbon valence = 4
    sbond_arr = np.array([1, 1], dtype=np.int32)
    bonds = np.array([[0, 1]], dtype=np.int32)
    atoms_to_bonds = {0: [0], 1: [0]}
    bond_mask_arr = np.array([1], dtype=np.uint8)

    h_floor = compute_cc_h_floor(cc_atom_ids, ve_arr, sbond_arr, 0, bonds, atoms_to_bonds, bond_mask_arr)
    assert h_floor == 2


# ---------------------------------------------------------------------------
# 11. compute_cc_h_floor — single isolated atom
# ---------------------------------------------------------------------------


def test_h_floor_single_isolated_carbon():
    """Single carbon with no bonds in its CC: h_floor = max(valence - 0, 0) = 4."""
    cc_atom_ids = np.array([0], dtype=np.int32)
    ve_arr = np.array([4], dtype=np.int32)
    sbond_arr = np.array([0], dtype=np.int32)
    bonds = np.empty((0, 2), dtype=np.int32)
    atoms_to_bonds: dict = {0: []}
    bond_mask_arr = np.empty(0, dtype=np.uint8)

    h_floor = compute_cc_h_floor(cc_atom_ids, ve_arr, sbond_arr, 0, bonds, atoms_to_bonds, bond_mask_arr)
    assert h_floor == 4
