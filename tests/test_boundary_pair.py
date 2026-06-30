"""Unit tests for compute_boundary_pair_idxs in frag_utils.py."""

import torch as th

from fragnnet.utils.frag_utils import compute_boundary_pair_idxs


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _prefix_sum(counts: list[int]) -> th.Tensor:
    result = [0]
    for c in counts:
        result.append(result[-1] + c)
    return th.tensor(result, dtype=th.long)


def _pairs_as_set(frag_idxs, in_global, out_global):
    """Return a set of (frag_idx, in_atom, out_atom) tuples for assertion."""
    return set(zip(frag_idxs.tolist(), in_global.tolist(), out_global.tolist()))


# ---------------------------------------------------------------------------
# Basic single-molecule tests
# ---------------------------------------------------------------------------


def test_linear_chain_single_cut():
    """Linear 3-atom chain 0-1-2, cut between atoms 1 and 2.

    Fragments:
      frag 0 (root): atoms {0,1,2} → no pairs (root)
      frag 1 (left): atoms {0,1}   → cut bond (1, 2): inside=1, outside=2
      frag 2 (right): atom {2}     → cut bond (2, 1): inside=2, outside=1
    """
    mask_size = 3
    frag_node_mask = th.tensor(
        [
            [True, True, True],   # root
            [True, True, False],  # left: atoms 0,1
            [False, False, True], # right: atom 2
        ],
        dtype=th.bool,
    )
    # Undirected edges both directions
    mol_edge_index = th.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=th.long)
    mol_num_nodes = _prefix_sum([3])
    frag_num_nodes = _prefix_sum([3])

    frag_idxs, in_g, out_g = compute_boundary_pair_idxs(
        frag_node_mask, mol_edge_index, mol_num_nodes, frag_num_nodes
    )

    pairs = _pairs_as_set(frag_idxs, in_g, out_g)

    # Root must have no pairs
    assert all(f != 0 for f, _, _ in pairs), "root fragment must have no boundary pairs"

    # Left fragment (idx 1): cut bond → inside atom 1, outside atom 2
    assert (1, 1, 2) in pairs, "left frag: inside=1, outside=2"

    # Right fragment (idx 2): cut bond → inside atom 2, outside atom 1
    assert (2, 2, 1) in pairs, "right frag: inside=2, outside=1"


def test_root_always_empty():
    """Root fragment (all atoms) produces zero pairs."""
    mask_size = 4
    frag_node_mask = th.tensor([[True, True, True, True]], dtype=th.bool)
    mol_edge_index = th.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=th.long)
    mol_num_nodes = _prefix_sum([4])
    frag_num_nodes = _prefix_sum([1])

    frag_idxs, in_g, out_g = compute_boundary_pair_idxs(
        frag_node_mask, mol_edge_index, mol_num_nodes, frag_num_nodes
    )

    assert frag_idxs.numel() == 0, "root has no cut bonds → no pairs"


def test_no_edges_gives_no_pairs():
    """Molecule with no edges produces no pairs for any fragment."""
    frag_node_mask = th.tensor([[True, True], [True, False]], dtype=th.bool)
    mol_edge_index = th.zeros(2, 0, dtype=th.long)
    mol_num_nodes = _prefix_sum([2])
    frag_num_nodes = _prefix_sum([2])

    frag_idxs, in_g, out_g = compute_boundary_pair_idxs(
        frag_node_mask, mol_edge_index, mol_num_nodes, frag_num_nodes
    )
    assert frag_idxs.numel() == 0


def test_single_atom_fragment():
    """Fragment containing one atom adjacent to one outside atom.

    mol: 0-1 (atoms 0 and 1)
    frag 1: atom {0} → cut bond (0,1): inside=0, outside=1
    """
    frag_node_mask = th.tensor(
        [[True, True], [True, False]],  # root, frag with atom 0
        dtype=th.bool,
    )
    mol_edge_index = th.tensor([[0, 1], [1, 0]], dtype=th.long)
    mol_num_nodes = _prefix_sum([2])
    frag_num_nodes = _prefix_sum([2])

    frag_idxs, in_g, out_g = compute_boundary_pair_idxs(
        frag_node_mask, mol_edge_index, mol_num_nodes, frag_num_nodes
    )

    pairs = _pairs_as_set(frag_idxs, in_g, out_g)
    assert (1, 0, 1) in pairs
    # Root must have no pairs
    assert all(f != 0 for f, _, _ in pairs)


def test_two_cuts_two_pairs():
    """Fragment cut at two bonds has two inside/outside pairs.

    Linear chain 0-1-2-3. Fragment: atoms {1,2} (middle two).
    Cut bonds: (1,0) and (2,3).
    """
    mask_size = 4
    frag_node_mask = th.tensor(
        [
            [True, True, True, True],    # root
            [False, True, True, False],  # middle frag: atoms 1,2
        ],
        dtype=th.bool,
    )
    mol_edge_index = th.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=th.long
    )
    mol_num_nodes = _prefix_sum([4])
    frag_num_nodes = _prefix_sum([2])

    frag_idxs, in_g, out_g = compute_boundary_pair_idxs(
        frag_node_mask, mol_edge_index, mol_num_nodes, frag_num_nodes
    )

    pairs = _pairs_as_set(frag_idxs, in_g, out_g)

    # Both cut bonds should appear for frag 1
    assert (1, 1, 0) in pairs, "cut bond 1-0: inside=1, outside=0"
    assert (1, 2, 3) in pairs, "cut bond 2-3: inside=2, outside=3"
    # Root has no pairs
    assert all(f != 0 for f, _, _ in pairs)


# ---------------------------------------------------------------------------
# Multi-molecule batch
# ---------------------------------------------------------------------------


def test_two_molecule_batch():
    """Batch of two molecules.

    mol0: 0-1 (2 atoms). Frags: root {0,1}, frag {0}.
      cut bond → frag {0}: inside=0 (global 0), outside=1 (global 1)
    mol1: 0-1-2 (3 atoms, global 2,3,4). Frags: root {0,1,2}, frag {2}.
      cut bond → frag {2}: inside=4 (global), outside=3 (global)
    """
    mask_size = 3
    frag_node_mask = th.tensor(
        [
            [True, True, False],   # mol0 root
            [True, False, False],  # mol0 frag: atom 0
            [True, True, True],    # mol1 root
            [False, False, True],  # mol1 frag: atom 2 (local)
        ],
        dtype=th.bool,
    )
    # mol0 edges (global 0↔1), mol1 edges (global 2↔3, 3↔4)
    mol_edge_index = th.tensor([[0, 1, 2, 3, 3, 4], [1, 0, 3, 2, 4, 3]], dtype=th.long)
    mol_num_nodes = _prefix_sum([2, 3])
    frag_num_nodes = _prefix_sum([2, 2])

    frag_idxs, in_g, out_g = compute_boundary_pair_idxs(
        frag_node_mask, mol_edge_index, mol_num_nodes, frag_num_nodes
    )

    pairs = _pairs_as_set(frag_idxs, in_g, out_g)

    # mol0 roots (frag 0) and mol1 root (frag 2) must have no pairs
    assert all(f not in (0, 2) for f, _, _ in pairs), "roots must have no pairs"

    # mol0 frag (global idx 1): atom 0 inside, atom 1 outside
    assert (1, 0, 1) in pairs

    # mol1 frag (global idx 3): atom 4 inside (local 2 + offset 2), atom 3 outside
    assert (3, 4, 3) in pairs


# ---------------------------------------------------------------------------
# Output properties
# ---------------------------------------------------------------------------


def test_output_dtypes():
    """All returned tensors must be long."""
    frag_node_mask = th.tensor([[True, True], [True, False]], dtype=th.bool)
    mol_edge_index = th.tensor([[0, 1], [1, 0]], dtype=th.long)
    mol_num_nodes = _prefix_sum([2])
    frag_num_nodes = _prefix_sum([2])

    frag_idxs, in_g, out_g = compute_boundary_pair_idxs(
        frag_node_mask, mol_edge_index, mol_num_nodes, frag_num_nodes
    )

    assert frag_idxs.dtype == th.long
    assert in_g.dtype == th.long
    assert out_g.dtype == th.long


def test_output_lengths_match():
    """All three returned tensors must have the same length."""
    frag_node_mask = th.tensor([[True, True, True], [True, True, False]], dtype=th.bool)
    mol_edge_index = th.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=th.long)
    mol_num_nodes = _prefix_sum([3])
    frag_num_nodes = _prefix_sum([2])

    frag_idxs, in_g, out_g = compute_boundary_pair_idxs(
        frag_node_mask, mol_edge_index, mol_num_nodes, frag_num_nodes
    )

    assert frag_idxs.shape == in_g.shape == out_g.shape
