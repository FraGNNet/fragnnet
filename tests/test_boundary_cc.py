"""Unit tests for compute_boundary_mask in fragnnet_model.py."""

import torch as th

from fragnnet.utils.frag_utils import compute_boundary_mask

# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _prefix_sum(counts: list[int]) -> th.Tensor:
    """Build a cumulative prefix-sum tensor from a list of per-molecule counts."""
    result = [0]
    for c in counts:
        result.append(result[-1] + c)
    return th.tensor(result, dtype=th.long)


# ---------------------------------------------------------------------------
# Basic single-molecule tests
# ---------------------------------------------------------------------------


def test_linear_chain_single_cut():
    """Linear 3-atom chain 0-1-2, cut between atoms 1 and 2.

    Two fragments:
      frag 0 (root): atoms {0, 1, 2}  → no boundary (root)
      frag 1 (left): atoms {0, 1}     → boundary atom 1 (adjacent to cut atom 2)
      frag 2 (right): atoms {2}       → boundary atom 2 (adjacent to cut atom 1)
    """
    mask_size = 3
    # frag_node_mask: (3, 3) bool
    frag_node_mask = th.tensor(
        [
            [True, True, True],  # root: all atoms
            [True, True, False],  # left fragment: atoms 0, 1
            [False, False, True],  # right fragment: atom 2
        ],
        dtype=th.bool,
    )
    # Undirected edges: 0-1 and 1-2 (both directions)
    mol_edge_index = th.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=th.long)
    mol_num_nodes = _prefix_sum([3])  # one molecule, 3 atoms
    frag_num_nodes = _prefix_sum([3])  # one molecule, 3 fragments

    bm = compute_boundary_mask(frag_node_mask, mol_edge_index, mol_num_nodes, frag_num_nodes)

    assert bm.shape == (3, mask_size)
    # Root: all-zero
    assert not bm[0].any(), "root fragment should have no boundary atoms"
    # Left fragment: atom 1 is boundary (neighbor 2 is outside)
    assert bm[1, 1].item(), "atom 1 should be boundary in left fragment"
    assert not bm[1, 0].item(), "atom 0 should not be boundary in left fragment"
    assert not bm[1, 2].item(), "atom 2 not in left fragment"
    # Right fragment: atom 2 is boundary (neighbor 1 is outside)
    assert bm[2, 2].item(), "atom 2 should be boundary in right fragment"
    assert not bm[2, 0].item()
    assert not bm[2, 1].item()


def test_single_atom_fragment():
    """Single-atom fragment: that atom is a boundary if it has any neighbor outside."""
    mask_size = 2
    frag_node_mask = th.tensor(
        [
            [True, True],  # root
            [True, False],  # frag containing atom 0 only
        ],
        dtype=th.bool,
    )
    mol_edge_index = th.tensor([[0, 1], [1, 0]], dtype=th.long)
    mol_num_nodes = _prefix_sum([2])
    frag_num_nodes = _prefix_sum([2])

    bm = compute_boundary_mask(frag_node_mask, mol_edge_index, mol_num_nodes, frag_num_nodes)

    assert not bm[0].any(), "root has no boundary"
    assert bm[1, 0].item(), "atom 0 is boundary (neighbor 1 is outside)"
    assert not bm[1, 1].item()


def test_no_edges():
    """Molecule with no edges: no boundary atoms for any fragment."""
    mask_size = 2
    frag_node_mask = th.tensor([[True, True], [True, False]], dtype=th.bool)
    mol_edge_index = th.zeros(2, 0, dtype=th.long)
    mol_num_nodes = _prefix_sum([2])
    frag_num_nodes = _prefix_sum([2])

    bm = compute_boundary_mask(frag_node_mask, mol_edge_index, mol_num_nodes, frag_num_nodes)
    assert not bm.any(), "no edges → no boundary atoms"


def test_root_always_zero():
    """Root fragment (all atoms True) must always produce an all-zero boundary row."""
    mask_size = 4
    # Root = all atoms
    frag_node_mask = th.tensor(
        [[True, True, True, True]],
        dtype=th.bool,
    )
    # Fully connected (star from 0)
    mol_edge_index = th.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]], dtype=th.long)
    mol_num_nodes = _prefix_sum([4])
    frag_num_nodes = _prefix_sum([1])

    bm = compute_boundary_mask(frag_node_mask, mol_edge_index, mol_num_nodes, frag_num_nodes)
    assert not bm.any(), "root (all atoms in frag) must have zero boundary"


# ---------------------------------------------------------------------------
# Multi-molecule batch tests
# ---------------------------------------------------------------------------


def test_two_molecule_batch():
    """Batch of two molecules: mol0 has 2 atoms, mol1 has 3 atoms.

    mol0: 0-1 (single bond), 2 frags (root + left {atom 0})
    mol1: 0-1-2 chain, 2 frags (root + right {atom 2})
    Global atom indices: mol0 → 0,1; mol1 → 2,3,4.
    """
    # mol0 local: 0,1  → global 0,1
    # mol1 local: 0,1,2 → global 2,3,4
    mask_size = 3

    # frag 0 (mol0 root): atoms 0,1 local → mask [T,T,F]
    # frag 1 (mol0 frag): atom 0 local   → mask [T,F,F]
    # frag 2 (mol1 root): atoms 0,1,2 local → mask [T,T,T]
    # frag 3 (mol1 frag): atom 2 local   → mask [F,F,T]
    frag_node_mask = th.tensor(
        [
            [True, True, False],  # mol0 root
            [True, False, False],  # mol0 frag: atom 0
            [True, True, True],  # mol1 root
            [False, False, True],  # mol1 frag: atom 2
        ],
        dtype=th.bool,
    )
    # mol0 edges: 0↔1 (global 0↔1)
    # mol1 edges: 0↔1, 1↔2 (global 2↔3, 3↔4)
    mol_edge_index = th.tensor([[0, 1, 2, 3, 3, 4], [1, 0, 3, 2, 4, 3]], dtype=th.long)
    mol_num_nodes = _prefix_sum([2, 3])  # mol0=2 atoms, mol1=3 atoms
    frag_num_nodes = _prefix_sum([2, 2])  # mol0=2 frags, mol1=2 frags

    bm = compute_boundary_mask(frag_node_mask, mol_edge_index, mol_num_nodes, frag_num_nodes)

    assert bm.shape == (4, mask_size)
    # mol0 root → all zero
    assert not bm[0].any()
    # mol0 frag (atom 0): atom 0 is boundary (neighbor 1 outside)
    assert bm[1, 0].item()
    assert not bm[1, 1].item()
    # mol1 root → all zero
    assert not bm[2].any()
    # mol1 frag (atom 2 local): atom 2 is boundary (neighbor 1 outside)
    assert bm[3, 2].item()
    assert not bm[3, 0].item()
    assert not bm[3, 1].item()


# ---------------------------------------------------------------------------
# Edge case: empty fragment list for one molecule
# ---------------------------------------------------------------------------


def test_molecule_with_no_fragments_skipped():
    """mol0 has 0 frags, mol1 has 2 frags — mol0 slice is skipped without error."""
    mask_size = 2
    frag_node_mask = th.tensor(
        [
            [True, True],  # mol1 root
            [True, False],  # mol1 frag: atom 0
        ],
        dtype=th.bool,
    )
    mol_edge_index = th.tensor([[0, 1], [1, 0]], dtype=th.long)
    # mol0=1 atom (global 0, but no frags), mol1=2 atoms (global 1,2... wait, mol0=1 atom)
    # Actually: mol0 has 1 atom, mol1 has 2 atoms (global 1,2). mol1 frags use local [0,1].
    mol_num_nodes = _prefix_sum([1, 2])
    frag_num_nodes = _prefix_sum([0, 2])  # mol0: 0 frags, mol1: 2 frags

    # mol1 edges (global): 1↔2
    mol_edge_index = th.tensor([[1, 2], [2, 1]], dtype=th.long)

    bm = compute_boundary_mask(frag_node_mask, mol_edge_index, mol_num_nodes, frag_num_nodes)
    assert bm.shape == (2, mask_size)
    # mol1 root: no boundary
    assert not bm[0].any()
    # mol1 frag atom 0: boundary (neighbor local-1 is outside)
    assert bm[1, 0].item()
    assert not bm[1, 1].item()


# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------


def test_output_dtype_and_shape():
    """Return tensor must be bool with shape (F, mask_size)."""
    frag_node_mask = th.tensor([[True, False, True]], dtype=th.bool)
    mol_edge_index = th.zeros(2, 0, dtype=th.long)
    mol_num_nodes = _prefix_sum([3])
    frag_num_nodes = _prefix_sum([1])

    bm = compute_boundary_mask(frag_node_mask, mol_edge_index, mol_num_nodes, frag_num_nodes)
    assert bm.dtype == th.bool
    assert bm.shape == (1, 3)
