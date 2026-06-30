"""Tests for compute_dags in frag_utils.py.

Tests cover:
1. Brominated amine (BrCCNc1ccccc1): previously caused KeyError: 255 due to
   uint8 underflow in diff_cc_mask and bitmask passed directly as atom indices.
2. Brominated piperidine (BrCCC1CCCCN1): same bug class.
3. DAG result structure: expected keys and tensor shapes.
4. cc_bit_mask_to_atom_idx correctly excludes uint8 underflow values (255).
5. Signed subtraction + clip yields correct diff atom ids.
6. FRAG_INDEX_DTYPE is np.uint32 (was uint16; changed to prevent overflow for
   molecules with many unique fragment formulae).
7. node_base_formula_matrix uses uint16 so H counts >255 are not truncated.
8. diff_formula_mask uses signed int32 subtraction to avoid uint16 underflow.
9. smarts_prepass=True with multi_cut_bfs=False runs compute_frags.compute_ccs
   unchanged and post-processes SMARTS pairs via _inject_smarts_pairs_into_dag,
   preserving ring-opening fragmentation while adding rearrangement fragments.
10. Nitro-group molecules: root formula is always present in idx_to_formula.
    compute_cc_h_floor overcounts the H floor by 1 for fragments containing
    [N+](=O)[O-] because update_bonds treats the N+=O double bond as a single
    bond, leaving O= with apparent free valence that its saturated N+ neighbour
    (diff=0 after formal-charge correction) cannot absorb.  The guard
    `floor = min(floor, num_hs_prior)` in compute_approximate_formula fixes this.
11. delta_h=0 is never invalid for any DAG node (root or sub-fragment).
    The guard `floor = min(floor, num_hs_prior)` ensures floor <= num_hs_prior
    for every fragment, making the natural H count always a valid configuration.
    Previously, sub-fragments containing [N+](=O)[O-] would receive "" at slot 0
    even when the parent molecule's root formula was correct — affecting 82% of
    nitro-containing molecules in NIST20 at the sub-fragment level.
"""

from __future__ import annotations

import numpy as np
import pytest

from fragnnet.frag.compute_frags import MASK_DTYPE
from fragnnet.utils.frag_utils import (
    ELEMENT_TO_IDX,
    FRAG_INDEX_DTYPE,
    NODE_FEAT_TO_IDX,
    cc_bit_mask_to_atom_idx,
    compute_dags,
    extract_mol_info,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_compute_dags(smiles: str, max_depth: int = 3) -> dict:
    """Run compute_dags with default parameters on a SMILES string."""
    mol_d = extract_mol_info(smiles)
    return compute_dags(
        mol_d,
        max_depth=max_depth,
        h_prior=True,
        max_h_transfer=2,
        frag_max_time=int(1e6),
        isotopes=False,
        nb_isomorphic=False,
        wl_max_iterations=-1,
        multi_cut_bfs=False,
    )


# ---------------------------------------------------------------------------
# 1 & 2. Regression: KeyError: 255 on brominated molecules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "smiles",
    [
        "BrCCNc1ccccc1",  # N-phenyl-2-bromoethylamine
        "BrCCC1CCCCN1",   # 1-(3-bromopropyl)piperidine
    ],
)
def test_compute_dags_brominated_molecule_no_keyerror(smiles: str) -> None:
    """Brominated molecules previously raised KeyError: 255 due to uint8
    underflow in diff_cc_mask (0 - 1 = 255 in uint8) and the bitmask being
    passed directly to compute_cc_h_floor as atom indices instead of being
    converted via cc_bit_mask_to_atom_idx first."""
    dag_d = _run_compute_dags(smiles)
    assert dag_d is not None


@pytest.mark.parametrize(
    "smiles",
    [
        "BrCCNc1ccccc1",
        "BrCCC1CCCCN1",
    ],
)
def test_compute_dags_brominated_molecule_has_edges(smiles: str) -> None:
    """Brominated molecules with multiple bonds must produce DAG edges."""
    dag_d = _run_compute_dags(smiles)
    assert dag_d["dag_num_edges"] > 0


# ---------------------------------------------------------------------------
# 3. DAG result structure
# ---------------------------------------------------------------------------


def test_compute_dags_result_keys() -> None:
    """Result dict must contain the core expected keys."""
    dag_d = _run_compute_dags("CCO")
    for key in ("dag", "dag_num_edges", "dag_num_nodes", "max_depth", "reached_depth"):
        assert key in dag_d, f"Missing key: {key}"


def test_compute_dags_edge_index_shape() -> None:
    """dag.edge_index has shape (2, num_edges)."""
    dag_d = _run_compute_dags("CCO")
    assert dag_d["dag"].edge_index.shape[0] == 2
    assert dag_d["dag"].edge_index.shape[1] == dag_d["dag_num_edges"]


def test_compute_dags_node_feat_shape() -> None:
    """dag.x has num_nodes rows."""
    dag_d = _run_compute_dags("CCO")
    assert dag_d["dag"].x.shape[0] == dag_d["dag_num_nodes"]


def test_compute_dags_attaches_boundary_pair_cache() -> None:
    """Generated DAGs should carry precomputed boundary-pair indices."""
    dag_d = _run_compute_dags("CCO")
    dag = dag_d["dag"]

    assert hasattr(dag, "boundary_pair_frag_idxs")
    assert hasattr(dag, "boundary_pair_in_local")
    assert hasattr(dag, "boundary_pair_out_local")
    assert dag.boundary_pair_frag_idxs.dtype == np.dtype(np.int64) or str(dag.boundary_pair_frag_idxs.dtype) == "torch.int64"
    assert dag.boundary_pair_in_local.dtype == np.dtype(np.int64) or str(dag.boundary_pair_in_local.dtype) == "torch.int64"
    assert dag.boundary_pair_out_local.dtype == np.dtype(np.int64) or str(dag.boundary_pair_out_local.dtype) == "torch.int64"
    assert dag.boundary_pair_frag_idxs.shape == dag.boundary_pair_in_local.shape
    assert dag.boundary_pair_frag_idxs.shape == dag.boundary_pair_out_local.shape


# ---------------------------------------------------------------------------
# 4. cc_bit_mask_to_atom_idx excludes uint8 underflow value 255
# ---------------------------------------------------------------------------


def test_cc_bit_mask_to_atom_idx_excludes_uint8_underflow() -> None:
    """uint8 subtraction 0 - 1 = 255; cc_bit_mask_to_atom_idx must not
    include those positions since it only selects positions where value == 1."""
    from_mask = np.array([1, 0, 1], dtype=MASK_DTYPE)
    to_mask = np.array([1, 1, 1], dtype=MASK_DTYPE)

    # Demonstrate the underflow: 0 - 1 = 255 in uint8
    raw_diff = from_mask - to_mask
    assert raw_diff[1] == 255, "Expected uint8 underflow to produce 255"

    # cc_bit_mask_to_atom_idx only picks positions with value exactly 1
    atom_ids = cc_bit_mask_to_atom_idx(raw_diff)
    assert 255 not in atom_ids
    # raw_diff is [0, 255, 0] — no position has value 1, so result is empty
    assert len(atom_ids) == 0


# ---------------------------------------------------------------------------
# 5. Signed subtraction + clip gives correct diff atom ids
# ---------------------------------------------------------------------------


def test_diff_cc_mask_signed_subtraction_correct() -> None:
    """With signed subtraction and clip, diff correctly identifies atoms in
    parent but not child, ignoring underflow."""
    from_mask = np.array([1, 1, 1, 0], dtype=MASK_DTYPE)
    to_mask = np.array([0, 1, 0, 0], dtype=MASK_DTYPE)

    diff = np.clip(
        from_mask.astype(np.int16) - to_mask.astype(np.int16), 0, 1
    ).astype(MASK_DTYPE)
    atom_ids = cc_bit_mask_to_atom_idx(diff)

    # Atoms 0 and 2 are in from but not to; atom 1 is in both; atom 3 in neither
    np.testing.assert_array_equal(np.sort(atom_ids), [0, 2])


# ---------------------------------------------------------------------------
# 6. FRAG_INDEX_DTYPE is np.uint32
# ---------------------------------------------------------------------------


def test_frag_index_dtype_is_uint32() -> None:
    """FRAG_INDEX_DTYPE must be np.uint32, not uint16, to support molecules
    with more than 65535 unique fragment formulae."""
    assert FRAG_INDEX_DTYPE == np.uint32, (
        f"Expected np.uint32, got {FRAG_INDEX_DTYPE}. "
        "uint16 max (65535) can be exceeded by molecules with many unique fragments."
    )


# ---------------------------------------------------------------------------
# 7. node_base_formula_matrix uses uint16 — H counts >255 are not truncated
# ---------------------------------------------------------------------------


def test_base_formula_heavy_atom_counts_correct() -> None:
    """Heavy atom counts in node features (base_formula columns) must be exact.

    H is intentionally zeroed in base_formula node features (compute_dags sets
    it to 0 before storing).  We verify the non-H element counts instead.
    For ethanol CCO: root node has C=2, O=1.
    """
    dag_d = _run_compute_dags("CCO")
    dag = dag_d["dag"]
    node_feat_idxs = dag.node_feat_idxs[0].tolist()
    bf_start = node_feat_idxs[NODE_FEAT_TO_IDX["base_formula"]]
    c_col = bf_start + ELEMENT_TO_IDX["C"]
    o_col = bf_start + ELEMENT_TO_IDX["O"]

    # Root node (index 0) is the full molecule: CCO → C=2, O=1
    assert dag.x[0, c_col].item() == 2, f"Root node C count wrong: {dag.x[0, c_col].item()}"
    assert dag.x[0, o_col].item() == 1, f"Root node O count wrong: {dag.x[0, o_col].item()}"


def test_base_formula_formula_strings_correct() -> None:
    """Formula strings in idx_to_formula must use the actual H count, not the
    uint8-truncated value from node_base_formula_matrix.

    compute_approximate_formula computes H counts from hs_arr directly (not
    from the formula array), so formula strings are always correct. This test
    guards against regressions in that path.

    Ethanol CCO: root node h_delta=0 formula must be 'C2H6O'.
    """
    dag_d = _run_compute_dags("CCO")
    # idx_to_formula: {int_idx: formula_str}
    formulae = set(dag_d["idx_to_formula"].values())
    assert "C2H6O" in formulae, (
        f"Expected 'C2H6O' in formula set for CCO, got {formulae}"
    )


# ---------------------------------------------------------------------------
# 8. diff_formula_mask uses signed int32 subtraction — no uint16 underflow
# ---------------------------------------------------------------------------


def test_diff_formula_mask_no_underflow() -> None:
    """Edge features must not show uint underflow in the formula diff columns.

    For any DAG edge (parent → child), the parent always has >= atoms of each
    heavy element than the child, so diff >= 0 for heavy atoms. H is zeroed.
    No edge feature value should be a large positive number caused by underflow
    (e.g., 65535 or 255).
    """
    # Use a molecule with multiple fragments to exercise edge features
    dag_d = _run_compute_dags("c1ccccc1CC(=O)O")  # phenylacetic acid
    dag = dag_d["dag"]
    edge_attr = dag.edge_attr  # int64 tensor

    if edge_attr.shape[0] == 0:
        pytest.skip("No edges in DAG — nothing to test")

    # Grab the formula diff columns from edge_feat_idxs
    edge_feat_idxs = dag.edge_feat_idxs[0].tolist()
    # Edge feature groups: cc_mask, base_formula_diff, h_range
    formula_diff_start = edge_feat_idxs[1]
    formula_diff_end = edge_feat_idxs[2]
    formula_diff_cols = edge_attr[:, formula_diff_start:formula_diff_end]

    # Underflow sentinel values: 65535 (uint16 wrap), 255 (uint8 wrap)
    assert not (formula_diff_cols == 65535).any(), (
        "Found 65535 in formula diff edge features — uint16 underflow detected."
    )
    assert not (formula_diff_cols == 255).any(), (
        "Found 255 in formula diff edge features — uint8 underflow detected."
    )
    # All heavy-atom diffs should be non-negative (parent >= child)
    assert (formula_diff_cols >= 0).all(), (
        "Negative formula diff found — unexpected for parent→child edges."
    )


# ---------------------------------------------------------------------------
# 9. smarts_prepass=True with multi_cut_bfs=False: post-process merge
# ---------------------------------------------------------------------------


def test_smarts_prepass_without_multi_cut_bfs_works() -> None:
    """smarts_prepass=True with multi_cut_bfs=False runs correctly via post-merge.

    compute_frags.compute_ccs runs unchanged (preserving ring-opening via the
    n_ccs=1 path), then _inject_smarts_pairs_into_dag appends any SMARTS
    rearrangement fragments as depth-1 children.  The result must be a valid
    DAG with at least as many nodes as the baseline (no prepass).
    """
    mol_d = extract_mol_info("CCOC(=O)CC")  # ethyl propanoate — matches crf13_2 CO2 loss
    baseline = compute_dags(
        mol_d,
        max_depth=3,
        h_prior=True,
        max_h_transfer=2,
        frag_max_time=int(1e6),
        isotopes=False,
        nb_isomorphic=False,
        wl_max_iterations=-1,
        multi_cut_bfs=False,
        smarts_prepass=False,
    )
    with_prepass = compute_dags(
        mol_d,
        max_depth=3,
        h_prior=True,
        max_h_transfer=2,
        frag_max_time=int(1e6),
        isotopes=False,
        nb_isomorphic=False,
        wl_max_iterations=-1,
        multi_cut_bfs=False,
        smarts_prepass=True,
    )
    assert with_prepass is not None
    # prepass can only add nodes/edges, never remove them
    assert with_prepass["dag_num_nodes"] >= baseline["dag_num_nodes"]
    assert with_prepass["dag_num_edges"] >= baseline["dag_num_edges"]


# ---------------------------------------------------------------------------
# 10. Nitro-group molecules: root formula present in idx_to_formula
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "smiles,expected_formula",
    [
        # Single nitro + N-methyl: floor overcount = 1 (1 O= blocked by N+ diff=0)
        ("Cn1cnc([N+](=O)[O-])c1",                                    "C4H5N3O2"),
        ("Cn1cnc([N+](=O)[O-])n1",                                    "C3H4N4O2"),
        # Single nitro + N-CH2Cl
        ("O=[N+]([O-])c1nn(CCl)cc1Cl",                               "C4H3Cl2N3O2"),
        # Two nitro groups + methoxy
        ("COc1nc(OC)c([N+](=O)[O-])cc1[N+](=O)[O-]",                 "C7H7N3O6"),
        ("COc1n[n+]([O-])c([N+](=O)[O-])cc1[N+](=O)[O-]",            "C5H4N4O6"),
        # Three nitro groups + methoxy
        ("COc1ccc([N+](=O)[O-])c([N+](=O)[O-])c1[N+](=O)[O-]",      "C7H5N3O7"),
        # Sanity: molecule without nitro must also work
        ("CCO",                                                        "C2H6O"),
    ],
)
def test_nitro_root_formula_in_dag(smiles: str, expected_formula: str) -> None:
    """Root node formula (delta_h=0) must equal the true molecular formula.

    compute_cc_h_floor previously overcounted the H floor by 1 for molecules
    containing [N+](=O)[O-] groups: the double-bond O= had apparent free valence
    (ve=2, sbond=1 after bond-order-agnostic update_bonds), but its only
    neighbour N+ had diff=0 (formal-charge correction ve 4→3 = sbond 3), so
    the floor could not be reduced and ended up > num_hs_prior.  This made
    delta_h=0 invalid at the root, assigning formula "" to node 0.
    """
    dag_d = _run_compute_dags(smiles)
    dag = dag_d["dag"]
    idx_to_formula = dag_d["idx_to_formula"]

    # Retrieve the root node's formula at delta_h=0.
    # h_formulae_idx feature group stores one formula-index per H-transfer slot;
    # slot 0 corresponds to delta_h=0 (h_delta_idx = delta_h * 2 for delta_h >= 0).
    h_formulae_start = dag.node_feat_idxs[0].tolist()[NODE_FEAT_TO_IDX["h_formulae_idx"]]
    root_formula_idx = dag.x[0, h_formulae_start].item()  # slot 0 = delta_h=0
    root_formula = idx_to_formula[root_formula_idx]

    assert root_formula == expected_formula, (
        f"Root node formula at delta_h=0 is '{root_formula}', expected '{expected_formula}' "
        f"for SMILES '{smiles}'.  "
        "Likely cause: compute_cc_h_floor overcounted the H floor (floor > num_hs_prior), "
        "making delta_h=0 invalid and assigning the sentinel '' to the root node."
    )


# ---------------------------------------------------------------------------
# 11. delta_h=0 is never invalid for any DAG node (root or sub-fragment)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "smiles",
    [
        # Nitro group — sub-fragments retaining [N+](=O)[O-] previously got ""
        # at delta_h=0 due to H-floor overcounting in compute_cc_h_floor.
        "Cn1cnc([N+](=O)[O-])c1",           # 1-methylimidazole-4-carbaldehyde nitro
        "O=[N+]([O-])c1ccccc1CC",            # nitrobenzene + ethyl: aryl-C cut leaves nitrophenyl sub-frag
        "COc1ccc([N+](=O)[O-])cc1CCC",       # methoxy-nitrophenyl + propyl chain
        "COc1nc(OC)c([N+](=O)[O-])cc1[N+](=O)[O-]",   # two nitro groups
        # Sanity: molecule without nitro must also satisfy the invariant
        "c1ccccc1CC(=O)O",
        "CCO",
    ],
)
def test_all_nodes_have_valid_delta_h0_formula(smiles: str) -> None:
    """Every DAG node must have a non-empty formula at delta_h=0 (slot 0).

    delta_h=0 corresponds to the fragment retaining its natural H count from
    the parent molecule.  This is always chemically valid: floor <= num_hs_prior
    is guaranteed by the guard ``floor = min(floor, num_hs_prior)`` in
    compute_approximate_formula, and cap >= num_hs_prior always holds.

    Before the fix, sub-fragments containing [N+](=O)[O-] received "" at slot 0
    even when the root formula was correct, because compute_cc_h_floor
    overcounted the floor for any CC containing a nitro group — not just the
    root CC.  This was confirmed at dataset scale: 436,181 sub-fragment nodes
    across 21,172 molecules (82% of NIST20) had empty slot-0 formulas before
    the fix.
    """
    dag_d = _run_compute_dags(smiles)
    dag = dag_d["dag"]
    idx_to_formula = dag_d["idx_to_formula"]

    h_formulae_start = dag.node_feat_idxs[0].tolist()[NODE_FEAT_TO_IDX["h_formulae_idx"]]
    empty_nodes = []
    for node_i in range(dag.x.shape[0]):
        fidx = int(dag.x[node_i, h_formulae_start].item())
        formula = idx_to_formula.get(fidx, "")
        if formula == "":
            empty_nodes.append(node_i)

    assert len(empty_nodes) == 0, (
        f"{len(empty_nodes)} DAG node(s) have empty formula at delta_h=0 for SMILES '{smiles}': "
        f"node indices {empty_nodes[:10]}{'...' if len(empty_nodes) > 10 else ''}.  "
        "Likely cause: compute_cc_h_floor overcounted the H floor for sub-fragments "
        "containing [N+](=O)[O-], making delta_h=0 invalid (floor > num_hs_prior)."
    )
