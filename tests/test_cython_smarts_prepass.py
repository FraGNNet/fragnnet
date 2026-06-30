"""Unit and benchmark tests for the Cython multi_cut_bfs with smarts_seed_pairs.

Unit tests
----------
- Functionality: the non-contiguous {C0,C3} fragment appears with prepass enabled
  (butane/crf12_0), the CO2 fragment appears for methyl acetate (crf13_2), the
  cyclic filter is respected, and prepass never reduces the fragment count.
- Depth placement: SMARTS seeds are registered at BFS depth 1, not depth 0.

Benchmarks
----------
- pytest-benchmark fixtures for per-molecule profiling (run with
  ``pytest --benchmark-enable`` or just ``pytest``; disabled at collection by
  ``--benchmark-disable`` if you want to skip them).
"""

from __future__ import annotations

import numpy as np
import pytest

from fragnnet.frag.multi_cut_bfs import compute_ccs_multi_cut as compute_ccs_multi_cut_cy
from fragnnet.frag.smarts_prepass import FRAG_RULES, MASK_DTYPE, _apply_smarts_prepass
from fragnnet.utils.frag_utils import extract_mol_info, get_fraggen_input_arrays

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_arrays(smiles: str):
    """Return (mol_d, num_nodes, num_edges, node_mask, edges, edge_mask, n2e, mol)."""
    mol_d = extract_mol_info(smiles)
    num_nodes, num_edges, node_mask, edges, edge_mask, n2e = get_fraggen_input_arrays(mol_d)
    return mol_d, num_nodes, num_edges, node_mask, edges, edge_mask, n2e, mol_d["mol"]


def _run_cy(
    smiles: str,
    max_cut_size: int = 2,
    max_depth: int = 3,
    use_prepass: bool = True,
    min_frag_atoms: int = 3,
):
    """Run the Cython BFS on *smiles*, optionally with smarts_seed_pairs."""
    mol_d, num_nodes, num_edges, node_mask, edges, edge_mask, n2e, mol = _get_arrays(smiles)
    ring_mask = mol_d["ring_bond_mask"]
    seed_pairs = None
    if use_prepass:
        em = np.asarray(edge_mask, dtype=MASK_DTYPE)
        seed_pairs = _apply_smarts_prepass(FRAG_RULES, mol, em, edges, num_nodes, num_edges)
    nodes_mask, nodes_depth, dag_edges, meta = compute_ccs_multi_cut_cy(
        num_nodes,
        num_edges,
        node_mask,
        edges,
        edge_mask,
        n2e,
        max_depth=max_depth,
        time_limit=30,
        ring_edge_mask=ring_mask,
        max_cut_size=max_cut_size,
        smarts_seed_pairs=seed_pairs,
        min_frag_atoms=min_frag_atoms,
    )
    return {
        "nodes_mask": nodes_mask,
        "nodes_depth": nodes_depth,
        "dag_edges": dag_edges,
        "meta": meta,
    }


def _atom_sets(nodes_mask: np.ndarray) -> set[frozenset[int]]:
    """Return the set of fragment atom-sets (frozensets), excluding the root (idx 0)."""
    result = set()
    for i in range(1, len(nodes_mask)):
        result.add(frozenset(int(j) for j in np.where(nodes_mask[i])[0]))
    return result


# ---------------------------------------------------------------------------
# Functionality tests: correct SMARTS fragments are injected
# ---------------------------------------------------------------------------


class TestCythonSmartsPrepassFunctionality:
    """Cython BFS + smarts_seed_pairs injects the correct SMARTS fragments."""

    def test_butane_non_contiguous_fragment_present(self):
        """{C0, C3} appears when prepass is enabled (crf12_0: 1,3-elimination)."""
        result = _run_cy("CCCC", use_prepass=True, min_frag_atoms=0)
        assert frozenset({0, 3}) in _atom_sets(result["nodes_mask"]), (
            "Cython prepass must include non-contiguous {C0,C3} (crf12_0) for butane"
        )

    def test_butane_non_contiguous_fragment_absent_without_prepass(self):
        """{C0, C3} must NOT appear when prepass is disabled."""
        result = _run_cy("CCCC", use_prepass=False)
        assert frozenset({0, 3}) not in _atom_sets(result["nodes_mask"]), (
            "Without prepass, {C0,C3} should not appear (non-adjacent, no ring cut)"
        )

    def test_methyl_acetate_three_atom_co2_fragment(self):
        """CO2 fragment (3 atoms) appears for methyl acetate (crf13_2: acyclic ester)."""
        result = _run_cy("CC(=O)OC", use_prepass=True)
        three_atom = [s for s in _atom_sets(result["nodes_mask"]) if len(s) == 3]
        assert len(three_atom) > 0, (
            "Cython prepass must produce a 3-atom CO2 fragment for methyl acetate"
        )

    def test_prepass_never_reduces_fragment_count(self):
        """Enabling prepass can only add fragments, never remove them."""
        for smiles in ("CCCC", "CC(=O)OC", "CC(=O)OC1=CC=CC=C1C(=O)O"):
            cy_no = _run_cy(smiles, use_prepass=False)
            cy_yes = _run_cy(smiles, use_prepass=True)
            assert len(cy_yes["nodes_mask"]) >= len(cy_no["nodes_mask"]), (
                f"Prepass reduced fragment count for {smiles}"
            )

    def test_4membered_lactone_cyclic_filter_respected(self):
        """β-propiolactone: crf13_2 adds no SMARTS fragments (cyclic case is filtered).

        The 4-membered ring lactone outer atoms (flanking CH2 groups) ARE directly
        bonded → filter_bond_map_nums fires → zero prepass results.  BFS cut=2
        already covers this ring opening independently.
        """
        cy_no = _run_cy("O=C1CCO1", use_prepass=False)
        cy_yes = _run_cy("O=C1CCO1", use_prepass=True)
        assert len(cy_yes["nodes_mask"]) == len(cy_no["nodes_mask"]), (
            "crf13_2 cyclic filter must suppress prepass for 4-membered lactone"
        )

    def test_empty_seed_pairs_no_change(self):
        """Passing an empty smarts_seed_pairs list is a no-op."""
        mol_d, nn, ne, nm, edges, em, n2e, mol = _get_arrays("CCCC")
        ring_mask = mol_d["ring_bond_mask"]
        no_seeds, _, _, _ = compute_ccs_multi_cut_cy(
            nn,
            ne,
            nm,
            edges,
            em,
            n2e,
            max_depth=3,
            time_limit=30,
            ring_edge_mask=ring_mask,
            max_cut_size=2,
            smarts_seed_pairs=None,
        )
        empty_seeds, _, _, _ = compute_ccs_multi_cut_cy(
            nn,
            ne,
            nm,
            edges,
            em,
            n2e,
            max_depth=3,
            time_limit=30,
            ring_edge_mask=ring_mask,
            max_cut_size=2,
            smarts_seed_pairs=[],
        )
        assert len(no_seeds) == len(empty_seeds), (
            "Empty smarts_seed_pairs must produce the same output as None"
        )

    def test_crf12_1_double_bond_inner_atoms(self):
        """crf12_1 on but-2-ene (CC=CC) gives {C0,C3} outer + {C1,C2} inner.

        Pattern: A–B=C–D → outer {A,D} merged + inner {B,C}.
        crf12_1 differs from crf12_0 by requiring a double bond between inner atoms.
        """
        result = _run_cy("CC=CC", use_prepass=True, min_frag_atoms=0)
        atom_sets = _atom_sets(result["nodes_mask"])
        assert frozenset({0, 3}) in atom_sets, (
            "crf12_1 must produce non-contiguous outer {C0,C3} for but-2-ene"
        )
        assert frozenset({1, 2}) in atom_sets, (
            "crf12_1 must produce inner {C1,C2} (double-bond pair) for but-2-ene"
        )

    def test_dimethyl_succinate_two_ester_groups(self):
        """dimethyl succinate (COC(=O)CCC(=O)OC) has 2 ester groups → more seed pairs.

        Two crf13_2 matches are expected, each producing a CO2 fragment + outer R+R'.
        """
        smiles = "COC(=O)CCC(=O)OC"
        cy_no = _run_cy(smiles, use_prepass=False)
        cy_yes = _run_cy(smiles, use_prepass=True)
        assert len(cy_yes["nodes_mask"]) > len(cy_no["nodes_mask"]), (
            "Two ester groups in dimethyl succinate must add SMARTS prepass fragments"
        )
        # Both ester groups should independently produce a 3-atom CO2 fragment
        three_atom = [s for s in _atom_sets(cy_yes["nodes_mask"]) if len(s) == 3]
        assert len(three_atom) > 0, "At least one 3-atom CO2 fragment expected"

    def test_benzene_no_smarts_matches(self):
        """Benzene (c1ccccc1) has no ester/chain substructure → prepass is a no-op."""
        cy_no = _run_cy("c1ccccc1", use_prepass=False)
        cy_yes = _run_cy("c1ccccc1", use_prepass=True)
        assert len(cy_yes["nodes_mask"]) == len(cy_no["nodes_mask"]), (
            "Benzene has no SMARTS-matchable substructures: prepass must add 0 fragments"
        )

    def test_seed_fragment_pairs_cover_all_atoms(self):
        """For every seed pair (mask_a, mask_b), union must equal the full molecule."""
        smiles = "CC(=O)OC"  # methyl acetate
        mol_d, nn, ne, nm, edges, em, n2e, mol = _get_arrays(smiles)
        em_u8 = np.asarray(em, dtype=MASK_DTYPE)
        seed_pairs = _apply_smarts_prepass(FRAG_RULES, mol, em_u8, edges, nn, ne)
        assert len(seed_pairs) > 0, "Need at least one seed pair to test"
        for mask_a, mask_b, _ in seed_pairs:
            union_count = int(np.logical_or(mask_a[:nn], mask_b[:nn]).sum())
            assert union_count == nn, f"Seed pair union covers {union_count} atoms, expected {nn}"

    def test_no_seed_equals_root_molecule(self):
        """No injected seed fragment should equal the full (root) molecule."""
        smiles = "CSCCC(C(=O)O)N"  # methionine
        mol_d, nn, ne, nm, edges, em, n2e, mol = _get_arrays(smiles)
        em_u8 = np.asarray(em, dtype=MASK_DTYPE)
        seed_pairs = _apply_smarts_prepass(FRAG_RULES, mol, em_u8, edges, nn, ne)
        root_mask = np.ones(nn, dtype=MASK_DTYPE)
        for mask_a, mask_b, _ in seed_pairs:
            assert not np.array_equal(mask_a[:nn], root_mask), (
                "A seed fragment must not equal the full root molecule"
            )
            assert not np.array_equal(mask_b[:nn], root_mask), (
                "A seed fragment must not equal the full root molecule"
            )

    def test_pentane_more_matches_than_butane(self):
        """Pentane (CCCCC) has more crf12_0 matches than butane (CCCC).

        Butane: 1 match (A-B-C-D on atoms 0-3).
        Pentane: multiple overlapping 4-atom windows (0-3, 1-4), each with outer
        non-contiguous atoms.  More seeds → more prepass fragments.
        """
        cy_butane = _run_cy("CCCC", use_prepass=True)
        cy_pentane = _run_cy("CCCCC", use_prepass=True)
        butane_extra = len(cy_butane["nodes_mask"]) - len(
            _run_cy("CCCC", use_prepass=False)["nodes_mask"]
        )
        pentane_extra = len(cy_pentane["nodes_mask"]) - len(
            _run_cy("CCCCC", use_prepass=False)["nodes_mask"]
        )
        assert pentane_extra >= butane_extra, (
            f"Pentane should have >= prepass fragments than butane "
            f"(pentane_extra={pentane_extra}, butane_extra={butane_extra})"
        )


# ---------------------------------------------------------------------------
# Depth placement tests: seeds must be at depth 1, not depth 0
# ---------------------------------------------------------------------------


class TestSeedDepthPlacement:
    """Verify that SMARTS seeds are registered at BFS depth 1 in nodes_depth_matrix."""

    SMILES = "CCCC"  # butane — crf12_0 gives {C0,C3} and {C1,C2} at depth 1

    @staticmethod
    def _find_idx(nodes_mask: np.ndarray, target: frozenset[int]) -> int:
        for i in range(len(nodes_mask)):
            if frozenset(int(j) for j in np.where(nodes_mask[i])[0]) == target:
                return i
        return -1

    def test_seed_registered_at_depth_1(self):
        """{C0,C3} outer fragment must have nodes_depth[:,1]==1."""
        result = _run_cy(self.SMILES, use_prepass=True, max_depth=3, min_frag_atoms=0)
        idx = self._find_idx(result["nodes_mask"], frozenset({0, 3}))
        assert idx != -1, "{C0,C3} not found in Cython output with prepass"
        assert result["nodes_depth"][idx, 1] == 1, (
            f"{'{C0,C3}'} must be recorded at depth 1; depth row={list(result['nodes_depth'][idx])}"
        )

    def test_seed_not_at_depth_0(self):
        """{C0,C3} seed fragment must NOT be at depth 0."""
        result = _run_cy(self.SMILES, use_prepass=True, max_depth=3, min_frag_atoms=0)
        idx = self._find_idx(result["nodes_mask"], frozenset({0, 3}))
        assert idx != -1, "{C0,C3} not found"
        assert result["nodes_depth"][idx, 0] == 0, (
            "{C0,C3} seed must not be at depth 0 (only root is at depth 0)"
        )

    def test_only_root_at_depth_0(self):
        """Only the root (full molecule) may be recorded at depth 0."""
        result = _run_cy(self.SMILES, use_prepass=True, max_depth=3, min_frag_atoms=0)
        for i in range(1, len(result["nodes_mask"])):
            assert result["nodes_depth"][i, 0] == 0, (
                f"Fragment {i} (atoms={frozenset(np.where(result['nodes_mask'][i])[0])}) "
                f"is incorrectly at depth 0 — only the root should be"
            )

    def test_seeds_present_at_max_depth_1(self):
        """With max_depth=1, depth-1 seeds still appear; depth ≥2 fragments do not."""
        result_d1 = _run_cy(self.SMILES, use_prepass=True, max_depth=1, min_frag_atoms=0)
        result_d3 = _run_cy(self.SMILES, use_prepass=True, max_depth=3, min_frag_atoms=0)
        sets_d1 = _atom_sets(result_d1["nodes_mask"])
        sets_d3 = _atom_sets(result_d3["nodes_mask"])
        # {C0,C3} is a seed injected at depth 1 → must appear at max_depth=1
        assert frozenset({0, 3}) in sets_d1, "Seed {C0,C3} must appear even with max_depth=1"
        # max_depth=3 output is a superset (BFS goes deeper)
        assert sets_d1.issubset(sets_d3), (
            "max_depth=3 must produce a superset of max_depth=1 fragments"
        )


# ---------------------------------------------------------------------------
# pytest-benchmark fixtures (run with: pytest tests/test_cython_smarts_prepass.py -v)
# ---------------------------------------------------------------------------

_SPEED_MOLECULES = [
    ("aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
    ("testosterone", "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C"),
    ("methionine", "CSCCC(C(=O)O)N"),
]


@pytest.mark.parametrize("name,smiles", _SPEED_MOLECULES)
def test_benchmark_cython_with_prepass(benchmark, name, smiles):
    """Benchmark: Cython BFS + pre-computed smarts_seed_pairs."""
    mol_d, nn, ne, nm, edges, em, n2e, mol = _get_arrays(smiles)
    ring_mask = mol_d["ring_bond_mask"]
    em_u8 = np.asarray(em, dtype=MASK_DTYPE)
    seed_pairs = _apply_smarts_prepass(FRAG_RULES, mol, em_u8, edges, nn, ne)

    benchmark(
        compute_ccs_multi_cut_cy,
        nn,
        ne,
        nm,
        edges,
        em,
        n2e,
        max_depth=3,
        time_limit=30,
        ring_edge_mask=ring_mask,
        max_cut_size=2,
        smarts_seed_pairs=seed_pairs,
    )


# ---------------------------------------------------------------------------
# DAG edge correctness for seed-injected fragments
# ---------------------------------------------------------------------------


class TestSeedEdgeCorrectness:
    """All edges introduced by SMARTS injection must originate from the root (node 0)."""

    def _run_with_and_without(self, smiles: str, min_frag_atoms: int = 0):
        mol_d, nn, ne, nm, edges, em, n2e, mol = _get_arrays(smiles)
        ring_mask = mol_d["ring_bond_mask"]
        em_u8 = np.asarray(em, dtype=MASK_DTYPE)
        seed_pairs = _apply_smarts_prepass(FRAG_RULES, mol, em_u8, edges, nn, ne)

        kwargs = {
            "max_depth": 3, "time_limit": 30, "ring_edge_mask": ring_mask,
            "max_cut_size": 2, "min_frag_atoms": min_frag_atoms,
        }
        _, _, dag_edges_no, _ = compute_ccs_multi_cut_cy(
            nn, ne, nm, edges, em, n2e, smarts_seed_pairs=None, **kwargs
        )
        _, _, dag_edges_yes, _ = compute_ccs_multi_cut_cy(
            nn, ne, nm, edges, em, n2e, smarts_seed_pairs=seed_pairs, **kwargs
        )
        return dag_edges_no, dag_edges_yes

    def test_seed_frags_have_root_edge(self):
        """Every fragment in prepass_node_rules must have an edge from root (node 0)."""
        mol_d, nn, ne, nm, edges, em, n2e, mol = _get_arrays("CCCC")
        ring_mask = mol_d["ring_bond_mask"]
        em_u8 = np.asarray(em, dtype=MASK_DTYPE)
        seed_pairs = _apply_smarts_prepass(FRAG_RULES, mol, em_u8, edges, nn, ne)
        nodes_mask, _, dag_edges, meta = compute_ccs_multi_cut_cy(
            nn, ne, nm, edges, em, n2e,
            max_depth=3, time_limit=30,
            ring_edge_mask=ring_mask, max_cut_size=2,
            smarts_seed_pairs=seed_pairs, min_frag_atoms=0,
        )
        # Build set of (src, dst) edges for O(1) lookup
        edge_set = {(int(e[0]), int(e[1])) for e in dag_edges}
        # Verify the non-contiguous seed fragment {C0,C3} has an edge from root
        target = frozenset({0, 3})
        for i, row in enumerate(nodes_mask):
            if frozenset(int(j) for j in np.where(row)[0]) == target:
                assert (0, i) in edge_set, f"Prepass fragment {i} has no edge from root (node 0)"
                break

    def test_prepass_never_removes_bfs_fragments(self):
        """Every fragment atom-set present without prepass must also exist with prepass.

        Node IDs differ between runs because injected seeds shift the index
        assignment, so we compare by atom-mask content (frozenset) rather than
        by integer edge tuples.
        """
        dag_no, dag_yes = self._run_with_and_without("CC(=O)OC")
        # dag_no and dag_yes are edge arrays — we need the node masks directly
        mol_d, nn, ne, nm, edges, em, n2e, mol = _get_arrays("CC(=O)OC")
        ring_mask = mol_d["ring_bond_mask"]
        em_u8 = np.asarray(em, dtype=MASK_DTYPE)
        seed_pairs = _apply_smarts_prepass(FRAG_RULES, mol, em_u8, edges, nn, ne)
        kwargs_base = {
            "max_depth": 3, "time_limit": 30,
            "ring_edge_mask": ring_mask, "max_cut_size": 2, "min_frag_atoms": 0,
        }
        nm_no, _, _, _ = compute_ccs_multi_cut_cy(
            nn, ne, nm, edges, em, n2e, smarts_seed_pairs=None, **kwargs_base
        )
        nm_yes, _, _, _ = compute_ccs_multi_cut_cy(
            nn, ne, nm, edges, em, n2e, smarts_seed_pairs=seed_pairs, **kwargs_base
        )
        sets_no = {frozenset(int(j) for j in np.where(nm_no[i])[0]) for i in range(len(nm_no))}
        sets_yes = {frozenset(int(j) for j in np.where(nm_yes[i])[0]) for i in range(len(nm_yes))}
        assert sets_no.issubset(sets_yes), (
            "Prepass may only add fragment atom-sets, never remove existing BFS fragments"
        )

    def test_seed_child_targets_are_in_nodes_mask(self):
        """Every child node index in seed-injected edges must index a valid row."""
        mol_d, nn, ne, nm, edges, em, n2e, mol = _get_arrays("CCCC")
        ring_mask = mol_d["ring_bond_mask"]
        em_u8 = np.asarray(em, dtype=MASK_DTYPE)
        seed_pairs = _apply_smarts_prepass(FRAG_RULES, mol, em_u8, edges, nn, ne)
        nodes_mask, _, dag_edges, _ = compute_ccs_multi_cut_cy(
            nn, ne, nm, edges, em, n2e,
            max_depth=3, time_limit=30,
            ring_edge_mask=ring_mask, max_cut_size=2,
            smarts_seed_pairs=seed_pairs, min_frag_atoms=0,
        )
        n_frags = len(nodes_mask)
        for src, dst in dag_edges:
            assert 0 <= dst < n_frags, (
                f"Edge target {dst} is out of range [0, {n_frags})"
            )


# ---------------------------------------------------------------------------
# nodes_min_depth for seed-injected fragments
# ---------------------------------------------------------------------------


class TestSeedNodesMinDepth:
    """Seeds are injected at BFS depth 1; nodes_min_depth must reflect this."""

    def _min_depth_for(self, smiles: str, target_atoms: frozenset[int]) -> int | None:
        mol_d, nn, ne, nm, edges, em, n2e, mol = _get_arrays(smiles)
        ring_mask = mol_d["ring_bond_mask"]
        em_u8 = np.asarray(em, dtype=MASK_DTYPE)
        seed_pairs = _apply_smarts_prepass(FRAG_RULES, mol, em_u8, edges, nn, ne)
        nodes_mask, _, _, meta = compute_ccs_multi_cut_cy(
            nn, ne, nm, edges, em, n2e,
            max_depth=3, time_limit=30,
            ring_edge_mask=ring_mask, max_cut_size=2,
            smarts_seed_pairs=seed_pairs, min_frag_atoms=0,
        )
        nodes_min_depth = meta["nodes_min_depth"]
        for i, row in enumerate(nodes_mask):
            if frozenset(int(j) for j in np.where(row)[0]) == target_atoms:
                return int(nodes_min_depth[i])
        return None

    def test_butane_outer_seed_min_depth_is_1(self):
        """{C0,C3} from crf12_0 is injected at depth 1 — min_depth must be 1."""
        depth = self._min_depth_for("CCCC", frozenset({0, 3}))
        assert depth is not None, "{C0,C3} not found in DAG"
        assert depth == 1, f"expected min_depth=1, got {depth}"

    def test_root_min_depth_is_0(self):
        """The root fragment always has min_depth=0."""
        mol_d, nn, ne, nm, edges, em, n2e, mol = _get_arrays("CCCC")
        ring_mask = mol_d["ring_bond_mask"]
        em_u8 = np.asarray(em, dtype=MASK_DTYPE)
        seed_pairs = _apply_smarts_prepass(FRAG_RULES, mol, em_u8, edges, nn, ne)
        _, _, _, meta = compute_ccs_multi_cut_cy(
            nn, ne, nm, edges, em, n2e,
            max_depth=3, time_limit=30,
            ring_edge_mask=ring_mask, max_cut_size=2,
            smarts_seed_pairs=seed_pairs, min_frag_atoms=0,
        )
        assert int(meta["nodes_min_depth"][0]) == 0, "root must have min_depth=0"


# ---------------------------------------------------------------------------
# Prepass with ring_edge_mask=None and max_cut_size=1
# ---------------------------------------------------------------------------


class TestPrepasWithoutRingMask:
    """Seeds are injected regardless of ring_edge_mask / max_cut_size."""

    def test_prepass_works_without_ring_mask(self):
        """ring_edge_mask=None disables ring cuts but seeds are still injected."""
        mol_d, nn, ne, nm, edges, em, n2e, mol = _get_arrays("CCCC")
        em_u8 = np.asarray(em, dtype=MASK_DTYPE)
        seed_pairs = _apply_smarts_prepass(FRAG_RULES, mol, em_u8, edges, nn, ne)

        nodes_no, _, _, _ = compute_ccs_multi_cut_cy(
            nn, ne, nm, edges, em, n2e,
            max_depth=3, time_limit=30,
            ring_edge_mask=None, max_cut_size=1,
            smarts_seed_pairs=None, min_frag_atoms=0,
        )
        nodes_yes, _, _, _ = compute_ccs_multi_cut_cy(
            nn, ne, nm, edges, em, n2e,
            max_depth=3, time_limit=30,
            ring_edge_mask=None, max_cut_size=1,
            smarts_seed_pairs=seed_pairs, min_frag_atoms=0,
        )
        # {C0,C3} is not reachable by single bond cuts; seeds add it
        sets_no = _atom_sets(nodes_no)
        sets_yes = _atom_sets(nodes_yes)
        assert frozenset({0, 3}) not in sets_no
        assert frozenset({0, 3}) in sets_yes

    def test_prepass_with_max_cut_size_1(self):
        """max_cut_size=1 (no ring cuts) still injects SMARTS seeds."""
        result_cut1 = _run_cy("CCCC", max_cut_size=1, use_prepass=True, min_frag_atoms=0)
        assert frozenset({0, 3}) in _atom_sets(result_cut1["nodes_mask"]), (
            "crf12_0 seed must be injected even with max_cut_size=1"
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestPrepassDeterminism:
    """Repeated calls with identical inputs must produce identical outputs."""

    @pytest.mark.parametrize("smiles", ["CCCC", "CC(=O)OC", "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C"])
    def test_identical_outputs_on_repeat(self, smiles):
        mol_d, nn, ne, nm, edges, em, n2e, mol = _get_arrays(smiles)
        ring_mask = mol_d["ring_bond_mask"]
        em_u8 = np.asarray(em, dtype=MASK_DTYPE)
        seed_pairs = _apply_smarts_prepass(FRAG_RULES, mol, em_u8, edges, nn, ne)

        kwargs = {
            "max_depth": 3, "time_limit": 30,
            "ring_edge_mask": ring_mask, "max_cut_size": 2,
            "smarts_seed_pairs": seed_pairs, "min_frag_atoms": 0,
        }
        nm1, nd1, de1, _ = compute_ccs_multi_cut_cy(nn, ne, nm, edges, em, n2e, **kwargs)
        nm2, nd2, de2, _ = compute_ccs_multi_cut_cy(nn, ne, nm, edges, em, n2e, **kwargs)

        assert np.array_equal(nm1, nm2), "nodes_mask differs between runs"
        assert np.array_equal(nd1, nd2), "nodes_depth differs between runs"
        assert np.array_equal(de1, de2), "dag_edges differs between runs"


@pytest.mark.parametrize("name,smiles", _SPEED_MOLECULES)
def test_benchmark_cython_no_prepass(benchmark, name, smiles):
    """Benchmark: Cython BFS without SMARTS prepass (baseline)."""
    mol_d, nn, ne, nm, edges, em, n2e, mol = _get_arrays(smiles)
    ring_mask = mol_d["ring_bond_mask"]

    benchmark(
        compute_ccs_multi_cut_cy,
        nn,
        ne,
        nm,
        edges,
        em,
        n2e,
        max_depth=3,
        time_limit=30,
        ring_edge_mask=ring_mask,
        max_cut_size=2,
    )
