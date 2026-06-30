"""Tests for the multi-bond-cut BFS fragmentation module.

Tests cover:
1. Acyclic molecule: multi-cut output matches standard BFS (no ring bonds → same result).
2. Cyclohexane: ring fragment appears at depth 1 with max_cut_size=2 but NOT with
   max_cut_size=1 (single-bond BFS can never disconnect a ring at depth 1).
3. Naphthalene: fused bicyclic — 2-bond cut separates the two rings.
4. Retro-Diels-Alder scaffold: cyclohexene-like 2-bond cut.
5. max_cut_size=1 falls back to standard single-bond BFS on a ring.
"""

from __future__ import annotations

import numpy as np

from fragnnet.frag.multi_cut_bfs import compute_ccs_multi_cut
from fragnnet.utils.frag_utils import extract_mol_info, get_fraggen_input_arrays

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(smiles: str, max_cut_size: int = 2, max_depth: int = 3, min_frag_atoms: int = 0) -> dict:
    """Run multi-cut BFS on a SMILES and return result components."""
    mol_d = extract_mol_info(smiles)
    ring_mask = mol_d["ring_bond_mask"]
    ring_bond_groups = [list(r) for r in mol_d["mol"].GetRingInfo().BondRings()]
    num_nodes, num_edges, node_mask, edges, edge_mask, n2e = get_fraggen_input_arrays(mol_d)
    nodes_mask, nodes_depth, dag_edges, meta = compute_ccs_multi_cut(
        num_nodes,
        num_edges,
        node_mask,
        edges,
        edge_mask,
        n2e,
        max_depth=max_depth,
        ring_edge_mask=ring_mask,
        max_cut_size=max_cut_size,
        ring_bond_groups=ring_bond_groups,
        min_frag_atoms=min_frag_atoms,
    )
    return {
        "nodes_mask": nodes_mask,
        "nodes_depth": nodes_depth,
        "dag_edges": dag_edges,
        "meta": meta,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
    }


def _atom_counts(nodes_mask: np.ndarray) -> list[int]:
    """Return sorted list of atom counts per DAG node (excl. root)."""
    counts = [int(nodes_mask[i].sum()) for i in range(1, len(nodes_mask))]
    return sorted(counts)


def _node_depths(nodes_mask: np.ndarray, nodes_depth: np.ndarray) -> dict[int, int]:
    """Return {atom_count: min_depth} for non-root nodes."""
    result = {}
    for i in range(1, len(nodes_mask)):
        count = int(nodes_mask[i].sum())
        min_d = int(np.where(nodes_depth[i])[0].min())
        result[count] = min(result.get(count, 999), min_d)
    return result


# ---------------------------------------------------------------------------
# Test 1: acyclic molecule — multi-cut and standard BFS agree
# ---------------------------------------------------------------------------


class TestAcyclicMolecule:
    """n-pentane: no ring bonds → max_cut_size has no effect."""

    SMILES = "CCCCC"  # n-pentane, 5C acyclic

    def test_same_node_count_as_single_cut(self):
        r1 = _run(self.SMILES, max_cut_size=1)
        r2 = _run(self.SMILES, max_cut_size=2)
        assert len(r1["nodes_mask"]) == len(r2["nodes_mask"])

    def test_same_dag_edge_count(self):
        r1 = _run(self.SMILES, max_cut_size=1)
        r2 = _run(self.SMILES, max_cut_size=2)
        assert len(r1["dag_edges"]) == len(r2["dag_edges"])

    def test_depth1_has_fragments(self):
        r = _run(self.SMILES, max_cut_size=2)
        depths = _node_depths(r["nodes_mask"], r["nodes_depth"])
        # Breaking any bond in n-pentane gives fragments at depth 1
        assert 1 in depths.values()

    def test_ring_bond_mask_all_zero(self):
        mol_d = extract_mol_info(self.SMILES)
        assert mol_d["ring_bond_mask"].sum() == 0


# ---------------------------------------------------------------------------
# Test 2: cyclohexane — ring fragment requires 2-bond cut
# ---------------------------------------------------------------------------


class TestCyclohexane:
    """Cyclohexane: every bond is a ring bond.

    Standard BFS (max_cut_size=1) can never disconnect it at any depth
    (single bond break → num_ccs=1 always).
    Multi-cut BFS (max_cut_size=2) finds the 2-bond cuts at depth 2
    (cut=2 consumes 2 depth units, so children land at depth 0+2=2).
    """

    SMILES = "C1CCCCC1"  # cyclohexane, 6C ring

    def test_standard_bfs_finds_no_fragments(self):
        """max_cut_size=1 on a simple ring finds no fragments (only root node)."""
        r = _run(self.SMILES, max_cut_size=1, max_depth=1)
        # Only the root node should exist since single-bond breaks on a ring
        # never disconnect
        assert len(r["nodes_mask"]) == 1, (
            f"Expected only root node, got {len(r['nodes_mask'])} nodes"
        )

    def test_multi_cut_finds_ring_fragments_at_depth2(self):
        """max_cut_size=2 finds ring-opening fragments at depth 2.

        Cut=2 consumes 2 depth units, so max_depth must be >= 2 for
        ring-pair cuts to produce any fragments.
        """
        r = _run(self.SMILES, max_cut_size=2, max_depth=2)
        # Should have more than just the root node
        assert len(r["nodes_mask"]) > 1, "Multi-cut BFS should find ring fragments"

    def test_multi_cut_blocked_when_max_depth_1(self):
        """With max_depth=1, cut=2 is blocked (would need depth 2 > max_depth=1)."""
        r = _run(self.SMILES, max_cut_size=2, max_depth=1)
        assert len(r["nodes_mask"]) == 1, (
            "Cut=2 should be blocked when max_depth=1 (needs 2 depth units)"
        )

    def test_ring_cut_fragment_sizes(self):
        """2-bond cuts of cyclohexane produce complementary fragment size pairs.

        Adjacent bond pair (e.g. bonds 0-1 and 1-2): 1+5 split
        Skip-1 bond pair (e.g. bonds 0-1 and 2-3):   2+4 split
        Opposite bond pair (e.g. bonds 0-1 and 3-4):  3+3 split

        Each cut produces 2 fragments summing to 6 atoms.
        All sizes 1..5 should appear in the non-root node set.
        Requires max_depth=2 so cut=2 from root (depth 0+2=2) can fire.
        """
        r = _run(self.SMILES, max_cut_size=2, max_depth=2)
        counts = _atom_counts(r["nodes_mask"])
        for expected_size in [1, 2, 3, 4, 5]:
            assert expected_size in counts, (
                f"Expected fragment of size {expected_size} from cyclohexane 2-bond cuts, "
                f"got sizes: {sorted(set(counts))}"
            )

    def test_ring_bond_mask_all_ring(self):
        mol_d = extract_mol_info(self.SMILES)
        assert mol_d["ring_bond_mask"].sum() == 6  # all 6 bonds are ring bonds

    def test_depth2_nodes_exist_in_multi_cut(self):
        """Cut=2 from root produces depth-2 nodes (depth consumed = cut size)."""
        r = _run(self.SMILES, max_cut_size=2, max_depth=2)
        depths = _node_depths(r["nodes_mask"], r["nodes_depth"])
        # Cyclohexane has no bridges → no depth-1 nodes; cut=2 fragments at depth 2
        assert 2 in depths.values(), "No depth-2 fragments found for cyclohexane"


# ---------------------------------------------------------------------------
# Test 3: naphthalene — fused bicyclic, 2-bond cut separates the two rings
# ---------------------------------------------------------------------------


class TestNaphthalene:
    """Naphthalene: two fused 6-membered aromatic rings.

    A 2-bond cut on the two bonds connecting the rings separates them into
    two 5-atom (C5H5) and 5-atom (C5H5) fragments — each being a cyclopentadienyl
    fragment at the formula level.
    """

    SMILES = "c1ccc2ccccc2c1"  # naphthalene, 10C

    def test_multi_cut_finds_more_nodes_than_single_cut(self):
        # max_depth=2 required so cut=2 (depth consumed=2) can fire from root
        r1 = _run(self.SMILES, max_cut_size=1, max_depth=2)
        r2 = _run(self.SMILES, max_cut_size=2, max_depth=2)
        # fused ring cuts are only possible with 2-bond cuts
        assert len(r2["nodes_mask"]) >= len(r1["nodes_mask"])

    def test_ring_bond_mask_count(self):
        mol_d = extract_mol_info(self.SMILES)
        # All 11 bonds of naphthalene are ring bonds
        assert mol_d["ring_bond_mask"].sum() == 11

    def test_achievable_fragment_sizes(self):
        """Verify the fragment sizes that are actually achievable with 2-bond cuts.

        Naphthalene min-edge-cut = 3 (bridgehead atoms have degree 3), so a
        5+5 split requires 3 bonds — NOT achievable with max_cut_size=2.

        Valid 2-bond cuts:
          1+9: cut both bonds around any of the 8 non-bridgehead (degree-2) atoms
          2+8: cut both external bonds of any adjacent pair of non-bridgehead atoms
          3+7: three consecutive non-bridgehead atoms (each ring has 2 such triples)
          4+6: four consecutive non-bridgehead atoms (one chain per ring: C4 subpath)
        """
        # max_depth=2 required so cut=2 (depth consumed=2) can fire from root
        r = _run(self.SMILES, max_cut_size=2, max_depth=2)
        counts = set(_atom_counts(r["nodes_mask"]))
        # All of these splits must be achievable
        for expected in [1, 2, 3, 4, 6, 7, 8, 9]:
            assert expected in counts, (
                f"Expected fragment of size {expected} from naphthalene 2-bond cuts, "
                f"got sizes: {sorted(counts)}"
            )
        # 5-atom split requires a 3-bond cut — must NOT appear with max_cut_size=2
        assert 5 not in counts, (
            "5-atom split of naphthalene should require 3-bond cut, "
            "but appeared with max_cut_size=2"
        )


# ---------------------------------------------------------------------------
# Test 4: retro-Diels-Alder scaffold — cyclohexene 2-bond cut
# ---------------------------------------------------------------------------


class TestCyclohexene:
    """Cyclohexene: the diene + dienophile 2-bond cut should appear at depth 2.

    Cut=2 consumes 2 depth units, so requires max_depth >= 2.
    """

    SMILES = "C1CC=CCC1"  # cyclohex-3-ene (simplified RDA scaffold)

    def test_multi_cut_produces_depth2_ring_fragments(self):
        r = _run(self.SMILES, max_cut_size=2, max_depth=2)
        assert len(r["nodes_mask"]) > 1, "Should find fragments from 2-bond ring cut"

    def test_ring_bonds_present(self):
        mol_d = extract_mol_info(self.SMILES)
        assert mol_d["ring_bond_mask"].sum() > 0


# ---------------------------------------------------------------------------
# Test 5: output format compatibility with compute_ccs
# ---------------------------------------------------------------------------


class TestOutputFormat:
    """Verify output arrays match expected shapes and dtypes."""

    SMILES = "C1CCCCC1"  # cyclohexane

    def test_nodes_mask_matrix_shape(self):
        r = _run(self.SMILES, max_cut_size=2)
        assert r["nodes_mask"].ndim == 2
        assert r["nodes_mask"].shape[1] == r["num_nodes"]
        assert r["nodes_mask"].dtype == np.uint8

    def test_nodes_depth_matrix_shape(self):
        r = _run(self.SMILES, max_cut_size=2, max_depth=3)
        assert r["nodes_depth"].ndim == 2
        assert r["nodes_depth"].shape[0] == r["nodes_mask"].shape[0]
        assert r["nodes_depth"].shape[1] == 4  # max_depth + 1

    def test_dag_edges_shape(self):
        r = _run(self.SMILES, max_cut_size=2)
        if len(r["dag_edges"]) > 0:
            assert r["dag_edges"].ndim == 2
            assert r["dag_edges"].shape[1] == 2
            assert r["dag_edges"].dtype == np.int64

    def test_meta_keys_present(self):
        r = _run(self.SMILES, max_cut_size=2)
        for key in ("reached_depth", "edges_min_depth", "nodes_min_depth", "force_stopped"):
            assert key in r["meta"], f"Missing key: {key}"

    def test_root_node_is_index_0(self):
        r = _run(self.SMILES, max_cut_size=2)
        # Root node should contain all atoms
        assert r["nodes_mask"][0].sum() == r["num_nodes"]

    def test_root_node_at_depth_0(self):
        r = _run(self.SMILES, max_cut_size=2)
        assert r["nodes_depth"][0, 0] == 1  # root at depth 0

    def test_no_all_zero_cc_rows(self):
        """MBFS output must not contain empty connected-component masks."""
        r = _run(
            "COc1ccc2nccc(NC(=O)C3CCN(CCc4ccc(Cl)cc4)CC3)c2n1",
            max_cut_size=2,
            max_depth=3,
        )
        sums = r["nodes_mask"].sum(axis=1)
        zero_rows = [i for i, s in enumerate(sums) if int(s) == 0]
        assert zero_rows == [], f"Found all-zero CC rows at indices {zero_rows[:10]}"


# ---------------------------------------------------------------------------
# Test 6: ring_bond_mask in mol_d
# ---------------------------------------------------------------------------


class TestRingBondMask:
    """Verify ring_bond_mask is correctly populated in mol_d."""

    def test_benzene_all_ring(self):
        mol_d = extract_mol_info("c1ccccc1")
        assert mol_d["ring_bond_mask"].sum() == 6

    def test_acyclic_all_zero(self):
        mol_d = extract_mol_info("CCCC")
        assert mol_d["ring_bond_mask"].sum() == 0

    def test_mixed_ring_acyclic(self):
        # methylcyclohexane: 6 ring bonds + 1 exocyclic C-C bond = 7 total, 6 ring
        mol_d = extract_mol_info("CC1CCCCC1")
        num_bonds = len(mol_d["bonds"])
        num_ring = int(mol_d["ring_bond_mask"].sum())
        assert num_bonds == 7
        assert num_ring == 6

    def test_mask_length_matches_bonds(self):
        for smiles in ["C1CCCCC1", "c1ccccc1", "CCCCC", "CC1CCCCC1"]:
            mol_d = extract_mol_info(smiles)
            assert len(mol_d["ring_bond_mask"]) == len(mol_d["bonds"])


# ---------------------------------------------------------------------------
# Test 7: depth bitmask — nodes_depth_matrix correctness after optimization
# ---------------------------------------------------------------------------


class TestDepthMatrix:
    """Verify nodes_depth_matrix values are exact (bitmask optimization must
    reproduce the same matrix as the original set-based approach)."""

    def test_root_only_at_depth_0(self):
        """Root node appears at depth 0 only, not at deeper depths."""
        r = _run("C1CCCCC1", max_cut_size=2, max_depth=2)
        # root = index 0; depth column 0 must be 1, columns 1/2 must be 0
        assert r["nodes_depth"][0, 0] == 1
        assert r["nodes_depth"][0, 1] == 0
        assert r["nodes_depth"][0, 2] == 0

    def test_depth1_fragments_not_at_depth0(self):
        """Depth-1 fragments (ring cuts) should NOT have depth-0 set."""
        r = _run("C1CCCCC1", max_cut_size=2, max_depth=2)
        for i in range(1, len(r["nodes_mask"])):
            if r["nodes_depth"][i, 1] == 1:
                # This node is a depth-1 child; it must not have depth 0 set
                assert r["nodes_depth"][i, 0] == 0, (
                    f"Node {i} has depth-0 set but was only found at depth 1"
                )

    def test_multi_ring_system_depth_correct(self):
        """Steroid scaffold (multi-ring system): depth-1 fragments only at depth 1."""
        # androstane skeleton — 4 fused rings, many ring bonds in different systems
        r = _run("C1CCC2CCCC3CCCC4CCCCC4C3C2C1", max_cut_size=2, max_depth=2)
        for i in range(1, len(r["nodes_mask"])):
            depths_set = set(int(d) for d in range(3) if r["nodes_depth"][i, d] == 1)
            # If found at depth 1 only, depth 0 and depth 2 must be 0
            if depths_set == {1}:
                assert r["nodes_depth"][i, 0] == 0
                assert r["nodes_depth"][i, 2] == 0

    def test_nodes_min_depth_consistent_with_depth_matrix(self):
        """nodes_min_depth must match the minimum set column in nodes_depth_matrix."""
        r = _run("c1ccc2ccccc2c1", max_cut_size=2, max_depth=3)
        min_depths = r["meta"]["nodes_min_depth"]
        for i in range(len(r["nodes_mask"])):
            cols_set = [int(d) for d in range(4) if r["nodes_depth"][i, d] == 1]
            assert len(cols_set) > 0, f"Node {i} has no depth set in matrix"
            assert int(min_depths[i]) == min(cols_set), (
                f"Node {i}: nodes_min_depth={min_depths[i]} but matrix min={min(cols_set)}"
            )


# ---------------------------------------------------------------------------
# Test 8: pre-grouped ring-system cut=2 — multi-ring-system molecules
# ---------------------------------------------------------------------------


class TestMultiRingSystemCuts:
    """Molecules with multiple distinct ring systems exercise the ring-grouping
    optimization: cross-system pairs must never produce a valid 2-bond cut."""

    SPIRO = "C1CCC2(CC1)CCCC2"  # spiro[5.4]decane — two rings sharing ONE atom

    def test_spiro_multi_ring_system_fragments(self):
        """Spiro compound: 2-bond cuts within each ring produce fragments at depth 2."""
        r = _run(self.SPIRO, max_cut_size=2, max_depth=2)
        assert len(r["nodes_mask"]) > 1, "Should find fragments in spiro compound"

    def test_spiro_no_3plus_cc_fragments(self):
        """2-bond cuts can never produce 3+ connected components (would be pruned)."""
        r = _run(self.SPIRO, max_cut_size=2, max_depth=2)
        # All non-root fragments sum with their pair to the full atom count
        total_atoms = r["num_nodes"]
        counts = sorted(_atom_counts(r["nodes_mask"]))
        for c in counts:
            # Each fragment must have a complementary fragment (partner sums to total)
            complement = total_atoms - c
            assert complement in counts or complement == total_atoms, (
                f"Fragment of size {c} has no complement in {counts}"
            )

    def test_bicyclo_cuts_within_system(self):
        """Bicyclo[2.2.1]heptane (norbornane): both ring bonds in same system."""
        # norbornane — bridged bicyclic; all ring bonds in one system
        r = _run("C1CC2CCC1C2", max_cut_size=2, max_depth=2)
        assert len(r["nodes_mask"]) > 1

    def test_two_disconnected_rings_no_cross_cut(self):
        """Biphenyl: two phenyl rings connected by one bond.

        The single bond between rings disconnects in cut=1.
        The two ring systems are distinct; cross-ring-system cut=2 pairs
        cannot disconnect the molecule and must not appear as valid cuts.
        """
        r1 = _run("c1ccccc1-c1ccccc1", max_cut_size=1, max_depth=1)
        r2 = _run("c1ccccc1-c1ccccc1", max_cut_size=2, max_depth=2)
        # cut=2 must find AT LEAST as many nodes as cut=1
        assert len(r2["nodes_mask"]) >= len(r1["nodes_mask"])
        # The single inter-ring bond cut (6+6 split) should appear in both
        counts2 = set(_atom_counts(r2["nodes_mask"]))
        assert 6 in counts2, "Biphenyl 6+6 split must appear"


# ---------------------------------------------------------------------------
# Test 9: selective scratch reset — correctness on back-to-back cuts
# ---------------------------------------------------------------------------


class TestSelectiveScratchReset:
    """These tests exercise the selective scratch reset by running many cuts
    in sequence on the same BFS node and checking correctness of each result."""

    def test_many_cuts_cyclohexane_consistent(self):
        """Cyclohexane has C(6,2)=15 ring pairs; all must produce valid results."""
        r = _run("C1CCCCC1", max_cut_size=2, max_depth=2)
        # Cyclohexane 2-bond cuts produce fragments of sizes 1..5
        counts = sorted(set(_atom_counts(r["nodes_mask"])))
        for expected in [1, 2, 3, 4, 5]:
            assert expected in counts

    def test_large_ring_cuts(self):
        """Cyclooctane (8-membered ring): C(8,2)=28 ring pairs."""
        r = _run("C1CCCCCCC1", max_cut_size=2, max_depth=2)
        counts = set(_atom_counts(r["nodes_mask"]))
        # Valid splits: 1+7, 2+6, 3+5, 4+4
        for expected in [1, 2, 3, 4, 5, 6, 7]:
            assert expected in counts, f"Expected {expected}-atom fragment from cyclooctane"

    def test_no_phantom_atoms_in_fragment(self):
        """No fragment should have atom count 0 or equal to the full molecule (except root)."""
        r = _run("C1CCCCC1", max_cut_size=2, max_depth=2)
        total = r["num_nodes"]
        for i in range(1, len(r["nodes_mask"])):
            count = int(r["nodes_mask"][i].sum())
            assert 0 < count < total, (
                f"Non-root node {i} has invalid atom count {count} (total={total})"
            )


# ---------------------------------------------------------------------------
# Test 10: bridge-finding correctness (cut=1 isolated)
# ---------------------------------------------------------------------------


class TestBridgeFindingCutOne:
    """Verify the Tarjan bridge-finding path against known graph structures.

    All tests use max_cut_size=1 (ring_edge_mask still passed so it's active)
    and max_depth=1 so only depth-1 fragments from cut=1 are produced.
    """

    def _run1(self, smiles: str) -> dict:
        return _run(smiles, max_cut_size=1, max_depth=1)

    def test_ethane_single_bridge(self):
        """Ethane (2 atoms, 1 bond): the bond is a bridge → 2 single-atom children."""
        r = self._run1("CC")
        counts = sorted(_atom_counts(r["nodes_mask"]))
        assert counts == [1, 1]
        assert len(r["nodes_mask"]) == 3  # root + 2 children

    def test_isobutane_three_bridges(self):
        """Isobutane: tree graph — all 3 C-C bonds are bridges.

        Cuts: C0-C1, C1-C2, C1-C3 (star from central C).
        Each gives size-1 + size-3 split.  6 unique fragments (3 × 2 by atom index).
        """
        r = self._run1("CC(C)C")
        counts = sorted(set(_atom_counts(r["nodes_mask"])))
        assert 1 in counts
        assert 3 in counts
        assert len(r["nodes_mask"]) - 1 == 6  # 3 bridges × 2 children

    def test_neopentane_four_bridges(self):
        """Neopentane: 4 bonds (all bridges) → 8 unique child fragments (4 × 2).

        Each bridge cut gives 1-atom (CH3) + 4-atom (rest) fragments.
        """
        r = self._run1("CC(C)(C)C")
        counts = sorted(set(_atom_counts(r["nodes_mask"])))
        assert 1 in counts
        assert 4 in counts
        assert len(r["nodes_mask"]) - 1 == 8  # 4 bridges × 2 children

    def test_methylcyclohexane_one_bridge(self):
        """Methylcyclohexane: exactly 1 bridge (exocyclic CH3 bond) → 2 fragments.

        The ring bonds are not bridges; only the CH3-ring bond disconnects.
        """
        r = self._run1("CC1CCCCC1")
        counts = sorted(set(_atom_counts(r["nodes_mask"])))
        assert 1 in counts  # isolated CH3
        assert 6 in counts  # cyclohexyl ring remains
        # Exactly 2 non-root fragments from the 1 bridge
        assert len(r["nodes_mask"]) - 1 == 2

    def test_biphenyl_one_inter_ring_bridge(self):
        """Biphenyl: 1 inter-ring bond (bridge) → 6+6 split; ring bonds are not bridges."""
        r = self._run1("c1ccccc1-c1ccccc1")
        counts = sorted(set(_atom_counts(r["nodes_mask"])))
        assert counts == [6]  # both fragments are 6-atom phenyl rings
        assert len(r["nodes_mask"]) - 1 == 2  # 1 bridge × 2 children

    def test_simple_ring_has_no_bridges(self):
        """No bond in a simple ring disconnects the molecule — zero fragments."""
        for smiles in ["C1CCCCC1", "c1ccccc1", "C1CCCCCCCC1"]:
            r = self._run1(smiles)
            assert len(r["nodes_mask"]) == 1, (
                f"Ring {smiles} should have no bridges, got {len(r['nodes_mask']) - 1} fragment(s)"
            )

    def test_linear_octane_all_bridges(self):
        """Linear C8: all 7 bonds are bridges — cut=1 depth=1 finds all sizes 1..7."""
        r = self._run1("CCCCCCCC")
        counts = set(_atom_counts(r["nodes_mask"]))
        for expected in range(1, 8):
            assert expected in counts, f"Expected size-{expected} fragment from C8 linear chain"


# ---------------------------------------------------------------------------
# Test 11: no duplicate fragments in output DAG
# ---------------------------------------------------------------------------


class TestNoDuplicateFragments:
    """The queued_set dedup must ensure each (node_key, edge_key) is enqueued once.

    nodes_mask_matrix must never contain two identical rows.
    """

    def _check_no_duplicates(self, smiles: str, max_cut_size: int = 2, max_depth: int = 2):
        r = _run(smiles, max_cut_size=max_cut_size, max_depth=max_depth)
        nm = r["nodes_mask"]
        rows = [bytes(nm[i]) for i in range(len(nm))]
        assert len(rows) == len(set(rows)), (
            f"Duplicate fragment masks in {smiles}: {len(rows)} total rows, {len(set(rows))} unique"
        )

    def test_naphthalene_no_duplicates(self):
        self._check_no_duplicates("c1ccc2ccccc2c1")

    def test_androstane_no_duplicates(self):
        """4-ring steroid: many overlapping cut paths can regenerate the same fragment."""
        self._check_no_duplicates("C1CCC2CCCC3CCCC4CCCCC4C3C2C1", max_depth=1)

    def test_ibuprofen_no_duplicates(self):
        """Ring + acyclic chain: bridge cuts and ring cuts may share sub-fragments."""
        self._check_no_duplicates("CC(C)Cc1ccc(cc1)C(C)C(=O)O")

    def test_adamantane_no_duplicates(self):
        self._check_no_duplicates("C1C2CC3CC1CC(C2)C3", max_depth=2)

    def test_linear_chain_no_duplicates(self):
        """Long linear chain — deep BFS generates many overlapping sub-chains."""
        self._check_no_duplicates("CCCCCCCC", max_depth=3)


# ---------------------------------------------------------------------------
# Test 12: DAG structural validity
# ---------------------------------------------------------------------------


class TestDAGStructure:
    """Verify structural properties of the output DAG are always satisfied."""

    def _check_dag(self, smiles: str, max_depth: int = 2):
        r = _run(smiles, max_cut_size=2, max_depth=max_depth)
        n_nodes = len(r["nodes_mask"])
        total_atoms = r["num_nodes"]

        # Root covers all atoms
        assert int(r["nodes_mask"][0].sum()) == total_atoms

        # All non-root nodes have valid (non-zero, non-full) atom counts
        for i in range(1, n_nodes):
            c = int(r["nodes_mask"][i].sum())
            assert 0 < c < total_atoms, f"Node {i} invalid count {c}"

        # DAG edges reference valid indices, no self-loops
        if len(r["dag_edges"]) > 0:
            assert int(r["dag_edges"].max()) < n_nodes
            assert int(r["dag_edges"].min()) >= 0
            for src, dst in r["dag_edges"]:
                assert int(src) != int(dst), f"Self-loop at node {src}"

        return r

    def test_ethane_dag(self):
        r = self._check_dag("CC", max_depth=1)
        # 1 bridge → 2 children → 2 DAG edges
        assert len(r["dag_edges"]) == 2

    def test_cyclohexane_dag(self):
        self._check_dag("C1CCCCC1")

    def test_naphthalene_dag(self):
        self._check_dag("c1ccc2ccccc2c1")

    def test_ibuprofen_dag(self):
        self._check_dag("CC(C)Cc1ccc(cc1)C(C)C(=O)O")

    def test_all_fragment_masks_are_submasks_of_root(self):
        """Every fragment's atoms must be a subset of the root's atoms."""
        r = _run("CC(C)Cc1ccc(cc1)C(C)C(=O)O", max_cut_size=2, max_depth=2)
        root_mask = r["nodes_mask"][0]
        for i in range(1, len(r["nodes_mask"])):
            frag = r["nodes_mask"][i]
            # frag must not activate any atom that root has off
            assert int(((frag > 0) & (root_mask == 0)).sum()) == 0, (
                f"Fragment {i} activates atoms outside the root molecule"
            )


# ---------------------------------------------------------------------------
# Test 13: polycyclic cage molecules (adamantane, norbornane)
# ---------------------------------------------------------------------------


class TestPolycyclicCages:
    """Bridged cage molecules exercise the ring-system cut logic.

    Adamantane (C10, 12 bonds) is 3-edge-connected:
      - cut=1 finds 0 bridges.
      - cut=2 finds only 1+9 splits (isolating one CH2 group at a time).
      - cut=3 finds many more fragments.

    Norbornane (C7, 8 bonds, bicyclo[2.2.1]heptane) is 2-edge-connected:
      - cut=1 finds 0 bridges.
      - cut=2 finds fragments of sizes {1, 2, 5, 6}.
    """

    ADAMANTANE = "C1C2CC3CC1CC(C2)C3"
    NORBORNANE = "C1CC2CCC1C2"

    def test_adamantane_no_cut1_bridges(self):
        r = _run(self.ADAMANTANE, max_cut_size=1, max_depth=1)
        assert len(r["nodes_mask"]) == 1, (
            f"Adamantane should have no bridges, got {len(r['nodes_mask']) - 1} fragment(s)"
        )

    def test_adamantane_cut2_sizes(self):
        """Cut=2 on adamantane only isolates single CH2 groups → size 1 and 9."""
        r = _run(self.ADAMANTANE, max_cut_size=2, max_depth=2)
        counts = set(_atom_counts(r["nodes_mask"]))
        assert counts == {1, 9}, f"Expected only 1+9 splits, got {sorted(counts)}"
        # 6 CH2 groups × 2 children = 12 unique non-root fragments
        assert len(r["nodes_mask"]) - 1 == 12

    def test_adamantane_cut3_finds_more(self):
        """Cut=3 on adamantane with SSSR grouping: within-ring triples cannot
        disconnect the cage (bridge bonds outside the ring keep it connected),
        so cut=3 produces the same fragments as cut=2 for adamantane."""
        r2 = _run(self.ADAMANTANE, max_cut_size=2, max_depth=3)
        r3 = _run(self.ADAMANTANE, max_cut_size=3, max_depth=3)
        assert len(r3["nodes_mask"]) == len(r2["nodes_mask"])

    def test_norbornane_no_cut1_bridges(self):
        r = _run(self.NORBORNANE, max_cut_size=1, max_depth=1)
        assert len(r["nodes_mask"]) == 1

    def test_norbornane_cut2_sizes(self):
        r = _run(self.NORBORNANE, max_cut_size=2, max_depth=2)
        counts = set(_atom_counts(r["nodes_mask"]))
        for expected in [1, 2, 5, 6]:
            assert expected in counts, (
                f"Expected size-{expected} fragment from norbornane, got {sorted(counts)}"
            )

    def test_norbornane_no_duplicates(self):
        r = _run(self.NORBORNANE, max_cut_size=3, max_depth=1)
        nm = r["nodes_mask"]
        rows = [bytes(nm[i]) for i in range(len(nm))]
        assert len(rows) == len(set(rows))


# ---------------------------------------------------------------------------
# Test 14: macrocycle (16-membered ring)
# ---------------------------------------------------------------------------


class TestMacrocycle16:
    """16-membered ring: no bridges; cut=2 opens the ring at all positions.

    Expected: sizes 1..15 all present (C(16,2) = 120 distinct bond pairs,
    but many give the same fragment sizes; all 15 complementary sizes appear).
    """

    SMILES = "C1CCCCCCCCCCCCCCC1"

    def test_no_bridges(self):
        r = _run(self.SMILES, max_cut_size=1, max_depth=1)
        assert len(r["nodes_mask"]) == 1, "16-membered ring should have no bridges"

    def test_all_split_sizes_present(self):
        r = _run(self.SMILES, max_cut_size=2, max_depth=2)
        counts = set(_atom_counts(r["nodes_mask"]))
        for expected in range(1, 16):
            assert expected in counts, f"Expected size-{expected} fragment from 16-membered ring"

    def test_fragment_pairs_sum_to_16(self):
        """Each non-root fragment must have a complement summing to 16."""
        r = _run(self.SMILES, max_cut_size=2, max_depth=2)
        counts = sorted(_atom_counts(r["nodes_mask"]))
        count_set = set(counts)
        for c in count_set:
            complement = 16 - c
            assert complement in count_set, (
                f"Fragment size {c} has no complement {complement} in {sorted(count_set)}"
            )

    def test_no_duplicates(self):
        r = _run(self.SMILES, max_cut_size=2, max_depth=2)
        nm = r["nodes_mask"]
        rows = [bytes(nm[i]) for i in range(len(nm))]
        assert len(rows) == len(set(rows))


# ---------------------------------------------------------------------------
# Test 15: mixed ring + acyclic molecule (ibuprofen)
# ---------------------------------------------------------------------------


class TestIbuprofen:
    """Ibuprofen (CC(C)Cc1ccc(cc1)C(C)C(=O)O): 15 atoms, 6 ring bonds, 9 acyclic.

    Cut=1 finds the 9 acyclic bridges (verified from probe: 18 unique non-root
    fragments, sizes {1, 3, 4, 5, 10, 11}).
    Cut=2 adds ring cuts from the benzene ring: 48 unique non-root fragments.
    """

    SMILES = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"

    def test_cut1_fragment_sizes(self):
        """All acyclic bridges produce known fragment sizes."""
        r = _run(self.SMILES, max_cut_size=1, max_depth=1)
        counts = set(_atom_counts(r["nodes_mask"]))
        for expected in [1, 3, 4, 5, 10, 11]:
            assert expected in counts, (
                f"Expected size-{expected} from ibuprofen acyclic bridges, got {sorted(counts)}"
            )

    def test_cut1_fragment_count(self):
        """18 unique non-root fragments from 9 acyclic bridges × 2 (minus symmetry)."""
        r = _run(self.SMILES, max_cut_size=1, max_depth=1)
        assert len(r["nodes_mask"]) - 1 == 18

    def test_cut2_adds_ring_fragments(self):
        """Cut=2 must produce more fragments than cut=1 (adds benzene ring cuts)."""
        r1 = _run(self.SMILES, max_cut_size=1, max_depth=2)
        r2 = _run(self.SMILES, max_cut_size=2, max_depth=2)
        assert len(r2["nodes_mask"]) > len(r1["nodes_mask"])

    def test_depth2_increases_fragments(self):
        r1 = _run(self.SMILES, max_cut_size=2, max_depth=1)
        r2 = _run(self.SMILES, max_cut_size=2, max_depth=2)
        assert len(r2["nodes_mask"]) >= len(r1["nodes_mask"])

    def test_all_fragments_are_subsets_of_molecule(self):
        r = _run(self.SMILES, max_cut_size=2, max_depth=2)
        root_mask = r["nodes_mask"][0]
        for i in range(1, len(r["nodes_mask"])):
            frag = r["nodes_mask"][i]
            assert int(((frag > 0) & (root_mask == 0)).sum()) == 0, (
                f"Fragment {i} has atoms outside the full molecule"
            )

    def test_no_duplicates_depth2(self):
        r = _run(self.SMILES, max_cut_size=2, max_depth=2)
        nm = r["nodes_mask"]
        rows = [bytes(nm[i]) for i in range(len(nm))]
        assert len(rows) == len(set(rows))


# ---------------------------------------------------------------------------
# Test 16: depth-bitmask consistency across diverse molecules
# ---------------------------------------------------------------------------


class TestDepthBitsConsistencyMultiMol:
    """nodes_min_depth must always match the minimum set column of nodes_depth_matrix
    across a variety of molecule types and BFS depths."""

    MOLECULES = [
        ("ethane", "CC"),
        ("isobutane", "CC(C)C"),
        ("methylcychex", "CC1CCCCC1"),
        ("biphenyl", "c1ccccc1-c1ccccc1"),
        ("naphthalene", "c1ccc2ccccc2c1"),
        ("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
        ("norbornane", "C1CC2CCC1C2"),
    ]

    def test_min_depth_consistent_with_matrix(self):
        for name, smi in self.MOLECULES:
            r = _run(smi, max_cut_size=2, max_depth=3)
            min_depths = r["meta"]["nodes_min_depth"]
            nd = r["nodes_depth"]
            for i in range(len(r["nodes_mask"])):
                cols_set = [int(d) for d in range(4) if nd[i, d] == 1]
                assert len(cols_set) > 0, f"{name}: node {i} has no depth set"
                assert int(min_depths[i]) == min(cols_set), (
                    f"{name}: node {i} min_depth={min_depths[i]} but matrix min={min(cols_set)}"
                )

    def test_root_only_at_depth0_multi_mol(self):
        for name, smi in self.MOLECULES:
            r = _run(smi, max_cut_size=2, max_depth=2)
            assert r["nodes_depth"][0, 0] == 1, f"{name}: root not at depth 0"
            assert r["nodes_depth"][0, 1] == 0, f"{name}: root wrongly at depth 1"
            assert r["nodes_depth"][0, 2] == 0, f"{name}: root wrongly at depth 2"


# ---------------------------------------------------------------------------
# Test 17: cut=3 triple-bond cuts
# ---------------------------------------------------------------------------


class TestCut3TripleBonds:
    """max_cut_size=3 unlocks triple simultaneous ring-bond cuts at depth 0."""

    def test_adamantane_cut3_unlocks_new_sizes(self):
        """Adamantane with SSSR grouping: within-ring triple cuts cannot
        disconnect the cage because bridge bonds outside the SSSR ring keep
        it connected. cut=3 produces the same sizes as cut=2 ({1, 9}).
        Cross-ring triples (old ring-system behaviour) are no longer tried."""
        r2 = _run("C1C2CC3CC1CC(C2)C3", max_cut_size=2, max_depth=3)
        r3 = _run("C1C2CC3CC1CC(C2)C3", max_cut_size=3, max_depth=3)
        counts2 = set(_atom_counts(r2["nodes_mask"]))
        counts3 = set(_atom_counts(r3["nodes_mask"]))
        assert counts3 == counts2, (
            f"Expected same sizes for cut=2 and cut=3 on adamantane, "
            f"got cut2={sorted(counts2)} cut3={sorted(counts3)}"
        )

    def test_cut3_no_duplicates(self):
        r = _run("C1C2CC3CC1CC(C2)C3", max_cut_size=3, max_depth=1)
        nm = r["nodes_mask"]
        rows = [bytes(nm[i]) for i in range(len(nm))]
        assert len(rows) == len(set(rows))

    def test_cut3_all_fragments_valid_size(self):
        r = _run("C1C2CC3CC1CC(C2)C3", max_cut_size=3, max_depth=1)
        total = r["num_nodes"]
        for i in range(1, len(r["nodes_mask"])):
            c = int(r["nodes_mask"][i].sum())
            assert 0 < c < total

    def test_norbornane_cut3_more_than_cut2(self):
        r2 = _run("C1CC2CCC1C2", max_cut_size=2, max_depth=1)
        r3 = _run("C1CC2CCC1C2", max_cut_size=3, max_depth=1)
        assert len(r3["nodes_mask"]) >= len(r2["nodes_mask"])


# ---------------------------------------------------------------------------
# Test 18: min_frag_atoms filter
# ---------------------------------------------------------------------------


class TestMinFragAtoms:
    """min_frag_atoms prunes child fragments smaller than the threshold.

    min_frag_atoms=0 disables the filter (default in _run helper).
    min_frag_atoms=3 drops 1- and 2-atom shards.
    """

    def test_filter_removes_small_fragments(self):
        """n-pentane bridges produce size-1 and size-2 fragments.

        With min_frag_atoms=3, those must not appear.
        """
        r0 = _run("CCCCC", max_cut_size=1, max_depth=1, min_frag_atoms=0)
        r3 = _run("CCCCC", max_cut_size=1, max_depth=1, min_frag_atoms=3)
        counts0 = set(_atom_counts(r0["nodes_mask"]))
        counts3 = set(_atom_counts(r3["nodes_mask"]))
        # Without filter: sizes 1,2,3,4 all present
        assert 1 in counts0 and 2 in counts0
        # With filter: sizes 1 and 2 must be absent
        assert 1 not in counts3 and 2 not in counts3
        # Large fragments (3+) still present
        assert 3 in counts3 and 4 in counts3

    def test_filter_fewer_nodes_than_no_filter(self):
        """Filtering must produce <= node count compared to no filter."""
        r0 = _run("CCCCCC", max_cut_size=1, max_depth=2, min_frag_atoms=0)
        r3 = _run("CCCCCC", max_cut_size=1, max_depth=2, min_frag_atoms=3)
        assert len(r3["nodes_mask"]) <= len(r0["nodes_mask"])

    def test_filter_zero_is_identity(self):
        """min_frag_atoms=0 must produce identical node count as no filter."""
        r0 = _run("CC(C)Cc1ccc(cc1)C(C)C(=O)O", max_cut_size=2, max_depth=2, min_frag_atoms=0)
        r1 = _run("CC(C)Cc1ccc(cc1)C(C)C(=O)O", max_cut_size=2, max_depth=2, min_frag_atoms=0)
        assert len(r0["nodes_mask"]) == len(r1["nodes_mask"])

    def test_filter3_all_fragments_at_least_3_atoms(self):
        """With min_frag_atoms=3, every non-root fragment has >= 3 atoms."""
        for smiles in ["CCCCC", "CC(C)C", "c1ccccc1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"]:
            r = _run(smiles, max_cut_size=2, max_depth=2, min_frag_atoms=3)
            for i in range(1, len(r["nodes_mask"])):
                count = int(r["nodes_mask"][i].sum())
                assert count >= 3, (
                    f"{smiles}: non-root node {i} has {count} atoms (< min_frag_atoms=3)"
                )

    def test_filter_ring_cut_sizes(self):
        """Cyclohexane cut=2: sizes 1 and 2 pruned, 3+ kept with min_frag_atoms=3."""
        r = _run("C1CCCCC1", max_cut_size=2, max_depth=2, min_frag_atoms=3)
        counts = set(_atom_counts(r["nodes_mask"]))
        assert 1 not in counts and 2 not in counts
        # 3-, 4-, 5-atom splits should still appear
        for expected in [3, 4, 5]:
            assert expected in counts, (
                f"Expected {expected}-atom fragment to survive min_frag_atoms=3 filter"
            )
