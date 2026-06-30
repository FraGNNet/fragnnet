"""Tests for FRAG_RULE prepass behavior and depth semantics.

Verifies that:
1. _apply_smarts_prepass returns (mask_a, mask_b, rule_idx) triples.
2. The fragment DAG no longer stores a prepass rule-id node feature.
3. Structural constraints: prepass only adds fragments, never removes them.
4. Prepass-injected fragments have node depth 2 (not 1): prepass rules require
   >=2 bond cuts + 1 new bond, so depth 2 correctly encodes the two-cut
   distance from the root. See docs/frag_gen_prepass.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from fragnnet.frag.smarts_prepass import FRAG_RULES, _apply_smarts_prepass
from fragnnet.utils.frag_utils import (
    NODE_FEAT_TO_IDX,
    compute_dags,
    extract_mol_info,
    get_fraggen_input_arrays,
    get_node_feats,
)

NUM_RULES = len(FRAG_RULES)


def _run(
    smiles: str,
    smarts_prepass: bool,
    multi_cut_bfs: bool = False,
    max_depth: int = 3,
) -> dict:
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
        multi_cut_bfs=multi_cut_bfs,
        smarts_prepass=smarts_prepass,
    )


def _get_depth_onehot(dag_d: dict) -> np.ndarray:
    """Extract the depth one-hot matrix from DAG node features."""

    dag = dag_d["dag"]
    x = dag.x.long()
    node_feat_idxs = dag.node_feat_idxs.long()
    feat = get_node_feats(x, node_feat_idxs[0], "depth")
    return feat.numpy()


def _get_cc_masks(dag_d: dict) -> np.ndarray:
    """Extract the cc mask matrix from DAG node features."""

    dag = dag_d["dag"]
    x = dag.x.long()
    node_feat_idxs = dag.node_feat_idxs.long()
    feat = get_node_feats(x, node_feat_idxs[0], "cc")
    return feat.numpy()


def _get_prepass_rows(dag_on: dict, dag_off: dict) -> np.ndarray:
    """Return node indices reached by root edges added only with prepass."""

    on_edges = set(map(tuple, dag_on["dag"].edge_index.t().tolist()))
    off_edges = set(map(tuple, dag_off["dag"].edge_index.t().tolist()))
    prepass_rows = sorted(dst for src, dst in on_edges - off_edges if src == 0)
    return np.asarray(prepass_rows, dtype=np.int64)


# ---------------------------------------------------------------------------
# 1. _apply_smarts_prepass returns (mask_a, mask_b, rule_idx) triples
# ---------------------------------------------------------------------------


class TestApplySmartsPrepassReturnType:
    """Verify _apply_smarts_prepass returns tuples of length 3 with valid rule_idx."""

    def test_return_tuple_length(self):
        """Each result tuple must have exactly 3 elements."""
        mol_d = extract_mol_info("CCCC")  # butane — matches crf12_0
        num_nodes, num_edges, _, edges, edge_mask, _ = get_fraggen_input_arrays(mol_d)
        results = _apply_smarts_prepass(
            FRAG_RULES, mol_d["mol"], edge_mask, edges, num_nodes, num_edges
        )
        assert len(results) > 0, "butane should match crf12_0"
        for item in results:
            assert len(item) == 3, f"expected 3-tuple, got {len(item)}-tuple"

    def test_rule_idx_is_int_in_range(self):
        """rule_idx must be a valid 0-based index into FRAG_RULES."""
        mol_d = extract_mol_info("CC(=O)OC")  # methyl acetate — matches crf13_2
        num_nodes, num_edges, _, edges, edge_mask, _ = get_fraggen_input_arrays(mol_d)
        results = _apply_smarts_prepass(
            FRAG_RULES, mol_d["mol"], edge_mask, edges, num_nodes, num_edges
        )
        assert len(results) > 0
        for _mask_a, _mask_b, rule_idx in results:
            assert isinstance(rule_idx, int)
            assert 0 <= rule_idx < NUM_RULES

    def test_crf13_2_rule_idx(self):
        """Methyl acetate must produce results with rule_idx == RULE_IDX['crf13_2']."""
        mol_d = extract_mol_info("CC(=O)OC")
        num_nodes, num_edges, _, edges, edge_mask, _ = get_fraggen_input_arrays(mol_d)
        results = _apply_smarts_prepass(
            FRAG_RULES, mol_d["mol"], edge_mask, edges, num_nodes, num_edges
        )
        rule_idxs = {r for _, _, r in results}
        assert RULE_IDX["crf13_2"] in rule_idxs

    def test_no_match_returns_empty(self):
        """A molecule that matches no rule returns an empty list."""
        mol_d = extract_mol_info("c1ccccc1")  # benzene
        num_nodes, num_edges, _, edges, edge_mask, _ = get_fraggen_input_arrays(mol_d)
        results = _apply_smarts_prepass(
            FRAG_RULES, mol_d["mol"], edge_mask, edges, num_nodes, num_edges
        )
        assert results == []


RULE_IDX = {rule.name: i for i, rule in enumerate(FRAG_RULES)}


# ---------------------------------------------------------------------------
# 2. prepass rule-id feature removed from DAG node features
# ---------------------------------------------------------------------------


class TestRuleIdFeatureRemoved:
    """The fragment DAG must not expose a prepass rule-id node feature."""

    def test_rule_id_feature_removed(self):
        assert "prepass_rule_id" not in NODE_FEAT_TO_IDX


# ---------------------------------------------------------------------------
# 3. Structural constraints
# ---------------------------------------------------------------------------


class TestStructuralConstraints:
    """Prepass only adds fragments/edges, never removes them."""

    @pytest.mark.parametrize("smiles", ["c1ccccc1", "CCO"])
    def test_no_match_node_count_unchanged(self, smiles):
        d_off = _run(smiles, smarts_prepass=False)
        d_on = _run(smiles, smarts_prepass=True)
        assert d_on["dag"].num_nodes == d_off["dag"].num_nodes

    @pytest.mark.parametrize("smiles", ["c1ccccc1", "CCO"])
    def test_no_match_edge_count_unchanged(self, smiles):
        d_off = _run(smiles, smarts_prepass=False)
        d_on = _run(smiles, smarts_prepass=True)
        assert d_on["dag"].num_edges == d_off["dag"].num_edges

    @pytest.mark.parametrize("smiles", ["CCC(C)CC", "CC(=O)OC", "CCCCC"])
    def test_prepass_never_removes_fragments(self, smiles):
        d_off = _run(smiles, smarts_prepass=False)
        d_on = _run(smiles, smarts_prepass=True)
        assert d_on["dag"].num_nodes >= d_off["dag"].num_nodes


# ---------------------------------------------------------------------------
# 4. Prepass fragment depth is 2 (not 1)
# ---------------------------------------------------------------------------


class TestPrepassFragmentDepth:
    """Prepass-injected fragments must carry node depth 2, not 1."""

    def test_methyl_acetate_prepass_frags_at_depth2(self):
        """All fragments added by prepass must have depth-2 one-hot set."""
        dag_on = _run("CC(=O)OC", smarts_prepass=True, max_depth=4)
        dag_off = _run("CC(=O)OC", smarts_prepass=False, max_depth=4)
        depth = _get_depth_onehot(dag_on)
        prepass_rows = _get_prepass_rows(dag_on, dag_off)
        assert len(prepass_rows) > 0, "expected at least one prepass fragment"
        for row in prepass_rows:
            assert depth[row, 2] == 1, (
                f"fragment {row} tagged as prepass but depth one-hot is {depth[row]} "
                f"(expected depth index 2 to be 1)"
            )

    def test_3methylpentane_prepass_frags_at_depth2(self):
        """Same check for a crf12_0 match (3-methylpentane)."""
        dag_on = _run("CCC(C)CC", smarts_prepass=True, max_depth=4)
        dag_off = _run("CCC(C)CC", smarts_prepass=False, max_depth=4)
        depth = _get_depth_onehot(dag_on)
        prepass_rows = _get_prepass_rows(dag_on, dag_off)
        assert len(prepass_rows) > 0
        for row in prepass_rows:
            assert depth[row, 2] == 1, (
                f"fragment {row} tagged as prepass but depth one-hot is {depth[row]}"
            )

    def test_prepass_frags_not_at_depth1(self):
        """No prepass fragment should have depth-1 one-hot set."""
        for smiles in ("CC(=O)OC", "CCC(C)CC"):
            dag_on = _run(smiles, smarts_prepass=True, max_depth=4)
            dag_off = _run(smiles, smarts_prepass=False, max_depth=4)
            depth = _get_depth_onehot(dag_on)
            prepass_rows = _get_prepass_rows(dag_on, dag_off)
            for row in prepass_rows:
                assert depth[row, 1] == 0, (
                    f"{smiles}: prepass fragment {row} has depth 1 "
                    f"(one-hot {depth[row]}); expected depth 2"
                )

    def test_bfs_depth1_frags_unchanged(self):
        """BFS-generated depth-1 fragments must still have depth 1."""
        dag_on = _run("CC(=O)OC", smarts_prepass=True, max_depth=4)
        dag_off = _run("CC(=O)OC", smarts_prepass=False, max_depth=4)
        depth = _get_depth_onehot(dag_on)
        prepass_rows = set(_get_prepass_rows(dag_on, dag_off).tolist())
        bfs_rows = np.asarray(
            [idx for idx in range(dag_on["dag"].num_nodes) if idx not in prepass_rows],
            dtype=np.int64,
        )
        bfs_non_root = bfs_rows[bfs_rows != 0]
        has_depth1 = np.any(depth[bfs_non_root, 1] == 1)
        assert has_depth1, "BFS-generated depth-1 fragments should still exist"

    def test_nodes_min_depth_is_2_for_prepass(self):
        """nodes_min_depth in dag_frag_meta must be 2 for prepass fragments."""
        dag_on = _run("CC(=O)OC", smarts_prepass=True, max_depth=4)
        dag_off = _run("CC(=O)OC", smarts_prepass=False, max_depth=4)
        nodes_min_depth = dag_on["nodes_min_depth"]
        prepass_rows = _get_prepass_rows(dag_on, dag_off)
        assert len(prepass_rows) > 0
        for row in prepass_rows:
            assert nodes_min_depth[row] == 2, (
                f"fragment {row}: nodes_min_depth={nodes_min_depth[row]}, expected 2"
            )
