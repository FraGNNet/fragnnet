"""Tests for scripts/debug/debug_compare_mbfs_dags.py helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch as th
from torch_geometric.data import Data

from fragnnet.utils import frag_utils


def load_module():
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "debug" / "debug_compare_mbfs_dags.py"
    spec = importlib.util.spec_from_file_location("debug_compare_mbfs_dags", str(script_path))
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Unable to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_dag(mask_rows: list[list[int]], edge_pairs: list[list[int]], formulas: dict[int, str]) -> dict:
    padded = th.zeros((len(mask_rows), frag_utils.MASK_SIZE), dtype=th.bool)
    for idx, row in enumerate(mask_rows):
        padded[idx, : len(row)] = th.tensor(row, dtype=th.bool)

    cc_long = frag_utils.th_mask_to_long(padded).to(th.int64)
    node_feats = th.cat(
        [
            th.zeros((len(mask_rows), 1), dtype=th.int64),
            cc_long,
            th.zeros((len(mask_rows), 1), dtype=th.int64),
        ],
        dim=1,
    )
    edge_index = (
        th.tensor(edge_pairs, dtype=th.int64).t()
        if edge_pairs
        else th.zeros((2, 0), dtype=th.int64)
    )
    edge_attr = th.zeros((edge_index.shape[1], 1), dtype=th.int64)
    dag = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_attr)
    dag.node_feat_idxs = th.tensor([[0, 1, 1 + frag_utils.MASK_SIZE // 64, 2 + frag_utils.MASK_SIZE // 64]])
    dag.edge_feat_idxs = th.tensor([[0, 1]])

    return {
        "dag": dag,
        "dag_num_nodes": len(mask_rows),
        "dag_num_edges": edge_index.shape[1],
        "reached_depth": 1,
        "force_stopped": False,
        "nodes_min_depth": th.tensor([0] + [1] * (len(mask_rows) - 1)).numpy(),
        "edges_min_depth": th.tensor([1] * edge_index.shape[1]).numpy(),
        "idx_to_formula": formulas,
        "dag_num_nodes_by_depth": {0: 1, 1: max(len(mask_rows) - 1, 0)},
        "dag_num_edges_by_depth": {0: 0, 1: edge_index.shape[1]},
    }


def test_compare_dag_summaries_identical():
    module = load_module()
    dag_d = _make_dag(
        mask_rows=[[1, 1, 1], [1, 0, 0], [0, 1, 1]],
        edge_pairs=[[0, 1], [0, 2]],
        formulas={0: "", 1: "C3H8", 2: "CH4"},
    )

    summary = module.summarize_dag(dag_d)
    diff = module.compare_dag_summaries(summary, summary)

    assert diff["node_count_delta"] == 0
    assert diff["edge_count_delta"] == 0
    assert diff["new_nodes_in_mbfs"] == []
    assert diff["missing_edges_in_mbfs"] == []
    assert diff["new_formulas_in_mbfs"] == []


def test_compare_dag_summaries_detects_extra_mbfs_fragment():
    module = load_module()
    baseline = _make_dag(
        mask_rows=[[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1]],
        edge_pairs=[[0, 1], [0, 2]],
        formulas={0: "", 1: "C4H10", 2: "C2H6"},
    )
    mbfs = _make_dag(
        mask_rows=[[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]],
        edge_pairs=[[0, 1], [0, 2], [0, 3]],
        formulas={0: "", 1: "C4H10", 2: "C2H6", 3: "C2H4"},
    )

    diff = module.compare_dag_summaries(module.summarize_dag(baseline), module.summarize_dag(mbfs))

    assert diff["node_count_delta"] == 1
    assert diff["edge_count_delta"] == 1
    assert diff["new_nodes_in_mbfs"] == [(1, 0, 1, 0)]
    assert diff["new_formulas_in_mbfs"] == ["C2H4"]
