"""Benchmarks for multi_cut_bfs.compute_ccs_multi_cut (Cython, optimized).

Run with::

    pytest tests/bench_multi_cut_bfs.py --benchmark-only -v

Or to save a baseline and compare later::

    pytest tests/bench_multi_cut_bfs.py --benchmark-only --benchmark-save=baseline
    # ... after changes ...
    pytest tests/bench_multi_cut_bfs.py --benchmark-only --benchmark-compare=baseline

Benchmark categories
--------------------
- Acyclic: no ring bonds — exercises cut=1 path only.
- Monocyclic: one ring — exercises cut=2 within a single system.
- Multi-ring-system: several disconnected ring systems — exercises the ring-
  grouping optimization (cross-system pairs are never attempted).
- Fused bicyclic / polycyclic: large fused ring count — exercises cut=2/3
  with many bonds in a single system.
- Drug-like: representative drug molecules covering a mix of the above.

Each benchmark is parameterized by (SMILES, max_cut_size, max_depth) and
repeated enough times to get stable timing via pytest-benchmark.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from fragnnet.frag.multi_cut_bfs import compute_ccs_multi_cut
from fragnnet.utils.frag_utils import extract_mol_info, get_fraggen_input_arrays

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _prepare(smiles: str):
    """Pre-compute all inputs for compute_ccs_multi_cut (excluded from timing)."""
    mol_d = extract_mol_info(smiles)
    ring_mask = mol_d["ring_bond_mask"]
    num_nodes, num_edges, node_mask, edges, edge_mask, n2e = get_fraggen_input_arrays(mol_d)
    return dict(
        num_nodes=num_nodes,
        num_edges=num_edges,
        node_mask_arr=node_mask,
        edges_arr=edges,
        edge_mask_arr=edge_mask,
        node_to_edge_idx_arr=n2e,
        ring_edge_mask=ring_mask,
    )


def _run(inputs: dict, max_cut_size: int = 2, max_depth: int = 3) -> int:
    """Run BFS and return number of DAG nodes produced."""
    nodes_mask, _, _, _ = compute_ccs_multi_cut(
        inputs["num_nodes"],
        inputs["num_edges"],
        inputs["node_mask_arr"],
        inputs["edges_arr"],
        inputs["edge_mask_arr"],
        inputs["node_to_edge_idx_arr"],
        max_depth=max_depth,
        time_limit=60,
        ring_edge_mask=inputs["ring_edge_mask"],
        max_cut_size=max_cut_size,
    )
    return len(nodes_mask)


# ---------------------------------------------------------------------------
# Molecule library
# ---------------------------------------------------------------------------

MOLECULES = {
    # Acyclic
    "n_butane":          "CCCC",
    "n_octane":          "CCCCCCCC",
    "n_hexadecane":      "CCCCCCCCCCCCCCCC",
    # Monocyclic
    "cyclohexane":       "C1CCCCC1",
    "cyclooctane":       "C1CCCCCCC1",
    "benzene":           "c1ccccc1",
    # Multi-ring-system (separate rings → grouping skips cross-system pairs)
    "biphenyl":          "c1ccccc1-c1ccccc1",
    "diphenylmethane":   "c1ccccc1Cc1ccccc1",
    "terphenyl":         "c1ccccc1-c1ccccc1-c1ccccc1",
    # Fused bicyclic
    "naphthalene":       "c1ccc2ccccc2c1",
    "tetralin":          "C1CCCc2ccccc21",
    "decalin":           "C1CCC2CCCCC2C1",
    # Fused polycyclic (large ring system — many same-system pairs)
    "pyrene":            "c1cc2ccc3cccc4ccc(c1)c2c34",
    "coronene":          "c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61",
    "steroid_skeleton":  "C1CCC2CCCC3CCCC4CCCCC4C3C2C1",   # androstane
    # Drug-like
    "ibuprofen":         "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "naproxen":          "CC(C(=O)O)c1ccc2cc(ccc2c1)OC",
    "caffeine":          "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "aspirin":           "CC(=O)Oc1ccccc1C(=O)O",
    "sildenafil":        "CCCC1=NN(C2=CC(=C(C=C2)S(=O)(=O)N3CCN(CC3)C)OCC)C(=O)C1=O",  # ~33 atoms
}


# ---------------------------------------------------------------------------
# pytest-benchmark tests
# ---------------------------------------------------------------------------


@pytest.fixture(params=list(MOLECULES.keys()))
def mol_inputs(request):
    """Fixture: yield pre-computed inputs for each molecule."""
    smiles = MOLECULES[request.param]
    return request.param, _prepare(smiles)


def test_bench_cut2_depth3(benchmark, mol_inputs):
    """Benchmark: max_cut_size=2, max_depth=3 (standard production setting)."""
    name, inputs = mol_inputs
    result = benchmark(_run, inputs, max_cut_size=2, max_depth=3)
    # Sanity: at least root node was returned
    assert result >= 1


def test_bench_cut2_depth1(benchmark, mol_inputs):
    """Benchmark: max_cut_size=2, max_depth=1 (single BFS pass — isolates cut overhead)."""
    name, inputs = mol_inputs
    result = benchmark(_run, inputs, max_cut_size=2, max_depth=1)
    assert result >= 1


# ---------------------------------------------------------------------------
# Raw wall-clock throughput test (no benchmark plugin needed)
# ---------------------------------------------------------------------------


class TestThroughput:
    """Coarse wall-clock tests: process N molecules and check mol/s thresholds.

    These are NOT strict performance SLAs — they just ensure we haven't
    introduced a catastrophic regression (e.g. accidentally going quadratic).
    The thresholds are intentionally conservative (10× below expected speed
    on a modern laptop).
    """

    N_REPEATS = 50  # how many times to repeat each molecule

    @pytest.mark.parametrize("smiles,min_mol_per_sec", [
        ("CCCCCCCC",              500),   # acyclic octane: trivially fast
        ("C1CCCCC1",              200),   # cyclohexane: 15 ring pairs
        ("c1ccc2ccccc2c1",         50),   # naphthalene: large same-system ring pairs
        ("C1CCC2CCCC3CCCC4CCCCC4C3C2C1",  10),  # steroid: large fused ring system
    ])
    def test_min_throughput(self, smiles: str, min_mol_per_sec: int):
        inputs = _prepare(smiles)
        t0 = time.perf_counter()
        for _ in range(self.N_REPEATS):
            _run(inputs, max_cut_size=2, max_depth=3)
        elapsed = time.perf_counter() - t0
        mol_per_sec = self.N_REPEATS / elapsed
        assert mol_per_sec >= min_mol_per_sec, (
            f"{smiles}: {mol_per_sec:.1f} mol/s < required {min_mol_per_sec} mol/s"
        )


# ---------------------------------------------------------------------------
# Quick comparison: cut=2 vs cut=1 on ring-heavy molecules
# ---------------------------------------------------------------------------


class TestCutSizeOverhead:
    """Measure the relative overhead of cut=2 vs cut=1 on different molecule types.

    Expected: ring-heavy molecules take longer with cut=2, but the overhead
    should be bounded (not explosive) due to ring-system grouping.
    """

    @pytest.mark.parametrize("smiles,max_overhead_factor", [
        # Comparison at max_depth=1 (single BFS pass) isolates the per-BFS-node
        # cut overhead without confounding from the exponentially larger DAG that
        # cut=2 generates at depth=2/3 (which dominates at deeper depths).
        ("CCCCCCCC",              2.0),   # acyclic: no ring pairs → overhead ≈ 0
        ("c1ccccc1-c1ccccc1",     8.0),   # biphenyl: 2 separate 6-bond systems
        ("c1ccc2ccccc2c1",        8.0),   # naphthalene: 1 large same-system group
        ("C1CCC2CCCC3CCCC4CCCCC4C3C2C1", 20.0),  # steroid: many ring bonds, 1 fused system
    ])
    def test_cut2_overhead_bounded(self, smiles: str, max_overhead_factor: float):
        """cut=2 overhead vs cut=1, measured at max_depth=1 (single BFS pass)."""
        inputs = _prepare(smiles)
        n = 100

        t0 = time.perf_counter()
        for _ in range(n):
            _run(inputs, max_cut_size=1, max_depth=1)
        t_cut1 = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(n):
            _run(inputs, max_cut_size=2, max_depth=1)
        t_cut2 = time.perf_counter() - t0

        factor = t_cut2 / max(t_cut1, 1e-9)
        assert factor <= max_overhead_factor, (
            f"{smiles}: cut=2 is {factor:.1f}× slower than cut=1 "
            f"(limit={max_overhead_factor}×). cut1={t_cut1*1000/n:.3f}ms, "
            f"cut2={t_cut2*1000/n:.3f}ms per mol"
        )
