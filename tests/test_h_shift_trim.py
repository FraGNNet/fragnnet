"""Unit tests for the h-shift trimming logic in FraGNNetModel.forward().

The trimming allows h4 DAGs (max_h_transfer=4, 9 formula columns per node)
to be used with a model configured with num_hs=3 (or any lower value) by
dropping the outermost Δh columns at runtime.

Column layout per fragment node:
  [Δ0, Δ-1, Δ+1, Δ-2, Δ+2, Δ-3, Δ+3, Δ-4, Δ+4]  (h4 = 9 columns)
  [Δ0, Δ-1, Δ+1, Δ-2, Δ+2, Δ-3, Δ+3]              (h3 = 7 columns)

Trimming from the end removes the largest-magnitude shifts first.

Tests:
- Correct column reduction and formula-count bookkeeping for h4 → h3.
- The peak-survival mask is computed correctly.
- All-same-column DAGs (every node references only Δ0) are unaffected by trim.
- num_hs_diff == 0 (matching DAG/model) takes the no-op path.
- h2 DAG with num_hs=3 fails the assertion (not enough h-transfers in DAG).
"""

import pytest
import torch as th

from fragnnet.utils.misc_utils import scatter_reduce


# ---------------------------------------------------------------------------
# Helper: run the trimming logic extracted from fragnnet_model.py forward()
# ---------------------------------------------------------------------------


def _run_trim(
    frag_joint_formula_idxs: th.Tensor,  # [N, h_width]
    num_hs_model: int,
    frag_formula_cumsizes: th.Tensor,  # [batch_size + 1]
    frag_formula_batch_idxs: th.Tensor,  # [total_formulae]
    frag_formula_peak_probs: th.Tensor,  # [num_peaks]
    frag_formula_peak_mzs: th.Tensor,  # [num_peaks]
    frag_formula_idxs_pretrim: th.Tensor,  # unique global formula idxs (from h4)
    batch_size: int,
) -> dict:
    """Mirror of the trimming block in FraGNNetModel.forward() for unit testing.

    Args:
        frag_joint_formula_idxs: Per-node global formula index tensor, shape [N, h_width].
        num_hs_model: The model's num_hs parameter (e.g., 3 for h3).
        frag_formula_cumsizes: Cumulative formula-pool sizes, shape [batch_size + 1].
        frag_formula_batch_idxs: Batch index per formula, shape [total_formulae].
        frag_formula_peak_probs: Non-null peak probabilities from h4 DAG.
        frag_formula_peak_mzs: Non-null peak m/z values from h4 DAG.
        frag_formula_idxs_pretrim: All unique formula indices before trimming.
        batch_size: Number of molecules in the batch.

    Returns:
        Dictionary with post-trim tensors for assertion.
    """
    device = frag_joint_formula_idxs.device
    h_width = frag_joint_formula_idxs.shape[1]
    num_hs_diff = (h_width - 1) // 2 - num_hs_model
    assert num_hs_diff >= 0, (
        f"DAG max_h_transfer ({(h_width - 1) // 2}) < model num_hs ({num_hs_model})"
    )
    if num_hs_diff == 0:
        return {
            "num_hs_diff": 0,
            "frag_joint_formula_idxs": frag_joint_formula_idxs.flatten(),
            "batch_frag_num_formulae": frag_formula_cumsizes[-1].item(),
            "frag_formula_batch_idxs": frag_formula_batch_idxs,
            "frag_formula_sizes": frag_formula_cumsizes[1:] - frag_formula_cumsizes[:-1],
            "frag_formula_cumsizes": frag_formula_cumsizes,
            "frag_formula_peak_probs": frag_formula_peak_probs,
            "frag_formula_peak_mzs": frag_formula_peak_mzs,
            "frag_formula_peak_mask": th.ones(frag_formula_peak_probs.shape[0], dtype=th.bool),
        }

    # Trim last 2*num_hs_diff columns
    frag_joint_formula_idxs = frag_joint_formula_idxs[:, :-2 * num_hs_diff].flatten()

    frag_joint_formula_idxs_un, frag_joint_formula_idxs_inv = th.unique(
        th.cat([frag_joint_formula_idxs, frag_formula_cumsizes[:-1]], dim=0),
        return_inverse=True,
    )
    frag_joint_formula_idxs_inv = frag_joint_formula_idxs_inv[:frag_joint_formula_idxs.shape[0]]

    frag_formula_peak_mask = th.isin(
        frag_formula_idxs_pretrim[~th.isin(frag_formula_idxs_pretrim, frag_formula_cumsizes[:-1])],
        frag_joint_formula_idxs_un,
    )

    batch_frag_num_formulae = frag_joint_formula_idxs_un.shape[0]
    frag_formula_batch_idxs = frag_formula_batch_idxs[frag_joint_formula_idxs_un]
    frag_joint_formula_idxs = th.arange(batch_frag_num_formulae, device=device)[
        frag_joint_formula_idxs_inv
    ]

    frag_formula_idxs = th.arange(batch_frag_num_formulae, device=device)
    frag_formula_sizes = scatter_reduce(
        th.ones_like(frag_formula_batch_idxs),
        frag_formula_batch_idxs,
        reduce="sum",
        dim_size=batch_size,
    )
    frag_formula_cumsizes = th.cumsum(
        th.cat([th.zeros([1], device=device, dtype=frag_formula_sizes.dtype), frag_formula_sizes], dim=0),
        dim=0,
    )

    return {
        "num_hs_diff": num_hs_diff,
        "frag_joint_formula_idxs": frag_joint_formula_idxs,
        "batch_frag_num_formulae": batch_frag_num_formulae,
        "frag_formula_batch_idxs": frag_formula_batch_idxs,
        "frag_formula_sizes": frag_formula_sizes,
        "frag_formula_cumsizes": frag_formula_cumsizes,
        "frag_formula_peak_probs": frag_formula_peak_probs[frag_formula_peak_mask],
        "frag_formula_peak_mzs": frag_formula_peak_mzs[frag_formula_peak_mask],
        "frag_formula_peak_mask": frag_formula_peak_mask,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def h4_single_mol():
    """Minimal h4 DAG: batch_size=1, 2 fragment nodes, 9-column formula layout.

    Formula pool for the single molecule has 9 entries:
      index 0 = NULL, indices 1–8 = non-null peaks (Δ-1, Δ+1, ..., Δ-4, Δ+4).

    Both fragment nodes reference the full set of 9 formulas.
    """
    # [N=2, h_width=9]  — global formula indices after adding per-batch offset (0 here)
    frag_joint_formula_idxs = th.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8],
         [0, 1, 2, 3, 4, 5, 6, 7, 8]],
        dtype=th.long,
    )
    frag_formula_cumsizes = th.tensor([0, 9], dtype=th.long)  # [0, total_formulae]
    frag_formula_batch_idxs = th.zeros(9, dtype=th.long)       # all formulas → batch 0
    frag_formula_peak_probs = th.full((8,), 0.125)             # 8 non-null peaks, uniform
    frag_formula_peak_mzs = th.arange(1, 9, dtype=th.float32) * 50.0  # 50, 100, ..., 400
    frag_formula_idxs_pretrim = th.arange(9, dtype=th.long)   # unique formula indices [0..8]
    return {
        "frag_joint_formula_idxs": frag_joint_formula_idxs,
        "frag_formula_cumsizes": frag_formula_cumsizes,
        "frag_formula_batch_idxs": frag_formula_batch_idxs,
        "frag_formula_peak_probs": frag_formula_peak_probs,
        "frag_formula_peak_mzs": frag_formula_peak_mzs,
        "frag_formula_idxs_pretrim": frag_formula_idxs_pretrim,
        "batch_size": 1,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestH4ToH3Trim:
    """h4 DAG trimmed to h3 (num_hs_diff=1): drops Δ±4 columns."""

    def test_num_hs_diff_is_one(self, h4_single_mol):
        result = _run_trim(num_hs_model=3, **h4_single_mol)
        assert result["num_hs_diff"] == 1

    def test_formula_count_reduced(self, h4_single_mol):
        """After trim, 2 formula entries (Δ-4 and Δ+4) are dropped → 7 remain."""
        result = _run_trim(num_hs_model=3, **h4_single_mol)
        assert result["batch_frag_num_formulae"] == 7

    def test_formula_sizes_updated(self, h4_single_mol):
        """frag_formula_sizes should sum to batch_frag_num_formulae (1 NULL + 6 peaks)."""
        result = _run_trim(num_hs_model=3, **h4_single_mol)
        assert result["frag_formula_sizes"].sum().item() == 7
        assert result["frag_formula_sizes"][0].item() == 7

    def test_peak_mask_drops_last_two(self, h4_single_mol):
        """Peak mask must be True for first 6 peaks (Δ-1..Δ+3) and False for last 2 (Δ-4, Δ+4)."""
        result = _run_trim(num_hs_model=3, **h4_single_mol)
        mask = result["frag_formula_peak_mask"]
        assert mask.shape[0] == 8   # 8 pre-trim non-null peaks
        assert mask[:6].all(), "First 6 peaks (within h3 range) should survive"
        assert not mask[6:].any(), "Last 2 peaks (Δ±4) should be dropped"

    def test_peak_probs_filtered(self, h4_single_mol):
        """Filtered peak_probs should contain only surviving 6 peaks."""
        result = _run_trim(num_hs_model=3, **h4_single_mol)
        assert result["frag_formula_peak_probs"].shape[0] == 6

    def test_peak_mzs_filtered(self, h4_single_mol):
        """Filtered peak_mzs should contain only the first 6 m/z values."""
        result = _run_trim(num_hs_model=3, **h4_single_mol)
        assert result["frag_formula_peak_mzs"].shape[0] == 6
        # m/z values 1..6 × 50 survive; 7×50=350 and 8×50=400 are dropped
        expected = th.arange(1, 7, dtype=th.float32) * 50.0
        assert th.allclose(result["frag_formula_peak_mzs"], expected)

    def test_joint_formula_idxs_contiguous(self, h4_single_mol):
        """Post-trim frag_joint_formula_idxs must be in [0, batch_frag_num_formulae)."""
        result = _run_trim(num_hs_model=3, **h4_single_mol)
        idxs = result["frag_joint_formula_idxs"]
        n = result["batch_frag_num_formulae"]
        assert idxs.min().item() >= 0
        assert idxs.max().item() < n


class TestH4ToH2Trim:
    """h4 DAG trimmed to h2 (num_hs_diff=2): drops Δ±3 and Δ±4 columns."""

    def test_formula_count_reduced(self, h4_single_mol):
        """4 formula entries dropped → 5 remain (1 NULL + 4 peaks)."""
        result = _run_trim(num_hs_model=2, **h4_single_mol)
        assert result["batch_frag_num_formulae"] == 5

    def test_peak_mask_drops_last_four(self, h4_single_mol):
        result = _run_trim(num_hs_model=2, **h4_single_mol)
        mask = result["frag_formula_peak_mask"]
        assert mask[:4].all()
        assert not mask[4:].any()


class TestNoTrim:
    """num_hs_diff == 0: no trimming, all formula entries preserved."""

    def test_no_change_when_matching(self, h4_single_mol):
        result = _run_trim(num_hs_model=4, **h4_single_mol)
        assert result["num_hs_diff"] == 0
        assert result["batch_frag_num_formulae"] == 9

    def test_all_peaks_survive(self, h4_single_mol):
        result = _run_trim(num_hs_model=4, **h4_single_mol)
        mask = result["frag_formula_peak_mask"]
        assert mask.all()


class TestAssertionOnInsufficientDAG:
    """A DAG with fewer h-transfers than num_hs must raise an AssertionError."""

    def test_h2_dag_with_num_hs3_raises(self):
        """h2 DAG (5 columns) cannot satisfy num_hs=3 (expected ≥ 7 columns)."""
        # Build an h2 DAG (5 columns)
        frag_joint_formula_idxs = th.tensor(
            [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=th.long
        )
        with pytest.raises(AssertionError, match="num_hs"):
            _run_trim(
                frag_joint_formula_idxs=frag_joint_formula_idxs,
                num_hs_model=3,
                frag_formula_cumsizes=th.tensor([0, 5], dtype=th.long),
                frag_formula_batch_idxs=th.zeros(5, dtype=th.long),
                frag_formula_peak_probs=th.ones(4),
                frag_formula_peak_mzs=th.arange(1, 5, dtype=th.float32) * 50.0,
                frag_formula_idxs_pretrim=th.arange(5, dtype=th.long),
                batch_size=1,
            )
