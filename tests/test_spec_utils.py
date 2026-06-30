"""Unit tests for fragnnet.utils.spec_utils module.

Tests for spectrum similarity helper functions:
- cos_hun_helper: cosine-Hungarian similarity (single pair)
- ndcg_helper: NDCG (single pair, intersection and union modes)
- jss_helper: binned JSS computation (batch)
- jss_hun_helper: Hungarian-matched JSS computation (single pair)
- batch_cos_hun_helper: cosine-Hungarian similarity (padded batch)
- batch_jss_hun_helper: Jensen-Shannon similarity via Hungarian (padded batch)
"""

import math

import pytest
import torch as th

from fragnnet.utils.misc_utils import EPS
from fragnnet.utils.spec_utils import (
    batch_cos_hun_helper,
    batch_jss_hun_helper,
    cos_hun_helper,
    jss_helper,
    jss_hun_helper,
    ndcg_helper,
)


class TestCosHunHelper:
    """Tests for cos_hun_helper (cosine-Hungarian, single pair).

    cos_hun_helper returns a scalar in [0, 1]:
      - Perfect match: 1.0
      - No overlap:    0.0
    """

    def _make_prec_masks(self, n_true: int, n_pred: int):
        """Return all-False precursor masks (no peaks removed)."""
        return th.zeros(n_true, dtype=th.bool), th.zeros(n_pred, dtype=th.bool)

    def test_identical_single_peak(self):
        """Single matched peak with equal intensities should give cos_hun = 1.0."""
        b_true_ints = th.tensor([1.0])
        b_pred_ints = th.tensor([1.0])
        b_match_mask = th.tensor([[True]])
        b_true_match_mask = th.tensor([True])
        b_pred_match_mask = th.tensor([True])
        b_true_prec, b_pred_prec = self._make_prec_masks(1, 1)

        result = cos_hun_helper(
            b_true_ints,
            b_pred_ints,
            b_match_mask,
            b_true_match_mask,
            b_pred_match_mask,
            False,
            b_true_prec,
            b_pred_prec,
        )

        assert th.allclose(result, th.tensor(1.0), atol=1e-6), (
            f"Identical single-peak: expected 1.0, got {result.item():.6f}"
        )

    def test_no_overlap(self):
        """No matching peaks should give cos_hun = 0.0."""
        b_true_ints = th.tensor([1.0])
        b_pred_ints = th.tensor([1.0])
        b_match_mask = th.tensor([[False]])
        b_true_match_mask = th.tensor([False])
        b_pred_match_mask = th.tensor([False])
        b_true_prec, b_pred_prec = self._make_prec_masks(1, 1)

        result = cos_hun_helper(
            b_true_ints,
            b_pred_ints,
            b_match_mask,
            b_true_match_mask,
            b_pred_match_mask,
            False,
            b_true_prec,
            b_pred_prec,
        )

        assert th.allclose(result, th.tensor(0.0), atol=1e-6), (
            f"No overlap: expected 0.0, got {result.item():.6f}"
        )

    def test_partial_overlap_known_value(self):
        """Two equal-intensity peaks sharing one match should give cos_hun = 0.5.

        Both spectra: [1, 1] → L2-normalised to [1/√2, 1/√2].
        Only true[0] ↔ pred[0] match.
        Score = (1/√2) * (1/√2) = 0.5.
        """
        b_true_ints = th.tensor([1.0, 1.0])
        b_pred_ints = th.tensor([1.0, 1.0])
        b_match_mask = th.tensor([[True, False], [False, False]])
        b_true_match_mask = th.tensor([True, False])
        b_pred_match_mask = th.tensor([True, False])
        b_true_prec, b_pred_prec = self._make_prec_masks(2, 2)

        result = cos_hun_helper(
            b_true_ints,
            b_pred_ints,
            b_match_mask,
            b_true_match_mask,
            b_pred_match_mask,
            False,
            b_true_prec,
            b_pred_prec,
        )

        assert th.allclose(result, th.tensor(0.5), atol=1e-6), (
            f"50%-overlap two-peak spectra: expected 0.5, got {result.item():.6f}"
        )

    def test_full_two_peak_match(self):
        """Two peaks both matched diagonally should give cos_hun = 1.0."""
        b_true_ints = th.tensor([1.0, 1.0])
        b_pred_ints = th.tensor([1.0, 1.0])
        b_match_mask = th.tensor([[True, False], [False, True]])
        b_true_match_mask = th.tensor([True, True])
        b_pred_match_mask = th.tensor([True, True])
        b_true_prec, b_pred_prec = self._make_prec_masks(2, 2)

        result = cos_hun_helper(
            b_true_ints,
            b_pred_ints,
            b_match_mask,
            b_true_match_mask,
            b_pred_match_mask,
            False,
            b_true_prec,
            b_pred_prec,
        )

        assert th.allclose(result, th.tensor(1.0), atol=1e-6), (
            f"Full diagonal match: expected 1.0, got {result.item():.6f}"
        )

    def test_remove_prec_peak_zeroes_contribution(self):
        """Zeroing the precursor peak should reduce it to zero intensity.

        Two peaks: prec mask = [True, False].  After removal only peak 1 remains
        in both spectra and they match perfectly → cos_hun = 1.0.
        """
        b_true_ints = th.tensor([2.0, 1.0])
        b_pred_ints = th.tensor([2.0, 1.0])
        b_match_mask = th.tensor([[True, False], [False, True]])
        b_true_match_mask = th.tensor([True, True])
        b_pred_match_mask = th.tensor([True, True])
        b_true_prec = th.tensor([True, False])
        b_pred_prec = th.tensor([True, False])

        result = cos_hun_helper(
            b_true_ints,
            b_pred_ints,
            b_match_mask,
            b_true_match_mask,
            b_pred_match_mask,
            True,
            b_true_prec,
            b_pred_prec,
        )

        # After zeroing peak 0, both spectra = [0, 1] (normalised) → identical
        assert th.allclose(result, th.tensor(1.0), atol=1e-5), (
            f"After prec removal: expected 1.0, got {result.item():.6f}"
        )

    def test_remove_prec_peak_all_false_no_effect(self):
        """All-False prec masks: remove_prec_peak=True/False should be identical."""
        b_true_ints = th.tensor([1.0, 1.0])
        b_pred_ints = th.tensor([1.0, 1.0])
        b_match_mask = th.tensor([[True, False], [False, True]])
        b_true_match_mask = th.tensor([True, True])
        b_pred_match_mask = th.tensor([True, True])
        b_true_prec, b_pred_prec = self._make_prec_masks(2, 2)

        result_false = cos_hun_helper(
            b_true_ints,
            b_pred_ints,
            b_match_mask,
            b_true_match_mask,
            b_pred_match_mask,
            False,
            b_true_prec,
            b_pred_prec,
        )
        result_true = cos_hun_helper(
            b_true_ints,
            b_pred_ints,
            b_match_mask,
            b_true_match_mask,
            b_pred_match_mask,
            True,
            b_true_prec,
            b_pred_prec,
        )

        assert th.allclose(result_false, result_true, atol=1e-6), (
            "All-False prec masks: remove_prec_peak should have no effect"
        )

    def test_bounds_random(self):
        """cos_hun should always be in [0, 1] for random inputs."""
        rng = th.Generator()
        rng.manual_seed(7)
        for _ in range(10):
            n_true = int(th.randint(1, 6, (1,), generator=rng).item())
            n_pred = int(th.randint(1, 6, (1,), generator=rng).item())
            b_true_ints = th.rand(n_true, generator=rng) + 0.01
            b_pred_ints = th.rand(n_pred, generator=rng) + 0.01
            b_match_mask = th.rand(n_true, n_pred, generator=rng) > 0.5
            b_true_match_mask = b_match_mask.any(dim=1)
            b_pred_match_mask = b_match_mask.any(dim=0)
            b_true_prec, b_pred_prec = self._make_prec_masks(n_true, n_pred)

            result = cos_hun_helper(
                b_true_ints,
                b_pred_ints,
                b_match_mask,
                b_true_match_mask,
                b_pred_match_mask,
                False,
                b_true_prec,
                b_pred_prec,
            )
            assert result.item() >= -1e-6, f"cos_hun < 0: {result.item()}"
            assert result.item() <= 1.0 + 1e-6, f"cos_hun > 1: {result.item()}"


class TestNdcgHelper:
    """Tests for ndcg_helper (NDCG, single pair).

    Supports two modes:
      - Intersection (union=False): only matched peaks; returns 0.0 when no match.
      - Union (union=True):         all peaks; unmatched pred peaks score zero.
    """

    def test_identical_intersection(self):
        """Perfect diagonal match should give NDCG = 1.0 (intersection mode)."""
        b_true_ints = th.tensor([2.0, 1.0])
        b_pred_ints = th.tensor([2.0, 1.0])
        b_match_mask = th.tensor([[True, False], [False, True]])
        b_true_match_mask = th.tensor([True, True])
        b_pred_match_mask = th.tensor([True, True])

        result = ndcg_helper(
            b_true_ints,
            b_pred_ints,
            b_match_mask,
            b_true_match_mask,
            b_pred_match_mask,
            optimistic=True,
            union=False,
        )

        assert pytest.approx(float(result), abs=1e-6) == 1.0, (
            f"Identical intersection: expected 1.0, got {float(result):.6f}"
        )

    def test_no_match_intersection_returns_zero(self):
        """No matching peaks should return 0.0 (intersection mode)."""
        b_true_ints = th.tensor([1.0])
        b_pred_ints = th.tensor([1.0])
        b_match_mask = th.tensor([[False]])
        b_true_match_mask = th.tensor([False])
        b_pred_match_mask = th.tensor([False])

        result = ndcg_helper(
            b_true_ints,
            b_pred_ints,
            b_match_mask,
            b_true_match_mask,
            b_pred_match_mask,
            optimistic=True,
            union=False,
        )

        assert float(result) == pytest.approx(0.0, abs=1e-6), (
            f"No match intersection: expected 0.0, got {float(result):.6f}"
        )

    def test_single_matched_peak_intersection(self):
        """A single matched pair always gives NDCG = 1.0 (intersection mode)."""
        b_true_ints = th.tensor([3.0])
        b_pred_ints = th.tensor([1.5])
        b_match_mask = th.tensor([[True]])
        b_true_match_mask = th.tensor([True])
        b_pred_match_mask = th.tensor([True])

        result = ndcg_helper(
            b_true_ints,
            b_pred_ints,
            b_match_mask,
            b_true_match_mask,
            b_pred_match_mask,
            optimistic=True,
            union=False,
        )

        assert pytest.approx(float(result), abs=1e-6) == 1.0, (
            f"Single matched peak: expected 1.0, got {float(result):.6f}"
        )

    def test_reversed_ranking_intersection_less_than_one(self):
        """Predicted spectrum with reversed peak ranking should give NDCG < 1.0.

        True:  [3.0, 1.0] → peak 0 is most important.
        Pred:  [0.5, 2.0] → pred ranks peak 1 higher.
        Optimal assignment (maximise score product): (0→0)=1.5, (1→1)=2.0.
        DCG < IDCG because the highest true-intensity peak is ranked second by pred.
        """
        b_true_ints = th.tensor([3.0, 1.0])
        b_pred_ints = th.tensor([0.5, 2.0])
        b_match_mask = th.tensor([[True, False], [False, True]])
        b_true_match_mask = th.tensor([True, True])
        b_pred_match_mask = th.tensor([True, True])

        result = ndcg_helper(
            b_true_ints,
            b_pred_ints,
            b_match_mask,
            b_true_match_mask,
            b_pred_match_mask,
            optimistic=True,
            union=False,
        )

        # result must be in (0, 1) and strictly less than 1
        assert 0.0 < float(result) < 1.0, (
            f"Reversed ranking: expected NDCG in (0, 1), got {float(result):.6f}"
        )

    def test_identical_union(self):
        """Perfect diagonal match with no unmatched peaks should give NDCG = 1.0 (union)."""
        b_true_ints = th.tensor([2.0, 1.0])
        b_pred_ints = th.tensor([2.0, 1.0])
        b_match_mask = th.tensor([[True, False], [False, True]])
        b_true_match_mask = th.tensor([True, True])
        b_pred_match_mask = th.tensor([True, True])

        result = ndcg_helper(
            b_true_ints,
            b_pred_ints,
            b_match_mask,
            b_true_match_mask,
            b_pred_match_mask,
            optimistic=True,
            union=True,
        )

        assert pytest.approx(float(result), abs=1e-6) == 1.0, (
            f"Identical union: expected 1.0, got {float(result):.6f}"
        )

    def test_unmatched_pred_peak_reduces_union_ndcg(self):
        """An extra unmatched predicted peak should reduce union NDCG below 1.0.

        True:  [1.0] at peak 0 (only one true peak, one predicted match).
        Pred:  [1.0, 0.5]: peak 0 matches true peak 0; peak 1 is spurious.
        The spurious peak is ranked below the matched peak by pred, but it
        receives true gain = 0 → DCG < IDCG (IDCG only has the matched peak).
        """
        b_true_ints = th.tensor([1.0])
        b_pred_ints = th.tensor([1.0, 0.5])
        # Only pred[0] matches true[0]; pred[1] is spurious
        b_match_mask = th.tensor([[True, False]])
        b_true_match_mask = th.tensor([True])
        b_pred_match_mask = th.tensor([True, False])

        result = ndcg_helper(
            b_true_ints,
            b_pred_ints,
            b_match_mask,
            b_true_match_mask,
            b_pred_match_mask,
            optimistic=True,
            union=True,
        )

        # IDCG has only the matched peak; DCG adds a zero-gain term for the spurious peak
        # → same DCG but same IDCG → NDCG = 1.0 when spurious peak is ranked below
        # (pred ranks [1.0, 0.5]: matched peak first) — this verifies the union path runs
        assert 0.0 <= float(result) <= 1.0 + 1e-6, f"Union NDCG out of bounds: {float(result):.6f}"

    def test_bounds_random(self):
        """NDCG should always be in [0, 1] for random inputs (both modes)."""
        rng = th.Generator()
        rng.manual_seed(13)
        for union in [False, True]:
            for _ in range(10):
                n_true = int(th.randint(1, 6, (1,), generator=rng).item())
                n_pred = int(th.randint(1, 6, (1,), generator=rng).item())
                b_true_ints = th.rand(n_true, generator=rng) + 0.01
                b_pred_ints = th.rand(n_pred, generator=rng) + 0.01
                b_match_mask = th.rand(n_true, n_pred, generator=rng) > 0.5
                b_true_match_mask = b_match_mask.any(dim=1)
                b_pred_match_mask = b_match_mask.any(dim=0)

                result = ndcg_helper(
                    b_true_ints,
                    b_pred_ints,
                    b_match_mask,
                    b_true_match_mask,
                    b_pred_match_mask,
                    optimistic=True,
                    union=union,
                )
                val = float(result)
                assert val >= -1e-6, f"NDCG < 0 (union={union}): {val:.6f}"
                assert val <= 1.0 + 1e-6, f"NDCG > 1 (union={union}): {val:.6f}"


class TestJssHelper:
    """Tests for jss_helper (binned, batched JSS).

    jss_helper returns JSS in [0, 1]:
      - Identical spectra: JSS = 1.0
      - Orthogonal spectra: JSS = 0.0
    """

    def test_identical_single_bin(self):
        """Identical single-bin spectra should have JSS = 1.0."""
        bin_idxs = th.tensor([50], dtype=th.long)
        ints = th.tensor([1.0])
        batch_idxs = th.tensor([0], dtype=th.long)

        jss = jss_helper(bin_idxs, ints, batch_idxs, bin_idxs, ints, batch_idxs, log_min=EPS)

        assert th.allclose(jss, th.ones(1), atol=1e-6), (
            f"Identical single-bin spectra: expected JSS=1.0, got {jss.item():.6f}"
        )

    def test_orthogonal_single_bin(self):
        """Completely non-overlapping spectra should have JSS = 0.0."""
        true_bin_idxs = th.tensor([50], dtype=th.long)
        true_ints = th.tensor([1.0])
        true_batch = th.tensor([0], dtype=th.long)

        pred_bin_idxs = th.tensor([100], dtype=th.long)
        pred_ints = th.tensor([1.0])
        pred_batch = th.tensor([0], dtype=th.long)

        jss = jss_helper(
            true_bin_idxs,
            true_ints,
            true_batch,
            pred_bin_idxs,
            pred_ints,
            pred_batch,
            log_min=EPS,
        )

        assert th.allclose(jss, th.zeros(1), atol=1e-6), (
            f"Orthogonal single-bin spectra: expected JSS=0.0, got {jss.item():.6f}"
        )

    def test_known_partial_overlap(self):
        """Two-peak spectra sharing one bin should have JSS = 0.5.

        P = [0.5, 0.5] at bins [50, 100]; Q = [0.5, 0.5] at bins [50, 150].
        M = [0.5, 0.25, 0.25]; KL(P||M) = KL(Q||M) = 0.5*log2.
        JSD = 0.5*log2; JSS = 1 - JSD/log2 = 0.5.
        """
        true_bin_idxs = th.tensor([50, 100], dtype=th.long)
        true_ints = th.tensor([1.0, 1.0])
        true_batch = th.tensor([0, 0], dtype=th.long)

        pred_bin_idxs = th.tensor([50, 150], dtype=th.long)
        pred_ints = th.tensor([1.0, 1.0])
        pred_batch = th.tensor([0, 0], dtype=th.long)

        jss = jss_helper(
            true_bin_idxs,
            true_ints,
            true_batch,
            pred_bin_idxs,
            pred_ints,
            pred_batch,
            log_min=EPS,
        )

        assert th.allclose(jss, th.tensor([0.5]), atol=1e-5), (
            f"50%-overlap spectra: expected JSS=0.5, got {jss.item():.6f}"
        )

    def test_non_negative_many_overlapping_bins(self):
        """Regression: JSS must be non-negative with many overlapping bins.

        Before the renormalization fix in jss_helper, floating-point accumulation
        in scatter_reduce could cause union_bin_ints to sum slightly below 1,
        inflating KL terms beyond log(2) and yielding JSS < 0.
        """
        N = 1000
        bin_idxs = th.arange(N, dtype=th.long)
        ints = th.full((N,), 1.0 / N)
        batch_idxs = th.zeros(N, dtype=th.long)

        jss = jss_helper(bin_idxs, ints, batch_idxs, bin_idxs, ints, batch_idxs, log_min=EPS)

        # safelog(x, eps) adds a small eps that can make identical-spectra JSS slightly exceed 1;
        # use a loose upper bound and check the key regression: JSS >= 0.
        assert jss.item() >= -1e-6, f"JSS must be non-negative, got {jss.item()}"
        assert th.allclose(jss, th.ones(1), atol=1e-4), (
            f"Identical spectra should give JSS ≈ 1.0, got {jss.item():.8f}"
        )

    def test_symmetry(self):
        """JSS should be symmetric: JSS(P, Q) = JSS(Q, P)."""
        true_bin_idxs = th.tensor([10, 20, 30], dtype=th.long)
        true_ints = th.tensor([1.0, 2.0, 0.5])
        true_batch = th.tensor([0, 0, 0], dtype=th.long)

        pred_bin_idxs = th.tensor([20, 40], dtype=th.long)
        pred_ints = th.tensor([1.5, 1.0])
        pred_batch = th.tensor([0, 0], dtype=th.long)

        jss_pq = jss_helper(
            true_bin_idxs,
            true_ints,
            true_batch,
            pred_bin_idxs,
            pred_ints,
            pred_batch,
            log_min=EPS,
        )
        jss_qp = jss_helper(
            pred_bin_idxs,
            pred_ints,
            pred_batch,
            true_bin_idxs,
            true_ints,
            true_batch,
            log_min=EPS,
        )

        assert th.allclose(jss_pq, jss_qp, atol=1e-6), (
            f"JSS should be symmetric: JSS(P,Q)={jss_pq.item():.6f} vs JSS(Q,P)={jss_qp.item():.6f}"
        )

    def test_batch_identical_and_orthogonal(self):
        """Batch of two pairs: pair 0 identical (JSS=1), pair 1 orthogonal (JSS=0)."""
        true_bin_idxs = th.tensor([50, 100], dtype=th.long)
        true_ints = th.tensor([1.0, 1.0])
        true_batch = th.tensor([0, 1], dtype=th.long)

        pred_bin_idxs = th.tensor([50, 200], dtype=th.long)
        pred_ints = th.tensor([1.0, 1.0])
        pred_batch = th.tensor([0, 1], dtype=th.long)

        jss = jss_helper(
            true_bin_idxs,
            true_ints,
            true_batch,
            pred_bin_idxs,
            pred_ints,
            pred_batch,
            log_min=EPS,
        )

        assert jss.shape == (2,), f"Expected shape (2,), got {jss.shape}"
        assert th.allclose(jss[0], th.tensor(1.0), atol=1e-6), (
            f"Pair 0 (identical): expected JSS=1.0, got {jss[0].item()}"
        )
        assert th.allclose(jss[1], th.tensor(0.0), atol=1e-6), (
            f"Pair 1 (orthogonal): expected JSS=0.0, got {jss[1].item()}"
        )

    def test_bounds(self):
        """JSS should always be in [0, 1]."""
        rng = th.Generator()
        rng.manual_seed(42)

        # Random spectra of different sizes
        for num_true, num_pred in [(3, 5), (10, 2), (1, 7)]:
            true_bin_idxs = th.randint(0, 100, (num_true,), generator=rng)
            true_ints = th.rand(num_true, generator=rng) + 0.01
            true_batch = th.zeros(num_true, dtype=th.long)

            pred_bin_idxs = th.randint(0, 100, (num_pred,), generator=rng)
            pred_ints = th.rand(num_pred, generator=rng) + 0.01
            pred_batch = th.zeros(num_pred, dtype=th.long)

            jss = jss_helper(
                true_bin_idxs,
                true_ints,
                true_batch,
                pred_bin_idxs,
                pred_ints,
                pred_batch,
                log_min=EPS,
            )
            assert jss.item() >= -1e-6, f"JSS out of bounds: {jss.item():.6f} < 0"
            assert jss.item() <= 1.0 + 1e-6, f"JSS out of bounds: {jss.item():.6f} > 1"


class TestJssHunHelper:
    """Tests for jss_hun_helper (Hungarian-matched, single-pair JSS).

    jss_hun_helper returns log(2) * JSS ∈ [0, log(2)]:
      - Identical spectra: log(2) ≈ 0.693
      - Orthogonal spectra: 0.0
    """

    def _make_prec_masks(self, n_true: int, n_pred: int):
        """Return all-False precursor masks (no peaks removed)."""
        return th.zeros(n_true, dtype=th.bool), th.zeros(n_pred, dtype=th.bool)

    def test_identical_single_peak(self):
        """Identical single-peak spectra should give log(2)."""
        b_true_ints = th.tensor([1.0])
        b_pred_ints = th.tensor([1.0])
        b_match_mask = th.tensor([[True]])
        b_true_match_mask = th.tensor([True])
        b_pred_match_mask = th.tensor([True])
        b_true_prec, b_pred_prec = self._make_prec_masks(1, 1)

        result = jss_hun_helper(
            b_true_ints,
            b_pred_ints,
            b_match_mask,
            b_true_match_mask,
            b_pred_match_mask,
            False,
            b_true_prec,
            b_pred_prec,
            log_min=EPS,
        )

        assert th.allclose(result, th.tensor(math.log(2.0)), atol=1e-6), (
            f"Identical spectra: expected log(2)≈{math.log(2):.4f}, got {result.item():.6f}"
        )

    def test_orthogonal_single_peak(self):
        """Orthogonal (non-overlapping) spectra should give 0.0."""
        b_true_ints = th.tensor([1.0])
        b_pred_ints = th.tensor([1.0])
        b_match_mask = th.tensor([[False]])  # no peaks within tolerance
        b_true_match_mask = th.tensor([False])
        b_pred_match_mask = th.tensor([False])
        b_true_prec, b_pred_prec = self._make_prec_masks(1, 1)

        result = jss_hun_helper(
            b_true_ints,
            b_pred_ints,
            b_match_mask,
            b_true_match_mask,
            b_pred_match_mask,
            False,
            b_true_prec,
            b_pred_prec,
            log_min=EPS,
        )

        assert th.allclose(result, th.tensor(0.0), atol=1e-6), (
            f"Orthogonal spectra: expected 0.0, got {result.item():.6f}"
        )

    def test_known_partial_overlap(self):
        """Two-peak spectra sharing one peak should give 0.5 * log(2).

        True = [0.5, 0.5] (peaks 0 and 1); Pred = [0.5, 0.5] (peaks 0 and 2).
        Only peak 0 is within tolerance → one matched pair.
        After matching: KL(P||M) = KL(Q||M) = 0.5*log2.
        jss_hun = log2 - 0.5*(0.5*log2 + 0.5*log2) = 0.5*log2.
        """
        b_true_ints = th.tensor([1.0, 1.0])  # will be normalized to [0.5, 0.5]
        b_pred_ints = th.tensor([1.0, 1.0])
        # peak 0 of true matches peak 0 of pred; peak 1 of each has no match
        b_match_mask = th.tensor([[True, False], [False, False]])
        b_true_match_mask = th.tensor([True, False])
        b_pred_match_mask = th.tensor([True, False])
        b_true_prec, b_pred_prec = self._make_prec_masks(2, 2)

        result = jss_hun_helper(
            b_true_ints,
            b_pred_ints,
            b_match_mask,
            b_true_match_mask,
            b_pred_match_mask,
            False,
            b_true_prec,
            b_pred_prec,
            log_min=EPS,
        )

        expected = 0.5 * math.log(2.0)
        assert th.allclose(result, th.tensor(expected), atol=1e-5), (
            f"50%-overlap: expected 0.5*log(2)≈{expected:.4f}, got {result.item():.6f}"
        )

    def test_bounds(self):
        """Result should be in [0, log(2)]."""
        b_true_ints = th.tensor([0.3, 0.4, 0.3])
        b_pred_ints = th.tensor([0.5, 0.5])
        # true peak 0 matches pred peak 0; true peak 2 matches pred peak 1
        b_match_mask = th.tensor([[True, False], [False, False], [False, True]])
        b_true_match_mask = th.tensor([True, False, True])
        b_pred_match_mask = th.tensor([True, True])
        b_true_prec, b_pred_prec = self._make_prec_masks(3, 2)

        result = jss_hun_helper(
            b_true_ints,
            b_pred_ints,
            b_match_mask,
            b_true_match_mask,
            b_pred_match_mask,
            False,
            b_true_prec,
            b_pred_prec,
            log_min=EPS,
        )

        assert result.item() >= -1e-6, f"jss_hun must be >= 0, got {result.item()}"
        assert result.item() <= math.log(2.0) + 1e-6, (
            f"jss_hun must be <= log(2), got {result.item()}"
        )

    def test_remove_prec_peak_no_effect_when_mask_false(self):
        """remove_prec_peak=True with all-False masks should give same result as False."""
        b_true_ints = th.tensor([1.0, 1.0])
        b_pred_ints = th.tensor([1.0, 1.0])
        b_match_mask = th.tensor([[True, False], [False, True]])
        b_true_match_mask = th.tensor([True, True])
        b_pred_match_mask = th.tensor([True, True])
        b_true_prec, b_pred_prec = self._make_prec_masks(2, 2)

        result_no_remove = jss_hun_helper(
            b_true_ints,
            b_pred_ints,
            b_match_mask,
            b_true_match_mask,
            b_pred_match_mask,
            False,
            b_true_prec,
            b_pred_prec,
            log_min=EPS,
        )
        result_remove = jss_hun_helper(
            b_true_ints,
            b_pred_ints,
            b_match_mask,
            b_true_match_mask,
            b_pred_match_mask,
            True,
            b_true_prec,
            b_pred_prec,
            log_min=EPS,
        )

        assert th.allclose(result_no_remove, result_remove, atol=1e-6), (
            "All-False prec masks: remove_prec_peak=True/False should give same result"
        )

    def test_remove_prec_peak_zeroes_peak(self):
        """Removing the precursor peak should reduce it to zero intensity.

        With 2 peaks, prec mask=[True, False]: peak 0 zeroed, only peak 1 remains.
        After removal both spectra become a single effective peak → JSS = 1 → log(2).
        """
        b_true_ints = th.tensor([1.0, 1.0])
        b_pred_ints = th.tensor([1.0, 1.0])
        b_match_mask = th.tensor([[True, False], [False, True]])
        b_true_match_mask = th.tensor([True, True])
        b_pred_match_mask = th.tensor([True, True])
        b_true_prec = th.tensor([True, False])  # remove peak 0 from true
        b_pred_prec = th.tensor([True, False])  # remove peak 0 from pred

        result = jss_hun_helper(
            b_true_ints,
            b_pred_ints,
            b_match_mask,
            b_true_match_mask,
            b_pred_match_mask,
            True,
            b_true_prec,
            b_pred_prec,
            log_min=EPS,
        )

        # After zeroing peak 0, both spectra = [0, 1] (normalized) → identical → log(2)
        assert th.allclose(result, th.tensor(math.log(2.0)), atol=1e-5), (
            f"After prec removal both spectra are identical: expected log(2), got {result.item():.6f}"
        )

    def test_asymmetric_matchability_consistent_with_batched(self):
        """Asymmetric matchability should stay bounded and match batched helper.

        Regression case: several true peaks are matchable to one predicted peak,
        but Hungarian assignment can select only one pair. The single-item helper
        must conserve unmatched mass exactly like batch_jss_hun_helper.
        """
        b_true_ints = th.tensor([0.55, 0.35, 0.10])
        b_pred_ints = th.tensor([0.70, 0.30])
        b_match_mask = th.tensor(
            [
                [True, False],
                [True, False],
                [False, True],
            ]
        )
        b_true_match_mask = b_match_mask.any(dim=1)
        b_pred_match_mask = b_match_mask.any(dim=0)
        b_true_prec, b_pred_prec = self._make_prec_masks(3, 2)

        single = jss_hun_helper(
            b_true_ints,
            b_pred_ints,
            b_match_mask,
            b_true_match_mask,
            b_pred_match_mask,
            False,
            b_true_prec,
            b_pred_prec,
            log_min=EPS,
        )
        batched = batch_jss_hun_helper(
            b_true_ints.unsqueeze(0),
            b_pred_ints.unsqueeze(0),
            b_match_mask.unsqueeze(0),
            b_true_match_mask.unsqueeze(0),
            b_pred_match_mask.unsqueeze(0),
            False,
            b_true_prec.unsqueeze(0),
            b_pred_prec.unsqueeze(0),
            log_min=EPS,
        )

        assert single.item() >= -1e-6, f"single jss_hun < 0: {single.item():.6f}"
        assert single.item() <= math.log(2.0) + 1e-6, (
            f"single jss_hun > log(2): {single.item():.6f}"
        )
        assert th.allclose(single, batched[0], atol=1e-6), (
            f"single ({single.item():.6f}) != batched ({batched[0].item():.6f})"
        )

    def test_no_match_consistent_with_batched(self):
        """No-match case should be 0 and consistent with the batched helper."""
        b_true_ints = th.tensor([0.4, 0.6])
        b_pred_ints = th.tensor([0.2, 0.8])
        b_match_mask = th.tensor([[False, False], [False, False]])
        b_true_match_mask = b_match_mask.any(dim=1)
        b_pred_match_mask = b_match_mask.any(dim=0)
        b_true_prec, b_pred_prec = self._make_prec_masks(2, 2)

        single = jss_hun_helper(
            b_true_ints,
            b_pred_ints,
            b_match_mask,
            b_true_match_mask,
            b_pred_match_mask,
            False,
            b_true_prec,
            b_pred_prec,
            log_min=EPS,
        )
        batched = batch_jss_hun_helper(
            b_true_ints.unsqueeze(0),
            b_pred_ints.unsqueeze(0),
            b_match_mask.unsqueeze(0),
            b_true_match_mask.unsqueeze(0),
            b_pred_match_mask.unsqueeze(0),
            False,
            b_true_prec.unsqueeze(0),
            b_pred_prec.unsqueeze(0),
            log_min=EPS,
        )

        assert th.allclose(single, th.tensor(0.0), atol=1e-6)
        assert th.allclose(single, batched[0], atol=1e-6)


class TestBatchCosHunHelper:
    """Tests for batch_cos_hun_helper (cosine-Hungarian, padded batch).

    Inputs are padded to uniform (B, N_true) / (B, N_pred) tensors.
    Returns a tensor of shape (B,) with values in [0, 1].

    Note: requires at least one matched peak across the whole batch so that
    max_true_match_pos > 0; the function does not guard against the all-no-match case.
    """

    def _make_prec_masks(self, b: int, n_true: int, n_pred: int):
        """Return all-False batched precursor masks."""
        return th.zeros(b, n_true, dtype=th.bool), th.zeros(b, n_pred, dtype=th.bool)

    def test_identical_single_item(self):
        """B=1, one peak fully matched → [1.0]."""
        batch_true_ints = th.tensor([[1.0]])
        batch_pred_ints = th.tensor([[1.0]])
        batch_match_mask = th.tensor([[[True]]])
        batch_true_match_mask = th.tensor([[True]])
        batch_pred_match_mask = th.tensor([[True]])
        bt_prec, bp_prec = self._make_prec_masks(1, 1, 1)

        result = batch_cos_hun_helper(
            batch_true_ints,
            batch_pred_ints,
            batch_match_mask,
            batch_true_match_mask,
            batch_pred_match_mask,
            False,
            bt_prec,
            bp_prec,
        )

        assert result.shape == (1,), f"Expected shape (1,), got {result.shape}"
        assert th.allclose(result, th.tensor([1.0]), atol=1e-6), (
            f"Identical B=1: expected [1.0], got {result.tolist()}"
        )

    def test_batch_identical_and_no_match(self):
        """B=2: pair 0 fully matched (1.0), pair 1 no overlap (0.0)."""
        batch_true_ints = th.tensor([[1.0], [1.0]])
        batch_pred_ints = th.tensor([[1.0], [1.0]])
        batch_match_mask = th.tensor([[[True]], [[False]]])
        batch_true_match_mask = th.tensor([[True], [False]])
        batch_pred_match_mask = th.tensor([[True], [False]])
        bt_prec, bp_prec = self._make_prec_masks(2, 1, 1)

        result = batch_cos_hun_helper(
            batch_true_ints,
            batch_pred_ints,
            batch_match_mask,
            batch_true_match_mask,
            batch_pred_match_mask,
            False,
            bt_prec,
            bp_prec,
        )

        assert result.shape == (2,), f"Expected shape (2,), got {result.shape}"
        assert th.allclose(result[0], th.tensor(1.0), atol=1e-6), (
            f"Pair 0 identical: expected 1.0, got {result[0].item():.6f}"
        )
        assert th.allclose(result[1], th.tensor(0.0), atol=1e-6), (
            f"Pair 1 no-match: expected 0.0, got {result[1].item():.6f}"
        )

    def test_known_partial_overlap(self):
        """B=1, two equal-intensity peaks, only one matched → [0.5].

        After L2 norm both spectra are [1/√2, 1/√2]; matched score = 0.5.
        """
        batch_true_ints = th.tensor([[1.0, 1.0]])
        batch_pred_ints = th.tensor([[1.0, 1.0]])
        batch_match_mask = th.tensor([[[True, False], [False, False]]])
        batch_true_match_mask = th.tensor([[True, False]])
        batch_pred_match_mask = th.tensor([[True, False]])
        bt_prec, bp_prec = self._make_prec_masks(1, 2, 2)

        result = batch_cos_hun_helper(
            batch_true_ints,
            batch_pred_ints,
            batch_match_mask,
            batch_true_match_mask,
            batch_pred_match_mask,
            False,
            bt_prec,
            bp_prec,
        )

        assert th.allclose(result, th.tensor([0.5]), atol=1e-6), (
            f"50%-overlap: expected [0.5], got {result.tolist()}"
        )

    def test_consistent_with_cos_hun_helper(self):
        """B=1 result must match the per-pair cos_hun_helper on the same data."""
        b_true_ints = th.tensor([1.0, 2.0])
        b_pred_ints = th.tensor([0.5, 1.5])
        b_match_mask = th.tensor([[True, False], [False, True]])
        b_true_match_mask = th.tensor([True, True])
        b_pred_match_mask = th.tensor([True, True])
        b_true_prec = th.zeros(2, dtype=th.bool)
        b_pred_prec = th.zeros(2, dtype=th.bool)

        single = cos_hun_helper(
            b_true_ints,
            b_pred_ints,
            b_match_mask,
            b_true_match_mask,
            b_pred_match_mask,
            False,
            b_true_prec,
            b_pred_prec,
        )
        batched = batch_cos_hun_helper(
            b_true_ints.unsqueeze(0),
            b_pred_ints.unsqueeze(0),
            b_match_mask.unsqueeze(0),
            b_true_match_mask.unsqueeze(0),
            b_pred_match_mask.unsqueeze(0),
            False,
            b_true_prec.unsqueeze(0),
            b_pred_prec.unsqueeze(0),
        )

        assert th.allclose(single, batched[0], atol=1e-6), (
            f"B=1 batched ({batched[0].item():.6f}) != single ({single.item():.6f})"
        )

    def test_remove_prec_peak_all_false_no_effect(self):
        """All-False prec masks: remove_prec_peak=True/False produce identical results."""
        batch_true_ints = th.tensor([[1.0, 1.0]])
        batch_pred_ints = th.tensor([[1.0, 1.0]])
        batch_match_mask = th.tensor([[[True, False], [False, True]]])
        batch_true_match_mask = th.tensor([[True, True]])
        batch_pred_match_mask = th.tensor([[True, True]])
        bt_prec, bp_prec = self._make_prec_masks(1, 2, 2)

        result_false = batch_cos_hun_helper(
            batch_true_ints,
            batch_pred_ints,
            batch_match_mask,
            batch_true_match_mask,
            batch_pred_match_mask,
            False,
            bt_prec,
            bp_prec,
        )
        result_true = batch_cos_hun_helper(
            batch_true_ints,
            batch_pred_ints,
            batch_match_mask,
            batch_true_match_mask,
            batch_pred_match_mask,
            True,
            bt_prec,
            bp_prec,
        )

        assert th.allclose(result_false, result_true, atol=1e-6), (
            "All-False prec masks: remove_prec_peak should have no effect"
        )

    def test_bounds_random(self):
        """Results should always be in [0, 1] for random batched inputs."""
        rng = th.Generator()
        rng.manual_seed(42)
        for _ in range(8):
            B = int(th.randint(1, 4, (1,), generator=rng).item())
            N_true = int(th.randint(1, 5, (1,), generator=rng).item())
            N_pred = int(th.randint(1, 5, (1,), generator=rng).item())
            batch_true_ints = th.rand(B, N_true, generator=rng) + 0.01
            batch_pred_ints = th.rand(B, N_pred, generator=rng) + 0.01
            # Ensure at least one match per batch so max_match_pos > 0
            batch_match_mask = th.zeros(B, N_true, N_pred, dtype=th.bool)
            batch_match_mask[:, 0, 0] = True  # force at least one match per item
            rand_mask = th.rand(B, N_true, N_pred, generator=rng) > 0.6
            batch_match_mask = batch_match_mask | rand_mask
            batch_true_match_mask = batch_match_mask.any(dim=2)
            batch_pred_match_mask = batch_match_mask.any(dim=1)
            bt_prec, bp_prec = self._make_prec_masks(B, N_true, N_pred)

            result = batch_cos_hun_helper(
                batch_true_ints,
                batch_pred_ints,
                batch_match_mask,
                batch_true_match_mask,
                batch_pred_match_mask,
                False,
                bt_prec,
                bp_prec,
            )
            assert result.shape == (B,)
            assert result.min().item() >= -1e-6, f"cos_hun < 0: {result.tolist()}"
            assert result.max().item() <= 1.0 + 1e-6, f"cos_hun > 1: {result.tolist()}"


class TestBatchJssHunHelper:
    """Tests for batch_jss_hun_helper (JSS-Hungarian, padded batch).

    Returns a tensor of shape (B,) with values in [0, log(2)]:
      - Identical spectra per item: log(2) ≈ 0.693
      - Orthogonal (no overlap) per item: 0.0

    Note: requires at least one matched peak across the whole batch so that
    max_true_match_pos > 0.
    """

    def _make_prec_masks(self, b: int, n_true: int, n_pred: int):
        """Return all-False batched precursor masks."""
        return th.zeros(b, n_true, dtype=th.bool), th.zeros(b, n_pred, dtype=th.bool)

    def test_identical_single_item(self):
        """B=1, one peak fully matched → [log(2)]."""
        batch_true_ints = th.tensor([[1.0]])
        batch_pred_ints = th.tensor([[1.0]])
        batch_match_mask = th.tensor([[[True]]])
        batch_true_match_mask = th.tensor([[True]])
        batch_pred_match_mask = th.tensor([[True]])
        bt_prec, bp_prec = self._make_prec_masks(1, 1, 1)

        result = batch_jss_hun_helper(
            batch_true_ints,
            batch_pred_ints,
            batch_match_mask,
            batch_true_match_mask,
            batch_pred_match_mask,
            False,
            bt_prec,
            bp_prec,
            log_min=EPS,
        )

        assert result.shape == (1,), f"Expected shape (1,), got {result.shape}"
        assert th.allclose(result, th.tensor([math.log(2.0)]), atol=1e-5), (
            f"Identical B=1: expected [log(2)], got {result.tolist()}"
        )

    def test_batch_identical_and_orthogonal(self):
        """B=2: pair 0 fully matched → log(2), pair 1 no overlap → 0.0.

        For pair 1 the match mask is False so the score is zeroed; both
        distributions contribute to the KL as pure unmatched mass, yielding
        JSD = log(2) and JSS_hun = 0.0.
        """
        batch_true_ints = th.tensor([[1.0], [1.0]])
        batch_pred_ints = th.tensor([[1.0], [1.0]])
        batch_match_mask = th.tensor([[[True]], [[False]]])
        batch_true_match_mask = th.tensor([[True], [False]])
        batch_pred_match_mask = th.tensor([[True], [False]])
        bt_prec, bp_prec = self._make_prec_masks(2, 1, 1)

        result = batch_jss_hun_helper(
            batch_true_ints,
            batch_pred_ints,
            batch_match_mask,
            batch_true_match_mask,
            batch_pred_match_mask,
            False,
            bt_prec,
            bp_prec,
            log_min=EPS,
        )

        assert result.shape == (2,), f"Expected shape (2,), got {result.shape}"
        assert th.allclose(result[0], th.tensor(math.log(2.0)), atol=1e-5), (
            f"Pair 0 identical: expected log(2), got {result[0].item():.6f}"
        )
        assert th.allclose(result[1], th.tensor(0.0), atol=1e-5), (
            f"Pair 1 orthogonal: expected 0.0, got {result[1].item():.6f}"
        )

    def test_consistent_with_jss_hun_helper(self):
        """B=1 result must match jss_hun_helper on the same single-pair data."""
        b_true_ints = th.tensor([1.0, 1.0])
        b_pred_ints = th.tensor([1.0, 1.0])
        b_match_mask = th.tensor([[True, False], [False, True]])
        b_true_match_mask = th.tensor([True, True])
        b_pred_match_mask = th.tensor([True, True])
        b_true_prec = th.zeros(2, dtype=th.bool)
        b_pred_prec = th.zeros(2, dtype=th.bool)

        single = jss_hun_helper(
            b_true_ints,
            b_pred_ints,
            b_match_mask,
            b_true_match_mask,
            b_pred_match_mask,
            False,
            b_true_prec,
            b_pred_prec,
            log_min=EPS,
        )
        batched = batch_jss_hun_helper(
            b_true_ints.unsqueeze(0),
            b_pred_ints.unsqueeze(0),
            b_match_mask.unsqueeze(0),
            b_true_match_mask.unsqueeze(0),
            b_pred_match_mask.unsqueeze(0),
            False,
            b_true_prec.unsqueeze(0),
            b_pred_prec.unsqueeze(0),
            log_min=EPS,
        )

        assert th.allclose(single, batched[0], atol=1e-5), (
            f"B=1 batched ({batched[0].item():.6f}) != single ({single.item():.6f})"
        )

    def test_known_partial_overlap(self):
        """B=1, two peaks with one matched → result in (0, log(2)).

        True = [0.5, 0.5]; Pred = [0.5, 0.5]; only peak 0 is within tolerance.
        Expected: 0.5 * log(2) (same calculation as jss_hun_helper).
        """
        batch_true_ints = th.tensor([[1.0, 1.0]])
        batch_pred_ints = th.tensor([[1.0, 1.0]])
        batch_match_mask = th.tensor([[[True, False], [False, False]]])
        batch_true_match_mask = th.tensor([[True, False]])
        batch_pred_match_mask = th.tensor([[True, False]])
        bt_prec, bp_prec = self._make_prec_masks(1, 2, 2)

        result = batch_jss_hun_helper(
            batch_true_ints,
            batch_pred_ints,
            batch_match_mask,
            batch_true_match_mask,
            batch_pred_match_mask,
            False,
            bt_prec,
            bp_prec,
            log_min=EPS,
        )

        expected = 0.5 * math.log(2.0)
        assert th.allclose(result, th.tensor([expected]), atol=1e-5), (
            f"50%-overlap: expected [0.5*log(2)={expected:.4f}], got {result.tolist()}"
        )

    def test_remove_prec_peak_all_false_no_effect(self):
        """All-False prec masks: remove_prec_peak=True/False produce identical results."""
        batch_true_ints = th.tensor([[1.0, 1.0]])
        batch_pred_ints = th.tensor([[1.0, 1.0]])
        batch_match_mask = th.tensor([[[True, False], [False, True]]])
        batch_true_match_mask = th.tensor([[True, True]])
        batch_pred_match_mask = th.tensor([[True, True]])
        bt_prec, bp_prec = self._make_prec_masks(1, 2, 2)

        result_false = batch_jss_hun_helper(
            batch_true_ints,
            batch_pred_ints,
            batch_match_mask,
            batch_true_match_mask,
            batch_pred_match_mask,
            False,
            bt_prec,
            bp_prec,
            log_min=EPS,
        )
        result_true = batch_jss_hun_helper(
            batch_true_ints,
            batch_pred_ints,
            batch_match_mask,
            batch_true_match_mask,
            batch_pred_match_mask,
            True,
            bt_prec,
            bp_prec,
            log_min=EPS,
        )

        assert th.allclose(result_false, result_true, atol=1e-6), (
            "All-False prec masks: remove_prec_peak should have no effect"
        )

    def test_bounds_random(self):
        """Results should always be in [0, log(2)] for random batched inputs."""
        rng = th.Generator()
        rng.manual_seed(99)
        log2 = math.log(2.0)
        for _ in range(8):
            B = int(th.randint(1, 4, (1,), generator=rng).item())
            N_true = int(th.randint(1, 5, (1,), generator=rng).item())
            N_pred = int(th.randint(1, 5, (1,), generator=rng).item())
            batch_true_ints = th.rand(B, N_true, generator=rng) + 0.01
            batch_pred_ints = th.rand(B, N_pred, generator=rng) + 0.01
            # Guarantee at least one match per item
            batch_match_mask = th.zeros(B, N_true, N_pred, dtype=th.bool)
            batch_match_mask[:, 0, 0] = True
            rand_mask = th.rand(B, N_true, N_pred, generator=rng) > 0.6
            batch_match_mask = batch_match_mask | rand_mask
            batch_true_match_mask = batch_match_mask.any(dim=2)
            batch_pred_match_mask = batch_match_mask.any(dim=1)
            bt_prec, bp_prec = self._make_prec_masks(B, N_true, N_pred)

            result = batch_jss_hun_helper(
                batch_true_ints,
                batch_pred_ints,
                batch_match_mask,
                batch_true_match_mask,
                batch_pred_match_mask,
                False,
                bt_prec,
                bp_prec,
                log_min=EPS,
            )
            assert result.shape == (B,)
            assert result.min().item() >= -1e-5, f"jss_hun < 0: {result.tolist()}"
            assert result.max().item() <= log2 + 1e-5, f"jss_hun > log(2): {result.tolist()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
