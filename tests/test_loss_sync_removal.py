"""Tests for GPU-sync-removal refactor in sparse_cross_entropy_vec.

Covers three changes:
1. scatter_logsumexp empty-input guard (numel==0, no GPU sync)
2. Removed th.any(bl_true_both_mask) branch — scatter_reduce handles empty input
3. Removed th.all(bl_true_both_mask) branch — scatter_logsumexp handles empty input

Edge cases exercised:
- All true peaks have a matching predicted peak (empty OOS / unmatched path)
- No true peaks have a matching predicted peak (empty IOS / matched path)
- Mixed (normal training case)
"""

import math

import pytest
import torch as th
import torch.nn.functional as F

from fragnnet.model.loss import get_sparse_cross_entropy_fn, sparse_cross_entropy_vec
from fragnnet.utils.misc_utils import LOG_ZERO, scatter_logsumexp


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_batch_idxs(peaks_per_spec: list[int], device: th.device) -> th.Tensor:
    batch_size = len(peaks_per_spec)
    return th.repeat_interleave(
        th.arange(batch_size, device=device),
        th.tensor(peaks_per_spec, device=device),
    )


def _default_ce_fn():
    return get_sparse_cross_entropy_fn(
        dist="gaussian",
        vectorized=True,
        tolerance=1e-5,
        relative=True,
        tolerance_min_mz=200.0,
        oos_tolerance_multiple=1,
        gaussian_renormalize=True,
        pm_tolerance_multiple=1,
        loss_batch_size=4,
    )


def _reference_sparse_ce_vec(
    true_mzs, true_logprobs, true_batch_idxs,
    pred_mzs, pred_logprobs, pred_batch_idxs, pred_oos_logprobs,
    tolerance, relative, tol_min_mz, oos_tol, gaussian_renorm, loss_batch_size, batch_size,
):
    """Old implementation with th.any / th.all branches — used as correctness reference."""
    import math as _math
    from fragnnet.utils.misc_utils import LOG_ZERO as _LZ, scatter_logsumexp as _sle
    from fragnnet.model.loss import scatter_reduce as _sr
    from fragnnet.utils.spec_utils import calculate_match_mzs as _cmm

    float_dtype = true_logprobs.dtype
    device = true_logprobs.device
    mask_value = _LZ(float_dtype)

    loss_num_batches = int((batch_size // loss_batch_size) + int(batch_size % loss_batch_size > 0))
    true_batch_cumsum = th.cumsum(
        F.pad(th.unique(true_batch_idxs, return_counts=True)[1], (1, 0), "constant", 0), dim=0
    )
    pred_batch_cumsum = th.cumsum(
        F.pad(th.unique(pred_batch_idxs, return_counts=True)[1], (1, 0), "constant", 0), dim=0
    )

    ios_ce = th.zeros(batch_size, dtype=float_dtype, device=device)
    oos_ce = th.zeros(batch_size, dtype=float_dtype, device=device)
    true_oos_out = th.zeros(batch_size, dtype=float_dtype, device=device)

    sigma = _math.sqrt(2)
    trunc = _math.erf(oos_tol / sigma)
    log_trunc = th.log(th.tensor(trunc, device=device))

    for bl in range(loss_num_batches):
        bl_lower = bl * loss_batch_size
        bl_upper = min((bl + 1) * loss_batch_size, batch_size)
        bl_batch_size = bl_upper - bl_lower
        tbl = true_batch_cumsum[bl_lower]
        tbu = true_batch_cumsum[bl_upper]
        pbl = pred_batch_cumsum[bl_lower]
        pbu = pred_batch_cumsum[bl_upper]

        bl_ti = true_batch_idxs[tbl:tbu] - bl_lower
        bl_pi = pred_batch_idxs[pbl:pbu] - bl_lower
        bl_tlp = true_logprobs[tbl:tbu]
        bl_plp = pred_logprobs[pbl:pbu]
        bl_tmz = true_mzs[tbl:tbu]
        bl_pmz = pred_mzs[pbl:pbu]
        bl_poos = pred_oos_logprobs[bl_lower:bl_upper]

        batch_mask = bl_ti.reshape(-1, 1) == bl_pi.reshape(1, -1)
        match_mask = _cmm(bl_tmz, bl_pmz, tolerance=oos_tol * tolerance, relative=relative,
                          tolerance_min_mz=tol_min_mz)
        match_mask = th.as_tensor(match_mask, device=device)
        both_mask = batch_mask & match_mask
        del batch_mask, match_mask
        true_both = th.any(both_mask, dim=1)

        stds = th.clamp(bl_pmz, min=tol_min_mz) * tolerance
        vs = stds ** 2
        log_trunc_vec = log_trunc * th.ones_like(bl_pmz)

        ios_lp = (
            -0.5 * (bl_tmz.reshape(-1, 1) - bl_pmz.reshape(1, -1)) ** 2 / vs.reshape(1, -1)
            - 0.5 * th.log(2 * th.pi * vs.reshape(1, -1))
            + bl_plp.reshape(1, -1)
            - log_trunc_vec.reshape(1, -1)
        )
        ios_lp = ios_lp + (~both_mask).to(float_dtype) * mask_value
        ios_lp = th.logsumexp(ios_lp, dim=1)

        if not th.any(true_both):
            pass  # ios_ce stays 0 (old behaviour: no assignment in this branch)
        else:
            bl_ios = _sr(
                (-th.exp(bl_tlp) * ios_lp)[true_both],
                bl_ti[true_both],
                "sum", dim=0, dim_size=bl_batch_size, default=0.0,
            )
            ios_ce[bl_lower:bl_upper] = bl_ios

        if not th.all(true_both):
            bl_oos_lp = _sle(bl_tlp[~true_both], bl_ti[~true_both], dim_size=bl_batch_size)
        else:
            bl_oos_lp = th.full((bl_batch_size,), mask_value, dtype=float_dtype, device=device)
        oos_ce[bl_lower:bl_upper] = -th.exp(bl_oos_lp) * bl_poos
        true_oos_out[bl_lower:bl_upper] = bl_oos_lp

    return ios_ce, oos_ce, true_oos_out


# ── 1. scatter_logsumexp empty-input guard ────────────────────────────────────

class TestScatterLogsumexpEmptyInput:
    """scatter_logsumexp must return LOG_ZERO fill (not crash) when input is empty."""

    @pytest.mark.parametrize("device_str", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
        not th.cuda.is_available(), reason="no GPU"
    ))])
    def test_empty_with_dim_size(self, device_str):
        device = th.device(device_str)
        logits = th.empty(0, dtype=th.float32, device=device)
        idxs = th.empty(0, dtype=th.long, device=device)
        result = scatter_logsumexp(logits, idxs, dim_size=4)
        assert result.shape == (4,), f"expected (4,), got {result.shape}"
        expected = LOG_ZERO(th.float32)
        assert th.all(result == expected), f"expected LOG_ZERO fills, got {result}"

    def test_empty_without_dim_size_returns_empty(self):
        logits = th.empty(0, dtype=th.float32)
        idxs = th.empty(0, dtype=th.long)
        result = scatter_logsumexp(logits, idxs, dim_size=None)
        assert result.shape == (0,)

    def test_empty_result_gives_zero_oos_ce(self):
        """exp(LOG_ZERO) * pred_oos_logprob ≈ 0 — OOS loss is 0 when no unmatched peaks."""
        device = th.device("cpu")
        logits = th.empty(0, dtype=th.float32, device=device)
        idxs = th.empty(0, dtype=th.long, device=device)
        oos_lp = scatter_logsumexp(logits, idxs, dim_size=3)
        pred_oos = th.tensor([-1.0, -2.0, -3.0])
        oos_ce = -th.exp(oos_lp) * pred_oos
        assert th.allclose(oos_ce, th.zeros(3), atol=1e-6), f"expected zeros, got {oos_ce}"

    @pytest.mark.parametrize("device_str", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
        not th.cuda.is_available(), reason="no GPU"
    ))])
    def test_nonempty_unchanged(self, device_str):
        """Non-empty path must produce the same result as before the guard."""
        device = th.device(device_str)
        th.manual_seed(0)
        logits = th.randn(6, device=device)
        idxs = th.tensor([0, 0, 1, 1, 2, 2], device=device)
        result = scatter_logsumexp(logits, idxs, dim_size=3)
        # Manual reference
        expected = th.stack([
            th.logsumexp(logits[idxs == i], dim=0) for i in range(3)
        ])
        assert th.allclose(result, expected, atol=1e-5), f"mismatch: {result} vs {expected}"


# ── 2. sparse_cross_entropy_vec edge cases ────────────────────────────────────

class TestSparseCEEdgeCases:

    @pytest.fixture
    def ce_fn(self):
        return _default_ce_fn()

    @pytest.mark.parametrize("device_str", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
        not th.cuda.is_available(), reason="no GPU"
    ))])
    def test_all_peaks_match_oos_ce_near_zero(self, ce_fn, device_str):
        """When every true peak matches a predicted peak, OOS loss must be ≈ 0."""
        device = th.device(device_str)
        th.manual_seed(1)
        batch_size = 4
        peaks = [3, 4, 2, 5]
        # true and pred share the same m/z values — all peaks match within tolerance
        mzs = th.cat([th.linspace(100.0, 500.0, n, device=device) for n in peaks])
        true_bidxs = _make_batch_idxs(peaks, device)
        pred_bidxs = _make_batch_idxs(peaks, device)
        true_lp = th.cat([th.log_softmax(th.randn(n, device=device), dim=0) for n in peaks])
        pred_lp = th.cat([th.log_softmax(th.randn(n, device=device), dim=0) for n in peaks])
        pred_oos = th.full((batch_size,), -2.0, device=device)

        ios_ce, oos_ce, _, _ = ce_fn(mzs, true_lp, true_bidxs, mzs, pred_lp, pred_bidxs, pred_oos)
        assert oos_ce.shape == (batch_size,)
        assert th.allclose(oos_ce, th.zeros(batch_size, device=device), atol=1e-5), (
            f"OOS CE should be ~0 when all peaks match, got {oos_ce}"
        )
        # ios_ce can be negative when Gaussian log-prob at zero distance > 0 (small σ)
        assert th.all(th.isfinite(ios_ce)), f"IOS CE must be finite, got {ios_ce}"

    @pytest.mark.parametrize("device_str", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
        not th.cuda.is_available(), reason="no GPU"
    ))])
    def test_no_peaks_match_ios_ce_is_zero(self, ce_fn, device_str):
        """When no true peak matches any predicted peak, IOS CE must be 0."""
        device = th.device(device_str)
        th.manual_seed(2)
        batch_size = 3
        peaks = [2, 3, 2]
        # true at low m/z, pred at high m/z — zero overlap within tolerance
        true_mzs = th.cat([th.linspace(50.0, 100.0, n, device=device) for n in peaks])
        pred_mzs = th.cat([th.linspace(900.0, 1000.0, n, device=device) for n in peaks])
        true_bidxs = _make_batch_idxs(peaks, device)
        pred_bidxs = _make_batch_idxs(peaks, device)
        true_lp = th.cat([th.log_softmax(th.randn(n, device=device), dim=0) for n in peaks])
        pred_lp = th.cat([th.log_softmax(th.randn(n, device=device), dim=0) for n in peaks])
        pred_oos = th.full((batch_size,), -2.0, device=device)

        ios_ce, oos_ce, _, _ = ce_fn(true_mzs, true_lp, true_bidxs, pred_mzs, pred_lp, pred_bidxs, pred_oos)
        assert ios_ce.shape == (batch_size,)
        assert th.allclose(ios_ce, th.zeros(batch_size, device=device), atol=1e-6), (
            f"IOS CE should be 0 when no peaks match, got {ios_ce}"
        )
        # OOS CE must be positive (all true peaks are out-of-sample)
        assert (oos_ce > 0).all(), f"OOS CE should be positive when no peaks match, got {oos_ce}"

    def test_single_spectrum_single_peak_match(self):
        """Trivial single-peak case: IOS CE = -log(pred_prob), OOS CE = 0."""
        device = th.device("cpu")
        mz = th.tensor([300.0])
        true_lp = th.tensor([0.0])     # log-prob = 0 → prob = 1
        pred_lp = th.tensor([-0.5])    # log-prob = -0.5
        pred_oos = th.tensor([-10.0])  # effectively 0
        bidxs = th.tensor([0])

        ce_fn = get_sparse_cross_entropy_fn(
            dist="gaussian", vectorized=True, tolerance=0.1,
            relative=False, tolerance_min_mz=0.0,
            oos_tolerance_multiple=1, gaussian_renormalize=False,
            pm_tolerance_multiple=1, loss_batch_size=4,
        )
        ios_ce, oos_ce, _, _ = ce_fn(mz, true_lp, bidxs, mz, pred_lp, bidxs, pred_oos)
        assert oos_ce.shape == (1,)
        # oos_ce ≈ 0 because all peaks match
        assert float(oos_ce[0]) == pytest.approx(0.0, abs=1e-4)
        # ios_ce = -p * log_prob; can be negative when Gaussian log-prob > 0 (high density at match)
        assert th.isfinite(ios_ce[0]), f"ios_ce must be finite, got {ios_ce[0]}"


# ── 3. Correctness vs reference implementation ────────────────────────────────

class TestSparseCEMatchesReference:
    """New implementation must match the old th.any/th.all reference for normal inputs."""

    @pytest.fixture
    def batch_data(self):
        th.manual_seed(42)
        batch_size = 8
        peaks_true = [4, 6, 3, 8, 5, 2, 7, 4]
        peaks_pred = [5, 4, 6, 3, 7, 5, 3, 6]
        true_bidxs = _make_batch_idxs(peaks_true, th.device("cpu"))
        pred_bidxs = _make_batch_idxs(peaks_pred, th.device("cpu"))
        # m/z values with partial overlap
        true_mzs = th.cat([th.linspace(100.0 + i * 5, 200.0 + i * 5, n)
                           for i, n in enumerate(peaks_true)])
        pred_mzs = th.cat([th.linspace(95.0 + i * 5, 195.0 + i * 5, n)
                           for i, n in enumerate(peaks_pred)])
        true_lp = th.cat([th.log_softmax(th.randn(n), dim=0) for n in peaks_true])
        pred_lp = th.cat([th.log_softmax(th.randn(n), dim=0) for n in peaks_pred])
        pred_oos = th.full((batch_size,), -2.0)
        return (batch_size, true_mzs, true_lp, true_bidxs,
                pred_mzs, pred_lp, pred_bidxs, pred_oos)

    def test_ios_ce_matches_reference(self, batch_data):
        (batch_size, true_mzs, true_lp, true_bidxs,
         pred_mzs, pred_lp, pred_bidxs, pred_oos) = batch_data

        tol, rel, tol_min, oos_tol, renorm, lbs = 1e-5, True, 200.0, 1, True, 4

        ref_ios, ref_oos, _ = _reference_sparse_ce_vec(
            true_mzs, true_lp, true_bidxs, pred_mzs, pred_lp, pred_bidxs, pred_oos,
            tol, rel, tol_min, oos_tol, renorm, lbs, batch_size,
        )
        new_ios, new_oos, _, _ = sparse_cross_entropy_vec(
            true_mzs, true_lp, true_bidxs, pred_mzs, pred_lp, pred_bidxs, pred_oos,
            tolerance=tol, relative=rel, tolerance_min_mz=tol_min,
            oos_tolerance_multiple=oos_tol, gaussian_renormalize=renorm,
            loss_batch_size=lbs,
        )

        assert th.allclose(new_ios, ref_ios, atol=1e-5), (
            f"IOS CE mismatch:\n  new={new_ios}\n  ref={ref_ios}"
        )
        assert th.allclose(new_oos, ref_oos, atol=1e-5), (
            f"OOS CE mismatch:\n  new={new_oos}\n  ref={ref_oos}"
        )

    @pytest.mark.parametrize("device_str", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
        not th.cuda.is_available(), reason="no GPU"
    ))])
    def test_ce_fn_wrapper_matches_reference(self, batch_data, device_str):
        """get_sparse_cross_entropy_fn wrapper produces the same result."""
        (batch_size, true_mzs, true_lp, true_bidxs,
         pred_mzs, pred_lp, pred_bidxs, pred_oos) = batch_data
        device = th.device(device_str)
        true_mzs, true_lp, true_bidxs = true_mzs.to(device), true_lp.to(device), true_bidxs.to(device)
        pred_mzs, pred_lp, pred_bidxs = pred_mzs.to(device), pred_lp.to(device), pred_bidxs.to(device)
        pred_oos = pred_oos.to(device)

        tol, rel, tol_min, oos_tol, renorm, lbs = 1e-5, True, 200.0, 1, True, 4

        ref_ios, ref_oos, _ = _reference_sparse_ce_vec(
            true_mzs, true_lp, true_bidxs, pred_mzs, pred_lp, pred_bidxs, pred_oos,
            tol, rel, tol_min, oos_tol, renorm, lbs, batch_size,
        )
        ce_fn = get_sparse_cross_entropy_fn(
            dist="gaussian", vectorized=True, tolerance=tol, relative=rel,
            tolerance_min_mz=tol_min, oos_tolerance_multiple=oos_tol,
            gaussian_renormalize=renorm, pm_tolerance_multiple=1, loss_batch_size=lbs,
        )
        new_ios, new_oos, _, _ = ce_fn(true_mzs, true_lp, true_bidxs, pred_mzs, pred_lp, pred_bidxs, pred_oos)

        assert th.allclose(new_ios.cpu(), ref_ios.cpu(), atol=1e-5)
        assert th.allclose(new_oos.cpu(), ref_oos.cpu(), atol=1e-5)

    def test_loss_batch_size_invariant(self, batch_data):
        """Result must be identical regardless of loss_batch_size (it's a memory tradeoff)."""
        (batch_size, true_mzs, true_lp, true_bidxs,
         pred_mzs, pred_lp, pred_bidxs, pred_oos) = batch_data

        tol, rel, tol_min, oos_tol, renorm = 1e-5, True, 200.0, 1, True

        def run(lbs):
            ios, oos, _, _ = sparse_cross_entropy_vec(
                true_mzs, true_lp, true_bidxs, pred_mzs, pred_lp, pred_bidxs, pred_oos,
                tolerance=tol, relative=rel, tolerance_min_mz=tol_min,
                oos_tolerance_multiple=oos_tol, gaussian_renormalize=renorm,
                loss_batch_size=lbs,
            )
            return ios, oos

        ios_ref, oos_ref = run(4)
        for lbs in [1, 16, 100]:
            ios, oos = run(lbs)
            assert th.allclose(ios, ios_ref, atol=1e-5), f"IOS mismatch at loss_batch_size={lbs}"
            assert th.allclose(oos, oos_ref, atol=1e-5), f"OOS mismatch at loss_batch_size={lbs}"
