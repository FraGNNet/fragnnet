"""Unit tests for get_cross_jss_contrastive in fragnnet.model.loss."""

import math

import pytest
import torch as th

from fragnnet.model.loss import get_cross_jss_contrastive


def _make_spectrum(peaks: dict[float, float], batch_idx: int) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Build (mzs, logprobs, batch_idxs) from a {mz: intensity} dict."""
    mzs = th.tensor(list(peaks.keys()), dtype=th.float32)
    ints = th.tensor(list(peaks.values()), dtype=th.float32)
    logprobs = th.log(ints / ints.sum())
    batch_idxs = th.full((len(mzs),), batch_idx, dtype=th.long)
    return mzs, logprobs, batch_idxs


def _cat_spectra(spectra):
    """Concatenate list of (mzs, logprobs, batch_idxs) tuples."""
    return tuple(th.cat([s[i] for s in spectra]) for i in range(3))


MZ_MAX = 1500.0
MZ_BIN_RES = 0.01


class TestOutputShape:
    def test_shape_batch3(self):
        """Output is [B] for a batch of 3."""
        specs = [
            _make_spectrum({100.0: 0.8, 200.0: 0.2}, i) for i in range(3)
        ]
        true_mzs, true_lp, true_bi = _cat_spectra(specs)
        pred_mzs, pred_lp, pred_bi = true_mzs.clone(), true_lp.clone(), true_bi.clone()

        loss = get_cross_jss_contrastive(
            true_mzs, true_lp, true_bi,
            pred_mzs, pred_lp, pred_bi,
            batch_size=3, mz_max=MZ_MAX, mz_bin_res=MZ_BIN_RES,
        )
        assert loss.shape == (3,)

    def test_shape_batch1(self):
        """Batch of 1 returns shape [1]."""
        mzs = th.tensor([100.0, 200.0])
        lp = th.log(th.tensor([0.6, 0.4]))
        bi = th.tensor([0, 0])

        loss = get_cross_jss_contrastive(
            mzs, lp, bi, mzs, lp, bi,
            batch_size=1, mz_max=MZ_MAX, mz_bin_res=MZ_BIN_RES,
        )
        assert loss.shape == (1,)

    def test_empty_input_returns_zeros(self):
        """Empty spectra return zero loss without crashing."""
        empty = th.tensor([], dtype=th.float32)
        bi = th.tensor([], dtype=th.long)

        loss = get_cross_jss_contrastive(
            empty, empty, bi, empty, empty, bi,
            batch_size=3, mz_max=MZ_MAX, mz_bin_res=MZ_BIN_RES,
        )
        assert loss.shape == (3,)
        assert th.allclose(loss, th.zeros(3))


class TestLossValues:
    def test_batch1_is_zero(self):
        """With a single candidate, InfoNCE is trivially 0 (only correct answer)."""
        mzs = th.tensor([100.0, 200.0, 300.0])
        lp = th.log(th.tensor([0.5, 0.3, 0.2]))
        bi = th.zeros(3, dtype=th.long)

        loss = get_cross_jss_contrastive(
            mzs, lp, bi, mzs, lp, bi,
            batch_size=1, mz_max=MZ_MAX, mz_bin_res=MZ_BIN_RES,
        )
        assert loss[0].item() == pytest.approx(0.0, abs=1e-5)

    def test_perfect_prediction_lower_than_random(self):
        """Perfect pred (pred = true) has lower loss than a random pred."""
        spec_a = _make_spectrum({100.0: 0.7, 200.0: 0.3}, 0)
        spec_b = _make_spectrum({300.0: 0.6, 400.0: 0.4}, 1)
        true_mzs, true_lp, true_bi = _cat_spectra([spec_a, spec_b])

        # Perfect: pred == true
        loss_perfect = get_cross_jss_contrastive(
            true_mzs, true_lp, true_bi,
            true_mzs, true_lp.clone(), true_bi,
            batch_size=2, mz_max=MZ_MAX, mz_bin_res=MZ_BIN_RES,
        )

        # Random: pred is completely wrong molecule for both
        wrong_a = _make_spectrum({300.0: 0.6, 400.0: 0.4}, 0)
        wrong_b = _make_spectrum({100.0: 0.7, 200.0: 0.3}, 1)
        pred_mzs, pred_lp, pred_bi = _cat_spectra([wrong_a, wrong_b])
        loss_wrong = get_cross_jss_contrastive(
            true_mzs, true_lp, true_bi,
            pred_mzs, pred_lp, pred_bi,
            batch_size=2, mz_max=MZ_MAX, mz_bin_res=MZ_BIN_RES,
        )

        assert loss_perfect.mean() < loss_wrong.mean()

    def test_all_finite(self):
        """Loss values are always finite."""
        th.manual_seed(0)
        B = 5
        spectra = [
            _make_spectrum({float(100 + i * 50): 0.5, float(150 + i * 50): 0.5}, i)
            for i in range(B)
        ]
        true_mzs, true_lp, true_bi = _cat_spectra(spectra)
        pred_lp = true_lp + th.randn_like(true_lp) * 0.5

        loss = get_cross_jss_contrastive(
            true_mzs, true_lp, true_bi,
            true_mzs, pred_lp, true_bi,
            batch_size=B, mz_max=MZ_MAX, mz_bin_res=MZ_BIN_RES,
        )
        assert loss.isfinite().all()


class TestJSSMatrix:
    def test_diagonal_is_highest_for_identity_pred(self):
        """When pred_i = real_i for all i, loss equals the InfoNCE minimum for the batch size."""
        # Non-overlapping single-peak molecules: off-diagonal JSS = 0, diagonal JSS = 1
        spec_a = _make_spectrum({100.0: 1.0}, 0)
        spec_b = _make_spectrum({300.0: 1.0}, 1)
        spec_c = _make_spectrum({500.0: 1.0}, 2)
        true_mzs, true_lp, true_bi = _cat_spectra([spec_a, spec_b, spec_c])

        T, B = 1.0, 3
        loss = get_cross_jss_contrastive(
            true_mzs, true_lp, true_bi,
            true_mzs, true_lp.clone(), true_bi,
            batch_size=B, mz_max=MZ_MAX, mz_bin_res=MZ_BIN_RES,
            temperature=T,
        )
        # With JSS diagonal=1, off-diagonal=0:
        # InfoNCE = -1/T + log(exp(1/T) + (B-1)*1) = log(1 + (B-1)*exp(-1/T))
        import math
        expected = math.log(1 + (B - 1) * math.exp(-1.0 / T))
        assert loss.mean().item() == pytest.approx(expected, abs=1e-4)

    def test_cross_matrix_not_symmetric(self):
        """S_ij = JSS(real_i, pred_j) is generally not equal to S_ji = JSS(real_j, pred_i)."""
        spec_a = _make_spectrum({100.0: 0.9, 200.0: 0.1}, 0)
        spec_b = _make_spectrum({100.0: 0.3, 200.0: 0.7}, 1)
        true_mzs, true_lp, true_bi = _cat_spectra([spec_a, spec_b])

        # Use perturbed predictions so pred ≠ true
        th.manual_seed(1)
        pred_lp = true_lp + th.randn_like(true_lp) * 0.2

        # We're testing that the function doesn't silently symmetrise the matrix.
        # Two runs with swapped true/pred should give different losses.
        loss_fwd = get_cross_jss_contrastive(
            true_mzs, true_lp, true_bi,
            true_mzs, pred_lp, true_bi,
            batch_size=2, mz_max=MZ_MAX, mz_bin_res=MZ_BIN_RES,
        )
        loss_rev = get_cross_jss_contrastive(
            true_mzs, pred_lp, true_bi,
            true_mzs, true_lp, true_bi,
            batch_size=2, mz_max=MZ_MAX, mz_bin_res=MZ_BIN_RES,
        )
        # Should differ because the matrix rows and columns come from different distributions
        assert not th.allclose(loss_fwd, loss_rev, atol=1e-4)


class TestGradients:
    def test_gradient_flows_through_pred(self):
        """Gradient w.r.t. pred_logprobs is non-None and finite."""
        spec_a = _make_spectrum({100.0: 0.7, 200.0: 0.3}, 0)
        spec_b = _make_spectrum({300.0: 0.6, 400.0: 0.4}, 1)
        true_mzs, true_lp, true_bi = _cat_spectra([spec_a, spec_b])

        pred_lp = true_lp.clone().detach().requires_grad_(True)

        loss = get_cross_jss_contrastive(
            true_mzs, true_lp, true_bi,
            true_mzs, pred_lp, true_bi,
            batch_size=2, mz_max=MZ_MAX, mz_bin_res=MZ_BIN_RES,
        )
        loss.mean().backward()

        assert pred_lp.grad is not None
        assert pred_lp.grad.isfinite().all()
        assert pred_lp.grad.abs().sum() > 0.0

    def test_no_gradient_required_for_true(self):
        """True spectrum logprobs do not need requires_grad — function runs without error."""
        mzs = th.tensor([100.0, 200.0, 300.0, 400.0])
        lp = th.log(th.tensor([0.4, 0.3, 0.2, 0.1]))
        bi = th.tensor([0, 0, 1, 1])

        pred_lp = lp.clone().detach().requires_grad_(True)

        loss = get_cross_jss_contrastive(
            mzs, lp, bi, mzs, pred_lp, bi,
            batch_size=2, mz_max=MZ_MAX, mz_bin_res=MZ_BIN_RES,
        )
        loss.sum().backward()  # should not raise


class TestOptions:
    def test_sqrt_transform_changes_result(self):
        """sqrt_transform=True and False produce different JSS values for high-dynamic-range spectra.

        Requires overlapping molecules so that off-diagonal JSS is non-zero and
        sensitive to the intensity transform.  sqrt amplifies the minor shared
        peak (100 Da), changing JSS(real_A, pred_B) noticeably.
        """
        # Both molecules share peak at 100 Da; A has a distinctive minor peak at 200,
        # B has a distinctive minor peak at 300.  The dominant peak (100 Da) is shared.
        spec_a = _make_spectrum({100.0: 0.9, 200.0: 0.1}, 0)
        spec_b = _make_spectrum({100.0: 0.9, 300.0: 0.1}, 1)
        true_mzs, true_lp, true_bi = _cat_spectra([spec_a, spec_b])

        # Predictions deliberately wrong: A predicted like B and vice versa
        pred_a = _make_spectrum({100.0: 0.9, 300.0: 0.1}, 0)  # A's pred looks like B
        pred_b = _make_spectrum({100.0: 0.9, 200.0: 0.1}, 1)  # B's pred looks like A
        pred_mzs, pred_lp, pred_bi = _cat_spectra([pred_a, pred_b])

        loss_sqrt = get_cross_jss_contrastive(
            true_mzs, true_lp, true_bi,
            pred_mzs, pred_lp, pred_bi,
            batch_size=2, mz_max=MZ_MAX, mz_bin_res=MZ_BIN_RES,
            sqrt_transform=True,
        )
        loss_no_sqrt = get_cross_jss_contrastive(
            true_mzs, true_lp, true_bi,
            pred_mzs, pred_lp, pred_bi,
            batch_size=2, mz_max=MZ_MAX, mz_bin_res=MZ_BIN_RES,
            sqrt_transform=False,
        )
        # sqrt amplifies the minor distinctive peaks (10%) → more sensitive to
        # wrong-candidate predictions → losses must differ
        assert not th.allclose(loss_sqrt, loss_no_sqrt, atol=1e-4)

    def test_lower_temperature_sharpens_loss(self):
        """Lower temperature reduces loss for a good predictor (sharpens the correct answer).

        When diagonal >> off-diagonal (perfect predictions, non-overlapping molecules),
        low T → loss → 0, while high T → loss → log(B).  This verifies temperature
        acts as a sharpness knob on the InfoNCE softmax.
        """
        # Perfectly distinct molecules: off-diagonal JSS = 0, diagonal JSS = 1
        B = 4
        specs = [_make_spectrum({float(100 + i * 200): 1.0}, i) for i in range(B)]
        true_mzs, true_lp, true_bi = _cat_spectra(specs)

        loss_high_T = get_cross_jss_contrastive(
            true_mzs, true_lp, true_bi,
            true_mzs, true_lp.clone(), true_bi,
            batch_size=B, mz_max=MZ_MAX, mz_bin_res=MZ_BIN_RES,
            temperature=1.0,
        )
        loss_low_T = get_cross_jss_contrastive(
            true_mzs, true_lp, true_bi,
            true_mzs, true_lp.clone(), true_bi,
            batch_size=B, mz_max=MZ_MAX, mz_bin_res=MZ_BIN_RES,
            temperature=0.07,
        )
        # Lower temperature sharpens correct answer → loss drives toward 0 for perfect predictor
        assert loss_low_T.mean() < loss_high_T.mean()

    def test_non_overlapping_molecules_near_zero_loss(self):
        """Perfectly distinct molecules with perfect predictions give near-zero loss."""
        # Each molecule has unique peaks — wrong candidates score 0 JSS against real
        specs = [_make_spectrum({float(100 + i * 200): 1.0}, i) for i in range(4)]
        true_mzs, true_lp, true_bi = _cat_spectra(specs)

        loss = get_cross_jss_contrastive(
            true_mzs, true_lp, true_bi,
            true_mzs, true_lp.clone(), true_bi,
            batch_size=4, mz_max=MZ_MAX, mz_bin_res=MZ_BIN_RES,
            temperature=0.07,
        )
        # JSS diagonal = 1, off-diagonal = 0 → loss = -log(exp(1/T) / exp(1/T)) = 0
        assert loss.max().item() == pytest.approx(0.0, abs=1e-4)
