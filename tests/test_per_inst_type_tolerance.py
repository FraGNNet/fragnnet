"""Unit tests for per-instrument-type mass tolerance in loss functions.

Tests cover:
- calculate_match_mzs: tol_per_true and min_mz_per_true override scalar tolerance
- sparse_cross_entropy_vec: tol_per_sample / min_mz_per_sample pass-through
- sparse_cosine_distance_hungarian: tol_per_sample pass-through
- sparse_jensen_shannon_divergence_hungarian: tol_per_sample pass-through
- inst_tol_map construction: sorted index assignment and global fallback
- Mixed FT+QTOF batch: per-sample independence in cosine distance and CE
- QTOF truncation scenario: floor-truncated m/z that only matches with wider min_mz floor
"""

import math
from types import SimpleNamespace

import pytest
import torch as th

from fragnnet.model.loss import (
    sparse_cosine_distance_hungarian,
    sparse_cross_entropy_vec,
    sparse_jensen_shannon_divergence_hungarian,
)
from fragnnet.pl_model.spectrum_pl import SpectrumPL
from fragnnet.utils.spec_utils import calculate_match_mzs

# ---------------------------------------------------------------------------
# calculate_match_mzs — tol_per_true / min_mz_per_true
# ---------------------------------------------------------------------------


class TestCalculateMatchMzsPerTrue:
    """Tests for per-true-peak tolerance in calculate_match_mzs."""

    def test_uniform_tol_per_true_matches_scalar(self):
        """When all tol_per_true entries equal the scalar tolerance, results are identical."""
        true_mzs = th.tensor([300.0, 500.0])
        pred_mzs = th.tensor([300.005, 500.008])

        scalar = calculate_match_mzs(
            true_mzs, pred_mzs, tolerance=1e-5, relative=True, tolerance_min_mz=200.0
        )
        per_true = calculate_match_mzs(
            true_mzs,
            pred_mzs,
            tolerance=1e-5,
            relative=True,
            tolerance_min_mz=200.0,
            tol_per_true=th.full((2,), 1e-5),
            min_mz_per_true=th.full((2,), 200.0),
        )
        assert th.equal(scalar, per_true)

    def test_different_tolerances_per_peak(self):
        """Tight vs wide tolerance on two peaks from the same pred_mz: only the wide one matches.

        At m/z 500 with diff = 0.008 Da:
          - 10 ppm (1e-5): 0.008/500 = 16 ppm  → no match
          - 20 ppm (2e-5): 0.008/500 = 16 ppm  < 20 ppm → match
        """
        true_mzs = th.tensor([500.0, 500.0])
        pred_mzs = th.tensor([500.008])

        result = calculate_match_mzs(
            true_mzs,
            pred_mzs,
            tolerance=1e-5,  # ignored when tol_per_true provided
            relative=True,
            tolerance_min_mz=200.0,
            tol_per_true=th.tensor([1e-5, 2e-5]),
            min_mz_per_true=th.tensor([200.0, 200.0]),
        )
        assert result.shape == (2, 1)
        assert not result[0, 0].item(), "10 ppm should NOT match 16 ppm diff"
        assert result[1, 0].item(), "20 ppm should match 16 ppm diff"

    def test_min_mz_per_true_floor_controls_match(self):
        """Per-true-peak min_mz floor: larger floor → larger absolute tolerance → more matches.

        true peak at m/z 50 (below global min_mz=200), pred at m/z 50 + 0.008 Da.
          - min_mz=200:  floor = 200 * 1e-5 = 0.002 Da — 0.008 > 0.002 → no match
          - min_mz=1000: floor = 1000 * 1e-5 = 0.010 Da — 0.008 < 0.010 → match
        """
        true_mzs = th.tensor([50.0, 50.0])
        pred_mzs = th.tensor([50.008])

        result = calculate_match_mzs(
            true_mzs,
            pred_mzs,
            tolerance=1e-5,
            relative=True,
            tolerance_min_mz=200.0,
            tol_per_true=th.tensor([1e-5, 1e-5]),
            min_mz_per_true=th.tensor([200.0, 1000.0]),
        )
        assert result.shape == (2, 1)
        assert not result[0, 0].item(), (
            "min_mz=200 gives 0.002 Da floor; 0.008 Da > floor → no match"
        )
        assert result[1, 0].item(), "min_mz=1000 gives 0.010 Da floor; 0.008 Da < floor → match"

    def test_qtof_truncation_scenario(self):
        """QTOF floor-truncated m/z: true peak stored as floor(500.007) = 500.000.

        Theoretical formula m/z = 500.007; truncation error = 0.007 Da.
          - FT config  (10 ppm, min_mz=200):  floor = 200*1e-5 = 0.002 Da → 0.007 > 0.002 → NO match
          - QTOF config (20 ppm, min_mz=750): floor = 750*2e-5 = 0.015 Da → 0.007 < 0.015 → MATCH
        """
        true_mzs = th.tensor([500.000, 500.000])  # both floor-truncated from 500.007
        pred_mzs = th.tensor([500.007])  # theoretical formula m/z

        result = calculate_match_mzs(
            true_mzs,
            pred_mzs,
            tolerance=1e-5,
            relative=True,
            tolerance_min_mz=200.0,
            tol_per_true=th.tensor([1e-5, 2e-5]),  # FT, QTOF
            min_mz_per_true=th.tensor([200.0, 750.0]),
        )
        assert result.shape == (2, 1)
        assert not result[0, 0].item(), (
            "FT (10 ppm, min_mz=200): 0.007 Da > 0.002 Da floor → no match"
        )
        assert result[1, 0].item(), "QTOF (20 ppm, min_mz=750): 0.007 Da < 0.015 Da floor → match"

    def test_tol_per_true_without_min_mz_uses_global_floor(self):
        """When only tol_per_true is given (no min_mz_per_true), global tolerance_min_mz is used."""
        true_mzs = th.tensor([500.0])
        pred_mzs = th.tensor([500.008])  # 16 ppm at m/z 500

        # 20 ppm with global min_mz=200 → floor=0.002 Da → 0.008/500=16ppm < 20ppm → match
        result = calculate_match_mzs(
            true_mzs,
            pred_mzs,
            tolerance=1e-5,
            relative=True,
            tolerance_min_mz=200.0,
            tol_per_true=th.tensor([2e-5]),
            min_mz_per_true=None,
        )
        assert result[0, 0].item(), "20 ppm should match 16 ppm diff"


# ---------------------------------------------------------------------------
# sparse_cross_entropy_vec — tol_per_sample / min_mz_per_sample
# ---------------------------------------------------------------------------


class TestSparseCrossEntropyVecPerSample:
    """Tests for per-sample tolerance in sparse_cross_entropy_vec."""

    def _make_single_peak_batch(self, true_mz: float, pred_mz: float):
        """Build a 1-sample, 1-peak batch for CE testing."""
        true_mzs = th.tensor([true_mz])
        true_logprobs = th.tensor([0.0])  # log(1.0)
        true_batch_idxs = th.tensor([0], dtype=th.long)
        pred_mzs = th.tensor([pred_mz])
        pred_logprobs = th.tensor([0.0])
        pred_batch_idxs = th.tensor([0], dtype=th.long)
        pred_oos_logprobs = th.tensor([math.log(1e-10)])
        return (
            true_mzs,
            true_logprobs,
            true_batch_idxs,
            pred_mzs,
            pred_logprobs,
            pred_batch_idxs,
            pred_oos_logprobs,
        )

    def test_uniform_tol_per_sample_matches_scalar(self):
        """Uniform tol_per_sample gives identical IOS CE as scalar tolerance."""
        args = self._make_single_peak_batch(500.0, 500.003)  # 6 ppm — within 10 ppm
        scalar_ios, _, _, _ = sparse_cross_entropy_vec(
            *args,
            tolerance=1e-5,
            relative=True,
            tolerance_min_mz=200.0,
            oos_tolerance_multiple=3.0,
            gaussian_renormalize=True,
            loss_batch_size=32,
        )
        per_sample_ios, _, _, _ = sparse_cross_entropy_vec(
            *args,
            tolerance=1e-5,
            relative=True,
            tolerance_min_mz=200.0,
            oos_tolerance_multiple=3.0,
            gaussian_renormalize=True,
            loss_batch_size=32,
            tol_per_sample=th.tensor([1e-5]),
            min_mz_per_sample=th.tensor([200.0]),
        )
        assert th.allclose(scalar_ios, per_sample_ios, atol=1e-6)

    def test_wide_tol_recovers_oos_peak(self):
        """A peak outside tight tolerance (OOS) becomes IOS when tolerance is widened.

        true m/z = 200.000 (at the min_mz floor), pred m/z = 200.007 (0.007 Da off).
          - tight (10 ppm, min_mz=200): std = max(200,200)*1e-5 = 0.002 Da, 3σ = 0.006 Da
            → 0.007 > 0.006 → OOS → IOS CE = 0
          - wide  (20 ppm, min_mz=750): std = max(200,750)*2e-5 = 0.015 Da, 3σ = 0.045 Da
            → 0.007 < 0.045 → IOS → IOS CE > 0
        """
        true_mz, pred_mz = 200.000, 200.007
        args = self._make_single_peak_batch(true_mz, pred_mz)

        ios_tight, _, _, _ = sparse_cross_entropy_vec(
            *args,
            tolerance=1e-5,
            relative=True,
            tolerance_min_mz=200.0,
            oos_tolerance_multiple=3.0,
            gaussian_renormalize=True,
            loss_batch_size=32,
        )
        ios_wide, _, _, _ = sparse_cross_entropy_vec(
            *args,
            tolerance=1e-5,
            relative=True,
            tolerance_min_mz=200.0,
            oos_tolerance_multiple=3.0,
            gaussian_renormalize=True,
            loss_batch_size=32,
            tol_per_sample=th.tensor([2e-5]),
            min_mz_per_sample=th.tensor([750.0]),
        )
        # Tight: peak is OOS → IOS CE = 0 (no in-support pairs contribute)
        assert ios_tight.item() == pytest.approx(0.0, abs=1e-6), (
            "Tight tolerance: peak is OOS, IOS CE should be ~0"
        )
        # Wide: peak is IOS → IOS CE is a finite negative log-probability
        assert ios_wide.item() < 0.0, "Wide tolerance: peak is IOS, IOS log-prob should be < 0"


class TestSpectrumPLEvalTensorGating:
    """Tests for the validation eval-tensor gate in SpectrumPL."""

    def _fake_pl(self, active_metric_names, auxiliary_metric_names, raw_epoch=False):
        return SimpleNamespace(
            auxiliary_metric_names=set(auxiliary_metric_names),
            _get_active_metric_names=lambda split: set(active_metric_names),
            _should_store_raw_epoch_results=lambda split: raw_epoch,
        )

    def test_ce_metadata_alone_does_not_require_eval_tensors(self):
        """CE metadata logging should not trigger eval spectrum materialization by itself."""
        fake = self._fake_pl(active_metric_names={"loss"}, auxiliary_metric_names=set())
        assert not SpectrumPL._needs_eval_tensors(fake, "val")

    def test_auxiliary_metrics_require_eval_tensors(self):
        """Auxiliary validation metrics still require eval spectra."""
        fake = self._fake_pl(active_metric_names={"loss", "cos_sim_0.01"}, auxiliary_metric_names={"cos_sim_0.01"})
        assert SpectrumPL._needs_eval_tensors(fake, "val")

    def test_raw_epoch_results_require_eval_tensors(self):
        """Image/raw-spectrum logging still requires eval spectra."""
        fake = self._fake_pl(active_metric_names={"loss"}, auxiliary_metric_names=set(), raw_epoch=True)
        assert SpectrumPL._needs_eval_tensors(fake, "val")


# ---------------------------------------------------------------------------
# sparse_cosine_distance_hungarian — tol_per_sample / min_mz_per_sample
# ---------------------------------------------------------------------------


class TestSparseCosineHungarianPerSample:
    """Tests for per-sample tolerance in sparse_cosine_distance_hungarian."""

    def test_uniform_tol_per_sample_matches_scalar(self):
        """Uniform tol_per_sample gives identical cosine distance as scalar tolerance."""
        mzs = th.tensor([100.0, 200.0, 100.0, 200.0])
        logprobs = th.tensor([-0.693, -0.693, -0.693, -0.693])  # log(0.5) each
        batch_idxs = th.tensor([0, 0, 1, 1], dtype=th.long)

        scalar = sparse_cosine_distance_hungarian(
            mzs,
            logprobs,
            batch_idxs,
            mzs,
            logprobs,
            batch_idxs,
            tolerance=1e-5,
            relative=True,
            tolerance_min_mz=200.0,
        )
        per_sample = sparse_cosine_distance_hungarian(
            mzs,
            logprobs,
            batch_idxs,
            mzs,
            logprobs,
            batch_idxs,
            tolerance=1e-5,
            relative=True,
            tolerance_min_mz=200.0,
            tol_per_sample=th.tensor([1e-5, 1e-5]),
            min_mz_per_sample=th.tensor([200.0, 200.0]),
        )
        assert th.allclose(scalar, per_sample, atol=1e-6)

    def test_wide_tolerance_matches_truncated_peak(self):
        """A predicted peak that misses with tight tolerance matches with QTOF-wide tolerance.

        true m/z = 500.000 (truncated), pred m/z = 500.007 (theoretical).
        With tight FT tolerance the spectra appear orthogonal (cosine distance = 1.0).
        With wide QTOF tolerance the peak is matched (cosine distance = 0.0 for identical spectra).
        """
        true_mzs = th.tensor([500.000])
        pred_mzs = th.tensor([500.007])
        logprobs = th.tensor([0.0])
        batch_idxs = th.tensor([0], dtype=th.long)

        dist_tight = sparse_cosine_distance_hungarian(
            true_mzs,
            logprobs,
            batch_idxs,
            pred_mzs,
            logprobs,
            batch_idxs,
            tolerance=1e-5,
            relative=True,
            tolerance_min_mz=200.0,
        )
        dist_wide = sparse_cosine_distance_hungarian(
            true_mzs,
            logprobs,
            batch_idxs,
            pred_mzs,
            logprobs,
            batch_idxs,
            tolerance=1e-5,
            relative=True,
            tolerance_min_mz=200.0,
            tol_per_sample=th.tensor([2e-5]),
            min_mz_per_sample=th.tensor([750.0]),
        )
        # Tight: no match → cosine distance = 1 (fully orthogonal)
        assert dist_tight.item() == pytest.approx(1.0, abs=1e-5), (
            "Tight tolerance: unmatched peak → cosine distance should be 1.0"
        )
        # Wide: peaks match → cosine distance ≈ 0
        assert dist_wide.item() == pytest.approx(0.0, abs=1e-5), (
            "Wide tolerance: matched peak → cosine distance should be ~0.0"
        )


# ---------------------------------------------------------------------------
# sparse_jensen_shannon_divergence_hungarian — tol_per_sample / min_mz_per_sample
# ---------------------------------------------------------------------------


class TestSparseJSDHungarianPerSample:
    """Tests for per-sample tolerance in sparse_jensen_shannon_divergence_hungarian."""

    def test_uniform_tol_per_sample_matches_scalar(self):
        """Uniform tol_per_sample gives identical JSD as scalar tolerance."""
        mzs = th.tensor([100.0, 200.0, 100.0, 200.0])
        logprobs = th.tensor([-0.693, -0.693, -0.693, -0.693])
        batch_idxs = th.tensor([0, 0, 1, 1], dtype=th.long)

        scalar = sparse_jensen_shannon_divergence_hungarian(
            mzs, logprobs, batch_idxs,
            mzs, logprobs, batch_idxs,
            tolerance=1e-5, relative=True, tolerance_min_mz=200.0,
        )
        per_sample = sparse_jensen_shannon_divergence_hungarian(
            mzs, logprobs, batch_idxs,
            mzs, logprobs, batch_idxs,
            tolerance=1e-5, relative=True, tolerance_min_mz=200.0,
            tol_per_sample=th.tensor([1e-5, 1e-5]),
            min_mz_per_sample=th.tensor([200.0, 200.0]),
        )
        assert th.allclose(scalar, per_sample, atol=1e-6)

    def test_wide_tolerance_matches_truncated_peak(self):
        """QTOF-wide tolerance matches a floor-truncated peak; FT-tight tolerance misses it.

        true m/z = 500.000 (truncated), pred m/z = 500.007 (theoretical), diff = 0.007 Da.
          - tight (10 ppm, min_mz=200): floor = 0.002 Da → no match → JSD = max (1.0)
          - wide  (20 ppm, min_mz=750): floor = 0.015 Da → match → JSD = 0.0
        """
        true_mzs = th.tensor([500.000])
        pred_mzs = th.tensor([500.007])
        logprobs = th.tensor([0.0])
        batch_idxs = th.tensor([0], dtype=th.long)

        jsd_tight = sparse_jensen_shannon_divergence_hungarian(
            true_mzs, logprobs, batch_idxs,
            pred_mzs, logprobs, batch_idxs,
            tolerance=1e-5, relative=True, tolerance_min_mz=200.0,
        )
        jsd_wide = sparse_jensen_shannon_divergence_hungarian(
            true_mzs, logprobs, batch_idxs,
            pred_mzs, logprobs, batch_idxs,
            tolerance=1e-5, relative=True, tolerance_min_mz=200.0,
            tol_per_sample=th.tensor([2e-5]),
            min_mz_per_sample=th.tensor([750.0]),
        )
        assert jsd_tight.item() > jsd_wide.item(), (
            "Tight tolerance misses peak → higher JSD than wide tolerance"
        )
        assert jsd_wide.item() == pytest.approx(0.0, abs=1e-5), (
            "Wide tolerance: identical matched spectra → JSD = 0.0"
        )


# ---------------------------------------------------------------------------
# Mixed FT+QTOF batch — per-sample independence
# ---------------------------------------------------------------------------


class TestMixedBatchPerSampleCosine:
    """Verify per-sample independence in a 2-sample FT+QTOF batch (cosine distance).

    Both samples have the QTOF truncation scenario (true=500.000, pred=500.007).
    FT sample (tight) should get distance=1.0; QTOF sample (wide) should get 0.0.
    """

    def _make_batch(self):
        true_mzs = th.tensor([500.000, 500.000])
        pred_mzs = th.tensor([500.007, 500.007])
        logprobs = th.tensor([0.0, 0.0])
        batch_idxs = th.tensor([0, 1], dtype=th.long)
        return true_mzs, logprobs, batch_idxs, pred_mzs, logprobs, batch_idxs

    def test_ft_qtof_mixed_batch(self):
        """FT sample misses (tight); QTOF sample hits (wide) — verified per-sample."""
        args = self._make_batch()
        dists = sparse_cosine_distance_hungarian(
            *args,
            tolerance=1e-5, relative=True, tolerance_min_mz=200.0,
            tol_per_sample=th.tensor([1e-5, 2e-5]),
            min_mz_per_sample=th.tensor([200.0, 750.0]),
        )
        assert dists[0].item() == pytest.approx(1.0, abs=1e-5), (
            "FT sample (tight): no match → cosine distance = 1.0"
        )
        assert dists[1].item() == pytest.approx(0.0, abs=1e-5), (
            "QTOF sample (wide): match → cosine distance = 0.0"
        )

    def test_global_tight_both_miss(self):
        """Without per-sample tol, both samples use tight global tolerance and both miss."""
        args = self._make_batch()
        dists = sparse_cosine_distance_hungarian(
            *args,
            tolerance=1e-5, relative=True, tolerance_min_mz=200.0,
        )
        assert dists[0].item() == pytest.approx(1.0, abs=1e-5)
        assert dists[1].item() == pytest.approx(1.0, abs=1e-5)


class TestMixedBatchPerSampleCE:
    """Verify per-sample independence in a 2-sample FT+QTOF batch (cross-entropy).

    Both samples use true=200.000, pred=200.007 (0.007 Da, at the min_mz floor).
      - FT  (10 ppm, min_mz=200): std=0.002 Da, 3σ=0.006 Da → OOS → IOS CE = 0
      - QTOF (20 ppm, min_mz=750): std=0.015 Da, 3σ=0.045 Da → IOS → IOS CE < 0
    """

    _CE_KWARGS = dict(
        tolerance=1e-5,
        relative=True,
        tolerance_min_mz=200.0,
        oos_tolerance_multiple=3.0,
        gaussian_renormalize=True,
        loss_batch_size=32,
    )

    def _make_batch(self, true_mz: float, pred_mz: float, n_samples: int):
        true_mzs = th.tensor([true_mz] * n_samples)
        true_logprobs = th.tensor([0.0] * n_samples)
        true_batch_idxs = th.arange(n_samples, dtype=th.long)
        pred_mzs = th.tensor([pred_mz] * n_samples)
        pred_logprobs = th.tensor([0.0] * n_samples)
        pred_batch_idxs = th.arange(n_samples, dtype=th.long)
        pred_oos_logprobs = th.tensor([math.log(1e-10)] * n_samples)
        return (
            true_mzs, true_logprobs, true_batch_idxs,
            pred_mzs, pred_logprobs, pred_batch_idxs, pred_oos_logprobs,
        )

    def test_both_ft_both_oos(self):
        """Both samples as FT (tight): both OOS → per-sample IOS CE = 0."""
        args = self._make_batch(200.0, 200.007, n_samples=2)
        ios, _, _, _ = sparse_cross_entropy_vec(
            *args, **self._CE_KWARGS,
            tol_per_sample=th.tensor([1e-5, 1e-5]),
            min_mz_per_sample=th.tensor([200.0, 200.0]),
        )
        # CE returns per-sample tensor [n_samples]
        assert ios[0].item() == pytest.approx(0.0, abs=1e-6), "FT sample 0: OOS → IOS CE = 0"
        assert ios[1].item() == pytest.approx(0.0, abs=1e-6), "FT sample 1: OOS → IOS CE = 0"

    def test_both_qtof_both_ios(self):
        """Both samples as QTOF (wide): both IOS → per-sample IOS CE < 0."""
        args = self._make_batch(200.0, 200.007, n_samples=2)
        ios, _, _, _ = sparse_cross_entropy_vec(
            *args, **self._CE_KWARGS,
            tol_per_sample=th.tensor([2e-5, 2e-5]),
            min_mz_per_sample=th.tensor([750.0, 750.0]),
        )
        assert ios[0].item() < 0.0, "QTOF sample 0: IOS → log-prob < 0"
        assert ios[1].item() < 0.0, "QTOF sample 1: IOS → log-prob < 0"

    def test_mixed_ft_qtof_per_sample_independence(self):
        """FT sample is OOS (CE=0); QTOF sample is IOS (CE<0) — verified per sample."""
        args = self._make_batch(200.0, 200.007, n_samples=2)
        ios, _, _, _ = sparse_cross_entropy_vec(
            *args, **self._CE_KWARGS,
            tol_per_sample=th.tensor([1e-5, 2e-5]),
            min_mz_per_sample=th.tensor([200.0, 750.0]),
        )
        assert ios[0].item() == pytest.approx(0.0, abs=1e-6), (
            "FT sample (idx=0): OOS → IOS CE = 0"
        )
        assert ios[1].item() < 0.0, "QTOF sample (idx=1): IOS → log-prob < 0"

    def test_mixed_ft_qtof_qtof_matches_single(self):
        """QTOF sample in a mixed batch gives the same IOS CE as a solo QTOF sample."""
        args_single = self._make_batch(200.0, 200.007, n_samples=1)
        ios_single, _, _, _ = sparse_cross_entropy_vec(
            *args_single, **self._CE_KWARGS,
            tol_per_sample=th.tensor([2e-5]),
            min_mz_per_sample=th.tensor([750.0]),
        )
        args_mixed = self._make_batch(200.0, 200.007, n_samples=2)
        ios_mixed, _, _, _ = sparse_cross_entropy_vec(
            *args_mixed, **self._CE_KWARGS,
            tol_per_sample=th.tensor([1e-5, 2e-5]),
            min_mz_per_sample=th.tensor([200.0, 750.0]),
        )
        assert ios_mixed[1].item() == pytest.approx(ios_single[0].item(), abs=1e-5), (
            "QTOF result is identical whether batched with FT or run alone"
        )


# ---------------------------------------------------------------------------
# inst_tol_map construction — index assignment and global fallback
# ---------------------------------------------------------------------------


class TestInstTolMapConstruction:
    """Tests for the inst_tol_lookup / inst_min_mz_lookup building logic in SpectrumPL._setup_tolerance.

    Lookup tensors are flat lists indexed by the 0-based position of each instrument type in
    sorted(spec_params["inst_types"]). Unmapped types fall back to the global scalar.
    """

    def _build_lookup(
        self,
        inst_types: list[str],
        inst_type_loss_tol: dict,
        global_tol: float = 1e-5,
        global_min_mz: float = 200.0,
    ) -> tuple[list[float], list[float]]:
        """Replicate the inst_tol_lookup / inst_min_mz_lookup construction from _setup_tolerance."""
        inst_types_sorted = sorted(inst_types)
        tols, min_mzs = [], []
        for inst_type in inst_types_sorted:
            if inst_type in inst_type_loss_tol:
                cfg = inst_type_loss_tol[inst_type]
                tols.append(float(cfg["rel"]))
                min_mzs.append(float(cfg["min_mz"]))
            else:
                tols.append(global_tol)
                min_mzs.append(global_min_mz)
        return tols, min_mzs

    def test_ft_qtof_it_standard_mapping(self):
        """sorted(["FT","QTOF","IT"]) = ["FT","IT","QTOF"] → FT=0, IT=1, QTOF=2."""
        tols, min_mzs = self._build_lookup(
            ["FT", "QTOF", "IT"],
            {"FT": {"rel": 1e-5, "min_mz": 200.0}, "QTOF": {"rel": 2e-5, "min_mz": 750.0}},
        )
        assert len(tols) == 3
        assert (tols[0], min_mzs[0]) == (1e-5, 200.0), "FT idx=0"
        assert (tols[1], min_mzs[1]) == (1e-5, 200.0), "IT idx=1 → global fallback"
        assert (tols[2], min_mzs[2]) == (2e-5, 750.0), "QTOF idx=2"

    def test_unmapped_instrument_uses_global(self):
        """An instrument not in inst_type_loss_tol inherits the global tolerance."""
        tols, min_mzs = self._build_lookup(
            ["FT", "QTOF", "IT"],
            {"QTOF": {"rel": 2e-5, "min_mz": 750.0}},
            global_tol=1e-5,
            global_min_mz=300.0,
        )
        assert (tols[0], min_mzs[0]) == (1e-5, 300.0), "FT unmapped → global (1e-5, 300.0)"
        assert (tols[1], min_mzs[1]) == (1e-5, 300.0), "IT unmapped → global (1e-5, 300.0)"
        assert (tols[2], min_mzs[2]) == (2e-5, 750.0), "QTOF mapped"

    def test_sort_order_is_alphabetical(self):
        """Input order of inst_types is irrelevant; indices follow alphabetical sort."""
        # unsorted input: QTOF, IT, FT — sorted: FT=0, IT=1, QTOF=2
        tols, min_mzs = self._build_lookup(
            ["QTOF", "IT", "FT"],
            {
                "FT": {"rel": 1e-5, "min_mz": 200.0},
                "IT": {"rel": 3e-5, "min_mz": 100.0},
                "QTOF": {"rel": 2e-5, "min_mz": 750.0},
            },
        )
        assert (tols[0], min_mzs[0]) == (1e-5, 200.0), "FT = idx 0 (first alphabetically)"
        assert (tols[1], min_mzs[1]) == (3e-5, 100.0), "IT = idx 1"
        assert (tols[2], min_mzs[2]) == (2e-5, 750.0), "QTOF = idx 2"

    def test_all_unmapped_uses_global_everywhere(self):
        """When inst_type_loss_tol is empty, all entries get the global tolerance."""
        tols, min_mzs = self._build_lookup(
            ["FT", "QTOF", "IT"],
            {},  # empty — simulates null config
            global_tol=5e-6,
            global_min_mz=150.0,
        )
        for idx in range(3):
            assert (tols[idx], min_mzs[idx]) == (5e-6, 150.0), f"idx={idx} should use global"


# ---------------------------------------------------------------------------
# FT floor adequacy — is 10 ppm / 0.002 Da a good floor for Orbitrap?
# ---------------------------------------------------------------------------


class TestFTFloorAdequacy:
    """Validate that 10 ppm (1e-5) with min_mz=200 (floor 0.002 Da) is appropriate for FT.

    The effective tolerance at m/z M is:  max(M, 200.0) × 1e-5
      - M < 200: floor = 0.002 Da  (floor controls)
      - M = 200: floor = 0.002 Da  (boundary — identical)
      - M > 200: M × 1e-5 Da       (relative controls, wider than floor)

    Two claims are tested:
    1. Floor is wide enough to cover FT instrument noise (< 1-2 mDa below m/z 200).
    2. Floor is narrow enough to resolve all chemically meaningful isobar pairs
       (smallest common spacing ≈ 0.011 Da for CO vs N2 at m/z ~28).
    """

    FT_TOL = 1e-5
    FT_MIN_MZ = 200.0

    def _match(self, true_mz: float, pred_mz: float) -> bool:
        return calculate_match_mzs(
            th.tensor([true_mz]),
            th.tensor([pred_mz]),
            tolerance=self.FT_TOL,
            relative=True,
            tolerance_min_mz=self.FT_MIN_MZ,
        )[0, 0].item()

    # --- floor boundary ---

    def test_floor_boundary_at_min_mz(self):
        """At m/z = min_mz = 200, relative and absolute floors agree exactly (0.002 Da).

        A 1.9 mDa shift should match; a 2.1 mDa shift should not.
        """
        assert self._match(200.0, 200.0019), "1.9 mDa at m/z 200 (< 0.002 floor) → match"
        assert not self._match(200.0, 200.0021), "2.1 mDa at m/z 200 (> 0.002 floor) → no match"

    # --- below min_mz: floor (0.002 Da) controls ---

    def test_floor_controls_below_min_mz(self):
        """Below m/z 200 the 0.002 Da floor applies (relative 10 ppm would be < floor).

        At m/z 100: 10 ppm = 0.001 Da, but floor = max(100, 200) × 1e-5 = 0.002 Da.
        """
        assert self._match(100.0, 100.0019), "1.9 mDa at m/z 100: inside 0.002 floor → match"
        assert not self._match(100.0, 100.0021), "2.1 mDa at m/z 100: outside 0.002 floor → no match"
        # Very low m/z (m/z 28, common small fragments): floor still 0.002 Da
        assert self._match(28.0, 28.0019), "1.9 mDa at m/z 28: inside 0.002 floor → match"
        assert not self._match(28.0, 28.0021), "2.1 mDa at m/z 28: outside 0.002 floor → no match"

    # --- above min_mz: relative (10 ppm) controls, wider than floor ---

    def test_relative_controls_above_min_mz(self):
        """Above m/z 200 the relative 10 ppm tolerance is wider than the 0.002 Da floor.

        At m/z 500: 10 ppm = 0.005 Da.
        At m/z 1000: 10 ppm = 0.010 Da.
        """
        assert self._match(500.0, 500.0049), "4.9 mDa at m/z 500: within 10 ppm (0.005 Da) → match"
        assert not self._match(500.0, 500.0051), "5.1 mDa at m/z 500: outside 10 ppm → no match"
        assert self._match(1000.0, 1000.0099), "9.9 mDa at m/z 1000: within 10 ppm → match"
        assert not self._match(1000.0, 1000.0101), "10.1 mDa at m/z 1000: outside 10 ppm → no match"

    # --- FT instrument noise: < 1-2 mDa should always match ---

    def test_ft_noise_under_1mda_always_matches(self):
        """Orbitrap noise is < 1 mDa across typical m/z range; 1 mDa must always be IOS."""
        for mz in [28.0, 100.0, 200.0, 500.0, 1000.0]:
            assert self._match(mz, mz + 0.001), f"1 mDa noise at m/z {mz} should match"

    # --- key FT isobar: CO vs N2 at m/z ~28 ---

    def test_co_n2_isobar_not_confused(self):
        """CO vs N2 isobar at m/z ~28: spacing 0.011 Da >> 0.002 Da floor → resolved.

        CO fragment: 27.9949 Da (12C + 16O).
        N2 fragment: 28.0062 Da (2 × 14N).
        Difference: 0.0113 Da = 5.6× the 0.002 Da FT floor.

        The 0.002 Da floor is tight enough to preserve this resolution.
        """
        co_mz = 27.9949  # 12C=O
        n2_mz = 28.0062  # 14N2
        assert not self._match(co_mz, n2_mz), "CO vs N2: 11.3 mDa >> 2 mDa floor → NOT matched"
        assert not self._match(n2_mz, co_mz), "N2 vs CO: same reasoning"

    # --- FT floor rejects QTOF truncation artifacts (confirms need for QTOF override) ---

    def test_ft_floor_rejects_qtof_truncation_artifact(self):
        """FT floor (0.002 Da at low m/z; 10 ppm relative at high m/z) correctly rejects
        QTOF floor-truncation artifacts (up to 0.009 Da error).

        At m/z 500 (above min_mz): FT tolerance = 500 × 1e-5 = 0.005 Da.
          - QTOF max truncation: 0.009 Da > 0.005 Da → FT correctly rejects.
        This confirms FT needs tight tolerance while QTOF needs wider floor (min_mz=750 → 0.015 Da).
        """
        # max QTOF truncation error = 0.009 Da at any m/z
        for diff in [0.007, 0.009]:
            # above min_mz=200: FT tolerance = 500 × 1e-5 = 0.005 Da < diff
            assert not self._match(500.0, 500.0 + diff), (
                f"FT at m/z 500: {diff*1000:.0f} mDa > 5 mDa (10 ppm) → correctly rejected"
            )
        # QTOF with wide floor accepts the same peaks (contrast)
        for diff in [0.007, 0.009]:
            qtof_match = calculate_match_mzs(
                th.tensor([500.0]),
                th.tensor([500.0 + diff]),
                tolerance=2e-5,
                relative=True,
                tolerance_min_mz=750.0,  # floor = 750 × 2e-5 = 0.015 Da
            )[0, 0].item()
            assert qtof_match, (
                f"QTOF at m/z 500: {diff*1000:.0f} mDa < 15 mDa (QTOF floor) → correctly accepted"
            )

    # --- raising the floor globally would harm FT ---

    def test_raising_floor_globally_would_confuse_co_n2(self):
        """If min_mz were raised to 1000 (floor = 0.010 Da), CO/N2 isobar stays safe,
        but a pair separated by only 0.005 Da at low m/z would be falsely matched.

        This motivates per-instrument override rather than a single global floor.
        """
        co_mz = 27.9949
        n2_mz = 28.0062  # diff = 0.011 Da — still safe even at 0.010 Da floor

        # with raised floor (0.010 Da) CO/N2 is still distinguished
        co_n2_with_large_floor = calculate_match_mzs(
            th.tensor([co_mz]),
            th.tensor([n2_mz]),
            tolerance=1e-5,
            relative=True,
            tolerance_min_mz=1000.0,  # floor = 1000 × 1e-5 = 0.010 Da
        )[0, 0].item()
        assert not co_n2_with_large_floor, "CO/N2 still separated even with 0.010 Da floor"

        # but a pair 0.005 Da apart at m/z 50 would be falsely matched with large floor
        close_pair_tight = calculate_match_mzs(
            th.tensor([50.0]),
            th.tensor([50.005]),
            tolerance=1e-5,
            relative=True,
            tolerance_min_mz=200.0,  # floor = 0.002 Da < 0.005 Da → no match
        )[0, 0].item()
        close_pair_loose = calculate_match_mzs(
            th.tensor([50.0]),
            th.tensor([50.005]),
            tolerance=1e-5,
            relative=True,
            tolerance_min_mz=1000.0,  # floor = 0.010 Da > 0.005 Da → match (false positive)
        )[0, 0].item()
        assert not close_pair_tight, "0.005 Da pair at m/z 50 — correct floor (0.002 Da) → no match"
        assert close_pair_loose, "0.005 Da pair at m/z 50 — raised floor (0.010 Da) → false match"
