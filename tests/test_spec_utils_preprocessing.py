"""Unit tests for spec_utils data preprocessing and transformation pipeline.

Tests for:
- get_ints_transform_func / get_ints_untransform_func (intensity transforms)
- transform_ce, transform_nce_to_ace (collision energy transforms)
- filter_func, bin_func, batch_func (single-spectrum operations)
- batched_filter_func, batched_bin_func (batched operations)
- merge_sparse_specs, calculate_spectrum_entropy (spectrum utilities)
- round_aggregate_peaks (peak deduplication and aggregation)
"""

import math

import numpy as np
import pytest
import torch as th

from fragnnet.utils.spec_utils import (
    batch_func,
    batched_bin_func,
    batched_filter_func,
    bin_func,
    calculate_spectrum_entropy,
    filter_func,
    get_ints_transform_func,
    get_ints_untransform_func,
    merge_sparse_specs,
    round_aggregate_peaks,
    transform_ce,
    transform_nce_to_ace,
)


class TestGetIntsTransformFunc:
    """Tests for get_ints_transform_func — returns a forward intensity transform callable."""

    def test_log10_known_value(self):
        """log10(9 + 1) == 1.0."""
        f = get_ints_transform_func("log10")
        assert f(th.tensor([9.0])).item() == pytest.approx(1.0)

    def test_log10_zero_input(self):
        """log10(0 + 1) == 0.0."""
        f = get_ints_transform_func("log10")
        assert f(th.tensor([0.0])).item() == pytest.approx(0.0)

    def test_log10t3_known_value(self):
        """log10t3(27) = log10(27/3 + 1) = log10(10) = 1.0."""
        f = get_ints_transform_func("log10t3")
        assert f(th.tensor([27.0])).item() == pytest.approx(1.0)

    def test_loge_known_value(self):
        """loge(e - 1) == 1.0."""
        f = get_ints_transform_func("loge")
        assert f(th.tensor([math.e - 1.0])).item() == pytest.approx(1.0)

    def test_sqrt_known_value(self):
        """sqrt(4) == 2.0."""
        f = get_ints_transform_func("sqrt")
        assert f(th.tensor([4.0])).item() == pytest.approx(2.0)

    def test_none_is_identity(self):
        """'none' transform returns the input unchanged."""
        f = get_ints_transform_func("none")
        x = th.tensor([1.0, 5.0, 100.0])
        assert th.allclose(f(x), x)

    def test_invalid_raises_value_error(self):
        """Unknown transform name raises ValueError."""
        with pytest.raises(ValueError):
            get_ints_transform_func("invalid_transform")

    def test_all_transforms_monotonically_increasing(self):
        """All valid transforms are monotonically non-decreasing on positive inputs."""
        x = th.tensor([1.0, 10.0, 100.0, 1000.0])
        for name in ["log10", "log10t3", "loge", "sqrt", "none"]:
            f = get_ints_transform_func(name)
            y = f(x)
            assert (y[1:] >= y[:-1]).all(), f"Transform '{name}' is not monotone"


class TestGetIntsUntransformFunc:
    """Tests for get_ints_untransform_func — batched inverse intensity transform callable."""

    def _call(self, name: str, raw_ints: list[float]) -> th.Tensor:
        """Call untransform func on a single batch."""
        func = get_ints_untransform_func(name)
        ints = th.tensor(raw_ints)
        batch_idxs = th.zeros(len(raw_ints), dtype=th.long)
        return func(ints, batch_idxs)

    def test_output_non_negative_all_transforms(self):
        """Untransformed values are always >= 0 for all valid transforms."""
        for name in ["log10", "log10t3", "loge", "sqrt", "none"]:
            result = self._call(name, [0.5, 1.0, 2.0])
            assert (result >= 0).all(), f"Negative output for '{name}'"

    def test_monotonic_order_preserved(self):
        """Higher input → higher untransformed output (monotone increasing)."""
        for name in ["log10", "log10t3", "loge", "sqrt", "none"]:
            result = self._call(name, [0.1, 0.5, 1.0])
            assert result[0] <= result[1] <= result[2], f"Not monotone for '{name}'"

    def test_invalid_raises(self):
        """Unknown transform name raises ValueError."""
        with pytest.raises(ValueError):
            get_ints_untransform_func("bad_transform")

    def test_multi_batch_independent(self):
        """Each batch item is normalized by its own max (independent batches)."""
        func = get_ints_untransform_func("none")
        ints = th.tensor([2.0, 1.0])
        batch_idxs = th.tensor([0, 1])
        result = func(ints, batch_idxs)
        assert result.shape == (2,)
        assert (result >= 0).all()

    def test_log10_near_1000_at_max(self):
        """When input equals log10 max_ints, untransform gives ~1000."""
        max_ints_log10 = float(np.log10(1000.0 + 1.0))
        result = self._call("log10", [max_ints_log10])
        assert result.item() == pytest.approx(1000.0, abs=1.0)


class TestTransformCe:
    """Tests for transform_ce — z-score normalization of collision energy."""

    def test_scalar_positive_normalized(self):
        """CE >= 0 is z-score normalized: (ce - mean) / std."""
        result = transform_ce(30.0, ce_mean=25.0, ce_std=10.0)
        assert result.item() == pytest.approx(0.5)

    def test_scalar_zero_ce(self):
        """CE == 0 is normalized when mean=0."""
        result = transform_ce(0.0, ce_mean=0.0, ce_std=10.0)
        assert result.item() == pytest.approx(0.0)

    def test_scalar_unknown_ce_returned_unchanged(self):
        """CE < 0 (missing/unknown) is returned as-is without normalization."""
        result = transform_ce(-2.0, ce_mean=25.0, ce_std=10.0)
        assert result.item() == pytest.approx(-2.0)

    def test_tensor_mixed_known_and_unknown(self):
        """Tensor with mixed CE: >= 0 normalized, < 0 passed through unchanged."""
        ce = th.tensor([30.0, -2.0])
        result = transform_ce(ce, ce_mean=25.0, ce_std=10.0)
        assert result[0].item() == pytest.approx(0.5)
        assert result[1].item() == pytest.approx(-2.0)

    def test_tensor_all_positive(self):
        """All-positive tensor CE values are all normalized."""
        ce = th.tensor([25.0, 35.0])
        result = transform_ce(ce, ce_mean=25.0, ce_std=10.0)
        assert result[0].item() == pytest.approx(0.0)
        assert result[1].item() == pytest.approx(1.0)


class TestTransformNceToAce:
    """Tests for transform_nce_to_ace — converts normalized to absolute collision energy."""

    def test_known_value_charge1(self):
        """ACE = NCE * mw / 500 * charge_factor for charge_factor=1."""
        assert transform_nce_to_ace(20.0, 500.0, charge_factor=1) == pytest.approx(20.0)

    def test_double_mw_doubles_ace(self):
        """Doubling molecular weight doubles the ACE."""
        ace_500 = transform_nce_to_ace(20.0, 500.0)
        ace_1000 = transform_nce_to_ace(20.0, 1000.0)
        assert ace_1000 == pytest.approx(2 * ace_500)

    def test_charge_factor_scales_linearly(self):
        """charge_factor=2 doubles ACE relative to charge_factor=1."""
        ace_1 = transform_nce_to_ace(20.0, 500.0, charge_factor=1)
        ace_2 = transform_nce_to_ace(20.0, 500.0, charge_factor=2)
        assert ace_2 == pytest.approx(2 * ace_1)

    def test_zero_nce_gives_zero(self):
        """NCE=0 always gives ACE=0 regardless of mw."""
        assert transform_nce_to_ace(0.0, 1000.0) == pytest.approx(0.0)



class TestFilterFunc:
    """Tests for filter_func — intensity threshold and m/z upper bound filtering."""

    def test_tensor_filters_by_thresh_and_mz_max(self):
        """Peaks with ints <= thresh or mzs >= mz_max are removed."""
        mzs = th.tensor([100.0, 200.0, 300.0])
        ints = th.tensor([0.5, 1.5, 2.5])
        out_mzs, out_ints = filter_func(mzs, ints, ints_thresh=1.0, mz_max=250.0)
        assert th.equal(out_mzs, th.tensor([200.0]))
        assert th.equal(out_ints, th.tensor([1.5]))

    def test_ndarray_same_result(self):
        """Works identically with numpy arrays."""
        mzs = np.array([100.0, 200.0, 300.0])
        ints = np.array([0.5, 1.5, 2.5])
        out_mzs, out_ints = filter_func(mzs, ints, ints_thresh=1.0, mz_max=250.0)
        np.testing.assert_array_equal(out_mzs, [200.0])
        np.testing.assert_array_equal(out_ints, [1.5])

    def test_mixed_types_raises_type_error(self):
        """Tensor mzs + ndarray ints raises TypeError."""
        with pytest.raises(TypeError):
            filter_func(th.tensor([100.0]), np.array([1.0]), ints_thresh=0.0, mz_max=1000.0)

    def test_mz_max_zero_disables_mz_filter(self):
        """mz_max <= 0 skips the m/z upper bound check."""
        mzs = th.tensor([100.0, 5000.0])
        ints = th.tensor([2.0, 3.0])
        out_mzs, _ = filter_func(mzs, ints, ints_thresh=0.0, mz_max=0.0)
        assert out_mzs.shape[0] == 2

    def test_all_filtered_returns_empty(self):
        """If all peaks fail the filter, empty tensors are returned."""
        mzs = th.tensor([100.0, 200.0])
        ints = th.tensor([0.1, 0.2])
        out_mzs, out_ints = filter_func(mzs, ints, ints_thresh=1.0, mz_max=1000.0)
        assert out_mzs.shape[0] == 0
        assert out_ints.shape[0] == 0

    def test_thresh_is_exclusive(self):
        """Intensity exactly at threshold is excluded (strict >)."""
        mzs = th.tensor([100.0])
        ints = th.tensor([1.0])
        out_mzs, _ = filter_func(mzs, ints, ints_thresh=1.0, mz_max=1000.0)
        assert out_mzs.shape[0] == 0

    def test_mz_max_is_exclusive(self):
        """m/z exactly at mz_max is excluded (strict <)."""
        mzs = th.tensor([250.0])
        ints = th.tensor([2.0])
        out_mzs, _ = filter_func(mzs, ints, ints_thresh=0.0, mz_max=250.0)
        assert out_mzs.shape[0] == 0


class TestBinFunc:
    """Tests for bin_func — histogram binning of a single spectrum.

    Uses mz_max=10.0, mz_bin_res=1.0 → 10 bins with edges [1,2,...,10].
    A peak at mz x lands in bin searchsorted(edges, x, right=True).
    """

    def test_single_peak_one_nonzero_bin(self):
        """A single peak produces exactly one non-zero bin."""
        mzs = th.tensor([2.5])
        ints = th.tensor([3.0])
        out = bin_func(mzs, ints, mz_max=10.0, mz_bin_res=1.0, return_index=False, sum_ints=True)
        assert out.shape[0] == 10
        assert out.sum().item() == pytest.approx(3.0)
        assert (out > 0).sum().item() == 1

    def test_sum_ints_two_peaks_same_bin(self):
        """Two peaks in the same bin are summed when sum_ints=True."""
        mzs = th.tensor([2.3, 2.7])  # both map to bin for (2, 3]
        ints = th.tensor([1.0, 2.0])
        out = bin_func(mzs, ints, mz_max=10.0, mz_bin_res=1.0, return_index=False, sum_ints=True)
        assert out.sum().item() == pytest.approx(3.0)
        assert (out > 0).sum().item() == 1

    def test_amax_two_peaks_same_bin(self):
        """Two peaks in the same bin take the max when sum_ints=False."""
        mzs = th.tensor([2.3, 2.7])
        ints = th.tensor([1.0, 2.0])
        out = bin_func(mzs, ints, mz_max=10.0, mz_bin_res=1.0, return_index=False, sum_ints=False)
        assert out.max().item() == pytest.approx(2.0)
        assert (out > 0).sum().item() == 1

    def test_two_peaks_different_bins(self):
        """Peaks at well-separated m/zs land in two distinct non-zero bins."""
        mzs = th.tensor([2.5, 7.5])
        ints = th.tensor([1.0, 2.0])
        out = bin_func(mzs, ints, mz_max=10.0, mz_bin_res=1.0, return_index=False, sum_ints=True)
        assert (out > 0).sum().item() == 2
        assert out.sum().item() == pytest.approx(3.0)

    def test_return_index_shape_and_uniqueness(self):
        """return_index=True returns per-peak bin indices, one per input peak."""
        mzs = th.tensor([2.5, 7.5])
        ints = th.tensor([1.0, 2.0])
        idx = bin_func(mzs, ints, mz_max=10.0, mz_bin_res=1.0, return_index=True, sum_ints=True)
        assert idx.shape == (2,)
        assert idx[0].item() != idx[1].item()  # different bins

    def test_return_index_same_bin_for_close_peaks(self):
        """Peaks in the same bin return the same index."""
        mzs = th.tensor([2.3, 2.7])
        ints = th.tensor([1.0, 1.0])
        idx = bin_func(mzs, ints, mz_max=10.0, mz_bin_res=1.0, return_index=True, sum_ints=True)
        assert idx[0].item() == idx[1].item()


class TestBatchFunc:
    """Tests for batch_func — concatenates per-item tensor lists with batch indices."""

    def test_two_spectra_concatenated(self):
        """Two spectra are concatenated with correct batch indices."""
        mzs = [th.tensor([1.0, 2.0]), th.tensor([3.0])]
        ints = [th.tensor([10.0, 20.0]), th.tensor([30.0])]
        b_mzs, b_ints, batch_idxs = batch_func(mzs, ints)
        assert th.equal(b_mzs, th.tensor([1.0, 2.0, 3.0]))
        assert th.equal(b_ints, th.tensor([10.0, 20.0, 30.0]))
        assert th.equal(batch_idxs, th.tensor([0, 0, 1]))

    def test_offset_flag_shifts_second_batch(self):
        """offset_flags=[True, False] adds cumulative size offsets to first list."""
        mzs = [th.tensor([1, 2]), th.tensor([3])]
        ints = [th.tensor([10.0, 20.0]), th.tensor([30.0])]
        b_mzs_off, b_ints, _ = batch_func(mzs, ints, offset_flags=[True, False])
        # offsets = [0, 2] (first batch has 2 items)
        # batch 1 mz: 3 + 2 = 5
        assert th.equal(b_mzs_off, th.tensor([1, 2, 5]))
        assert th.equal(b_ints, th.tensor([10.0, 20.0, 30.0]))

    def test_single_spectrum_all_zeros_batch_idx(self):
        """Single spectrum → all batch indices are 0."""
        mzs = [th.tensor([5.0, 6.0, 7.0])]
        (b_mzs,), batch_idxs = batch_func(mzs)[:1], batch_func(mzs)[1]
        assert th.equal(batch_idxs, th.zeros(3, dtype=th.long))


class TestBatchedFilterFunc:
    """Tests for batched_filter_func — batched intensity and m/z filtering."""

    def test_filters_within_each_batch(self):
        """Peaks below threshold or above mz_max are removed; batch indices updated."""
        mzs = th.tensor([100.0, 200.0, 300.0, 400.0])
        ints = th.tensor([0.5, 1.5, 2.5, 0.1])
        batch_idxs = th.tensor([0, 0, 1, 1])
        out_mzs, out_ints, out_batch = batched_filter_func(
            mzs, ints, batch_idxs, ints_thresh=1.0, mz_max=350.0
        )
        assert th.equal(out_mzs, th.tensor([200.0, 300.0]))
        assert th.equal(out_ints, th.tensor([1.5, 2.5]))
        assert th.equal(out_batch, th.tensor([0, 1]))

    def test_mz_max_zero_disables_upper_bound(self):
        """mz_max <= 0 skips the m/z upper bound filter."""
        mzs = th.tensor([100.0, 5000.0])
        ints = th.tensor([2.0, 3.0])
        batch_idxs = th.tensor([0, 0])
        out_mzs, _, _ = batched_filter_func(mzs, ints, batch_idxs, ints_thresh=0.0, mz_max=0.0)
        assert out_mzs.shape[0] == 2

    def test_thresh_is_exclusive(self):
        """Intensity exactly equal to threshold is excluded (strict >)."""
        mzs = th.tensor([100.0])
        ints = th.tensor([1.0])
        batch_idxs = th.tensor([0])
        out_mzs, _, _ = batched_filter_func(mzs, ints, batch_idxs, ints_thresh=1.0, mz_max=0.0)
        assert out_mzs.shape[0] == 0

    def test_preserves_batch_structure(self):
        """Batch indices in output correspond to the surviving peaks."""
        mzs = th.tensor([1.0, 2.0, 3.0, 4.0])
        ints = th.tensor([5.0, 0.1, 0.1, 5.0])  # peaks 0 and 3 survive
        batch_idxs = th.tensor([0, 0, 1, 1])
        out_mzs, out_ints, out_batch = batched_filter_func(
            mzs, ints, batch_idxs, ints_thresh=1.0, mz_max=0.0
        )
        assert th.equal(out_batch, th.tensor([0, 1]))

    def test_top_k_peaks_applies_per_batch_item(self):
        """top_k_peaks keeps the strongest peaks independently for each spectrum."""
        mzs = th.tensor([100.0, 200.0, 300.0, 400.0, 500.0])
        ints = th.tensor([1.0, 4.0, 2.0, 10.0, 3.0])
        batch_idxs = th.tensor([0, 0, 0, 1, 1])
        out_mzs, out_ints, out_batch = batched_filter_func(
            mzs,
            ints,
            batch_idxs,
            ints_thresh=0.0,
            mz_max=0.0,
            top_k_peaks=2,
        )
        assert th.equal(out_mzs, th.tensor([200.0, 300.0, 400.0, 500.0]))
        assert th.equal(out_ints, th.tensor([4.0, 2.0, 10.0, 3.0]))
        assert th.equal(out_batch, th.tensor([0, 0, 1, 1]))

    def test_negative_top_k_peaks_disables_top_k_filter(self):
        """top_k_peaks=-1 is the config default and leaves peak counts unchanged."""
        mzs = th.tensor([100.0, 200.0, 300.0])
        ints = th.tensor([1.0, 4.0, 2.0])
        batch_idxs = th.tensor([0, 0, 0])
        out_mzs, out_ints, out_batch = batched_filter_func(
            mzs,
            ints,
            batch_idxs,
            ints_thresh=0.0,
            mz_max=0.0,
            top_k_peaks=-1,
        )
        assert th.equal(out_mzs, mzs)
        assert th.equal(out_ints, ints)
        assert th.equal(out_batch, batch_idxs)

    def test_drop_min_int_peak_applies_per_batch_item(self):
        """drop_min_int_peak removes one weakest peak from each non-singleton spectrum."""
        mzs = th.tensor([100.0, 200.0, 300.0, 400.0, 500.0])
        ints = th.tensor([1.0, 4.0, 2.0, 10.0, 3.0])
        batch_idxs = th.tensor([0, 0, 0, 1, 1])
        out_mzs, out_ints, out_batch = batched_filter_func(
            mzs,
            ints,
            batch_idxs,
            ints_thresh=0.0,
            mz_max=0.0,
            drop_min_int_peak=True,
        )
        assert th.equal(out_mzs, th.tensor([200.0, 300.0, 400.0]))
        assert th.equal(out_ints, th.tensor([4.0, 2.0, 10.0]))
        assert th.equal(out_batch, th.tensor([0, 0, 1]))

    def test_rank_filters_run_after_threshold_filter(self):
        """Threshold filtering happens before weakest-peak and top-k filtering."""
        mzs = th.tensor([100.0, 200.0, 300.0, 400.0])
        ints = th.tensor([0.5, 2.0, 5.0, 7.0])
        batch_idxs = th.tensor([0, 0, 0, 0])
        out_mzs, out_ints, out_batch = batched_filter_func(
            mzs,
            ints,
            batch_idxs,
            ints_thresh=1.0,
            mz_max=0.0,
            drop_min_int_peak=True,
            top_k_peaks=1,
        )
        assert th.equal(out_mzs, th.tensor([400.0]))
        assert th.equal(out_ints, th.tensor([7.0]))
        assert th.equal(out_batch, th.tensor([0]))


class TestBatchedBinFunc:
    """Tests for batched_bin_func — batched histogram binning.

    Uses mz_max=10.0, mz_bin_res=1.0 → 10 bins per batch item.
    """

    def test_dense_output_shape(self):
        """Dense mode (sparse=False) returns tensor of shape (batch_size, num_bins)."""
        mzs = th.tensor([2.5, 7.5])
        ints = th.tensor([1.0, 2.0])
        batch_idxs = th.tensor([0, 1])
        out = batched_bin_func(mzs, ints, batch_idxs, mz_max=10.0, mz_bin_res=1.0, agg="sum")
        assert out.shape == (2, 10)

    def test_dense_sum_aggregation(self):
        """Two peaks in the same bin within one batch item are summed."""
        mzs = th.tensor([2.3, 2.7])
        ints = th.tensor([1.0, 2.0])
        batch_idxs = th.tensor([0, 0])
        out = batched_bin_func(mzs, ints, batch_idxs, mz_max=10.0, mz_bin_res=1.0, agg="sum")
        assert out[0].sum().item() == pytest.approx(3.0)
        assert (out[0] > 0).sum().item() == 1

    def test_dense_amax_aggregation(self):
        """Two peaks in the same bin take the max in amax mode."""
        mzs = th.tensor([2.3, 2.7])
        ints = th.tensor([1.0, 2.0])
        batch_idxs = th.tensor([0, 0])
        out = batched_bin_func(mzs, ints, batch_idxs, mz_max=10.0, mz_bin_res=1.0, agg="amax")
        assert out[0].max().item() == pytest.approx(2.0)

    def test_dense_lse_aggregation(self):
        """Two equal-valued peaks in the same bin aggregate to log(2) in lse mode."""
        mzs = th.tensor([2.3, 2.7])
        ints = th.tensor([0.0, 0.0])  # log-space: exp(0)=1 each
        batch_idxs = th.tensor([0, 0])
        out = batched_bin_func(mzs, ints, batch_idxs, mz_max=10.0, mz_bin_res=1.0, agg="lse")
        # LSE([0, 0]) = log(e^0 + e^0) = log(2)
        assert out[0].max().item() == pytest.approx(math.log(2), abs=1e-4)

    def test_batch_separation_dense(self):
        """Each batch item only contributes to its own row in the output."""
        mzs = th.tensor([2.5, 7.5])
        ints = th.tensor([1.0, 2.0])
        batch_idxs = th.tensor([0, 1])
        out = batched_bin_func(mzs, ints, batch_idxs, mz_max=10.0, mz_bin_res=1.0, agg="sum")
        assert out[0].sum().item() == pytest.approx(1.0)
        assert out[1].sum().item() == pytest.approx(2.0)

    def test_sparse_returns_triplet(self):
        """Sparse mode (sparse=True) returns a 3-tuple of (idxs, ints, batch_idxs)."""
        mzs = th.tensor([2.5, 7.5])
        ints = th.tensor([1.0, 2.0])
        batch_idxs = th.tensor([0, 0])
        result = batched_bin_func(
            mzs, ints, batch_idxs, mz_max=10.0, mz_bin_res=1.0, agg="sum", sparse=True
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        _, out_ints, out_batch = result
        assert out_ints.shape[0] == 2
        assert th.equal(out_batch, th.tensor([0, 0]))

    def test_sparse_return_mzs_gives_bin_centers(self):
        """return_mzs=True returns bin center m/z values instead of bin indices."""
        mzs = th.tensor([2.5])
        ints = th.tensor([1.0])
        batch_idxs = th.tensor([0])
        out_mzs, out_ints, _ = batched_bin_func(
            mzs, ints, batch_idxs, mz_max=10.0, mz_bin_res=1.0,
            agg="sum", sparse=True, return_mzs=True,
        )
        # Bin for 2.5 has right edge 3.0 and center 3.0 - 0.5 = 2.5
        assert out_mzs.item() == pytest.approx(2.5)
        assert out_ints.item() == pytest.approx(1.0)

    def test_remove_prec_peaks_zeroes_precursor_bin(self):
        """remove_prec_peaks=True zeros out the bin containing each precursor m/z."""
        mzs = th.tensor([2.5, 4.5])
        ints = th.tensor([1.0, 2.0])
        batch_idxs = th.tensor([0, 0])
        prec_mzs = th.tensor([2.5])  # one precursor per batch item
        out = batched_bin_func(
            mzs, ints, batch_idxs, mz_max=10.0, mz_bin_res=1.0, agg="sum",
            remove_prec_peaks=True, prec_mzs=prec_mzs,
        )
        # Peak at 2.5 → bin index 2; should be zeroed
        assert out[0, 2].item() == pytest.approx(0.0)
        # Peak at 4.5 → bin index 4; should be intact
        assert out[0, 4].item() == pytest.approx(2.0)

    def test_empty_mzs_raises_value_error(self):
        """Empty input tensor raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            batched_bin_func(
                th.tensor([]), th.tensor([]), th.tensor([], dtype=th.long),
                mz_max=10.0, mz_bin_res=1.0, agg="sum",
            )


class TestMergeSparseSpecs:
    """Tests for merge_sparse_specs — merges multiple peak lists into one."""

    def test_no_overlap_union_of_peaks(self):
        """Non-overlapping peaks from different spectra are all kept."""
        peaks1 = [(100.0, 1.0), (200.0, 2.0)]
        peaks2 = [(300.0, 3.0)]
        merged = merge_sparse_specs(peaks1, peaks2)
        assert len(merged) == 3
        mzs = {m for m, _ in merged}
        assert mzs == {100.0, 200.0, 300.0}

    def test_overlap_sum_ints(self):
        """Overlapping peaks are summed when sum_ints=True (default)."""
        peaks1 = [(100.0, 1.0)]
        peaks2 = [(100.0, 2.0)]
        merged = merge_sparse_specs(peaks1, peaks2, sum_ints=True)
        assert len(merged) == 1
        assert merged[0] == (100.0, 3.0)

    def test_overlap_max_ints(self):
        """Overlapping peaks take the maximum when sum_ints=False."""
        peaks1 = [(100.0, 1.0)]
        peaks2 = [(100.0, 2.0)]
        merged = merge_sparse_specs(peaks1, peaks2, sum_ints=False)
        assert len(merged) == 1
        assert merged[0] == (100.0, 2.0)

    def test_output_sorted_by_mz(self):
        """Output is always sorted in ascending m/z order."""
        peaks1 = [(300.0, 1.0), (100.0, 2.0)]
        merged = merge_sparse_specs(peaks1)
        mzs = [m for m, _ in merged]
        assert mzs == sorted(mzs)

    def test_renormalize_sums_to_one(self):
        """renormalize=True scales intensities so they sum to 1."""
        peaks1 = [(100.0, 2.0), (200.0, 3.0)]
        merged = merge_sparse_specs(peaks1, renormalize=True)
        total = sum(i for _, i in merged)
        assert total == pytest.approx(1.0)

    def test_three_spectra_partial_overlap(self):
        """Three spectra with partial overlap merge correctly per peak."""
        a = [(100.0, 1.0)]
        b = [(100.0, 1.0)]
        c = [(200.0, 5.0)]
        merged = merge_sparse_specs(a, b, c, sum_ints=True)
        merged_dict = dict(merged)
        assert merged_dict[100.0] == pytest.approx(2.0)
        assert merged_dict[200.0] == pytest.approx(5.0)


class TestCalculateSpectrumEntropy:
    """Tests for calculate_spectrum_entropy — entropy over log-intensity distribution.

    Note: this computes H = -sum(p * log(p)) where p = softmax(log_ints).
    """

    def test_single_peak_zero_entropy(self):
        """Single peak has H = 0 (the distribution is a point mass)."""
        log_ints = th.tensor([0.0])
        batch_idxs = th.tensor([0])
        entropy = calculate_spectrum_entropy(log_ints, batch_idxs)
        assert entropy.item() == pytest.approx(0.0, abs=1e-6)

    def test_two_equal_peaks_log2_entropy(self):
        """Two equal peaks → H = log(2) ≈ 0.6931."""
        log_ints = th.tensor([0.0, 0.0])
        batch_idxs = th.tensor([0, 0])
        entropy = calculate_spectrum_entropy(log_ints, batch_idxs)
        assert entropy.item() == pytest.approx(math.log(2), abs=1e-5)

    def test_three_equal_peaks_log3_entropy(self):
        """Three equal peaks → H = log(3)."""
        log_ints = th.tensor([0.0, 0.0, 0.0])
        batch_idxs = th.tensor([0, 0, 0])
        entropy = calculate_spectrum_entropy(log_ints, batch_idxs)
        assert entropy.item() == pytest.approx(math.log(3), abs=1e-5)

    def test_entropy_non_negative(self):
        """Entropy is always >= 0 for any valid input."""
        log_ints = th.tensor([1.0, 2.0, 0.5])
        batch_idxs = th.tensor([0, 0, 0])
        entropy = calculate_spectrum_entropy(log_ints, batch_idxs)
        assert entropy.item() >= 0.0

    def test_multi_batch_independent(self):
        """Entropy is computed independently per batch item."""
        # Batch 0: two equal peaks → H = log(2); Batch 1: one peak → H = 0
        log_ints = th.tensor([0.0, 0.0, 0.0])
        batch_idxs = th.tensor([0, 0, 1])
        entropy = calculate_spectrum_entropy(log_ints, batch_idxs)
        assert entropy[0].item() == pytest.approx(math.log(2), abs=1e-5)
        assert entropy[1].item() == pytest.approx(0.0, abs=1e-6)


class TestRoundAggregatePeaks:
    """Tests for round_aggregate_peaks — rounds m/z to decimals, aggregates duplicates."""

    def test_no_duplicates_pass_through(self):
        """Peaks that do not collide after rounding are returned unchanged."""
        mzs = th.tensor([1.0, 2.0])
        ints = th.tensor([1.0, 2.0])
        batch_idxs = th.tensor([0, 0])
        out_mzs, out_ints, _ = round_aggregate_peaks(mzs, ints, batch_idxs, decimals=4)
        assert out_mzs.shape[0] == 2
        assert out_ints.sum().item() == pytest.approx(3.0)

    def test_duplicates_summed_agg_sum(self):
        """Two peaks that round to the same m/z are summed with agg='sum'."""
        mzs = th.tensor([1.00001, 1.00002])  # both round to 1.0000 at 4 decimals
        ints = th.tensor([1.0, 2.0])
        batch_idxs = th.tensor([0, 0])
        out_mzs, out_ints, _ = round_aggregate_peaks(mzs, ints, batch_idxs, decimals=4, agg="sum")
        assert out_mzs.shape[0] == 1
        assert out_ints.item() == pytest.approx(3.0)

    def test_duplicates_amax(self):
        """Duplicates take the max intensity with agg='amax'."""
        mzs = th.tensor([1.00001, 1.00002])
        ints = th.tensor([1.0, 2.0])
        batch_idxs = th.tensor([0, 0])
        _, out_ints, _ = round_aggregate_peaks(mzs, ints, batch_idxs, decimals=4, agg="amax")
        assert out_ints.item() == pytest.approx(2.0)

    def test_duplicates_lse(self):
        """Duplicates use logsumexp with agg='lse'."""
        mzs = th.tensor([1.00001, 1.00002])
        ints = th.tensor([0.0, 0.0])  # log-space values
        batch_idxs = th.tensor([0, 0])
        _, out_ints, _ = round_aggregate_peaks(mzs, ints, batch_idxs, decimals=4, agg="lse")
        assert out_ints.item() == pytest.approx(math.log(2), abs=1e-4)

    def test_output_mzs_sorted_ascending(self):
        """Output m/zs are in ascending order within each batch."""
        mzs = th.tensor([3.0, 1.0, 2.0])
        ints = th.tensor([1.0, 2.0, 3.0])
        batch_idxs = th.tensor([0, 0, 0])
        out_mzs, _, _ = round_aggregate_peaks(mzs, ints, batch_idxs, decimals=4)
        assert out_mzs.tolist() == sorted(out_mzs.tolist())

    def test_multi_batch_independent_aggregation(self):
        """Each batch item's peaks are rounded and aggregated independently."""
        mzs = th.tensor([1.00001, 1.00002, 2.00001, 2.00002])
        ints = th.tensor([1.0, 2.0, 3.0, 4.0])
        batch_idxs = th.tensor([0, 0, 1, 1])
        out_mzs, out_ints, out_batch = round_aggregate_peaks(
            mzs, ints, batch_idxs, decimals=4, agg="sum"
        )
        assert out_mzs.shape[0] == 2
        assert out_ints[out_batch == 0].item() == pytest.approx(3.0)
        assert out_ints[out_batch == 1].item() == pytest.approx(7.0)

    def test_higher_decimals_fewer_collisions(self):
        """Higher decimal precision → fewer peaks merge (more distinct m/zs)."""
        mzs = th.tensor([1.001, 1.002])
        ints = th.tensor([1.0, 1.0])
        batch_idxs = th.tensor([0, 0])
        out_2dec, _, _ = round_aggregate_peaks(mzs, ints, batch_idxs, decimals=2)
        out_4dec, _, _ = round_aggregate_peaks(mzs, ints, batch_idxs, decimals=4)
        # At 2 decimals: 1.001→1.00, 1.002→1.00 → 1 peak
        # At 4 decimals: 1.001→1.001, 1.002→1.002 → 2 peaks
        assert out_2dec.shape[0] == 1
        assert out_4dec.shape[0] == 2
