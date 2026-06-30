"""Tests for detect_cross_ce_orphan_peaks.

Covers:
- Single-spectrum group: all-False (nothing to compare against)
- Two-spectrum group: basic orphan labelling
- Three-spectrum group: min_other_ces=1 vs min_other_ces=2
- Empty spectra and empty groups
- Instrument-specific tolerance selection (FT vs QTOF vs IT/default)
- mz_tol override
- Input validation errors
- Intensity not used (orphan decision is m/z-only)
"""

import numpy as np
import pytest

from fragnnet.utils.isotope_utils import detect_cross_ce_orphan_peaks


def _make(mzs: list[float], ints: list[float]) -> tuple[np.ndarray, np.ndarray]:
    return np.array(mzs, dtype=np.float64), np.array(ints, dtype=np.float64)


class TestSingleSpectrum:
    def test_single_spectrum_all_false(self):
        """One spectrum → no comparisons possible → all non-orphan."""
        masks = detect_cross_ce_orphan_peaks([_make([100.0, 200.0], [1000.0, 500.0])])
        assert len(masks) == 1
        assert not np.any(masks[0])

    def test_single_empty_spectrum(self):
        masks = detect_cross_ce_orphan_peaks([_make([], [])])
        assert masks[0].shape == (0,)


class TestTwoSpectra:
    def test_shared_peaks_not_orphan(self):
        """Peaks present in both spectra are not orphans."""
        a = _make([100.0, 150.0, 200.0], [1000.0, 500.0, 200.0])
        b = _make([100.0, 150.0, 200.0], [900.0, 450.0, 180.0])
        masks = detect_cross_ce_orphan_peaks([a, b], inst_type="FT")
        assert not np.any(masks[0])
        assert not np.any(masks[1])

    def test_unique_peak_is_orphan(self):
        """Peak present in only one spectrum is labelled orphan."""
        a = _make([100.0, 150.0, 999.0], [1000.0, 500.0, 10.0])
        b = _make([100.0, 150.0], [900.0, 450.0])
        masks = detect_cross_ce_orphan_peaks([a, b], inst_type="FT")
        # 999.0 is only in spectrum a
        assert masks[0][2]
        assert not masks[0][0]
        assert not masks[0][1]
        # b has no unique peaks
        assert not np.any(masks[1])

    def test_both_have_unique_peaks(self):
        a = _make([100.0, 777.0], [1000.0, 50.0])
        b = _make([100.0, 888.0], [900.0, 30.0])
        masks = detect_cross_ce_orphan_peaks([a, b], inst_type="FT")
        # 100.0 shared
        assert not masks[0][0]
        assert not masks[1][0]
        # 777.0 and 888.0 are unique
        assert masks[0][1]
        assert masks[1][1]


class TestThreeSpectra:
    def test_min_other_ces_1(self):
        """Peak in 2/3 spectra passes min_other_ces=1."""
        a = _make([100.0, 200.0], [1000.0, 500.0])
        b = _make([100.0, 200.0, 300.0], [900.0, 450.0, 10.0])
        c = _make([100.0], [800.0])
        # 200.0: in a and b, not c → matches 1 other CE from a's perspective → not orphan
        # 300.0: only in b → orphan for b
        masks = detect_cross_ce_orphan_peaks([a, b, c], inst_type="FT", min_other_ces=1)
        assert not masks[0][1]  # 200.0 in a matched by b
        assert masks[1][2]  # 300.0 only in b

    def test_min_other_ces_2(self):
        """Peak must appear in ≥2 other spectra to pass min_other_ces=2."""
        a = _make([100.0, 200.0], [1000.0, 500.0])
        b = _make([100.0, 200.0], [900.0, 450.0])
        c = _make([100.0], [800.0])
        # 200.0: in a and b, not c → matches 1 other CE → orphan when min_other_ces=2
        masks = detect_cross_ce_orphan_peaks([a, b, c], inst_type="FT", min_other_ces=2)
        assert masks[0][1]  # 200.0 in a: only 1 match (b), need 2
        assert masks[1][1]  # 200.0 in b: only 1 match (a), need 2
        # 100.0 is in all 3 → both a and b have 2 other matches
        assert not masks[0][0]
        assert not masks[1][0]


class TestTolerance:
    def test_ft_tolerance_within(self):
        """FT default 0.02 Da: peaks 0.015 Da apart match."""
        a = _make([100.000], [1000.0])
        b = _make([100.015], [900.0])
        masks = detect_cross_ce_orphan_peaks([a, b], inst_type="FT")
        assert not masks[0][0]
        assert not masks[1][0]

    def test_ft_tolerance_outside(self):
        """FT default 0.02 Da: peaks 0.025 Da apart do not match → both orphan."""
        a = _make([100.000], [1000.0])
        b = _make([100.025], [900.0])
        masks = detect_cross_ce_orphan_peaks([a, b], inst_type="FT")
        assert masks[0][0]
        assert masks[1][0]

    def test_qtof_tolerance(self):
        """QTOF default 0.05 Da: peaks 0.04 Da apart match."""
        a = _make([200.000], [1000.0])
        b = _make([200.040], [900.0])
        masks = detect_cross_ce_orphan_peaks([a, b], inst_type="QTOF")
        assert not masks[0][0]
        assert not masks[1][0]

    def test_default_tolerance_it(self):
        """Unknown/IT default 0.5 Da: peaks 0.4 Da apart match."""
        a = _make([300.000], [1000.0])
        b = _make([300.400], [900.0])
        masks = detect_cross_ce_orphan_peaks([a, b], inst_type="IT")
        assert not masks[0][0]
        assert not masks[1][0]

    def test_mz_tol_override(self):
        """Explicit mz_tol overrides inst_type default."""
        a = _make([100.000], [1000.0])
        b = _make([100.100], [900.0])
        # With FT default (0.02 Da): orphans
        masks_ft = detect_cross_ce_orphan_peaks([a, b], inst_type="FT")
        assert masks_ft[0][0]
        # With explicit 0.2 Da: not orphans
        masks_wide = detect_cross_ce_orphan_peaks([a, b], inst_type="FT", mz_tol=0.2)
        assert not masks_wide[0][0]


class TestIntensityIgnored:
    def test_low_intensity_peak_is_not_auto_orphan(self):
        """Orphan detection is m/z-based only; a high-intensity unique peak is still orphan."""
        a = _make([100.0, 500.0], [1000.0, 9999.0])
        b = _make([100.0], [900.0])
        masks = detect_cross_ce_orphan_peaks([a, b], inst_type="FT")
        # 500.0 is high intensity but unique → orphan
        assert masks[0][1]


class TestEmptySpectra:
    def test_one_empty_one_nonempty(self):
        """Empty spectrum contributes nothing; peaks in the other are orphans."""
        a = _make([100.0, 200.0], [1000.0, 500.0])
        b = _make([], [])
        masks = detect_cross_ce_orphan_peaks([a, b], inst_type="FT")
        # b is empty, so all peaks in a are orphans
        assert np.all(masks[0])
        assert masks[1].shape == (0,)

    def test_both_empty(self):
        masks = detect_cross_ce_orphan_peaks([_make([], []), _make([], [])])
        assert masks[0].shape == (0,)
        assert masks[1].shape == (0,)


class TestValidation:
    def test_empty_peaks_list(self):
        with pytest.raises(ValueError, match="peaks_list must contain"):
            detect_cross_ce_orphan_peaks([])

    def test_mismatched_mzs_ints(self):
        with pytest.raises(ValueError, match="peaks_list\\[0\\]"):
            detect_cross_ce_orphan_peaks(
                [(np.array([1.0, 2.0]), np.array([1.0]))]
            )

    def test_2d_input_rejected(self):
        with pytest.raises(ValueError, match="peaks_list\\[0\\]"):
            detect_cross_ce_orphan_peaks(
                [(np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]]))]
            )


class TestOutputShape:
    def test_output_length_matches_input(self):
        specs = [_make([100.0 + i, 200.0], [1000.0, 500.0]) for i in range(5)]
        masks = detect_cross_ce_orphan_peaks(specs, inst_type="FT")
        assert len(masks) == 5
        for i, (mzs, _) in enumerate(specs):
            assert masks[i].shape == (len(mzs),)
            assert masks[i].dtype == bool
