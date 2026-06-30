"""Tests for isotope peak detection.

Covers:
- ``_parse_adduct_counts``: correct parsing of adduct formula diffs
- ``_expected_mk_ratio``: adduct atoms applied at full weight
- ``detect_isotope_peaks_for_training``: co-isolation R_k based detection

Design of ``detect_isotope_peaks_for_training`` spectra
-------------------------------------------------------
The function estimates R_k from precursor M+k peaks *above* prec_mz.  Test
spectra are therefore built with two series:
  1. Fragment peaks: [M+0, M+1, ...] at ``frag_base + i * C13_SPACING``
  2. Precursor peaks: [M+0, M+1, ...] at ``prec_mz + i * C13_SPACING``

The tight FT tolerance (4 mDa for k=1, 8 mDa for k≥2) means peaks must be
placed at exact C13_SPACING multiples so the search window reliably finds them.
"""

import numpy as np
import pytest

from fragnnet.utils.formula_utils import C13_SPACING
from fragnnet.utils.isotope_utils import (
    _expected_mk_ratio,
    _parse_adduct_counts,
    detect_isotope_peaks_for_training,
    detect_isotope_peaks_for_training_cleanup,
    estimate_coisolation_fraction_from_precursor,
)

# ---------------------------------------------------------------------------
# _parse_adduct_counts
# ---------------------------------------------------------------------------


class TestParseAdductCounts:
    def test_mh_plus(self):
        counts, mass = _parse_adduct_counts("[M+H]+")
        assert counts == {"H": 1}
        assert mass == pytest.approx(1.00794, abs=0.002)

    def test_mk_plus(self):
        counts, mass = _parse_adduct_counts("[M+K]+")
        assert counts == {"K": 1}
        assert mass == pytest.approx(38.964, abs=0.005)

    def test_mna_plus(self):
        counts, mass = _parse_adduct_counts("[M+Na]+")
        assert counts == {"Na": 1}
        assert mass == pytest.approx(22.990, abs=0.005)

    def test_mnh4_plus(self):
        counts, mass = _parse_adduct_counts("[M+NH4]+")
        assert counts.get("N", 0) == 1
        assert counts.get("H", 0) == 4
        assert mass == pytest.approx(18.034, abs=0.005)

    def test_mcl_minus(self):
        counts, mass = _parse_adduct_counts("[M+Cl]-")
        assert counts == {"Cl": 1}
        assert mass == pytest.approx(34.969, abs=0.005)

    def test_mh_minus_negligible_mass(self):
        # [M-H]- removes H; parse_formula strips at '-', leaving 'H' with count 1.
        # H isotope contribution is 0.015% — effectively zero for our purposes.
        _, mass = _parse_adduct_counts("[M-H]-")
        assert mass < 2.0

    def test_none(self):
        counts, mass = _parse_adduct_counts(None)
        assert counts == {}
        assert mass == 0.0

    def test_unknown_adduct(self):
        counts, mass = _parse_adduct_counts("[M+Weird]+")
        assert counts == {}
        assert mass == 0.0


# ---------------------------------------------------------------------------
# _expected_mk_ratio
# ---------------------------------------------------------------------------


class TestExpectedMkRatio:
    """Verify that adduct atoms are applied at full weight, not scaled."""

    def test_no_adduct_m1_scales_with_mol_scale(self):
        # C10 molecule, scale 0.5: M+1 ≈ 10 * 0.01103 * 0.5 = 0.0552
        counts = {"C": 10}
        ratio = _expected_mk_ratio(counts, mol_scale=0.5, k=1)
        expected = 10 * 0.01103 * 0.5
        assert ratio == pytest.approx(expected, rel=0.05)

    def test_k_adduct_m2_at_full_weight(self):
        # Molecule has no Cl/Br/K/S — m2_mol ≈ tiny.
        # K adduct adds 0.0673 at full weight.
        counts = {"C": 10}
        adduct = {"K": 1}
        ratio_with = _expected_mk_ratio(counts, mol_scale=0.5, k=2, adduct_counts=adduct)
        ratio_without = _expected_mk_ratio(counts, mol_scale=0.5, k=2)
        assert ratio_with > ratio_without
        # K's M+2 contribution (0.0673) dominates
        assert (ratio_with - ratio_without) == pytest.approx(0.0673, abs=0.005)

    def test_k_adduct_m2_independent_of_mol_scale(self):
        # K's contribution must not scale with mol_scale
        counts = {"C": 5}
        adduct = {"K": 1}
        ratio_half = _expected_mk_ratio(counts, mol_scale=0.5, k=2, adduct_counts=adduct)
        ratio_full = _expected_mk_ratio(counts, mol_scale=1.0, k=2, adduct_counts=adduct)
        # Difference is only from the molecule part (C5 has no Cl/Br/K/S M+2)
        # so the two ratios differ only by the molecule's M+2 Poisson term
        # which is tiny for C5 — both ratios dominated by K's 0.0673
        assert abs(ratio_full - ratio_half) < 0.01  # small delta, not 0.0673

    def test_cl_adduct_m2_at_full_weight(self):
        counts = {"C": 5}
        adduct = {"Cl": 1}
        ratio = _expected_mk_ratio(counts, mol_scale=0.5, k=2, adduct_counts=adduct)
        # Cl contributes 0.3220 at full weight; molecule adds small poisson + O term
        assert ratio == pytest.approx(0.3220, abs=0.02)

    def test_floor_at_1e4(self):
        ratio = _expected_mk_ratio({}, mol_scale=0.0, k=2)
        assert ratio == pytest.approx(1e-4)


# ---------------------------------------------------------------------------
# detect_isotope_peaks_for_training — co-isolation R_k based detection
# ---------------------------------------------------------------------------

_NM = C13_SPACING  # 1.003355 Da


def _build_spectrum(
    frag_base: float,
    frag_ints: tuple[float, ...],
    prec_mz: float,
    prec_ints: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Build a spectrum with a fragment peak series and precursor isotope peaks.

    Fragment peaks are placed at ``frag_base + i * C13_SPACING``.
    Precursor peaks are placed at ``prec_mz + i * C13_SPACING`` (M+0 at prec_mz,
    M+1 at prec_mz + C13_SPACING, etc.).  The precursor M+k peaks above prec_mz
    serve as the reference from which R_k is estimated.

    Args:
        frag_base: m/z of the monoisotopic fragment peak (M+0).
        frag_ints: Intensities for fragment M+0, M+1, M+2, ...
        prec_mz: Precursor m/z (M+0 position).
        prec_ints: Intensities for precursor M+0, M+1, M+2, ...

    Returns:
        Tuple of (mzs, ints) arrays for the combined spectrum.
    """
    frag_mzs = np.array([frag_base + i * _NM for i in range(len(frag_ints))])
    prec_mzs = np.array([prec_mz + i * _NM for i in range(len(prec_ints))])
    mzs = np.concatenate([frag_mzs, prec_mzs])
    ints = np.concatenate([np.array(frag_ints, dtype=float), np.array(prec_ints, dtype=float)])
    return mzs, ints


class TestDetectIsotopePeaksForTraining:
    """Tests for co-isolation-ratio-based isotope leak detection."""

    # ----- M+1 detection -----

    def test_m1_detected_when_ratio_matches_prec(self):
        """Fragment M+1 is flagged when its ratio matches the precursor R_1."""
        # prec_mz=179.13, frag at 109.05.
        # R_1 = 57/1000 = 0.057; fragment M+1 intensity = 57 → ratio matches.
        prec_mz = 179.13
        mzs, ints = _build_spectrum(
            frag_base=109.05,
            frag_ints=(1000.0, 57.0),
            prec_mz=prec_mz,
            prec_ints=(1000.0, 57.0),  # R_1 = 0.057
        )
        mask = detect_isotope_peaks_for_training(mzs, ints, prec_mz=prec_mz, inst_type="FT")
        assert mask[1], "Fragment M+1 with ratio matching prec R_1 should be detected"

    def test_m1_not_detected_when_no_above_prec_peaks(self):
        """Without any above-prec peaks, R_k cannot be estimated → nothing flagged."""
        prec_mz = 179.13
        # Fragment only — no precursor peaks in spectrum
        mzs = np.array([109.05, 109.05 + _NM])
        ints = np.array([1000.0, 57.0])
        mask = detect_isotope_peaks_for_training(mzs, ints, prec_mz=prec_mz, inst_type="FT")
        assert not mask.any(), "Without above-prec peaks, R_k is unknown → all-False"

    def test_m1_not_detected_when_ratio_far_from_prec(self):
        """Fragment M+1 whose ratio differs greatly from prec R_1 is not flagged."""
        prec_mz = 179.13
        # R_1 = 30/1000 = 0.03; fragment M+1 = 30% → obs/R_k = 10.0 => ratio_hi=10.0, we will push it to 20.0
        mzs, ints = _build_spectrum(
            frag_base=109.05,
            frag_ints=(1000.0, 600.0),
            prec_mz=prec_mz,
            prec_ints=(1000.0, 30.0),  # R_1 = 0.03
        )
        mask = detect_isotope_peaks_for_training(mzs, ints, prec_mz=prec_mz, inst_type="FT")
        assert not mask[1], "Fragment M+1 with obs/R_k=20 >> ratio_hi=10.0 should not be flagged"

    # ----- M+2 detection with Cl precursor -----

    def test_cl_m2_detected_for_fragment_retaining_cl(self):
        """Fragment that retained Cl has M+2 ≈ 32.2%, matching prec R_2 → detected."""
        # [M+Cl]- precursor.  Precursor R_2 = 322/1000 = 0.322 (37Cl dominates).
        # Fragment [frag+Cl]- retains Cl → its M+2 ≈ 32.2% of M+0.
        prec_mz = 121.04
        frag_base = 93.02
        mzs, ints = _build_spectrum(
            frag_base=frag_base,
            frag_ints=(1000.0, 35.0, 322.0),
            prec_mz=prec_mz,
            prec_ints=(1000.0, 35.0, 322.0),  # R_1=0.035, R_2=0.322
        )
        mask = detect_isotope_peaks_for_training(mzs, ints, prec_mz=prec_mz, inst_type="FT")
        assert mask[2], "Cl-retaining fragment M+2 matching prec R_2 should be detected"

    def test_cmf_fragment_not_falsely_flagged(self):
        """CMF fragment [frag-H]- lost Cl: M+2 ≈ 1.2%, far below prec R_2 = 32.2%."""
        # Even though the precursor was [M+Cl]-, this fragment lost Cl via CMF.
        # Its M+2 comes only from 13C (C3 → ~1.2%).
        # prec R_2 = 0.322; fragment ratio = 0.014 -> 0.014 / 0.322 = 0.043 < ratio_lo=0.05. We reduce it to 14.0 (0.014 / 0.322 = 0.043)
        # 14.0 also fails the false-positive M+1 link check (14.0 / 35.0 = 0.40 > 10.0 * 0.035 = 0.35).
        prec_mz = 121.04
        frag_base = 57.03
        mzs, ints = _build_spectrum(
            frag_base=frag_base,
            frag_ints=(1000.0, 35.0, 14.0),
            prec_mz=prec_mz,
            prec_ints=(1000.0, 35.0, 322.0),  # R_2 = 0.322
        )
        mask = detect_isotope_peaks_for_training(mzs, ints, prec_mz=prec_mz, inst_type="FT")
        assert not mask[2], "CMF fragment with tiny M+2 should NOT be flagged (ratio << R_2)"

    # ----- [M+H]+ baseline -----

    def test_mh_plus_m1_detected(self):
        """[M+H]+ M+1 detection works via observed prec R_1."""
        # C9 fragment ~121.1 Da; prec ~241.2 Da.  R_1 = 95/1000 = 0.095.
        # Fragment M+1 = 9.5% → ratio matches R_1.
        prec_mz = 241.19
        mzs, ints = _build_spectrum(
            frag_base=121.10,
            frag_ints=(1000.0, 95.0),
            prec_mz=prec_mz,
            prec_ints=(1000.0, 95.0),  # R_1 = 0.095
        )
        mask = detect_isotope_peaks_for_training(mzs, ints, prec_mz=prec_mz, inst_type="FT")
        assert mask[1], "[M+H]+ M+1 should be detected"

    # ----- instrument gating -----

    def test_qtof_always_returns_false(self):
        prec_mz = 200.0
        mzs, ints = _build_spectrum(
            frag_base=100.0,
            frag_ints=(1000.0, 110.0),
            prec_mz=prec_mz,
            prec_ints=(1000.0, 110.0),
        )
        mask = detect_isotope_peaks_for_training(mzs, ints, prec_mz=prec_mz, inst_type="QTOF")
        assert not mask.any(), "QTOF should never be filtered"


# ---------------------------------------------------------------------------
# detect_isotope_peaks_for_training — aggressive mode (cross-pair consistency)
# ---------------------------------------------------------------------------


class TestDetectIsotopePeaksAggressiveMode:
    """Tests for aggressive=True: R_k estimated from cross-pair consistency.

    The key identity: if co-isolation is present with ratio R, then for any two
    fragment pairs (M_i, M+1_i) and (M_j, M+1_j):

        I(M+1_i) / I(M+1_j) = I(M_i) / I(M_j)   (= R is the same for all i)

    This cross-pair consistency (low CV of r_i = I(M+1_i)/I(M_i)) allows R to
    be estimated from the spectrum itself, without requiring above-prec peaks.
    """

    def _make_consistent_spectrum(
        self,
        R: float,
        prec_mz: float,
        frag_bases: tuple[float, ...],
        m0_ints: tuple[float, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a spectrum with N fragments each having M+1 = R * M+0.

        No precursor M+k peaks above prec_mz are included, so the default mode
        would return all-False.  Aggressive mode should recover R from consistency.
        """
        mz_list, int_list = [], []
        for base, m0 in zip(frag_bases, m0_ints):
            mz_list += [base, base + _NM]
            int_list += [m0, R * m0]
        return np.array(mz_list), np.array(int_list)

    def test_consistent_ratio_detected_in_aggressive_mode(self):
        """Three fragments all with R=0.06 → aggressive mode detects all M+1 peaks."""
        R = 0.06
        prec_mz = 400.0
        mzs, ints = self._make_consistent_spectrum(
            R,
            prec_mz,
            frag_bases=(80.0, 120.0, 160.0),
            m0_ints=(2000.0, 1500.0, 800.0),
        )
        mask = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=prec_mz,
            inst_type="FT",
            aggressive=True,
            aggressive_min_pairs=3,
            aggressive_cv=0.3,
        )
        # Indices 1, 3, 5 are the M+1 peaks (every second element)
        m1_indices = [1, 3, 5]
        assert all(mask[i] for i in m1_indices), (
            "All M+1 peaks should be detected in aggressive mode"
        )

    def test_aggressive_not_triggered_when_above_prec_available(self):
        """When above-prec peaks exist, aggressive fallback is not needed."""
        prec_mz = 300.0
        # Build with above-prec peaks — primary path should handle this (R_1 = 0.07)
        mzs, ints = _build_spectrum(
            frag_base=100.0,
            frag_ints=(1000.0, 70.0),
            prec_mz=prec_mz,
            prec_ints=(2000.0, 140.0),
        )
        mask_default = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=prec_mz,
            inst_type="FT",
            aggressive=False,
        )
        mask_aggressive = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=prec_mz,
            inst_type="FT",
            aggressive=True,
        )
        # Both modes should give the same result when above-prec peaks are present
        np.testing.assert_array_equal(mask_default, mask_aggressive)

    def test_inconsistent_ratios_not_flagged_in_aggressive_mode(self):
        """Fragments with scattered r_i (high CV) → aggressive mode returns all-False."""
        prec_mz = 400.0
        # Three fragments with very different M+1/M+0 ratios → not co-isolation
        mzs = np.array(
            [
                80.0,
                80.0 + _NM,  # r = 0.02
                120.0,
                120.0 + _NM,  # r = 0.20
                160.0,
                160.0 + _NM,  # r = 0.50
            ]
        )
        ints = np.array([2000.0, 40.0, 1500.0, 300.0, 800.0, 400.0])
        mask = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=prec_mz,
            inst_type="FT",
            aggressive=True,
            aggressive_min_pairs=3,
            aggressive_cv=0.3,
        )
        assert not mask.any(), "Inconsistent ratios (high CV) should not be flagged"

    def test_aggressive_requires_min_pairs(self):
        """Fewer pairs than min_pairs → aggressive mode returns all-False."""
        R = 0.06
        prec_mz = 400.0
        # Only 2 pairs, but min_pairs=3
        mzs, ints = self._make_consistent_spectrum(
            R,
            prec_mz,
            frag_bases=(80.0, 120.0),
            m0_ints=(2000.0, 1500.0),
        )
        mask = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=prec_mz,
            inst_type="FT",
            aggressive=True,
            aggressive_min_pairs=3,
        )
        assert not mask.any(), "Fewer than min_pairs should not trigger aggressive detection"


# ---------------------------------------------------------------------------
# detect_isotope_peaks_for_training — formula-aware co-isolation (corrected physics)
# ---------------------------------------------------------------------------


class TestDetectIsotopePeaksFormulaMode:
    """Tests for formula-aware co-isolation detection.

    Physics: the isolation window co-selects fraction f of the M+k precursor
    isotopologue.  Every fragment gains a satellite at +k×C13 with intensity

        I(M+k_frag) = f × R_exp_frag_i × I(M+0_frag)

    At the precursor level:  R_k / R_exp_prec = f  (always ≤ 1.0).
    The old ``min_leak_factor ≥ 1.5`` was impossible; the correct gate is
    ``f ≥ min_coiso_fraction`` (default 0.05).
    """

    @staticmethod
    def _build_coiso_spectrum(
        formula: str,
        prec_type: str,
        prec_mz: float,
        f: float,
        frag_bases: tuple[float, ...],
        m0_ints: tuple[float, ...],
        with_prec_residuals: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a spectrum with physics-correct co-isolation intensities."""
        from fragnnet.utils.formula_utils import parse_formula

        mol_counts = parse_formula(formula)
        adduct_counts, adduct_mass = _parse_adduct_counts(prec_type)
        prec_neutral = prec_mz - adduct_mass

        r_exp_prec1 = _expected_mk_ratio(
            mol_counts, mol_scale=1.0, k=1, adduct_counts=adduct_counts
        )

        mz_list: list[float] = []
        int_list: list[float] = []

        for frag_mz, m0_int in zip(frag_bases, m0_ints):
            mol_scale = max(0.0, min(1.0, (frag_mz - adduct_mass) / prec_neutral))
            r_exp_frag = _expected_mk_ratio(
                mol_counts, mol_scale=mol_scale, k=1, adduct_counts=adduct_counts
            )
            mz_list += [frag_mz, frag_mz + _NM]
            int_list += [m0_int, f * r_exp_frag * m0_int]

        if with_prec_residuals:
            mz_list += [prec_mz, prec_mz + _NM]
            int_list += [1000.0, f * r_exp_prec1 * 1000.0]

        return np.array(mz_list), np.array(int_list)

    def test_genuine_coiso_detected_primary_mode(self):
        """f=0.3 co-isolation detected via precursor residuals in formula mode."""
        mzs, ints = self._build_coiso_spectrum(
            formula="C10H14N2O5",
            prec_type="[M+H]+",
            prec_mz=243.098,
            f=0.3,
            frag_bases=(120.0,),
            m0_ints=(2000.0,),
            with_prec_residuals=True,
        )
        mask = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=243.098,
            inst_type="FT",
            formula="C10H14N2O5",
            prec_type="[M+H]+",
        )
        # Index 1 is the fragment M+1 peak
        assert mask[1], "Fragment M+1 from f=0.3 co-isolation should be detected"
        assert not mask[0], "Fragment M+0 should not be flagged"

    def test_precursor_f_estimator_recovers_coiso_fraction(self):
        """Precursor residuals provide f that can be reused at group level."""
        mzs, ints = self._build_coiso_spectrum(
            formula="C10H14N2O5",
            prec_type="[M+H]+",
            prec_mz=243.098,
            f=0.3,
            frag_bases=(120.0,),
            m0_ints=(2000.0,),
            with_prec_residuals=True,
        )
        f_by_k = estimate_coisolation_fraction_from_precursor(
            mzs,
            ints,
            prec_mz=243.098,
            inst_type="FT",
            formula="C10H14N2O5",
            prec_type="[M+H]+",
        )
        assert f_by_k[1] == pytest.approx(0.3, rel=1e-6)

    def test_precursor_f_estimator_recovers_m1_and_m2_when_envelope_matches(self):
        """Consistent above-precursor M2/M1 residuals recover both f values."""
        from fragnnet.utils.formula_utils import parse_formula

        formula = "C10H14N2O5"
        prec_type = "[M+H]+"
        prec_mz = 243.098
        f = 0.3
        mol_counts = parse_formula(formula)
        adduct_counts, _ = _parse_adduct_counts(prec_type)
        r1 = _expected_mk_ratio(mol_counts, mol_scale=1.0, k=1, adduct_counts=adduct_counts)
        r2 = _expected_mk_ratio(mol_counts, mol_scale=1.0, k=2, adduct_counts=adduct_counts)
        mzs = np.array([prec_mz, prec_mz + _NM, prec_mz + 2 * _NM])
        ints = np.array([1000.0, f * r1 * 1000.0, f * r2 * 1000.0])

        f_by_k = estimate_coisolation_fraction_from_precursor(
            mzs,
            ints,
            prec_mz=prec_mz,
            inst_type="FT",
            formula=formula,
            prec_type=prec_type,
            max_isotope=2,
        )

        assert f_by_k[1] == pytest.approx(f, rel=1e-6)
        assert f_by_k[2] == pytest.approx(f, rel=1e-6)

    def test_precursor_f_estimator_rejects_bad_m2_m1_envelope(self):
        """A stray M+2 residual is rejected by the explicit M2/M1 envelope check."""
        from fragnnet.utils.formula_utils import parse_formula

        formula = "C10H14N2O5"
        prec_type = "[M+H]+"
        prec_mz = 243.098
        mol_counts = parse_formula(formula)
        adduct_counts, _ = _parse_adduct_counts(prec_type)
        r1 = _expected_mk_ratio(mol_counts, mol_scale=1.0, k=1, adduct_counts=adduct_counts)
        r2 = _expected_mk_ratio(mol_counts, mol_scale=1.0, k=2, adduct_counts=adduct_counts)
        mzs = np.array([prec_mz, prec_mz + _NM, prec_mz + 2 * _NM])
        # Both individual f values pass min/max gates, and f2/f1=8 would pass the
        # broad fragment ratio_hi=10 gate.  The tighter precursor M2/M1 envelope
        # check rejects the pair as incoherent above-precursor evidence.
        ints = np.array([1000.0, 0.05 * r1 * 1000.0, 0.4 * r2 * 1000.0])

        f_by_k = estimate_coisolation_fraction_from_precursor(
            mzs,
            ints,
            prec_mz=prec_mz,
            inst_type="FT",
            formula=formula,
            prec_type=prec_type,
            max_isotope=2,
            ratio_hi=10.0,
            precursor_envelope_hi=5.0,
        )

        assert f_by_k == {}

    def test_group_level_f_detects_spectrum_without_precursor_residuals(self):
        """Borrowed group f fills the gap when this spectrum has no precursor f."""
        mzs, ints = self._build_coiso_spectrum(
            formula="C10H14N2O5",
            prec_type="[M+H]+",
            prec_mz=243.098,
            f=0.3,
            frag_bases=(120.0,),
            m0_ints=(2000.0,),
            with_prec_residuals=False,
        )
        mask_without_group_f = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=243.098,
            inst_type="FT",
            formula="C10H14N2O5",
            prec_type="[M+H]+",
            aggressive=False,
        )
        mask_with_group_f = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=243.098,
            inst_type="FT",
            formula="C10H14N2O5",
            prec_type="[M+H]+",
            aggressive=False,
            coiso_fraction_by_k={1: 0.3},
        )
        assert not mask_without_group_f.any(), "No local f and no fallback should not fire"
        assert mask_with_group_f[1], "Group-level f should recover the M+1 coisolation peak"

    def test_coiso_below_min_fraction_not_detected(self):
        """f=0.01 (1%) is below the default min_coiso_fraction=0.05 → all-False."""
        mzs, ints = self._build_coiso_spectrum(
            formula="C10H14N2O5",
            prec_type="[M+H]+",
            prec_mz=243.098,
            f=0.01,
            frag_bases=(120.0,),
            m0_ints=(2000.0,),
            with_prec_residuals=True,
        )
        mask = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=243.098,
            inst_type="FT",
            formula="C10H14N2O5",
            prec_type="[M+H]+",
        )
        assert not mask.any(), "1% co-isolation below min_coiso_fraction=0.05 should not fire"

    def test_genuine_coiso_detected_aggressive_formula(self):
        """f=0.3, no above-prec peaks: aggressive formula mode detects from fragment pairs."""
        mzs, ints = self._build_coiso_spectrum(
            formula="C10H14N2O5",
            prec_type="[M+H]+",
            prec_mz=400.0,
            f=0.3,
            frag_bases=(80.0, 120.0, 160.0),
            m0_ints=(2000.0, 1500.0, 800.0),
            with_prec_residuals=False,
        )
        mask = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=400.0,
            inst_type="FT",
            formula="C10H14N2O5",
            prec_type="[M+H]+",
            aggressive=True,
            aggressive_min_pairs=3,
        )
        m1_indices = [1, 3, 5]
        assert all(mask[i] for i in m1_indices), (
            "All M+1 co-isolation peaks should be detected in aggressive formula mode"
        )

    def test_k1_kept_when_k2_absent(self):
        """k=2 below noise for small molecule: k=1 is still kept (not discarded)."""
        # Small molecule: M+2 residual ≈ f × R_exp_prec[2] × 1000 ≈ very small → absent
        mzs, ints = self._build_coiso_spectrum(
            formula="C5H8O2",
            prec_type="[M+H]+",
            prec_mz=101.060,
            f=0.5,
            frag_bases=(57.0,),
            m0_ints=(2000.0,),
            with_prec_residuals=True,  # only k=1 residual; k=2 too tiny to include
        )
        mask = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=101.060,
            inst_type="FT",
            formula="C5H8O2",
            prec_type="[M+H]+",
            max_isotope=2,
        )
        assert mask[1], "Fragment M+1 should be detected even without k=2 residual peak"

    def test_ratio_lo_20pct_detected(self):
        """Fragment M+1 at 25% of expected R_1 is caught with ratio_lo=0.2."""
        # prec R_1 = 100/1000 = 0.10; fragment M+1 = 2.5% → obs/R_k = 0.25.
        # Below old ratio_lo=0.5 but above new ratio_lo=0.2.
        prec_mz = 300.0
        mzs, ints = _build_spectrum(
            frag_base=150.0,
            frag_ints=(1000.0, 25.0),
            prec_mz=prec_mz,
            prec_ints=(1000.0, 100.0),
        )
        mask = detect_isotope_peaks_for_training(mzs, ints, prec_mz=prec_mz, inst_type="FT")
        assert mask[1], "Fragment M+1 at 25% of R_k should be flagged with ratio_lo=0.2"

    def test_two_pair_aggressive_detected(self):
        """Two consistent pairs (aggressive_min_pairs=2) are enough to detect co-isolation."""
        R = 0.07
        prec_mz = 350.0
        # Only 2 fragment pairs; old default (min_pairs=3) would miss this.
        mzs = np.array(
            [
                80.0,
                80.0 + _NM,
                130.0,
                130.0 + _NM,
            ]
        )
        ints = np.array([2000.0, R * 2000.0, 1500.0, R * 1500.0])
        mask = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=prec_mz,
            inst_type="FT",
            aggressive=True,
        )
        assert mask[1] and mask[3], "Two consistent pairs should be enough for aggressive detection"

    def test_single_pair_formula_fallback(self):
        """Single fragment pair with large f detected via formula single-pair fallback."""
        # f=0.4 >> aggressive_single_pair_min_f=0.15; only 1 pair visible.
        from fragnnet.utils.formula_utils import parse_formula
        from fragnnet.utils.isotope_utils import _parse_adduct_counts

        formula = "C10H14N2O5"
        prec_type = "[M+H]+"
        prec_mz = 243.098
        mol_counts = parse_formula(formula)
        adduct_counts, adduct_mass = _parse_adduct_counts(prec_type)
        prec_neutral = prec_mz - adduct_mass
        f = 0.4
        frag_mz = 120.0
        mol_scale = max(0.0, min(1.0, (frag_mz - adduct_mass) / prec_neutral))
        r_exp_frag = _expected_mk_ratio(mol_counts, mol_scale, k=1, adduct_counts=adduct_counts)

        # Only one fragment pair, no above-prec peaks.
        mzs = np.array([frag_mz, frag_mz + _NM])
        ints = np.array([2000.0, f * r_exp_frag * 2000.0])
        mask = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=prec_mz,
            inst_type="FT",
            formula=formula,
            prec_type=prec_type,
            aggressive=True,
        )
        assert mask[1], "Single-pair formula fallback should detect large-f co-isolation"

    def test_single_pair_formula_fallback_small_f_rejected(self):
        """Single fragment pair with f=0.01 (below threshold) is not flagged."""
        from fragnnet.utils.formula_utils import parse_formula
        from fragnnet.utils.isotope_utils import _parse_adduct_counts

        formula = "C10H14N2O5"
        prec_type = "[M+H]+"
        prec_mz = 243.098
        mol_counts = parse_formula(formula)
        adduct_counts, adduct_mass = _parse_adduct_counts(prec_type)
        prec_neutral = prec_mz - adduct_mass
        f = 0.01  # below aggressive_single_pair_min_f=0.05
        frag_mz = 120.0
        mol_scale = max(0.0, min(1.0, (frag_mz - adduct_mass) / prec_neutral))
        r_exp_frag = _expected_mk_ratio(mol_counts, mol_scale, k=1, adduct_counts=adduct_counts)

        mzs = np.array([frag_mz, frag_mz + _NM])
        ints = np.array([2000.0, f * r_exp_frag * 2000.0])
        mask = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=prec_mz,
            inst_type="FT",
            formula=formula,
            prec_type=prec_type,
            aggressive=True,
        )
        assert not mask.any(), "Single-pair with f below threshold should not be flagged"

    def test_single_pair_fallback_uses_denominator_corrected_k1_math(self):
        """For f≈1, M+1 precursor neutral-loss contribution must not under-estimate f."""
        from fragnnet.utils.formula_utils import parse_formula

        formula = "C10H22"
        frag_formula = "C5H11"
        prec_mz = 143.179
        frag_mz = 71.086
        f = 1.0

        r_prec = _expected_mk_ratio(parse_formula(formula), 1.0, k=1)
        r_frag = _expected_mk_ratio(parse_formula(frag_formula), 1.0, k=1)
        obs = (f * r_frag) / (1.0 + f * (r_prec - r_frag))

        mzs = np.array([frag_mz, frag_mz + _NM])
        ints = np.array([2000.0, obs * 2000.0])
        mask = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=prec_mz,
            inst_type="FT",
            formula=formula,
            prec_type="[M+H]+",
            aggressive=True,
            aggressive_single_pair_min_f=0.99,
            dag_formula_mzs=np.array([frag_mz]),
            dag_formula_strs=np.array([frag_formula], dtype=object),
        )
        assert mask[1], "Corrected k=1 fallback should recover f≈1 and flag the isotope peak"

    def test_dag_mono_protection_keeps_plausible_monoisotopic_peak(self):
        """A shifted peak can be protected when the DAG also explains it as monoisotopic."""
        from fragnnet.utils.formula_utils import parse_formula

        formula = "C10H22"
        frag_formula = "C5H11"
        competing_formula = "C4H9N"
        prec_mz = 143.179
        frag_mz = 71.086
        isotope_mz = frag_mz + _NM

        r_prec = _expected_mk_ratio(parse_formula(formula), 1.0, k=1)
        r_frag = _expected_mk_ratio(parse_formula(frag_formula), 1.0, k=1)
        obs = r_frag / (1.0 + (r_prec - r_frag))

        mzs = np.array([frag_mz, isotope_mz])
        ints = np.array([2000.0, obs * 2000.0])
        kwargs = {
            "mzs": mzs,
            "ints": ints,
            "prec_mz": prec_mz,
            "inst_type": "FT",
            "formula": formula,
            "prec_type": "[M+H]+",
            "aggressive": True,
            "aggressive_single_pair_min_f": 0.99,
            "dag_formula_mzs": np.array([frag_mz, isotope_mz]),
            "dag_formula_strs": np.array([frag_formula, competing_formula], dtype=object),
        }
        mask_unprotected = detect_isotope_peaks_for_training(**kwargs)
        mask_protected = detect_isotope_peaks_for_training(**kwargs, protect_dag_mono=True)

        assert mask_unprotected[1]
        assert not mask_protected[1]


# ---------------------------------------------------------------------------
# detect_isotope_peaks_formula_aware
# ---------------------------------------------------------------------------


class TestIsotopeOffsetsFromFormula:
    """Tests for pyopenms-backed isotopologue offset computation."""

    def test_carbon_only_m1_dominates(self):
        """For a carbon-only formula, M+1 should be the largest offset."""
        from fragnnet.utils.formula_utils import isotope_offsets_from_formula, parse_formula

        offsets = isotope_offsets_from_formula(parse_formula("C10H14N2O5"))
        assert len(offsets) >= 1
        # First offset should be ~1.003 Da (13C spacing)
        assert offsets[0][0] == pytest.approx(1.003355, abs=0.001)

    def test_bromine_m2_larger_than_m1(self):
        """For C10H8Br2, M+2 relative intensity should exceed M+1."""
        from fragnnet.utils.formula_utils import isotope_offsets_from_formula, parse_formula

        offsets = isotope_offsets_from_formula(parse_formula("C10H8Br2"))
        # Build offset → intensity map
        offset_map = {round(off, 1): rel for off, rel in offsets}
        m1_rel = offset_map.get(1.0, 0.0)
        m2_rel = offset_map.get(2.0, 0.0)
        assert m2_rel > m1_rel, "Br2 M+2 should dominate over M+1"
        assert m2_rel > 0.4, "Br2 M+2 relative intensity should be ~0.46"

    def test_empty_formula_returns_empty(self):
        from fragnnet.utils.formula_utils import isotope_offsets_from_formula

        assert isotope_offsets_from_formula({}) == []

    def test_threshold_filters_low_peaks(self):
        from fragnnet.utils.formula_utils import isotope_offsets_from_formula, parse_formula

        ec = parse_formula("C6H12O6")
        offsets_loose = isotope_offsets_from_formula(ec, threshold=0.0001)
        offsets_strict = isotope_offsets_from_formula(ec, threshold=0.05)
        assert len(offsets_loose) > len(offsets_strict)


# ---------------------------------------------------------------------------
# detect_isotope_peaks_formula_aware
# ---------------------------------------------------------------------------


class TestDetectIsotopePeaksFormulaAware:
    """Tests for formula-aware isotope peak detection via pyopenms offsets."""

    def test_basic_carbon_m1_detected(self):
        """M+1 of a carbon-rich molecule at correct spacing is flagged."""
        from fragnnet.utils.formula_utils import parse_formula
        from fragnnet.utils.isotope_utils import detect_isotope_peaks_formula_aware

        ec = parse_formula("C10H14N2O5")
        mono = 242.090
        mzs = np.array([mono, mono + 1.003355])
        ints = np.array([1000.0, 100.0])
        mask = detect_isotope_peaks_formula_aware(mzs, ints, ec)
        assert mask[1], "M+1 peak at 13C spacing should be flagged"
        assert not mask[0], "Monoisotopic peak should not be flagged"

    def test_bromine_m2_detected(self):
        """For C10H8Br2, the large M+2 peak should be flagged as isotope."""
        from fragnnet.utils.formula_utils import parse_formula
        from fragnnet.utils.isotope_utils import detect_isotope_peaks_formula_aware

        ec = parse_formula("C10H8Br2")
        mono = 285.899
        mzs = np.array([mono, mono + 1.003, mono + 2.007, mono + 3.010, mono + 4.013])
        ints = np.array([100.0, 5.0, 97.0, 2.0, 24.0])
        mask = detect_isotope_peaks_formula_aware(mzs, ints, ec, mz_tol=0.02)
        assert mask[2], "M+2 (large Br peak) should be detected as isotope"

    def test_intensity_gate_rejects_noise(self):
        """A peak at the right offset but wrong intensity is not flagged when check_intensity=True."""
        from fragnnet.utils.formula_utils import parse_formula
        from fragnnet.utils.isotope_utils import detect_isotope_peaks_formula_aware

        ec = parse_formula("C6H12O6")
        mono = 180.063
        # Noise at M+1 position that is MORE intense than monoisotopic — not a real isotope
        mzs = np.array([mono, mono + 1.003])
        ints = np.array([100.0, 500.0])  # noise is 5× monoisotopic
        mask = detect_isotope_peaks_formula_aware(mzs, ints, ec, check_intensity=True)
        assert not mask[1], "A peak more intense than its monoisotopic should not be flagged"

    def test_empty_spectrum(self):
        from fragnnet.utils.formula_utils import parse_formula
        from fragnnet.utils.isotope_utils import detect_isotope_peaks_formula_aware

        ec = parse_formula("C6H12O6")
        mask = detect_isotope_peaks_formula_aware(np.array([]), np.array([]), ec)
        assert mask.shape == (0,)

    def test_wrong_shape_raises(self):
        from fragnnet.utils.formula_utils import parse_formula
        from fragnnet.utils.isotope_utils import detect_isotope_peaks_formula_aware

        ec = parse_formula("C6H12O6")
        with pytest.raises(ValueError):
            detect_isotope_peaks_formula_aware(np.array([100.0, 101.0]), np.array([1.0]), ec)


class TestTrainingIsotopeCleanup:
    """Tests for monoisotopic target cleanup masks."""

    def test_regular_isotope_peak_removed(self):
        mono = 120.0
        mzs = np.array([mono, mono + C13_SPACING, 150.0])
        ints = np.array([1000.0, 80.0, 500.0])
        mask = detect_isotope_peaks_for_training_cleanup(
            mzs,
            ints,
            prec_mz=243.098,
            inst_type="FT",
            formula="C10H14N2O5",
            prec_type="[M+H]+",
            remove_regular_isotopes=True,
            remove_coisolation=False,
        )
        assert mask.tolist() == [False, True, False]

    def test_group_level_coiso_peak_removed_without_regular_path(self):
        mzs, ints = TestDetectIsotopePeaksFormulaMode._build_coiso_spectrum(
            formula="C10H14N2O5",
            prec_type="[M+H]+",
            prec_mz=243.098,
            f=0.3,
            frag_bases=(120.0,),
            m0_ints=(2000.0,),
            with_prec_residuals=False,
        )
        mask = detect_isotope_peaks_for_training_cleanup(
            mzs,
            ints,
            prec_mz=243.098,
            inst_type="FT",
            formula="C10H14N2O5",
            prec_type="[M+H]+",
            remove_regular_isotopes=False,
            remove_coisolation=True,
            coiso_fraction_by_k={1: 0.3},
        )
        assert mask.tolist() == [False, True]
