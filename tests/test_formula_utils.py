"""Unit tests for fragnnet.utils.formula_utils module.

Tests for formula parsing, isotopic peak generation, and mass calculations.
"""

import numpy as np
import pytest
import rdkit.Chem as Chem

from fragnnet.utils import formula_utils as fu


class TestParseFormula:
    """Parsing and element utilities."""

    def test_parse_formula_simple(self):
        result = fu.parse_formula("C6H12O6")
        assert result == {"C": 6, "H": 12, "O": 6}

    def test_parse_formula_invalid_trailing(self):
        with pytest.raises(ValueError):
            fu.parse_formula("CH4*")

    def test_get_elements_set(self):
        elems = fu.get_elements_set("C2H6O")
        assert elems == {"C", "H", "O"}


class TestMassAndIsotopes:
    """Mass table and isotope-related helpers."""

    def test_mass_matches_periodic_table(self):
        expected = Chem.GetPeriodicTable().GetMostCommonIsotopeMass("H")
        assert fu.MASS("H") == pytest.approx(expected, rel=1e-8, abs=1e-8)

class TestPeaks:
    """Peak generation via get_peaks_for_formula (pyopenms-backed)."""

    def test_probs_sum_to_one(self):
        masses, probs = fu.get_peaks_for_formula({"C": 6, "H": 12, "O": 6}, threshold=0.0)
        assert sum(probs) == pytest.approx(1.0, rel=1e-4, abs=1e-4)

    def test_monoisotopic_is_most_abundant_small_molecule(self):
        masses, probs = fu.get_peaks_for_formula({"C": 6, "H": 12, "O": 6})
        assert probs[0] == max(probs)

    def test_bromine_m2_dominates(self):
        # For C10H8Br2, M+2 relative intensity should exceed M+1
        masses, probs = fu.get_peaks_for_formula({"C": 10, "H": 8, "Br": 2})
        sorted_pairs = sorted(zip(masses, probs))
        m2_prob = sorted_pairs[2][1]
        m1_prob = sorted_pairs[1][1]
        assert m2_prob > m1_prob

    def test_empty_raises(self):
        with pytest.raises((ValueError, Exception)):
            fu.get_peaks_for_formula({})

    def test_get_peaks_for_formula(self):
        masses, probs = fu.get_peaks_for_formula({"H": 2, "O": 1}, threshold=0.0)
        assert len(masses) == len(probs) > 0
        assert sum(probs) == pytest.approx(1.0, rel=1e-3, abs=1e-3)

    def test_formula_to_peak_mzs_no_isotopes(self):
        peak_mzs = fu.formula_to_peak_mzs("H2O", prec_type="[M+H]+", isotopes=False)
        assert len(peak_mzs) == 1
        expected = (
            2 * Chem.GetPeriodicTable().GetMostCommonIsotopeMass("H")
            + Chem.GetPeriodicTable().GetMostCommonIsotopeMass("O")
            + fu.PREC_TYPE_TO_MASS_DIFF["[M+H]+"]
        )
        assert peak_mzs[0] == pytest.approx(expected, rel=1e-6, abs=1e-6)

    def test_formula_to_peak_mzs_with_probs(self):
        peak_mzs, peak_probs = fu.formula_to_peak_mzs(
            "H2",
            prec_type="[M+H]+",
            isotopes=True,
            return_probs=True,
        )
        assert len(peak_mzs) == len(peak_probs) > 0
        assert sum(peak_probs) == pytest.approx(1.0, rel=5e-4, abs=5e-4)


class TestHillNotation:
    """Hill notation helper."""

    def test_get_formulae_hill_notation(self):
        result = fu.get_formulae_hill_notation({"O": 1, "H": 2})
        assert result == "H2O"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
