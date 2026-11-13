"""Unit tests for fragnnet.utils.formula_utils module.

Tests for formula parsing, isotopic peak generation, and mass calculations.
"""

import re

import numpy as np
import pytest
import rdkit.Chem as Chem
from pyteomics.mass import Composition

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


class TestPrecTypeDiffs:
    """PREC_TYPE_TO_MASS_DIFF must be the mass counterpart of PREC_TYPE_TO_FORMULA_DIFF.

    Both are added to a *fragment* (formula and mass respectively) to produce the predicted ion,
    so a disagreement between them silently shifts predicted m/z away from predicted formulas.
    """

    @staticmethod
    def _mass_from_formula_diff(prec_type: str) -> float:
        """Return the mass diff implied by the composition diff, including the charge electron."""
        comp = fu.PREC_TYPE_TO_COMP_DIFF[prec_type]
        mass = sum(fu.MASS(element) * count for element, count in comp.items())
        return mass - fu.ELECTRON_MASS if prec_type.endswith("+") else mass + fu.ELECTRON_MASS

    @pytest.mark.parametrize("prec_type", sorted(fu.PREC_TYPE_TO_FORMULA_DIFF))
    def test_mass_diff_matches_formula_diff(self, prec_type):
        assert fu.PREC_TYPE_TO_MASS_DIFF[prec_type] == pytest.approx(
            self._mass_from_formula_diff(prec_type), abs=1e-6
        )

    def test_neutral_loss_adduct_values(self):
        """Known values: the loss is applied on top of the ionizing adduct."""
        h_plus = fu.PREC_TYPE_TO_MASS_DIFF["[M+H]+"]
        minus_h = fu.PREC_TYPE_TO_MASS_DIFF["[M-H]-"]
        nh3 = fu.N_MASS + 3 * fu.H_MASS
        co2 = fu.C_MASS + 2 * fu.O_MASS
        assert fu.PREC_TYPE_TO_MASS_DIFF["[M+H-NH3]+"] == pytest.approx(h_plus - nh3, abs=1e-6)
        assert fu.PREC_TYPE_TO_MASS_DIFF["[M-H-CO2]-"] == pytest.approx(minus_h - co2, abs=1e-6)

    def test_radical_cation_diff_is_electron_only(self):
        """Edge case: [M]+ adds no atoms, so its diff is a bare electron loss."""
        assert fu.PREC_TYPE_TO_FORMULA_DIFF["[M]+"] == ""
        assert fu.PREC_TYPE_TO_MASS_DIFF["[M]+"] == pytest.approx(-fu.ELECTRON_MASS, abs=1e-12)


def _nonzero(delta_formula: str) -> dict[str, int]:
    return {el: n for el, n in Composition(formula=delta_formula).items() if n != 0}


def _parse_signed(rendered: str) -> dict[str, int]:
    """Read a signed rendering back into element counts, independently of the renderer."""
    counts: dict[str, int] = {}
    for sign, body in re.findall(r"([+-])([A-Za-z0-9]*)", rendered):
        mult = 1 if sign == "+" else -1
        for element, num in re.findall(r"([A-Z][a-z]?)(\d*)", body):
            counts[element] = counts.get(element, 0) + mult * int(num or 1)
    return counts


class TestFormulaDeltaToSignedStr:
    """Signed rendering of a precursor delta, for fragment annotation."""

    @pytest.mark.parametrize(
        ("delta", "expected"),
        [
            ("H1", "+H"),
            ("H-1", "-H"),
            ("Na1", "+Na"),
            ("Cl1", "+Cl"),
            ("C1H1O2", "+CHO2"),
            ("H-1C-1O-2", "-CHO2"),
            ("N1H4", "+H4N"),
            ("", ""),
            # A delta that both removes and adds atoms keeps one group per sign, losses
            # first, matching how an adduct name is written. Synthetic: no shipped
            # precursor delta mixes signs (the one that did, [M-H+2i]-, was an iodine
            # misparse of NIST's isotope-selected label and is no longer supported).
            ("H-1Na1", "-H+Na"),
        ],
    )
    def test_rendering(self, delta, expected):
        assert fu.formula_delta_to_signed_str(delta) == expected

    def test_every_shipped_delta_round_trips(self):
        """No shipped adduct may lose, invent or flip an atom in its rendering."""
        for table in (fu.PREC_TYPE_TO_FORMULA_DIFF, fu.PREC_TYPE_TO_CMF_FORMULA_DIFF):
            for prec_type, delta in table.items():
                rendered = fu.formula_delta_to_signed_str(delta)
                assert rendered == "" or rendered[0] in "+-", (prec_type, rendered)
                assert _parse_signed(rendered) == _nonzero(delta), (prec_type, rendered)

    def test_mixed_sign_delta_keeps_both_groups(self):
        assert fu.formula_delta_to_signed_str("C1H-1") == "-H+C"


class TestFormulaRowPrecDeltas:
    """CRF/CMF split across the batch-concatenated formula rows."""

    def test_crf_only_layout(self):
        # 1 sample, NULL + 3 CRF rows. The NULL row stands for no fragment.
        deltas = fu.formula_row_prec_deltas(
            ["[M+H]+"], [0, 4], include_cmf=False, max_h_transfer=4
        )
        assert deltas.tolist() == ["", "H1", "H1", "H1"]

    def test_negative_mode_delta(self):
        deltas = fu.formula_row_prec_deltas(
            ["[M-H]-"], [0, 3], include_cmf=False, max_h_transfer=4
        )
        assert deltas.tolist() == ["", "H-1", "H-1"]

    def test_cmf_half_gets_the_migration_delta(self):
        # NULL + CRF[1:2] + CMF[1:2] = 5 rows; [M+Na]+ keeps both channels.
        deltas = fu.formula_row_prec_deltas(
            ["[M+Na]+"], [0, 5], include_cmf=True, max_h_transfer=4
        )
        assert deltas.tolist() == ["", "Na1", "Na1", "H1", "H1"]

    def test_cmf_ignored_when_it_duplicates_a_crf_h_shift(self):
        # [M+H]+ CMF is the CRF state at delta_h = -1, so the dataset never emits it.
        assert fu.cmf_duplicate_crf_h_shift("[M+H]+", 4) == -1
        deltas = fu.formula_row_prec_deltas(
            ["[M+H]+"], [0, 5], include_cmf=True, max_h_transfer=4
        )
        assert deltas.tolist() == ["", "H1", "H1", "H1", "H1"]

    def test_cmf_emitted_when_h_transfer_cannot_reach_it(self):
        # With no hydrogen transfer the CRF ladder never reaches the CMF state, so the
        # dataset does emit the CMF half even for [M+H]+.
        assert fu.cmf_duplicate_crf_h_shift("[M+H]+", 0) is None
        deltas = fu.formula_row_prec_deltas(
            ["[M+H]+"], [0, 5], include_cmf=True, max_h_transfer=0
        )
        assert deltas.tolist() == ["", "H1", "H1", "", ""]

    def test_per_sample_precursor_across_a_batch(self):
        deltas = fu.formula_row_prec_deltas(
            ["[M+H]+", "[M-H]-"], [0, 3, 6], include_cmf=False, max_h_transfer=4
        )
        assert deltas.tolist() == ["", "H1", "H1", "", "H-1", "H-1"]

    def test_cumsizes_length_is_validated(self):
        with pytest.raises(ValueError, match="batch_size \\+ 1"):
            fu.formula_row_prec_deltas(
                ["[M+H]+", "[M-H]-"], [0, 3], include_cmf=False, max_h_transfer=4
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
