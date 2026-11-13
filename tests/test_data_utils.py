"""Unit tests for fragnnet.utils.data_utils module.

Tests for normalization, tokenization, charge parsing, and energy conversions.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fragnnet.utils import data_utils as du


def test_par_apply_uses_unordered_generator():
    """Request unordered joblib results without changing the existing defaults."""
    with patch.object(du.joblib, "Parallel") as parallel_cls:
        du.par_apply([1], str, return_as_generator=True, n_jobs=2, unordered=True)

    assert parallel_cls.call_args.kwargs["return_as"] == "generator_unordered"


def test_par_apply_rejects_unordered_list():
    """Unordered output is only defined for streaming results."""
    with pytest.raises(ValueError, match="return_as_generator"):
        du.par_apply([1], str, unordered=True)


def test_par_apply_falls_back_for_older_joblib():
    """A joblib/backend combo that rejects generator_unordered still streams ordered results."""
    ordered_parallel = MagicMock(return_value=["result"])
    with (
        patch.object(
            du.joblib,
            "Parallel",
            side_effect=[ValueError("unsupported return_as"), ordered_parallel],
        ) as parallel_cls,
        pytest.warns(RuntimeWarning, match="falling back"),
    ):
        result = du.par_apply([1], str, return_as_generator=True, n_jobs=2, unordered=True)

    assert result == ["result"]
    assert parallel_cls.call_args_list[1].kwargs["return_as"] == "generator"


class TestNormalizeInts:
    """Tests for normalize_ints."""

    def test_normalize_basic(self):
        vals = [1.0, 2.0, 3.0]
        norm = du.normalize_ints(vals)
        assert pytest.approx(sum(norm), rel=1e-9) == 1.0
        assert norm == [pytest.approx(1 / 6), pytest.approx(2 / 6), pytest.approx(3 / 6)]


class TestFormulaMass:
    """Tests for formula mass data used by subformula filtering."""

    def test_formula_to_mass_uses_supported_monoisotopic_masses(self):
        expected_mass = du.ELEMENT_TO_MASS["C"] + 4 * du.ELEMENT_TO_MASS["H"]

        assert du.formula_to_mass("CH4") == pytest.approx(expected_mass)


class TestSplitSmiles:
    """Tests for split_smiles tokenizer."""

    @pytest.mark.parametrize(
        "smiles,expected",
        [
            ("CCO", ["C", "C", "O"]),
            ("C1=CC=CC=C1", ["C", "1", "=", "C", "C", "=", "C", "C", "=", "C", "1"]),
            ("[NH4+]", ["[NH4+]"]),
            ("C12CC1CC2", ["C", "1", "2", "C", "C", "1", "C", "C", "2"]),
        ],
    )
    def test_split_smiles_tokens(self, smiles, expected):
        tokens = du.split_smiles(smiles)
        assert tokens == expected


class TestListReplace:
    """Tests for list_replace mapping."""

    def test_mapping(self):
        vals = [0, 1, 2]
        mapping = {0: "a", 1: "b", 2: "c"}
        assert du.list_replace(vals, mapping) == ["a", "b", "c"]


class TestParseAceNce:
    """Tests for collision energy string parsing."""

    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("10", 10.0),
            ("10 eV", 10.0),
            ("NCE=30% 15eV", 15.0),
            ("25HCD", 25.0),
            ("CE 20", 20.0),
            ("bad", np.nan),
        ],
    )
    def test_parse_ace_str(self, input_str, expected):
        result = du.parse_ace_str(input_str)
        if np.isnan(expected):
            assert np.isnan(result)
        else:
            assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("NCE=30% 15eV", 30.0),
            ("NCE=25%", 25.0),
            ("40% resonant relative/normalized", 40.0),
            ("50(NCE)", 50.0),
            ("HCD (NCE 35%)", 35.0),
            ("bad", np.nan),
        ],
    )
    def test_parse_nce_str(self, input_str, expected):
        result = du.parse_nce_str(input_str)
        if np.isnan(expected):
            assert np.isnan(result)
        else:
            assert result == pytest.approx(expected)


class TestChargeAndEnergy:
    """Tests for charge parsing and energy conversions."""

    def test_get_charge(self):
        assert du.get_charge("[M+H]+") == 1
        assert du.get_charge("[M+H]2+") == 2
        assert du.get_charge("[M-H]-") == -1
        assert du.get_charge("EI") == 1

    def test_nce_ace_roundtrip(self):
        nce = 30.0
        charge = 2  # factor 0.9
        prec_mz = 500.0
        ace = du.nce_to_ace_helper(nce, charge, prec_mz)
        assert ace == pytest.approx(27.0)
        back_nce = du.ace_to_nce_helper(ace, charge, prec_mz)
        assert back_nce == pytest.approx(nce)


class TestNormalizationHelpers:
    """Tests for min-max and z-score normalization helpers."""

    def test_min_max_roundtrip(self):
        val = 5.0
        mn, mx = 0.0, 10.0
        norm = du.min_max_normalize(val, mn, mx)
        assert norm == pytest.approx(0.5, abs=1e-6)
        denorm = du.min_max_denormalize(norm, mn, mx)
        assert denorm == pytest.approx(val)

    def test_zscore_roundtrip(self):
        val = 5.0
        mean, std = 2.0, 3.0
        norm = du.zscore_normalize(val, mean, std)
        assert norm == pytest.approx(1.0, rel=1e-7)
        denorm = du.zscore_denormalize(norm, mean, std)
        assert denorm == pytest.approx(val)


class TestMolFromSmiles:
    """Tests for mol_from_smiles."""

    def test_valid_smiles_returns_mol(self):
        mol = du.mol_from_smiles("CCO", ml_standardize=False)
        assert mol is not None and not (isinstance(mol, float) and np.isnan(mol))

    def test_invalid_smiles_returns_nan(self):
        result = du.mol_from_smiles("not_a_smiles", ml_standardize=False)
        assert isinstance(result, float) and np.isnan(result)

    def test_unparseable_smiles_returns_nan(self):
        # "!!" is not valid SMILES — RDKit returns None which mol_from_smiles converts to nan
        result = du.mol_from_smiles("!!", ml_standardize=False)
        assert isinstance(result, float) and np.isnan(result)


class TestMolProperties:
    """Tests for mol_to_* property extractors."""

    @pytest.fixture
    def ethanol(self):
        return du.mol_from_smiles("CCO", ml_standardize=False)

    @pytest.fixture
    def aspirin(self):
        return du.mol_from_smiles("CC(=O)OC1=CC=CC=C1C(=O)O", ml_standardize=False)

    def test_mol_to_formula_ethanol(self, ethanol):
        assert du.mol_to_formula(ethanol) == "C2H6O"

    def test_mol_to_formula_aspirin(self, aspirin):
        assert du.mol_to_formula(aspirin) == "C9H8O4"

    def test_mol_to_mol_weight_exact_ethanol(self, ethanol):
        mw = du.mol_to_mol_weight(ethanol, exact=True)
        assert mw == pytest.approx(46.0418, abs=1e-3)

    def test_mol_to_mol_weight_average_ethanol(self, ethanol):
        mw = du.mol_to_mol_weight(ethanol, exact=False)
        assert mw == pytest.approx(46.068, abs=0.01)

    def test_mol_to_num_atoms_ethanol(self, ethanol):
        assert du.mol_to_num_atoms(ethanol) == 3  # C, C, O (heavy only)

    def test_mol_to_num_bonds_ethanol(self, ethanol):
        assert du.mol_to_num_bonds(ethanol) == 2

    def test_mol_to_charge_neutral(self, ethanol):
        assert du.mol_to_charge(ethanol) == 0

    def test_mol_to_num_radicals_none(self, ethanol):
        assert du.mol_to_num_radicals(ethanol) == 0

    def test_mol_to_inchikey_length(self, ethanol):
        ik = du.mol_to_inchikey(ethanol)
        assert isinstance(ik, str) and len(ik) == 27  # standard InChIKey length

    def test_mol_to_inchikey_s_length(self, ethanol):
        iks = du.mol_to_inchikey_s(ethanol)
        assert isinstance(iks, str) and len(iks) == 14


class TestCheckMolHelpers:
    """Tests for check_neutral_charge and check_single_mol."""

    def test_neutral_molecule_is_neutral(self):
        mol = du.mol_from_smiles("CCO", ml_standardize=False)
        assert du.check_neutral_charge(mol) is True

    def test_single_connected_mol(self):
        mol = du.mol_from_smiles("CCO", ml_standardize=False)
        assert du.check_single_mol(mol) is True

    def test_disconnected_mol_not_single(self):
        # Two disconnected fragments: [Na+].[Cl-]
        mol = du.mol_from_smiles("[Na+].[Cl-]", ml_standardize=False)
        assert du.check_single_mol(mol) is False


class TestGetMurckoScaffold:
    """Tests for get_murcko_scaffold."""

    def test_benzene_scaffold(self):
        mol = du.mol_from_smiles("c1ccccc1", ml_standardize=False)
        scaffold = du.get_murcko_scaffold(mol)
        assert isinstance(scaffold, str) and len(scaffold) > 0

    def test_acyclic_molecule_gives_empty_scaffold(self):
        mol = du.mol_from_smiles("CCCC", ml_standardize=False)
        scaffold = du.get_murcko_scaffold(mol)
        # Acyclic molecules have no ring scaffold → empty string
        assert scaffold == ""

    def test_nan_input_returns_nan(self):
        result = du.get_murcko_scaffold(np.nan)
        assert isinstance(result, float) and np.isnan(result)


class TestParsePeaksStr:
    """Tests for parse_peaks_str."""

    def test_basic_peaks(self):
        peaks_str = "100.0 500\n200.5 1000"
        result = du.parse_peaks_str(peaks_str)
        assert result == [("100.0", "500"), ("200.5", "1000")]

    def test_nan_input_returns_nan(self):
        result = du.parse_peaks_str(np.nan)
        assert isinstance(result, float) and np.isnan(result)

    def test_empty_lines_skipped(self):
        peaks_str = "100.0 500\n\n200.5 1000\n"
        result = du.parse_peaks_str(peaks_str)
        assert len(result) == 2


class TestCombineFormulae:
    """Tests for combine_formulae."""

    def test_combine_two_formulae(self):
        result = du.combine_formulae("C2H6O", "CH4")
        assert "C" in result and "H" in result

    def test_combine_returns_string(self):
        # Just verify the result is a non-empty string; exact content depends
        # on pyteomics Composition parsing behaviour.
        result = du.combine_formulae("C2H6O", "CH4")
        assert isinstance(result, str) and len(result) > 0

    def test_combine_exact_counts(self):
        assert du.combine_formulae("C2H6O", "CH4") == "C3H10O1"

    @pytest.mark.parametrize(
        ("neutral", "delta", "expected"),
        [
            # Digit-free formulas in the one-letter amino-acid alphabet: pyteomics reads a
            # positional string as a peptide sequence, so these used to come back as
            # cysteine/pyrrolysine compositions (C -> C3H8N1O2S1).
            ("C", "H1", "C1H1"),
            ("N", "H1", "H1N1"),
            ("O", "H1", "H1O1"),
            ("CO", "H1", "C1H1O1"),
            ("CN", "H1", "C1H1N1"),
            ("CNO", "H1", "C1H1N1O1"),
            # Negative deltas ([M-H]-) and multi-element adducts must also survive.
            ("C7H6O2", "H-1", "C7H5O2"),
            ("C", "Na1", "C1Na1"),
            ("CO", "", "C1O1"),
        ],
    )
    def test_amino_acid_letter_formulae_are_not_peptides(self, neutral, delta, expected):
        assert du.combine_formulae(neutral, delta) == expected

    def test_composition_argument_is_accepted(self):
        from pyteomics.mass import Composition

        assert du.combine_formulae(Composition(formula="CO"), "H1") == "C1H1O1"

    def test_dict_argument_is_accepted(self):
        # The isinstance(str) branch must not be the only one that normalises: a plain
        # mapping still has to reach Composition rather than hitting dict + dict.
        assert du.combine_formulae({"C": 1, "O": 1}, "H1") == "C1H1O1"


class TestParseAnnotationsPrecursorParsing:
    """`parse_annotations` parses the molecular formula, not a peptide sequence.

    Hill notation drops the "1" on count-one elements, so a molecule whose every element
    appears once ("CHN", "CHNO") is a digit-free string in the one-letter amino-acid
    alphabet. Read positionally, pyteomics takes it as a sequence: the precursor, and every
    product and neutral loss derived from it, come back as a peptide's composition.
    """

    @staticmethod
    def _row(formula, prec_type="[M+H]+"):
        return {
            "peaks": '50.0 100 "p"',
            "prec_type": prec_type,
            "dset": "nist20_hr",
            "formula": formula,
            "notes": "",
        }

    @pytest.mark.parametrize("formula", ["CHN", "CHNO", "CHNOS"])
    def test_digit_free_formula_is_not_read_as_a_peptide(self, formula):
        from pyteomics.mass import Composition

        _, products, _, _, _, error = du.parse_annotations(self._row(formula))
        assert error == repr(None), error
        expected = Composition(formula=formula)
        expected["H"] += 1  # [M+H]+
        # "p" annotates the precursor peak, so the product is the protonated molecule.
        assert products == [du.composition_to_string(expected)]

    def test_formula_with_digits_is_unaffected(self):
        _, products, _, _, _, error = du.parse_annotations(self._row("C8H10N4O2"))
        assert error == repr(None), error
        assert products == ["C8H11N4O2"]


class TestParseMassGymCeStr:
    """Tests for parse_mass_gym_ce_str."""

    def test_normalized_energy(self):
        ce, normalized, ramped = du.parse_mass_gym_ce_str("30 normalized=True")
        assert normalized is True
        assert ramped is False
        assert "NCE=30%" in ce

    def test_absolute_energy(self):
        ce, normalized, ramped = du.parse_mass_gym_ce_str("20")
        assert normalized is False
        assert "20 eV" in ce

    def test_nan_input(self):
        ce, normalized, ramped = du.parse_mass_gym_ce_str(float("nan"))
        assert ce == ""
        assert normalized is False
        assert ramped is False


class TestFillMissingCe:
    """Tests for fill_missing_nce and fill_missing_ace."""

    def _make_row(self, nce, ace, prec_type="[M+H]+", prec_mz=500.0):
        return {
            "nce": nce,
            "ace": ace,
            "prec_type": prec_type,
            "prec_mz": prec_mz,
        }

    def test_fill_missing_nce_from_ace(self):
        row = self._make_row(nce=np.nan, ace=27.0)
        result = du.fill_missing_nce(row, ace_colname="ace", nce_colname="nce")
        assert not np.isnan(result)

    def test_fill_missing_nce_keeps_existing(self):
        row = self._make_row(nce=30.0, ace=27.0)
        result = du.fill_missing_nce(row, ace_colname="ace", nce_colname="nce")
        assert result == pytest.approx(30.0)

    def test_fill_missing_ace_from_nce(self):
        row = self._make_row(nce=30.0, ace=np.nan)
        result = du.fill_missing_ace(row, ace_colname="ace", nce_colname="nce")
        assert not np.isnan(result)

    def test_fill_missing_ace_keeps_existing(self):
        row = self._make_row(nce=30.0, ace=27.0)
        result = du.fill_missing_ace(row, ace_colname="ace", nce_colname="nce")
        assert result == pytest.approx(27.0)

    def test_both_missing_returns_nan(self):
        row = self._make_row(nce=np.nan, ace=np.nan)
        result = du.fill_missing_nce(row, ace_colname="ace", nce_colname="nce")
        assert np.isnan(result)


class TestRandomizeSmiles:
    """Tests for randomize_smiles."""

    def test_rseed_minus1_returns_original(self):
        smiles = "CCO"
        assert du.randomize_smiles(smiles, rseed=-1) == smiles

    @pytest.mark.xfail(
        reason="randomize_smiles uses `rng.shuffle()` on the value yielded by "
        "np_temp_seed, which yields None (not a Generator); bug in misc_utils.py",
        strict=True,
    )
    def test_same_seed_same_output(self):
        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        out1 = du.randomize_smiles(smiles, rseed=42)
        out2 = du.randomize_smiles(smiles, rseed=42)
        assert out1 == out2

    @pytest.mark.xfail(
        reason="same np_temp_seed / Generator mismatch as test_same_seed_same_output",
        strict=True,
    )
    def test_different_seeds_may_differ(self):
        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        outputs = {du.randomize_smiles(smiles, rseed=i) for i in range(10)}
        assert len(outputs) > 1


class TestTautomerCanonicalization:
    """Tests for the canonicalize_tautomers flag in rdkit_ml_standardize,
    mol_from_smiles, and mol_from_inchi.

    Uses the 2-hydroxypyridine / 2-pyridinone pair, a classic RDKit tautomer test case.
    """

    # Tautomeric pair: enol form vs keto/lactam form
    ENOL = "Oc1ccccn1"  # 2-hydroxypyridine (aromatic enol)
    KETO = "O=C1C=CC=CN1"  # 2(1H)-pyridinone (correct unsaturated keto form)

    def _to_smi(self, mol) -> str:
        return du.mol_to_smiles(mol)

    def test_canonicalize_true_collapses_tautomers(self):
        """Both tautomers should produce the same canonical SMILES when canonicalize_tautomers=True."""
        mol_enol = du.mol_from_smiles(self.ENOL, canonicalize_tautomers=True)
        mol_keto = du.mol_from_smiles(self.KETO, canonicalize_tautomers=True)
        assert self._to_smi(mol_enol) == self._to_smi(mol_keto)

    def test_canonicalize_false_preserves_distinct_forms(self):
        """Without tautomer canonicalization the two forms must stay distinct."""
        mol_enol = du.mol_from_smiles(self.ENOL, canonicalize_tautomers=False)
        mol_keto = du.mol_from_smiles(self.KETO, canonicalize_tautomers=False)
        assert self._to_smi(mol_enol) != self._to_smi(mol_keto)

    def test_canonicalize_default_is_true(self):
        """Default behaviour (no flag) must equal canonicalize_tautomers=True."""
        mol_default = du.mol_from_smiles(self.ENOL)
        mol_explicit = du.mol_from_smiles(self.ENOL, canonicalize_tautomers=True)
        assert self._to_smi(mol_default) == self._to_smi(mol_explicit)

    def test_rdkit_ml_standardize_flag(self):
        """rdkit_ml_standardize itself must honour the flag."""
        from rdkit import Chem

        mol = Chem.MolFromSmiles(self.ENOL)
        smi_canon = du.mol_to_smiles(du.rdkit_ml_standardize(mol, canonicalize_tautomers=True))
        smi_raw = du.mol_to_smiles(du.rdkit_ml_standardize(mol, canonicalize_tautomers=False))
        # With the flag off the enol form is preserved; with the flag on RDKit may
        # pick a different canonical tautomer — at minimum the results must differ.
        assert smi_canon != smi_raw

    def test_mol_from_inchi_flag_passes_through(self):
        """mol_from_inchi with canonicalize_tautomers=False must return a valid mol."""
        inchi = "InChI=1S/C5H5NO/c7-5-3-1-2-4-6-5/h1-4H,(H,6,7)"  # 2-pyridinone
        mol = du.mol_from_inchi(inchi, canonicalize_tautomers=False)
        assert mol is not None and not (isinstance(mol, float) and np.isnan(mol))

    def test_mol_from_inchi_flag_true_returns_valid_mol(self):
        inchi = "InChI=1S/C5H5NO/c7-5-3-1-2-4-6-5/h1-4H,(H,6,7)"
        mol = du.mol_from_inchi(inchi, canonicalize_tautomers=True)
        assert mol is not None and not (isinstance(mol, float) and np.isnan(mol))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
