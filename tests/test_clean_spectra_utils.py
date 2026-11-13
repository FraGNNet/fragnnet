import pandas as pd
import pytest

from fragnnet.utils import clean_spectra_utils, data_utils


def test_merge_peaks_uses_intensity_weighted_centroid():
    peaks = [(100.0, 1.0), (100.001, 3.0), (101.0, 2.0)]

    merged = clean_spectra_utils.merge_peaks(peaks, tolerance_ppm=20.0)

    assert merged[0] == pytest.approx((100.00075, 4.0))
    assert merged[1] == pytest.approx((101.0, 2.0))


def test_merge_peaks_does_not_use_transitive_chaining():
    peaks = [(100.0, 1.0), (100.0019, 1.0), (100.0038, 1.0)]

    merged = clean_spectra_utils.merge_peaks(peaks, tolerance_ppm=20.0)

    assert len(merged) == 2
    assert merged[0] == pytest.approx((100.00095, 2.0))
    assert merged[1] == pytest.approx((100.0038, 1.0))


def test_merge_peaks_applies_low_mass_tolerance_floor():
    peaks = [(100.0, 1.0), (100.003, 1.0)]

    merged = clean_spectra_utils.merge_peaks(
        peaks, tolerance_ppm=20.0, tolerance_min_mz=200.0
    )

    assert merged == pytest.approx([(100.0015, 2.0)])


def test_merge_instrument_tolerance_uses_overlapping_per_peak_windows():
    peaks = [(100.0, 1.0), (100.01, 1.0)]
    spec_df = pd.DataFrame(
        {
            "inst_type": ["FT", "QTOF"],
            "peaks": [peaks, peaks],
        }
    )

    cleaned = clean_spectra_utils.clean_spectra(
        spec_df,
        pd.DataFrame(),
        merge_peak_signals=True,
        merge_ppm=20.0,
        filter_by_subformula=False,
        subformula_ppm=5.0,
        drop_empty_spectra=False,
        use_merge_instrument_tolerance=True,
    )

    assert cleaned.loc[0, "peaks"] == peaks
    assert cleaned.loc[1, "peaks"] == pytest.approx([(100.005, 2.0)])


def test_merge_instrument_tolerance_requires_cluster_wide_overlap():
    spec_df = pd.DataFrame(
        {
            "inst_type": ["QTOF"],
            "peaks": [[(100.0, 1.0), (100.028, 1.0), (100.042, 1.0)]],
        }
    )

    cleaned = clean_spectra_utils.clean_spectra(
        spec_df,
        pd.DataFrame(),
        merge_peak_signals=True,
        merge_ppm=20.0,
        filter_by_subformula=False,
        subformula_ppm=5.0,
        drop_empty_spectra=False,
        use_merge_instrument_tolerance=True,
    )

    assert len(cleaned.loc[0, "peaks"]) == 2
    assert cleaned.loc[0, "peaks"][0] == pytest.approx((100.014, 2.0))
    assert cleaned.loc[0, "peaks"][1] == pytest.approx((100.042, 1.0))


def test_merge_instrument_tolerance_can_select_naive_centroid_method():
    spec_df = pd.DataFrame(
        {
            "inst_type": ["QTOF"],
            "peaks": [[(100.0, 1.0), (100.028, 1.0), (100.042, 1.0)]],
        }
    )

    cleaned = clean_spectra_utils.clean_spectra(
        spec_df,
        pd.DataFrame(),
        merge_peak_signals=True,
        merge_ppm=20.0,
        filter_by_subformula=False,
        subformula_ppm=5.0,
        drop_empty_spectra=False,
        use_merge_instrument_tolerance=True,
        merge_method="naive_centroid",
    )

    assert len(cleaned.loc[0, "peaks"]) == 1
    assert cleaned.loc[0, "peaks"][0] == pytest.approx((100.0233333333, 3.0))


def test_fixed_ppm_merge_can_select_common_overlap_method():
    spec_df = pd.DataFrame(
        {"peaks": [[(100.0, 1.0), (100.0038, 3.0)]]}
    )

    cleaned = clean_spectra_utils.clean_spectra(
        spec_df,
        pd.DataFrame(),
        merge_peak_signals=True,
        merge_ppm=20.0,
        filter_by_subformula=False,
        subformula_ppm=5.0,
        drop_empty_spectra=False,
        merge_method="common_overlap",
    )

    assert len(cleaned.loc[0, "peaks"]) == 1
    assert cleaned.loc[0, "peaks"][0] == pytest.approx((100.00285, 4.0))


def test_merge_only_does_not_require_molecule_metadata():
    spec_df = pd.DataFrame({"peaks": [[(100.0, 1.0), (100.001, 2.0)]]})

    cleaned = clean_spectra_utils.clean_spectra(
        spec_df,
        pd.DataFrame(),
        merge_peak_signals=True,
        merge_ppm=20.0,
        filter_by_subformula=False,
        subformula_ppm=5.0,
        drop_empty_spectra=False,
    )

    assert cleaned.loc[0, "peaks"][0] == pytest.approx((100.0006666667, 3.0))


def test_filter_subformula_peaks_keeps_only_mass_matches():
    carbon_ion_mz = data_utils.formula_to_mass("C") - data_utils.ELECTRON_MASS
    peaks = [(carbon_ion_mz + 4e-5, 10.0), (50.0, 5.0)]

    filtered = clean_spectra_utils.filter_subformula_peaks(
        peaks,
        formula="CH4",
        precursor_type="[M]+",
        tolerance_ppm=5.0,
    )

    assert filtered == [peaks[0]]


@pytest.mark.parametrize(
    ("precursor_type", "mass_shift"),
    [
        ("[M+H]+", clean_spectra_utils.PREC_TYPE_TO_MASS_DIFF["[M+H]+"]),
        ("[M+Na]+", clean_spectra_utils.PREC_TYPE_TO_MASS_DIFF["[M+Na]+"]),
        ("[M-H]-", clean_spectra_utils.PREC_TYPE_TO_MASS_DIFF["[M-H]-"]),
        ("[M+Cl]-", clean_spectra_utils.PREC_TYPE_TO_MASS_DIFF["[M+Cl]-"]),
    ],
)
def test_filter_subformula_peaks_applies_adduct_mass_shift(precursor_type, mass_shift):
    ion_mz = data_utils.formula_to_mass("CH4") + mass_shift
    neutral_mz = data_utils.formula_to_mass("CH4")
    peaks = [(ion_mz, 10.0), (neutral_mz, 5.0)]

    filtered = clean_spectra_utils.filter_subformula_peaks(
        peaks,
        formula="CH4",
        precursor_type=precursor_type,
        tolerance_ppm=5.0,
    )

    assert filtered == [peaks[0]]


def test_filter_subformula_peaks_accepts_nist_dehydration_adduct_alias():
    normalized_type = "[M-H2O+H]+"
    nist_type = "[M+H-H2O]+"
    ion_mz = (
        data_utils.formula_to_mass("CH4O")
        + clean_spectra_utils.PREC_TYPE_TO_MASS_DIFF[normalized_type]
    )

    filtered = clean_spectra_utils.filter_subformula_peaks(
        [(ion_mz, 10.0)],
        formula="CH4O",
        precursor_type=nist_type,
        tolerance_ppm=5.0,
    )

    assert filtered == [(ion_mz, 10.0)]


def test_deprotonated_subformula_requires_a_hydrogen_to_remove():
    impossible_mz = (
        data_utils.formula_to_mass("C")
        + clean_spectra_utils.PREC_TYPE_TO_MASS_DIFF["[M-H]-"]
    )

    filtered = clean_spectra_utils.filter_subformula_peaks(
        [(impossible_mz, 10.0)],
        formula="CH4",
        precursor_type="[M-H]-",
        tolerance_ppm=5.0,
    )

    assert filtered == []


@pytest.mark.parametrize(
    ("precursor_type", "cmf_shift"),
    [
        ("[M+Na]+", clean_spectra_utils.PREC_TYPE_TO_MASS_DIFF["[M+H]+"]),
        ("[M+Cl]-", clean_spectra_utils.PREC_TYPE_TO_MASS_DIFF["[M-H]-"]),
    ],
)
def test_charge_migration_fragments_are_candidates(precursor_type, cmf_shift):
    """[M+Na]+ fragments to [frag+H]+ and [M+Cl]- loses HCl to give [frag-H]-."""
    migrated_mz = data_utils.formula_to_mass("CH4O") + cmf_shift
    peaks = [(migrated_mz, 10.0)]

    kept = clean_spectra_utils.filter_subformula_peaks(
        peaks, "CH4O", precursor_type, tolerance_ppm=5.0
    )

    assert kept == peaks
    # Nothing that keeps the carrier lands here, so it is the carrier-free subformula of the
    # precursor ion that explains the peak.
    carrier = "Na" if precursor_type == "[M+Na]+" else "Cl"
    crf_budget = clean_spectra_utils._budget_ion_masses(
        {"C": 1, "H": 4, "O": 1},
        clean_spectra_utils.PREC_TYPE_TO_COMP_DIFF[precursor_type],
        clean_spectra_utils._precursor_ion_charge(precursor_type),
    )
    carrier_retaining = crf_budget[crf_budget > data_utils.formula_to_mass(carrier)]
    assert min(abs(carrier_retaining - migrated_mz)) > 5e-6 * migrated_mz


@pytest.mark.parametrize("precursor_type", ["[M+Na]+", "[M+Cl]-"])
def test_carrier_retaining_fragments_still_match_with_cmf(precursor_type):
    """Adding the CMF channel must widen the candidate set, never replace it."""
    retained_mz = (
        data_utils.formula_to_mass("CH4O")
        + clean_spectra_utils.PREC_TYPE_TO_MASS_DIFF[precursor_type]
    )

    filtered = clean_spectra_utils.filter_subformula_peaks(
        [(retained_mz, 10.0)], "CH4O", precursor_type, tolerance_ppm=5.0
    )

    assert filtered == [(retained_mz, 10.0)]


@pytest.mark.parametrize(
    ("precursor_type", "budget_hydrogens"),
    [
        # The proton is already [M+H]+'s own carrier, so there is nothing to migrate to.
        ("[M+H]+", [5]),
        # Likewise [M-H]-: a budget of M would hold more hydrogens than the precursor ion does.
        ("[M-H]-", [3]),
        # [M+Na]+ fragments to [frag+H]+, which reaches a hydrogen beyond what M+Na holds.
        ("[M+Na]+", [4, 5]),
        ("[M+K]+", [4, 5]),
        # [M+Cl]- loses HCl, so its migrated products are already subformulas of M+Cl.
        ("[M+Cl]-", [4]),
        ("[M+NH4]+", [8]),
    ],
)
def test_budgets_are_precursor_ions_including_the_migrated_one(precursor_type, budget_hydrogens):
    """Each budget is an ion the fragment could have come from, not the bare neutral."""
    budgets = [
        clean_spectra_utils._apply_comp_diff({"C": 1, "H": 4}, comp_diff)
        for comp_diff in clean_spectra_utils._precursor_ion_budgets(precursor_type)
    ]

    assert [counts["H"] for counts in budgets] == budget_hydrogens


def test_charge_migration_reaches_one_hydrogen_past_the_sodiated_ion():
    """[frag+H]+ carries a proton M+Na cannot supply, so the CMF budget must allow it."""
    protonated_mz = (
        data_utils.formula_to_mass("CH4O")
        + clean_spectra_utils.PREC_TYPE_TO_MASS_DIFF["[M+H]+"]
    )

    filtered = clean_spectra_utils.filter_subformula_peaks(
        [(protonated_mz, 10.0)], "CH4O", "[M+Na]+", tolerance_ppm=5.0
    )

    assert filtered == [(protonated_mz, 10.0)]


def test_fragment_ion_cannot_outrun_the_deprotonated_precursor_hydrogens():
    """[M-H]- holds one less hydrogen than M, so a full-hydrogen fragment anion is impossible."""
    impossible_mz = data_utils.formula_to_mass("CH4") + clean_spectra_utils.ELECTRON_MASS

    filtered = clean_spectra_utils.filter_subformula_peaks(
        [(impossible_mz, 10.0)], "CH4O", "[M-H]-", tolerance_ppm=5.0
    )

    assert filtered == []


def test_neutral_loss_precursor_types_are_gone():
    """The _NL entries were never present in real data and are no longer supported."""
    assert not [
        prec_type
        for prec_type in clean_spectra_utils.PREC_TYPE_TO_MASS_DIFF
        if "_NL" in prec_type
    ]


@pytest.mark.parametrize(
    ("ion_formula", "precursor_type", "source_formula", "kept"),
    [
        # The molecular part is what must be drawable, so the carrier is stripped before the
        # RDBE test. A saturated molecule keeps its carrier-retaining adduct ions.
        ("CH4OCl", "[M+Cl]-", "CH4O", True),  # [CH3OH+Cl]- is real; RDBE scores CH4O, not CH4OCl
        ("C2H10NO", "[M+NH4]+", "C2H6O", True),  # likewise the ammonium adduct of ethanol
        ("CH5O", "[M+H]+", "CH4O", True),  # protonated methanol, molecular part CH4O at RDBE 0
        ("CH3O", "[M-H]-", "CH4O", True),  # methoxide: the loss comes out of the molecule
        # An impossible molecular part is still rejected in either polarity.
        ("CH7", "[M+H]+", "CH6", False),
        ("CH5O2", "[M-H]-", "CH6O2", False),
    ],
)
def test_rdbe_is_taken_on_the_neutral_form_with_the_carrier_stripped(
    ion_formula, precursor_type, source_formula, kept
):
    """RDBE bounds a neutral, so it scores the molecular part and exempts the charge carrier."""
    charge = clean_spectra_utils._precursor_ion_charge(precursor_type)
    ion_mz = data_utils.formula_to_mass(ion_formula) - charge * clean_spectra_utils.ELECTRON_MASS

    filtered = clean_spectra_utils.filter_subformula_peaks(
        [(ion_mz, 10.0)], source_formula, precursor_type, tolerance_ppm=5.0
    )

    assert bool(filtered) is kept


def test_precursor_ion_charge_is_signed():
    """Polarity drives the electron correction on the ion mass."""
    assert clean_spectra_utils._precursor_ion_charge("[M+H]+") == 1
    assert clean_spectra_utils._precursor_ion_charge("[M-H]-") == -1


def test_carrier_free_fragment_ion_is_a_candidate():
    """The bare chloride of an [M+Cl]- precursor is a subformula of that precursor ion."""
    chloride_mz = data_utils.formula_to_mass("Cl") + clean_spectra_utils.ELECTRON_MASS

    filtered = clean_spectra_utils.filter_subformula_peaks(
        [(chloride_mz, 10.0)], "CH4O", "[M+Cl]-", tolerance_ppm=5.0
    )

    assert filtered == [(chloride_mz, 10.0)]


def test_cmf_channel_still_requires_a_removable_hydrogen():
    """[M+Cl]- CMF is [frag-H]-, so a hydrogen-free subformula cannot supply it."""
    impossible_mz = (
        data_utils.formula_to_mass("C")
        + clean_spectra_utils.PREC_TYPE_TO_MASS_DIFF["[M-H]-"]
    )

    filtered = clean_spectra_utils.filter_subformula_peaks(
        [(impossible_mz, 10.0)], "CCl4", "[M+Cl]-", tolerance_ppm=5.0
    )

    assert filtered == []


def test_cmf_channel_respects_the_rdbe_rule():
    """A CMF candidate with negative RDBE is rejected like any other subformula."""
    # H4 is a subformula of CH4O but has negative RDBE, so the CMF channel must
    # reject it rather than emit [H4+H]+ as a candidate.
    impossible_mz = (
        data_utils.formula_to_mass("H4")
        + clean_spectra_utils.PREC_TYPE_TO_MASS_DIFF["[M+H]+"]
    )

    filtered = clean_spectra_utils.filter_subformula_peaks(
        [(impossible_mz, 10.0)], "CH4O", "[M+Na]+", tolerance_ppm=5.0
    )

    assert filtered == []


def test_qtof_instrument_tolerance_uses_0p015_da_low_mass_floor():
    theoretical_mz = (
        data_utils.formula_to_mass("CH4")
        + clean_spectra_utils.PREC_TYPE_TO_MASS_DIFF["[M+H]+"]
    )
    observed_peak = (theoretical_mz + 0.01, 10.0)

    fixed_5ppm = clean_spectra_utils.filter_subformula_peaks(
        [observed_peak], "CH4", "[M+H]+", tolerance_ppm=5.0
    )
    qtof_inst_tol = clean_spectra_utils.filter_subformula_peaks(
        [observed_peak],
        "CH4",
        "[M+H]+",
        tolerance_ppm=20.0,
        tolerance_min_mz=750.0,
    )

    assert fixed_5ppm == []
    assert qtof_inst_tol == [observed_peak]


def test_cleanup_options_are_independent_and_can_drop_empty_spectra():
    spec_df = pd.DataFrame(
        {
            "spec_id": [0, 1],
            "mol_id": [0, 0],
            "prec_type": ["[M]+", "[M]+"],
            "peaks": [[(100.0, 1.0), (100.001, 2.0)], [(50.0, 1.0)]],
        }
    )
    mol_df = pd.DataFrame({"mol_id": [0], "formula": ["CH4"]})

    merged_only = clean_spectra_utils.clean_spectra(
        spec_df,
        mol_df,
        merge_peak_signals=True,
        merge_ppm=20.0,
        filter_by_subformula=False,
        subformula_ppm=5.0,
        drop_empty_spectra=False,
    )
    filtered = clean_spectra_utils.clean_spectra(
        spec_df.iloc[[1]],
        mol_df,
        merge_peak_signals=False,
        merge_ppm=20.0,
        filter_by_subformula=True,
        subformula_ppm=5.0,
        drop_empty_spectra=True,
    )

    assert merged_only.loc[0, "peaks"][0] == pytest.approx((100.0006666667, 3.0))
    assert filtered.empty


def test_cleanup_stats_report_peak_and_intensity_retention():
    spec_df = pd.DataFrame(
        {
            "mol_id": [0, 0],
            "prec_type": ["[M]+", "[M]+"],
            "peaks": [[(12.0, 3.0), (50.0, 1.0)], [(50.0, 2.0)]],
        }
    )
    mol_df = pd.DataFrame({"mol_id": [0], "formula": ["CH4"]})

    cleaned, stats = clean_spectra_utils.clean_spectra_with_stats(
        spec_df,
        mol_df,
        merge_peak_signals=False,
        merge_ppm=20.0,
        filter_by_subformula=True,
        subformula_ppm=100.0,
        drop_empty_spectra=True,
    )

    assert len(cleaned) == 1
    assert stats["input"]["peaks"] == 3
    assert stats["after_subformula_filter"]["peaks"] == 1
    assert stats["subformula_retention"]["aggregate_peak_fraction"] == pytest.approx(1 / 3)
    assert stats["subformula_retention"]["spectra"] == 2
    assert stats["subformula_retention"]["aggregate_intensity_fraction"] == pytest.approx(0.5)
    assert stats["subformula_retention"]["spectra_with_no_retained_peaks"] == 1
    assert stats["subformula_retention"]["spectra_with_base_peak_retained"] == 1
    assert stats["subformula_retention_by_prec_type"]["[M]+"] == stats[
        "subformula_retention"
    ]


def test_select_precursor_types_supports_four_adduct_evaluation_scope():
    spec_df = pd.DataFrame(
        {
            "spec_id": [0, 1, 2, 3, 4],
            "prec_type": ["[M+H]+", "[M-H]-", "[M+Na]+", "[M+Cl]-", "[M+K]+"],
            "peaks": [[], [], [], [], []],
        }
    )

    selected = clean_spectra_utils.select_precursor_types(
        spec_df, ["[M+H]+", "[M-H]-", "[M+Na]+", "[M+Cl]-"]
    )

    assert selected["spec_id"].tolist() == [0, 1, 2, 3]


def test_select_instrument_types_focuses_on_ft_and_qtof():
    spec_df = pd.DataFrame(
        {
            "spec_id": [0, 1, 2],
            "inst_type": ["FT", "QTOF", "IT"],
            "peaks": [[], [], []],
        }
    )

    selected = clean_spectra_utils.select_instrument_types(spec_df, ["FT", "QTOF"])

    assert selected["spec_id"].tolist() == [0, 1]


def test_select_dsets_focuses_on_requested_source_dataset():
    spec_df = pd.DataFrame(
        {
            "spec_id": [0, 1, 2],
            "dset": ["nist20_hr", "mona23", "nist20_hr"],
            "peaks": [[], [], []],
        }
    )

    selected = clean_spectra_utils.select_dsets(spec_df, ["nist20_hr"])

    assert selected["spec_id"].tolist() == [0, 2]


def test_select_supported_formula_spectra_reports_exclusions():
    spec_df = pd.DataFrame({"spec_id": [0, 1], "mol_id": [0, 1], "peaks": [[], []]})
    mol_df = pd.DataFrame({"mol_id": [0, 1], "formula": ["CH4", "CH4Sn"]})

    selected, summary = clean_spectra_utils.select_supported_formula_spectra(spec_df, mol_df)

    assert selected["spec_id"].tolist() == [0]
    assert summary == {
        "excluded_spectra": 1,
        "excluded_molecules": 1,
        "unsupported_elements": ["Sn"],
    }
