"""Merge nearby spectrum peaks and filter them by molecular subformulas."""

import argparse
import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from fragnnet.utils.data_utils import ELEMENT_TO_MASS, VALID_ELEMENTS
from fragnnet.utils.formula_utils import (
    ELECTRON_MASS,
    PREC_TYPE_TO_CMF_COMP_DIFF,
    PREC_TYPE_TO_COMP_DIFF,
    PREC_TYPE_TO_MASS_DIFF,
    parse_formula,
)

logger = logging.getLogger(__name__)

PRECURSOR_TYPE_ALIASES = {
    "[M+H-H2O]+": "[M-H2O+H]+",
    "[M+H-2H2O]+": "[M-2H2O+H]+",
    "[M-H4O2+H]+": "[M-2H2O+H]+",
    "[M+H3N+H]+": "[M+NH4]+",
}

# Twice-RDBE contribution per element; a subformula with negative RDBE is not a
# feasible neutral and is rejected before the carrier mass shift is applied.
RDBE_COEFFICIENTS = {"C": 2, "N": 1, "P": 1, "H": -1, "Cl": -1, "Br": -1, "I": -1, "F": -1}

INSTRUMENT_TOLERANCES = {
    "FT": {"relative": 1.0e-5, "min_mz": 200.0},
    "QTOF": {"relative": 2.0e-5, "min_mz": 750.0},
}
MERGE_METHODS = ("common_overlap", "naive_centroid")


def ppm_error(observed_mz: float, theoretical_mz: float) -> float:
    """Return the absolute mass error in parts per million."""
    return abs(observed_mz - theoretical_mz) / theoretical_mz * 1e6


def merge_peaks(
    peaks: list[tuple[float, float]],
    tolerance_ppm: float,
    tolerance_min_mz: float = 0.0,
    require_common_tolerance_overlap: bool = False,
) -> list[tuple[float, float]]:
    """Merge peaks and represent each cluster by its intensity-weighted centroid.

    When ``require_common_tolerance_overlap`` is true, every peak defines a
    ``m/z +/- tolerance`` interval and a peak joins the current cluster only if
    all cluster intervals retain a common intersection. Otherwise, use the
    naive-centroid method and compare each peak with the moving centroid.
    """
    if tolerance_ppm <= 0:
        raise ValueError("tolerance_ppm must be positive")
    if tolerance_min_mz < 0:
        raise ValueError("tolerance_min_mz must be nonnegative")
    if not peaks:
        return []

    sorted_peaks = sorted((float(mz), float(intensity)) for mz, intensity in peaks)
    for mz, intensity in sorted_peaks:
        if not np.isfinite(mz) or mz <= 0:
            raise ValueError(f"Peak m/z must be finite and positive, got {mz}")
        if not np.isfinite(intensity) or intensity < 0:
            raise ValueError(f"Peak intensity must be finite and nonnegative, got {intensity}")

    merged = []
    cluster = [sorted_peaks[0]]
    first_mz = sorted_peaks[0][0]
    first_tolerance_da = tolerance_ppm * 1e-6 * max(first_mz, tolerance_min_mz)
    overlap_lower = first_mz - first_tolerance_da
    overlap_upper = first_mz + first_tolerance_da
    for peak in sorted_peaks[1:]:
        peak_tolerance_da = tolerance_ppm * 1e-6 * max(peak[0], tolerance_min_mz)
        peak_lower = peak[0] - peak_tolerance_da
        peak_upper = peak[0] + peak_tolerance_da
        if require_common_tolerance_overlap:
            should_merge = peak_lower <= overlap_upper and peak_upper >= overlap_lower
        else:
            centroid = _cluster_centroid(cluster)
            centroid_tolerance_da = tolerance_ppm * 1e-6 * max(
                centroid, tolerance_min_mz
            )
            should_merge = abs(peak[0] - centroid) <= centroid_tolerance_da

        if should_merge:
            cluster.append(peak)
            if require_common_tolerance_overlap:
                overlap_lower = max(overlap_lower, peak_lower)
                overlap_upper = min(overlap_upper, peak_upper)
        else:
            merged.append((_cluster_centroid(cluster), sum(item[1] for item in cluster)))
            cluster = [peak]
            overlap_lower = peak_lower
            overlap_upper = peak_upper
    merged.append((_cluster_centroid(cluster), sum(item[1] for item in cluster)))
    return merged


def _cluster_centroid(cluster: list[tuple[float, float]]) -> float:
    total_intensity = sum(intensity for _, intensity in cluster)
    if total_intensity == 0:
        return float(np.mean([mz for mz, _ in cluster]))
    return sum(mz * intensity for mz, intensity in cluster) / total_intensity


def _precursor_ion_charge(normalized_precursor_type: str) -> int:
    """Return the precursor's charge from its adduct string, 0 when it names no polarity."""
    if normalized_precursor_type.endswith("+"):
        return 1
    if normalized_precursor_type.endswith("-"):
        return -1
    return 0


def _budget_ion_masses(neutral_counts: dict[str, int], comp_diff, charge: int) -> np.ndarray:
    """Enumerate ion masses for every subformula of one precursor-ion budget.

    The budget is a precursor ion, ``M`` plus an adduct delta, and a candidate is split the way
    the chemistry splits it: a molecular part drawn from ``M`` and whatever of the carrier the ion
    still holds. Bounding the walk by an ion rather than by the neutral is what lets one budget
    cover several charge-carrier outcomes at once — a fragment that keeps the carrier, one that
    gives it up, and the bare carrier are all subformulas of the same parent.

    ``RDBE_COEFFICIENTS`` sums twice the ring-plus-double-bond equivalent, and the rule it bounds
    is about a neutral: a formula whose neutral form cannot be drawn is not a formula. So the test
    is taken on the neutral form, which means stripping the charge carrier before applying it —
    the molecular part is what has to be drawable, and the carrier is exempt. A chloride adduct is
    the case that makes this matter: ``[CH3OH+Cl]-`` is a real ion, but scoring RDBE across the
    whole anion counts the chloride against a saturated molecule and rejects it.

    A delta that removes atoms is the other half of the same idea. Its loss comes out of the
    molecular part, so a deprotonated precursor is a neutral that has already given up a hydrogen
    and needs no separate sign correction. The ion mass is the formula mass less the charge's
    electrons.
    """
    required_counts = {
        element: -count_change for element, count_change in comp_diff.items() if count_change < 0
    }
    if any(neutral_counts.get(element, 0) < count for element, count in required_counts.items()):
        return np.array([], dtype=float)

    masses = np.array([0.0])
    rdbe_twice = np.array([2.0])
    for element, max_count in neutral_counts.items():
        min_count = required_counts.get(element, 0)
        if min_count > max_count:
            return np.array([], dtype=float)
        levels = np.arange(min_count, max_count + 1)
        masses = (masses[:, None] + levels * ELEMENT_TO_MASS[element]).ravel()
        rdbe_twice = (rdbe_twice[:, None] + levels * RDBE_COEFFICIENTS.get(element, 0)).ravel()
    masses = masses[rdbe_twice >= 0]
    if masses.size == 0:
        return np.array([], dtype=float)

    masses = masses + sum(
        count * ELEMENT_TO_MASS[element]
        for element, count in comp_diff.items()
        if count < 0
    )
    for element, count in comp_diff.items():
        if count > 0:
            levels = np.arange(0, count + 1)
            masses = (masses[:, None] + levels * ELEMENT_TO_MASS[element]).ravel()

    masses = masses[masses > 0]
    ion_masses = masses - charge * ELECTRON_MASS
    return ion_masses[ion_masses > 0]


def _apply_comp_diff(element_counts: dict[str, int], comp_diff) -> dict:
    """Return the ion element counts for one adduct delta applied to a neutral formula."""
    return {
        element: element_counts.get(element, 0) + comp_diff.get(element, 0)
        for element in set(element_counts) | set(comp_diff)
    }


def _precursor_ion_budgets(normalized_precursor_type: str) -> list:
    """Return the adduct deltas whose ions bound what a fragment of this precursor can be.

    This is a chemical-feasibility test on an observed peak, not a question about what any
    fragmentation model can enumerate. A precursor presents two parents. Direct charge retention
    (CRF) keeps the precursor's own adduct, so its budget is the precursor ion. Charge migration
    (CMF) swaps that carrier for the protonated or deprotonated one, which is how sodiated and
    chloride adducts mostly fragment: ``[M+Na]+`` yields ``[frag+H]+`` and ``[M+Cl]-`` loses HCl
    to yield ``[frag-H]-`` (see the references on ``PREC_TYPE_TO_CMF_MASS_DIFF``).

    The migrated carrier is why CMF can need a budget of its own rather than a share of the CRF
    one: ``[frag+H]+`` carries a proton, so it reaches one hydrogen beyond what ``M+Na`` holds,
    and no subformula of the sodiated ion can express it. That second budget is added only when
    it buys something, which rules out two cases.

    An empty CMF delta means the precursor's carrier was already the proton, so there is no other
    carrier to migrate to and no second parent: ``[M+H]+`` is bounded by ``M+H`` alone and
    ``[M-H]-`` by ``M-H`` alone. Giving those a budget of ``M`` would let a fragment ion hold more
    hydrogens than the precursor ion itself has, which fragmentation cannot produce.

    A CMF delta contained in the CRF delta would only repeat masses the first budget already
    covers. ``[M+Cl]-`` is that case: its products lose HCl, so every one is already a subformula
    of ``M+Cl``. Ammonium and the iodine adducts behave the same way.

    The carrier tables come from ``formula_utils`` and are shared with the fragmentation code,
    but nothing here consults a DAG: a peak this filter cannot explain is kept or dropped on
    chemistry alone, and a peak no DAG can explain is the model's problem to handle downstream,
    not a reason to delete it from the target.
    """
    crf_diff = PREC_TYPE_TO_COMP_DIFF[normalized_precursor_type]
    budgets = [crf_diff]
    cmf_diff = PREC_TYPE_TO_CMF_COMP_DIFF.get(normalized_precursor_type)
    if (
        cmf_diff
        and any(
            cmf_diff.get(element, 0) > crf_diff.get(element, 0)
            for element in set(cmf_diff) | set(crf_diff)
        )
    ):
        budgets.append(cmf_diff)
    return budgets


@lru_cache(maxsize=2)
def _subformula_masses(formula: str, precursor_type: str) -> np.ndarray:
    normalized_precursor_type = PRECURSOR_TYPE_ALIASES.get(precursor_type, precursor_type)
    if normalized_precursor_type not in PREC_TYPE_TO_MASS_DIFF:
        supported_types = sorted(set(PREC_TYPE_TO_MASS_DIFF) | set(PRECURSOR_TYPE_ALIASES))
        raise ValueError(
            f"Unsupported precursor type {precursor_type!r}; supported types are "
            f"{supported_types}"
        )

    element_counts = parse_formula(formula)
    unsupported_elements = set(element_counts) - set(VALID_ELEMENTS)
    if unsupported_elements:
        raise ValueError(f"Unsupported formula elements: {sorted(unsupported_elements)}")

    charge = _precursor_ion_charge(normalized_precursor_type)
    ion_masses = np.concatenate(
        [
            _budget_ion_masses(element_counts, comp_diff, charge)
            for comp_diff in _precursor_ion_budgets(normalized_precursor_type)
        ]
    )
    return np.unique(ion_masses)


def filter_subformula_peaks(
    peaks: list[tuple[float, float]],
    formula: str,
    precursor_type: str,
    tolerance_ppm: float,
    tolerance_min_mz: float = 0.0,
) -> list[tuple[float, float]]:
    """Keep peaks matching a precursor-bounded, nonnegative-RDBE subformula.

    Candidates always include charge-migration ions, so ``[M+Na]+`` accepts
    ``[frag+H]+`` and ``[M+Cl]-`` accepts ``[frag-H]-``. Those are real ions; excluding
    them discards most genuine peaks for those two adducts, so it is not an option.
    """
    if tolerance_ppm <= 0:
        raise ValueError("tolerance_ppm must be positive")

    candidate_masses = _subformula_masses(formula, precursor_type)
    if candidate_masses.size == 0:
        return []

    filtered = []
    for peak in peaks:
        mz = float(peak[0])
        insertion_idx = int(np.searchsorted(candidate_masses, mz))
        candidate_idxs = [insertion_idx - 1, insertion_idx]
        if any(
            0 <= idx < len(candidate_masses)
            and abs(mz - float(candidate_masses[idx]))
            <= tolerance_ppm * 1e-6 * max(mz, tolerance_min_mz)
            for idx in candidate_idxs
        ):
            filtered.append((float(peak[0]), float(peak[1])))
    return filtered


def summarize_spectra(spec_df: pd.DataFrame) -> dict[str, int | float]:
    """Summarize spectrum and peak counts for a processing stage."""
    peak_counts = spec_df["peaks"].apply(len).to_numpy()
    total_intensity = float(
        sum(sum(float(intensity) for _, intensity in peaks) for peaks in spec_df["peaks"])
    )
    return {
        "spectra": int(len(spec_df)),
        "nonempty_spectra": int((peak_counts > 0).sum()),
        "empty_spectra": int((peak_counts == 0).sum()),
        "peaks": int(peak_counts.sum()),
        "mean_peaks_per_spectrum": float(peak_counts.mean()) if len(peak_counts) else 0.0,
        "median_peaks_per_spectrum": float(np.median(peak_counts)) if len(peak_counts) else 0.0,
        "total_intensity": total_intensity,
    }


def _safe_fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summarize_retention(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
) -> dict[str, int | float]:
    """Summarize per-spectrum peak and intensity retention between aligned tables."""
    if len(before_df) != len(after_df):
        raise ValueError("Retention statistics require aligned tables with equal row counts")

    peak_fractions = []
    intensity_fractions = []
    spectra_with_no_retained_peaks = 0
    spectra_with_base_peak_retained = 0
    for before_peaks, after_peaks in zip(before_df["peaks"], after_df["peaks"], strict=True):
        if not after_peaks:
            spectra_with_no_retained_peaks += 1
        if before_peaks:
            peak_fractions.append(len(after_peaks) / len(before_peaks))
            before_base_intensity = max(float(intensity) for _, intensity in before_peaks)
            after_base_intensity = max(
                (float(intensity) for _, intensity in after_peaks), default=-np.inf
            )
            spectra_with_base_peak_retained += after_base_intensity >= before_base_intensity

        before_intensity = sum(float(intensity) for _, intensity in before_peaks)
        after_intensity = sum(float(intensity) for _, intensity in after_peaks)
        if before_intensity > 0:
            intensity_fractions.append(after_intensity / before_intensity)

    before_stats = summarize_spectra(before_df)
    after_stats = summarize_spectra(after_df)
    return {
        "spectra": int(len(before_df)),
        "aggregate_peak_fraction": _safe_fraction(after_stats["peaks"], before_stats["peaks"]),
        "aggregate_intensity_fraction": _safe_fraction(
            after_stats["total_intensity"], before_stats["total_intensity"]
        ),
        "mean_per_spectrum_peak_fraction": float(np.mean(peak_fractions))
        if peak_fractions
        else 0.0,
        "median_per_spectrum_peak_fraction": float(np.median(peak_fractions))
        if peak_fractions
        else 0.0,
        "mean_per_spectrum_intensity_fraction": float(np.mean(intensity_fractions))
        if intensity_fractions
        else 0.0,
        "median_per_spectrum_intensity_fraction": float(np.median(intensity_fractions))
        if intensity_fractions
        else 0.0,
        "spectra_with_no_retained_peaks": spectra_with_no_retained_peaks,
        "spectra_with_base_peak_retained": spectra_with_base_peak_retained,
    }


def summarize_retention_by_column(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    column: str,
) -> dict[str, dict[str, int | float]]:
    """Calculate aligned retention statistics separately for each group."""
    summaries = {}
    for value in sorted(before_df[column].dropna().unique()):
        mask = before_df[column] == value
        summaries[str(value)] = summarize_retention(before_df[mask], after_df[mask])
    return summaries


def summarize_retention_by_columns(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    columns: list[str],
) -> dict[str, dict[str, int | float]]:
    """Calculate retention for intersections such as precursor type × instrument."""
    summaries = {}
    for values, indices in before_df.groupby(columns, dropna=False, sort=True).groups.items():
        values = values if isinstance(values, tuple) else (values,)
        group_name = " | ".join(f"{column}={value}" for column, value in zip(columns, values))
        summaries[group_name] = summarize_retention(
            before_df.loc[indices], after_df.loc[indices]
        )
    return summaries


def select_precursor_types(
    spec_df: pd.DataFrame,
    precursor_types: list[str] | None,
) -> pd.DataFrame:
    """Select requested precursor types without changing spectrum identifiers."""
    if precursor_types is None:
        return spec_df
    if "prec_type" not in spec_df:
        raise ValueError("spec_df must contain prec_type when --precursor-types is used")
    selected_df = spec_df[spec_df["prec_type"].isin(precursor_types)].copy()
    if selected_df.empty:
        raise ValueError(f"No spectra found for precursor types: {precursor_types}")
    return selected_df.reset_index(drop=True)


def select_instrument_types(
    spec_df: pd.DataFrame,
    instrument_types: list[str] | None,
) -> pd.DataFrame:
    """Select requested instrument types without changing spectrum identifiers."""
    if instrument_types is None:
        return spec_df
    if "inst_type" not in spec_df:
        raise ValueError("spec_df must contain inst_type when --instrument-types is used")
    selected_df = spec_df[spec_df["inst_type"].isin(instrument_types)].copy()
    if selected_df.empty:
        raise ValueError(f"No spectra found for instrument types: {instrument_types}")
    return selected_df.reset_index(drop=True)


def select_dsets(
    spec_df: pd.DataFrame,
    dsets: list[str] | None,
) -> pd.DataFrame:
    """Select requested source datasets without changing spectrum identifiers."""
    if dsets is None:
        return spec_df
    if "dset" not in spec_df:
        raise ValueError("spec_df must contain dset when --dsets is used")
    selected_df = spec_df[spec_df["dset"].isin(dsets)].copy()
    if selected_df.empty:
        raise ValueError(f"No spectra found for datasets: {dsets}")
    return selected_df.reset_index(drop=True)


def select_supported_formula_spectra(
    spec_df: pd.DataFrame,
    mol_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int | list[str]]]:
    """Remove spectra whose molecular formulas contain unsupported RDBE elements."""
    formula_by_mol_id = mol_df.set_index("mol_id")["formula"]
    relevant_formulas = formula_by_mol_id[formula_by_mol_id.index.isin(spec_df["mol_id"])]
    unsupported_by_mol_id = relevant_formulas.map(
        lambda formula: sorted(set(parse_formula(formula)) - set(VALID_ELEMENTS))
    )
    unsupported_by_mol_id = unsupported_by_mol_id[unsupported_by_mol_id.apply(bool)]
    unsupported_mask = spec_df["mol_id"].isin(unsupported_by_mol_id.index)
    summary = {
        "excluded_spectra": int(unsupported_mask.sum()),
        "excluded_molecules": int(unsupported_by_mol_id.size),
        "unsupported_elements": sorted(
            {element for elements in unsupported_by_mol_id for element in elements}
        ),
    }
    return spec_df[~unsupported_mask].reset_index(drop=True), summary


def clean_spectra(
    spec_df: pd.DataFrame,
    mol_df: pd.DataFrame,
    merge_peak_signals: bool,
    merge_ppm: float,
    filter_by_subformula: bool,
    subformula_ppm: float,
    drop_empty_spectra: bool,
    use_instrument_tolerance: bool = False,
    use_merge_instrument_tolerance: bool = False,
    merge_method: str | None = None,
) -> pd.DataFrame:
    """Apply the selected per-spectrum cleanup operations."""
    cleaned_df, _ = clean_spectra_with_stats(
        spec_df,
        mol_df,
        merge_peak_signals,
        merge_ppm,
        filter_by_subformula,
        subformula_ppm,
        drop_empty_spectra,
        use_instrument_tolerance,
        use_merge_instrument_tolerance,
        merge_method,
    )
    return cleaned_df


def clean_spectra_with_stats(
    spec_df: pd.DataFrame,
    mol_df: pd.DataFrame,
    merge_peak_signals: bool,
    merge_ppm: float,
    filter_by_subformula: bool,
    subformula_ppm: float,
    drop_empty_spectra: bool,
    use_instrument_tolerance: bool = False,
    use_merge_instrument_tolerance: bool = False,
    merge_method: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Apply cleanup and return stage-level and retention statistics."""
    if merge_method is None:
        merge_method = (
            "common_overlap" if use_merge_instrument_tolerance else "naive_centroid"
        )
    if merge_method not in MERGE_METHODS:
        raise ValueError(
            f"merge_method must be one of {MERGE_METHODS}, got {merge_method!r}"
        )
    required_spec_columns = {"peaks"}
    if filter_by_subformula:
        required_spec_columns.update({"mol_id", "prec_type"})
    if (merge_peak_signals and use_merge_instrument_tolerance) or (
        filter_by_subformula and use_instrument_tolerance
    ):
        required_spec_columns.add("inst_type")
    missing_spec_columns = required_spec_columns - set(spec_df.columns)
    if missing_spec_columns:
        raise ValueError(f"spec_df is missing required columns: {sorted(missing_spec_columns)}")
    if filter_by_subformula and not {"mol_id", "formula"}.issubset(mol_df.columns):
        raise ValueError("mol_df must contain mol_id and formula columns for subformula filtering")

    stats = {
        "parameters": {
            "merge_peaks": merge_peak_signals,
            "merge_ppm": merge_ppm,
            "merge_instrument_tolerance": use_merge_instrument_tolerance,
            "merge_method": merge_method if merge_peak_signals else None,
            "filter_subformula_peaks": filter_by_subformula,
            "subformula_ppm": subformula_ppm,
            "subformula_instrument_tolerance": use_instrument_tolerance,
            "drop_empty_spectra": drop_empty_spectra,
        },
        "input": summarize_spectra(spec_df),
    }

    # 1. Attach molecular formulas only when subformula filtering is requested.
    result = spec_df.copy()
    if filter_by_subformula:
        formula_by_mol_id = mol_df.set_index("mol_id")["formula"]
        if not result["mol_id"].isin(formula_by_mol_id.index).all():
            missing_ids = sorted(set(result["mol_id"]) - set(formula_by_mol_id.index))
            raise ValueError(f"No molecular formula found for mol_id values: {missing_ids[:10]}")
        formulas = result["mol_id"].map(formula_by_mol_id)

    # 2. Group nearby peaks and replace each group with its intensity-weighted centroid.
    if merge_peak_signals:
        before_merge = result.copy()
        if use_merge_instrument_tolerance:
            unsupported_instruments = set(result["inst_type"]) - set(INSTRUMENT_TOLERANCES)
            if unsupported_instruments:
                raise ValueError(
                    "No merge instrument tolerance for: "
                    f"{sorted(unsupported_instruments)}"
                )
            use_common_overlap = merge_method == "common_overlap"
            tolerance_multiplier = 1.0 if use_common_overlap else 2.0
            merged_peaks = (
                merge_peaks(
                    peaks,
                    tolerance_ppm=tolerance_multiplier
                    * INSTRUMENT_TOLERANCES[instrument_type]["relative"]
                    * 1e6,
                    tolerance_min_mz=INSTRUMENT_TOLERANCES[instrument_type]["min_mz"],
                    require_common_tolerance_overlap=use_common_overlap,
                )
                for peaks, instrument_type in zip(
                    result["peaks"], result["inst_type"], strict=True
                )
            )
            result.loc[:, "peaks"] = pd.Series(
                merged_peaks, index=result.index, dtype=object
            )
        else:
            result.loc[:, "peaks"] = result["peaks"].apply(
                merge_peaks,
                tolerance_ppm=merge_ppm,
                require_common_tolerance_overlap=merge_method == "common_overlap",
            )
        stats["after_merge"] = summarize_spectra(result)
        stats["merge_retention"] = summarize_retention(before_merge, result)
        if "prec_type" in result:
            stats["merge_retention_by_prec_type"] = summarize_retention_by_column(
                before_merge, result, "prec_type"
            )
        if "inst_type" in result:
            stats["merge_retention_by_inst_type"] = summarize_retention_by_column(
                before_merge, result, "inst_type"
            )
            if "prec_type" in result:
                stats["merge_retention_by_prec_type_and_inst_type"] = (
                    summarize_retention_by_columns(before_merge, result, ["prec_type", "inst_type"])
                )

    # 3. Retain only peaks explained by a feasible molecular subformula.
    if filter_by_subformula:
        before_subformula_filter = result.copy()
        if use_instrument_tolerance:
            unsupported_instruments = set(result["inst_type"]) - set(INSTRUMENT_TOLERANCES)
            if unsupported_instruments:
                raise ValueError(
                    "No subformula instrument tolerance for: "
                    f"{sorted(unsupported_instruments)}"
                )
            filtered_peaks = (
                filter_subformula_peaks(
                    peaks,
                    formula,
                    precursor_type,
                    INSTRUMENT_TOLERANCES[instrument_type]["relative"] * 1e6,
                    INSTRUMENT_TOLERANCES[instrument_type]["min_mz"],
                )
                for peaks, formula, precursor_type, instrument_type in zip(
                    result["peaks"],
                    formulas,
                    result["prec_type"],
                    result["inst_type"],
                    strict=True,
                )
            )
        else:
            filtered_peaks = (
                filter_subformula_peaks(peaks, formula, precursor_type, subformula_ppm)
                for peaks, formula, precursor_type in zip(
                    result["peaks"], formulas, result["prec_type"], strict=True
                )
            )
        result.loc[:, "peaks"] = pd.Series(filtered_peaks, index=result.index, dtype=object)
        stats["after_subformula_filter"] = summarize_spectra(result)
        stats["subformula_retention"] = summarize_retention(before_subformula_filter, result)
        stats["subformula_retention_by_prec_type"] = summarize_retention_by_column(
            before_subformula_filter, result, "prec_type"
        )
        if "inst_type" in result:
            stats["subformula_retention_by_inst_type"] = summarize_retention_by_column(
                before_subformula_filter, result, "inst_type"
            )
            stats["subformula_retention_by_prec_type_and_inst_type"] = (
                summarize_retention_by_columns(
                    before_subformula_filter, result, ["prec_type", "inst_type"]
                )
            )

    if drop_empty_spectra:
        result = result[result["peaks"].apply(len) > 0].copy()
    result = result.reset_index(drop=True)
    stats["output"] = summarize_spectra(result)
    return result, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-fp", type=Path, required=True, help="Input spec_df.pkl")
    parser.add_argument(
        "--mol-fp",
        type=Path,
        help="Input mol_df.pkl; required with --filter-subformula-peaks",
    )
    parser.add_argument("--output-fp", type=Path, help="Output pickle path")
    parser.add_argument(
        "--stats-fp",
        type=Path,
        help="JSON statistics path; defaults to <output-fp stem>.stats.json",
    )
    parser.add_argument(
        "--precursor-types",
        nargs="+",
        help=(
            'Only process these precursor types, e.g. "[M+H]+" "[M-H]-" '
            '"[M+Na]+" "[M+Cl]-"'
        ),
    )
    parser.add_argument(
        "--instrument-types",
        nargs="+",
        help="Only process these instrument types, e.g. FT QTOF",
    )
    parser.add_argument(
        "--dsets",
        nargs="+",
        help="Only process these source datasets, e.g. nist20_hr",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Calculate statistics without writing a cleaned spectrum pickle",
    )
    parser.add_argument(
        "--skip-unsupported-formulas",
        action="store_true",
        help="Exclude and report formulas containing elements unsupported by the RDBE table",
    )
    parser.add_argument(
        "--merge-peaks",
        action="store_true",
        help="Merge peaks using an intensity-weighted centroid",
    )
    parser.add_argument("--merge-ppm", type=float, default=20.0)
    parser.add_argument(
        "--merge-inst-tol",
        action="store_true",
        help="Merge peaks using FraGNNet FT/QTOF tolerances instead of --merge-ppm",
    )
    parser.add_argument(
        "--merge-method",
        choices=MERGE_METHODS,
        default=None,
        help=(
            "Peak grouping rule. Defaults to naive_centroid with --merge-ppm and "
            "common_overlap "
            "with --merge-inst-tol"
        ),
    )
    parser.add_argument(
        "--filter-subformula-peaks",
        action="store_true",
        help="Keep only peaks matching precursor-compatible molecular subformulas",
    )
    parser.add_argument("--subformula-ppm", type=float, default=5.0)
    parser.add_argument(
        "--subformula-inst-tol",
        action="store_true",
        help="Use FraGNNet FT/QTOF tolerance and m/z floors instead of --subformula-ppm",
    )
    parser.add_argument(
        "--drop-empty-spectra",
        action="store_true",
        help="Drop spectra with no peaks after the selected operations",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.merge_peaks and not args.filter_subformula_peaks:
        raise ValueError("Enable --merge-peaks, --filter-subformula-peaks, or both")
    if args.filter_subformula_peaks and args.mol_fp is None:
        raise ValueError("--mol-fp is required with --filter-subformula-peaks")
    if not args.stats_only and args.output_fp is None:
        raise ValueError("--output-fp is required unless --stats-only is used")
    if args.stats_only and args.stats_fp is None:
        raise ValueError("--stats-fp is required with --stats-only")
    if args.output_fp is not None and args.output_fp.resolve() == args.spec_fp.resolve():
        raise ValueError("--output-fp must differ from --spec-fp")

    source_spec_df = pd.read_pickle(args.spec_fp)
    spec_df = select_dsets(source_spec_df, args.dsets)
    spec_df = select_precursor_types(spec_df, args.precursor_types)
    spec_df = select_instrument_types(spec_df, args.instrument_types)
    mol_df = pd.read_pickle(args.mol_fp) if args.mol_fp is not None else pd.DataFrame()
    selected_spectra = len(spec_df)
    unsupported_summary = None
    if args.filter_subformula_peaks and args.skip_unsupported_formulas:
        spec_df, unsupported_summary = select_supported_formula_spectra(spec_df, mol_df)
    cleaned_df, stats = clean_spectra_with_stats(
        spec_df=spec_df,
        mol_df=mol_df,
        merge_peak_signals=args.merge_peaks,
        merge_ppm=args.merge_ppm,
        filter_by_subformula=args.filter_subformula_peaks,
        subformula_ppm=args.subformula_ppm,
        drop_empty_spectra=args.drop_empty_spectra,
        use_instrument_tolerance=args.subformula_inst_tol,
        use_merge_instrument_tolerance=args.merge_inst_tol,
        merge_method=args.merge_method,
    )
    stats["selection"] = {
        "precursor_types": args.precursor_types,
        "instrument_types": args.instrument_types,
        "dsets": args.dsets,
        "source_spectra": len(source_spec_df),
        "selected_spectra": selected_spectra,
        "processed_spectra": len(spec_df),
        "unsupported_formula_exclusions": unsupported_summary,
    }

    if not args.stats_only:
        args.output_fp.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_pickle(args.output_fp)
        logger.info("Saved %d spectra to %s", len(cleaned_df), args.output_fp)
    stats_fp = args.stats_fp or args.output_fp.with_suffix(".stats.json")
    stats_fp.parent.mkdir(parents=True, exist_ok=True)
    stats_fp.write_text(json.dumps(stats, indent=2) + "\n")
    logger.info("Saved cleanup statistics to %s", stats_fp)


if __name__ == "__main__":
    main()
