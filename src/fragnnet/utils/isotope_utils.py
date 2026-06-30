from collections.abc import Mapping

import numpy as np

from fragnnet.utils.formula_utils import (
    C13_SPACING,
    MASS,
    NEUTRON_MASS,
    PREC_TYPE_TO_FORMULA_DIFF,
    isotope_offsets_from_formula,
    parse_formula,
)


def detect_isotope_peaks(
    mzs: np.ndarray,
    ints: np.ndarray,
    mz_tol: float = 0.02,
    max_isotope: int = 3,
    check_intensity: bool = True,
) -> np.ndarray:
    """Detect isotope peaks in a reference spectrum.

    A peak is considered an isotope peak (M+k, k >= 1) if there exists a
    lower-m/z peak separated by approximately k * NEUTRON_MASS within
    ``mz_tol``.  Optionally, that lower-m/z peak must also be more intense
    than the candidate isotope peak.

    The function works with raw (non-normalised) m/z and intensity arrays as
    produced by standard reference libraries such as NIST.

    Args:
        mzs: Peak m/z values of shape (N,).  Need not be sorted.
        ints: Peak intensities of shape (N,).  Same order as ``mzs``.
        mz_tol: Absolute m/z tolerance in Da for matching isotope spacing.
            The default of 0.02 Da covers both the exact 13C–12C spacing
            (~1.0034 Da) and the NEUTRON_MASS approximation (1.0087 Da) used
            in the rest of the codebase.  Defaults to 0.02.
        max_isotope: Maximum isotope order to consider (1 = M+1 only,
            2 = up to M+2, etc.). Defaults to 3.
        check_intensity: If True, require that the candidate monoisotopic
            peak (the one at mz - k * NEUTRON_MASS) is at least as intense
            as the current peak.  This suppresses false positives from
            noise but may miss isotope peaks of low-intensity fragments.
            Defaults to True.

    Returns:
        Boolean array of shape (N,) where True means the peak is an
        isotope peak.

    Example:
        >>> mzs = np.array([100.0, 101.008, 200.0, 201.012])
        >>> ints = np.array([1000.0, 100.0, 500.0, 50.0])
        >>> mask = detect_isotope_peaks(mzs, ints)
        >>> mask  # peaks at 101.008 and 201.012 flagged as isotopes
        array([False,  True, False,  True])
    """
    mzs = np.asarray(mzs, dtype=np.float64)
    ints = np.asarray(ints, dtype=np.float64)
    if mzs.shape != ints.shape or mzs.ndim != 1:
        raise ValueError(f"mzs and ints must be 1-D arrays of the same length, got {mzs.shape}")

    n = len(mzs)
    is_isotope = np.zeros(n, dtype=bool)
    if n == 0:
        return is_isotope

    sort_idx = np.argsort(mzs)
    sorted_mzs = mzs[sort_idx]
    sorted_ints = ints[sort_idx]

    for i in range(n):
        for k in range(1, max_isotope + 1):
            target_mz = sorted_mzs[i] - k * NEUTRON_MASS
            lo = np.searchsorted(sorted_mzs, target_mz - mz_tol, side="left")
            hi = np.searchsorted(sorted_mzs, target_mz + mz_tol, side="right")
            if lo >= hi:
                continue
            # At least one candidate monoisotopic peak exists in the window.
            if check_intensity:
                if np.any(sorted_ints[lo:hi] >= sorted_ints[i]):
                    is_isotope[sort_idx[i]] = True
                    break
            else:
                is_isotope[sort_idx[i]] = True
                break

    return is_isotope


def detect_isotope_peaks_formula_aware(
    mzs: np.ndarray,
    ints: np.ndarray,
    element_counts: dict[str, int],
    mz_tol: float = 0.02,
    max_isotope: int = 3,
    check_intensity: bool = True,
    intensity_ratio_lo: float = 0.1,
    intensity_ratio_hi: float = 10.0,
) -> np.ndarray:
    """Detect isotope peaks using exact pyopenms isotopologue mass offsets.

    Unlike :func:`detect_isotope_peaks`, which uses a fixed ``k * NEUTRON_MASS``
    step, this function queries pyopenms ``CoarseIsotopePatternGenerator`` for
    the actual M+k mass offsets predicted for the given molecular formula.  This
    improves accuracy for molecules containing Cl, Br, S, or K where even-mass
    isotopes shift the M+2/M+3 positions away from the simple neutron-mass
    approximation.

    An optional intensity ratio gate rejects peaks whose observed/expected ratio
    falls outside ``[intensity_ratio_lo * exp, intensity_ratio_hi * exp]``.

    Requires pyopenms to be installed.

    Args:
        mzs: Peak m/z values of shape (N,).  Need not be sorted.
        ints: Peak intensities of shape (N,).  Same order as ``mzs``.
        element_counts: Molecular formula as element count dict from
            :func:`~fragnnet.utils.formula_utils.parse_formula`.
        mz_tol: Absolute m/z tolerance in Da.  Defaults to 0.02.
        max_isotope: Maximum isotope order to query from pyopenms.
            Defaults to 3.
        check_intensity: If True, require that the candidate monoisotopic peak
            is at least as intense as the candidate isotope peak.  Defaults to
            True.
        intensity_ratio_lo: Lower multiplier on the pyopenms-predicted
            relative intensity for the ratio gate.  Set to 0.0 to disable.
            Defaults to 0.1.
        intensity_ratio_hi: Upper multiplier on the pyopenms-predicted
            relative intensity for the ratio gate.  Set to ``np.inf`` to
            disable.  Defaults to 10.0.

    Returns:
        Boolean array of shape (N,) where True means the peak is an isotope
        peak.

    Example:
        >>> import numpy as np
        >>> from fragnnet.utils.formula_utils import parse_formula
        >>> ec = parse_formula("C10H8Br2")
        >>> mzs = np.array([285.899, 286.902, 287.902, 288.902, 289.902])
        >>> ints = np.array([100.0, 5.0, 97.0, 2.0, 24.0])
        >>> mask = detect_isotope_peaks_formula_aware(mzs, ints, ec)
        >>> mask[2]  # M+2 (large Br peak) is correctly flagged
        True
    """
    mzs = np.asarray(mzs, dtype=np.float64)
    ints = np.asarray(ints, dtype=np.float64)
    if mzs.shape != ints.shape or mzs.ndim != 1:
        raise ValueError(f"mzs and ints must be 1-D arrays of the same length, got {mzs.shape}")

    n = len(mzs)
    is_isotope = np.zeros(n, dtype=bool)
    if n == 0:
        return is_isotope

    offsets = isotope_offsets_from_formula(element_counts, max_isotope=max_isotope)
    if not offsets:
        return is_isotope

    sort_idx = np.argsort(mzs)
    sorted_mzs = mzs[sort_idx]
    sorted_ints = ints[sort_idx]

    for i in range(n):
        for offset, exp_ratio in offsets:
            target_mz = sorted_mzs[i] - offset
            if target_mz < 0:
                continue
            lo = np.searchsorted(sorted_mzs, target_mz - mz_tol, side="left")
            hi = np.searchsorted(sorted_mzs, target_mz + mz_tol, side="right")
            if lo >= hi:
                continue
            mono_int = sorted_ints[lo:hi].max()
            if check_intensity and exp_ratio <= 1.0 and mono_int < sorted_ints[i]:
                continue
            if mono_int > 0:
                obs_ratio = sorted_ints[i] / mono_int
                if not (
                    intensity_ratio_lo * exp_ratio <= obs_ratio <= intensity_ratio_hi * exp_ratio
                ):
                    continue
            is_isotope[sort_idx[i]] = True
            break

    return is_isotope


# ---------------------------------------------------------------------------
# Training-data isotope cleaning
# ---------------------------------------------------------------------------

# Natural abundance M+1/M+0 contribution per atom (Poisson/binomial approx)
_M1_CONTRIB: dict[str, float] = {
    "C": 0.01103,  # 13C
    "H": 0.000149,  # 2H
    "N": 0.00368,  # 15N
    "O": 0.000404,  # 17O
    "S": 0.00421,  # 33S
    "Si": 0.04683,  # 29Si
}

# Additive M+2/M+0 contributions from heavy isotopes (Cl, Br dominate)
_M2_ADDITIVE: dict[str, float] = {
    "Cl": 0.3220,  # 37Cl
    "Br": 0.9735,  # 81Br
    "K": 0.0673,  # 41K
    "S": 0.0425,  # 34S
    "Si": 0.0337,  # 30Si
    "O": 0.00204,  # 18O
}

# mz_tol per instrument type for isotope detection.
# The M+0 search is centred on k × C13_SPACING (1.003355 Da per step) not NEUTRON_MASS.
# H-gain (+1H = 1.007825 Da) sits 4.47 mDa from C13_SPACING, so 4 mDa excludes it
# while retaining a 0.47 mDa safety margin for FT mass accuracy (~1–3 ppm).
# For k ≥ 2, the tolerance is doubled (8 mDa) to cover heavy-isotope spacings
# (³⁷Cl 1.9970 Da, ⁴¹K 1.9981 Da, ³⁴S 1.9985 Da) that differ from 2 × C13_SPACING
# (2.0067 Da) by up to ~9.7 mDa.
# All other instruments: None → never filter (H-gain and isotope indistinguishable).
_INST_MZ_TOL: dict[str, float | None] = {
    "FT": 0.004,
}


def _expected_mk_ratio(
    mol_counts: dict[str, int],
    mol_scale: float,
    k: int,
    adduct_counts: dict[str, int] | None = None,
) -> float:
    """Expected M+k/M+0 intensity ratio for a fragment, adduct-aware.

    Molecule atoms are scaled by ``mol_scale`` (≈ neutral fragment mass / neutral
    precursor mass).  Adduct atoms (e.g. K from ``[M+K]+``, Cl from ``[M+Cl]-``)
    are applied at **full weight** because the adduct charge carrier either stays
    with a fragment entirely or is absent—it does not distribute proportionally
    across all fragments the way carbon does.

    Args:
        mol_counts: Element counts for the neutral molecule from ``parse_formula``.
        mol_scale: Fraction of the neutral molecule mass present in this neutral
            fragment, i.e. ``(frag_mz - adduct_delta_mass) / (prec_mz - adduct_delta_mass)``.
        k: Isotope order (1 = M+1, 2 = M+2).
        adduct_counts: Element counts for atoms *added* by the adduct (positive
            atoms only, e.g. ``{"K": 1}`` for ``[M+K]+``).  None or empty means
            no adduct correction is applied.

    Returns:
        Expected M+k/M+0 intensity ratio, floored at 1e-4.
    """
    import math

    if adduct_counts is None:
        adduct_counts = {}

    m1_mol = sum(mol_counts.get(elem, 0) * c for elem, c in _M1_CONTRIB.items())
    m1_add = sum(adduct_counts.get(elem, 0) * c for elem, c in _M1_CONTRIB.items())
    total_m1 = m1_mol * mol_scale + m1_add
    poisson = total_m1**k / math.factorial(k)
    if k == 2:
        m2_mol = sum(mol_counts.get(elem, 0) * c for elem, c in _M2_ADDITIVE.items())
        m2_add = sum(adduct_counts.get(elem, 0) * c for elem, c in _M2_ADDITIVE.items())
        return max(poisson + m2_mol * mol_scale + m2_add, 1e-4)
    return max(poisson, 1e-4)


def _parse_adduct_counts(prec_type: str | None) -> tuple[dict[str, int], float]:
    """Return positive adduct atom counts and their total monoisotopic mass.

    Only atoms that are *added* by the adduct are returned (positive counts).
    Atoms removed by the adduct (e.g. the ``-H`` in ``[M-H]-``) are ignored
    because they reduce the precursor mass and have negligible isotope impact.

    Args:
        prec_type: Adduct/precursor type string (e.g. ``"[M+K]+"``, ``"[M+Cl]-"``).
            If None or not found in ``PREC_TYPE_TO_FORMULA_DIFF``, returns empty
            counts and zero mass.

    Returns:
        Tuple of (adduct_counts, adduct_delta_mass) where adduct_counts maps
        element symbol to integer count for atoms added by the adduct, and
        adduct_delta_mass is the total monoisotopic mass of those atoms in Da.
    """
    if prec_type is None:
        return {}, 0.0
    formula_diff = PREC_TYPE_TO_FORMULA_DIFF.get(prec_type, "")
    if not formula_diff:
        return {}, 0.0
    try:
        diff_counts = parse_formula(formula_diff)
    except (ValueError, AssertionError):
        return {}, 0.0
    adduct_counts = {elem: cnt for elem, cnt in diff_counts.items() if cnt > 0}
    adduct_delta_mass = sum(MASS(elem) * cnt for elem, cnt in adduct_counts.items())
    return adduct_counts, adduct_delta_mass


def _estimate_leak_ratio_from_spectrum(
    smzs: np.ndarray,
    sints: np.ndarray,
    k: int,
    mz_tol: float,
    min_pairs: int,
    consistency_cv: float,
    mol_counts: dict[str, int] | None = None,
    adduct_counts: dict[str, int] | None = None,
    adduct_delta_mass: float = 0.0,
    prec_neutral_mass: float = 0.0,
    r_exp_prec: Mapping[int, float] | None = None,
    dag_formula_mzs: np.ndarray | None = None,
    dag_formula_strs: np.ndarray | None = None,
    dag_mz_tol: float = 0.015,
) -> float | None:
    """Estimate co-isolation ratio from cross-pair consistency of M+k peaks.

    When ``mol_counts`` is provided, each raw ratio ``r_i = I(M+k_i) / I(M_i)``
    is converted into a co-isolation fraction using the formula-predicted
    natural M+k/M+0 ratio for that fragment:

        f_i = r_i / R_exp_frag_i

    For ``k == 1`` and a precursor isotope ratio is available, use the
    denominator-corrected approximation:

        f_i = r_i / (R_frag_i - r_i * (R_prec - R_frag_i))

    because an M+1 precursor can contribute to observed fragment M+0 when the
    heavy isotope is carried by the neutral loss.  Without ``mol_counts`` the raw
    ``r_i`` values are used and the function returns the median as ``R_k`` (the
    old behaviour, appropriate when the formula is unavailable).

    Args:
        smzs: Sorted m/z array.
        sints: Sorted intensity array (same order as smzs).
        k: Isotope order (1 = M+1, 2 = M+2).
        mz_tol: Per-step m/z tolerance in Da (doubled for k ≥ 2 by caller).
        min_pairs: Minimum number of (M, M+k) pairs required to accept consensus.
        consistency_cv: Maximum coefficient of variation (std/mean) of values to
            accept the consensus as a genuine co-isolation signal.
        mol_counts: Element counts of the neutral precursor molecule.  When given,
            enables formula-normalised f estimation.
        adduct_counts: Element counts added by the adduct (positive atoms only).
        adduct_delta_mass: Monoisotopic mass of adduct atoms (Da).
        prec_neutral_mass: Neutral precursor mass ``prec_mz - adduct_delta_mass``
            used as denominator for the per-fragment mol_scale estimate.
        r_exp_prec: Precursor expected isotope ratios by order.  When present,
            enables denominator correction for k=1.
        dag_formula_mzs: Optional DAG monoisotopic fragment m/z values.
        dag_formula_strs: Formula strings aligned to ``dag_formula_mzs``.
        dag_mz_tol: Absolute tolerance for matching observed M+0 peaks to DAG
            formula entries.

    Returns:
        Consensus f (when ``mol_counts`` given) or R_k (when not), or None
        if the signal is inconsistent or there are too few pairs.
    """
    k_tol = mz_tol if k == 1 else mz_tol * 2
    n = len(smzs)
    r_vals: list[float] = []
    use_formula = mol_counts is not None and prec_neutral_mass > 0

    for i in range(n):
        m0_int = sints[i]
        if m0_int <= 0:
            continue
        target = smzs[i] + k * C13_SPACING
        lo = np.searchsorted(smzs, target - k_tol, side="left")
        hi = np.searchsorted(smzs, target + k_tol, side="right")
        if lo >= hi:
            continue
        mk_int = sints[lo + int(np.argmax(sints[lo:hi]))]
        # Monotonicity: M+0 must be more intense (R_k < 1 for typical co-isolation)
        if mk_int >= m0_int:
            continue
        r_raw = mk_int / m0_int
        if use_formula:
            r_exp = _fragment_expected_ratio(
                smzs[i],
                k,
                mol_counts,
                adduct_counts or {},
                adduct_delta_mass,
                prec_neutral_mass,
                dag_formula_mzs=dag_formula_mzs,
                dag_formula_strs=dag_formula_strs,
                dag_mz_tol=dag_mz_tol,
            )
            if r_exp <= 0:
                continue
            if k == 1 and r_exp_prec is not None:
                r_prec = float(r_exp_prec.get(k, 0.0))
                denom = r_exp - r_raw * (r_prec - r_exp)
                if denom <= 0:
                    continue
                r_vals.append(r_raw / denom)  # denominator-corrected f
            else:
                r_vals.append(r_raw / r_exp)  # first-order f
        else:
            r_vals.append(r_raw)  # unnormalised → R_k

    if len(r_vals) < min_pairs:
        return None

    r_arr = np.array(r_vals)
    r_median = float(np.median(r_arr))
    if r_median <= 0:
        return None

    cv = float(np.std(r_arr) / r_median)
    if cv > consistency_cv:
        return None  # values too scattered → not a coherent co-isolation signal

    return r_median


def _find_dag_formula(
    query_mz: float,
    dag_formula_mzs: np.ndarray | None,
    dag_formula_strs: np.ndarray | None,
    dag_mz_tol: float,
) -> str | None:
    """Return the closest DAG formula string for ``query_mz`` if available."""
    if dag_formula_mzs is None or dag_formula_strs is None:
        return None
    mzs = np.asarray(dag_formula_mzs, dtype=np.float64)
    formulas = np.asarray(dag_formula_strs, dtype=object)
    if mzs.ndim != 1 or formulas.ndim != 1 or len(mzs) != len(formulas) or len(mzs) == 0:
        return None
    valid = np.isfinite(mzs) & (mzs > 0)
    if not np.any(valid):
        return None
    candidate_idx = np.where(valid & (np.abs(mzs - query_mz) <= dag_mz_tol))[0]
    if len(candidate_idx) == 0:
        return None
    best = candidate_idx[int(np.argmin(np.abs(mzs[candidate_idx] - query_mz)))]
    formula = formulas[best]
    if formula is None:
        return None
    formula = str(formula)
    return formula or None


def _fragment_expected_ratio(
    m0_mz: float,
    k: int,
    mol_counts: dict[str, int],
    adduct_counts: dict[str, int],
    adduct_delta_mass: float,
    prec_neutral_mass: float,
    dag_formula_mzs: np.ndarray | None = None,
    dag_formula_strs: np.ndarray | None = None,
    dag_mz_tol: float = 0.015,
) -> float:
    """Return expected natural fragment M+k/M+0 ratio.

    DAG formulas are preferred because they provide the fragment composition.
    The m/z-scaled precursor formula is retained as a fallback.
    """
    dag_formula = _find_dag_formula(m0_mz, dag_formula_mzs, dag_formula_strs, dag_mz_tol)
    if dag_formula is not None:
        try:
            return _expected_mk_ratio(parse_formula(dag_formula), 1.0, k)
        except Exception:
            pass
    neutral_frag = m0_mz - adduct_delta_mass
    mol_scale = max(0.0, min(1.0, neutral_frag / prec_neutral_mass))
    return _expected_mk_ratio(mol_counts, mol_scale, k, adduct_counts)


def _expected_observed_isotope_ratio(
    f: float,
    r_frag: float,
    k: int,
    r_exp_prec: Mapping[int, float] | None = None,
) -> float:
    """Expected observed I(M+k_frag)/I(M+0_frag) for co-isolation cleanup.

    In MS2, the isolation window selects the monoisotopic M+0 precursor (all-12C).
    Its fragments are all M+0 — no natural M+k contribution.  Only the co-isolated
    M+k precursor produces M+k satellite peaks in the fragment spectrum.

    For a fragment with formula-predicted natural ratio ``r_frag``, the probability
    that the single 13C from the M+1 precursor lands in this fragment is
    ``mol_scale ≈ r_frag / r_prec``.  So the expected M+k satellite ratio is:

        obs = f × r_frag                      (k ≥ 2, no denominator correction)

    For k=1, the M+1 precursor also dilutes the observed M+0 fragment intensity
    (when 13C is carried by the neutral loss rather than retained in the fragment).
    The denominator correction accounts for this:

        obs = (f × r_frag) / (1 + f × (r_prec − r_frag))

    Args:
        f: Co-isolation efficiency (``R_k_obs_prec / R_k_exp_prec``).
        r_frag: Expected natural M+k/M+0 ratio for this fragment (used as a proxy
            for the fraction of 13C that would land in the fragment).
        k: Isotope order.
        r_exp_prec: Precursor expected isotope ratios by order.

    Returns:
        Expected observed I(M+k_frag)/I(M+0_frag).
    """
    if k == 1 and r_exp_prec is not None:
        r_prec = float(r_exp_prec.get(k, 0.0))
        if r_prec > 0:
            return (f * r_frag) / (1.0 + f * (r_prec - r_frag))
    return f * r_frag


def _find_peak_sorted(
    smzs: np.ndarray,
    sints: np.ndarray,
    target: float,
    tol: float,
    tol_floor: float = 1e-6,
) -> tuple[float, float] | None:
    """Return (m/z, intensity) of the most intense sorted-spectrum peak near target."""
    tol = max(float(tol), tol_floor)
    lo = np.searchsorted(smzs, target - tol, side="left")
    hi = np.searchsorted(smzs, target + tol, side="right")
    if lo >= hi:
        return None
    best = lo + int(np.argmax(sints[lo:hi]))
    return smzs[best], sints[best]


def _estimate_coiso_fraction_from_precursor_sorted(
    smzs: np.ndarray,
    sints: np.ndarray,
    prec_mz: float,
    mz_tol: float,
    max_isotope: int,
    r_exp_prec: Mapping[int, float],
    min_coiso_fraction: float,
    max_leak_factor: float,
    ratio_lo: float,
    ratio_hi: float,
    precursor_envelope_lo: float,
    precursor_envelope_hi: float,
) -> tuple[dict[int, float], dict[int, float]]:
    """Estimate precursor-derived co-isolation f and R_k from sorted peaks.

    When precursor M+1 and M+2 residual peaks are both present, validate their
    direct M2/M1 ratio against the expected precursor isotope envelope.  This is
    equivalent to checking ``f[2] / f[1]`` but keeps the above-precursor envelope
    test explicit and independently tunable from the fragment ratio gate.
    """
    if not r_exp_prec:
        return {}, {}

    m0_tol = 0.05
    min_prec_m0_frac = 0.005
    base_int = float(sints.max()) if len(sints) > 0 else 0.0
    prec_m0 = _find_peak_sorted(smzs, sints, prec_mz, m0_tol)
    if (
        prec_m0 is None
        or prec_m0[1] <= 0
        or (base_int > 0 and prec_m0[1] / base_int < min_prec_m0_frac)
    ):
        return {}, {}

    prec_m0_int = prec_m0[1]
    leak_ratios: dict[int, float] = {}
    r_leak: dict[int, float] = {}
    precursor_residual_ints: dict[int, float] = {}

    for k in range(1, max_isotope + 1):
        k_tol = mz_tol if k == 1 else mz_tol * 2
        peak = _find_peak_sorted(smzs, sints, prec_mz + k * C13_SPACING, k_tol)
        if peak is None:
            continue
        pk_mz, pk_int = peak
        if pk_mz <= prec_mz + k_tol:
            continue
        r_exp_k = r_exp_prec.get(k, 0.0)
        if r_exp_k <= 0:
            continue
        R_k = pk_int / prec_m0_int
        factor = R_k / r_exp_k
        if factor < min_coiso_fraction:
            continue
        if factor > max_leak_factor:
            continue
        r_leak[k] = factor
        leak_ratios[k] = R_k
        precursor_residual_ints[k] = pk_int

    if max_isotope >= 2 and 1 in r_leak and 2 in r_leak:
        exp_m2_m1 = r_exp_prec[2] / r_exp_prec[1]
        obs_m2_m1 = precursor_residual_ints[2] / precursor_residual_ints[1]
        envelope_factor = obs_m2_m1 / exp_m2_m1
        if not (precursor_envelope_lo <= envelope_factor <= precursor_envelope_hi):
            return {}, {}
        ratio = r_leak[2] / r_leak[1]
        if not (ratio_lo <= ratio <= ratio_hi):
            return {}, {}

    return r_leak, leak_ratios


def estimate_coisolation_fraction_from_precursor(
    mzs: np.ndarray,
    ints: np.ndarray,
    prec_mz: float,
    inst_type: str,
    formula: str,
    prec_type: str | None = None,
    max_isotope: int = 2,
    min_coiso_fraction: float = 0.05,
    max_leak_factor: float = 1.5,
    ratio_lo: float = 0.05,
    ratio_hi: float = 10.0,
    precursor_envelope_lo: float = 0.2,
    precursor_envelope_hi: float = 5.0,
) -> dict[int, float]:
    """Estimate co-isolation fraction f from precursor M+k residual peaks only.

    This is intended for group-level imputation: compute reliable f values from
    spectra in a ``mol_id + prec_type + inst_type`` group that contain precursor
    residuals, aggregate those f values within the group, then pass the aggregate
    to ``detect_isotope_peaks_for_training(..., coiso_fraction_by_k=...)`` for
    spectra in the same group that have no local precursor-derived f.

    Returns:
        Mapping from isotope order k to precursor-derived co-isolation fraction
        f. Empty if the instrument is unsupported, formula parsing fails, no
        precursor residual is found, or the precursor estimate is implausible.
    """
    mzs = np.asarray(mzs, dtype=np.float64)
    ints = np.asarray(ints, dtype=np.float64)
    if mzs.shape != ints.shape or mzs.ndim != 1:
        raise ValueError(f"mzs and ints must be 1-D arrays of the same length, got {mzs.shape}")

    mz_tol = _INST_MZ_TOL.get(inst_type)
    if mz_tol is None or len(mzs) == 0 or prec_mz <= 0:
        return {}

    try:
        mol_counts = parse_formula(formula)
        adduct_counts, _ = _parse_adduct_counts(prec_type)
        r_exp_prec = {
            k: _expected_mk_ratio(mol_counts, 1.0, k, adduct_counts)
            for k in range(1, max_isotope + 1)
        }
    except Exception:
        return {}

    sort_idx = np.argsort(mzs)
    smzs = mzs[sort_idx]
    sints = ints[sort_idx]
    r_leak, _ = _estimate_coiso_fraction_from_precursor_sorted(
        smzs,
        sints,
        prec_mz,
        mz_tol,
        max_isotope,
        r_exp_prec,
        min_coiso_fraction,
        max_leak_factor,
        ratio_lo,
        ratio_hi,
        precursor_envelope_lo,
        precursor_envelope_hi,
    )
    return r_leak


def detect_isotope_peaks_for_training(
    mzs: np.ndarray,
    ints: np.ndarray,
    prec_mz: float,
    inst_type: str,
    max_isotope: int = 2,
    require_monotone: bool = True,
    require_envelope: bool = True,
    ratio_lo: float = 0.05,
    ratio_hi: float = 10.0,
    precursor_envelope_lo: float = 0.2,
    precursor_envelope_hi: float = 5.0,
    aggressive: bool = True,
    aggressive_min_pairs: int = 2,
    aggressive_cv: float = 0.6,
    aggressive_single_pair_min_f: float = 0.05,
    formula: str | None = None,
    prec_type: str | None = None,
    min_coiso_fraction: float = 0.05,
    max_leak_factor: float = 1.5,
    coiso_fraction_by_k: Mapping[int, float] | None = None,
    dag_formula_mzs: np.ndarray | None = None,
    dag_formula_strs: np.ndarray | None = None,
    dag_mz_tol: float = 0.015,
    protect_dag_mono: bool = False,
) -> np.ndarray:
    """Detect isotope leak peaks for training-data cleaning.

    Designed for high precision (minimal false positives) rather than high recall,
    because false positives delete real H-transfer peaks that the model must learn.

    **Physics model** (``formula`` provided):

    The isolation window co-selects a fraction ``f`` of the M+k precursor isotopologue
    alongside M+0.  Both fragment identically, so every fragment ion gains a satellite
    at ``+k × C13_SPACING`` with intensity proportional to ``f``:

        observed ratio at fragment i  =  R_exp_frag_i × f

    where ``R_exp_frag_i`` is the natural M+k/M+0 ratio for that fragment's formula
    (proportional to carbon/heteroatom count) and ``f`` is the co-isolation fraction
    (0 < f ≤ 1).  ``f`` is estimated from the precursor residual peaks above
    ``prec_mz`` (primary path) or from cross-pair consistency of in-spectrum
    fragment M+k pairs (aggressive fallback):

        f  =  R_k_prec / R_exp_prec          (primary: precursor residuals)
        f  ≈  median(r_raw_i / R_exp_frag_i) (aggressive: formula-normalised pairs)

    Only spectra where ``f ≥ min_coiso_fraction`` are flagged.  Because ``f ≤ 1``
    always.

    Per-fragment expected ratio is estimated as
    ``R_exp_frag_i ≈ _expected_mk_ratio(mol_counts, frag_mz/prec_neutral, k)``
    using the fragment's m/z as a proxy for its formula size.  This correctly
    scales the detection threshold: small fragments require a much smaller
    absolute M+k/M+0 ratio to be considered a leak than large ones.

    **Without formula** (``formula=None``): falls back to comparing ``obs ≈ R_k``
    uniformly across all fragments (previous behaviour).  No ``R_leak`` gating.

    **Default mode** (``aggressive=False``): ``R_k`` / ``R_leak`` is read from
    the precursor's own M+k peaks above ``prec_mz``.  Returns all-False when
    those peaks are absent.

    **Aggressive mode** (``aggressive=True``): when no above-prec reference is
    found, estimates ``R_leak`` from cross-pair consistency of in-spectrum M+k
    candidates, formula-normalised when ``mol_counts`` is available.

    Key design choices:

    * **Instrument gating**: Only FT spectra are filtered.
    * **Tight mz_tol**: 4 mDa (k=1), 8 mDa (k≥2), centred on C13_SPACING.
    * **Monotonicity**: Requires I(M+0_frag) > I(M+k_frag).
    * **Envelope coherence**: M+k requires M+1…M+(k-1) confirmed first.

    Args:
        mzs: Peak m/z values of shape (N,).  Need not be sorted.
        ints: Peak intensities of shape (N,).  Same order as ``mzs``.
        prec_mz: Precursor m/z.
        inst_type: Instrument type string (``"FT"``, ``"QTOF"``, ``"IT"``).
            Only ``"FT"`` is currently filtered.
        max_isotope: Maximum isotope order to consider.  Defaults to 2.
        require_monotone: If True, require I(M+0_frag) > I(M+k_frag).
        require_envelope: If True, require all intermediate M+j confirmed first.
        ratio_lo: Lower bound multiplier on the per-fragment expected ratio.
        ratio_hi: Upper bound multiplier on the per-fragment expected ratio.
        precursor_envelope_lo: Lower bound for the normalized above-precursor
            M2/M1 envelope ratio when both precursor residuals are visible.
        precursor_envelope_hi: Upper bound for the normalized above-precursor
            M2/M1 envelope ratio when both precursor residuals are visible.
        aggressive: If True and no above-prec reference found, estimate R_leak
            from cross-pair consistency.  Defaults to True.
        aggressive_min_pairs: Min pairs for aggressive consensus.  Defaults to 2.
        aggressive_cv: Max CV of (normalised) r_i for aggressive mode.  Defaults to 0.6.
        aggressive_single_pair_min_f: When ``formula`` is given and only one
            (M, M+k) pair is found, accept it as genuine co-isolation if the
            estimated f exceeds this threshold.  Allows detection in spectra with
            very few visible fragments.  Ignored when ``formula`` is None or when
            ``aggressive_min_pairs`` pairs are already available.  Defaults to 0.05.
        formula: Neutral molecular formula string (e.g. ``"C10H14N2"``).  When
            given, enables R_leak gating and per-fragment expected-ratio correction.
        prec_type: Adduct/precursor type (e.g. ``"[M+H]+"``, ``"[M-H]-"``).
            Used together with ``formula`` to account for adduct atoms.
        min_coiso_fraction: Minimum co-isolation fraction ``f = R_k / R_exp_prec``
            required to trigger flagging.  For genuine co-isolation ``f ≤ 1``; below
            this threshold the signal is too weak to distinguish from noise.
            Only active when ``formula`` is provided.  Defaults to 0.05.
        max_leak_factor: Maximum ``f = R_k / R_exp_prec`` accepted for the above-prec
            reference peak.  Genuine co-isolation gives ``f ≤ 1``; values above this
            indicate a stray fragment coincidentally at the isotope position.
            Only active when ``formula`` is provided.  Defaults to 1.5.
        coiso_fraction_by_k: Optional group-level co-isolation fractions ``f`` by
            isotope order, estimated from other spectra in the same acquisition-like
            group (for example ``mol_id + prec_type + inst_type``). These values
            are used only for isotope orders where this spectrum lacks its own
            precursor-derived reference.
        dag_formula_mzs: Optional DAG monoisotopic fragment m/z values.  When
            provided with ``dag_formula_strs``, fragment isotope ratios are
            computed from DAG formulas instead of the m/z-scaled precursor proxy.
        dag_formula_strs: Formula strings aligned to ``dag_formula_mzs``.
        dag_mz_tol: Absolute m/z tolerance for matching observed M+0 peaks to DAG
            formula entries.
        protect_dag_mono: If True, never flag a peak whose m/z is itself matched
            by a DAG monoisotopic formula.  This is conservative and may reduce
            recall when isotope and monoisotopic explanations overlap.

    Returns:
        Boolean array of shape (N,) where True means co-isolation isotope leak.
        All-False for non-FT instruments or when co-isolation cannot be confirmed.

    Example:
        >>> mzs = np.array([100.0, 101.003, 300.0, 301.003])
        >>> ints = np.array([1000.0, 50.0, 2000.0, 100.0])  # R_1 = 100/2000 = 0.05
        >>> mask = detect_isotope_peaks_for_training(mzs, ints, prec_mz=300.0, inst_type="FT")
        >>> mask[1]  # fragment M+1: ratio 50/1000=0.05 == R_1 → True
        True
    """
    mzs = np.asarray(mzs, dtype=np.float64)
    ints = np.asarray(ints, dtype=np.float64)
    if mzs.shape != ints.shape or mzs.ndim != 1:
        raise ValueError(f"mzs and ints must be 1-D arrays of the same length, got {mzs.shape}")

    n = len(mzs)
    is_isotope = np.zeros(n, dtype=bool)

    # Gate: only FT spectra are filtered
    mz_tol = _INST_MZ_TOL.get(inst_type)
    if mz_tol is None:
        return is_isotope

    if n == 0 or prec_mz <= 0:
        return is_isotope

    sort_idx = np.argsort(mzs)
    smzs = mzs[sort_idx]
    sints = ints[sort_idx]

    def _find_peak(target: float, tol: float) -> tuple[float, float] | None:
        """Return (mz, intensity) of the most intense peak within tol, or None."""
        return _find_peak_sorted(smzs, sints, target, tol)

    # Precompute precursor formula-based expected ratios when formula is known.
    # R_exp_prec[k] = natural M+k/M+0 for the intact precursor (adduct-inclusive).
    # R_leak[k]     = R_k / R_exp_prec[k] — pure co-isolation amplification factor.
    # Only genuine co-isolation (R_leak >= min_leak_factor) triggers flagging.
    mol_counts: dict[str, int] | None = None
    adduct_counts_parsed: dict[str, int] = {}
    adduct_delta_mass: float = 0.0
    r_exp_prec: dict[int, float] = {}

    if formula is not None:
        try:
            mol_counts = parse_formula(formula)
            adduct_counts_parsed, adduct_delta_mass = _parse_adduct_counts(prec_type)
            for _k in range(1, max_isotope + 1):
                r_exp_prec[_k] = _expected_mk_ratio(mol_counts, 1.0, _k, adduct_counts_parsed)
        except Exception:
            mol_counts = None
            r_exp_prec = {}

    prec_neutral_mass = prec_mz - adduct_delta_mass if mol_counts is not None else 0.0

    leak_ratios: dict[int, float] = {}  # k -> R_k  (= R_leak * R_exp_prec[k])
    r_leak: dict[int, float] = {}  # k -> R_leak (only when formula known)

    # Step 1: Estimate f from precursor M+k peaks above prec_mz (primary path).
    if r_exp_prec:
        r_leak, leak_ratios = _estimate_coiso_fraction_from_precursor_sorted(
            smzs,
            sints,
            prec_mz,
            mz_tol,
            max_isotope,
            r_exp_prec,
            min_coiso_fraction,
            max_leak_factor,
            ratio_lo,
            ratio_hi,
            precursor_envelope_lo,
            precursor_envelope_hi,
        )
    else:
        # Legacy no-formula mode: estimate raw R_k from precursor residuals and
        # compare every fragment against that uniform spectrum-level ratio.
        m0_tol = 0.05
        min_prec_m0_frac = 0.005
        base_int = float(sints.max()) if len(sints) > 0 else 0.0
        prec_m0 = _find_peak(prec_mz, m0_tol)
        if (
            prec_m0 is not None
            and prec_m0[1] > 0
            and (base_int <= 0 or prec_m0[1] / base_int >= min_prec_m0_frac)
        ):
            prec_m0_int = prec_m0[1]
            for k in range(1, max_isotope + 1):
                k_tol = mz_tol if k == 1 else mz_tol * 2
                peak = _find_peak(prec_mz + k * C13_SPACING, k_tol)
                if peak is None:
                    continue
                pk_mz, pk_int = peak
                if pk_mz <= prec_mz + k_tol:
                    continue
                leak_ratios[k] = pk_int / prec_m0_int

    # Step 1c: Borrow group-level f for orders without a local precursor f.
    # This covers spectra in the same mol/adduct/instrument group where the
    # precursor residual peaks are absent but another spectrum in the group had
    # a reliable precursor-derived estimate.
    if coiso_fraction_by_k is not None and r_exp_prec:
        for k, f in coiso_fraction_by_k.items():
            if k in leak_ratios or k < 1 or k > max_isotope:
                continue
            if not np.isfinite(f):
                continue
            if f < min_coiso_fraction or f > max_leak_factor:
                continue
            r_exp_k = r_exp_prec.get(k, 0.0)
            if r_exp_k <= 0:
                continue
            r_leak[k] = float(f)
            leak_ratios[k] = float(f) * r_exp_k

    # Step 2 (aggressive fallback): when above-prec reference is unavailable,
    # estimate R_leak (formula-normalised) or R_k (no formula) from cross-pair
    # consistency of in-spectrum M+k candidates.
    # Step 2b: single-pair fallback (formula mode only) — if only one (M, M+k) pair
    # is found but the estimated f is large enough, accept it.  This covers sparse
    # spectra where the multi-pair CV check cannot fire.
    if aggressive:
        for k in range(1, max_isotope + 1):
            if k in leak_ratios:
                continue
            r_consensus = _estimate_leak_ratio_from_spectrum(
                smzs,
                sints,
                k,
                mz_tol=mz_tol,
                min_pairs=aggressive_min_pairs,
                consistency_cv=aggressive_cv,
                mol_counts=mol_counts,
                adduct_counts=adduct_counts_parsed if mol_counts is not None else None,
                adduct_delta_mass=adduct_delta_mass,
                prec_neutral_mass=prec_neutral_mass,
                r_exp_prec=r_exp_prec,
                dag_formula_mzs=dag_formula_mzs,
                dag_formula_strs=dag_formula_strs,
                dag_mz_tol=dag_mz_tol,
            )
            if r_consensus is None and mol_counts is not None and prec_neutral_mass > 0:
                # Single-pair fallback: try with min_pairs=1 (no CV check possible).
                # Only accepted when formula is known so f can be estimated, and only
                # when f ≥ aggressive_single_pair_min_f to guard against real fragments.
                r_single = _estimate_leak_ratio_from_spectrum(
                    smzs,
                    sints,
                    k,
                    mz_tol=mz_tol,
                    min_pairs=1,
                    consistency_cv=1.0,  # CV irrelevant for a single pair
                    mol_counts=mol_counts,
                    adduct_counts=adduct_counts_parsed,
                    adduct_delta_mass=adduct_delta_mass,
                    prec_neutral_mass=prec_neutral_mass,
                    r_exp_prec=r_exp_prec,
                    dag_formula_mzs=dag_formula_mzs,
                    dag_formula_strs=dag_formula_strs,
                    dag_mz_tol=dag_mz_tol,
                )
                if r_single is not None and r_single >= aggressive_single_pair_min_f:
                    r_consensus = r_single
            if r_consensus is None:
                continue
            if mol_counts is not None:
                # r_consensus ≈ f (co-isolation fraction); apply min_coiso_fraction gate
                if r_consensus < min_coiso_fraction:
                    continue
                r_leak[k] = r_consensus
                r_exp_k = r_exp_prec.get(k, 0.0)
                leak_ratios[k] = r_consensus * r_exp_k if r_exp_k > 0 else r_consensus
            else:
                leak_ratios[k] = r_consensus

    if not leak_ratios:
        return is_isotope

    # Step 3: Flag each peak that is the M+k satellite of a lower-m/z fragment.
    # When formula is known, per-fragment expected ratio = R_exp_frag_i × R_leak[k]
    # where R_exp_frag_i is estimated via m/z-based mol_scale.  This correctly
    # scales the window for each fragment's composition instead of applying the
    # precursor-level R_k uniformly.
    def _frag_expected(m0_mz: float, k: int) -> float:
        """Return per-fragment expected M+k/M+0 ratio given its M+0 m/z.

        Two code paths depending on whether the precursor formula is known:

        **Formula known** (``mol_counts is not None``):
            ``r_leak[k]`` = co-isolation fraction f (formula-normalised).
            Returns ``_expected_observed_isotope_ratio(f, r_nat_frag, k, r_exp_prec)``
            where r_nat_frag comes from the DAG or an m/z proxy
            (handled by ``_fragment_expected_ratio``).

        **No formula** (``mol_counts is None``):
            ``leak_ratios[k]`` = observed M+k / M+0 ratio of the precursor residual
            peaks (or cross-pair-consistent estimate from fragment pairs).  Returned
            directly as R_expected.  Applying mol_scale here would lower the absolute
            flagging floor from ``ratio_lo × R_k`` to ``ratio_lo × R_k × mol_scale``,
            which dramatically increases FP on small fragments because many small noise
            peaks fall within the shrunken window.  The formula-aware path handles
            per-fragment natural-abundance scaling correctly via ``r_leak`` +
            ``_fragment_expected_ratio``.
        """
        r_k = leak_ratios.get(k, 0.0)

        if k not in r_leak:
            # No-formula path: return the spectrum-level R_k directly.
            return r_k

        if prec_neutral_mass <= 0:
            return r_k

        # Formula path: r_leak[k] = f; compute exact per-fragment expected ratio.
        r_exp_frag = _fragment_expected_ratio(
            m0_mz,
            k,
            mol_counts,
            adduct_counts_parsed,
            adduct_delta_mass,
            prec_neutral_mass,
            dag_formula_mzs=dag_formula_mzs,
            dag_formula_strs=dag_formula_strs,
            dag_mz_tol=dag_mz_tol,
        )

        if r_exp_frag is None or r_exp_frag <= 0:
            mol_scale = max(
                0.0,
                min(1.0, (m0_mz - adduct_delta_mass) / prec_neutral_mass),
            )
            r_exp_frag = r_k * mol_scale

        return _expected_observed_isotope_ratio(r_leak[k], r_exp_frag, k, r_exp_prec)

    for i in range(n):
        frag_mz = smzs[i]
        frag_int = sints[i]

        if protect_dag_mono and _find_dag_formula(
            frag_mz, dag_formula_mzs, dag_formula_strs, dag_mz_tol
        ):
            continue

        for k in leak_ratios:
            k_tol = mz_tol if k == 1 else mz_tol * 2
            m0 = _find_peak(frag_mz - k * C13_SPACING, k_tol)
            if m0 is None:
                continue
            m0_mz, m0_int = m0

            if m0_int <= 0:
                continue

            obs = frag_int / m0_int
            R_expected = _frag_expected(m0_mz, k)

            if require_monotone and R_expected <= 1.0 and m0_int <= frag_int:
                continue

            if not (ratio_lo * R_expected <= obs <= ratio_hi * R_expected):
                continue

            # Envelope coherence: all intermediate M+j peaks must be present
            # and match their per-fragment expected ratio before accepting M+k.
            if require_envelope and k > 1:
                # Disable envelope requirement for Cl/Br where M+2 dominates M+1
                if k == 2 and R_expected > _frag_expected(m0_mz, 1):
                    pass
                else:
                    envelope_ok = True
                    for j in range(1, k):
                        if j not in leak_ratios:
                            envelope_ok = False
                            break
                        j_tol = mz_tol if j == 1 else mz_tol * 2
                        mj = _find_peak(frag_mz - (k - j) * C13_SPACING, j_tol)
                        if mj is None:
                            envelope_ok = False
                            break
                        _, mj_int = mj
                        R_exp_j = _frag_expected(m0_mz, j)
                        if require_monotone and R_exp_j <= 1.0 and mj_int >= m0_int:
                            envelope_ok = False
                            break
                        if m0_int > 0:
                            obs_j = mj_int / m0_int
                            if not (ratio_lo * R_exp_j <= obs_j <= ratio_hi * R_exp_j):
                                envelope_ok = False
                                break
                    if not envelope_ok:
                        continue

            is_isotope[sort_idx[i]] = True
            break

    return is_isotope


def detect_isotope_peaks_for_training_cleanup(
    mzs: np.ndarray,
    ints: np.ndarray,
    prec_mz: float,
    inst_type: str,
    formula: str | None,
    prec_type: str | None = None,
    max_isotope: int = 2,
    remove_regular_isotopes: bool = True,
    remove_coisolation: bool = True,
    regular_mz_tol: float | None = None,
    coiso_fraction_by_k: Mapping[int, float] | None = None,
    min_coiso_fraction: float = 0.05,
    max_leak_factor: float = 1.5,
    precursor_envelope_lo: float = 0.2,
    precursor_envelope_hi: float = 5.0,
    dag_formula_mzs: np.ndarray | None = None,
    dag_formula_strs: np.ndarray | None = None,
    dag_mz_tol: float = 0.015,
    protect_dag_mono: bool = False,
) -> np.ndarray:
    """Detect isotope peaks to remove from monoisotopic training targets.

    This combines two related but distinct cleanup sources:
    - regular isotope peaks, detected by formula-aware isotope offsets; and
    - co-isolation satellite peaks, detected by the precursor/co-isolation ``f`` model.

    Args:
        mzs: Peak m/z values of shape (N,).
        ints: Peak intensities of shape (N,).
        prec_mz: Precursor m/z.
        inst_type: Instrument type string.
        formula: Neutral molecular formula string. Required for regular isotope
            removal and formula-aware co-isolation scaling.
        prec_type: Adduct/precursor type string.
        max_isotope: Maximum isotope order to remove.
        remove_regular_isotopes: If True, remove regular M+k isotope peaks.
        remove_coisolation: If True, remove co-isolation satellite peaks.
        regular_mz_tol: Absolute m/z tolerance for regular isotope matching.
            Defaults to the instrument-specific isotope tolerance when available,
            otherwise 0.02 Da.
        coiso_fraction_by_k: Optional group-level co-isolation fractions.
        min_coiso_fraction: Minimum co-isolation fraction for co-isolation cleanup.
        max_leak_factor: Maximum accepted co-isolation fraction.
        precursor_envelope_lo: Lower bound for the normalized above-precursor
            M2/M1 envelope ratio.
        precursor_envelope_hi: Upper bound for the normalized above-precursor
            M2/M1 envelope ratio.
        dag_formula_mzs: Optional DAG monoisotopic fragment m/z values.
        dag_formula_strs: Formula strings aligned to ``dag_formula_mzs``.
        dag_mz_tol: Absolute tolerance for matching peaks to DAG formula m/z.
        protect_dag_mono: If True, never remove peaks that match a DAG
            monoisotopic formula.

    Returns:
        Boolean mask of shape (N,), where True means remove this isotope peak.
    """
    mzs = np.asarray(mzs, dtype=np.float64)
    ints = np.asarray(ints, dtype=np.float64)
    if mzs.shape != ints.shape or mzs.ndim != 1:
        raise ValueError(f"mzs and ints must be 1-D arrays of the same length, got {mzs.shape}")

    remove_mask = np.zeros(len(mzs), dtype=bool)
    if len(mzs) == 0:
        return remove_mask

    if formula:
        try:
            element_counts = parse_formula(formula)
        except Exception:
            element_counts = None
        if remove_regular_isotopes and element_counts is not None:
            mz_tol = regular_mz_tol
            if mz_tol is None:
                mz_tol = _INST_MZ_TOL.get(inst_type) or 0.02
            remove_mask |= detect_isotope_peaks_formula_aware(
                mzs,
                ints,
                element_counts,
                mz_tol=mz_tol,
                max_isotope=max_isotope,
                check_intensity=True,
                intensity_ratio_lo=0.0,
                intensity_ratio_hi=np.inf,
            )

    if remove_coisolation:
        remove_mask |= detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=prec_mz,
            inst_type=inst_type,
            max_isotope=max_isotope,
            formula=formula,
            prec_type=prec_type,
            min_coiso_fraction=min_coiso_fraction,
            max_leak_factor=max_leak_factor,
            precursor_envelope_lo=precursor_envelope_lo,
            precursor_envelope_hi=precursor_envelope_hi,
            coiso_fraction_by_k=coiso_fraction_by_k,
            dag_formula_mzs=dag_formula_mzs,
            dag_formula_strs=dag_formula_strs,
            dag_mz_tol=dag_mz_tol,
            protect_dag_mono=protect_dag_mono,
        )

    return remove_mask


# ---------------------------------------------------------------------------
# Cross-CE orphan peak detection
# ---------------------------------------------------------------------------

# Per-instrument absolute m/z tolerance for cross-CE peak matching.
# FT instruments have ~1–5 ppm accuracy so 0.02 Da covers up to m/z ~4000.
# QTOF is typically 5–20 ppm / 20–50 mDa.
# IT and unknown instruments have low mass accuracy; use 0.5 Da.
_CROSS_CE_MZ_TOL: dict[str, float] = {
    "FT": 0.02,
    "QTOF": 0.05,
}
_CROSS_CE_MZ_TOL_DEFAULT = 0.5


def detect_cross_ce_orphan_peaks(
    peaks_list: list[tuple[np.ndarray, np.ndarray]],
    inst_type: str = "",
    mz_tol: float | None = None,
    min_other_ces: int = 1,
) -> list[np.ndarray]:
    """Detect peaks in each CE spectrum that are absent from all other CE spectra.

    Given a set of spectra acquired at different collision energies for the same
    molecule / adduct / instrument (i.e. one ``group_id``), a peak is labelled
    "orphan" if its m/z does not match any peak in at least ``min_other_ces``
    of the remaining spectra within the instrument-appropriate m/z tolerance.

    Real fragment ions tend to appear across multiple CE spectra; orphan peaks
    are more likely to be noise, chimeric contamination, or co-isolation
    artefacts that happen to fall inside the isolation window only at a
    particular CE.

    The function is intentionally conservative: it uses a wide tolerance
    (relative to per-spectrum isotope detection) so that genuine fragments
    that shift slightly with CE are not mis-labelled.

    Args:
        peaks_list: List of ``(mzs, ints)`` tuples, one per CE spectrum.
            Each ``mzs`` and ``ints`` must be 1-D float arrays of the same
            length.  Arrays need not be sorted.  The list must contain at
            least one entry; groups with a single spectrum are returned
            all-False (no information to compare against).
        inst_type: Instrument type string (e.g. ``"FT"``, ``"QTOF"``,
            ``"IT"``).  Used to select a default m/z tolerance when
            ``mz_tol`` is ``None``.  Defaults to ``""`` (maps to 0.5 Da).
        mz_tol: Absolute m/z tolerance in Da for matching peaks across
            spectra.  Overrides the instrument-default when provided.
        min_other_ces: Minimum number of *other* CE spectra in which a
            matching peak must appear for the peak to be considered
            consistent.  Defaults to 1.

    Returns:
        List of boolean arrays, one per input spectrum.  ``True`` means the
        peak at that position is an orphan (absent from all other spectra).

    Raises:
        ValueError: If ``peaks_list`` is empty or any (mzs, ints) pair has
            mismatched lengths or wrong dimensionality.

    Example:
        >>> import numpy as np
        >>> # Three CE spectra for the same molecule
        >>> spec_a = (np.array([100.0, 150.0, 200.0]), np.array([1000., 500., 200.]))
        >>> spec_b = (np.array([100.0, 150.0, 999.0]), np.array([900., 450., 10.]))
        >>> spec_c = (np.array([100.0, 150.0, 200.0]), np.array([800., 300., 150.]))
        >>> masks = detect_cross_ce_orphan_peaks([spec_a, spec_b, spec_c], inst_type="FT")
        >>> masks[1]  # peak at 999.0 Da only in spec_b → orphan
        array([False, False,  True])
    """
    if not peaks_list:
        raise ValueError("peaks_list must contain at least one spectrum")

    validated: list[tuple[np.ndarray, np.ndarray]] = []
    for i, (mzs, ints) in enumerate(peaks_list):
        mzs = np.asarray(mzs, dtype=np.float64)
        ints = np.asarray(ints, dtype=np.float64)
        if mzs.ndim != 1 or ints.ndim != 1 or mzs.shape != ints.shape:
            raise ValueError(
                f"peaks_list[{i}]: mzs and ints must be 1-D arrays of equal length, "
                f"got mzs.shape={mzs.shape}, ints.shape={ints.shape}"
            )
        validated.append((mzs, ints))

    tol = mz_tol if mz_tol is not None else _CROSS_CE_MZ_TOL.get(inst_type, _CROSS_CE_MZ_TOL_DEFAULT)

    # Pre-sort each spectrum's m/z array for binary search.
    sorted_mzs: list[np.ndarray] = []
    for mzs, _ in validated:
        sorted_mzs.append(np.sort(mzs))

    n_spec = len(validated)
    result: list[np.ndarray] = []

    for i, (mzs, _) in enumerate(validated):
        n_peaks = len(mzs)
        orphan = np.ones(n_peaks, dtype=bool)
        if n_spec == 1:
            # No other spectra to compare against → cannot label anything.
            orphan[:] = False
            result.append(orphan)
            continue

        for j, peak_mz in enumerate(mzs):
            matches = 0
            for k in range(n_spec):
                if k == i:
                    continue
                smz = sorted_mzs[k]
                lo = int(np.searchsorted(smz, peak_mz - tol, side="left"))
                hi = int(np.searchsorted(smz, peak_mz + tol, side="right"))
                if lo < hi:
                    matches += 1
                    if matches >= min_other_ces:
                        break
            if matches >= min_other_ces:
                orphan[j] = False

        result.append(orphan)

    return result
