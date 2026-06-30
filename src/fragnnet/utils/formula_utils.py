import re
from typing import Literal

import rdkit.Chem as Chem
from pyteomics.mass import Composition

from fragnnet.utils.misc_utils import none_or_nan

PERIODIC_TABLE = Chem.GetPeriodicTable()
ELECTRON_MASS = 0.00054858

H_MASS = PERIODIC_TABLE.GetMostCommonIsotopeMass("H")
NA_MASS = PERIODIC_TABLE.GetMostCommonIsotopeMass("Na")
N_MASS = PERIODIC_TABLE.GetMostCommonIsotopeMass("N")
O_MASS = PERIODIC_TABLE.GetMostCommonIsotopeMass("O")
CL_MASS = PERIODIC_TABLE.GetMostCommonIsotopeMass("Cl")
K_MASS = PERIODIC_TABLE.GetMostCommonIsotopeMass("K")
I_MASS = PERIODIC_TABLE.GetMostCommonIsotopeMass("I")
C_MASS = PERIODIC_TABLE.GetMostCommonIsotopeMass("C")

PREC_TYPE_TO_MASS_DIFF = {
    "[M+H]+": H_MASS - ELECTRON_MASS,
    "[M+Na]+": NA_MASS - ELECTRON_MASS,
    "[M+H+2i]+": H_MASS + 2 * I_MASS - ELECTRON_MASS,
    "[M+K]+": K_MASS - ELECTRON_MASS,
    "[M+NH4]+": N_MASS + 4 * H_MASS - ELECTRON_MASS,
    "[M-H2O+H]+": -H_MASS - O_MASS - ELECTRON_MASS,
    "[M-2H2O+H]+": -3 * H_MASS - 2 * O_MASS - ELECTRON_MASS,
    "[M+H-NH3]+": -2 * H_MASS + N_MASS - ELECTRON_MASS,
    "[M]+": -ELECTRON_MASS,
    "": 0.0,
    # negative stuff
    "[M-H]-": -H_MASS + ELECTRON_MASS,
    "[M-H-CO2]-": -H_MASS + ELECTRON_MASS,
    "[M+Cl]-": CL_MASS + ELECTRON_MASS,
    "[M-H-H2O]-": -3 * H_MASS - O_MASS + ELECTRON_MASS,
    "[M-H+2i]-": -H_MASS + 2 * I_MASS + ELECTRON_MASS,
    "[M+HCOOH-H]-": C_MASS + 1 * H_MASS + 2 * O_MASS + ELECTRON_MASS,
    "[M+CH3COOH-H]-": 2 * C_MASS + 4 * H_MASS + 2 * O_MASS + ELECTRON_MASS,
    "[M+CH3COO]-": 2 * C_MASS + 4 * H_MASS + 2 * O_MASS + ELECTRON_MASS,
    # some NL adducts
    "[M+H]+_NL": -H_MASS + ELECTRON_MASS,
    "[M-H]-_NL": H_MASS - ELECTRON_MASS,
}
# charge transfer
PREC_TYPE_TO_CMF_MASS_DIFF = {
    "[M+H]+": H_MASS - ELECTRON_MASS,
    "[M+Na]+": H_MASS - ELECTRON_MASS,
    "[M+H+2i]+": H_MASS - ELECTRON_MASS,
    "[M+K]+": H_MASS - ELECTRON_MASS,
    "[M+NH4]+": H_MASS - ELECTRON_MASS,
    "[M-H2O+H]+": H_MASS - ELECTRON_MASS,
    "[M-2H2O+H]+": H_MASS - ELECTRON_MASS,
    "[M+H-NH3]+": H_MASS - ELECTRON_MASS,
    # negative stuff
    "[M-H]-": -H_MASS + ELECTRON_MASS,
    "[M-H-CO2]-": -H_MASS + ELECTRON_MASS,
    "[M+Cl]-": -H_MASS + ELECTRON_MASS,  # fragments are [frag-H]-, same carrier as [M-H]-
    "[M-H-H2O]-": -H_MASS + ELECTRON_MASS,
    "[M-H+2i]-": -H_MASS + ELECTRON_MASS,
    "[M+CH3COOH-H]-": -H_MASS + ELECTRON_MASS,
    "[M+CH3COO]-": -H_MASS + ELECTRON_MASS,
    "[M+HCOOH-H]-": -H_MASS + ELECTRON_MASS,
}

PREC_TYPE_TO_FORMULA_DIFF = {
    "[M+H]+": "H1",
    "[M+Na]+": "Na1",
    "[M+H+2i]+": "H1I2",
    "[M+K]+": "K1",
    "[M+NH4]+": "N1H4",
    "[M-H2O+H]+": "O-1H-1",
    "[M-2H2O+H]+": "O-2H-3",
    "[M+H-NH3]+": "H-2N-1",
    "[M]+": "",
    # negative stuff
    "[M-H]-": "H-1",
    "[M-H-CO2]-": "H-1C-1O-2",
    "[M+Cl]-": "Cl1",
    "[M-H-H2O]-": "H-2O-2",
    "[M-H+2i]-": "H-1I2",
    "[M+HCOOH-H]-": "CH2O2",
    "[M+CH3COOH-H]-": "C2H3O2",
    "[M+CH3COO]-": "C2H3O2",
}

PREC_TYPE_TO_COMP_DIFF = {k: Composition(v) for k, v in PREC_TYPE_TO_FORMULA_DIFF.items()}

NEUTRON_MASS = 1.008665
C13_SPACING = 1.003355  # ¹³C – ¹²C monoisotopic mass difference (Da); true M+1 peak spacing


def MASS(element: str | int) -> float:
    return PERIODIC_TABLE.GetMostCommonIsotopeMass(element)


def get_peaks_for_formula(
    element_counts: dict[str, int],
    threshold: float = 0.001,
    peaks_for_element_cache: dict = None,  # kept for API compatibility, unused
) -> tuple[list[float], list[float]]:
    """Return isotope peak masses and relative intensities for a molecular formula.

    Uses pyopenms ``CoarseIsotopePatternGenerator`` with IUPAC 2016 natural
    abundances.  Replaces the previous multinomial convolution implementation,
    which was 100–1000× slower and used an older RDKit abundance table.

    Args:
        element_counts: Element counts dict, e.g. ``{"C": 6, "H": 12, "O": 6}``.
        threshold: Minimum relative intensity to include.  Defaults to 0.001.
        peaks_for_element_cache: Ignored.  Kept for API compatibility.

    Returns:
        Tuple of ``(masses, probs)`` where each is a list of floats.  Masses
        are monoisotopic-based exact masses in Da; probs are renormalised
        relative intensities summing to ≤ 1 (peaks below threshold excluded).

    Raises:
        ValueError: If ``element_counts`` is empty or contains no positive counts.
    """
    from pyopenms import CoarseIsotopePatternGenerator, EmpiricalFormula

    formula_str = "".join(f"{el}{n}" for el, n in element_counts.items() if n > 0)
    if not formula_str:
        raise ValueError(f"element_counts has no positive counts: {element_counts}")
    emp = EmpiricalFormula(formula_str)
    gen = CoarseIsotopePatternGenerator(5)
    iso = gen.run(emp)
    iso.renormalize()
    container = iso.getContainer()
    keep = [(p.getMZ(), p.getIntensity()) for p in container if p.getIntensity() >= threshold]
    if not keep:
        return [container[0].getMZ()], [1.0]
    return [m for m, p in keep], [p for m, p in keep]


def formula_to_peak_mzs(
    formula,
    prec_type: Literal["[M+H]+", "[M-H]-", ""],
    isotopes=True,
    return_map=False,
    return_probs=False,
    peaks_for_element_cache: dict | None = None,
):
    """_summary_

    Args:
        formula (_type_): _description_
        prec_type (_type_): _description_
        isotopes (bool, optional): _description_. Defaults to True.
        return_map (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    assert not (return_map and return_probs)
    element_counts = parse_formula(formula)
    mass_diff = PREC_TYPE_TO_MASS_DIFF[prec_type]
    if not isotopes:
        peak_mz = 0.0
        for k, v in element_counts.items():
            peak_mz += v * PERIODIC_TABLE.GetMostCommonIsotopeMass(k)
        peak_mz += mass_diff
        peak_mzs = [peak_mz]
        peak_probs = [1.0]
    else:
        peak_mzs, peak_probs = get_peaks_for_formula(
            element_counts, peaks_for_element_cache=peaks_for_element_cache
        )
        peak_mzs = [peak_mz + mass_diff for peak_mz in peak_mzs]
    if return_map:
        return dict.fromkeys(peak_mzs, formula)
    elif return_probs:
        return peak_mzs, peak_probs
    else:
        return peak_mzs


def get_formulae_hill_notation(element_counts: dict[str, int]) -> str:
    """return formulae in string following hill notation

    Args:
        element_counts (Dict[str,int]): element configs eg:  {"C":6,"H":7,"O":6,"N":0}

    Returns:
        str: formulae string eg: BrClH2Si
    """
    formulae_string = ""
    sorted_keys = list(element_counts.keys())
    sorted_keys.sort()
    for element in sorted_keys:
        formulae_string += element
        if element_counts[element] > 1:
            formulae_string += str(element_counts[element])
    return formulae_string


def parse_formula(formula: str) -> dict[str, int]:
    """
        Return a Dict of count of each elemnt in the forumla
        NOTE: THIS DOES NOT HANDLE Condensed formulas eg. CH3CH2OH
    Args:
        formula (str): chemical forumla eg.CH4 or chemical forumla with adducts eg. CH4+H

    Raises:
        ValueError: _description_

    Returns:
        Dict[str,int]: count per element eg {"C":1, "H":4}
    """

    assert not none_or_nan(formula)
    cur_element = None
    cur_count = 1
    element_counts = {}
    if "-" in formula:
        formula = formula[: formula.index("-")]
    if "+" in formula:
        formula = formula[: formula.index("+")]
    # Use finditer to strictly parse tokens and detect unexpected characters
    pattern = re.finditer(r"[A-Z][a-z]?|\d+", formula)
    pos = 0
    for m in pattern:
        # if there's any gap between last position and this match, it's invalid
        if m.start() != pos:
            offending = formula[pos : m.start()]
            raise ValueError(
                f"Invalid substring '{offending}' at position {pos} in formula '{formula}'"
            )
        token = m.group(0)
        if token.isalpha():
            if cur_element is not None:
                assert cur_element not in element_counts
                element_counts[cur_element] = cur_count
            cur_element = token
            cur_count = 1
        elif token.isdigit():
            cur_count = int(token)
        pos = m.end()
    # any trailing characters after last token are invalid
    if pos != len(formula):
        offending = formula[pos:]
        raise ValueError(
            f"Invalid trailing substring '{offending}' at position {pos} in formula '{formula}'"
        )
    if cur_element is not None:
        assert cur_element not in element_counts
        element_counts[cur_element] = cur_count
    return element_counts


def isotope_offsets_from_formula(
    element_counts: dict[str, int],
    max_isotope: int = 4,
    threshold: float = 0.001,
) -> list[tuple[float, float]]:
    """Return isotopologue mass offsets and relative intensities for a molecular formula.

    Uses pyopenms ``CoarseIsotopePatternGenerator`` to compute the exact mass
    offset of each M+k peak from the monoisotopic M+0 peak.  These offsets
    differ from ``k * NEUTRON_MASS`` for heavy-isotope elements such as Cl
    and Br where even-mass isotopes dominate.

    Args:
        element_counts: Element counts dict as returned by ``parse_formula``.
        max_isotope: Maximum number of isotope peaks beyond M+0 to return.
            Defaults to 4.
        threshold: Minimum relative intensity to include.  Peaks below this
            threshold are discarded.  Defaults to 0.001.

    Returns:
        List of ``(offset_da, relative_intensity)`` tuples sorted by offset,
        excluding M+0 (offset ≈ 0).  Empty list if no peaks pass the threshold.

    Example:
        >>> ec = parse_formula("C10H8Br2")
        >>> offsets = isotope_offsets_from_formula(ec)
        >>> # M+2 ~0.46 dominates (81Br), not M+1 ~0.03
        >>> offsets[0][0]  # offset_da ≈ 1.0034
        1.003...
    """
    from pyopenms import CoarseIsotopePatternGenerator, EmpiricalFormula

    formula_str = "".join(f"{el}{n}" for el, n in element_counts.items() if n > 0)
    if not formula_str:
        return []
    emp = EmpiricalFormula(formula_str)
    gen = CoarseIsotopePatternGenerator(max_isotope + 1)
    iso = gen.run(emp)
    iso.renormalize()
    container = iso.getContainer()

    peaks = sorted((p.getMZ(), p.getIntensity()) for p in container)
    if not peaks:
        return []

    mono_mass = peaks[0][0]
    mono_intensity = peaks[0][1]
    return [
        (mass - mono_mass, intensity / mono_intensity)
        for mass, intensity in peaks[1:]
        if (intensity / mono_intensity) >= threshold
    ]


def get_elements_set(formula: str) -> set[str]:
    """
    method to get set of elements in a formula
    Args:
        formula (str): chemical formula eg.CH4 or chemical formula with adducts eg. CH4+H
    """
    return set(list(parse_formula(formula).keys()))
