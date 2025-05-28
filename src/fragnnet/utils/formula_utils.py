import rdkit.Chem as Chem
import itertools
import numpy as np
import scipy
import re
from pyteomics.mass import Composition

from typing import List, Tuple, Dict, Literal
from fragnnet.utils.misc_utils import none_or_nan

PERIODIC_TABLE = Chem.GetPeriodicTable()
ELECTRON_MASS = 0.00054858

H_MASS = PERIODIC_TABLE.GetMostCommonIsotopeMass("H")
NA_MASS = PERIODIC_TABLE.GetMostCommonIsotopeMass("Na")
N_MASS = PERIODIC_TABLE.GetMostCommonIsotopeMass("N")
O_MASS = PERIODIC_TABLE.GetMostCommonIsotopeMass("O")
PREC_TYPE_TO_MASS_DIFF = {
	"[M+H]+": H_MASS - ELECTRON_MASS,
	"[M-H]-": -H_MASS + ELECTRON_MASS,
	"[M+Na]+": NA_MASS - ELECTRON_MASS,
	"[M+NH4]+": N_MASS + 4*H_MASS - ELECTRON_MASS,
	"[M-H2O+H]+": H_MASS - O_MASS - ELECTRON_MASS,
	"[M]+": - ELECTRON_MASS,
	"": 0.,
	"[M+H]+_NL": -H_MASS + ELECTRON_MASS,
	"[M-H]-_NL": H_MASS - ELECTRON_MASS,
}
PREC_TYPE_TO_FORMULA_DIFF = {
	"[M+H]+": "H1",
	"[M-H]-": "H-1",
	"[M+Na]+": "Na1",
	"[M+NH4]+": "N1H4",
	"[M-H2O+H]+": "O-1H-1",
	"[M]+": ""
}
PREC_TYPE_TO_COMP_DIFF = {
	k: Composition(v) for k, v in PREC_TYPE_TO_FORMULA_DIFF.items()
}

NEUTRON_MASS = 1.008665

def MASS(element: str | int) -> float:
	return PERIODIC_TABLE.GetMostCommonIsotopeMass(element)

# Just huge table grabbed for atomic number and its stable isotopes
# if atomic number is not in this table, that means we should even see it in the sample
# After all this is MS not Cyclotron and we are not try to finding super hevay elemet
# with this we don't need scan atomicno - atomicno * 4
# src http://moltensalt.org/references/static/downloads/pdf/stable-isotopes.pdf
# src https://periodictable.com/Elements/043/data.html
ISOTOPES_DICT = {1: [1, 2],
				 2: [3, 4],
				 3: [6, 7],
				 4: [9],
				 5: [10, 11],
				 6: [12, 13],
				 7: [14, 15],
				 8: [16, 17, 18],
				 9: [19],
				 10: [20, 21, 22],
				 11: [23],
				 12: [24, 25, 26],
				 13: [27],
				 14: [28, 29, 30],
				 15: [31],
				 16: [32, 33, 34, 36],
				 17: [35, 37],
				 18: [36, 38, 40],
				 19: [39, 41],
				 20: [40, 42, 43, 44, 46],
				 21: [45],
				 22: [46, 47, 48, 49, 50],
				 23: [51],
				 24: [50, 52, 53, 54],
				 25: [55],
				 26: [54, 56, 57, 58],
				 27: [59],
				 28: [58, 60, 61, 62, 64],
				 29: [63, 65],
				 30: [64, 66, 67, 68, 70],
				 31: [69, 71],
				 32: [70, 72, 73, 74],
				 33: [75],
				 34: [74, 76, 77, 78, 80],
				 35: [79, 81],
				 36: [78, 80, 82, 83, 84, 86],
				 37: [85],
				 38: [84, 86, 87, 88],
				 39: [89],
				 40: [90, 91, 92, 94],
				 41: [93],
				 42: [92, 94, 95, 96, 97, 98],
				 44: [100, 101, 102, 104, 96, 98, 99],
				 45: [103],
				 46: [102, 104, 105, 106, 108, 110],
				 47: [107, 109],
				 48: [106, 108, 110, 111, 112, 114],
				 49: [113],
				 50: [112, 114, 115, 116, 117, 118, 119, 120, 122, 124],
				 51: [121, 123],
				 52: [120, 122, 124, 125, 126],
				 53: [127],
				 54: [124, 126, 128, 129, 130, 131, 132, 134, 136],
				 55: [133],
				 56: [130, 132, 134, 135, 136, 137, 138],
				 57: [139],
				 58: [136, 138, 140, 142],
				 59: [141],
				 60: [142, 143, 145, 146, 148],
				 62: [144, 149, 150, 152, 154],
				 63: [151, 153],
				 64: [154, 155, 156, 157, 158, 160],
				 65: [159],
				 66: [156, 158, 160, 161, 162, 163, 164],
				 67: [165],
				 68: [162, 164, 166, 167, 168, 170],
				 69: [169],
				 70: [168, 170, 171, 172, 173, 174, 176],
				 71: [175],
				 72: [176, 177, 178, 179, 180],
				 73: [181],
				 74: [180, 182, 183, 184, 186],
				 75: [185],
				 76: [184, 187, 188, 189, 190, 192],
				 77: [191, 193],
				 78: [192, 194, 195, 196, 198],
				 79: [197],
				 80: [196, 198, 199, 200, 201, 202, 204],
				 81: [203, 205],
				 82: [204, 206, 207, 208],
				 90: [232]
				 }

# global cache for peak per element
# so we don't compute this a million time
# https://docs.python.org/3/glossary.html#term-global-interpreter-lock
# GIL will ensure this implicitly safe against concurrent access
# peaks_for_element_cache = {}

def enumerate_multinomial(N: int, p: List[float], cutoff_thold: float = 0.0001):
	"""
	Enumerate the state space of rolling a die with |p| faces
	N times

	returns: [(prob_1, (tuple_of_counts)), 
			   (prob_2, (tuple_of_counts)), 
			   ...]

	cutoff_thold: Do not return points in the state space 
	with prob less than cutoff_thold
	"""

	K = len(p)
	counts = []
	for s in itertools.combinations_with_replacement(range(K), N):
		count = np.zeros(K, dtype=np.int32)
		for i in s:
			count[i] += 1
		counts.append(count)
	counts = np.array(counts)

	p = p / np.sum(p)
	rv = scipy.stats.multinomial(N, p)

	probs = rv.pmf(counts)
	sort_idx = np.argsort(probs)[::-1]

	probs_sorted = probs[sort_idx]
	thold_idx = np.argwhere(np.cumsum(probs_sorted) >=
							1-cutoff_thold).flatten()[0]
	thold_idx += 1

	return list(zip(probs_sorted[:thold_idx],
					[tuple(a) for a in counts[sort_idx][:thold_idx]]))


def get_isotopes(atomicno: int, pct_thold: float = 0.01) -> List[Tuple[int, float, float]]:
	"""  
	Get all isotopes and their masses for atomic number atomicno

	Args:
		atomicno (int): atomic number
		pct_thold (float, optional): probility theorshold. Defaults to 0.01.

	Returns:
		List[Tuple[int,float,float]]:  [(int_mass, prob, exact_mass),...] for each isotope up to 4
	"""

	mass_probs = []
	pt = Chem.GetPeriodicTable()

	if atomicno not in ISOTOPES_DICT:
		return mass_probs
	else:
		for m in ISOTOPES_DICT[atomicno]: 
			#range(atomicno, 4*atomicno):
			pct = pt.GetAbundanceForIsotope(atomicno, m)
			if pct >= pct_thold:
				exact_mass = pt.GetMassForIsotope(atomicno, m)
				mass_probs.append((m, pct/100, exact_mass))
	return sorted(mass_probs, key=lambda r: -r[1])


def get_peaks_for_element(element: str, num_atoms: int) -> List:
	"""  
	Get the peaks for a formula with num_atoms of atomic no
	Args:
		element (str): element symbol
		num_atoms (int): num of atoms

	Returns:
		_type_: _description_
	"""

	atomicno = Chem.GetPeriodicTable().GetAtomicNumber(element)

	isotopes = get_isotopes(atomicno)
	probs = [i[1] for i in isotopes]

	comb_probs = enumerate_multinomial(num_atoms, probs, 0.001)

	exact_masses = [
		np.sum([isotopes[a_i][2]*a for a_i, a in enumerate(m)]) for p, m in comb_probs]

	peaks_masses_probs = [(comb_probs[i][0], m)
						  for i, m in enumerate(exact_masses)]
	return peaks_masses_probs


def get_peaks_for_formula(element_counts: Dict[str, int], threshold: float = 0.001, peaks_for_element_cache:dict = None) -> Tuple[List[float], List[float]]:
	"""for a given element configs, return masses with all possible isotopic configus

	Args:
		element_counts (Dict): element configs eg:  {"C":6,"H":7,"O":6,"N":0}
		threshold (float, optional): _description_. Defaults to 0.001.

	Returns:
		Tuple[List[float],List[float]]: ([mass,mass,mass,mass],[prob,prob,prob,prob])
	"""
	if peaks_for_element_cache is None:
		peaks_for_element_cache = {}

	peaks_masses_probs = {}
	for k, v in element_counts.items():
		if v > 0:
			key_str = k + str(v)
			if key_str not in peaks_for_element_cache:
				peaks_for_element_cache[key_str] = get_peaks_for_element(k, v)
			peaks_masses_probs[k] = peaks_for_element_cache[key_str]

	assert len(peaks_masses_probs) > 0, peaks_masses_probs
	probs = [[item[0] for item in v] for v in peaks_masses_probs.values()]
	masses = [[item[1] for item in v] for v in peaks_masses_probs.values()]
	all_probs = [np.prod(tup) for tup in itertools.product(*probs)]
	all_masses = [np.sum(tup) for tup in itertools.product(*masses)]
	keep_masses, keep_probs = [], []
	for mass, prob in zip(all_masses, all_probs):
		if prob >= threshold:
			keep_masses.append(mass)
			keep_probs.append(prob)
	return keep_masses, keep_probs


def formula_to_peak_mzs(formula, prec_type:Literal["[M+H]+","[M-H]-",""], 
			isotopes=True, return_map=False, 
			return_probs=False, peaks_for_element_cache: dict = None):
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
		peak_mz = 0.
		for k, v in element_counts.items():
			peak_mz += v*PERIODIC_TABLE.GetMostCommonIsotopeMass(k)
		peak_mz += mass_diff
		peak_mzs = [peak_mz]
		peak_probs = [1.0]
	else:	
		peak_mzs, peak_probs = get_peaks_for_formula(element_counts, peaks_for_element_cache = peaks_for_element_cache)
		peak_mzs = [peak_mz+mass_diff for peak_mz in peak_mzs]
	if return_map:
		return {peak_mz:formula for peak_mz in peak_mzs}
	elif return_probs:
		return peak_mzs, peak_probs
	else:
		return peak_mzs

def get_formulae_hill_notation(element_counts: Dict[str,int]) -> str:
	"""return formulae in string following hill notation

	Args:
		element_counts (Dict[str,int]): element configs eg:  {"C":6,"H":7,"O":6,"N":0}

	Returns:
		str: formulae string eg: BrClH2Si
	"""
	formulae_string = ''
	sorted_keys = list(element_counts.keys())
	sorted_keys.sort()
	for element in sorted_keys:
		formulae_string += element
		if element_counts[element] > 1:
			formulae_string += str(element_counts[element])
	return formulae_string

def parse_formula(formula:str) -> Dict[str,int]:
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
		formula = formula[:formula.index("-")]
	if "+" in formula:
		formula = formula[:formula.index("+")]
	for token in re.findall('[A-Z][a-z]?|\d+|.', formula):
		if token.isalpha():
			if cur_element is not None:
				assert cur_element not in element_counts
				element_counts[cur_element] = cur_count
			cur_element = token
			cur_count = 1
		elif token.isdigit():
			cur_count = int(token)
		else:
			print(formula)
			raise ValueError(f"Invalid token {token}")
	if cur_element is not None:
		assert cur_element not in element_counts
		element_counts[cur_element] = cur_count
	return element_counts

def get_elements_set(formula:str) -> set[str]:
    """
    """
    return set(list(parse_formula(formula).keys()))

