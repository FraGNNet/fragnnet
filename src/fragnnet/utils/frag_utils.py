#from copy import deepcopy
import pickle
import sys
import tarfile
import traceback
from zipfile import ZipFile
import _pickle as cPickle
import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import rdkit.Chem.MolStandardize.rdMolStandardize as rdMolStandardize
import rdkit.Chem.rdmolops as rdmolops
import torch as th
import torch.nn.functional as F
import bz2
import os
import torch_geometric as pyg
from collections import Counter
from hashlib import blake2b

import fragnnet.frag.compute_frags as compute_frags
from fragnnet.utils.formula_utils import PREC_TYPE_TO_MASS_DIFF, formula_to_peak_mzs
from fragnnet.utils.misc_utils import PPM
from fragnnet.frag.compute_frags import (
	MASK_DTYPE,
	MAX_NUM_EDGES,
	MAX_NUM_NODES,
)

from fragnnet.utils.data_utils import mol_from_smiles

# full list of common elements
# NOTE: this maybe a bad idea to not cover all the element
# we should handle all the elemnet
ELEMENT_TO_VE = {
	"C": 4,
	"O": 2,
	"N": 3,
	"P": 3, # up to 5
	"S": 2, # up to 6
	"F": 1,
	"Cl": 1,
	"Br": 1,
	"I": 1,
	"Se": 2, # up to 6, same as S
	"Si": 4
}
HEAVY_ELEMENTS = list(ELEMENT_TO_VE.keys())
NUM_HEAVY_ELEMENTS = len(HEAVY_ELEMENTS)
ELEMENTS = HEAVY_ELEMENTS + ["H"]
NUM_ELEMENTS = len(ELEMENTS)

CANONICAL_ELEMENT_ORDER = ["C","H"] + sorted([elem for elem in HEAVY_ELEMENTS if elem != "C"])
CANONICAL_H_IDX = CANONICAL_ELEMENT_ORDER.index("H")

ELEMENT_TO_IDX = { elem: idx for idx, elem in enumerate(ELEMENT_TO_VE.keys())}
ELEMENT_TO_IDX["H"] = len(ELEMENT_TO_VE.keys())

IDX_TO_ELEMENT = { idx: elem for elem, idx in ELEMENT_TO_IDX.items()}

MAX_H_TRANSFER = 4
MAX_NUM_MZS_PER_FORMULA = 5

NODE_FEAT_DTYPE = th.int64
EDGE_FEAT_DTYPE = th.int64
META_DATA_DTYPE = th.float32

MASK_SIZE = 128
assert MASK_SIZE >= MAX_NUM_NODES, "MASK_SIZE should be larger than MAX_NUM_NODES"
assert MASK_SIZE % 64 == 0, "MASK_SIZE should be a multiple of 64"

BOND_TYPE_TO_IDX = {
	Chem.rdchem.BondType.names["AROMATIC"]: 2,
	Chem.rdchem.BondType.names["DOUBLE"]: 2,
	Chem.rdchem.BondType.names["TRIPLE"]: 3,
	Chem.rdchem.BondType.names["SINGLE"]: 1,
}

def convert_cc_mask_to_int(cc:list|np.ndarray)->int:
	"""covert a cc mask to an int

	Args: a list of 0 and 1s

	Returns:
		int: int presentation of input mask
	"""

	return sum([2**i for i in cc])

def convert_cc_int_to_mask(num_nodes:int,cc_int:int,bitmask=True) -> tuple[int]:
	"""_summary_

	Args:
		num_nodes (int): number of nodes
		cc_int (int): int version of cc mask
		bitmask (bool, optional): _description_. Defaults to True.

	Returns:
		list|np.ndarray: a list of 0 and 1s
	"""

	cc = []
	quot = cc_int
	for i in range(num_nodes):
		quot, rem = divmod(quot,2)
		if bitmask:
			cc.append(int(rem))
		else:
			if rem == 1:
				cc.append(i)
	return cc


def convert_cc_int_to_np_mask(num_nodes:int,cc_int:int,bitmask=True) -> np.ndarray:
	"""_summary_

	Args:
		num_nodes (int): number of nodes
		cc_int (int): int version of cc mask
		bitmask (bool, optional): _description_. Defaults to True.

	Returns:
		list|np.ndarray: a list of 0 and 1s
	"""

	cc = convert_cc_int_to_mask(num_nodes, cc_int, bitmask)
	np_mask = np.array(cc, dtype=MASK_DTYPE)
	return np_mask


def cc_bit_mask_to_atom_idx(cc:list|np.ndarray) -> np.ndarray:
	"""_summary_

	Args:
		cc (list | np.ndarray): _description_

	Returns:
		_type_: _description_
	"""
	cc_np = np.array(cc) if isinstance(cc,list) else cc
	atom_ids = np.where(cc_np == 1)[0].astype(np.int32)
	return atom_ids

def get_fraggen_input_arrays(mol_d:dict):
	"""_summary_

	Args:
		mol_d (_type_): _description_

	Returns:
		_type_: _description_
	"""
	num_nodes = mol_d["atom_mask_arr"].shape[0]
	num_edges = mol_d["bond_mask_arr"].shape[0]
	assert num_nodes <= MAX_NUM_NODES, num_nodes
	assert num_edges <= MAX_NUM_EDGES, num_edges
	edges = np.zeros((MAX_NUM_EDGES,2),dtype=np.intc)
	for bond_idx, bond in enumerate(mol_d["bonds"]):
		edges[bond_idx,0] = bond[0]
		edges[bond_idx,1] = bond[1]
	node_to_edge_idx = compute_frags.py_compute_node_to_edge_idx(num_nodes,num_edges,edges)
	# print(node_to_edge_idx)
	node_mask = np.zeros((num_nodes,),dtype=MASK_DTYPE)
	node_mask[:num_nodes] = 1
	edge_mask = np.zeros((num_edges,),dtype=MASK_DTYPE)
	edge_mask[:num_edges] = 1
	return num_nodes, num_edges, node_mask, edges, edge_mask, node_to_edge_idx

def extract_mol_info(smiles_or_mol, use_default_valence=False) ->dict:
	"""Method to exatract mol infomation in to dict

	Args:
		smiles_or_mol (_type_): _description_
		use_default_valence (bool, optional): _description_. Defaults to False.

	Raises:
		ValueError: _description_

	Returns:
		dict : 
		mol_d["mol"]: mol object
		mol_d["num_hs"]: number of totoal Hs
		mol_d["sbond_arr"]: single bond array
		mol_d["ve_arr"]: max velance array
		mol_d["hs_arr"]: max Hs per atom arary
		mol_d["elems"]: elemns per atom
		mol_d["bonds"]: a list of from atom - to atom pairs
		mol_d["bond_mask_arr"]: defualt bond mask, this should be just 1s
		mol_d["atom_mask_arr"] : defualt atom  mask, this should be just 1s
		mol_d["atoms_to_bonds"]: 
		mol_d["element_counts"] : atom count per element
	"""
	if isinstance(smiles_or_mol,str):
		# TODO: change this to be consistent with the approach in data_utils.py
		# rdkit mol stuff
		# mol = Chem.MolFromSmiles(smiles_or_mol)
		# rdMolStandardize.Cleanup(mol)
		# te = rdMolStandardize.TautomerEnumerator()
		# mol = te.Canonicalize(mol)
		mol = mol_from_smiles(smiles_or_mol)
	else:
		assert isinstance(smiles_or_mol,Chem.rdchem.Mol), type(smiles_or_mol)
		mol = smiles_or_mol

	pt = Chem.GetPeriodicTable()
	#GetValenceList
	# some checks
	charge = rdmolops.GetFormalCharge(mol)
	assert charge == 0, charge
	# enumerate atoms
	sbond_arr = []
	ve_arr = []
	hs_arr = []
	elems = []
	elem_idxs = []
	num_hs = 0
	num_atoms = 0
	num_radicals = 0
	element_counts = {elem:0 for elem in ELEMENT_TO_VE.keys()}

	for atom in mol.GetAtoms():
		cur_idx = atom.GetIdx()
		cur_num_hs = atom.GetTotalNumHs()
		cur_deg = atom.GetTotalDegree()
		# cur_num_bonds = atom.GetNumBonds()
		cur_element = atom.GetSymbol()

		if cur_element not in element_counts:
			raise ValueError(f"Molecules with {cur_element} atom(s) currently not supported")
		
		element_counts[cur_element] += 1
		elems.append(cur_element)
		elem_idxs.append(ELEMENT_TO_IDX[cur_element])
		# number of single bond need to attach to this atom to keep atom connected
		# this equals to replace all all bond to single, and count how many single bond
		sbond_arr.append(cur_deg - cur_num_hs)
		#ve_arr.append(ELEMENT_TO_VE[cur_element])

		# set valence value for each atom
		# assumption we will use current valance unless
		# use default valence flag is set to true
		# not defualt valence can be -1 for transition metals
		# on paper we should not encounter them at all
		ve_value = atom.GetTotalValence()
		if use_default_valence:
			default_valence = pt.GetDefaultValence(cur_element)
			if default_valence != -1:
				ve_value = min(ve_value, default_valence)

		ve_arr.append(ve_value)
		hs_arr.append(cur_num_hs)
		num_hs += cur_num_hs
		num_atoms += 1
		num_radicals += atom.GetNumRadicalElectrons()

	element_counts["H"] = num_hs
	assert num_radicals == 0, num_radicals
	sbond_arr = np.array(sbond_arr, dtype=np.int32)
	ve_arr = np.array(ve_arr, dtype=np.int32)
	hs_arr = np.array(hs_arr, dtype=np.int32)
	# enumerate bonds
	atoms_to_bonds = {}
	bonds, bond_type_idxs = [], []
	num_bonds = 0
	adj = np.zeros((num_atoms, num_atoms), dtype=np.int32)
	#adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
	for bond in mol.GetBonds():
		cur_idx = bond.GetIdx()
		from_idx = bond.GetBeginAtomIdx()
		to_idx = bond.GetEndAtomIdx()
		cur_type_idx = BOND_TYPE_TO_IDX[bond.GetBondType()]
		assert from_idx != to_idx
		adj[from_idx, to_idx] = 1

		bonds.append((from_idx, to_idx))
		bond_type_idxs.append(cur_type_idx)

		if from_idx not in atoms_to_bonds:
			atoms_to_bonds[from_idx] = [cur_idx]
		else:
			atoms_to_bonds[from_idx].append(cur_idx)

		if to_idx not in atoms_to_bonds:
			atoms_to_bonds[to_idx] = [cur_idx]
		else:
			atoms_to_bonds[to_idx].append(cur_idx)
		num_bonds += 1
	bonds = np.array(bonds, dtype=np.int32)
	mol_d = {}
	mol_d["mol"] = mol
	mol_d["num_hs"] = num_hs
	mol_d["sbond_arr"] = sbond_arr
	mol_d["ve_arr"] = ve_arr
	mol_d["hs_arr"] = hs_arr
	mol_d["elems"] = elems
	mol_d["elem_idxs"] = elem_idxs
	mol_d["bonds"] = bonds
	mol_d["bond_type_idxs"] = bond_type_idxs
	mol_d["bond_mask_arr"] = np.ones((num_bonds,), dtype=bool)
	mol_d["atom_mask_arr"] = np.ones((num_atoms,), dtype=bool) # can be computed
	mol_d["atoms_to_bonds"] = atoms_to_bonds # can be computed
	mol_d["element_counts"] = element_counts
	return mol_d

def compute_cc_h_cap(cc_atom_ids: np.ndarray,ve_arr:np.ndarray,sbond_arr:np.ndarray,num_radicals:int):
	"""compute max amount of Hs a cc can have.  
		For any ccs the max amount of Hs it can have is the congifcation where all the bond are single
		And all the atom has max amount of Hs
	Args:
		cc (list|np.ndarray): cc mask in list form
		ve_arr (list|np.ndarray): max velance each atom can have
		sbond_arr (_type_): _description_
		num_radicals (_type_): _description_

	Returns:
		_type_: _description_
	"""
	
	assert num_radicals == 0
	if not isinstance(cc_atom_ids,np.ndarray):
		cc_atom_ids = np.array(list(cc_atom_ids))
	cap_sbond_arr = sbond_arr[cc_atom_ids]
	cap_ve_arr = ve_arr[cc_atom_ids]
	cap_ve_mask = cap_ve_arr < cap_sbond_arr
	cap_ve_arr[cap_ve_mask] = cap_sbond_arr[cap_ve_mask]
	return np.sum(cap_ve_arr) - np.sum(cap_sbond_arr) - num_radicals

def compute_cc_h_floor(cc_atom_ids: np.ndarray,ve_arr: np.ndarray, sbond_arr: np.ndarray, \
    	num_radicals:int,bonds:np.ndarray,atoms_to_bonds:dict,bond_mask_arr:np.ndarray):
	"""compute min amount of Hs a cc can have.

	Args:
		cc_atom_ids (np.ndarray): atom ids in the cc
		ve_arr (np.ndarray): _description_
		sbond_arr (np.ndarray): _description_
		bonds (np.ndarray): _description_
		atoms_to_bonds (dict): _description_
		bond_mask_arr (np.ndarray): _description_

	Returns:
		_type_: _description_
	"""
	assert num_radicals == 0
	# if we could use update from single bond to double bond to get an electron pair
	diff_arr = np.maximum(ve_arr - sbond_arr,0)
	#cc_atoms = list(cc_atom_ids)
	# print(diff_arr)
	# this computes a lower bound
	h_arr = np.copy(diff_arr)

	for _, atom in enumerate(cc_atom_ids):
		bond_idxs = atoms_to_bonds[atom]
		for bond_idx in bond_idxs:
			if h_arr[atom] == 0:
				break
			if not bond_mask_arr[bond_idx]:
				continue
			bond = bonds[bond_idx]
			if bond[0] == atom:
				other = bond[1]
			else:
				other = bond[0]
			# dont't form more than 3 bonds with anything!
			h_arr[atom] = max(0,h_arr[atom]-min(diff_arr[other],2))
	# print(h_arr)
	cc_floor = sum(h_arr[atom] for atom in cc_atom_ids)
	cc_floor -= num_radicals
	cc_floor = max(cc_floor,0) # why can cc_floor be negative?
	return cc_floor

def compute_approximate_formula(cc:list|np.ndarray,mol_d:dict
				,max_h_transfer:int
				,formula_strs:bool=False
				,bitmask:bool=True
				,base_formula = None) ->dict[int,str]|dict[int,np.ndarray]:
	"""given a connected component, comupute all formula within give h shift

	Args:
		cc (list|np.ndarray): connected component
		mol_d (dict): mol dictionary
		max_h_transfer (_type_): max number of h movement allowed
		formula_strs (bool, optional): _description_. Defaults to False.

	Returns:
		_type_: _description_
	"""

	bonds = mol_d["bonds"] # bond definition, atom id-atom id, never mutated
	atoms_to_bonds = mol_d["atoms_to_bonds"] # never mutated
	ve_arr = np.copy(mol_d["ve_arr"])
	sbond_arr = np.copy(mol_d["sbond_arr"])
	num_hs = mol_d["num_hs"]
	hs_arr = mol_d["hs_arr"]
	elem_idxs = mol_d["elem_idxs"]
	bond_mask_arr = mol_d["bond_mask_arr"]
	
	if base_formula is None and bitmask:
		base_formula = cc_bitmask_to_formula_arr(cc,elem_idxs)
	elif base_formula is None and not bitmask:
		base_formula = cc_to_formula_arr(cc,elem_idxs, bitmask = False)

	atom_ids = cc_bit_mask_to_atom_idx(cc) if bitmask else cc

	sbond_arr, bond_mask_arr = compute_frags.update_bonds(atom_ids,sbond_arr,bond_mask_arr,bonds,atoms_to_bonds)
	cap = compute_cc_h_cap(atom_ids,ve_arr,sbond_arr,0)
	floor = compute_frags.compute_cc_h_floor(atom_ids,ve_arr,sbond_arr,0,bonds,atoms_to_bonds,bond_mask_arr)

	# check floor and cap
	assert floor <= cap, (floor,cap,atom_ids)
	assert floor >= 0, floor

	# check how many Hs each atom have on the mol
	num_hs_prior = 0
	for atom in atom_ids:
		num_hs_prior += hs_arr[atom]

	# update min and max
	floor = max(floor,num_hs_prior-max_h_transfer)
	cap = min(cap,num_hs_prior+max_h_transfer,num_hs)

	delta_h_to_formula, delta_h_to_h_count = {}, {}

	if formula_strs:
		formula_template, formual_no_h = formula_arr_to_str(base_formula, get_h_template=True)

	for delta_h in range(-max_h_transfer,max_h_transfer+1):
		h = num_hs_prior + delta_h
		# delta_h_to_h_count[delta_h] = h
		if h < floor or h > cap:
			# special invalid formula
			formula = np.zeros_like(base_formula)
			formula_str = ""
			delta_h_to_h_count[delta_h] = -1
		else:
			formula = np.copy(base_formula)
			formula[ELEMENT_TO_IDX["H"]] = h
			formula = tuple(formula)
			if h == 0:
				formula_str = formual_no_h
			else:
				formula_str = formula_template.format(h)
			delta_h_to_h_count[delta_h] = h
		if formula_strs:
			formula = formula_str
		delta_h_to_formula[delta_h] = formula
	return delta_h_to_formula, delta_h_to_h_count

def update_bonds(cc_atom_ids:np.ndarray,sbond_arr:np.ndarray,bond_mask_arr:np.ndarray,bonds:np.ndarray,atoms_to_bonds:dict):
	"""_summary_

	Args:
		frag (np.ndarray): _description_
		sbond_arr (np.ndarray): _description_
		bond_mask_arr (np.ndarray): _description_
		bonds (np.ndarray): _description_
		atoms_to_bonds (dict): _description_

	Returns:
		_type_: _description_
	"""
	sbond_arr = np.zeros_like(sbond_arr,dtype=sbond_arr.dtype)
	bond_mask_arr = np.zeros_like(bond_mask_arr,dtype=bool)
	for atom in cc_atom_ids:
		for bond_idx in atoms_to_bonds[atom]:
			bond = bonds[bond_idx]
			if bond[0] in cc_atom_ids and bond[1] in cc_atom_ids:
				sbond_arr[atom] += 1
				bond_mask_arr[bond_idx] = True
	return sbond_arr, bond_mask_arr

def compute_approximate_cchs(ccs,mol_d,h_prior=False,max_h_transfer=MAX_H_TRANSFER):

	bonds = mol_d["bonds"] # never mutated
	atoms_to_bonds = mol_d["atoms_to_bonds"] # never mutated
	ve_arr = np.copy(mol_d["ve_arr"])
	sbond_arr = np.copy(mol_d["sbond_arr"])
	num_hs = mol_d["num_hs"]
	hs_arr = mol_d["hs_arr"]
	mol_d["elems"]
	bond_mask_arr = mol_d["bond_mask_arr"]
	all_cchs = set()
	for cc in list(set(ccs)):
		sbond_arr, bond_mask_arr = compute_frags.update_bonds(cc,sbond_arr,bond_mask_arr,bonds,atoms_to_bonds)
		cc_atom_ids = cc_bit_mask_to_atom_idx(cc)
		cap = compute_cc_h_cap(cc_atom_ids,ve_arr,sbond_arr,0)
		floor = compute_frags.compute_cc_h_floor(cc_atom_ids,ve_arr,sbond_arr,0,bonds,atoms_to_bonds,bond_mask_arr)
		assert floor <= cap, (floor,cap,cc)
		assert floor >= 0, floor
		if h_prior:
			num_hs_prior = 0
			for atom in cc:
				num_hs_prior += hs_arr[atom]
			floor = max(floor,num_hs_prior-max_h_transfer)
			cap = min(cap,num_hs_prior+max_h_transfer,num_hs)
			# cap = max(cap,floor)
		else:
			cap = min(cap,num_hs)
		# cc_cchs = []
		for h in range(floor,cap+1):
			# cc_cchs.append((cc,h))
			all_cchs.add((cc,h))
		# cchs.append(cc_cchs)
		# assert floor <= cap, (floor,cap,cc)
		# assert floor >= 0, floor
	return all_cchs

def cc_to_formula_arr(cc,elems,bitmask=False):
	# does not include h count
	formula_arr = np.zeros([len(ELEMENT_TO_IDX)],dtype=int)
	if bitmask:
		for i in range(len(cc)):
			if cc[i]:
				elem = elems[i]
				formula_arr[ELEMENT_TO_IDX[elem]] += 1
	else:
		for atom in cc:
			elem = elems[atom]
			formula_arr[ELEMENT_TO_IDX[elem]] += 1
	#print(">cc_to_formula_arr", formula_arr)
	return formula_arr


def cc_bitmask_to_formula_arr(cc,elem_idxs) -> np.ndarray:
	""" fast cc bit mask to formula arr

	Args:
		cc (_type_): _description_
		elem_idxs (_type_): _description_

	Returns:
		np.ndarray: _description_
	"""
	# does not include h count
	formula_arr = np.zeros([len(ELEMENT_TO_IDX)],dtype=MASK_DTYPE)
	elem_idxs_np = np.array(elem_idxs) + 1
	#print(cc, elem_idxs_np)
	elem_idx_np = np.multiply(cc, elem_idxs_np)
	unique, counts = np.unique(elem_idx_np, return_counts=True)
	for elem_idx, count in zip(unique, counts):
		#print(unique, counts)
		if elem_idx == 0:
			continue
		else:
			formula_arr[elem_idx-1] = count
	#print(">cc_bitmask_to_formula_arr", formula_arr)
	return formula_arr

def formula_arr_to_str(formula_arr, get_h_template = False):
	""" use canonical order """
  
	elem_d = {}

	for idx, count in enumerate(formula_arr):
		elem = IDX_TO_ELEMENT[idx]
		elem_d[elem] = count

	if not get_h_template:
		formula_str = ""
		for elem in CANONICAL_ELEMENT_ORDER:
			if elem in elem_d:
				count = elem_d[elem]
				if count > 0:
					formula_str += elem
				if count > 1:
					formula_str += str(count)
		return formula_str
	else:
		formula_str = ""
		formula_str_no_h = ""
		for elem in CANONICAL_ELEMENT_ORDER:
			if elem in elem_d:
				if elem == 'H':
					formula_str += "H{:d}"
				else:
					count = elem_d[elem]
					if count > 0:
						formula_str += elem
						formula_str_no_h += elem
					if count > 1:
						formula_str += str(count)
						formula_str_no_h += str(count)
		return formula_str, formula_str_no_h


def compute_frag_peak_stats(peaks,formula_peak_mzs,formula_peak_probs,idx_by_h_delta,\
							 prec_mz, allowed_h_transfer, tolerance=0.01,\
							 prec_type="[M+H]+", is_ppm=False):
	"""_summary_

	Args:
		peaks (_type_): _description_
		formula_peak_mzs (_type_): _description_
		formula_peak_probs (_type_): _description_
		idx_by_h_delta (_type_): _description_
		allowed_h_transfer (_type_): _description_
		tolerance (float, optional): _description_. Defaults to 0.01.
		is_ppm (bool, optional): _description_. Defaults to False.

	Returns:
		_type_: _description_
	"""
	true_mzs, true_ints = list(zip(*peaks))
	theoretical_mzs = formula_peak_mzs
	theoretical_probs = formula_peak_probs

	allowed_idx = list(idx_by_h_delta[0])
	for h in range(1, allowed_h_transfer):
		allowed_idx += list(idx_by_h_delta[2 * h - 1])
		allowed_idx += list(idx_by_h_delta[2 * h])
	allowed_idx = list(set(allowed_idx))
	#print(allowed_idx)
	indices = th.tensor(allowed_idx)
	theoretical_mzs = th.index_select(theoretical_mzs,0, indices)
	theoretical_probs = th.index_select(theoretical_probs,0, indices)

	prec_mask = (theoretical_probs > 0.).type(th.float32)
	# account for adduct mass
	theoretical_mzs = theoretical_mzs + prec_mask * PREC_TYPE_TO_MASS_DIFF[prec_type]
	# compute overlap
	overlap_true_idxs = []
	overlap_true_ints = []
	overlap_pred_idxs = []
	overlap_pred_peak_counts = []
	overlap_pred_formula_counts = []

	# check true_mzs
	for true_idx, true_mz in enumerate(true_mzs):
		mz_diffs = th.abs(theoretical_mzs-true_mz)
		if not is_ppm:
			mz_close = mz_diffs < tolerance 
		else:
			mz_close = mz_diffs < (true_mz * tolerance * PPM)
		if th.any(mz_close):
			pred_idx = th.nonzero(mz_close,as_tuple=False)
			num_formula_match = th.sum(th.any(mz_close,dim=1).type(th.int32)).item()
			num_peak_match = pred_idx.shape[0]
			overlap_true_idxs.append(true_idx)
			overlap_true_ints.append(true_ints[true_idx])
			overlap_pred_idxs.append(pred_idx)
			overlap_pred_peak_counts.append(num_peak_match)
			overlap_pred_formula_counts.append(num_formula_match)
	# remove duplicates
	if len(overlap_pred_idxs) > 0:
		overlap_pred_idxs = th.unique(th.cat(overlap_pred_idxs,dim=0),dim=0)
	else:
		overlap_pred_idxs = th.zeros((0,2))
	recall = len(overlap_true_idxs) / len(true_mzs)
	w_recall = np.sum(overlap_true_ints) / np.sum(true_ints)
	prec = len(overlap_pred_idxs) / th.sum((theoretical_probs>0.).type(th.int32)).item()
	if len(overlap_pred_peak_counts) > 0:
		ppt_peak = np.mean(overlap_pred_peak_counts)
		ppt_formula = np.mean(overlap_pred_formula_counts)
	else:
		ppt_peak = np.nan
		ppt_formula = np.nan
	# check prec_mz stuff
	prec_recalls = []
	for comp_mzs in [theoretical_mzs,th.tensor(true_mzs)]:
		prec_mz_diffs = th.abs(comp_mzs-prec_mz)
		if not is_ppm:
			prec_mz_close = prec_mz_diffs < tolerance 
		else:
			prec_mz_close = prec_mz_diffs < (prec_mz * tolerance * PPM)
		if th.any(prec_mz_close):
			prec_recalls.append(1.)
		else:
			prec_recalls.append(0.)
	prec_recall, prec_spec_recall = prec_recalls
	return pd.Series([recall,w_recall,prec,ppt_peak,ppt_formula,prec_recall,prec_spec_recall])

def th_long_to_mask(long):
	"""_summary_

	Args:
		long (_type_): _description_

	Returns:
		_type_: _description_
	"""
	# long is N x MASK_SIZE//64
	num_dims = long.shape[1]
	long = long.reshape(long.shape[0],num_dims,1)
	mask = 2**th.arange(64-1,-1,-1,device=long.device)
	mask = mask.reshape(1,1,64)
	return long.bitwise_and(mask).ne(0).reshape(long.shape[0],-1)


def th_mask_to_long(mask):
	# mask is N x MASK_SIZE
	num_dims = mask.shape[1]//64
	mask = mask.reshape(mask.shape[0],num_dims,64)
	long = 2**th.arange(64-1,-1,-1,device=mask.device).expand(1,num_dims,64)
	return th.sum(long*mask,dim=2).long()


def compute_dags(
		mol_d:dict,
		max_depth:int,
		h_prior:bool,
		max_h_transfer:int,
		frag_max_time:int,
		isotopes:bool = False,
		nb_isomorphic:bool = False,
		b_isomorphic:bool = False,
		max_iterations:int = -1) -> dict:
	"""_summary_

	Args:
		mol_d (dict): _description_
		max_depth (int): _description_
		h_prior (bool): _description_
		max_h_transfer (int): _description_
		frag_max_time (int): _description_
		isotopes (bool, optional): _description_. Defaults to False.

	Raises:
		ValueError: _description_

	Returns:
		dict: _description_
	"""

	assert h_prior in [True,False], h_prior
	num_nodes, num_edges, node_mask, edges, edge_mask, node_to_edge_idx = get_fraggen_input_arrays(mol_d)
	# time the recursive part
	if frag_max_time is None:
		frag_max_time = int(1e6)
	
	node_mask = node_mask.astype(MASK_DTYPE)
	edge_mask = edge_mask.astype(MASK_DTYPE)
	nodes_mask_matrix, nodes_depth_matrix, dag_edges_matrix, dag_frag_meta = compute_frags.compute_ccs(
		num_nodes,
		num_edges,
		node_mask,
		edges,
		edge_mask,
		node_to_edge_idx,
		max_depth,
		frag_max_time
	)

	if nb_isomorphic:
		node_nb_hashes = get_subgraph_hashes(
			nodes_mask_matrix=nodes_mask_matrix,
			elems=mol_d["elems"], 
			bond_type_idxs=mol_d["bond_type_idxs"], 
			edges=edges[:num_edges], 
			node_to_edge_idx=node_to_edge_idx[:num_nodes], 
			include_bond_type=False,
			max_iterations=max_iterations)
	else:
		node_nb_hashes = None

	# if b_isomorphic:
	# 	node_b_hashes = get_subgraph_hashes(
	# 		nodes_mask_matrix=nodes_mask_matrix,
	# 		elems=mol_d["elems"], 
	# 		bond_type_idxs=mol_d["bond_type_idxs"], 
	# 		edges=edges[:num_edges], 
	# 		node_to_edge_idx=node_to_edge_idx[:num_nodes], 
	# 		include_bond_type=True)
	# else:
	# 	node_b_hashes = None

	# get meta
	reached_depth = dag_frag_meta["reached_depth"]
	edges_min_depth = dag_frag_meta["edges_min_depth"]
	nodes_min_depth = dag_frag_meta["nodes_min_depth"]

	# add node depth information
	# convert to a one hot encoding
	depth_node_feat_size = max_depth+1 #depth
	cc_node_feat_size = MASK_SIZE//64 # mask
	base_formula_node_feat_size = len(ELEMENT_TO_IDX) # base_formula
	formula_node_feat_size = 1+2*max_h_transfer # h_formulae_idx, at max we can have 1 + 2 * max_h_transfer different formula
	h_count_node_feat_size = formula_node_feat_size
	nb_iso_node_feat_size = 1 if nb_isomorphic else 0
	cc_edge_feat_size = cc_node_feat_size
	base_formula_edge_feat_size = base_formula_node_feat_size
	h_range_edge_feat_size = 2
	node_feat_shapes = [depth_node_feat_size,cc_node_feat_size,base_formula_node_feat_size,formula_node_feat_size,h_count_node_feat_size,nb_iso_node_feat_size]
	edge_feat_shapes = [cc_edge_feat_size,base_formula_edge_feat_size,h_range_edge_feat_size]
	# total_node_feat_size = sum(node_feat_shapes)

	# add node h count and element information
	hs_arr = mol_d["hs_arr"]
	elem_idxs = mol_d["elem_idxs"]
	cc_formula_list = []
	hs_arr_np = np.array(hs_arr)
	for cc_mask in nodes_mask_matrix:
		cc_h_count = np.sum(np.multiply(cc_mask, hs_arr_np))
		cc_formula = cc_bitmask_to_formula_arr(cc_mask,elem_idxs)
		cc_formula[ELEMENT_TO_IDX["H"]] = cc_h_count
		cc_formula_list.append(cc_formula)
	node_base_formula_matrix = np.stack(cc_formula_list, dtype=MASK_DTYPE)
	
	# map nodes to formulae
	formula_d_list, h_count_d_list = [], []
	formula_counts = {}
	for idx, cc_mask in enumerate(nodes_mask_matrix):
		base_formula = node_base_formula_matrix[idx]
		base_formula[ELEMENT_TO_IDX["H"]] = 0 # remove Hs
		delta_h_to_formula, delta_h_to_h_count = compute_approximate_formula(cc_mask,mol_d,max_h_transfer,formula_strs=True, base_formula = base_formula)
		formula_d_list.append(delta_h_to_formula)
		h_count_d_list.append(delta_h_to_h_count)
		formulae = list(delta_h_to_formula.values())
		for formula in formulae:
			formula_counts[formula] = formula_counts.get(formula,0) + 1
   
	# map nodes to formulae indices
	formula_to_idx = {formula:idx for idx,formula in enumerate(sorted(list(formula_counts.keys())))}
	idx_to_formula = {idx:formula for formula,idx in formula_to_idx.items()}
	formula_idx_by_h_delta = [set() for _ in range(1 + 2 * max_h_transfer)]

	formula_idx_list, h_count_list = [], []
	for formulae_dict, h_count_dict in zip(formula_d_list,h_count_d_list):
		formulae_idxs = np.zeros(formula_node_feat_size, dtype=np.int16)
		h_counts = np.zeros(h_count_node_feat_size, dtype=np.int16)
		for h_delta in formulae_dict:
			formula = formulae_dict[h_delta]
			h_count = h_count_dict[h_delta]
			formula_idx = formula_to_idx[formula]
			# h_delta [0,-1,1,-2,2,-3,3,-4,4]
			h_delta_idx = h_delta * 2 if h_delta >= 0 else (-h_delta * 2) - 1
			formulae_idxs[h_delta_idx] = formula_idx
			h_counts[h_delta_idx] = h_count
			formula_idx_by_h_delta[h_delta_idx].add(formula_idx)
		formula_idx_list.append(formulae_idxs)
		h_count_list.append(h_counts)

	node_formulae_matrix = np.stack(formula_idx_list, dtype=np.int16)
	node_h_count_matrix = np.stack(h_count_list, dtype=np.int16)

	# map nodes to nb_isomorphism indices
	if nb_isomorphic:
		nb_iso_map = {hash:idx for idx,hash in enumerate(sorted(list(set(node_nb_hashes))))}
		node_nb_iso_idx = np.array([nb_iso_map[hash] for hash in node_nb_hashes],dtype=np.int16).reshape(-1,1)
		dag_num_nodes_nb = len(set(node_nb_hashes))
	else:
		node_nb_iso_idx = np.zeros((nodes_mask_matrix.shape[0],0),dtype=np.int16)
		dag_num_nodes_nb = -1

	assert nodes_mask_matrix.shape[1] <= MASK_SIZE
	node_cc_mask = th.as_tensor(nodes_mask_matrix,dtype=th.bool)
	node_cc_mask = F.pad(node_cc_mask, (0, MASK_SIZE-node_cc_mask.shape[1]), "constant", 0)
	node_cc_long = th_mask_to_long(node_cc_mask).type(NODE_FEAT_DTYPE)
	# the order is important!
	pyg_node_feats = th.cat(
		[ 
			th.as_tensor(nodes_depth_matrix,dtype=NODE_FEAT_DTYPE),
			node_cc_long,
			th.as_tensor(node_base_formula_matrix,dtype=NODE_FEAT_DTYPE), 
			th.as_tensor(node_formulae_matrix,dtype=NODE_FEAT_DTYPE),
			th.as_tensor(node_h_count_matrix,dtype=NODE_FEAT_DTYPE),
			th.as_tensor(node_nb_iso_idx,dtype=NODE_FEAT_DTYPE)
		], dim=1
	)

	# peak mzs array
	formula_peak_mzs = []
	formula_peak_probs = []
	peaks_for_element_cache = {}
	for formula, idx in formula_to_idx.items():
		if formula == "":
			peak_mzs = np.zeros(MAX_NUM_MZS_PER_FORMULA,dtype=np.float32)
			peak_probs = np.zeros(MAX_NUM_MZS_PER_FORMULA,dtype=np.float32)
		else:
			peak_mzs, peak_probs = formula_to_peak_mzs(formula,"", isotopes = isotopes, return_probs=True, peaks_for_element_cache = peaks_for_element_cache)
			peak_mzs, peak_probs = zip(*sorted(zip(peak_mzs, peak_probs), key=lambda x: x[1], reverse=True))
			peak_mzs = np.array(peak_mzs[:MAX_NUM_MZS_PER_FORMULA],dtype=np.float32)
			peak_mzs = np.pad(peak_mzs,(0,MAX_NUM_MZS_PER_FORMULA-len(peak_mzs)),"constant",constant_values=0)
			peak_probs = np.array(peak_probs[:MAX_NUM_MZS_PER_FORMULA],dtype=np.float32)
			peak_probs = np.pad(peak_probs,(0,MAX_NUM_MZS_PER_FORMULA-len(peak_probs)),"constant",constant_values=0)
		formula_peak_mzs.append(peak_mzs)
		formula_peak_probs.append(peak_probs)

	# save as float32 for speed and lower ram usage
	formula_peak_mzs = th.as_tensor(np.stack(formula_peak_mzs,axis=0), dtype = META_DATA_DTYPE)
	formula_peak_probs = th.as_tensor(np.stack(formula_peak_probs,axis=0), dtype = META_DATA_DTYPE)

	# add edges info
	edge_diff_cc_mask, edge_diff_formula_mask, edge_diff_h_range = [], [], []

	for edge in dag_edges_matrix:
		from_idx, to_idx = edge
		from_cc_mask = nodes_mask_matrix[from_idx]
		to_cc_mask = nodes_mask_matrix[to_idx]
		diff_cc_mask =  from_cc_mask - to_cc_mask

		from_formula_mask = node_base_formula_matrix[from_idx]
		to_formula_mask = node_base_formula_matrix[to_idx]
		diff_formula_mask = from_formula_mask - to_formula_mask
		diff_formula_mask[CANONICAL_H_IDX] = 0 # we don't care Hs for this

		diff_cc_atom_ids = cc_bit_mask_to_atom_idx(diff_cc_mask)

		diff_h_floor = compute_frags.compute_cc_h_floor(
			diff_cc_atom_ids,
			mol_d["ve_arr"],
			mol_d["sbond_arr"],
			0,
			mol_d["bonds"],
			mol_d["atoms_to_bonds"],
			mol_d["bond_mask_arr"]
		)
		diff_h_cap = compute_cc_h_cap(
			diff_cc_atom_ids,
			mol_d["ve_arr"],
			mol_d["sbond_arr"],
			0
		)
		assert diff_h_floor <= diff_h_cap, (diff_h_floor,diff_h_cap)
		assert diff_h_floor >= 0, diff_h_floor
		#print(diff_h_floor,diff_h_cap)
		diff_h_range = [diff_h_floor,diff_h_cap]
		edge_diff_cc_mask.append(diff_cc_mask)
		edge_diff_formula_mask.append(diff_formula_mask)
		edge_diff_h_range.append(diff_h_range)

	assert len(edge_diff_cc_mask) == len(edge_diff_formula_mask)
	assert len(edge_diff_cc_mask) > 0, "DAG has no edges"

	edge_diff_cc_mask = th.as_tensor(np.stack(edge_diff_cc_mask,axis=0), dtype=th.bool)
	edge_diff_cc_mask = F.pad(edge_diff_cc_mask, (0, MASK_SIZE-edge_diff_cc_mask.shape[1]), "constant", 0)
	edge_diff_cc_long = th_mask_to_long(edge_diff_cc_mask).type(EDGE_FEAT_DTYPE)
	edge_diff_formula_mask = th.as_tensor(np.stack(edge_diff_formula_mask,axis=0), dtype=EDGE_FEAT_DTYPE)
	edge_diff_h_range = th.as_tensor(np.stack(edge_diff_h_range,axis=0), dtype=EDGE_FEAT_DTYPE)
	pyg_edge_feats = th.cat([edge_diff_cc_long,edge_diff_formula_mask,edge_diff_h_range],dim=1)

	# edge index need to be int64 or it will throw error where compute degree
	pyg_edge_index = th.tensor(dag_edges_matrix.T, dtype=th.int64)

	pyg_cc_g = pyg.data.Data(pyg_node_feats,pyg_edge_index,pyg_edge_feats)

	pyg_cc_g.node_feat_idxs = th.cumsum(th.tensor([0]+node_feat_shapes,dtype=th.long),0).reshape(1,-1)
	pyg_cc_g.edge_feat_idxs = th.cumsum(th.tensor([0]+edge_feat_shapes,dtype=th.long),0).reshape(1,-1)

	# convert to pyg
	frag_d = {}
	frag_d["max_depth"] = max_depth
	frag_d["reached_depth"] = reached_depth
	frag_d["h_prior"] = h_prior
	frag_d["max_h_transfer"] = max_h_transfer
	frag_d["formula_peak_mzs"] = formula_peak_mzs
	frag_d["formula_peak_probs"] = formula_peak_probs
	frag_d["idx_to_formula"] = idx_to_formula # useful for annotation
	frag_d["idx_by_h_delta"] = formula_idx_by_h_delta # formula idx for each h_delta
	frag_d["dag"] = pyg_cc_g
	
	frag_d["edges_min_depth"] = edges_min_depth
	frag_d["nodes_min_depth"] = nodes_min_depth

	# add stats here
	# we need change data type again
	# we just need to change this one place
	frag_d["dag_num_edges"] = pyg_cc_g.num_edges
	frag_d["dag_num_nodes"] = pyg_cc_g.num_nodes
	frag_d["dag_num_nodes_nb"] = dag_num_nodes_nb
	frag_d["dag_sparsity"] = 2*pyg_cc_g.num_edges/(pyg_cc_g.num_nodes *(pyg_cc_g.num_nodes - 1))
	frag_d["formula_redundancy"] = sum([v for k,v in formula_counts.items() if k != ""])/len([k for k in formula_counts.keys() if k != ""])
	frag_d["node_feature_size"] = pyg_cc_g.num_features
	frag_d["edge_feature_size"] = pyg_cc_g.num_edge_features
	frag_d["is_directed"] = pyg_cc_g.is_directed()
	frag_d["dag_num_edges_by_depth"] = { k:np.count_nonzero(edges_min_depth == k) for k in range(reached_depth+1)}
	frag_d["dag_num_nodes_by_depth"] = { k:np.count_nonzero(nodes_min_depth == k) for k in range(reached_depth+1)} 
	return frag_d

NODE_FEAT_TO_IDX = {
	"depth":0,
	"cc":1,
	"base_formula":2,
	"h_formulae_idx":3,
	"h_counts":4,
	"nb_iso_idx":5
}

EDGE_FEAT_TO_IDX = {
	"cc":0,
	"base_formula":1,
	"h_range":2,
	"complement":3
}

def get_node_feats(node_feats:th.Tensor,node_feat_idxs:th.Tensor,key:str):
	"""get node features by key used for pyg

	Args:
		node_feats (_type_): _description_
		node_feat_idxs (_type_): _description_
		key (_type_): _description_

	Returns:
		_type_: _description_
	"""

	node_feat_idx = NODE_FEAT_TO_IDX[key]
	node_feats = node_feats[:,node_feat_idxs[node_feat_idx]:node_feat_idxs[node_feat_idx+1]]
	#print(f"get_node_feats, node_feats tensor shape: {node_feats.shape}, num nodes: {len(node_feats)}, feature name: {key}" )
	return node_feats

def get_edge_feats(edge_feats:th.Tensor,edge_feat_idxs:int,key:str):
	"""get edege feats used for pyg

	Args:
		edge_feats (_type_): _description_
		edge_feat_idxs (_type_): _description_
		key (_type_): _description_

	Returns:
		_type_: _description_
	"""

	edge_feat_idx = EDGE_FEAT_TO_IDX[key]
	edge_feats = edge_feats[:,edge_feat_idxs[edge_feat_idx]:edge_feat_idxs[edge_feat_idx+1]]
	return edge_feats

def get_frag_name(mol_id:str, is_compressed:bool):
	name = f'{mol_id}.pickle'
	if is_compressed:
		name += ".bz2"
	return name

def get_frag_fp(mol_id:str, frag_dp:str, is_compressed:bool):

	fp = os.path.join(frag_dp,get_frag_name(mol_id, is_compressed))

	return fp

def save_frag_d(frag_d:dict, mol_id:int, frag_dp:str, is_compressed:bool = False):
	"""save frag_d use pickle if is_compressed save as .pbz

	Args:
		frag_d (dict): _description_
		filepath (str): _description_
	"""
	
	fp = get_frag_fp(mol_id, frag_dp, is_compressed)
	try:
		if not is_compressed:
			with open(fp, 'wb') as fileout:
				pickle.dump(frag_d, fileout, protocol=pickle.HIGHEST_PROTOCOL)
		else:
			with bz2.BZ2File(fp, 'wb') as fileout: 
				cPickle.dump(frag_d, fileout)
	except Exception as e:
		print(e,fp)
      
def load_frag_d(mol_id:str, frag_dp:str, is_compressed:bool = False):
	"""_summary_

	Args:
		filepath (str): _description_

	Returns:
		_type_: _description_
	"""
	frag_d = None

	if os.path.isfile(frag_dp) and str(frag_dp).endswith(".tar"):
		frag_filename = get_frag_name(mol_id, is_compressed)
		with tarfile.open(frag_dp, "r") as tar_read:
			for member in tar_read.getmembers():
				if member.name == frag_filename:
					f = tar_read.extractfile(member)
					content = f.read()
					frag_d = pickle.loads(bz2.decompress(content))
					break

	elif os.path.isfile(frag_dp) and str(frag_dp).endswith(".zip"):
		frag_filename = get_frag_name(mol_id, is_compressed)
		with ZipFile(frag_dp, 'r') as zip_read:
			with zip_read.open(frag_filename) as f:
				content = f.read()
				if frag_filename.endswith("bz2"):
					frag_d = pickle.loads(bz2.decompress(content))
				else:
					frag_d = pickle.loads(content)
	else:
		fp = get_frag_fp(mol_id, frag_dp, is_compressed)
		if not is_compressed:
			with open(fp, 'rb') as filein:
				frag_d = pickle.load(filein)
		else:
			with bz2.BZ2File(fp, 'rb') as filein: 
				frag_d = cPickle.load(filein)

	return frag_d

def _hash_label(label, digest_size=32):
	"""
	Adapted from https://networkx.org/documentation/stable/_modules/networkx/algorithms/graph_hashing.html
	"""
	return blake2b(label.encode("ascii"), digest_size=digest_size).hexdigest()

def wl_hash(
	elems: list,
	bond_type_idxs: list,
	node_mask: np.ndarray,
	edges: np.ndarray,
	node_to_edge_idx: np.ndarray,
	include_bond_type: bool = False,
	max_iterations: int = -1
) -> int:
	""" 
	Adapted from https://networkx.org/documentation/stable/_modules/networkx/algorithms/graph_hashing.html
	"""

	cur_hashes = []
	num_nodes = len(elems)
	for i in range(num_nodes):
		if node_mask[i]:
			cur_hashes.append(str(elems[i]))
		else:
			cur_hashes.append("")
	cur_counter = Counter(cur_hashes)
	cur_counter.pop("", None)
	graph_hash_counts = sorted(cur_counter.items(), key=lambda x: x[0])
	iterations = np.sum(node_mask)
	assert iterations <= num_nodes, (iterations, num_nodes)
	if max_iterations == -1:
		max_iterations = iterations
	else:
		assert max_iterations >= 0, max_iterations
	ct = 0
	while ct < iterations and ct < max_iterations:
		# print(cur_hashes)
		new_hashes = []
		temp_atoms = 0
		# Step 2: Update hashes with local neighborhoods
		for node_idx in range(num_nodes):
			cur_hash = cur_hashes[node_idx]
			if not node_mask[node_idx]:
				new_hashes.append(cur_hash)
				continue
			# Count num atoms in this loop
			temp_atoms += 1
			# Get local neighbors
			neighbor_labels = []
			for edge_idx in node_to_edge_idx[node_idx]:
				if edge_idx == -1:
					break
				node_idx_1, node_idx_2 = edges[edge_idx]
				if node_idx_1 == node_idx:
					targ_node_idx = node_idx_2
				else:
					targ_node_idx = node_idx_1
				assert targ_node_idx != node_idx
				if not node_mask[targ_node_idx]:
					continue
				targ_hash = cur_hashes[targ_node_idx]
				if include_bond_type:
					bondtype = bond_type_idxs[edge_idx]
					neighbor_label = f"_{bondtype}_{targ_hash}"
				else:
					neighbor_label = f"_{targ_hash}"
				neighbor_labels.append(neighbor_label)
			new_hash = cur_hash + "".join(sorted(neighbor_labels))
			new_hash = _hash_label(new_hash)
			new_hashes.append(new_hash)
		assert temp_atoms == iterations, (temp_atoms, iterations)
		new_counter = Counter(new_hashes)
		new_counter.pop("", None)
		graph_hash_counts.extend(sorted(new_counter.items(), key=lambda x: x[0]))
		cur_hashes = new_hashes
		# print(f"> {ct}")
		# print(new_graph_hash)
		# print(cur_hashes)
		ct += 1
	graph_hash = _hash_label(str(tuple(graph_hash_counts)))
	return graph_hash

def get_subgraph_hashes(
	nodes_mask_matrix: np.ndarray,
	elems: list, 
	bond_type_idxs: list, 
	edges: np.ndarray, 
	node_to_edge_idx: np.ndarray, 
	include_bond_type: bool,
	max_iterations: int):

	subgraph_hashes = []
	num_subgraphs = nodes_mask_matrix.shape[0]
	for i in range(num_subgraphs):
		subgraph_mask = nodes_mask_matrix[i]
		subgraph_hash = wl_hash(
			elems, 
			bond_type_idxs, 
			subgraph_mask, 
			edges, 
			node_to_edge_idx,
			include_bond_type=include_bond_type,
			max_iterations=max_iterations)
		subgraph_hashes.append(subgraph_hash)
	return subgraph_hashes

def timed_get_dags(mol, 
			mol_id, 
			max_depth,
			h_prior,
			max_h_transfer, 
			max_time,
			isotopes: bool,
			nb_isomorphic: bool,
			max_iterations: int,
			output_dir: str, 
			use_cached = True, 
			compressed = False,
			save_dag = True):

		if save_dag:
			output_file = get_frag_fp(mol_id, output_dir, compressed)

		need_compute = not use_cached or not os.path.exists(output_file)
		if not need_compute:
			assert save_dag
			try:
				dag_d = load_frag_d(mol_id, output_dir, compressed)
			except Exception as e:
				print(e, "cache is not usable")
				need_compute = True
		if need_compute:
			dag_d = {}
			try:
				mol_d = extract_mol_info(mol)
				# this maybe dangers in multi processing because of scopes
				dag_d = compute_dags(
					mol_d,
					max_depth,
					h_prior,
					max_h_transfer, 
					max_time, 
					isotopes,
					nb_isomorphic,
					max_iterations
				)
			except KeyboardInterrupt as e:
				# let these through
				raise e
			except Exception as e:
				# don't retry, theres a bug
				if type(mol) is not str:
					mol = Chem.MolToSmiles(mol)
				print(f">> Non-timeout error, aborting: {type(e)} {repr(e)} Input {mol}",file=sys.stderr)
				print("> Traceback",file=sys.stderr)
				traceback.print_exc(file=sys.stderr)
				dag_d = {}
			else:
				if save_dag:
					save_frag_d(dag_d, mol_id, output_dir, compressed)
					del dag_d["dag"]
		return dag_d