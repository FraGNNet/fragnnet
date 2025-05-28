import sys
import argparse
import logging
import json
from pathlib import Path
import numpy as np
import copy
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from pprint import pformat
from rdkit import RDLogger

# Custom import
import fragnnet.iceberg.common as common
import fragnnet.iceberg.fragmentation as fragmentation
import fragnnet.utils.frag_utils as frag_utils
from fragnnet.utils.proc_utils import filter_spec_mol, merge_spec_df
from fragnnet.utils.misc_utils import booltype
from fragnnet.utils.data_utils import seq_apply, par_apply


def greedy_prune(
	fe: fragmentation.FragmentEngine, included_nodes: list, tree_nodes: list
):

	"""greedy_prune.

	Multiple paths can be used to access each of the nodes in the graph. This
	can double or triple the total number of nodes in the tree. We must prune
	these in order to get better nodes.

	We use a greedy set cover at each depth, attempting to find useful parents

	Args:
		fe (FragmentEngine): Fragment engine
		included_nodes: List of included nodes in the tree to be pruned
		tree_nodes: Nodes that must be included in the output arboresence

	Return:
		List of new nodes to include
	"""
	# Get a queue for each node's priority based upon how many children it has
	# included in the tree nodes
	tree_set = set(tree_nodes)

	if len(tree_set) == 0:
		return []

	# Sort
	included_nodes = sorted(included_nodes)

	# Add whether or not the node is a member of existing nodes
	node_priorities = np.zeros(len(included_nodes))
	# Hash to index
	hash_to_pos = dict(zip(included_nodes, np.arange(len(included_nodes))))

	output_mask = np.array([j in tree_set for j in included_nodes])
	node_priorities += output_mask
	incoming_edges, outgoing_edges = fe.export_edges_dict(included_nodes)

	# Get adj dicts
	entries = [fe.frag_to_entry[i] for i in included_nodes]

	# Tie breaking
	node_scores = [i["score"] for i in entries]
	max_broken = np.array([i["max_broken"] for i in entries])
	tree_depths = np.array([i["tree_depth"] for i in entries])
	highest_depth = max(tree_depths)

	# Loop over tree depth from leaves up
	for depth in range(highest_depth, 0, -1):

		cur_layer_inds = tree_depths == depth

		# Define all parents of current layer (do this in loop below)
		cover_options = np.zeros(len(cur_layer_inds)).astype(bool)

		# Mask to cover --> all nodes in the current row requiring covering
		to_cover = np.logical_and(output_mask, cur_layer_inds)

		# Update parent priorities based upon what's in to_cover
		for node in np.where(to_cover)[0]:
			hash_key = included_nodes[node]

			# Get all parents
			incoming_parents = incoming_edges[hash_key]
			for p in incoming_parents:
				p_pos = hash_to_pos[p]
				node_priorities[p_pos] += 1
				cover_options[p_pos] = True

		num_to_cover = np.sum(to_cover)
		while num_to_cover > 0:

			# Get all cover options
			max_score = np.max(node_priorities[cover_options])
			best_cands_mask = node_priorities == max_score
			best_cands_mask[~cover_options] = False
			best_cands = np.where(best_cands_mask)[0]

			# Break ties by choosing those with min score
			best_cands = sorted(best_cands, key=lambda x: node_scores[x])
			new_output = best_cands[0]

			# Add this node to the return set and make sure it comes up for nex
			# time node priorities are computed
			output_mask[new_output] = True

			# Find all nodes that this covers and remove them from to cover
			# Decrement scores for the parents above based upon added value
			for child in outgoing_edges[included_nodes[new_output]]:
				new_covered = hash_to_pos[child]
				to_cover[new_covered] = False

				# Decrement scores for parents not in the output already
				for p in incoming_edges[child]:
					p_pos = hash_to_pos[p]
					if not output_mask[p_pos]:
						node_priorities[hash_to_pos[p]] -= 1

			# Remove cover options
			cover_options[new_output] = False
			num_to_cover = np.sum(to_cover)
	output_hashes = np.array(included_nodes)[output_mask].tolist()
	return output_hashes


def get_formula_dict(
	spec_name: str,
	spec: np.ndarray,
	form: str,
	mass_diff_type: str,
	mass_diff_thresh: float,
	inten_thresh: float,
	adduct_type: str,
	max_formulae: int,
	use_all: bool,
	smiles: str,
	use_magma: bool,
) -> dict:
	"""get_output_dict.

	This function attemps to take an array of mass intensity values and assign
	formula subsets to subpeaks

	Args:
		spec_name (str): spec_name
		spec (np.ndarray): spec
		form (str): form
		mass_diff_type (str): mass_diff_type
		mass_diff_thresh (float): mass_diff_thresh
		inten_thresh (float): inten_thresh
		adduct_type (str): adduct_type
		max_formulae (int): max_formulae
		use_all:
		smiles (str): smiles
		use_magma (bool): use_magma

	Returns:
		dict:
	"""
	# This is the case for some erroneous MS2 files for which proc_spec_file return None
	# All the MS2 subpeaks in these erroneous MS2 files has mz larger than parentmass
	if spec is None:
		output_dict = {
			"cand_form": form,
			"spec_name": spec_name,
			"cand_ion": adduct_type,
			"output_tbl": None,
		}
		return output_dict

	# Filter down
	spec = common.max_inten_spec(spec, max_formulae, inten_thresh=inten_thresh)
	spec_masses, spec_intens = spec[:, 0], spec[:, 1]
	adduct_masses = common.ion2mass[adduct_type]

	if use_all:
		output_tbl = {
			"mz": list(spec_masses),
			"ms2_inten": list(spec_intens),
			"rel_inten": list(spec_intens),  # rel_inten),
			"mono_mass": list(spec_masses),
			"formula_mass_no_adduct": list(spec_masses - adduct_masses),
			"mass_diff": [0] * len(spec_masses),
			"formula": [""] * len(spec_masses),
			"ions": [adduct_masses] * len(spec_masses),
		}

		if len(spec_intens) == 0:
			output_tbl = None
		output_dict = {
			"cand_form": form,
			"spec_name": spec_name,
			"cand_ion": adduct_type,
			"output_tbl": output_tbl,
		}
		return output_dict
	elif use_magma:
		fe = fragmentation.FragmentEngine(
			mol_str=smiles,
			# max_tree_depth=3,
		)
		try:
			fe.generate_fragments()
		except:
			print(f"Error with generating fragments for spec {smiles}")
			return {
				"cand_form": form,
				"spec_name": spec_name,
				"cand_ion": adduct_type,
				"output_tbl": None,
			}
		cross_prod, masses = fe.get_frag_forms()
	else:
		# exhaustive enumeration
		cross_prod, masses = common.get_all_subsets(form)
	
	masses_with_adduct = masses + adduct_masses
	adduct_types = np.array([adduct_type] * len(masses_with_adduct))
	mass_diffs = np.abs(spec_masses[:, None] - masses_with_adduct[None, :])

	formula_inds = mass_diffs.argmin(-1)
	min_mass_diff = mass_diffs[np.arange(len(mass_diffs)), formula_inds]

	if mass_diff_type == "ppm":
		mass_divisior = np.copy(spec_masses)
		mass_divisior[mass_divisior <= 200] = 200
		min_mass_diff = (min_mass_diff / mass_divisior) * 1e6
	elif mass_diff_type == "abs":
		pass

	# Filter by abs mass diff
	valid_mask = min_mass_diff < mass_diff_thresh
	spec_masses = spec_masses[valid_mask]
	spec_intens = spec_intens[valid_mask]
	min_mass_diff = min_mass_diff[valid_mask]
	formula_inds = formula_inds[valid_mask]

	formulas = np.array([common.vec_to_formula(j) for j in cross_prod[formula_inds]])
	formula_masses = masses_with_adduct[formula_inds]
	formula_mass_no_adduct = masses[formula_inds]
	adduct_types = adduct_types[formula_inds]

	# Build mask for uniqueness on formula and adduct
	# note that adduct are all the same for one subformula assignment
	# hence we only need to consider the uniqueness of the formula
	formula_idx_dict = {}
	uniq_mask = []
	for idx in range(len(formulas)):
		formula = formulas[idx]
		if formula not in formula_idx_dict:
			uniq_mask.append(True)
			formula_idx_dict[formula] = idx
		else:
			merge_idx = formula_idx_dict[formula]
			uniq_mask.append(False)
			spec_intens[merge_idx] = max(spec_intens[idx], spec_intens[merge_idx])

	spec_masses = spec_masses[uniq_mask]
	spec_intens = spec_intens[uniq_mask]
	min_mass_diff = min_mass_diff[uniq_mask]
	formula_masses = formula_masses[uniq_mask]
	formula_mass_no_adduct = formula_mass_no_adduct[uniq_mask]
	formulas = formulas[uniq_mask]
	adduct_types = adduct_types[uniq_mask]

	# Renormalize
	# to calculate explained intensity, let's preserve the original normalized intensity
	if spec_intens.size == 0:
		output_dict = {
			"cand_form": form,
			"spec_name": spec_name,
			"cand_ion": adduct_type,
			"output_tbl": None,
		}
	else:
		# if mass_diff_type = ppm, then mass_diff is calculated using ppm
		# if mass_diff_type = abs, then mass_diff is in the unit of Dalton

		# Use rel inten, but assume it's already been processed
		# rel_inten = spec_intens / (spec_intens.max())
		# rel_inten = np.sqrt(rel_inten)
		output_tbl = {
			"mz": list(spec_masses),
			"ms2_inten": list(spec_intens),
			"rel_inten": list(spec_intens),  # rel_inten),
			"mono_mass": list(formula_masses),
			"formula_mass_no_adduct": list(formula_mass_no_adduct),
			"mass_diff": list(min_mass_diff),
			"formula": list(formulas),
			"ions": [adduct for adduct in list(adduct_types)],
		}

		output_dict = {
			"cand_form": form,
			"spec_name": spec_name,
			"cand_ion": adduct_type,
			"output_tbl": output_tbl,
		}
	return output_dict


def magma_augmentation(
	spec_entry: dict,
	magma_dir: Path,
	max_peaks: int,
	ppm_diff: float,
	mass_diff_thresh: float,
	mass_diff_type: str,
	inten_thresh: float,
	max_formulae: int,
	parallel: bool,
):
	"""magma_augmentation.

	Args:
		spec_file (Path): spec_file
		output_dir (Path): output_dir
		spec_to_smiles (dict): spec_to_smiles
		spec_to_adduct (dict): Spec to adduct
		max_peaks (int): max_peaks
		ppm_diff (float): Max diff ppm
		parallel (bool)
	"""

	spectra_name = str(spec_entry["group_id"]) # assumes merged
	tsv_dir = magma_dir / "magma_tsv"
	tree_dir = magma_dir / "magma_tree"
	assert tsv_dir.exists(), f"Missing tsv dir: {tsv_dir}"
	assert tree_dir.exists(), f"Missing tree dir: {tree_dir}"
	tsv_filename = tsv_dir / f"{spectra_name}.tsv"
	tree_filename = tree_dir / f"{spectra_name}.json"
	formula_dir = magma_dir / "magma_formula"
	assert formula_dir.exists(), f"Missing formula dir: {formula_dir}"
	formula_file = formula_dir / f"{spectra_name}.json"

	spectra_smiles = spec_entry["smiles"]
	spectra_adduct = spec_entry["prec_type"]
	spectra_parentmass = spec_entry["prec_mz"]
	spectra_formula = spec_entry["formula"]
	# formula is NOT needed
	meta = {
		"smiles": spectra_smiles,
		"adduct": spectra_adduct,
		"parentmass": spectra_parentmass,
	}
	spectras = [(None,np.array(spec_entry["peaks"]))]

	# Step 1 - Generate fragmentations inside fragmentation engine
	fe = fragmentation.FragmentEngine(mol_str=spectra_smiles, **fragmentation.FRAGMENT_ENGINE_PARAMS)

	# Outside try except loop
	if not parallel:
		fe.generate_fragments()
	else:
		RDLogger.DisableLog("rdApp.*")
		try:
			fe.generate_fragments()
		except:
			print(f"Error with generating fragments for spec {spectra_name}")
			return False

	# Step 2: Process spec and get comparison points
	# Read in file and filter it down
	try:
		spectra = common.process_spec_file(meta, spectras)
	except Exception as e:
		spectra = None
	
	if spectra is None:
		print(f"Error with reading file for spec {spectra_name}")
		return False

	spectra_f = common.max_inten_spec(
		spectra, max_num_inten=max_peaks, inten_thresh=inten_thresh
	)
	s_m, s_i = spectra_f[:, 0], spectra_f[:, 1]

	# Correct for s_m by subtracting it
	adjusted_m = s_m - common.ion2mass[spectra_adduct]

	# Step 3: Make all assignments
	frag_hashes, frag_inds, shift_inds, masses, scores = fe.get_frag_masses()
	# print(f"Number of raw fragments = {len(scores)}")

	# Argsort by bond breaking scores
	# Lower bond scores are better
	new_order = np.argsort(scores)
	frag_hashes, frag_inds, shift_inds, masses, scores = (
		frag_hashes[new_order],
		frag_inds[new_order],
		shift_inds[new_order],
		masses[new_order],
		scores[new_order],
	)
	ppm_diffs = (
		np.abs(masses[None, :] - adjusted_m[:, None]) / adjusted_m[:, None] * 1e6
	)

	# Need to catch _all_ equivalent fragments
	# How do I remove the symmetry problem at each step and avoid branching
	# trees for the same examples??
	min_ppms = ppm_diffs.min(-1)
	is_min = min_ppms[:, None] == ppm_diffs
	peak_mask = min_ppms < ppm_diff
	if peak_mask.sum() == 0:
		# just choose the one closest to the precursor (to prevent empty tree)
		adjusted_pm = spectra_parentmass - common.ion2mass[spectra_adduct]
		peak_idx = np.argmin(np.abs(adjusted_m - adjusted_pm) / adjusted_pm * 1e6)
		peak_mask = np.arange(peak_mask.shape[0]) == peak_idx
		mass_idx = np.argmin(np.abs(masses - adjusted_pm) / masses * 1e6)
		is_min = np.zeros_like(is_min)
		is_min[peak_idx][mass_idx] = True
		min_ppms = np.zeros_like(min_ppms)
		min_ppms[peak_idx] = np.abs(adjusted_m[peak_idx] - adjusted_pm) / adjusted_m[peak_idx] * 1e6
	assert peak_mask.sum() >= 1, peak_mask.sum()

	# Step 4: Make exports
	# Now collect all inds and results
	# Also record a map from hash, hshift to the peak_info
	tsv_export_list = []
	hash_to_peaks = defaultdict(lambda: [])
	max_labeled_inten = 0
	for ind, was_assigned in enumerate(peak_mask):
		new_entry = {
			"mz_observed": s_m[ind],
			"mz_corrected": adjusted_m[ind],
			"inten": s_i[ind],
			"ppm_diff": "",
			"frag_inds": "",
			"frag_mass": "",
			"frag_h_shift": "",
			"frag_base_form": "",
			"frag_hashes": "",
		}
		if was_assigned:
			# Find all the fragments that have min ppm tolerance
			matched_peaks = is_min[ind]
			min_inds = np.argwhere(matched_peaks).flatten()

			# Get min score for this assignment
			min_score = np.min(scores[min_inds])

			# Filter even further down to inds that have min score and min ppm
			min_score_ppm = min_inds[
				np.argwhere(scores[min_inds] == min_score).flatten()
			]

			frag_inds_temp = [frag_inds[temp_ind] for temp_ind in min_score_ppm]
			frag_masses_temp = [masses[temp_ind] for temp_ind in min_score_ppm]
			frag_hashes_temp = [frag_hashes[temp_ind] for temp_ind in min_score_ppm]
			shift_inds_temp = [shift_inds[temp_ind] for temp_ind in min_score_ppm]
			frag_entries_temp = [
				fe.frag_to_entry[frag_hash] for frag_hash in frag_hashes_temp
			]
			frag_forms_temp = [frag_entry["form"] for frag_entry in frag_entries_temp]

			str_join = lambda x: ",".join([str(xx) for xx in x])
			new_entry["ppm_diff"] = min_ppms[ind]
			new_entry["frag_inds"] = str_join(frag_inds_temp)
			new_entry["frag_hashes"] = ",".join(frag_hashes_temp)
			new_entry["frag_mass"] = str_join(frag_masses_temp)
			new_entry["frag_h_shift"] = str_join(shift_inds_temp)
			new_entry["frag_base_form"] = ",".join(frag_forms_temp)
			peak_info_base = {
				"mz_observed": s_m[ind],
				"mz_corrected": adjusted_m[ind],
				"inten": s_i[ind],
				"ppm_diff": min_ppms[0],
				"frag_mass": frag_masses_temp[0],
			}
			max_labeled_inten = max(max_labeled_inten, s_i[ind])
			for h, s, f in zip(frag_hashes_temp, shift_inds_temp, frag_forms_temp):
				peak_info_ex = copy.deepcopy(peak_info_base)
				peak_info_ex["frag_hash"] = h
				peak_info_ex["frag_h_shift"] = s
				peak_info_ex["frag_base_form"] = f
				hash_to_peaks[h].append(peak_info_ex)

		tsv_export_list.append(new_entry)
	# print(f"Number of peaks (with or without assignment): {len(tsv_export_list)}")

	df = pd.DataFrame(tsv_export_list)
	df.sort_values(by="mz_observed", inplace=True)
	df.to_csv(tsv_filename, sep="\t", index=None)

	# Build trees
	tree_nodes = [
		j for i in tsv_export_list for j in i["frag_hashes"].split(",") if len(j) > 0
	]
	tree_nodes = list(set(tree_nodes))
	# print(f"Number of nodes (with assignment to at least one peak): {len(tree_nodes)}")

	# Now do a breadth first search back on the tree via its parents
	explore_queue = copy.deepcopy(tree_nodes)
	explored = set()
	while len(explore_queue) > 0:
		new_explore = explore_queue.pop()
		explored.add(new_explore)

		# Get parents for current node and add all of them
		entry = fe.frag_to_entry[new_explore]
		parent_hashes = entry["parent_hashes"]

		# Note: Parents are singular, but each parent has multiple potential
		explore_queue.extend(set([i for i in parent_hashes if i not in explored]))

	included_nodes = list(explored)
	pruned_nodes = greedy_prune(fe, included_nodes, tree_nodes)

	# Export and use to construct tree viz or others
	out_frags = {}
	pruned_node_set = set(pruned_nodes)
	node_to_pulled = defaultdict(lambda: set())
	node_pulled_sib = defaultdict(lambda: set())
	node_to_parents = defaultdict(lambda: set())

	for frag in pruned_nodes:
		entry = fe.frag_to_entry[frag]

		peak_intens = hash_to_peaks[frag]
		inten_vec = np.zeros(fe.shift_bucket_inds.shape[0])
		for i in peak_intens:
			# Do not renormalize!
			inten_vec[i["frag_h_shift"]] = i["inten"]  # / max_labeled_inten

		new_entry = {
			"frag_hash": frag,
			"frag": entry["frag"],
			"is_observed": frag in tree_nodes,
			"atoms_pulled": [],
			"parents": [],
			"base_mass": entry["base_mass"],
			"intens": inten_vec.tolist(),
			"id": entry["id"],
			"sib": False,
			"max_broken": entry["max_broken"],
			"tree_depth": entry["tree_depth"],
			"max_remove_hs": entry["max_remove_hs"],
			"max_add_hs": entry["max_add_hs"],
		}
		out_frags[frag] = new_entry
		for parent, pulled_atom, sibling_hash in zip(
			entry["parent_hashes"], entry["parent_ind_removed"], entry["sibling_hashes"]
		):
			if parent in pruned_node_set:
				node_to_parents[frag].add(parent)
				node_to_pulled[parent].add(pulled_atom)
				node_pulled_sib[(parent, pulled_atom)] = sibling_hash

	# Build up a list of (node, pulled) tuples and have a dict of pulled to
	# sibling
	out_frags_keys = list(out_frags.keys())
	for k in out_frags_keys:
		out_frags[k]["atoms_pulled"] = node_to_pulled[k]
		out_frags[k]["parents"] = node_to_parents[k]

		# Add in siblings for all pulled nodes!
		for a in out_frags[k]["atoms_pulled"]:
			sib_entries = node_pulled_sib[(k, a)]
			for sib_node in sib_entries:
				fe_entry = fe.frag_to_entry[sib_node]

				cur_entry = out_frags.get(sib_node)
				# Make a new sib entry
				if cur_entry is None:
					inten_vec = np.zeros(fe.shift_bucket_inds.shape[0])
					new_entry = {
						"frag_hash": sib_node,
						"frag": fe_entry["frag"],
						"is_observed": False,
						"atoms_pulled": [],
						"parents": [k],
						"base_mass": fe_entry["base_mass"],
						"intens": inten_vec.tolist(),
						"id": fe_entry["id"],
						"sib": True,
						"max_broken": fe_entry["max_broken"],
						"tree_depth": fe_entry["tree_depth"],
						"max_remove_hs": fe_entry["max_remove_hs"],
						"max_add_hs": fe_entry["max_add_hs"],
					}
					out_frags[sib_node] = new_entry

				# If we already have a non sib entry, continue
				else:
					if k not in cur_entry["parents"]:
						cur_entry["parents"].append(k)

		out_frags[k]["atoms_pulled"] = list(out_frags[k]["atoms_pulled"])
		out_frags[k]["parents"] = list(out_frags[k]["parents"])

	export_tree = {
		"root_inchi": fe.inchi,
		"frags": out_frags,
	}
	# Export files when needed
	if len(export_tree["frags"]) > 0:
		with open(tree_filename, "w") as f:
			json.dump(export_tree, f, indent=2)

	# get formulae
	formula_dict = get_formula_dict(
		spec_name=spectra_name,
		spec=spectra,
		form=spectra_formula,
		mass_diff_type=mass_diff_type,
		mass_diff_thresh=mass_diff_thresh,
		inten_thresh=inten_thresh,
		adduct_type=spectra_adduct,
		max_formulae=max_formulae,
		use_all=True,
		smiles=spectra_smiles,
		use_magma=False
	)
	with open(formula_file, "w") as f:
		json.dump(formula_dict, f, indent=4)

	return True


def run_magma_augmentation(
	proc_dp: str,
	magma_dp: str,
	max_peaks: int,
	parallel: bool,
	ppm_diff: int,
	allowed_elements: list[str],
	dsets: list[str],
	num_entries: int,
	group_ids: list[int],
	mass_diff_thresh: float,
	mass_diff_type: str,
	inten_thresh: float,
	max_formulae: int
):
	"""run_magma_augmentation.

	Runs magma augmentation

	Args:
		spectra_dir (str): spectra_dir
		output_dir (str): output_dir
		spec_labels (str): spec_labels
		max_peaks (int): max_peaks
		ppm_diff (int): PPM diff threshold
	"""

	magma_dir = Path(magma_dp)
	magma_dir.mkdir(exist_ok=True)
	tsv_dir = magma_dir / "magma_tsv"
	tree_dir = magma_dir / "magma_tree"
	tsv_dir.mkdir(exist_ok=True)
	tree_dir.mkdir(exist_ok=True)
	formula_dir = magma_dir / "magma_formula"
	formula_dir.mkdir(exist_ok=True)

	proc_dir = Path(proc_dp)
	mol_df_file = proc_dir / "mol_df.pkl"
	assert mol_df_file.exists(), f"Missing mol_df file: {mol_df_file}"
	mol_df = pd.read_pickle(mol_df_file)
	spec_df_file = proc_dir / "spec_df.pkl"
	assert spec_df_file.exists(), f"Missing spec_df file: {spec_df_file}"
	spec_df = pd.read_pickle(spec_df_file)

	# filter and merge
	spec_df, mol_df = filter_spec_mol(
		spec_df,
		mol_df,
		elements=allowed_elements,
		dsets=dsets,
		prec_types=["[M+H]+"],
		num_entries=num_entries
	)
	m_spec_df = merge_spec_df(spec_df,renormalize=False,sum_ints=False)
	if len(group_ids) > 0:
		m_spec_df = m_spec_df[m_spec_df["group_id"].isin(group_ids)].reset_index(drop=True)

	# get smiles, peaks, and metadata
	both_df = m_spec_df[["group_id","mol_id","peaks","prec_type","prec_mz"]].merge(mol_df[["mol_id","smiles","formula"]],on="mol_id",how="inner")
	del spec_df, mol_df, m_spec_df

	spec_entries = both_df[["group_id","smiles","peaks","prec_type","prec_mz","formula"]].to_dict("records")
	del both_df

	# Run this over all files
	partial_aug_safe = lambda spec_entry: magma_augmentation(
		spec_entry=spec_entry,
		magma_dir=magma_dir,
		max_peaks=max_peaks,
		ppm_diff=ppm_diff,
		mass_diff_thresh=mass_diff_thresh,
		mass_diff_type=mass_diff_type,
		inten_thresh=inten_thresh,
		max_formulae=max_formulae,
		parallel=parallel,
	)
	spec_entry_iter = tqdm(
		spec_entries,
		desc=pformat(partial_aug_safe),
		total=len(spec_entries)
	)
	if not parallel:
		results = seq_apply(spec_entry_iter,partial_aug_safe)
	else:
		results = par_apply(spec_entry_iter,partial_aug_safe)
	# with open(magma_dir / "results.json","w") as f:
	# 	json.dump(results,f,indent=2)

def get_args():
	"""get args"""
	parser = argparse.ArgumentParser()
	parser.add_argument("--num_entries", type=int, default=-1)
	parser.add_argument("--magma_dp", type=str, default="data/magma/gen_2")
	parser.add_argument("--proc_dp", type=str, default="data/proc/nist")
	parser.add_argument("--allowed_elements", type=str, nargs="+", default=frag_utils.ELEMENTS)
	parser.add_argument("--dsets", type=str, nargs="+", default=["nist"])
	parser.add_argument("--group_ids", type=int, nargs="+", default=[])
	# iceberg magma params
	parser.add_argument(
		"--max-peaks",
		default=50,
		help="Maximum number of peaks",
		type=int,
	)
	parser.add_argument(
		"--ppm-diff",
		default=20,
		help="PPM threshold difference",
		type=int,
	)
	parser.add_argument("--parallel", type=booltype, default=True)
	# iceberg formula params
	parser.add_argument(
		"--mass-diff-type",
		default="ppm",
		type=str,
		help="Type of mass difference - absolute differece (abs) or relative difference (ppm).",
	)
	parser.add_argument(
		"--mass-diff-thresh",
		default=10,
		type=float,
		help="Threshold of mass difference.",
	)
	parser.add_argument(
		"--inten-thresh",
		default=0.001,
		type=float,
		help="Threshold of MS2 subpeak intensity (normalized to 1).",
	)
	parser.add_argument(
		"--max-formulae",
		default=50,
		type=int,
		help="Max number of peaks to keep",
	)
	args = parser.parse_args()
	return args


if __name__ == "__main__":
	# Define basic logger
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s %(levelname)s: %(message)s",
		handlers=[
			logging.StreamHandler(sys.stdout),
		],
	)
	RDLogger.DisableLog("rdApp.*")
	args = get_args()
	kwargs = args.__dict__
	run_magma_augmentation(**kwargs)



