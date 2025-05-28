
import os
import argparse
import numpy as np

from tqdm import tqdm
import pandas as pd
from fragnnet.utils.ms2c_utils import MolCandidateDB
from fragnnet.utils.misc_utils import booltype, PPM
import fragnnet.utils.formula_utils as formula_utils
import fragnnet.utils.frag_utils as frag_utils
import fragnnet.utils.data_utils as data_utils
from fragnnet.utils.misc_utils import booltype
from fragnnet.frag.compute_frags import (
	MAX_NUM_NODES
)

from rdkit import RDLogger
from rdkit import Chem

def get_args():
	parser = argparse.ArgumentParser()
	
	# frag candidates configs
	parser.add_argument("--proc_dp", type=str, default="./data/proc/nist20mona23")
	parser.add_argument("--split_dp", type=str, default="./data/split/nist20_d4_wr00_inchikey")
	parser.add_argument("--db_file", type=str, default= "./data/ms2c/pubchem/pubchem.sqlite")
	parser.add_argument("--tolerance", type=float, default= 10, help="Default 10ppm")
	parser.add_argument("--use_ppm", type=booltype, default= True)
	parser.add_argument("--max_candidates", type=int, default= 50, help="max canidates include target itself")
	parser.add_argument("--subset_size", type=int, default=-1)
	parser.add_argument("--fp_type", type=str, default='MORGAN-R2', help="")
	parser.add_argument("--split_type", type=str,default="test", help="")
	parser.add_argument("--candidate_dp",type=str,default='./data/ms2c/pubchem/candidates')
	
	args = parser.parse_args()
	return args

def filter_candidates(db_results, target_smiles, target_mol_id, top_k = None, morgen_radius = 3):

	chem, rd_fpgen, rd_ds, rd_moldesc, rd_logger = data_utils.rdkit_import("rdkit.Chem",
																			   "rdkit.Chem.rdFingerprintGenerator", 
																			   "rdkit.DataStructs", 
																			   "rdkit.Chem.rdMolDescriptors",
																			   "rdkit.RDLogger")
	
	rd_logger.DisableLog('rdApp.*') 

	mfpgen = rd_fpgen.GetMorganGenerator(radius=morgen_radius)
	t_mol = chem.MolFromSmiles(target_smiles)
	t_fp = mfpgen.GetFingerprint(t_mol)

	# make sure target in there
	db_results.append((-1, data_utils.mol_to_inchikey(t_mol),
					target_smiles, 
					data_utils.mol_to_formula(t_mol),
					data_utils.mol_to_mol_weight(t_mol,exact=True)))
	 
	allowed_elements = set(frag_utils.ELEMENTS)
 
	case_cdf = pd.DataFrame(db_results,columns=['mol_id','inchikey','smiles','formula','mw'])
	case_cdf = case_cdf[['mol_id','smiles','formula','mw']]
 
	# drop not single mol rows
	case_cdf = case_cdf[~(case_cdf['smiles'].str.contains('.', regex=False))]
  
	case_cdf['mol'] = case_cdf['smiles'].apply(lambda x: data_utils.mol_from_smiles(x))
	# drop duplicate no mol rows
	case_cdf = case_cdf.dropna()
	# drop duplicate by canonical smiles and always keep self
	case_cdf['smiles'] = case_cdf['mol'].apply(lambda x:data_utils.mol_to_smiles(x))
	case_cdf = case_cdf.drop_duplicates(subset="smiles", keep="last")
 
	# drop duplicate by inchikey_s smiles and always keep self
	case_cdf['inchikey_s'] = case_cdf['mol'].apply(lambda x:data_utils.mol_to_inchikey_s(x))
	# drop duplicate by inchikey and always keep self
	case_cdf = case_cdf.drop_duplicates(subset="inchikey_s", keep="last")
 
	# filter by heavy atom account
	case_cdf['num_heavy_atoms'] = case_cdf['mol'].apply(lambda x: rd_moldesc.CalcNumHeavyAtoms(x))
	case_cdf = case_cdf[case_cdf['num_heavy_atoms'] <= MAX_NUM_NODES]
	# filter by radicals
	case_cdf['num_radicals'] = case_cdf['mol'].apply(lambda x: sum([atom.GetNumRadicalElectrons() for atom in x.GetAtoms()]))
	case_cdf = case_cdf[case_cdf['num_radicals'] == 0]
 
	# filter by out_set_elements
	case_cdf['num_out_set_elements'] = case_cdf['formula'].apply(lambda x: len(set(list(formula_utils.parse_formula(x).keys())) - allowed_elements))
	case_cdf = case_cdf[case_cdf['num_out_set_elements'] == 0]
 
	# filter by charges
	case_cdf['charge'] = case_cdf["mol"].apply(lambda x: data_utils.mol_to_charge(x))
	case_cdf = case_cdf[case_cdf['charge'] == 0]

	# filter by tanimoto and sort
	case_cdf["tanimoto"] = case_cdf['mol'].apply(lambda x: rd_ds.TanimotoSimilarity(t_fp, mfpgen.GetFingerprint(x)))
	case_cdf = case_cdf[case_cdf['tanimoto'] >= 0.0]
	case_cdf = case_cdf.sort_values('tanimoto', ascending = False)
	
	if top_k is not None:
		case_cdf = case_cdf[:top_k]
 
	return target_mol_id, target_smiles, case_cdf

def fetch_db(db_file, mw, mass_tol ):
	with MolCandidateDB(db_file) as db:
		db_results = db.get_compounds_by_exact_mass_range(mw -mass_tol, mw + mass_tol)
	return db_results

if __name__ == "__main__":

	RDLogger.DisableLog('rdApp.*') 

	args = get_args()
	split_configs_d = {
		"test":"test_ids.csv",
		"val":"val_ids.csv",
		"train":"train_ids.csv"
	}

	task_configs = []
	#for split in args.included_splits:
	if args.split_type not in ["test", "val", "train"]:
		print("Error, unknown split name")
		exit(0)

	# disable rdkit warning
	if args.fp_type == 'MORGAN-R2':
		morgen_radius = 2
	elif args.fp_type == 'MORGAN-R3':
		morgen_radius = 3
	else:
		print("Not Supported Fingerprint")

	split_basename = os.path.basename(os.path.normpath(args.split_dp))
	split_type = args.split_type
	split_csv_filename = split_configs_d[split_type]
	tol_type = f"{args.tolerance}Da" if not args.use_ppm else f"{args.tolerance}ppm" 
	candidate_name = f'pubchem_{split_basename}_{split_type}_{tol_type}_{args.fp_type}_{args.max_candidates}'
	if args.subset_size > 0:
		candidate_name += f'_s{args.subset_size}'
	#candidate_dp = os.path.join(args.candidate_dp, split_basename)
	os.makedirs(args.candidate_dp, exist_ok= True)
	args_fp = os.path.join(args.candidate_dp, f'{candidate_name}.yaml')

	candidates_fp = os.path.join(args.candidate_dp,f'{candidate_name}.pkl.gz')

	mol_df_fp = os.path.join(args.proc_dp,"mol_df.pkl")
	print(f"> Read mol df from {mol_df_fp}")
	mol_df = pd.read_pickle(mol_df_fp)

	id_fp = os.path.join(args.split_dp, split_csv_filename)
	print(f"> Processing {split_type} split, read ids from {id_fp}")
	#val_sp_fp = os.path.join(args.split_dp,"val_ids.csv")
	sp_mol_ids = pd.read_csv(id_fp)['mol_id'].drop_duplicates().to_frame()
	sp_mol_df = sp_mol_ids.merge(mol_df, how="left", on="mol_id")

	db_file = args.db_file
	db_query_input = []
	filter_candidates_input_l = []
	if args.subset_size > 0:
		sp_mol_df = sp_mol_df[:args.subset_size]
  
	for _, row in tqdm(sp_mol_df.iterrows(), total = len(sp_mol_df), desc = "Prepare pubchem database input"):
		# we can not frag them, just ignore	
		if Chem.rdMolDescriptors.CalcNumHeavyAtoms(row['mol']) > MAX_NUM_NODES:
			continue

		mw = row['exact_mw']
		if args.use_ppm:
			mass_tol = mw * args.tolerance * PPM
		else:
			mass_tol = args.tolerance
		db_query_input.append([db_file,mw,mass_tol])
		#db_results = db.get_compounds_by_exact_mass_range(mw -mass_tol, mw + mass_tol)
		filter_candidates_input_l.append([None, row['smiles'],row['mol_id'], args.max_candidates, morgen_radius])

	db_results = data_utils.par_apply(iter(db_query_input),fetch_db, True, return_as_generator=True)	
	for idx, res in enumerate(tqdm(db_results, total =  len(db_query_input), desc = "Fetch pubchem database ")):
		filter_candidates_input_l[idx][0] = res
	
	candidate_data_df_l = []
	canidate_results = data_utils.par_apply(iter(filter_candidates_input_l),filter_candidates, True, return_as_generator=True)	
	for res in tqdm(canidate_results,  total =  len(filter_candidates_input_l), desc = "Filter candidates"):
		target_mol_id, targert_smiles, case_df = res
		case_df['target_mol_id'] = target_mol_id
		case_df['targert_smiles'] = targert_smiles
		case_df = case_df[['target_mol_id','targert_smiles','inchikey_s','smiles','tanimoto']] 
		candidate_data_df_l.append(case_df)

	candidates_df = pd.concat(candidate_data_df_l)
	candidates_df['mol_id'] = candidates_df['inchikey_s']
	# make sure there is no nan value in df
	candidates_df = candidates_df.dropna()
	print(f"> Save to {candidates_fp}")
	candidates_df.reset_index(drop=True, inplace=True)
	candidates_df.to_pickle(candidates_fp)