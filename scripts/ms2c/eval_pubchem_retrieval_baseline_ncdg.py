
import os
import pandas as pd
import argparse
import numpy as np
from tqdm import tqdm
import glob
from sklearn.metrics import ndcg_score
#from sklearn.metrics import ndcg_score

LAST_RANK = 51
LAST_SCORE = 0.0

def get_args():
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--inchikey_proc_dp", type=str,default="./data/ms2c/pubchem/run_240925/proc/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50")
	parser.add_argument("--scaffold_proc_dp", type=str,default="./data/ms2c/pubchem/run_240925/proc/pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50")
 
	args = parser.parse_args()
	return args

if __name__ == "__main__":

	args = get_args()

	def get_ncdg_baseline(proc_dp, dcg_form="linear", p = None):
		spec_input_df = pd.read_pickle(os.path.join(proc_dp,"spec_df.pkl.gz"))
		mol_input_df = pd.read_pickle(os.path.join(proc_dp,"mol_df.pkl.gz"))
		mol_input_df = mol_input_df[['mol_id','inchikey_s','smiles']]
		spec_input_df = spec_input_df.merge(mol_input_df, left_on='mol_id', right_on='mol_id')
		spec_input_df = spec_input_df[['group_id', 'target_mol_id', 'target_mol_smiles', 'target_inchikey_s', 'mol_id', 'smiles', 'inchikey_s', 'tanimoto']]
		spec_input_df = spec_input_df.drop_duplicates(subset=['group_id'])
		#spec_input_df['keep'] = spec_input_df.groupby('target_mol_id').apply(lambda x: True if len(x[x['target_inchikey_s'] == x['mol_id']]) == 1 else False).reset_index(level=0, drop=True)
		keep_df = spec_input_df.groupby('target_mol_id').apply(lambda x: True if (x['target_inchikey_s'] == x['mol_id']).sum() == 1 else False, include_groups=False).reset_index()
		keep_target_mol_ids = keep_df[keep_df.iloc[:,1]].iloc[:,0].to_list()
		spec_input_df = spec_input_df[spec_input_df['target_mol_id'].isin(keep_target_mol_ids)]
		spec_input_df['tanimoto_distance'] = 1 - spec_input_df['tanimoto'] 
		spec_input_df['tanimoto_rand'] = np.random.rand(len(spec_input_df) )

		if dcg_form == "linear":
			spec_input_df['rel_score'] = spec_input_df['tanimoto']
		elif dcg_form == "exp":
			spec_input_df['rel_score'] = spec_input_df['tanimoto'].apply(lambda x : 2 ** x -1)
		else:
			print(f"{dcg_form} is not supported")
   
		ndcg_neg_tanimoto = spec_input_df.groupby('target_mol_id').apply(
								lambda x: ndcg_score(x['rel_score'].to_numpy()[np.newaxis, :], \
											x['tanimoto_distance'].to_numpy()[np.newaxis, :], \
											k = p
								), include_groups=False).to_numpy()
  
		ndcg_rand_tanimoto = spec_input_df.groupby('target_mol_id').apply(
								lambda x: ndcg_score(x['rel_score'].to_numpy()[np.newaxis, :], \
											x['tanimoto_rand'].to_numpy()[np.newaxis, :], \
											k = p
								), include_groups=False).to_numpy()

		return ndcg_neg_tanimoto, ndcg_rand_tanimoto

	#cdg_form = "linear"
	p = 50
	for cdg_form in ["linear","exp"]:
		#for p in [5,10,50]:
		print(f"inchikey {cdg_form} ncdg @{p}")
		ndcg_neg_tanimoto, ndcg_rand_tanimoto = get_ncdg_baseline(args.inchikey_proc_dp, cdg_form, p)
		print(f"worest: {np.mean(ndcg_neg_tanimoto)}")
		print(f"random: {np.mean(ndcg_rand_tanimoto)}")
	
		print(f"scaffold {cdg_form} ncdg @{p}")
		ndcg_neg_tanimoto, ndcg_rand_tanimoto = get_ncdg_baseline(args.scaffold_proc_dp, cdg_form, p)
		print(f"worest: {np.mean(ndcg_neg_tanimoto)}")
		print(f"random: {np.mean(ndcg_rand_tanimoto)}")