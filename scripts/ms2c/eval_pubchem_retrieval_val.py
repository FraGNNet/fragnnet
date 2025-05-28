
import os
import pandas as pd
import argparse
import numpy as np
from tqdm import tqdm
import glob
from sklearn.metrics import ndcg_score

LAST_RANK = 51
LAST_SCORE = 0.0

def get_args():
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--save_dp",type=str,default="./data/ms2c/pubchem/run_240925/benchmark/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50")
	parser.add_argument("--proc_dp", type=str,default="./data/ms2c/pubchem/run_240925/proc/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50")
	parser.add_argument("--predicted_dp", type=str,default="./data/ms2c/pubchem/run_240925/predicted/ms2pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50_ranking_only")
 
	args = parser.parse_args()
	return args

if __name__ == "__main__":

	args = get_args()
	os.makedirs(args.save_dp, exist_ok=True)
	print(f"> {args.proc_dp}")
	print(f"> {args.predicted_dp}")
	print(f"> {args.save_dp}")

	print("> load target_spec_df")
	spec_input_df = pd.read_pickle(os.path.join(args.proc_dp,"spec_df.pkl.gz"))
	mol_input_df = pd.read_pickle(os.path.join(args.proc_dp,"mol_df.pkl.gz"))
	mol_input_df = mol_input_df[['mol_id','inchikey_s','smiles']]
	spec_input_df = spec_input_df.merge(mol_input_df, left_on='mol_id', right_on='mol_id')
	spec_input_df = spec_input_df[['group_id', 'target_mol_id', 'target_mol_smiles', 'target_inchikey_s', 'mol_id', 'smiles', 'inchikey_s', 'tanimoto']]
	spec_input_df = spec_input_df.drop_duplicates(subset=['group_id'])
	#spec_input_df['keep'] = spec_input_df.groupby('target_mol_id').apply(lambda x: True if len(x[x['target_inchikey_s'] == x['mol_id']]) == 1 else False).reset_index(level=0, drop=True)
	keep_df = spec_input_df.groupby('target_mol_id').apply(lambda x: True if (x['target_inchikey_s'] == x['mol_id']).sum() == 1 else False).reset_index()
	keep_target_mol_ids = keep_df[keep_df.iloc[:,1]].iloc[:,0].to_list()
	spec_input_df = spec_input_df[spec_input_df['target_mol_id'].isin(keep_target_mol_ids)]
	total_tasks =  spec_input_df['target_mol_id'].nunique()
	print(f"total_tasks {total_tasks}")

	spec_input_df['linear_rel_score'] = spec_input_df['tanimoto']
	spec_input_df['exp_rel_score'] = spec_input_df['tanimoto'].apply(lambda x : 2 ** x -1)

	# Get a list of all .gz files in the directory
	gz_files = glob.glob(os.path.join(args.predicted_dp, '*.gz'))
	pkl_files = glob.glob(os.path.join(args.predicted_dp, '*.pkl'))
	all_files = gz_files + pkl_files
	# Extract and print the basenames (without the .gz extension)
	overall_stats_rows = []
	ncdg_p = 50

	for file in tqdm(all_files):
		basename = os.path.basename(file).replace('.pkl', '').replace('.gz', '')
		print(f">> starting file = {basename}")
		predicted_spec_df = pd.read_pickle(file)
		print(f"> predicted_spec_df.shape (pre-merge) = {predicted_spec_df.shape}")
		predicted_spec_df = predicted_spec_df.merge(spec_input_df, left_on='spec_id', right_on='group_id')
		print(f"> predicted_spec_df.shape (post-merge) = {predicted_spec_df.shape}")
		predicted_spec_df['cos_sim_0.01_rank'] = predicted_spec_df.groupby('target_mol_id')['cos_sim_0.01'].rank(ascending=False)
		predicted_spec_df['cos_hun_rank'] = predicted_spec_df.groupby('target_mol_id')['cos_hun'].rank(ascending=False)

		stats_df = predicted_spec_df.groupby('target_mol_id').apply(lambda x: x[x['target_inchikey_s'] == x['inchikey_s']])
		print(f"> stats_df.shape = {stats_df.shape}")

		#for ncdg_p in [5,10,50]:
		stats_df['ndcg_linear_cos_sim_0.01'] = predicted_spec_df.groupby('target_mol_id').apply(
								lambda x: ndcg_score(x['linear_rel_score'].to_numpy()[np.newaxis, :], \
											x['cos_sim_0.01'].to_numpy()[np.newaxis, :], \
											k = ncdg_p
								)).to_numpy()

		stats_df['ndcg_exp_cos_sim_0.01'] = predicted_spec_df.groupby('target_mol_id').apply(
								lambda x: ndcg_score(x['exp_rel_score'].to_numpy()[np.newaxis, :], \
											x['cos_sim_0.01'].to_numpy()[np.newaxis, :], \
											k = ncdg_p
								)).to_numpy()
	
		stats_df['ndcg_linear_cos_hun'] = predicted_spec_df.groupby('target_mol_id').apply(
								lambda x: ndcg_score(x['linear_rel_score'].to_numpy()[np.newaxis, :], \
											x['cos_hun'].to_numpy()[np.newaxis, :], \
											k = ncdg_p
								)).to_numpy()
		stats_df['ndcg_exp_cos_hun'] = predicted_spec_df.groupby('target_mol_id').apply(
								lambda x: ndcg_score(x['exp_rel_score'].to_numpy()[np.newaxis, :], \
											x['cos_hun'].to_numpy()[np.newaxis, :], \
											k = ncdg_p
								)).to_numpy()

		stats_df = stats_df[['group_id','target_mol_smiles', 'target_inchikey_s', 'cos_sim_0.01','cos_hun','cos_sim_0.01_rank','cos_hun_rank',
                       'ndcg_linear_cos_sim_0.01','ndcg_exp_cos_sim_0.01','ndcg_linear_cos_hun','ndcg_exp_cos_hun']]
		stats_df.to_csv(os.path.join(args.save_dp, f"{basename}_summary.csv"),index = False)
  
		for rank_type in ['cos_sim_0.01', 'cos_hun']:
			top_stats = [basename,rank_type]
			for k in [1,3,5,10]:
				top_stats.append(len(stats_df[stats_df[rank_type+'_rank'] <= k]))
				top_stats.append(len(stats_df[stats_df[rank_type+'_rank'] <= k])/len(stats_df))
			# ans_score_mean
			top_stats.append(stats_df[rank_type].mean())
			top_stats.append(stats_df[rank_type].std())
			# avg_ranks and std
			top_stats.append(stats_df[rank_type+'_rank'].mean())
			top_stats.append(stats_df[rank_type+'_rank'].std())
			# median_ranks
			top_stats.append(stats_df[rank_type+'_rank'].median())
			# tanimoto
			top_one_stats_df = predicted_spec_df.groupby('target_mol_id').apply(lambda x: x[x[rank_type+'_rank'] == 1.0])
			top_stats.append(top_one_stats_df['tanimoto'].mean())
			top_stats.append(top_one_stats_df['tanimoto'].std())
			# ncdg
			top_stats.append(stats_df['ndcg_linear_'+ rank_type].mean())
			top_stats.append(stats_df['ndcg_linear_'+ rank_type].std())
			top_stats.append(stats_df['ndcg_exp_'+ rank_type].mean())
			top_stats.append(stats_df['ndcg_exp_'+ rank_type].std())
			overall_stats_rows.append(top_stats)

	overall_stats_df = pd.DataFrame(overall_stats_rows, \
		columns = ['model','ranking method', 'top-1','top-1 ratio','top-3','top-3 ratio','top-5','top-5 ratio','top-10','top-10 ratio', 
				'ans_score_mean', 'ans_score_std', 'rank_mean','rank_std','rank_median','top_1_tanimoto_mean','top_1_tanimoto_std', 
    			'ndcg_linear_mean','ndcg_linear_std','ndcg_exp_mean','ndcg_exp_std'])
	overall_stats_df = overall_stats_df[['model','ranking method', 'top-1','top-3','top-5','top-10',
									  'top-1 ratio','top-3 ratio','top-5 ratio','top-10 ratio',
									  'ans_score_mean', 'ans_score_std','rank_mean','rank_std','rank_median',
           								'top_1_tanimoto_mean','top_1_tanimoto_std', 
           								'ndcg_linear_mean','ndcg_linear_std','ndcg_exp_mean','ndcg_exp_std']]
	overall_stats_df = overall_stats_df.sort_values(by=['model','ranking method'], ascending=True)

	overall_stats_df.to_csv(os.path.join(args.save_dp, "overall_stats_summary.csv"),index = False)
