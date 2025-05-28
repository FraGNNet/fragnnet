
import os
import pandas as pd
import argparse
from tqdm import tqdm
import glob

LAST_RANK = 51
LAST_SCORE = 0.0

def get_args():
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--save_dp",type=str,default="./data/ms2c/pubchem/run_240925/predicted/ms2pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50_ranking_only")
	parser.add_argument("--predicted_dp", type=str,default="./data/ms2c/pubchem/run_240925/predicted/ms2pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50")
 
	args = parser.parse_args()
	return args

if __name__ == "__main__":

	args = get_args()
	os.makedirs(args.save_dp, exist_ok=True)
	print(f"> predicted_dp {args.predicted_dp}")
	print(f"> save_dp {args.save_dp}")
	gz_files = glob.glob(os.path.join(args.predicted_dp, '*.gz'))
	overall_stats_rows = []
	for gz_file in tqdm(gz_files):
		basename = os.path.basename(gz_file).replace('.pkl.gz', '')
		predicted_spec_df = pd.read_pickle(gz_file)
		predicted_spec_df = predicted_spec_df.drop(columns=['pred_mzs', 'pred_ints'])
		predicted_spec_df.to_pickle(os.path.join(args.save_dp,basename + '.pkl.gz'))