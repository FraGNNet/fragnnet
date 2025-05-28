import pandas as pd
from fragnnet.utils.misc_utils import booltype
from tqdm import tqdm
import argparse
import os
import fragnnet.utils.data_utils as data_utils
import fragnnet.utils.frag_utils as frag_utils

def get_args():
	parser = argparse.ArgumentParser()
	
	# frag feats configs
	parser.add_argument("--max_depth",type=int,default=3)
	parser.add_argument("--max_time",type=int,default=150)
	parser.add_argument("--project_dp",type=str,default=os.getcwd())
	parser.add_argument("--frag_dp",type=str,default="data/ms2c/casmi2016_neg/frag")
	parser.add_argument("--max_h_transfer",type=int,default=4)
	parser.add_argument("--allowed_elements",type=str,nargs="+",default=frag_utils.ELEMENTS)
	parser.add_argument("--nb_isomorphic",type=booltype,default=False)
	parser.add_argument("--max_iterations",type=int,default=-1)
	parser.add_argument("--isotopes",type=booltype,default=False)
	parser.add_argument("--proc_dp", type=str,default="data/ms2c/casmi2016_neg/proc", help="")
	parser.add_argument("--allow_cached", type=booltype,default=False, help="")
	parser.add_argument("--compressed", type=booltype,default=True, help="")

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	
	args = get_args()
	print(f">read mol_df {args.proc_dp} ")
	mol_df = pd.read_pickle(os.path.join(args.proc_dp, "mol_df.pkl.gz"))
	print("> mkdir dir ")
	os.makedirs(args.frag_dp, exist_ok=True)
	dag_feat_inputs_l = []
	print("prepare data")
	for _, row in mol_df.iterrows():
		dag_feat_inputs_l.append([
								row['mol'],
								row['mol_id'], 
								args.max_depth,
								True, # h_prior
								args.max_h_transfer,
								args.max_time,
								args.isotopes, 
								args.nb_isomorphic,
								args.max_iterations,
								args.frag_dp,
								args.allow_cached,
								args.compressed, # compressed
								])
	
	frag_results = data_utils.par_apply(iter(dag_feat_inputs_l),frag_utils.timed_get_dags,True, return_as_generator=True)
	for res in tqdm(frag_results, total = len(dag_feat_inputs_l), desc = "Compute Frags"):
		continue