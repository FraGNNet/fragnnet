import argparse
from tqdm import tqdm

from fragnnet.utils.misc_utils import booltype, get_best_ckpt_from_wandb
from fragnnet.utils.ms2c_utils import load_model_and_init_config, run_spectra_prediction

import os
import pandas as pd
import json
import time
import multiprocessing


def get_args():
    
	parser = argparse.ArgumentParser()
	parser.add_argument("--device", type=str,default="cuda:0", help="")
	parser.add_argument("--save_fp", type=str,default="./data/ms2c/pubchem/predicted/ms2pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50", help="")
	parser.add_argument("--proc_dp", type=str,default="./data/ms2c/pubchem/proc/pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50", help="")
	parser.add_argument("--frag_dp", type=str,default="./data/ms2c/pubchem/frags/d3_h4_isoFalse/pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50", help="frag dp only used in fragnnet")

	parser.add_argument("--use_cached_ckpt",type=booltype,default=False)
	parser.add_argument("--batch_size",type=int,default=50)
	parser.add_argument("--model_save_dp",type=str, default="./saved_ckpts")
	parser.add_argument("--model_ckpt",type=str, default=None)
	parser.add_argument("--wandb_run_id",type=str, default= 'wf1zkb7n')
	parser.add_argument("--custom_fp",type=str, required=False)
	parser.add_argument("--template_fp",type=str, default="./config/template.yml")
	parser.add_argument("--num_workers",type=int, default=multiprocessing.cpu_count())
	args = parser.parse_args()
	return args

if __name__ == "__main__":

	args = get_args()
	timestr = time.strftime("%Y%m%d-%H%M%S")
	save_dp = os.path.dirname(args.save_fp)
	os.makedirs(save_dp, exist_ok=True)
	records_dict = vars(args)

	if args.model_ckpt is None:
		os.makedirs(args.model_save_dp, exist_ok=True)	
		model_save_dp = args.model_save_dp
		print(f">> save model to {model_save_dp} from wandb")
		#assert args.wandb_run_id is not None
		print(f">> wandb run id: {args.wandb_run_id}")
		ckpt_fp = get_best_ckpt_from_wandb(model_save_dp, args.wandb_run_id, use_cached = args.use_cached_ckpt)
	else:
		ckpt_fp = args.model_ckpt
	records_dict['model_ckpt'] = ckpt_fp
	json.dump(records_dict, open(f"{args.save_fp}_{timestr}.json", "w"), indent=2)
 
	# preprocess load datasets
	print(f"> Loading datasets {args.proc_dp}")

	print("> preload mol_df")
	mol_df = pd.read_pickle(os.path.join(args.proc_dp,'mol_df.pkl.gz'))

	print("> preload spec_fp")
	spec_df = pd.read_pickle(os.path.join(args.proc_dp,'spec_df.pkl.gz'))

	print(f"> Loading models {ckpt_fp}")
	auxiliary_scores = ["cos_sim","cos_hun"]
	eval_mz_bin_res = [0.01]
	model, config_d, device = load_model_and_init_config(
		ckpt_fp = ckpt_fp, 
		device = args.device, 
        batch_size = args.batch_size, 
        auxiliary_scores = auxiliary_scores,
        eval_mz_bin_res = eval_mz_bin_res,
		custom_fp = args.custom_fp,
		template_fp = args.template_fp,
		num_workers = args.num_workers,
	)
 
	#if args.group_by_target:
	# NOTE this only one for version 1, as there is only on Adduct Type
	# TODO change this for multiple Adduct Type

	df = run_spectra_prediction(
		model = model,   
		config_d = config_d,
		mol_data_ptr = mol_df,
		spec_data_ptr = spec_df,
		frag_dp = args.frag_dp,
		device = device,
		validate = True
	)
 
	df.to_pickle(args.save_fp)