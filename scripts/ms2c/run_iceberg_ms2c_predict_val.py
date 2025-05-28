import time
import torch as th
import pandas as pd

import os
import argparse
import json
from fragnnet.iceberg.pl_model import IcebergGenPL
from fragnnet.iceberg.dataset import SpecMolMagmaGenDataset
from fragnnet.utils.misc_utils import booltype
from fragnnet.utils.data_utils import par_apply, par_apply_series
from rdkit import Chem
from tqdm import tqdm
from fragnnet.utils.misc_utils import booltype, get_best_ckpt_from_wandb
from fragnnet.utils.ms2c_utils import load_model_and_init_config, run_spectra_prediction
from rdkit import RDLogger
import multiprocessing

def run_iceberg_gen(iceberg_gen_inputs, iceberg_gen_ckpt_fp,
				 device = None, iceberg_gen_n_jobs = 8
				 ):

	def single_predict_mol(adduct, inchi, smi, spec_id, magma_dp, threshold, max_nodes):
	 
		# disbale  UserWarning: TypedStorage is deprecated.warning
		import warnings
		warnings.filterwarnings("ignore", category=DeprecationWarning) 
		th.set_num_threads(8)
		out_file =  f"{magma_dp}/magma_tree/{spec_id}.json"
		iceberg_gen_model = IcebergGenPL.load_from_checkpoint(iceberg_gen_ckpt_fp, map_location = device)
		iceberg_gen_model.eval()
		tree_processor = SpecMolMagmaGenDataset.init_tree_processor(
			pe_embed_k=iceberg_gen_model.hparams.magma_params["pe_embed_k"],
			root_encode=iceberg_gen_model.hparams.magma_params["root_encode"],
			add_hs=iceberg_gen_model.hparams.magma_params["add_hs"]
		)
		try:
			pred = iceberg_gen_model.predict_mol(
				smi,
				adduct=adduct,
				threshold=threshold,
				device=device,
				max_nodes=max_nodes,
				tree_processor=tree_processor
			)
		except Exception as error:
			print("Error", error, adduct, inchi, smi, spec_id)
			output = None
		else:
			output = {
				"root_inchi": inchi,
				"name": spec_id,
				"frags": pred,
				"adduct": adduct
			}

			with open(out_file, "w") as fp:
				json.dump(output, fp,indent=2)

	#if device == 'cpu':
	results = par_apply(iter(iceberg_gen_inputs),single_predict_mol,True, return_as_generator=True,n_jobs=iceberg_gen_n_jobs)
	for _ in tqdm(results, total = len(iceberg_gen_inputs), desc="Running iceberg gen", leave=True):
		continue


def get_args():
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--save_fp", type=str,default="./data/ms2c/pubchem/predicted/ms2pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50/iceberg_inchikey_s0_sanity2.pkl", help="")
	parser.add_argument("--proc_dp", type=str,default="./data/ms2c/pubchem/proc/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50_sanity", help="")

	parser.add_argument("--use_cached_ckpt",type=booltype,default=True)
	parser.add_argument("--model_save_dp",type=str, default="./saved_ckpts")

 	#gen params
	parser.add_argument("--iceberg_gen_wandb_run_id",type=str, default= 'i7k9gc3d')
	parser.add_argument("--iceberg_gen_ckpt_fp", type=str,default=None, help="")
	parser.add_argument("--magma_dp", type=str,default="data/ms2c/pubchem/magma/pred_magma_nist20v3_inchikey_iceberg_gen_s0_i7k9gc3d", help="")
	parser.add_argument("--threshold", type=float, default=0.0)
	parser.add_argument("--max_nodes", type=int, default=100)
	parser.add_argument("--iceberg_gen_device", type=str,default="cuda:0", help="")
	parser.add_argument("--iceberg_gen_n_jobs", type=int,default=8, help="")
	parser.add_argument("--run_iceberg_gen",type=booltype,default=True)
	parser.add_argument("--iceberg_gen_use_cached",type=booltype,default=True)
	
	#iceberg_inten params
	parser.add_argument("--iceberg_inten_wandb_run_id",type=str, default= 'c8iwu7il')
	parser.add_argument("--iceberg_inten_ckpt_fp", type=str,default=None, help="")
	parser.add_argument("--iceberg_inten_device", type=str,default="cuda:0", help="")
	parser.add_argument("--iceberg_inten_batch_size", type=int, default=32, help="")
	parser.add_argument("--run_iceberg_chunk_size", type=int, default=-1, help="runing number of prediction at a time. useful to prevent ram to brrrr")
	parser.add_argument("--run_iceberg_inten", type=booltype, default=True, help="")
	parser.add_argument("--iceberg_inten_custom_fp", type=str, required=False, help="")
	parser.add_argument("--iceberg_inten_template_fp", type=str, default="./config/template.yml", help="")
	parser.add_argument("--iceberg_inten_num_workers", type=int, default=multiprocessing.cpu_count(), help="")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	
	RDLogger.DisableLog("rdApp.warning")
	args = get_args()

	if not th.cuda.is_available():
		args.iceberg_gen_device = "cpu"
		args.iceberg_inten_device = "cpu"
		print("num parallelizing threads operations",th.get_num_threads())
  
	timestr = time.strftime("%Y%m%d-%H%M%S")
	save_dp = os.path.dirname(args.save_fp) or "."
	os.makedirs(save_dp, exist_ok=True)
	records_dict = vars(args)

	os.makedirs(args.model_save_dp, exist_ok=True)	
	if args.iceberg_gen_ckpt_fp is None:
		model_save_dp = args.model_save_dp
		print(f">> save model to {args.model_save_dp} from wandb with run id: {args.iceberg_gen_wandb_run_id}")
		iceberg_gen_ckpt_fp = get_best_ckpt_from_wandb(args.model_save_dp, args.iceberg_gen_wandb_run_id, use_cached = args.use_cached_ckpt)
	else:
		iceberg_gen_ckpt_fp = args.iceberg_gen_ckpt_fp
	print(f"> iceberg_gen_ckpt_fp: {iceberg_gen_ckpt_fp}")
	records_dict['iceberg_gen_ckpt_fp'] = iceberg_gen_ckpt_fp
 
	if args.iceberg_inten_ckpt_fp is None:
		model_save_dp = args.model_save_dp
		print(f">> save model to {args.model_save_dp} from wandb with run id: {args.iceberg_inten_wandb_run_id}")
		iceberg_inten_ckpt_fp = get_best_ckpt_from_wandb(args.model_save_dp, args.iceberg_inten_wandb_run_id, use_cached = args.use_cached_ckpt)
	else:
		iceberg_inten_ckpt_fp = args.iceberg_inten_ckpt_fp
	print(f"> iceberg_inten_ckpt_fp: {iceberg_inten_ckpt_fp}")
	records_dict['iceberg_inten_ckpt_fp'] = iceberg_inten_ckpt_fp

	json.dump(records_dict, open(f"{args.save_fp}_{timestr}.json", "w"), indent=2)

	print(f"> iceberg_gen_device {args.iceberg_gen_device }")
	print(f"> iceberg_inten_device {args.iceberg_inten_device }")
	print(f"> proc_dp {args.proc_dp}")
	mol_fp = os.path.join(args.proc_dp,'mol_df.pkl.gz')
	spec_fp = os.path.join(args.proc_dp,'spec_df.pkl.gz')

	print(f"> prepare mol df {mol_fp}")
	mol_df = pd.read_pickle(mol_fp)

	print(f"> prepare spec df {spec_fp}")
	spec_df = pd.read_pickle(spec_fp)

	#this is for debug
	#unique_group_ids = spec_df['group_id'].unique()
	#print(unique_group_ids)
	#unique_group_ids = unique_group_ids[0:1000]
	#spec_df = spec_df[spec_df['group_id'].isin(unique_group_ids)]

	if args.run_iceberg_gen:
		print(f"> prepare iceberg gen/magma")
		iceberg_gen_df = spec_df.drop_duplicates(subset='group_id').copy()
		iceberg_gen_df.loc[:,'magma_dp'] = args.magma_dp
		iceberg_gen_df.loc[:,'max_nodes'] = args.max_nodes
		iceberg_gen_df.loc[:,'threshold'] = args.threshold
		mol_df.loc[:,"inchi"] = par_apply_series(mol_df["mol"], lambda x : (RDLogger.DisableLog("rdApp.warning"), Chem.MolToInchi(x))[1])
		iceberg_gen_df = iceberg_gen_df.merge(mol_df[['mol_id','mol', 'smiles', "inchi"]], on = 'mol_id', how = 'left')

		# run_iceberg_gen 
		iceberg_gen_inputs = []
		print(f"> num of group ids {spec_df['group_id'].nunique()}")
		magma_tree_dp = f"{args.magma_dp}/magma_tree"
		if not args.iceberg_gen_use_cached or not os.path.exists(magma_tree_dp):
			os.makedirs(magma_tree_dp, exist_ok=True)
			iceberg_gen_inputs = iceberg_gen_df[['prec_type','inchi','smiles','group_id','magma_dp', 'threshold', 'max_nodes']].values.tolist()
		else:
			cached_group_ids = [int(f.split('.')[0]) for f in os.listdir(magma_tree_dp)]
			print(f"> num of cached group ids {len(cached_group_ids)}")
			iceberg_gen_inputs = iceberg_gen_df[~iceberg_gen_df['group_id'].isin(cached_group_ids)][['prec_type','inchi','smiles','group_id','magma_dp', 'threshold', 'max_nodes']].values.tolist()
		if len(iceberg_gen_inputs):
			run_iceberg_gen(iceberg_gen_inputs, iceberg_gen_ckpt_fp = iceberg_gen_ckpt_fp, device = args.iceberg_gen_device, iceberg_gen_n_jobs=args.iceberg_gen_n_jobs)
	
	if args.run_iceberg_inten:
		print(f"> Loading Iceberg_Inten models {iceberg_inten_ckpt_fp}")
		auxiliary_scores = ["cos_sim","cos_hun"]
		eval_mz_bin_res = [0.01]
		model, config_d, device = load_model_and_init_config(
			ckpt_fp = iceberg_inten_ckpt_fp, 
			device = args.iceberg_inten_device, 
			batch_size = args.iceberg_inten_batch_size, 
			auxiliary_scores = auxiliary_scores,
			eval_mz_bin_res = eval_mz_bin_res,
			custom_fp = args.iceberg_inten_custom_fp,
			template_fp = args.iceberg_inten_template_fp,
			num_workers = args.iceberg_inten_num_workers
		)

		if args.run_iceberg_chunk_size > 0:
			unique_group_ids = spec_df['group_id'].unique()
			chunk_save_fps = []
			for i in tqdm(range(0, len(unique_group_ids), args.run_iceberg_chunk_size)):
				unique_group_ids_chunk = unique_group_ids[i:i + args.run_iceberg_chunk_size]
				spec_chunk_df = spec_df[spec_df['group_id'].isin(unique_group_ids_chunk)]
				df = run_spectra_prediction(model = model,   
					config_d = config_d,
					mol_data_ptr = mol_df,
					spec_data_ptr = spec_chunk_df,
					magma_dp = args.magma_dp,
					device = args.iceberg_inten_device,
					validate=True
				)
				chunk_save_fp = args.save_fp.replace(".pkl",f"_chunk_{i}.pkl")
				df.to_pickle(chunk_save_fp)
				chunk_save_fps.append(chunk_save_fp)
			dfs = []
			for f in chunk_save_fps:
				print(f"> {f}")
				dfs.append(pd.read_pickle(f))		
			df = pd.concat(dfs)
			df.to_pickle(args.save_fp)
		else:
			df = run_spectra_prediction(
				model = model,   
				config_d = config_d,
				mol_data_ptr = mol_df,
				spec_data_ptr = spec_df,
				magma_dp = args.magma_dp,
				device = args.iceberg_inten_device,
				validate = True
			)
			df.to_pickle(args.save_fp)