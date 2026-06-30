from fragnnet.runner import load_config, init_dataset, init_dataloader
from fragnnet.pl_model import FragGNNPL
from fragnnet.utils.misc_utils import booltype, scatter_logsumexp
from fragnnet.utils.script_utils import init_wandb_ckpt, run_inference, select_model_vals, log_mean
from fragnnet.loss import sparse_conditional_entropy_fn

import torch as th
import pandas as pd
import argparse	
import wandb
import os
from pprint import pprint


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--depth", type=str, default="d4", choices=["d3","d4"])
	parser.add_argument("--debug", type=booltype, default=False)
	parser.add_argument("--mode", type=str, choices=["baseline","posfn","negfn","mix"], default="baseline")
	parser.add_argument("--eval_split", type=str, default="test", choices=["train","val","test"])
	parser.add_argument("--split_type", type=str, default="scaffold", choices=["scaffold","inchikey"])
	parser.add_argument("--output_dp", type=str, required=True)
	parser.add_argument("--entropy_exp", type=str, default="entropyv3-1")
	parser.add_argument("--overwrite", type=booltype, default=False)
	args = parser.parse_args()

	depth = args.depth
	debug = args.debug
	mode = args.mode
	eval_split = args.eval_split
	split_type = args.split_type
	overwrite = args.overwrite
	entropy_exp = args.entropy_exp
	pd.options.display.float_format = '{:.4f}'.format

	output_dir = f"{split_type}_{depth}_{mode}_{eval_split}_{debug}"
	output_dp = os.path.join(args.output_dp,output_dir)
	os.makedirs(output_dp,exist_ok=True)

	# load configs
	print(">>> Loading configs")
	template_fp = "config/template.yml"
	custom_fp = os.path.join("config",entropy_exp,split_type,f"{depth}_script_config.yml")
	assert os.path.isfile(custom_fp)
	config_d = load_config(template_fp, custom_fp)
	# modify config to record relevant metrics
	config_d["auxiliary_scores"] = ["cos_sim","cos_hun","true_oos_prob"]
	config_d["skip_extra_losses"] = False
	config_d["compile"] = False

	if debug:
		raise NotImplementedError
		print(">> debug mode, using small dataset")
		config_d["spec_fp"] = "data/proc/debug_smallv2/spec_df.pkl"
		config_d["mol_fp"] = "data/proc/debug_smallv2/mol_df.pkl"
		config_d["split_dp"] = "data/split/debug_smallv2"
		config_d["ann_fp"] = "data/proc/debug_smallv2/ann_df.pkl"
		batch_cutoff = 3
	else:
		batch_cutoff = int(1e6)

	vals_fp = os.path.join(output_dp,"vals.pkl")
	sims_fp = os.path.join(output_dp,"sims.pkl")
	entropies_fp = os.path.join(output_dp,"entropies.pkl")

	if os.path.isfile(sims_fp) and os.path.isfile(entropies_fp) and not args.overwrite:

		# don't load anything here
		pass

	elif not os.path.isfile(vals_fp) or (os.path.isfile(vals_fp) and args.overwrite):

		api = wandb.Api(timeout=30)
		
		if mode != "mix":
		
			runs = api.runs(
					path="fragnnet/fragnnet",
					filters={"group":f"{entropy_exp}_{split_type}_{depth}_{mode}_ens"})
			model_to_run_id = {run.name: run.id for run in runs}
		
		else:
		
			model_to_run_id = {}
			mode_to_seed_range = {
				"baseline": (0,5),
				"posfn": (5,10),
				"negfn": (10,15)
			}
			for smode, seed_range in mode_to_seed_range.items():
				runs = api.runs(
						path="fragnnet/fragnnet",
						filters={"group":f"{entropy_exp}_{split_type}_{depth}_{smode}_ens"})
				for run in runs:
					for i in range(seed_range[0],seed_range[1]):
						if f"s{i:02d}" in run.tags:
							model_to_run_id[run.name] = run.id
							break

		if debug:
			keys = sorted(list(model_to_run_id.keys()))
			model_to_run_id = {k: model_to_run_id[k] for k in keys[:2]}
	
		pprint(model_to_run_id)

		num_models = len(model_to_run_id.keys())
		assert num_models > 0, num_models

		# load datasets
		print(">>> Loading datasets")
		ds = init_dataset(config_d, splits=(eval_split,))[0]
		dl = init_dataloader(ds, config_d)

		model_to_vals = {}
		for model_name, run_id in model_to_run_id.items():
			# load model
			print(f">>> Loading model {model_name} - {run_id}")
			device = th.device("cuda:0") if config_d["accelerator"]=="gpu" else th.device("cpu")
			model = init_wandb_ckpt(run_id,last_ckpt=False,model_cls=FragGNNPL,config_d=config_d)
			model.to(device)
			model.eval()
			# run inference
			vals = run_inference(dl, model, device, eval_split, batch_cutoff, config_d["nb_iso"])
			model_to_vals[model_name] = vals
			del model

		th.save(model_to_vals,vals_fp)

		# delete dataset, dataloader
		del ds, dl

	else:

		print(f">>> Loading previous values from {vals_fp}")
		model_to_vals = th.load(vals_fp)
		num_models = len(model_to_vals.keys())

	# load model
	model = FragGNNPL(**config_d)

	# compute ensemble distributions
	print(">>> Cosine Similarity")

	if not os.path.isfile(sims_fp) or overwrite:

		select_keys = [
			"pred_mzs",
			"pred_logprobs",
			"pred_batch_idxs",
			"true_mzs",
			"true_logprobs",
			"true_batch_idxs",
			"true_prec_mzs"
		]
		select_d = select_model_vals(model_to_vals,select_keys,stack_dim=0)
		# get target stuff
		true_logprobs = select_d["true_logprobs"][0]
		true_mzs = select_d["true_mzs"][0]
		true_batch_idxs = select_d["true_batch_idxs"][0]
		true_prec_mzs = select_d["true_prec_mzs"][0]

		ind_cos_sim_bin = []
		ind_cos_hun = []

		for i in range(num_models):
			
			print(f">> model {i}")
			
			# get predicted stuff
			pred_logprobs = select_d["pred_logprobs"][i]
			pred_mzs = select_d["pred_mzs"][i]
			pred_batch_idxs = select_d["pred_batch_idxs"][i]
			
			# calculate metrics
			with th.inference_mode():
				metric_input_d = {
					"true_mzs": true_mzs,
					"true_ints": th.exp(true_logprobs),
					"true_batch_idxs": true_batch_idxs,
					"pred_mzs": pred_mzs,
					"pred_ints": th.exp(pred_logprobs),
					"pred_batch_idxs": pred_batch_idxs,
					"true_prec_mzs": true_prec_mzs
				}
				metric_output_d = model.metric_fn(**metric_input_d)
			
			cos_sim_bin = metric_output_d["cos_sim_0.01"]
			cos_hun = metric_output_d["cos_hun"]
			ind_cos_sim_bin.append(cos_sim_bin)
			ind_cos_hun.append(cos_hun)
		
		ind_cos_sim_bin = th.stack(ind_cos_sim_bin,dim=0)
		ind_cos_hun = th.stack(ind_cos_hun,dim=0)

		ens_pred_logprobs = log_mean(select_d["pred_logprobs"],dim=0)
		ens_pred_mzs = select_d["pred_mzs"][0]
		ens_pred_batch_idxs = select_d["pred_batch_idxs"][0]
		
		with th.inference_mode():
			metric_input_d = {
				"true_mzs": true_mzs,
				"true_ints": th.exp(true_logprobs),
				"true_batch_idxs": true_batch_idxs,
				"pred_mzs": ens_pred_mzs,
				"pred_ints": th.exp(ens_pred_logprobs),
				"pred_batch_idxs": ens_pred_batch_idxs,
				"true_prec_mzs": true_prec_mzs
			}
			metric_output_d = model.metric_fn(**metric_input_d)

		ens_cos_sim_bin = metric_output_d["cos_sim_0.01"]
		ens_cos_hun = metric_output_d["cos_hun"]

		sims_d = {
			"ind_cos_sim_bin": ind_cos_sim_bin,
			"ind_cos_hun": ind_cos_hun,
			"ens_cos_sim_bin": ens_cos_sim_bin,
			"ens_cos_hun": ens_cos_hun
		}

		th.save(sims_d, sims_fp)

	else:

		sims_d = th.load(sims_fp)
		ind_cos_sim_bin = sims_d["ind_cos_sim_bin"]
		ind_cos_hun = sims_d["ind_cos_hun"]
		ens_cos_sim_bin = sims_d["ens_cos_sim_bin"]
		ens_cos_hun = sims_d["ens_cos_hun"]

	print(">> average individual")
	print(f"> cos_sim_0.01: {ind_cos_sim_bin.mean()} +/- {ind_cos_sim_bin.mean(dim=1).std()}")
	print(f"> cos_hun: {ind_cos_hun.mean()} +/- {ind_cos_hun.mean(dim=1).std()}")

	print(">> ensemble")
	print(f"> cos_sim_0.01: {ens_cos_sim_bin.mean()}")
	print(f"> cos_hun: {ens_cos_hun.mean()}")

	print(">>> Entropy")

	if not os.path.isfile(entropies_fp) or overwrite:

		select_keys = [
			"pred_joint_logprobs",
			"pred_formula_logprobs",
			"pred_formula_node_logprobs",
			"pred_formula_batch_idxs",
			"pred_joint_formula_idxs",
			"pred_joint_batch_idxs",
			"pred_formula_formula_idxs",
			"pred_nb_joint_logprobs",
			"pred_nb_formula_node_logprobs",
			"pred_nb_joint_formula_idxs",
			"pred_nb_joint_batch_idxs"
		]
		select_d = select_model_vals(model_to_vals,select_keys,stack_dim=0)

		ind_formula_node_e = []
		ind_formula_node_ne = []
		ind_nb_formula_node_e = []
		ind_nb_formula_node_ne = []

		for i in range(num_models):
			
			print(f">> model {i}")
			
			formula_node_logprobs = select_d["pred_formula_node_logprobs"][i]
			formula_logprobs = select_d["pred_formula_logprobs"][i]
			formula_batch_idxs = select_d["pred_formula_batch_idxs"][i]
			joint_formula_idxs = select_d["pred_joint_formula_idxs"][i]
			joint_batch_idxs = select_d["pred_joint_batch_idxs"][i]
			nb_node_formula_logprobs = select_d["pred_nb_formula_node_logprobs"][i]
			nb_joint_formula_idxs = select_d["pred_nb_joint_formula_idxs"][i]
			nb_joint_batch_idxs = select_d["pred_joint_batch_idxs"][i]

			formula_node_e, formula_node_ne = sparse_conditional_entropy_fn(
				formula_logprobs,
				formula_batch_idxs,
				formula_node_logprobs,
				joint_formula_idxs,
				joint_batch_idxs
			)

			nb_formula_node_e, nb_formula_node_ne = sparse_conditional_entropy_fn(
				formula_logprobs,
				formula_batch_idxs,
				nb_node_formula_logprobs,
				nb_joint_formula_idxs,
				nb_joint_batch_idxs
			)
				
			ind_formula_node_e.append(formula_node_e)
			ind_formula_node_ne.append(formula_node_ne)
			ind_nb_formula_node_e.append(nb_formula_node_e)
			ind_nb_formula_node_ne.append(nb_formula_node_ne)

		ind_formula_node_e = th.stack(ind_formula_node_e,dim=0)
		ind_formula_node_ne = th.stack(ind_formula_node_ne,dim=0)
		ind_nb_formula_node_e = th.stack(ind_nb_formula_node_e,dim=0)
		ind_nb_formula_node_ne = th.stack(ind_nb_formula_node_ne,dim=0)

		ens_joint_logprobs = log_mean(select_d["pred_joint_logprobs"],dim=0)
		ens_formula_logprobs = log_mean(select_d["pred_formula_logprobs"],dim=0)
		ens_formula_batch_idxs = select_d["pred_formula_batch_idxs"][0]
		ens_joint_formula_idxs = select_d["pred_joint_formula_idxs"][0]
		ens_joint_batch_idxs = select_d["pred_joint_batch_idxs"][0]
		ens_formula_formula_idxs = select_d["pred_formula_formula_idxs"][0]
		assert th.all(th.arange(ens_formula_formula_idxs.shape[0]) == ens_formula_formula_idxs)
		# ens_formula_logprobs = ens_formula_logprobs[th.argsort(ens_formula_formula_idxs)]
		ens_formula_node_logprobs = ens_joint_logprobs - ens_formula_logprobs[ens_joint_formula_idxs]
		ens_formula_node_logprobs = th.clamp(ens_formula_node_logprobs, max=0.)
		ens_formula_node_lse = scatter_logsumexp(ens_formula_node_logprobs, ens_joint_formula_idxs, dim_size=ens_formula_formula_idxs.shape[0])
		# print(ens_formula_node_lse.max(), ens_formula_node_lse.min())

		ens_formula_node_e, ens_formula_node_ne = sparse_conditional_entropy_fn(
			ens_formula_logprobs,
			ens_formula_batch_idxs,
			ens_formula_node_logprobs,
			ens_joint_formula_idxs,
			ens_joint_batch_idxs
		)

		ens_nb_joint_logprobs = log_mean(select_d["pred_nb_joint_logprobs"],dim=0)
		ens_nb_joint_formula_idxs = select_d["pred_nb_joint_formula_idxs"][0]
		ens_nb_joint_batch_idxs = select_d["pred_nb_joint_batch_idxs"][0]
		ens_nb_formula_node_logprobs = ens_nb_joint_logprobs - ens_formula_logprobs[ens_nb_joint_formula_idxs]
		ens_nb_formula_node_logprobs = th.clamp(ens_nb_formula_node_logprobs, max=0.)
		ens_nb_formula_node_lse = scatter_logsumexp(ens_nb_formula_node_logprobs, ens_nb_joint_formula_idxs, dim_size=ens_formula_formula_idxs.shape[0])

		ens_nb_formula_node_e, ens_nb_formula_node_ne = sparse_conditional_entropy_fn(
			ens_formula_logprobs,
			ens_formula_batch_idxs,
			ens_nb_formula_node_logprobs,
			ens_nb_joint_formula_idxs,
			ens_nb_joint_batch_idxs
		)

		entropies_d = {
			"ind_formula_node_e": ind_formula_node_e,
			"ind_formula_node_ne": ind_formula_node_ne,
			"ind_nb_formula_node_e": ind_nb_formula_node_e,
			"ind_nb_formula_node_ne": ind_nb_formula_node_ne,
			"ens_formula_node_e": ens_formula_node_e,
			"ens_formula_node_ne": ens_formula_node_ne,
			"ens_nb_formula_node_e": ens_nb_formula_node_e,
			"ens_nb_formula_node_ne": ens_nb_formula_node_ne
		}

		th.save(entropies_d, entropies_fp)

	else:

		entropies_d = th.load(entropies_fp)
		ind_formula_node_e = entropies_d["ind_formula_node_e"]
		ind_formula_node_ne = entropies_d["ind_formula_node_ne"]
		ind_nb_formula_node_e = entropies_d["ind_nb_formula_node_e"]
		ind_nb_formula_node_ne = entropies_d["ind_nb_formula_node_ne"]
		ens_formula_node_e = entropies_d["ens_formula_node_e"]
		ens_formula_node_ne = entropies_d["ens_formula_node_ne"]
		ens_nb_formula_node_e = entropies_d["ens_nb_formula_node_e"]
		ens_nb_formula_node_ne = entropies_d["ens_nb_formula_node_ne"]

	print(">> average individual")
	# print(f"> formula_node_e: {ind_formula_node_e.mean()} +/- {ind_formula_node_e.mean(dim=1).std()}")
	print(f"> formula_node_ne: {ind_formula_node_ne.mean()} +/- {ind_formula_node_ne.mean(dim=1).std()}")
	# print(f"> nb_formula_node_e: {ind_nb_formula_node_e.mean()} +/- {ind_nb_formula_node_e.mean(dim=1).std()}")
	print(f"> nb_formula_node_ne: {ind_nb_formula_node_ne.mean()} +/- {ind_nb_formula_node_ne.mean(dim=1).std()}")

	print(">> ensemble")
	# print(f"> formula_node_e: {ens_formula_node_e.mean()}")
	print(f"> formula_node_ne: {ens_formula_node_ne.mean()}")
	# print(f"> nb_formula_node_e: {ens_nb_formula_node_e.mean()}")
	print(f"> nb_formula_node_ne: {ens_nb_formula_node_ne.mean()}")
