from fragnnet.runner import load_config, init_dataloader
from fragnnet.pl_model import FragGNNPL
from fragnnet.dataset import SpecMolFragDataset
from fragnnet.utils.misc_utils import NestedDefaultDict, booltype
from fragnnet.utils.script_utils import init_wandb_ckpt, run_inference

import torch as th
import torch._dynamo
import argparse
import wandb
from pprint import pprint
import os
import gc

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--eval_split", 
		type=str, 
		default="test", 
		choices=["train","val","test","secondary"]
	)
	parser.add_argument(
		"--num_seeds", 
		type=int, 
		default=5
	)
	parser.add_argument(
		"--vals_dp", 
		type=str, 
		required=True
		# default="./inference_outputs"
	)
	parser.add_argument(
		"--force_overwrite", 
		type=booltype, 
		default=False
	)
	parser.add_argument(
		"--auxiliary_scores", 
		type=str, 
		nargs="+", 
		default=[
			"cos_sim",
			"cos_hun",
			"true_oos_prob",
			"true_oos_e"
		]
	)
	parser.add_argument(
		"--skip_extra_losses", 
		type=booltype, 
		default=False
	)
	parser.add_argument(
		"--output_subset", 
		type=str, 
		nargs="+", 
		default=[
			"true_mzs",
			"true_logprobs",
			"true_batch_idxs",
			"pred_mzs",
			"pred_logprobs",
			"pred_batch_idxs",
			"true_oos_prob",
			"oos_prob",
		]
	)
	parser.add_argument(
		"--output_everything",
		type=booltype,
		default=False
	)
	parser.add_argument(
		"--untransform_spec",
		type=booltype,
		default=False
	)
	parser.add_argument(
		"--disable_preproc",
		type=booltype,
		default=True
	)
	parser.add_argument(
		"--ablation_dset",
		type=str,
		default="ablationsv3-1"
	)
	parser.add_argument(
		"--ablation_models",
		type=str,
		nargs="+",
		default=[
			"fragnnet_d3_edges",
			"fragnnet_d3_noce",
			# "fragnnet_d4_edges",
			"fragnnet_d4_noce",
		]
	)
	args = parser.parse_args()

	eval_split = args.eval_split
	num_seeds = args.num_seeds
	vals_dp = args.vals_dp
	force_overwrite = args.force_overwrite
	ablation_models = args.ablation_models
	ablation_dset = args.ablation_dset
	if args.output_everything:
		output_subset = None
	else:
		output_subset = set(args.output_subset)

	# this is necessary to avoid a pytorch compile issue...
	# TODO: replace this with a proper checkpoint decompilation
	th._dynamo.config.suppress_errors = True

	batch_cutoff = int(1e6)

	model_to_run_id = NestedDefaultDict()
	model_to_config = NestedDefaultDict()

	# find wandb checkpoints
	template_fp = "config/template.yml"
	api = wandb.Api()
	for ablation_model in ablation_models:
		model_to_config[ablation_model] = f"config/{ablation_dset}/{ablation_model}/s0.yml"
		group_name = f"{ablation_dset}_{ablation_model}"
		runs = api.runs(
			path="fragnnet/fragnnet",
			filters={"group":group_name})
		for run in runs:
			seed = int(run.name.split("_")[-1].removeprefix("s"))
			if seed < num_seeds:
				model_to_run_id[ablation_model][seed] = run.id
	del api
	pprint(model_to_config)
	pprint(model_to_run_id)

	for ablation_model in ablation_models:
		print(f">>> Starting model_type {ablation_model}")
		custom_fp = model_to_config[ablation_model]
		print(f">>> Loading config {custom_fp}")
		config_d = load_config(template_fp, custom_fp)
		# modify config to record relevant metrics
		config_d["auxiliary_scores"] = args.auxiliary_scores
		config_d["skip_extra_losses"] = args.skip_extra_losses
		config_d["compile"] = False
		if args.disable_preproc:
			for k in ["spec", "mol", "frag", "magma", "ann"]:
				key = f"{k}_params"
				config_d[key]["preprocess"] = False
				if k in ["frag"]:
					config_d[key]["preload"] = False
		# modify config for formula strings (don't output them)
		config_d["output_formula_str"] = False
		config_d["spec_params"]["prec_type_str"] = False
		config_d["frag_params"]["formula_str"] = False
		print(">>> Loading dataset")
		ds_cls = SpecMolFragDataset
		ds = ds_cls(split=eval_split, **config_d)
		dl = init_dataloader(ds, config_d)
		model_cls = FragGNNPL
		device = th.device("cuda:0") if config_d["accelerator"]=="gpu" else th.device("cpu")
		for seed in range(num_seeds):
			print(f">>> Starting seed {seed}")
			run_id = model_to_run_id[ablation_model][seed]
			model = init_wandb_ckpt(run_id,last_ckpt=False,model_cls=model_cls,config_d=config_d)
			model.to(device)
			model.eval()
			# define output subset
			if not args.output_everything:
				output_subset = output_subset.union(model.metric_names)
			# run inference
			vals = run_inference(
				dl, 
				model, 
				device, 
				eval_split, 
				batch_cutoff, 
				config_d["nb_iso"], 
				output_subset=output_subset, 
				untransform_spec=args.untransform_spec
			)
			model.cpu()
			# save vals
			seed_vals_dp = os.path.join(args.vals_dp,ablation_model,f"s{seed}")
			os.makedirs(seed_vals_dp,exist_ok=True)
			seed_vals_fp = os.path.join(seed_vals_dp,f"{eval_split}.pkl")
			if os.path.isfile(seed_vals_fp):
				print(f"> Info: {seed_vals_fp} exists already")
				if force_overwrite:
					print(f"> Warning: overwriting {seed_vals_fp}")
					th.save(vals, seed_vals_fp)
			else:
				th.save(vals, seed_vals_fp)
			del model
			del vals
			gc.collect()
			if config_d["accelerator"]=="gpu":
				with th.no_grad():
					th.cuda.empty_cache()
