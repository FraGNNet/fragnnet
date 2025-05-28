from fragnnet.runner import load_config, init_dataloader
from fragnnet.pl_model import FragGNNPL, NeimsPL, PrecursorPL
from fragnnet.massformer.pl_model import MassFormerPL
from fragnnet.iceberg.pl_model import IcebergIntenPL
from fragnnet.graff.pl_model import GrAFFPL
from fragnnet.dataset import SpecMolDataset, SpecMolFragDataset
from fragnnet.iceberg.dataset import SpecMolMagmaIntenDataset
from fragnnet.graff.dataset import SpecMolAnnDataset
from fragnnet.utils.misc_utils import NestedDefaultDict, booltype
from fragnnet.utils.script_utils import init_wandb_ckpt, run_inference

import torch as th
import argparse
import wandb
from pprint import pprint
from glob import glob
import os
import gc

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--split_types", 
		type=str, 
		nargs="+", 
		default=["inchikey","scaffold"]
	)
	parser.add_argument(
		"--wandb_model_types", 
		type=str, 
		nargs="+", 
		default=[
			"fragnnet_d3",
			"fragnnet_d4",
			"iceberg_inten",
			"iceberg_inten_opt",
			"neims",
			"massformer",
			"graff",
			"precursor"
		]
	)
	parser.add_argument(
		"--local_model_types", 
		type=str, 
		nargs="+", 
		default=[]
	)
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
		"--local_path", 
		type=str, 
		default="./local_runs"
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
		"--dset",
		type=str,
		default="nist20v3-1"
	)
	args = parser.parse_args()

	eval_split = args.eval_split
	num_seeds = args.num_seeds
	local_path = args.local_path
	vals_dp = args.vals_dp
	force_overwrite = args.force_overwrite
	if args.output_everything:
		output_subset = None
	else:
		output_subset = set(args.output_subset)

	# this is necessary to avoid a pytorch compile issue...
	# TODO: replace this with a proper checkpoint decompilation
	th._dynamo.config.suppress_errors = True

	split_types = args.split_types
	wandb_model_types = args.wandb_model_types
	local_model_types = args.local_model_types
	batch_cutoff = int(1e6)
	
	model_type_to_model_cls = {
		"neims": NeimsPL,
		"massformer": MassFormerPL,
		"fragnnet_d3": FragGNNPL,
		"fragnnet_d4": FragGNNPL,
		"iceberg_inten": IcebergIntenPL,
		"iceberg_inten_opt": IcebergIntenPL,
		"graff3": GrAFFPL,
		"precursor": PrecursorPL
	}

	model_type_to_ds_cls = {
		"neims": SpecMolDataset,
		"massformer": SpecMolDataset,
		"fragnnet_d3": SpecMolFragDataset,
		"fragnnet_d4": SpecMolFragDataset,
		"iceberg_inten": SpecMolMagmaIntenDataset,
		"iceberg_inten_opt": SpecMolMagmaIntenDataset,
		"graff3": SpecMolAnnDataset,
		"precursor": SpecMolDataset
	}

	model_to_run_id = NestedDefaultDict()
	model_to_config = NestedDefaultDict()

	# find wandb checkpoints
	template_fp = "config/template.yml"
	api = wandb.Api()
	for split_type in split_types:
		for model_type in wandb_model_types:
			dset = args.dset
			model_to_config[split_type][model_type] = f"config/{dset}_d4_wr00_{split_type}/{model_type}/s0.yml"
			if model_type != "precursor":
				group_name = f"{dset}_{split_type}_{model_type}"
				runs = api.runs(
					path="frag-gnn/frag-gnn",
					filters={"group":group_name})
				for run in runs:
					seed = int(run.name.split("_")[-1].removeprefix("s"))
					if seed < num_seeds:
						model_to_run_id[split_type][model_type][seed] = run.id
	del api
	pprint(model_to_config)
	pprint(model_to_run_id)

	# find local checkpoints
	model_to_ckpt = NestedDefaultDict()
	for split_type in split_types:
		for model_type in local_model_types:
			dset = args.dset
			model_to_config[split_type][model_type] = f"config/{dset}_d4_wr00_{split_type}/{model_type}/s0.yml"
			if model_type != "precursor":
				for seed in range(num_seeds):
					ckpt = glob(os.path.join(local_path,f"{model_type}/{split_type}/ckpts/s{seed}/*.ckpt"))[0]
					model_to_ckpt[split_type][model_type][seed] = ckpt
	pprint(model_to_config)
	pprint(model_to_ckpt)

	model_types = wandb_model_types + local_model_types

	for split_type in split_types:
		for model_type in model_types:
			print(f">>> Starting model_type {model_type}")
			custom_fp = model_to_config[split_type][model_type]
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
			# modify config for formula strings
			config_d["output_formula_str"] = True
			if "fraggnn" in model_type:
				config_d["spec_params"]["prec_type_str"] = True
				config_d["frag_params"]["formula_str"] = True
			elif "iceberg" in model_type:
				config_d["magma_params"]["adduct_form_deltas"] = True
			elif "graff" in model_type:
				config_d["ann_params"]["formula_str"] = True
			print(">>> Loading dataset")
			ds_cls = model_type_to_ds_cls[model_type]
			ds = ds_cls(split=eval_split, **config_d)
			dl = init_dataloader(ds, config_d)
			model_cls = model_type_to_model_cls[model_type]
			device = th.device("cuda:0") if config_d["accelerator"]=="gpu" else th.device("cpu")
			for seed in range(num_seeds):
				print(f">>> Starting seed {seed}")
				if model_type in model_to_run_id[split_type]:
					assert model_type in wandb_model_types, model_type
					run_id = model_to_run_id[split_type][model_type][seed]
					model = init_wandb_ckpt(run_id,last_ckpt=False,model_cls=model_cls,config_d=config_d)
				elif model_type in model_to_ckpt:
					assert model_type in local_model_types, model_type
					ckpt_fp = model_to_ckpt[split_type][model_type][seed]
					try:
						model = model_cls.load_from_checkpoint(ckpt_fp,strict=True,**config_d)
					except RuntimeError as e:
						print(f"> error when loading from checkpoint file {ckpt_fp}, will try with strict=False")
						model = model_cls.load_from_checkpoint(ckpt_fp,strict=False,**config_d)
				else:
					assert model_type == "precursor", model_type
					# does not need a checkpoint
					model = model_cls(**config_d)
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
				seed_vals_dp = os.path.join(args.vals_dp,split_type,model_type,f"s{seed}")
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
