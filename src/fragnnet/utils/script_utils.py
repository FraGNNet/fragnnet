import torch as th
import numpy as np
from tqdm import tqdm
import tempfile
import os

from fragnnet.utils.misc_utils import to_device, scatter_reduce, flatten_lol, LOG_ZERO, check_wandb
from fragnnet.pl_model import FragGNNPL
from fragnnet.iceberg.pl_model import IcebergIntenPL
from fragnnet.utils.nn_utils import decompile_jit_ckpt

def init_wandb_ckpt(run_id,last_ckpt,model_cls,config_d):
	import wandb
	# load model
	with tempfile.TemporaryDirectory(dir=os.getcwd()) as temp_dp:
		api = wandb.Api()
		run = api.run(f"frag-gnn/frag-gnn/{run_id}")
		if last_ckpt:
			ckpt_wandb_fp = "ckpt/last.ckpt"
		else:
			files = run.files(per_page=100)
			best_ckpt_wandb_fp, best_epoch = None, None
			for file in files:
				if file.name.endswith(".ckpt") and "last" not in file.name:
					ckpt_wandb_fp = file.name
					epoch = int(os.path.basename(file.name).removeprefix("model-epoch=").removesuffix(".ckpt"))
					if best_ckpt_wandb_fp is None or epoch > best_epoch:
						best_ckpt_wandb_fp = ckpt_wandb_fp
						best_epoch = epoch
		assert ckpt_wandb_fp is not None
		ckpt_file_local = run.file(ckpt_wandb_fp).download(root=temp_dp,replace=False)
		ckpt_fp = ckpt_file_local.name
		ckpt = th.load(ckpt_fp, map_location="cpu")
		# decompile
		assert not config_d["compile"]
		ckpt = decompile_jit_ckpt(ckpt)
		# init
		model = model_cls(**config_d)
		try:
			model.load_state_dict(ckpt["state_dict"], strict=True)
		except RuntimeError as e:
			print(f"> error when loading from checkpoint id {run_id}, will try with strict=False")
			model.load_state_dict(ckpt["state_dict"], strict=False)
	return model


def run_inference(dl, model, device, eval_split, batch_cutoff, nb_iso, output_subset=None, untransform_spec=False):
	
	# make predictions on validation data
	print(">>> Iterating over data")
	cum_num_datapoints = 0
	cum_num_nodes = 0
	cum_num_formulas = 0
	cum_num_nb_nodes = 0
	fraggnn_model = isinstance(model, FragGNNPL)
	iceberg_model = isinstance(model, IcebergIntenPL)
	if not (fraggnn_model or iceberg_model):
		assert not nb_iso
		BATCH_IDX_VARS = [
			"pred_batch_idxs",
			"true_batch_idxs"
		]
		NODE_IDX_VARS = []
		FORMULA_IDX_VARS = []
		NB_NODE_IDX_VARS = []
	elif iceberg_model:
		assert not nb_iso
		BATCH_IDX_VARS = [
			"pred_batch_idxs",
			"true_batch_idxs",
		]
		if model.model.output_formula_str:
			BATCH_IDX_VARS.extend([
				"pred_formula_batch_idxs"
			])
		NODE_IDX_VARS = []
		FORMULA_IDX_VARS = []
		NB_NODE_IDX_VARS = []
	else:
		BATCH_IDX_VARS = [
			"pred_batch_idxs",
			"true_batch_idxs",
			"pred_formula_batch_idxs",
			"pred_node_batch_idxs",
			"pred_joint_batch_idxs"
		]
		NODE_IDX_VARS = [
			"pred_node_node_idxs",
			"pred_joint_node_idxs"
		]
		FORMULA_IDX_VARS = [
			"pred_formula_formula_idxs",
			"pred_joint_formula_idxs"
		]
		NB_NODE_IDX_VARS = []
		if nb_iso:
			BATCH_IDX_VARS.extend([
				"pred_nb_node_batch_idxs",
				"pred_nb_joint_batch_idxs",
				"pred_nb_node_node_batch_idxs"
			])
			FORMULA_IDX_VARS.extend([
				"pred_nb_joint_formula_idxs"
			])
			NB_NODE_IDX_VARS.extend([
				"pred_nb_node_node_idxs",
				"pred_nb_joint_node_idxs",
				"pred_nb_node_node_node_idxs"
			])
	if output_subset is not None:
		BATCH_IDX_VARS = [k for k in BATCH_IDX_VARS if k in output_subset]
		NODE_IDX_VARS = [k for k in NODE_IDX_VARS if k in output_subset]
		FORMULA_IDX_VARS = [k for k in FORMULA_IDX_VARS if k in output_subset]
		NB_NODE_IDX_VARS = [k for k in NB_NODE_IDX_VARS if k in output_subset]
	vals = {}
	if model.training:
		print(f"> Warning: model {model} is in training mode!")
	with th.inference_mode(): 
		for b_idx, b_input in tqdm(enumerate(dl),total=min(len(dl),batch_cutoff)):
			# transfer data, make predictions
			b_input = to_device(b_input, device,non_blocking=False)
			b_output = model.inference_step(b_input, split=eval_split, untransform_spec=untransform_spec)
			b_output = to_device(b_output, "cpu",non_blocking=False)
			if output_subset is not None:
				b_output = {k: b_output[k] for k in b_output if k in output_subset}
			output_keys = b_output.keys()
			# rebatch stuff
			if "pred_batch_idxs" in output_keys:
				b_num_datapoints = b_output["pred_batch_idxs"].max()+1
			elif "true_batch_idxs" in output_keys:
				b_num_datapoints = b_output["true_batch_idxs"].max()+1
			else:
				b_num_datapoints = 0
			if fraggnn_model and "pred_node_node_idxs" in output_keys:
				b_num_nodes = b_output["pred_node_node_idxs"].max()+1
			else:
				b_num_nodes = 0
			if fraggnn_model and "pred_formula_formula_idxs" in output_keys:
				b_num_formulas = b_output["pred_formula_formula_idxs"].max()+1
			else:
				b_num_formulas = 0
			if fraggnn_model and nb_iso and "pred_nb_node_node_idxs" in output_keys:
				b_num_nb_nodes = b_output["pred_nb_node_node_idxs"].max()+1
			else:
				b_num_nb_nodes = 0
			for k in BATCH_IDX_VARS:
				b_output[k] = b_output[k] + cum_num_datapoints
			for k in NODE_IDX_VARS:
				b_output[k] = b_output[k] + cum_num_nodes
			for k in FORMULA_IDX_VARS:
				b_output[k] = b_output[k] + cum_num_formulas
			for k in NB_NODE_IDX_VARS:
				b_output[k] = b_output[k] + cum_num_nb_nodes
			# update counts
			cum_num_datapoints += b_num_datapoints
			cum_num_nodes += b_num_nodes
			cum_num_formulas += b_num_formulas
			cum_num_nb_nodes += b_num_nb_nodes
			# update dict
			for k in b_output.keys():
				if b_output[k] is None:
					continue
				if k not in vals.keys():
					assert b_idx == 0
					vals[k] = [b_output[k]]
				else:
					assert b_idx > 0
					vals[k].append(b_output[k])
			# early exit
			if b_idx >= batch_cutoff:
				break
	print(f"> num_datapoints = {cum_num_datapoints}")
	print(f"> num_nodes = {cum_num_nodes}")
	print(f"> num_formulas = {cum_num_formulas}")
	print(f"> num_nb_nodes = {cum_num_nb_nodes}")
	for k in vals.keys():
		if isinstance(vals[k][0], list):
			vals[k] = flatten_lol(vals[k])
		elif isinstance(vals[k][0], th.Tensor):
			if vals[k][0].dim() == 0:
				vals[k] = th.stack(vals[k], dim=0)
			else:
				vals[k] = th.cat(vals[k], dim=0)
		elif isinstance(vals[k][0], np.ndarray):
			if vals[k][0].ndim == 0:
				vals[k] = np.stack(vals[k], axis=0)
			else:
				vals[k] = np.concatenate(vals[k], axis=0)
		else:
			raise ValueError(f"unexpected type {type(vals[k][0])}")
	return vals

def select_model_vals(model_to_vals, keys, stack_dim=0):

	select_d = {k: [] for k in keys}
	for model, vals in model_to_vals.items():
		for k in keys:
			select_d[k].append(vals[k])
	select_d = {k: th.stack(v, dim=stack_dim) for k,v in select_d.items()}
	return select_d

def log_mean(log_p,dim=0):
	return th.logsumexp(log_p,dim=dim) - np.log(log_p.shape[dim])

def log_std(log_p,dim=0,correction=1):
	support_size = log_p.shape[dim]
	log_mean_p = log_mean(log_p,dim=dim)
	log_mean_p_sq = log_mean(2*log_p,dim=dim)
	log_var = th.log(th.exp(log_mean_p_sq) - th.exp(2*log_mean_p))
	return 0.5*(log_var + np.log(support_size) - np.log(support_size - correction))

# define sparse kl divergence
def sparse_kl(log_p,log_q,b_idx,dim_size):
	p = th.exp(log_p)
	kl = scatter_reduce(
		p * (log_p - th.clamp(log_q,min=LOG_ZERO(log_p.dtype))),
		b_idx,
		reduce="sum",
		dim_size=dim_size
	)
	kl = th.clamp(kl,min=0.0)
	return kl

def pearson_r(x,y,dim=0):
	x_mean = th.mean(x,dim=dim)
	y_mean = th.mean(y,dim=dim)
	x_std = th.std(x,dim=dim)
	y_std = th.std(y,dim=dim)
	r = th.mean((x-x_mean)*(y-y_mean),dim=dim) / (x_std * y_std)
	return r