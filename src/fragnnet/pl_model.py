import torch as th
import numpy as np
import logging
import inspect

# make cc happy
try:
	import lightning.pytorch as pl
	from lightning.fabric.utilities.seed import _collect_rng_states, _set_rng_states
except ModuleNotFoundError:
	import pytorch_lightning as pl
	from pytorch_lightning.utilities.seed import _collect_rng_states, _set_rng_states
	
import torch._dynamo as th_dynamo

from fragnnet.utils.nn_utils import build_lr_scheduler
from fragnnet.model import FragGNNModel, NeimsModel, PrecursorModel, GNNModel
from fragnnet.loss import get_sparse_cross_entropy_fn, sparse_entropy_fn, sparse_conditional_entropy_fn, get_edge_loss_fn, sparse_cosine_distance_hungarian, sparse_cosine_distance_binned
from fragnnet.utils.spec_utils import *
from fragnnet.utils.misc_utils import safelog, flatten_lol, th_temp_seed, scatter_logsumexp, to_cpu, scatter_l1normalize, LOG_ZERO
from fragnnet.utils.plot_utils import plot_spectra_sparse


class SpectrumPL(pl.LightningModule):

	def __init__(self,**kwargs):

		super().__init__()
		self.save_hyperparameters()
		# setup functions
		self.metric_names = set()
		self._setup_model()
		self._setup_tolerance()
		self._setup_loss_fn()
		self._setup_spec_fns()
		self._setup_metric_fns()
		self._setup_batch_metric_reduce_fns()
		self._setup_result_trackers()
		self._setup_sampler()

	def _setup_model(self):
		raise NotImplementedError

	def _setup_tolerance(self):

		# set tolerance
		if self.hparams.loss_tolerance_rel is not None:
			self.tolerance = self.hparams.loss_tolerance_rel
			self.relative = True
			self.tolerance_min_mz = self.hparams.loss_tolerance_min_mz
		else:
			assert self.hparams.loss_tolerance_abs is not None
			self.tolerance = self.hparams.loss_tolerance_abs
			self.relative = False
			self.tolerance_min_mz = None

	def _setup_loss_fn(self):
		raise NotImplementedError

	def _setup_spec_fns(self):

		def _filter_func(mzs, ints, batch_idxs):
			return batched_filter_func(mzs,ints,batch_idxs,self.hparams.ints_thresh,self.hparams.mz_max)
		self.filter_func = _filter_func
		def _bin_func(mzs, ints, batch_idxs):
			agg = "sum" if self.hparams.sum_ints else "amax"
			bin_idxs, bin_ints, bin_batch_idxs = batched_bin_func(
				mzs,
				ints,
				batch_idxs,
				self.hparams.mz_max,
				self.hparams.mz_bin_res,
				agg,
				sparse=True)
			return bin_idxs, bin_ints, bin_batch_idxs
		self.bin_func = _bin_func
		self.ints_transform_func = get_ints_transform_func(self.hparams.ints_transform)
		self.ints_untransform_func = get_ints_untransform_func(self.hparams.ints_transform)
		self.ints_normalize_func = batched_l1_normalize

	def _setup_metric_fns(self):

		# setup metrics
		self.metric_min_max = {"loss": "min"}
		self.metric_bests = ["loss"]
		for k in self.metric_names:
			if k not in self.metric_min_max:
				self.metric_min_max[k] = None
		self.auxiliary_metric_names = set()

		bin_sqrt_flags = [False]
		if self.hparams.eval_bin_sqrt:
			bin_sqrt_flags.append(True)
		bin_remove_prec_peak_flags = [False]
		if self.hparams.eval_bin_remove_prec_peaks:
			bin_remove_prec_peak_flags.append(True)
		hun_sqrt_flags = [False]
		if self.hparams.eval_hun_sqrt:
			hun_sqrt_flags.append(True)
		hun_remove_prec_peak_flags = [False]
		if self.hparams.eval_hun_remove_prec_peaks:
			hun_remove_prec_peak_flags.append(True)
		nb_flags = [False]
		if self.hparams.nb_iso:
			nb_flags.append(True)
		self.extra_metric_args = []

		eval_mz_bin_reses = self.hparams.eval_mz_bin_res

		if len(self.hparams.auxiliary_scores) > 0:
			assert self.hparams.sparse_cosine_similarity
		compute_match_mzs = False
		compute_rounded_match_mzs = False
		compute_bin_specs = False
		compute_entropies = False

		for metric_name in self.hparams.auxiliary_scores:
			if metric_name == "cos_sim":
				for remove_prec_peak in bin_remove_prec_peak_flags:
					for sqrt in bin_sqrt_flags:
						for mz_bin_res in eval_mz_bin_reses:
							fname_base = "cos_sim"
							if sqrt:
								fname_base += "_sqrt"
							if remove_prec_peak:
								fname_base += "_np"
							fname = f"{fname_base}_{mz_bin_res}"
							self.metric_min_max[fname] = "max"
							self.metric_bests.append(fname)
							self.auxiliary_metric_names.add(fname)
					compute_bin_specs = True
			elif metric_name == "jss":
				for remove_prec_peak in bin_remove_prec_peak_flags:
					for sqrt in bin_sqrt_flags:
						for mz_bin_res in eval_mz_bin_reses:
							fname_base = "jss"
							if sqrt:
								fname_base += "_sqrt"
							if remove_prec_peak:
								fname_base += "_np"
							fname = f"{fname_base}_{mz_bin_res}"
							self.metric_min_max[fname] = "max"
							self.metric_bests.append(fname)
							self.auxiliary_metric_names.add(fname)
			elif metric_name == "recall":
				self.metric_min_max["recall"] = "max"
				self.auxiliary_metric_names.add("recall")
				compute_match_mzs = True
			elif metric_name == "wrecall":
				self.metric_min_max["wrecall"] = "max"
				self.auxiliary_metric_names.add("wrecall")
				compute_match_mzs = True
			elif metric_name == "opt_cos_sim":
				for remove_prec_peak in bin_remove_prec_peak_flags:
					for sqrt in bin_sqrt_flags:
						for mz_bin_res in eval_mz_bin_reses:
							fname_base = "opt_cos_sim"
							if sqrt:
								fname_base += "_sqrt"
							if remove_prec_peak:
								fname_base += "_np"
							fname = f"{fname_base}_{mz_bin_res}"
							self.metric_min_max[fname] = None
							self.auxiliary_metric_names.add(fname)
					compute_bin_specs = True
			elif metric_name == "true_spec_e":
				self.metric_min_max["true_spec_e"] = None
				self.auxiliary_metric_names.add("true_spec_e")
				compute_entropies = True
			elif metric_name == "true_spec_ne":
				self.metric_min_max["true_spec_ne"] = None
				self.auxiliary_metric_names.add("true_spec_ne")
				compute_entropies = True
			elif metric_name == "cos_hun":
				for remove_prec_peak in hun_remove_prec_peak_flags:
					for sqrt in hun_sqrt_flags:
						fname_base = "cos_hun"
						if sqrt:
							fname_base += "_sqrt"
						if remove_prec_peak:
							fname_base += "_np"
						fname = f"{fname_base}"
						self.metric_min_max[fname] = "max"
						self.metric_bests.append(fname)
						self.auxiliary_metric_names.add(fname)
				compute_rounded_match_mzs = True
			elif metric_name == "precision":
				self.metric_min_max["precision"] = "max"
				self.metric_bests.append("precision")
				self.auxiliary_metric_names.add("precision")
				compute_match_mzs = True
			elif metric_name == "wprecision":
				self.metric_min_max["wprecision"] = "max"
				self.metric_bests.append("wprecision")
				self.auxiliary_metric_names.add("wprecision")
				compute_match_mzs = True
			# elif metric_name == "dice":
			# 	self.metric_min_max["dice"] = "max"
			# 	self.metric_bests.append("dice")
			# 	compute_match_mzs = True
			elif metric_name == "ndcg":
				for union in [True,False]:
					if union:
						fname_base = "ndcg_un"
						tie_break_flags = [True,False]
					else:
						fname_base = "ndcg_int"
						tie_break_flags = [True]
					for optimistic in tie_break_flags:
						if union:
							suffix = "_opt" if optimistic else "_pess"
						else:
							suffix = ""
						fname = f"{fname_base}{suffix}"
						self.metric_min_max[fname] = "max"
						self.metric_bests.append(fname)
						self.auxiliary_metric_names.add(fname)
				compute_rounded_match_mzs = True
			elif metric_name == "jss_hun":
				for remove_prec_peak in hun_remove_prec_peak_flags:
					for sqrt in hun_sqrt_flags:
						fname_base = "jss_hun"
						if sqrt:
							fname_base += "_sqrt"
						if remove_prec_peak:
							fname_base += "_np"
						fname = f"{fname_base}"
						self.metric_min_max[fname] = "max"
						self.metric_bests.append(fname)
						self.auxiliary_metric_names.add(fname)
				compute_rounded_match_mzs = True
			elif metric_name == "true_oos_prob":
				self.metric_min_max["true_oos_prob"] = None
				self.auxiliary_metric_names.add("true_oos_prob")
				compute_match_mzs = True
			elif metric_name == "true_oos_e":
				self.metric_min_max["true_oos_e"] = None
				self.auxiliary_metric_names.add("true_oos_e")
				compute_match_mzs = True
			elif metric_name == "pred_node_count":
				for nb_flag in nb_flags:
					if nb_flag:
						fname = "pred_nb_node_count"
						self.extra_metric_args.extend(["pred_nb_node_batch_idxs"])
					else:
						fname = "pred_node_count"
						self.extra_metric_args.extend(["pred_node_batch_idxs"])
					self.metric_min_max[fname] = None
					self.auxiliary_metric_names.add(fname)
			elif metric_name == "pred_formula_count":
				fname = "pred_formula_count"
				self.extra_metric_args.extend(["pred_formula_batch_idxs"])
				self.metric_min_max[fname] = None
				self.auxiliary_metric_names.add(fname)
			elif metric_name == "pred_edge_count":
				fname = "pred_edge_count"
				self.extra_metric_args.extend(["pred_edge_batch_idxs"])
				self.metric_min_max[fname] = None
				self.auxiliary_metric_names.add(fname)
			else:
				raise ValueError(f"metric_name {metric_name} not recognized")

		self.metric_names.update(self.auxiliary_metric_names)

		def calculate_all_auxiliary_metrics(
			true_mzs,
			true_ints,
			true_batch_idxs,
			pred_mzs,
			pred_ints,
			pred_batch_idxs,
			true_prec_mzs,
			**kwargs):

			# assumes both true/pred spectra are already L1-normalized

			batch_size = th.max(true_batch_idxs)+1
			assert batch_size == th.max(pred_batch_idxs)+1, (batch_size, th.max(pred_batch_idxs)+1)

			metric_d = {k: -th.ones([batch_size], dtype=true_ints.dtype, device=true_ints.device) for k in self.auxiliary_metric_names}

			# define aggregation
			agg = "sum" if self.hparams.sum_ints else "amax"

			# global calculations
			if compute_rounded_match_mzs:
				true_mzs_r, true_ints_r, true_batch_idxs_r = round_aggregate_peaks(
					true_mzs,
					true_ints,
					true_batch_idxs,
					agg=agg
				)
				pred_mzs_r, pred_ints_r, pred_batch_idxs_r = round_aggregate_peaks(
					pred_mzs,
					pred_ints,
					pred_batch_idxs,
					agg=agg
				)

			if compute_bin_specs:
				for remove_prec_peak in bin_remove_prec_peak_flags:
					for sqrt in bin_sqrt_flags:
						if sqrt:
							ints_transform = get_ints_transform_func("sqrt")
						else:
							ints_transform = get_ints_transform_func("none")
						for mz_bin_res in eval_mz_bin_reses:
							# bin, transform, normalize
							true_bin_idxs, true_bin_ints, true_bin_batch_idxs = batched_bin_func(
								true_mzs,
								true_ints,
								true_batch_idxs,
								mz_max=self.hparams.mz_max,
								mz_bin_res=mz_bin_res,
								agg=agg,
								sparse=True,
								remove_prec_peaks=remove_prec_peak,
								prec_mzs=true_prec_mzs
							)
							true_ints = scatter_l1normalize(ints_transform(true_ints), true_batch_idxs)
							pred_bin_idxs, pred_bin_ints, pred_bin_batch_idxs = batched_bin_func(
								pred_mzs,
								pred_ints,
								pred_batch_idxs,
								mz_max=self.hparams.mz_max,
								mz_bin_res=mz_bin_res,
								agg=agg,
								sparse=True,
								remove_prec_peaks=remove_prec_peak,
								prec_mzs=true_prec_mzs
							)
							pred_ints = scatter_l1normalize(ints_transform(pred_ints), pred_batch_idxs)
							if "cos_sim" in self.hparams.auxiliary_scores:
								fname_base = f"cos_sim"
								if sqrt:
									fname_base += "_sqrt"
								if remove_prec_peak:
									fname_base += "_np"
								fname = f"{fname_base}_{mz_bin_res}"
								cos_sim = cos_sim_helper(
									true_bin_idxs,
									true_bin_ints,
									true_bin_batch_idxs,
									pred_bin_idxs,
									pred_bin_ints,
									pred_bin_batch_idxs
								)
								# print(cos_sim)
								metric_d[fname] = cos_sim
							if "opt_cos_sim" in self.hparams.auxiliary_scores:
								fname_base = f"opt_cos_sim"
								if sqrt:
									fname_base += "_sqrt"
								if remove_prec_peak:
									fname_base += "_np"
								fname = f"{fname_base}_{mz_bin_res}"
								opt_cos_sim = opt_cos_sim_helper(
									true_bin_idxs,
									true_bin_ints,
									true_bin_batch_idxs,
									pred_bin_idxs,
									pred_bin_ints,
									pred_bin_batch_idxs,
								)
								metric_d[fname] = opt_cos_sim
							if "jss" in self.hparams.auxiliary_scores:
								fname_base = f"jss"
								if sqrt:
									fname_base += "_sqrt"
								if remove_prec_peak:
									fname_base += "_np"
								fname = f"{fname_base}_{mz_bin_res}"
								jss = jss_helper(
									true_bin_idxs,
									true_bin_ints,
									true_bin_batch_idxs,
									pred_bin_idxs,
									pred_bin_ints,
									pred_bin_batch_idxs,
									log_min=self.hparams.log_min
								)
								metric_d[fname] = jss

			if compute_entropies:
				true_log_probs = safelog(scatter_l1normalize(true_ints,true_batch_idxs), eps=self.hparams.log_min)
				true_spec_e, true_spec_ne = sparse_entropy_fn(true_log_probs,true_batch_idxs)
				if "true_spec_e" in self.hparams.auxiliary_scores:
					metric_d["true_spec_e"] = true_spec_e
				if "true_spec_ne" in self.hparams.auxiliary_scores:
					metric_d["true_spec_ne"] = true_spec_ne

			# local calculations
			for b_idx in range(batch_size):

				b_true_mask = (true_batch_idxs==b_idx)
				b_pred_mask = (pred_batch_idxs==b_idx)
				b_true_mzs = true_mzs[b_true_mask]
				b_pred_mzs = pred_mzs[b_pred_mask]
				b_true_ints = true_ints[b_true_mask]
				b_pred_ints = pred_ints[b_pred_mask]
				b_true_prec_mz = true_prec_mzs[b_idx:b_idx+1]

				if compute_match_mzs:
					b_match_mask = calculate_match_mzs(
						b_true_mzs,
						b_pred_mzs,
						tolerance=self.tolerance,
						relative=self.relative,
						tolerance_min_mz=self.tolerance_min_mz
					)
					b_true_match_mask = th.any(b_match_mask,dim=1)
					b_pred_match_mask = th.any(b_match_mask,dim=0)

				if compute_rounded_match_mzs:
					b_true_mask_r = (true_batch_idxs_r==b_idx)
					b_pred_mask_r = (pred_batch_idxs_r==b_idx)
					b_true_mzs_r = true_mzs_r[b_true_mask_r]
					b_pred_mzs_r = pred_mzs_r[b_pred_mask_r]
					b_true_ints_r = true_ints_r[b_true_mask_r]
					b_pred_ints_r = pred_ints_r[b_pred_mask_r]
					# precursor
					b_true_prec_mask_r = calculate_match_mzs(
						b_true_mzs_r,
						b_true_prec_mz,
						tolerance=self.tolerance,
						relative=self.relative,
						tolerance_min_mz=self.tolerance_min_mz
					).squeeze(1)
					b_pred_prec_mask_r = calculate_match_mzs(
						b_pred_mzs_r,
						b_true_prec_mz,
						tolerance=self.tolerance,
						relative=self.relative,
						tolerance_min_mz=self.tolerance_min_mz
					).squeeze(1)
					# match
					b_match_mask_r = calculate_match_mzs(
						b_true_mzs_r,
						b_pred_mzs_r,
						tolerance=self.tolerance,
						relative=self.relative,
						tolerance_min_mz=self.tolerance_min_mz
					)
					b_true_match_mask_r = th.any(b_match_mask_r,dim=1)
					b_pred_match_mask_r = th.any(b_match_mask_r,dim=0)
					

				if "recall" in self.hparams.auxiliary_scores:
					b_recall = th.sum(b_true_match_mask.float()) / b_true_match_mask.shape[0]
					metric_d["recall"][b_idx] = b_recall
								
				if "wrecall" in self.hparams.auxiliary_scores:
					b_wrecall = th.sum(b_true_ints[b_true_match_mask]) #/ th.sum(b_true_ints)
					metric_d["wrecall"][b_idx] = b_wrecall

				if "precision" in self.hparams.auxiliary_scores:
					b_precision = th.sum(b_pred_match_mask.float()) / b_pred_match_mask.shape[0]
					metric_d["precision"][b_idx] = b_precision

				if "wprecision" in self.hparams.auxiliary_scores:
					b_wprecision = th.sum(b_pred_ints[b_pred_match_mask])
					metric_d["wprecision"][b_idx] = b_wprecision

				if "cos_hun" in self.hparams.auxiliary_scores:
					for remove_prec_peak in hun_remove_prec_peak_flags:
						for sqrt in hun_sqrt_flags:
							if sqrt:
								ints_transform = get_ints_transform_func("sqrt")
							else:
								ints_transform = get_ints_transform_func("none")
							b_cos_hun = cos_hun_helper(
								ints_transform(b_true_ints_r),
								ints_transform(b_pred_ints_r),
								b_match_mask_r,
								b_true_match_mask_r,
								b_pred_match_mask_r,
								remove_prec_peak,
								b_true_prec_mask_r,
								b_pred_prec_mask_r
							)
							fname_base = f"cos_hun"
							if sqrt:
								fname_base += "_sqrt"
							if remove_prec_peak:
								fname_base += "_np"
							fname = f"{fname_base}"
							metric_d[fname][b_idx] = b_cos_hun

				if "ndcg" in self.hparams.auxiliary_scores:
					for union in [True,False]:
						if union:
							fname_base = "ndcg_un"
							tie_break_flags = [True,False]
						else:
							fname_base = "ndcg_int"
							tie_break_flags = [True]
						for optimistic in tie_break_flags:
							if union:
								suffix = "_opt" if optimistic else "_pess"
							else:
								suffix = ""
							b_ndcg = ndcg_helper(
								b_true_ints,
								b_pred_ints,
								b_match_mask,
								b_true_match_mask,
								b_pred_match_mask,
								optimistic,
								union
							)
							fname = f"{fname_base}{suffix}"
							metric_d[fname][b_idx] = b_ndcg

				if "jss_hun" in self.hparams.auxiliary_scores:
					for remove_prec_peak in hun_remove_prec_peak_flags:
						for sqrt in hun_sqrt_flags:
							if sqrt:
								ints_transform = get_ints_transform_func("sqrt")
							else:
								ints_transform = get_ints_transform_func("none")
							b_jss_hun = jss_hun_helper(
								ints_transform(b_true_ints_r),
								ints_transform(b_pred_ints_r),
								b_match_mask_r,
								b_true_match_mask_r,
								b_pred_match_mask_r,
								remove_prec_peak,
								b_true_prec_mask_r,
								b_pred_prec_mask_r,
								log_min=self.hparams.log_min
							)
							fname_base = f"jss_hun"
							if sqrt:
								fname_base += "_sqrt"
							if remove_prec_peak:
								fname_base += "_np"
							fname = f"{fname_base}"
							metric_d[fname][b_idx] = b_jss_hun

				if "true_oos_prob" in self.hparams.auxiliary_scores:
					
					b_true_oos_prob = th.sum(b_true_ints[~b_true_match_mask]) / th.sum(b_true_ints)
					metric_d["true_oos_prob"][b_idx] = b_true_oos_prob

				if "true_oos_e" in self.hparams.auxiliary_scores:

					b_true_oos_probs = b_true_ints[~b_true_match_mask] / th.sum(b_true_ints[~b_true_match_mask])
					b_true_oos_logprobs = safelog(b_true_oos_probs, eps=self.hparams.log_min)
					b_true_oos_e = -th.sum(b_true_oos_probs * b_true_oos_logprobs)
					metric_d["true_oos_e"][b_idx] = b_true_oos_e

			if "pred_node_count" in self.hparams.auxiliary_scores:

				for nb_flag in nb_flags:
					if nb_flag:
						key = "nb_node"
					else:
						key = "node"
					b_pred_node_count = scatter_reduce(
						th.ones_like(kwargs[f"pred_{key}_batch_idxs"]),
						kwargs[f"pred_{key}_batch_idxs"],
						reduce="sum",
						dim=0
					)
					assert th.min(b_pred_node_count) > 0, b_pred_node_count
					metric_d[f"pred_{key}_count"] = b_pred_node_count

			if "pred_formula_count" in self.hparams.auxiliary_scores:

				b_pred_formula_count = scatter_reduce(
					th.ones_like(kwargs["pred_formula_batch_idxs"]),
					kwargs["pred_formula_batch_idxs"],
					reduce="sum",
					dim=0
				)
				b_pred_formula_count = b_pred_formula_count - 1  # -1 for OOS!
				assert th.min(b_pred_formula_count) > 0, b_pred_formula_count
				metric_d["pred_formula_count"] = b_pred_formula_count

			if "pred_edge_count" in self.hparams.auxiliary_scores:

				b_pred_edge_count = scatter_reduce(
					th.ones_like(kwargs["pred_edge_batch_idxs"]),
					kwargs["pred_edge_batch_idxs"],
					reduce="sum",
					dim=0
				)
				assert th.min(b_pred_edge_count) > 0, b_pred_edge_count
				metric_d["pred_edge_count"] = b_pred_edge_count

			return metric_d
	
		self.metric_fn = calculate_all_auxiliary_metrics

	def _get_batch_metric_reduce_fn(self, sample_weight):

		if sample_weight == "none":
			calc_sample_weights = lambda spec_per_group, spec_per_mol, group_per_mol: th.ones_like(spec_per_group, dtype=th.float32)
		elif sample_weight == "group":
			calc_sample_weights = lambda spec_per_group, spec_per_mol, group_per_mol: 1. / spec_per_group
		elif sample_weight == "mol":
			calc_sample_weights = lambda spec_per_group, spec_per_mol, group_per_mol: 1. / spec_per_mol
		elif sample_weight == "group_mol":
			calc_sample_weights = lambda spec_per_group, spec_per_mol, group_per_mol: 1. / (spec_per_group*group_per_mol)
		def _batch_metric_reduce(b_metric, b_spec_per_group, b_spec_per_mol, b_group_per_mol, reduce, return_weights=False):
			b_sample_weight = calc_sample_weights(b_spec_per_group, b_spec_per_mol, b_group_per_mol)
			b_total_weight = th.sum(b_sample_weight, dim=0)
			if reduce == "w_mean":
				b_reduce_metric = th.sum(b_sample_weight * b_metric, dim=0) / b_total_weight
			elif reduce == "w_std":
				b_reduce_metric = th.sqrt(
					th.sum(b_sample_weight * (b_metric - th.sum(b_sample_weight * b_metric, dim=0) / b_total_weight)**2, dim=0) / b_total_weight
				)
			else:
				assert reduce == "w_sum", reduce
				b_reduce_metric = th.sum(b_sample_weight * b_metric, dim=0)
			if return_weights:
				return b_reduce_metric, b_total_weight
			else:
				return b_reduce_metric
		return _batch_metric_reduce

	def _setup_batch_metric_reduce_fns(self):
		
		self.train_batch_metric_reduce_fn = self._get_batch_metric_reduce_fn(self.hparams.train_sample_weight)
		self.eval_batch_metric_reduce_fn = self._get_batch_metric_reduce_fn(self.hparams.eval_sample_weight)
		def _batch_metric_reduce_fn(split,**kwargs):
			if split == "train":
				return self.train_batch_metric_reduce_fn(**kwargs)
			else:
				return self.eval_batch_metric_reduce_fn(**kwargs)
		self.batch_metric_reduce_fn = _batch_metric_reduce_fn

	def _setup_result_trackers(self):

		if self.hparams.spec_params["merge"]:
			max_num_datapoints = 22000
		else:
			max_num_datapoints = 270000
		for split in ["train","val","test"]:
			setattr(self,f"{split}_results",None)
			setattr(self,f"{split}_counter",0)
			setattr(self,f"{split}_mean_metrics",{})
			setattr(self,f"{split}_std_metrics",{})
			if self.hparams.track_datapoint_metrics:
				setattr(self,f"{split}_datapoint_metrics",{})
				setattr(self,f"{split}_num_datapoints",-th.ones([1],dtype=th.int64))
			for name in self.metric_names:
				_name = name.replace(".","-")
				mean_metrics = getattr(self,f"{split}_mean_metrics")
				std_metrics = getattr(self,f"{split}_std_metrics")
				if split != "test":
					mean_metrics[_name] = -th.ones([self.hparams.max_epochs],dtype=th.float32)
					std_metrics[_name] = -th.ones([self.hparams.max_epochs],dtype=th.float32)
				else:
					mean_metrics[_name] = -th.ones([1],dtype=th.float32)
					std_metrics[_name] = -th.ones([1],dtype=th.float32)
				if self.hparams.track_datapoint_metrics:
					datapoint_metrics = getattr(self,f"{split}_datapoint_metrics")
					datapoint_metrics[_name] = -th.ones([max_num_datapoints],dtype=th.float32)

	def _setup_sampler(self):

		self._cur_batch_size = 0
		self._cur_batch_weight = 0.
		self._max_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches
		self.automatic_optimization = self.hparams.automatic_optimization
		train_dl_generator = th.Generator()
		train_dl_generator.manual_seed(self.hparams.seed)
		self.train_dl_seeds = th.randint(
			low=0,
			high=2**32-1,
			size=[self.hparams.max_epochs+1],
			generator=train_dl_generator)

	def _preproc_spec(self,spec_mzs,spec_ints,spec_batch_idxs,filter_spec=False,bin_spec=False,transform_spec=False,normalize_spec=False):
		# assumes spec_ints are not logged, or transformed in any way
		# does not assume any particular kind of normalization

		if filter_spec:
			# filter
			spec_mzs, spec_ints, spec_batch_idxs = self.filter_func(
				spec_mzs,
				spec_ints,
				spec_batch_idxs
			)
		if bin_spec:
			# bin
			spec_mzs, spec_ints, spec_batch_idxs = self.bin_func(
				spec_mzs,
				spec_ints,
				spec_batch_idxs
			)
		if transform_spec:
			# normalize to mf1000
			spec_ints = batched_mf1000_normalize(spec_ints, spec_batch_idxs)
			# transform
			spec_ints = self.ints_transform_func(spec_ints)
		if normalize_spec:
			# renormalize
			spec_ints = self.ints_normalize_func(spec_ints, spec_batch_idxs)
		return spec_mzs, spec_ints, spec_batch_idxs

	def preproc_spec(
		self,
		spec_mzs,
		spec_ints,
		spec_batch_idxs,
		train: bool,
		pred: bool,
		log_in: bool = False,
		log_out: bool = False
	):
		if log_in:
			spec_ints = spec_ints.exp()
		if train and pred:
			# if you normalize here, you would mess up some losses (i.e. OOS)
			spec_mzs, spec_ints, spec_batch_idxs = self._preproc_spec(
				spec_mzs,
				spec_ints,
				spec_batch_idxs,
				filter_spec=False,
				bin_spec=self.binned_loss,
				transform_spec=False,
				normalize_spec=False
			)
		elif train and (not pred):
			spec_mzs, spec_ints, spec_batch_idxs = self._preproc_spec(
				spec_mzs,
				spec_ints,
				spec_batch_idxs,
				filter_spec=True,
				bin_spec=self.binned_loss,
				transform_spec=True,
				normalize_spec=True
			)
		elif (not train) and pred:
			# untransform
			spec_ints = self.ints_untransform_func(spec_ints, spec_batch_idxs)
			# normalize (note that this messes up OOS stuff, which is fine for eval...)
			spec_ints = self.ints_normalize_func(spec_ints, spec_batch_idxs)
		elif (not train) and (not pred):
			spec_mzs, spec_ints, spec_batch_idxs =  self._preproc_spec(
				spec_mzs,
				spec_ints,
				spec_batch_idxs,
				filter_spec=True,
				bin_spec=False,
				transform_spec=False,
				normalize_spec=True
			)
		if log_out:
			spec_ints = safelog(spec_ints, eps=self.hparams.log_min)
		return spec_mzs, spec_ints, spec_batch_idxs

	def predict_step(self,**batch_kwargs):
		
		return self.forward(**batch_kwargs)

	def forward(self,**batch_kwargs):

		# get predictions
		if self.hparams.activation_checkpointing:
			forward_keys = list(inspect.signature(self.model.forward).parameters.keys())
			forward_keys.remove("kwargs")
			forward_args = [batch_kwargs.get(k,None) for k in forward_keys]
			pred = th.utils.checkpoint.checkpoint(
				self.model.forward,
				*forward_args,
				use_reentrant=False)
		else:
			pred = self.model.forward(**batch_kwargs)
		return pred

	def _common_step(self, batch, split="train", log=True):

		# preprocess spec
		batch_size = batch["batch_size"]
		unique_id = batch["spec_unique_id"]
		smiles = batch["mol_smiles"]
		true_mzs = batch["spec_mzs"]
		true_ints = batch["spec_ints"]
		true_batch_idxs = batch["spec_batch_idxs"]
		true_prec_mzs = batch["spec_prec_mz"]
		spec_per_group = batch["spec_per_group"]
		spec_per_mol = batch["spec_per_mol"]
		group_per_mol = batch["group_per_mol"]
		pred_d = self.forward(**batch)
		pred_mzs = pred_d.pop("pred_mzs")
		pred_logprobs = pred_d.pop("pred_logprobs")
		pred_batch_idxs = pred_d.pop("pred_batch_idxs")
		pred_oos_logprobs = pred_d.pop("pred_oos_logprobs",None)
		# prepare training dict (TODO: do preprocessing in log space)
		train_true_mzs, train_true_logprobs, train_true_batch_idxs = self.preproc_spec(
			true_mzs,
			true_ints,
			true_batch_idxs,
			train=True,
			pred=False,
			log_in=False,
			log_out=True
		)
		train_pred_mzs, train_pred_logprobs, train_pred_batch_idxs = self.preproc_spec(
			pred_mzs,
			pred_logprobs,
			pred_batch_idxs,
			train=True,
			pred=True,
			log_in=True,
			log_out=True
		)
		train_d = {
			"true_mzs": train_true_mzs,
			"true_logprobs": train_true_logprobs,
			"true_batch_idxs": train_true_batch_idxs,
			"pred_mzs": train_pred_mzs,
			"pred_logprobs": train_pred_logprobs,
			"pred_batch_idxs": train_pred_batch_idxs,
			"pred_oos_logprobs": pred_oos_logprobs,
			**pred_d
		}
		loss_d = self.loss_fn(**train_d)
		loss = loss_d["loss"]
		mean_loss = self.batch_metric_reduce_fn(
			b_metric=loss,
			b_spec_per_group=spec_per_group,
			b_spec_per_mol=spec_per_mol,
			b_group_per_mol=group_per_mol,
			reduce="w_mean",
			split=split
		)
		total_loss, total_weight = self.batch_metric_reduce_fn(
			b_metric=loss,
			b_spec_per_group=spec_per_group,
			b_spec_per_mol=spec_per_mol,
			b_group_per_mol=group_per_mol,
			reduce="w_sum",
			return_weights=True,
			split=split
		)
		# prepare eval dict
		eval_true_mzs, eval_true_probs, eval_true_batch_idxs = self.preproc_spec(
			true_mzs,
			true_ints,
			true_batch_idxs,
			train=False,
			pred=False,
			log_in=False,
			log_out=False
		)
		eval_true_logprobs = safelog(eval_true_probs, eps=self.hparams.log_min)
		eval_pred_mzs, eval_pred_probs, eval_pred_batch_idxs = self.preproc_spec(
			pred_mzs,
			pred_logprobs,
			pred_batch_idxs,
			train=False,
			pred=True,
			log_in=True,
			log_out=False
		)
		eval_pred_logprobs = safelog(eval_pred_probs, eps=self.hparams.log_min)
		both_d = {
			"true_mzs": eval_true_mzs,
			"true_logprobs": eval_true_logprobs,
			"true_batch_idxs": eval_true_batch_idxs,
			"true_prec_mzs": true_prec_mzs,
			"pred_mzs": eval_pred_mzs,
			"pred_logprobs": eval_pred_logprobs,
			"pred_batch_idxs": eval_pred_batch_idxs,
			**pred_d
		}
		# just log loss
		if log:
			self.log(
				f"{split}_batch_loss",
				mean_loss,
				batch_size=batch_size,
				on_epoch=True
			)
		results = {
			"unique_id": unique_id,
			"smiles": smiles,
			"mean_loss": mean_loss,
			"total_loss": total_loss,
			"total_weight": total_weight,
			"spec_per_group": spec_per_group,
			"spec_per_mol": spec_per_mol,
			"group_per_mol": group_per_mol,
			**both_d,
			**loss_d
		}
		assert true_mzs.shape[0] > 0, true_mzs.shape[0]
		with th.inference_mode():
			metric_input_d = {
				"true_mzs": eval_true_mzs,
				"true_ints": eval_true_probs,
				"true_batch_idxs": eval_true_batch_idxs,
				"pred_mzs": eval_pred_mzs,
				"pred_ints": eval_pred_probs,
				"pred_batch_idxs": eval_pred_batch_idxs,
				"true_prec_mzs": true_prec_mzs
			}
			for arg in self.extra_metric_args:
				metric_input_d[arg] = both_d[arg]
			metric_output_d = self.metric_fn(**metric_input_d)
			for k,v in metric_output_d.items():
				assert k not in results, k
				results[k] = v
		for metric_name in self.metric_names:
			assert metric_name in results, metric_name
		return results

	def training_step(self, batch, batch_idx):
		""" training loop

		Args:
			batch (_type_): _description_
			batch_idx (_type_): _description_

		Raises:
			NotImplementedError: _description_

		Returns:
			_type_: _description_
		"""

		if self.hparams.automatic_optimization:
			batch_results = self._common_step(batch, split="train")
			mean_loss = batch_results["mean_loss"]
			self._update_results(batch_results,"train")
			return mean_loss
		else:
			raise NotImplementedError("Manual optimization not implemented")

	def validation_step(self, batch, batch_idx):

		assert not self.training
		batch_results = self._common_step(batch, split="val")
		mean_loss = batch_results["mean_loss"]
		self._update_results(batch_results,"val")
		return mean_loss

	def test_step(self, batch, batch_idx):

		assert not self.training
		batch_results = self._common_step(batch, split="test")
		mean_loss = batch_results["mean_loss"]
		self._update_results(batch_results,"test")
		return mean_loss
	
	def inference_step(self, batch, split, untransform_spec=False):

		if self.training:
			print("Warning: model is in training mode")
		batch_results = self._common_step(batch, split=split, log=False)
		if untransform_spec:
			raise NotImplementedError("untransform_spec not implemented")
			# pred_logprobs = batch_results["pred_logprobs"]
			# pred_batch_idxs = batch_results["pred_batch_idxs"]
			# true_logprobs = batch_results["true_logprobs"]
			# true_batch_idxs = batch_results["true_batch_idxs"]
			# pred_ints = self.ints_untransform_func(
			# 	pred_logprobs.exp(),
			# 	pred_batch_idxs
			# )
			# pred_ints = self.ints_normalize_func(
			# 	pred_ints, 
			# 	pred_batch_idxs
			# )
			# if "pred_oos_logprobs" in batch_results:
			# 	pred_oos_logprobs = batch_results["pred_oos_logprobs"][pred_batch_idxs]
			# 	pred_ints = pred_ints * (1.-pred_oos_logprobs.exp())
			# batch_results["pred_logprobs"] = safelog(pred_ints, eps=self.hparams.log_min)
			# true_ints = self.ints_untransform_func(
			# 	true_logprobs.exp(),
			# 	true_batch_idxs
			# )
			# true_ints = self.ints_normalize_func(
			# 	true_ints, 
			# 	true_batch_idxs
			# )
			# batch_results["true_logprobs"] = safelog(true_ints, eps=self.hparams.log_min)
		return batch_results

	def configure_optimizers(self):

		if self.hparams.optimizer == "adam":
			optimizer_cls = th.optim.Adam
		elif self.hparams.optimizer == "adamw":
			optimizer_cls = th.optim.AdamW
		elif self.hparams.optimizer == "sgd":
			optimizer_cls = th.optim.SGD
		else:
			raise ValueError(f"Unknown optimizer {self.optimizer}")
		optimizer = optimizer_cls(
			self.parameters(), 
			lr=self.hparams.lr, 
			weight_decay=self.hparams.weight_decay
		)
		ret = {
			"optimizer": optimizer,
		}
		if self.hparams.lr_schedule:
			scheduler = build_lr_scheduler(
				optimizer=optimizer, 
				decay_rate=self.hparams.lr_decay_rate, 
				warmup_steps=self.hparams.lr_warmup_steps,
				decay_steps=self.hparams.lr_decay_steps,
			)
			ret["lr_scheduler"] = {
				"scheduler": scheduler,
				"frequency": 1,
				"interval": "step",
			}
		return ret

	def _get_wandb_logger(self):

		wandb_logger = None
		for logger in self.loggers:
			if isinstance(logger, pl.loggers.WandbLogger):
				wandb_logger = logger		
		return wandb_logger

	def _update_results(self, batch_results, split):

		results_attr = f"{split}_results"
		counter_attr = f"{split}_counter"
		# filter keys (filtering first to save time/memory)
		keys = [
			"unique_ids",
			"smiles",
			"true_mzs",
			"true_logprobs",
			"true_unique_ids",
			"pred_mzs",
			"pred_logprobs",
			"pred_unique_ids",
			"spec_per_group",
			"spec_per_mol",
			"group_per_mol"
		]
		keys.extend(list(self.metric_names))
		unique_ids = batch_results.pop("unique_id")
		true_batch_idxs = batch_results.pop("true_batch_idxs")
		pred_batch_idxs = batch_results.pop("pred_batch_idxs")
		batch_results = {k:v for k,v in batch_results.items() if k in keys}
		# update unique_ids
		true_unique_ids = unique_ids[true_batch_idxs]
		pred_unique_ids = unique_ids[pred_batch_idxs]
		batch_results["unique_ids"] = unique_ids
		batch_results["true_unique_ids"] = true_unique_ids
		batch_results["pred_unique_ids"] = pred_unique_ids
		# check all metrics
		assert all([metric_name in batch_results for metric_name in self.metric_names])
		# transfer to cpu
		batch_results = to_cpu(batch_results,detach=True)
		# setup results dict
		if getattr(self,results_attr) is None:
			setattr(self,results_attr,{k:list() for k in keys})
		else:
			assert set(keys) == set(getattr(self,results_attr).keys())
		results_dict = getattr(self,results_attr)
		# add to results dict
		for k,v in batch_results.items():
			results_dict[k].append(v)
		# increment counter
		setattr(self,counter_attr,getattr(self,counter_attr)+unique_ids.shape[0])

	def _reduce_metrics(self, results, split):

		mean_metrics, std_metrics = {}, {}
		for k, v in results.items():
			if k in self.metric_names:
				mean_metrics[k] = self.batch_metric_reduce_fn(
					b_metric=v,
					b_spec_per_group=results["spec_per_group"],
					b_spec_per_mol=results["spec_per_mol"],
					b_group_per_mol=results["group_per_mol"],
					reduce="w_mean",
					split=split
				)
				std_metrics[k] = self.batch_metric_reduce_fn(
					b_metric=v,
					b_spec_per_group=results["spec_per_group"],
					b_spec_per_mol=results["spec_per_mol"],
					b_group_per_mol=results["group_per_mol"],
					reduce="w_std",
					split=split
				)
		return mean_metrics, std_metrics

	def _consolidate_results(self, split):
		
		results = getattr(self,f"{split}_results")
		if results is None:
			return
		keys = results.keys()
		for k in keys:
			if isinstance(results[k][0],th.Tensor):
				results[k] = th.cat(results[k],dim=0)
			else:
				assert isinstance(results[k][0],list)
				results[k] = flatten_lol(results[k])
		# log all the metrics
		mean_metrics, std_metrics = self._reduce_metrics(results, split)
		for k in mean_metrics.keys():
			self.log(
				f"{split}_{k}_epoch/mean",
				mean_metrics[k],
			)
			self.log(
				f"{split}_{k}_epoch/std",
				std_metrics[k],
			)
		if split != "test":
			# update epoch stats
			mean_metrics_epochs = getattr(self,f"{split}_mean_metrics")
			std_metrics_epochs = getattr(self,f"{split}_std_metrics")
			for k in mean_metrics.keys():
				mean_metrics_epochs[k.replace(".","-")][self.current_epoch] = mean_metrics[k]
				std_metrics_epochs[k.replace(".","-")][self.current_epoch] = std_metrics[k]
			# log histograms
			if self.hparams.log_hist_metrics:
				wandb_logger = self._get_wandb_logger()
				import wandb
				for k,v in results.items():
					if k in self.metric_names:
						log_d = {
							f"{split}_{k}_hist": wandb.Histogram(v.cpu()),
							"epoch": self.current_epoch
						}
						if wandb_logger is not None:
							wandb_logger.experiment.log(log_d)
			# log best metric
			update_datapoint_metrics = False
			checkpoint_metric = self.hparams.checkpoint_metric.removeprefix("train_").removeprefix("val_").removesuffix("/mean").removesuffix("/std").removesuffix("_epoch")
			for k in self.metric_bests:
				assert k in self.metric_names, (k, self.metric_names)
				mean_metric_epochs = mean_metrics_epochs[k.replace(".","-")][:self.current_epoch+1]
				std_metric_epochs = std_metrics_epochs[k.replace(".","-")][:self.current_epoch+1]
				if self.metric_min_max[k] == "min":
					argbest_metric = th.argmin(mean_metric_epochs)
				elif self.metric_min_max[k] == "max":
					argbest_metric = th.argmax(mean_metric_epochs)
				else:
					assert self.metric_min_max[k] is None, self.metric_min_max[k]
					continue
				mean_metric_best = mean_metric_epochs[argbest_metric]
				std_metric_best = std_metric_epochs[argbest_metric]
				self.log(
					f"{split}_{k}_best/mean",
					mean_metric_best
				)
				self.log(
					f"{split}_{k}_best/std",
					std_metric_best
				)
				# if it's the best, update the datapoint metrics
				if k == checkpoint_metric and argbest_metric == self.current_epoch:
					update_datapoint_metrics = True
			if checkpoint_metric == "epoch":
				assert not update_datapoint_metrics
				update_datapoint_metrics = True
		else:
			update_datapoint_metrics = True

		if self.hparams.track_datapoint_metrics and update_datapoint_metrics:
			datapoint_metrics = getattr(self,f"{split}_datapoint_metrics")
			num_datapoints_p = getattr(self,f"{split}_num_datapoints")
			example_key = list(mean_metrics.keys())[0]
			num_datapoints = num_datapoints_p.item()
			if num_datapoints == -1:
				num_datapoints_p[0] = len(results[example_key])
				num_datapoints = num_datapoints_p.item()
			assert num_datapoints <= datapoint_metrics[example_key.replace(".","-")].shape[0], (num_datapoints, datapoint_metrics[example_key.replace(".","-")].shape[0])
			for k in mean_metrics.keys():
				assert len(results[k]) == num_datapoints, (k, len(results[k]), num_datapoints)
				datapoint_metrics[k.replace(".","-")][:num_datapoints] = results[k]

	def _log_images(self, split):

		results = getattr(self,f"{split}_results")
		num_log_images = getattr(self.hparams,f"num_log_{split}_images")
		counter = getattr(self,f"{split}_counter")
		num_log_images = min(num_log_images,counter)
		wandb_logger = self._get_wandb_logger()
		if wandb_logger is None:
			return
		# randomly sample unique_ids
		unique_ids = th.unique(results["unique_ids"],sorted=True)
		if num_log_images == unique_ids.shape[0]:
			sample_idxs = th.arange(num_log_images)
			sample_unique_ids = unique_ids
		else:
			with th_temp_seed(420):
				sample_idxs = th.randperm(unique_ids.shape[0])[:num_log_images]
				sample_unique_ids = unique_ids[sample_idxs]
		# plot images
		for i in range(num_log_images):
			unique_id = sample_unique_ids[i].item()
			unique_idx = th.nonzero(results["unique_ids"] == unique_id,as_tuple=False).item()
			true_mask = results["true_unique_ids"] == unique_id
			pred_mask = results["pred_unique_ids"] == unique_id
			smiles = results["smiles"][unique_idx]
			loss = results["loss"][unique_idx].item()
			if "cos_sim" in results:
				cos_sim = results["cos_sim"][unique_idx].item()
			else:
				cos_sim = np.nan
			if "wrecall" in results:
				wrecall = results["wrecall"][unique_idx].item()
			else:
				wrecall = np.nan
			# note: these are already untransformed and normalized (with possibility of oos)
			true_mzs = results["true_mzs"][true_mask]
			true_ints = th.exp(results["true_logprobs"][true_mask])
			pred_mzs = results["pred_mzs"][pred_mask]
			pred_ints = th.exp(results["pred_logprobs"][pred_mask])
			# cast to numpy
			true_mzs = true_mzs.numpy()
			true_ints = true_ints.numpy()
			pred_mzs = pred_mzs.numpy()
			pred_ints = pred_ints.numpy()
			assert np.isclose(np.sum(true_ints),1.0), np.sum(true_ints)
			assert np.isclose(np.sum(pred_ints),1.0), np.sum(pred_ints)
			# plot
			data = plot_spectra_sparse(
				true_mzs,
				true_ints,
				pred_mzs,
				pred_ints,
				smiles,
				return_data=True
			)
			wandb_logger.log_image(
				key=f"{split}_example_{i}",
				caption=[f"unique_id = {unique_id}, epoch = {self.current_epoch:03d}, loss = {loss:.3f}, cos_sim = {cos_sim:.3f}, wrecall = {wrecall:.3f}"],
				images=[data]
			)

	def on_train_epoch_start(self):

		self._seed_dataloader(self.current_epoch)

	def on_train_epoch_end(self):

		# consolidate results
		self._consolidate_results("train")
		# log images
		self._log_images("train")
		# reset
		self.train_results = None
		self.train_counter = 0
		# seed dataloader
		self._seed_dataloader(self.current_epoch+1)

	def on_validation_epoch_end(self):
		
		if not self.trainer.sanity_checking:
			# consolidate results
			self._consolidate_results("val")
			# log images
			self._log_images("val")
		# reset
		self.val_results = None
		self.val_counter = 0

	def on_test_epoch_end(self):

		# consolidate results
		self._consolidate_results("test")
		# reset
		self.test_results = None
		self.test_counter = 0

	def _print_grad_norm(self,prefix=None,total_num_params=5):

		if prefix is None:
			prefix = "no_prefix"
		print(f">> {prefix}, {self._cur_batch_size}, {self._max_batch_size}")
		opt = self.optimizers()
		num_params = 0
		for pg in opt.param_groups:
			for p in pg['params']:
				if p.grad is not None:
					print(th.norm(p.grad))
					num_params += 1
				if num_params > total_num_params:
					break
			if num_params > total_num_params:
				break
	
	def on_after_backward(self):
		
		if self.hparams.check_gradient_norm:
			self._print_grad_norm(prefix="after_backward")

	def on_before_optimizer_step(self, optimizer):

		if self.hparams.check_gradient_norm:
			self._print_grad_norm(prefix="before_optimizer_step")

	def on_after_optimizer_step(self, optimizer):

		self._cur_batch_size = 0
		self._cur_batch_weight = 0.

	def _check_ce_params(self):

		# check merge params
		if self.hparams.spec_params["merge"]:
			assert (not self.hparams.spec_params["nce"]) or self.hparams.spec_params["merge_keep_ces"]
		elif self.hparams.spec_params["nce"]:
			assert not self.hparams.spec_params["ace"]
			assert self.hparams.spec_params["nce"] and (not self.hparams.spec_params["merge_keep_ces"])
		elif self.hparams.spec_params["ace"]:
			assert not self.hparams.spec_params["nce"]
			assert self.hparams.spec_params["ace"] and (not self.hparams.spec_params["merge_keep_ces"])
   
	def _seed_dataloader(self, seed):

		generator = th.Generator()
		generator.manual_seed(self.train_dl_seeds[seed].item())
		train_dataloader = self.trainer.train_dataloader
		batch_sampler = train_dataloader.batch_sampler
		sampler = train_dataloader.sampler
		if self.hparams.dynamic_batch_sampler:
			if hasattr(batch_sampler.sampler, "generator"):
				# print("p3")
				batch_sampler.sampler.generator = generator
			if hasattr(batch_sampler.sampler, "_pre_compute_batches"):
				# print("p4")
				batch_sampler.sampler._pre_compute_batches()
			if hasattr(batch_sampler, "_pre_compute_batches"):
				# print("p5")
				batch_sampler._pre_compute_batches()
		else:
			assert batch_sampler.sampler is sampler
			if hasattr(sampler, "generator"):
				# print("p1")
				sampler.generator = generator
			if hasattr(sampler, "_pre_compute_batches"):
				# print("p2")
				sampler._pre_compute_batches()

	def on_save_checkpoint(self, checkpoint):

		# rng
		checkpoint["rng_states"] = _collect_rng_states(include_cuda=th.cuda.is_available())
		# metrics
		checkpoint["train_mean_metrics"] = self.train_mean_metrics
		checkpoint["val_mean_metrics"] = self.val_mean_metrics
		checkpoint["train_std_metrics"] = self.train_std_metrics
		checkpoint["val_std_metrics"] = self.val_std_metrics
		if self.hparams.track_datapoint_metrics:
			checkpoint["train_num_datapoints"] = self.train_num_datapoints
			checkpoint["val_num_datapoints"] = self.val_num_datapoints
			checkpoint["train_datapoint_metrics"] = self.train_datapoint_metrics
			checkpoint["val_datapoint_metrics"] = self.val_datapoint_metrics

	def on_load_checkpoint(self, checkpoint):
		
		# rng
		if "torch" in checkpoint["rng_states"]:
			checkpoint["rng_states"]["torch"] = checkpoint["rng_states"]["torch"].cpu()
		if "torch.cuda" in checkpoint["rng_states"]:
			checkpoint["rng_states"]["torch.cuda"][0] = checkpoint["rng_states"]["torch.cuda"][0].cpu()
		_set_rng_states(checkpoint["rng_states"])
		# metrics
		self.train_mean_metrics = checkpoint["train_mean_metrics"]
		self.val_mean_metrics = checkpoint["val_mean_metrics"]
		self.train_std_metrics = checkpoint["train_std_metrics"]
		self.val_std_metrics = checkpoint["val_std_metrics"]
		if self.hparams.track_datapoint_metrics:
			self.train_num_datapoints = checkpoint["train_num_datapoints"]
			self.val_num_datapoints = checkpoint["val_num_datapoints"]
			self.train_datapoint_metrics = checkpoint["train_datapoint_metrics"]
			self.val_datapoint_metrics = checkpoint["val_datapoint_metrics"]

class FragGNNPL(SpectrumPL):

	def _setup_model(self):

		# frag GNN
		self.model = FragGNNModel(
			num_depth=self.hparams.num_depth,
			num_hs=self.hparams.num_hs,
			num_elements=self.hparams.num_elements,
			int_embedder=self.hparams.int_embedder,
			int_embedder_tight=self.hparams.int_embedder_tight,
			mol_node_feats=self.hparams.mol_params["pyg_node_feats"],
			mol_edge_feats=self.hparams.mol_params["pyg_edge_feats"],
			mol_pe_embed_k=self.hparams.mol_params["pyg_pe_embed_k"],
			mol_hidden_size=self.hparams.mol_hidden_size,
			mol_num_layers=self.hparams.mol_num_layers,
			mol_gnn_type=self.hparams.mol_gnn_type,
			mol_dropout=self.hparams.mol_dropout,
			mol_normalization=self.hparams.mol_normalization,
			mol_pool_type=self.hparams.mol_pool_type,
			frag_node_feats=self.hparams.frag_params["pyg_node_feats"],
			frag_edge_feats=self.hparams.frag_params["pyg_edge_feats"],
			frag_hidden_size=self.hparams.frag_hidden_size,
			frag_num_layers=self.hparams.frag_num_layers,
			frag_gnn_type=self.hparams.frag_gnn_type,
			frag_dropout=self.hparams.frag_dropout,
			frag_normalization=self.hparams.frag_normalization,
			frag_pool_type=self.hparams.frag_pool_type,
			frag_embed_combine=self.hparams.frag_embed_combine,
			frag_pool_combine=self.hparams.frag_pool_combine,
			mlp_output_format=self.hparams.mlp_output_format,
			mlp_hidden_size=self.hparams.mlp_hidden_size,
			mlp_normalization=self.hparams.mlp_normalization,
			mlp_dropout=self.hparams.mlp_dropout,
			mlp_num_layers=self.hparams.mlp_num_layers,
			mlp_use_residuals=self.hparams.mlp_use_residuals,
			cc_interstage_type=self.hparams.cc_interstage_type,
			nb_iso=self.hparams.nb_iso,
			skip_edge_loss=self.hparams.skip_edge_loss,
			mask_null_formula=self.hparams.mask_null_formula,
			predict_oos=self.hparams.predict_oos,
			bin_output=self.hparams.bin_output,
			mz_bin_res=self.hparams.mz_bin_res,
			mz_max=self.hparams.mz_max,
			ce_insert_type=self.hparams.ce_insert_type,
			ce_insert_location=self.hparams.ce_insert_location,
			ce_insert_merge=self.hparams.ce_insert_merge,
			ce_insert_size=self.hparams.ce_insert_size,
			ce_mean=self.hparams.ce_mean,
			ce_std=self.hparams.ce_std,
			ce_max=self.hparams.ce_max,
			prec_insert_location=self.hparams.prec_insert_location,
			prec_insert_size=self.hparams.prec_insert_size,
			prec_types=self.hparams.spec_params["prec_types"],
			inst_insert_location=self.hparams.inst_insert_location,
			inst_insert_size=self.hparams.inst_insert_size,
			inst_types=self.hparams.spec_params["inst_types"],
			output_formula_str=self.hparams.output_formula_str
		)
		
		self._check_ce_params()

		# check edge loss params
		if not self.hparams.skip_edge_loss:
			assert "h_counts" in self.hparams.frag_params["pyg_node_feats"]
			assert "h_range" in self.hparams.frag_params["pyg_edge_feats"]
		else:
			if "h_counts" in self.hparams.frag_params["pyg_node_feats"]:
				logging.warning("h_counts in frag pyg_node_feats but edge_loss is disabled!")
			if "h_range" in self.hparams.frag_params["pyg_edge_feats"]:
				logging.warning("h_range in frag pyg_edge_feats but edge_loss is disabled!")

		# compile
		if self.hparams.compile:
			th_dynamo.reset()
			self.dynamo_prof = th_dynamo.utils.CompileProfiler()
			self.model = self.model.get_compile(backend=self.dynamo_prof,dynamic=True)

	def _setup_loss_names(self):

		# flag losses for tracking
		loss_names = [
			"loss",
			"primary_loss",
			"null_formula_prob",
			"oos_prob",
		]
		if self.hparams.loss_type == "cross_entropy":
			loss_names.extend([
				"spec_ce",
				"oos_ce",
				"ios_ce",
			])
		if not self.hparams.skip_extra_losses:
			loss_names.extend([
				"spec_e",
				"spec_ne",
				"formula_e",
				"formula_ne",
				"node_e",
				"node_ne",
				"node_formula_e",
				"node_formula_ne",
				"formula_node_e",
				"formula_node_ne",
				"joint_e",
				"joint_ne",
				"node_formula_mi",
				"formula_node_mi",
				"h_mean",
				"h_e",
				"h_ne"
			])
			if not self.hparams.skip_edge_loss:
				loss_names.extend([
					"edge_h_range_loss",
					"edge_h_transfer_loss",
					"edge_e",
					"edge_ne"
				])
			if self.hparams.nb_iso:
				loss_names.extend([
					"nb_node_e",
					"nb_node_ne",
					"nb_node_formula_e",
					"nb_node_formula_ne",
					"nb_formula_node_e",
					"nb_formula_node_ne",
					"nb_joint_e",
					"nb_joint_ne",
					"nb_node_node_e",
					"nb_node_node_ne",
					"nb_node_formula_mi",
					"nb_formula_node_mi",
					"nb_node_node_mi"
				])

		else:
			if not (self.hparams.formula_entropy_weight == self.hparams.formula_normalized_entropy_weight == 0.):
				loss_names.extend([
					"formula_e",
					"formula_ne",
				])
			if not (self.hparams.node_entropy_weight == self.hparams.node_normalized_entropy_weight == 0.):
				loss_names.extend([
					"node_e",
					"node_ne",
				])
			if not (self.hparams.node_formula_entropy_weight == self.hparams.node_formula_normalized_entropy_weight == 0.):
				loss_names.extend([
					"node_formula_e",
					"node_formula_ne",
				])
			if not (self.hparams.formula_node_entropy_weight == self.hparams.formula_node_normalized_entropy_weight == 0.):
				loss_names.extend([
					"formula_node_e",
					"formula_node_ne",
				])
			if not (self.hparams.joint_entropy_weight == self.hparams.joint_normalized_entropy_weight == 0.):
				loss_names.extend([
					"joint_e",
					"joint_ne",
				])
			if self.hparams.skip_edge_loss:
				if not (self.hparams.edge_h_range_loss_weight == self.hparams.edge_h_transfer_loss_weight == 0.):
					loss_names.extend([
						"edge_h_range_loss",
						"edge_h_transfer_loss",
					])
				if not (self.hparams.edge_entropy_weight == self.hparams.edge_normalized_entropy_weight == 0.):
					loss_names.extend([
						"edge_e",
						"edge_ne",
					])
			if self.hparams.nb_iso:
				if not (self.hparams.nb_node_entropy_weight == self.hparams.nb_node_normalized_entropy_weight == 0.):
					loss_names.extend([
						"nb_node_e",
						"nb_node_ne",
					])
				if not (self.hparams.nb_node_formula_entropy_weight == self.hparams.nb_node_formula_normalized_entropy_weight == 0.):
					loss_names.extend([
						"nb_node_formula_e",
						"nb_node_formula_ne",
					])
				if not (self.hparams.nb_formula_node_entropy_weight == self.hparams.nb_formula_node_normalized_entropy_weight == 0.):
					loss_names.extend([
						"nb_formula_node_e",
						"nb_formula_node_ne",
					])
				if not (self.hparams.nb_joint_entropy_weight == self.hparams.nb_joint_normalized_entropy_weight == 0.):
					loss_names.extend([
						"nb_joint_e",
						"nb_joint_ne",
					])
				if not (self.hparams.nb_node_node_entropy_weight == self.hparams.nb_node_node_normalized_entropy_weight == 0.):
					loss_names.extend([
						"nb_node_node_e",
						"nb_node_node_ne",
					])
		self.loss_names = loss_names
		self.metric_names.update(loss_names)

	def _setup_loss_fn(self):

		# cross entropy
		sparse_ce_fn = get_sparse_cross_entropy_fn(
			dist=self.hparams.output_distribution,
			vectorized=self.hparams.loss_vectorized,
			tolerance=self.tolerance,
			relative=self.relative,
			tolerance_min_mz=self.tolerance_min_mz,
			oos_tolerance_multiple=self.hparams.oos_tolerance_multiple,
			gaussian_renormalize=self.hparams.gaussian_renormalize,
			pm_tolerance_multiple=self.hparams.pm_tolerance_multiple,
			loss_batch_size=self.hparams.loss_batch_size
		)
		# edge loss
		edge_h_range_loss_fn = get_edge_loss_fn(self.hparams.edge_h_range_loss_fn,4.0)
		edge_h_transfer_loss_fn = get_edge_loss_fn(self.hparams.edge_h_transfer_loss_fn,4.0) #,1.0)

		# binned cosine distance
		def cos_dist_fn(
			true_mzs,
			true_logprobs,
			true_batch_idxs,
			pred_mzs,
			pred_logprobs,
			pred_batch_idxs):
			return sparse_cosine_distance_binned(
				true_mzs,
				true_logprobs,
				true_batch_idxs,
				pred_mzs,
				pred_logprobs,
				pred_batch_idxs,
				log_distance=(self.hparams.loss_type == "log_cosine_distance")
			)

		# hungarian cosine distance
		def cos_hun_fn(
				true_mzs,
				true_logprobs,
				true_batch_idxs,
				pred_mzs,
				pred_logprobs,
				pred_batch_idxs):
				return sparse_cosine_distance_hungarian(
					true_mzs,
					true_logprobs,
					true_batch_idxs,
					pred_mzs,
					pred_logprobs,
					pred_batch_idxs,
					tolerance=self.tolerance,
					relative=self.relative,
					log_distance=(self.hparams.loss_type == "log_cosine_distance_hungarian")
				)

		assert self.hparams.sparse_cosine_similarity
		assert self.hparams.sum_ints
		if self.hparams.loss_type == "cross_entropy":
			self.binned_loss = False
		elif self.hparams.loss_type in ["cosine_distance", "log_cosine_distance"]:
			self.binned_loss = True
		elif self.hparams.loss_type in ["cosine_distance_hungarian", "log_cosine_distance_hungarian"]:
			self.binned_loss = False
		else:
			raise ValueError(f"Unknown loss type {self.hparams.loss_type}")
			
		if self.binned_loss:
			if self.hparams.ints_transform != "none" and not self.hparams.bin_output:
				print("> Warning: binned loss with ints transform; output is not binned; inverse transform will be biased for binned metrics")
		else:
			assert not self.hparams.bin_output, "cannot train binned model with unbinned loss"

		def edge_loss(
			pred_edge_logprobs,
			pred_edge_h_diffs,
			pred_edge_h_range_masks,
			pred_edge_h_logprobs,
			pred_edge_batch_idxs):
			edge_h_range_costs = edge_h_range_loss_fn(pred_edge_h_diffs * pred_edge_h_range_masks.float())
			edge_h_range_costs = edge_h_range_costs.reshape(edge_h_range_costs.shape[0],-1)
			edge_h_transfer_costs = edge_h_transfer_loss_fn(pred_edge_h_diffs)
			edge_h_transfer_costs = edge_h_transfer_costs.reshape(edge_h_transfer_costs.shape[0],-1)
			edge_h_logprobs = pred_edge_h_logprobs.reshape(pred_edge_h_logprobs.shape[0],-1)
			edge_h_range_avg_cost = th.logsumexp(edge_h_logprobs + safelog(edge_h_range_costs, eps=self.hparams.log_min),dim=1)
			edge_h_transfer_avg_cost = th.logsumexp(edge_h_logprobs + safelog(edge_h_transfer_costs, eps=self.hparams.log_min),dim=1)
			h_range_loss = scatter_logsumexp(
				pred_edge_logprobs + edge_h_range_avg_cost,
				pred_edge_batch_idxs
			)
			h_transfer_loss = scatter_logsumexp(
				pred_edge_logprobs + edge_h_transfer_avg_cost,
				pred_edge_batch_idxs
			)
			return h_range_loss, h_transfer_loss
		
		self._setup_loss_names()

		def _loss_fn(
			true_mzs,
			true_logprobs,
			true_batch_idxs,
			pred_mzs,
			pred_logprobs,
			pred_batch_idxs,
			pred_formula_logprobs,
			pred_formula_formula_idxs,
			pred_formula_batch_idxs,
			pred_node_logprobs,
			pred_node_node_idxs,
			pred_node_batch_idxs,
			pred_node_formula_logprobs,
			pred_formula_node_logprobs,
			pred_joint_logprobs,
			pred_joint_node_idxs,
			pred_joint_formula_idxs,
			pred_joint_batch_idxs,
			pred_null_formula_logprob,
			pred_edge_logprobs,
			pred_edge_h_diffs,
			pred_edge_h_range_masks,
			pred_edge_h_logprobs,
			pred_edge_batch_idxs,
			pred_oos_logprobs,
			pred_h_logprobs,
			pred_h_counts,
			pred_h_batch_idxs,
			pred_nb_node_logprobs,
			pred_nb_node_formula_logprobs,
			pred_nb_formula_node_logprobs,
			pred_nb_joint_logprobs,
			pred_nb_node_node_logprobs,
			pred_nb_node_node_idxs,
			pred_nb_node_batch_idxs,
			pred_nb_joint_node_idxs,
			pred_nb_joint_formula_idxs,
			pred_nb_joint_batch_idxs,
			pred_nb_node_node_node_idxs,
			pred_nb_node_node_batch_idxs,
			**kwargs):
			
			loss_d = {}
			
			if self.hparams.loss_type == "cross_entropy":
				ios_ce, oos_ce, _, _ = sparse_ce_fn(
					true_mzs,
					true_logprobs,
					true_batch_idxs,
					pred_mzs,
					pred_logprobs,
					pred_batch_idxs,
					pred_oos_logprobs
				)
				spec_ce = ios_ce + oos_ce
				if self.hparams.oos_loss:
					primary_loss = spec_ce
				else:
					primary_loss = ios_ce
				loss_d["ios_ce"] = ios_ce
				loss_d["oos_ce"] = oos_ce
				loss_d["spec_ce"] = spec_ce
			elif self.hparams.loss_type in ["cosine_distance", "log_cosine_distance"]:
				primary_loss = cos_dist_fn(
					true_mzs,
					true_logprobs,
					true_batch_idxs,
					pred_mzs,
					pred_logprobs,
					pred_batch_idxs
				)
			elif self.hparams.loss_type in ["cosine_distance_hungarian", "log_cosine_distance_hungarian"]:
				primary_loss = cos_hun_fn(
					true_mzs,
					true_logprobs,
					true_batch_idxs,
					pred_mzs,
					pred_logprobs,
					pred_batch_idxs
				)

			loss = self.hparams.primary_loss_weight * primary_loss
			loss_d["primary_loss"] = primary_loss			
			
			if self.hparams.null_loss:
				assert self.hparams.loss_type != "cross_entropy", self.hparams.loss_type
				if self.hparams.loss_type in ["cosine_distance","cosine_distance_hungarian"]:
					loss = loss + self.hparams.null_loss_weight * th.exp(pred_null_formula_logprob)
				elif self.hparams.loss_type in ["log_cosine_distance","log_cosine_distance_hungarian"]:
					loss = loss + self.hparams.null_loss_weight * pred_null_formula_logprob

			if "spec_e" in self.loss_names or "spec_ne" in self.loss_names:
				spec_e, spec_ne = sparse_entropy_fn(
					pred_logprobs,
					pred_batch_idxs,
					oos_logprobs=pred_oos_logprobs,
					renormalize=True
				)
				loss_d["spec_e"] = spec_e
				loss_d["spec_ne"] = spec_ne

			if "formula_e" in self.loss_names or "formula_ne" in self.loss_names:
				# formula logprobs do not consider NULL, OOS
				formula_e, formula_ne = sparse_entropy_fn(
					pred_formula_logprobs,
					pred_formula_batch_idxs,
					support_size_delta=-1. # for NULL formula
				)
				loss = loss + self.hparams.formula_entropy_weight * formula_e \
					+ self.hparams.formula_normalized_entropy_weight * formula_ne
				loss_d["formula_e"] = formula_e
				loss_d["formula_ne"] = formula_ne

			if "node_e" in self.loss_names or "node_ne" in self.loss_names:
				node_e, node_ne = sparse_entropy_fn(
					pred_node_logprobs,
					pred_node_batch_idxs
				)
				loss = loss + self.hparams.node_entropy_weight * node_e \
					+ self.hparams.node_normalized_entropy_weight * node_ne
				loss_d["node_e"] = node_e
				loss_d["node_ne"] = node_ne
			
			if "node_formula_e" in self.loss_names or "node_formula_ne" in self.loss_names:
				# node_formula logprobs do not consider NULL, OOS
				node_formula_e, node_formula_ne = sparse_conditional_entropy_fn(
					pred_node_logprobs,
					pred_node_batch_idxs,
					pred_node_formula_logprobs,
					pred_joint_node_idxs,
					pred_joint_batch_idxs,
				)
				loss = loss + self.hparams.node_formula_entropy_weight * node_formula_e \
					+ self.hparams.node_formula_normalized_entropy_weight * node_formula_ne
				loss_d["node_formula_e"] = node_formula_e
				loss_d["node_formula_ne"] = node_formula_ne
			
			if "formula_node_e" in self.loss_names or "formula_node_ne" in self.loss_names:
				# formula_node logprobs do not consider NULL, OOS
				formula_node_e, formula_node_ne = sparse_conditional_entropy_fn(
					pred_formula_logprobs,
					pred_formula_batch_idxs,
					pred_formula_node_logprobs,
					pred_joint_formula_idxs,
					pred_joint_batch_idxs
				)
				loss = loss + self.hparams.formula_node_entropy_weight * formula_node_e \
					+ self.hparams.formula_node_normalized_entropy_weight * formula_node_ne
				loss_d["formula_node_e"] = formula_node_e
				loss_d["formula_node_ne"] = formula_node_ne
			
			if "joint_e" in self.loss_names or "joint_ne" in self.loss_names:
				# joint
				joint_e, joint_ne = sparse_entropy_fn(
					pred_joint_logprobs,
					pred_joint_batch_idxs
				)
				# assert th.all(th.isclose(joint_e, node_formula_e+node_e,rtol=0.,atol=0.001))
				# assert th.all(th.isclose(joint_e, formula_node_e+formula_e,rtol=0.,atol=0.001))
				loss = loss + self.hparams.joint_entropy_weight * joint_e \
					+ self.hparams.joint_normalized_entropy_weight * joint_ne
				loss_d["joint_e"] = joint_e
				loss_d["joint_ne"] = joint_ne
			
			if "node_formula_mi" in self.loss_names:
				# mi
				node_formula_mi = formula_e - node_formula_e
				loss_d["node_formula_mi"] = node_formula_mi

			if "formula_node_mi" in self.loss_names:
				formula_node_mi = node_e - formula_node_e
				loss_d["formula_node_mi"] = formula_node_mi

			# special probabilities
			null_formula_prob = th.exp(pred_null_formula_logprob)
			oos_prob = th.exp(pred_oos_logprobs)
			loss_d["null_formula_prob"] = null_formula_prob
			loss_d["oos_prob"] = oos_prob

			if "h_mean" in self.loss_names or "h_e" in self.loss_names or "h_ne" in self.loss_names:
				# hydrogens
				h_mean = scatter_reduce(
					th.exp(pred_h_logprobs) * pred_h_counts,
					pred_h_batch_idxs,
					reduce="sum"
				)
				h_e, h_ne = sparse_entropy_fn(
					pred_h_logprobs,
					pred_h_batch_idxs
				)
				loss_d["h_mean"] = h_mean
				loss_d["h_e"] = h_e
				loss_d["h_ne"] = h_ne
			
			if not self.hparams.skip_edge_loss:
				if "edge_h_range_loss" in self.loss_names or "edge_h_transfer_loss" in self.loss_names:
					r_el, t_el = edge_loss(
						pred_edge_logprobs,
						pred_edge_h_diffs,
						pred_edge_h_range_masks,
						pred_edge_h_logprobs,
						pred_edge_batch_idxs
					)
					loss = loss + self.hparams.edge_h_range_loss_weight * r_el \
						+ self.hparams.edge_h_transfer_loss_weight * t_el
					loss_d["edge_h_range_loss"] = r_el
					loss_d["edge_h_transfer_loss"] = t_el
				if "edge_e" in self.loss_names or "edge_ne" in self.loss_names:
					edge_e, edge_ne = sparse_entropy_fn(
						pred_edge_logprobs,
						pred_edge_batch_idxs
					)
					loss = loss + self.hparams.edge_entropy_weight * edge_e \
						+ self.hparams.edge_normalized_entropy_weight * edge_ne
					loss_d["edge_e"] = edge_e
					loss_d["edge_ne"] = edge_ne
			else:
				assert self.hparams.edge_h_range_loss_weight == 0.0
				assert self.hparams.edge_h_transfer_loss_weight == 0.0
				assert self.hparams.edge_entropy_weight == 0.0
				assert self.hparams.edge_normalized_entropy_weight == 0.0
			
			if self.hparams.nb_iso:
				if "nb_node_e" in self.loss_names or "nb_node_ne" in self.loss_names:
					nb_node_e, nb_node_ne = sparse_entropy_fn(
						pred_nb_node_logprobs,
						pred_nb_node_batch_idxs
					)
					loss = loss + self.hparams.nb_node_entropy_weight * nb_node_e + \
						self.hparams.nb_node_normalized_entropy_weight * nb_node_ne
					loss_d["nb_node_e"] = nb_node_e
					loss_d["nb_node_ne"] = nb_node_ne
				if "nb_node_formula_e" in self.loss_names or "nb_node_formula_ne" in self.loss_names:
					nb_node_formula_e, nb_node_formula_ne = sparse_conditional_entropy_fn(
						pred_nb_node_logprobs,
						pred_nb_node_batch_idxs,
						pred_nb_node_formula_logprobs,
						pred_nb_joint_node_idxs,
						pred_nb_joint_batch_idxs,
					)
					loss = loss + self.hparams.nb_node_formula_entropy_weight * nb_node_formula_e + \
						self.hparams.nb_node_formula_normalized_entropy_weight * nb_node_formula_ne
					loss_d["nb_node_formula_e"] = nb_node_formula_e
					loss_d["nb_node_formula_ne"] = nb_node_formula_ne
				if "nb_formula_node_e" in self.loss_names or "nb_formula_node_ne" in self.loss_names:
					nb_formula_node_e, nb_formula_node_ne = sparse_conditional_entropy_fn(
						pred_formula_logprobs,
						pred_formula_batch_idxs,
						pred_nb_formula_node_logprobs,
						pred_nb_joint_formula_idxs,
						pred_nb_joint_batch_idxs
					)
					assert not th.any(nb_formula_node_ne < 0.), nb_formula_node_ne
					assert not th.any(nb_formula_node_ne > 1.), nb_formula_node_ne
					loss = loss + self.hparams.nb_formula_node_entropy_weight * nb_formula_node_e + \
						self.hparams.nb_formula_node_normalized_entropy_weight * nb_formula_node_ne
					loss_d["nb_formula_node_e"] = nb_formula_node_e
					loss_d["nb_formula_node_ne"] = nb_formula_node_ne
				if "nb_node_node_e" in self.loss_names or "nb_node_node_ne" in self.loss_names:
					nb_node_node_e, nb_node_node_ne = sparse_conditional_entropy_fn(
						pred_nb_node_logprobs,
						pred_nb_node_batch_idxs,
						pred_nb_node_node_logprobs,
						pred_nb_node_node_node_idxs,
						pred_nb_node_node_batch_idxs
					)
					loss = loss + self.hparams.nb_node_node_entropy_weight * nb_node_node_e + \
						self.hparams.nb_node_node_normalized_entropy_weight * nb_node_node_ne
					loss_d["nb_node_node_e"] = nb_node_node_e
					loss_d["nb_node_node_ne"] = nb_node_node_ne
				if "nb_joint_e" in self.loss_names or "nb_joint_ne" in self.loss_names:
					nb_joint_e, nb_joint_ne = sparse_entropy_fn(
						pred_nb_joint_logprobs,
						pred_nb_joint_batch_idxs
					)
					# assert th.all(th.isclose(nb_joint_e, nb_node_formula_e+nb_node_e,rtol=0.,atol=0.001))
					# assert th.all(th.isclose(nb_joint_e, nb_formula_node_e+formula_e,rtol=0.,atol=0.001))
					loss = loss + self.hparams.nb_joint_entropy_weight * nb_joint_e \
						+ self.hparams.nb_joint_normalized_entropy_weight * nb_joint_ne
					loss_d["nb_joint_e"] = nb_joint_e
					loss_d["nb_joint_ne"] = nb_joint_ne
				if "nb_node_formula_mi" in self.loss_names:
					nb_node_formula_mi = formula_e - nb_node_formula_e
					loss_d["nb_node_formula_mi"] = nb_node_formula_mi
				if "nb_formula_node_mi" in self.loss_names:
					nb_formula_node_mi = nb_node_e - nb_formula_node_e
					loss_d["nb_formula_node_mi"] = nb_formula_node_mi
				if "nb_node_node_mi" in self.loss_names:
					nb_node_node_mi = nb_node_e - nb_node_node_e
					loss_d["nb_node_node_mi"] = nb_node_node_mi
			else:
				assert self.hparams.nb_node_entropy_weight == 0.0
				assert self.hparams.nb_node_normalized_entropy_weight == 0.0
				assert self.hparams.nb_node_formula_entropy_weight == 0.0
				assert self.hparams.nb_node_formula_normalized_entropy_weight == 0.0
				assert self.hparams.nb_formula_node_entropy_weight == 0.0
				assert self.hparams.nb_formula_node_normalized_entropy_weight == 0.0
			
			# finally, update the loss
			loss_d["loss"] = loss
			
			return loss_d
		
		self.loss_fn = _loss_fn

	def on_train_epoch_end(self):

		if not self.hparams.automatic_optimization and self.hparams.dynamic_batch_sampler and self._cur_batch_size > 0:
			# this is used for manul opt when lr_schedulers is not available from torch lightning
			self._manual_opt()

		super().on_train_epoch_end()

	def _manual_opt(self):
		""" run manual opt
		"""
		opt = self.optimizers()
		# scale gradients
		if self.hparams.dynamic_batch_sampler:
			for pg in opt.param_groups:
				for p in pg['params']:
					if p.grad is not None:
						p.grad.data.mul_(self._max_batch_size / self._cur_batch_weight)					
		self.on_before_optimizer_step(opt)
		# clip gradients
		self.clip_gradients(opt, gradient_clip_val=self.hparams.gradient_clip_val, gradient_clip_algorithm=self.hparams.gradient_clip_algorithm)
		# call opt
		opt.step()
		self.on_after_optimizer_step(opt)
		# clean gradient
		opt.zero_grad()

	def training_step(self, batch, batch_idx):
		""" training loop for FragGNNPL

		Args:
			batch (_type_): _description_
			batch_idx (_type_): _description_

		Raises:
			NotImplementedError: _description_

		Returns:
			_type_: _description_
		"""

		if self.hparams.automatic_optimization:
			batch_results = self._common_step(batch, split="train")
			mean_loss = batch_results["mean_loss"]
			if self.hparams.debug_zero_loss:
				mean_loss = 0. * mean_loss
				self.print(mean_loss)
			self._cur_batch_size += self.hparams.train_batch_size
			self._update_results(batch_results,"train")
			return mean_loss
		
		elif self.hparams.dynamic_batch_sampler:
			batch_size = batch['batch_size'].item()
			assert batch_size > 0, batch_size
			batch_results = self._common_step(batch, split="train")
			total_loss = batch_results["total_loss"]
			total_weight = batch_results["total_weight"]
			mean_loss = total_loss / self._max_batch_size
			if self.hparams.debug_zero_loss:
				mean_loss = 0. * mean_loss
				self.print(mean_loss)
			self.manual_backward(mean_loss)
			# accumulate gradients of N samples
			self._cur_batch_size += batch_size
			self._cur_batch_weight += total_weight
			if self._cur_batch_size >= self._max_batch_size:
				self._manual_opt()
			self._update_results(batch_results,"train")
			return mean_loss
		
		else:
			batch_results = self._common_step(batch, split="train")
			mean_loss = batch_results["mean_loss"] / self.hparams.accumulate_grad_batches
			total_weight = batch_results["total_weight"]
			if self.hparams.debug_zero_loss:
				mean_loss = 0. * mean_loss
				self.print(mean_loss)
			self.manual_backward(mean_loss)
			# accumulate gradients of N samples
			self._cur_batch_size += self.hparams.train_batch_size
			self._cur_batch_weight += total_weight
			if (batch_idx + 1) % self.hparams.accumulate_grad_batches == 0:
				self._manual_opt()
			self._update_results(batch_results,"train")
			return mean_loss
		
	def optimizer_step(self, *args, **kwargs):
		
		super().optimizer_step(*args, **kwargs)
		opt = self.optimizers()
		self.on_after_optimizer_step(opt)

class BinnedPL(SpectrumPL):

	def _setup_loss_fn(self):
		
		# binned cosine distance
		def cos_dist_fn(
			true_mzs,
			true_logprobs,
			true_batch_idxs,
			pred_mzs,
			pred_logprobs,
			pred_batch_idxs):
			return sparse_cosine_distance_binned(
				true_mzs,
				true_logprobs,
				true_batch_idxs,
				pred_mzs,
				pred_logprobs,
				pred_batch_idxs,
				log_distance=(self.hparams.loss_type == "log_cosine_distance")
			)
		
		assert self.hparams.loss_type == "cosine_distance", self.hparams.loss_type
		assert self.hparams.sparse_cosine_similarity
		self.binned_loss = True

		def _loss_fn(
			true_mzs,
			true_logprobs,
			true_batch_idxs,
			pred_mzs,
			pred_logprobs,
			pred_batch_idxs,
			**kwargs):
			
			spec_cd = cos_dist_fn(
				true_mzs,
				true_logprobs,
				true_batch_idxs,
				pred_mzs,
				pred_logprobs,
				pred_batch_idxs
			)
			loss = spec_cd
			loss_d = {
				"loss": loss,
				"spec_cd": spec_cd
			}
			return loss_d

		self.loss_fn = _loss_fn
		# flag metrics for tracking
		loss_names = [
			"loss",
			"spec_cd"
		]
		self.metric_names.update(loss_names)

class NeimsPL(BinnedPL):

	def _setup_model(self):

		self.model = NeimsModel(
			mol_fingerprint_morgan=self.hparams.mol_params["fingerprint_morgan"],
			mol_fingerprint_rdkit=self.hparams.mol_params["fingerprint_rdkit"],
			mol_fingerprint_maccs=self.hparams.mol_params["fingerprint_maccs"],
			mlp_hidden_size=self.hparams.mlp_hidden_size,
			mlp_dropout=self.hparams.mlp_dropout,
			mlp_num_layers=self.hparams.mlp_num_layers,
			mlp_use_residuals=self.hparams.mlp_use_residuals,
			mz_max=self.hparams.mz_max,
			mz_bin_res=self.hparams.mz_bin_res,
			ff_prec_mz_offset=self.hparams.ff_prec_mz_offset,
			ff_bidirectional=self.hparams.ff_bidirectional,
			ff_output_map_size=self.hparams.ff_output_map_size,
			ff_output_activation=self.hparams.ff_output_activation,
			int_embedder=self.hparams.int_embedder,
			ce_insert_type=self.hparams.ce_insert_type,
			ce_insert_location=self.hparams.ce_insert_location,
			ce_insert_merge=self.hparams.ce_insert_merge,
			ce_insert_size=self.hparams.ce_insert_size,
   			ce_mean=self.hparams.ce_mean,
			ce_std=self.hparams.ce_std,
			ce_max=self.hparams.ce_max,
			prec_insert_location=self.hparams.prec_insert_location,
			prec_insert_size=self.hparams.prec_insert_size,
			prec_types=self.hparams.spec_params["prec_types"],
			inst_insert_location=self.hparams.inst_insert_location,
			inst_insert_size=self.hparams.inst_insert_size,
			inst_types=self.hparams.spec_params["inst_types"],
			log_min=self.hparams.log_min
		)
		
		self._check_ce_params()

		# compile
		if self.hparams.compile:
			th_dynamo.reset()
			self.dynamo_prof = th_dynamo.utils.CompileProfiler()
			self.model = th.compile(self.model,backend=self.dynamo_prof,dynamic=True)

class PrecursorPL(SpectrumPL):

	def _setup_model(self):

		assert not self.hparams.compile
		self.model = PrecursorModel()

	def _setup_loss_names(self):

		# flag losses for tracking
		loss_names = [
			"loss",
			"primary_loss",
		]
		self.loss_names = loss_names
		self.metric_names.update(loss_names)

	def _setup_loss_fn(self):

		assert self.hparams.loss_type == "cross_entropy"

		# cross entropy
		sparse_ce_fn = get_sparse_cross_entropy_fn(
			dist=self.hparams.output_distribution,
			vectorized=self.hparams.loss_vectorized,
			tolerance=self.tolerance,
			relative=self.relative,
			tolerance_min_mz=self.tolerance_min_mz,
			oos_tolerance_multiple=self.hparams.oos_tolerance_multiple,
			gaussian_renormalize=self.hparams.gaussian_renormalize,
			pm_tolerance_multiple=self.hparams.pm_tolerance_multiple,
			loss_batch_size=self.hparams.loss_batch_size
		)

		self._setup_loss_names()

		def loss_fn(
			true_mzs,
			true_logprobs,
			true_batch_idxs,
			pred_mzs,
			pred_logprobs,
			pred_batch_idxs,
			**kwargs):

			batch_size = th.max(true_batch_idxs).item()+1
			assert pred_mzs.shape[0] == pred_logprobs.shape[0] == batch_size
			ios_ce, oos_ce, true_oos_logprob, true_oos_e = sparse_ce_fn(
				true_mzs,
				true_logprobs,
				true_batch_idxs,
				pred_mzs,
				pred_logprobs,
				pred_batch_idxs,
				th.full((batch_size,), LOG_ZERO(true_logprobs.dtype), device=pred_logprobs.device, dtype=pred_logprobs.dtype)
			)
			spec_ce = ios_ce + oos_ce
			if self.hparams.oos_loss:
				raise NotImplementedError
				primary_loss = spec_ce
			else:
				primary_loss = ios_ce
			loss = primary_loss
			loss_d = {
				"loss": loss,
				"primary_loss": primary_loss,
			}
			return loss_d

		self.loss_fn = loss_fn
		self.binned_loss = False

class GNNPL(BinnedPL):
	def _setup_model(self):

		self.model = GNNModel( 	# TODO: initialize model based on params. finalize params
			mol_node_feats=self.hparams.mol_params["pyg_node_feats"],	# mol feats
			mol_edge_feats=self.hparams.mol_params["pyg_edge_feats"],
			mol_pe_embed_k=self.hparams.mol_params["pyg_pe_embed_k"],
			mol_hidden_size=self.hparams.mol_hidden_size,
			mol_num_layers=self.hparams.mol_num_layers,
			mol_gnn_type=self.hparams.mol_gnn_type,
			mol_normalization=self.hparams.mol_normalization,
			mol_dropout=self.hparams.mol_dropout,
			mol_pool_type=self.hparams.mol_pool_type,
			mlp_hidden_size=self.hparams.mlp_hidden_size,				# FFN
			mlp_dropout=self.hparams.mlp_dropout,
			mlp_num_layers=self.hparams.mlp_num_layers,
			mlp_use_residuals=self.hparams.mlp_use_residuals,
			mz_max=self.hparams.mz_max,
			mz_bin_res=self.hparams.mz_bin_res,
			ff_prec_mz_offset=self.hparams.ff_prec_mz_offset,
			ff_bidirectional=self.hparams.ff_bidirectional,
			ff_output_map_size=self.hparams.ff_output_map_size,
			ff_output_activation=self.hparams.ff_output_activation,
			int_embedder=self.hparams.int_embedder,						# cross entropy
			ce_insert_type=self.hparams.ce_insert_type,
			ce_insert_location=self.hparams.ce_insert_location,
			ce_insert_merge=self.hparams.ce_insert_merge,
			ce_insert_size=self.hparams.ce_insert_size,
      		ce_mean=self.hparams.ce_mean,
			ce_std=self.hparams.ce_std,
			ce_max=self.hparams.ce_max,
			prec_insert_location=self.hparams.prec_insert_location,		# precursor
			prec_insert_size=self.hparams.prec_insert_size,
			prec_types=self.hparams.spec_params["prec_types"],
			inst_insert_location=self.hparams.inst_insert_location,		# instrument
			inst_insert_size=self.hparams.inst_insert_size,
			inst_types=self.hparams.spec_params["inst_types"],
		)

		self._check_ce_params()
	
		# compile
		if self.hparams.compile:
			th_dynamo.reset()
			self.dynamo_prof = th_dynamo.utils.CompileProfiler()
			self.model = th.compile(self.model,backend=self.dynamo_prof,dynamic=True)

def test_models():

	from fragnnet.runner import load_config, init_dataset
	from fragnnet.utils.misc_utils import print_shapes
	from fragnnet.utils.misc_utils import to_device

	template_fp = "config/template.yml"
	custom_fp_to_model_cls = {
		# "config/debug_m/debug_d3_m.yml": FragGNNPL,
		# "config/debug_m/debug_d3_prec_m.yml": FragGNNPL,
		# "config/debug_m/debug_neims_prec_m.yml": NeimsPL,
		# "config/debug_um/debug_d3_prec_um.yml": FragGNNPL,
		'config/debug_m/debug_gnn.yml': GNNPL
	}

	for custom_fp, pl_model_cls in custom_fp_to_model_cls.items():

		print(">>>", custom_fp)

		config_d = load_config(template_fp, custom_fp)
		for k in ["spec_params","mol_params","frag_params"]:
			config_d[k]["preprocess"] = False
			if "preload" in config_d[k]:
				config_d[k]["preload"] = False
		device = "cuda:0" if config_d["accelerator"] == "gpu" else "cpu"

		ds = init_dataset(config_d, splits=("train",))[0]
		dl = th.utils.data.DataLoader(
			ds,
			batch_size=8,
			shuffle=False,
			num_workers=0,
			collate_fn=ds.get_collate_fn(),
		)
		dl_iter = iter(dl)
		batch = next(dl_iter)
		print("> batch")
		print_shapes(batch)
		batch = to_device(batch,device)

		pl_model = pl_model_cls(**config_d)
		pl_model.train()
		pl_model.to(device)

		outputs = pl_model.forward(**batch)
		print("> train outputs")
		print_shapes(outputs)
		print(th.exp(outputs["pred_logprobs"]).sum())

		results = pl_model._common_step(batch, split="train", log=False)
		print_shapes(results)

		mean_loss = results["mean_loss"]
		print("> loss")
		print(mean_loss)
		mean_loss.backward()

		print()


if __name__ == "__main__":

	from fragnnet.utils.misc_utils import th_temp_seed

	with th_temp_seed(420):
		test_models()
