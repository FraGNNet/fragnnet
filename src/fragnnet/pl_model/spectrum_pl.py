"""
Base PyTorch Lightning model for spectrum prediction in FragGNN.

This class includs model setup, loss, metrics, optimizer configuration, and training/validation/test logic.
It is intended to be subclassed for specific model architectures and tasks.

"""

import inspect
import warnings
from typing import Any

import lightning as L

# Try to import Lightning, supporting both lightning.pytorch and pytorch_lightning for compatibility
import numpy as np
import torch as th
from lightning.fabric.utilities.seed import _collect_rng_states, _set_rng_states
from lightning.pytorch.loggers import WandbLogger

from fragnnet.model.loss import sparse_entropy_fn
from fragnnet.utils.misc_utils import (
    TOLERANCE_MIN_MZ,
    flatten_lol,
    safelog,
    scatter_l1normalize,
    th_temp_generator,
    to_cpu,
)
from fragnnet.utils.nn_utils import build_lr_scheduler, get_optimizer_class
from fragnnet.utils.plot_utils import plot_spectra_sparse
from fragnnet.utils.spec_utils import (
    batch_cos_hun_helper,
    batch_jss_hun_helper,
    batched_bin_func,
    batched_filter_func,
    batched_l1_normalize,
    batched_mf1000_normalize,
    calculate_match_mzs,
    cos_sim_helper,
    get_ints_transform_func,
    get_ints_untransform_func,
    jss_helper,
    ndcg_helper,
    opt_cos_sim_helper,
    round_aggregate_peaks,
    scatter_reduce,
)


def get_split_fn(bs_idx):
    # check the validity of bs idx
    diffs = bs_idx[1:] - bs_idx[:-1]
    assert len(diffs) == 0 or diffs.min() >= 0
    batch_counts = th.bincount(bs_idx)
    batch_counts = batch_counts.tolist()

    def split_fn(inp_tensor):
        return th.split(inp_tensor, batch_counts)

    return split_fn


def list_to_batch_tensor(inp, pad):
    max_l = max([len(_) for _ in inp])

    def padv(v):
        if len(v) < max_l:
            return th.cat([v, v.new_zeros(max_l - len(v)) + pad], 0)
        return v

    return th.stack([padv(_) for _ in inp], 0)


def ragged_to_batch(feat: th.Tensor, batch_idx: th.Tensor, pad_value=0):
    assert batch_idx.ndim == 1
    assert feat.size(0) == batch_idx.size(0)

    device = feat.device
    N = batch_idx.size(0)

    lengths = th.bincount(batch_idx)
    B = lengths.size(0)
    max_len = lengths.max()

    offsets = th.cumsum(lengths, dim=0)
    offsets = th.cat([th.zeros(1, device=device, dtype=offsets.dtype), offsets[:-1]])
    local_idx = th.arange(N, device=device) - offsets[batch_idx]
    assert feat.ndim == 1

    out = th.full((B, max_len), pad_value, device=device, dtype=feat.dtype)
    out[batch_idx, local_idx] = feat
    return out


def primary_ce_to_batch(
    spec_ce: th.Tensor, spec_ce_batch_idxs: th.Tensor | None, batch_size: int
) -> th.Tensor:
    """Extract one collision energy value per batch element.

    Args:
        spec_ce: Collision energy tensor. This is typically flattened by collation and has
            shape (num_ce_values,) or (batch_size, num_ce_values_per_sample).
        spec_ce_batch_idxs: Batch indices for flattened CE values of shape (num_ce_values,).
        batch_size: Number of samples in the batch.

    Returns:
        Collision energy tensor of shape (batch_size,).
    """
    if spec_ce.ndim == 2:
        if spec_ce.size(1) == 1:
            # batch_func produced (total_ces, 1) — squeeze trailing dim and
            # fall through to the sparse batch_idxs path below.
            spec_ce = spec_ce.squeeze(-1)
        elif spec_ce.size(0) == batch_size:
            # Dense layout (batch_size, num_ces) — take first CE per sample.
            return spec_ce[:, 0].detach().cpu()
        else:
            raise ValueError(
                "Expected spec_ce to have batch_size rows when it is 2D, "
                f"got shape {tuple(spec_ce.shape)} and batch_size={batch_size}"
            )

    flat_spec_ce = spec_ce.reshape(-1).detach().cpu()
    if spec_ce_batch_idxs is None:
        if flat_spec_ce.numel() != batch_size:
            raise ValueError(
                "spec_ce_batch_idxs is required when spec_ce is flattened and does not "
                f"already have one value per batch item; got {flat_spec_ce.numel()} values "
                f"for batch_size={batch_size}"
            )
        return flat_spec_ce

    flat_batch_idxs = spec_ce_batch_idxs.reshape(-1).detach().cpu().long()
    if flat_batch_idxs.numel() != flat_spec_ce.numel():
        raise ValueError(
            "spec_ce and spec_ce_batch_idxs must have the same number of elements; "
            f"got {flat_spec_ce.numel()} and {flat_batch_idxs.numel()}"
        )

    ce_primary = th.full((batch_size,), float("nan"), dtype=flat_spec_ce.dtype)
    for batch_idx in range(batch_size):
        batch_mask = flat_batch_idxs == batch_idx
        if batch_mask.any():
            ce_primary[batch_idx] = flat_spec_ce[batch_mask][0]
    return ce_primary


class SpectrumPL(L.LightningModule):
    """
    Base PyTorch LightningModule for spectrum prediction.
    Provides modular setup and hooks for spectrum modeling in FragGNN.
    Subclasses should override _setup_model and _setup_loss_fn.
    """

    hparams: Any

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        # setup functions and state
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
        """
        Override in subclass to set up the model.
        """
        raise NotImplementedError

    def _setup_tolerance(self):
        """
        Setup m/z tolerance for matching peaks.

        Sets the global tolerance scalars and, optionally, a per-instrument-type tolerance map
        when ``inst_type_loss_tol`` is configured. All per-instrument entries must use relative
        tolerance (``rel`` + ``min_mz``).
        """
        if self.hparams["loss_tolerance_rel"] is not None:
            self.tolerance = self.hparams["loss_tolerance_rel"]
            self.relative = True
            self.tolerance_min_mz = self.hparams["loss_tolerance_min_mz"]
        else:
            assert self.hparams["loss_tolerance_abs"] is not None
            self.tolerance = self.hparams["loss_tolerance_abs"]
            self.relative = False
            self.tolerance_min_mz = None

        # Per-instrument-type tolerance lookup tensors: shape [num_inst_types].
        # Indexed by inst_type_idx in _common_step to build per-sample tolerance tensors
        # without a Python loop. None when inst_type_loss_tol is not configured (fast path).
        inst_type_loss_tol = self.hparams.get("inst_type_loss_tol")
        if inst_type_loss_tol is not None and len(inst_type_loss_tol) > 0:
            inst_types_sorted = sorted(self.hparams.spec_params["inst_types"])
            global_min_mz = (
                self.tolerance_min_mz if self.tolerance_min_mz is not None else TOLERANCE_MIN_MZ
            )
            tols, min_mzs = [], []
            for inst_type in inst_types_sorted:
                if inst_type in inst_type_loss_tol:
                    cfg = inst_type_loss_tol[inst_type]
                    tols.append(float(cfg["rel"]))
                    min_mzs.append(float(cfg["min_mz"]))
                else:
                    tols.append(self.tolerance)
                    min_mzs.append(global_min_mz)
            self.register_buffer(
                "inst_tol_lookup", th.tensor(tols, dtype=th.float32), persistent=False
            )
            self.register_buffer(
                "inst_min_mz_lookup", th.tensor(min_mzs, dtype=th.float32), persistent=False
            )
        else:
            self.register_buffer("inst_tol_lookup", None, persistent=False)
            self.register_buffer("inst_min_mz_lookup", None, persistent=False)

        # idx↔str maps are fetched lazily from the dataloader dataset via
        # _get_dataset_maps() — the dataset is the canonical source of truth.

    def _setup_loss_fn(self):
        """
        Override in subclass to set up the loss function.
        """
        raise NotImplementedError

    def _setup_spec_fns(self):
        """
        Setup spectrum preprocessing functions: filtering, binning, intensity transforms, and normalization.
        """

        def _filter_func(mzs, ints, batch_idxs):
            return batched_filter_func(
                mzs, ints, batch_idxs, self.hparams["target_ints_thresh"], self.hparams.mz_max
            )

        self.filter_func = _filter_func

        def _target_rank_filter_func(mzs, ints, batch_idxs):
            return batched_filter_func(
                mzs,
                ints,
                batch_idxs,
                ints_thresh=-float("inf"),
                mz_max=0.0,
                top_k_peaks=self.hparams["train_target_top_k_peaks"],
                drop_min_int_peak=self.hparams["train_target_drop_min_int_peak"],
            )

        self.target_rank_filter_func = _target_rank_filter_func

        def _bin_func(mzs, ints, batch_idxs):
            agg = "sum" if self.hparams.sum_ints else "amax"
            bin_idxs, bin_ints, bin_batch_idxs = batched_bin_func(
                mzs,
                ints,
                batch_idxs,
                self.hparams.mz_max,
                self.hparams.mz_bin_res,
                agg,
                sparse=True,
            )
            return bin_idxs, bin_ints, bin_batch_idxs

        self.bin_func = _bin_func
        self.ints_transform_func = get_ints_transform_func(self.hparams.ints_transform)
        self.ints_untransform_func = get_ints_untransform_func(self.hparams.ints_transform)
        self.ints_normalize_func = batched_l1_normalize

    def _setup_metric_fns(self):
        """
        Setup all auxiliary and main metric functions, including configuration for different evaluation modes.
        """

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
        compute_pred_entropies = False
        compute_inst_tol_match_mzs = False
        self.batch_metric_args: set[str] = set()

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
            elif metric_name == "pred_spec_e":
                self.metric_min_max["pred_spec_e"] = None
                self.auxiliary_metric_names.add("pred_spec_e")
                compute_pred_entropies = True
            elif metric_name == "pred_spec_ne":
                self.metric_min_max["pred_spec_ne"] = None
                self.auxiliary_metric_names.add("pred_spec_ne")
                compute_pred_entropies = True
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
            #   self.metric_min_max["dice"] = "max"
            #   self.metric_bests.append("dice")
            #   compute_match_mzs = True
            elif metric_name == "ndcg":
                for union in [True, False]:
                    if union:
                        fname_base = "ndcg_un"
                        tie_break_flags = [True, False]
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
                compute_match_mzs = True
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
            elif metric_name == "cos_hun_inst_tol":
                for remove_prec_peak in hun_remove_prec_peak_flags:
                    for sqrt in hun_sqrt_flags:
                        fname_base = "cos_hun_inst_tol"
                        if sqrt:
                            fname_base += "_sqrt"
                        if remove_prec_peak:
                            fname_base += "_np"
                        fname = fname_base
                        self.metric_min_max[fname] = "max"
                        self.metric_bests.append(fname)
                        self.auxiliary_metric_names.add(fname)
                compute_inst_tol_match_mzs = True
                compute_rounded_match_mzs = True
                self.batch_metric_args.add("spec_inst_type")
            elif metric_name == "jss_hun_inst_tol":
                for remove_prec_peak in hun_remove_prec_peak_flags:
                    for sqrt in hun_sqrt_flags:
                        fname_base = "jss_hun_inst_tol"
                        if sqrt:
                            fname_base += "_sqrt"
                        if remove_prec_peak:
                            fname_base += "_np"
                        fname = fname_base
                        self.metric_min_max[fname] = "max"
                        self.metric_bests.append(fname)
                        self.auxiliary_metric_names.add(fname)
                compute_inst_tol_match_mzs = True
                compute_rounded_match_mzs = True
                self.batch_metric_args.add("spec_inst_type")
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
            split="test",
            **kwargs,
        ):
            # assumes both true/pred spectra are already L1-normalized

            batch_size = th.max(true_batch_idxs) + 1
            assert batch_size == th.max(pred_batch_idxs) + 1, (
                batch_size,
                th.max(pred_batch_idxs) + 1,
            )

            if split == "train" or split == "val":
                metric_d = {
                    k: -th.ones([batch_size], dtype=true_ints.dtype, device=true_ints.device)
                    for k in self.auxiliary_metric_names
                }
            else:
                metric_d = {}
            # define aggregation
            agg = "sum" if self.hparams.sum_ints else "amax"

            # global calculations
            if compute_rounded_match_mzs:
                true_mzs_r, true_ints_r, true_batch_idxs_r = round_aggregate_peaks(
                    true_mzs, true_ints, true_batch_idxs, agg=agg
                )
                pred_mzs_r, pred_ints_r, pred_batch_idxs_r = round_aggregate_peaks(
                    pred_mzs, pred_ints, pred_batch_idxs, agg=agg
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
                                prec_mzs=true_prec_mzs,
                            )
                            pred_bin_idxs, pred_bin_ints, pred_bin_batch_idxs = batched_bin_func(
                                pred_mzs,
                                pred_ints,
                                pred_batch_idxs,
                                mz_max=self.hparams.mz_max,
                                mz_bin_res=mz_bin_res,
                                agg=agg,
                                sparse=True,
                                remove_prec_peaks=remove_prec_peak,
                                prec_mzs=true_prec_mzs,
                            )
                            # apply intensity transform (e.g. sqrt); helpers normalize internally
                            true_bin_ints = ints_transform(true_bin_ints)
                            pred_bin_ints = ints_transform(pred_bin_ints)
                            if "cos_sim" in self.hparams.auxiliary_scores:
                                fname_base = "cos_sim"
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
                                    pred_bin_batch_idxs,
                                )
                                # print(cos_sim)
                                metric_d[fname] = cos_sim
                            if "opt_cos_sim" in self.hparams.auxiliary_scores:
                                fname_base = "opt_cos_sim"
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
                                fname_base = "jss"
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
                                    log_min=self.hparams.log_min,
                                )
                                metric_d[fname] = jss

            if compute_entropies:
                true_log_probs = kwargs.get("true_logprobs")
                if true_log_probs is None:
                    true_log_probs = safelog(
                        scatter_l1normalize(true_ints, true_batch_idxs), eps=self.hparams.log_min
                    )
                true_spec_e, true_spec_ne = sparse_entropy_fn(true_log_probs, true_batch_idxs)
                if "true_spec_e" in self.hparams.auxiliary_scores:
                    metric_d["true_spec_e"] = true_spec_e
                if "true_spec_ne" in self.hparams.auxiliary_scores:
                    metric_d["true_spec_ne"] = true_spec_ne

            if compute_pred_entropies:
                pred_log_probs = kwargs.get("pred_logprobs")
                if pred_log_probs is None:
                    pred_log_probs = safelog(
                        scatter_l1normalize(pred_ints, pred_batch_idxs), eps=self.hparams.log_min
                    )
                pred_spec_e, pred_spec_ne = sparse_entropy_fn(pred_log_probs, pred_batch_idxs)
                if "pred_spec_e" in self.hparams.auxiliary_scores:
                    metric_d["pred_spec_e"] = pred_spec_e
                if "pred_spec_ne" in self.hparams.auxiliary_scores:
                    metric_d["pred_spec_ne"] = pred_spec_ne

            true_ion_num = th.bincount(true_batch_idxs)
            pred_ion_num = th.bincount(pred_batch_idxs)
            if compute_rounded_match_mzs:
                batched_true_ints_r = ragged_to_batch(true_ints_r, true_batch_idxs_r, pad_value=0)
                batched_pred_ints_r = ragged_to_batch(pred_ints_r, pred_batch_idxs_r, pad_value=0)
                sqrt_batched_true_ints_r = get_ints_transform_func("sqrt")(batched_true_ints_r)
                sqrt_batched_pred_ints_r = get_ints_transform_func("sqrt")(batched_pred_ints_r)

            if compute_match_mzs:
                batch_true_ints = ragged_to_batch(
                    true_ints, true_batch_idxs, pad_value=float("inf")
                )
                batch_pred_ints = ragged_to_batch(
                    pred_ints, pred_batch_idxs, pad_value=float("inf")
                )
                batch_true_mzs = ragged_to_batch(true_mzs, true_batch_idxs, pad_value=float("inf"))
                batch_pred_mzs = ragged_to_batch(pred_mzs, pred_batch_idxs, pad_value=float("inf"))
                batch_match_mask = calculate_match_mzs(
                    batch_true_mzs,
                    batch_pred_mzs,
                    tolerance=self.tolerance,
                    relative=self.relative,
                    tolerance_min_mz=self.tolerance_min_mz,
                )
                batch_true_match_mask = batch_match_mask.max(-1).values
                batch_pred_match_mask = batch_match_mask.max(-2).values
            if compute_rounded_match_mzs:
                batch_true_mzs_r = ragged_to_batch(
                    true_mzs_r, true_batch_idxs_r, pad_value=float("inf")
                )
                batch_pred_mzs_r = ragged_to_batch(
                    pred_mzs_r, pred_batch_idxs_r, pad_value=float("inf")
                )
                batch_true_prec_mask_r = calculate_match_mzs(
                    batch_true_mzs_r,
                    true_prec_mzs[..., None],
                    tolerance=self.tolerance,
                    relative=self.relative,
                    tolerance_min_mz=self.tolerance_min_mz,
                ).squeeze(-1)
                batch_pred_prec_mask_r = calculate_match_mzs(
                    batch_pred_mzs_r,
                    true_prec_mzs[..., None],
                    tolerance=self.tolerance,
                    relative=self.relative,
                    tolerance_min_mz=self.tolerance_min_mz,
                ).squeeze(-1)
                # match
                batch_match_mask_r = calculate_match_mzs(
                    batch_true_mzs_r,
                    batch_pred_mzs_r,
                    tolerance=self.tolerance,
                    relative=self.relative,
                    tolerance_min_mz=self.tolerance_min_mz,
                )
                batch_true_match_mask_r = batch_match_mask_r.max(
                    -1
                ).values  # th.any(b_match_mask_r, dim=1)
                batch_pred_match_mask_r = batch_match_mask_r.max(
                    -2
                ).values  # th.any(b_match_mask_r, dim=0)

            # calculate batched cosine and batched jss
            batched_metric = {}
            if "cos_hun" in self.hparams.auxiliary_scores:
                for remove_prec_peak in hun_remove_prec_peak_flags:
                    for sqrt in hun_sqrt_flags:
                        if sqrt:
                            inp_true = sqrt_batched_true_ints_r
                            inp_pred = sqrt_batched_pred_ints_r
                            # ints_transform = get_ints_transform_func("sqrt")
                        else:
                            inp_true = batched_true_ints_r
                            inp_pred = batched_pred_ints_r
                            # ints_transform = get_ints_transform_func("none")
                        fname_base = "cos_hun"
                        if sqrt:
                            fname_base += "_sqrt"
                        if remove_prec_peak:
                            fname_base += "_np"
                        fname = f"{fname_base}"
                        # metric_d[fname][b_idx] = b_cos_hun
                        batched_metric[fname] = batch_cos_hun_helper(
                            inp_true,
                            inp_pred,
                            batch_match_mask_r,
                            batch_true_match_mask_r,
                            batch_pred_match_mask_r,
                            remove_prec_peak,
                            batch_true_prec_mask_r,
                            batch_pred_prec_mask_r,
                        )
            if "jss_hun" in self.hparams.auxiliary_scores:
                for remove_prec_peak in hun_remove_prec_peak_flags:
                    for sqrt in hun_sqrt_flags:
                        if sqrt:
                            inp_true = sqrt_batched_true_ints_r
                            inp_pred = sqrt_batched_pred_ints_r
                            # ints_transform = get_ints_transform_func("sqrt")
                        else:
                            inp_true = batched_true_ints_r
                            inp_pred = batched_pred_ints_r
                        fname_base = "jss_hun"
                        if sqrt:
                            fname_base += "_sqrt"
                        if remove_prec_peak:
                            fname_base += "_np"
                        fname = f"{fname_base}"

                        batched_metric[fname] = batch_jss_hun_helper(
                            inp_true,
                            inp_pred,
                            batch_match_mask_r,
                            batch_true_match_mask_r,
                            batch_pred_match_mask_r,
                            remove_prec_peak,
                            batch_true_prec_mask_r,
                            batch_pred_prec_mask_r,
                            log_min=self.hparams.log_min,
                        )

            if "recall" in self.hparams.auxiliary_scores:
                metric_d["recall"] = batch_true_match_mask.float().sum(-1) / true_ion_num.float()
            if "wrecall" in self.hparams.auxiliary_scores:
                metric_d["wrecall"] = (batch_true_ints * batch_true_match_mask.float()).sum(-1)
            if "precision" in self.hparams.auxiliary_scores:
                metric_d["precision"] = batch_pred_match_mask.float().sum(-1) / pred_ion_num.float()
            if "wprecision" in self.hparams.auxiliary_scores:
                metric_d["wprecision"] = (batch_pred_ints * batch_pred_match_mask.float()).sum(-1)
            if "cos_hun" in self.hparams.auxiliary_scores:
                for remove_prec_peak in hun_remove_prec_peak_flags:
                    for sqrt in hun_sqrt_flags:
                        fname_base = "cos_hun"
                        if sqrt:
                            fname_base += "_sqrt"
                        if remove_prec_peak:
                            fname_base += "_np"
                        fname = f"{fname_base}"
                        metric_d[fname] = batched_metric[fname]
            if "jss_hun" in self.hparams.auxiliary_scores:
                for remove_prec_peak in hun_remove_prec_peak_flags:
                    for sqrt in hun_sqrt_flags:
                        fname_base = "jss_hun"
                        if sqrt:
                            fname_base += "_sqrt"
                        if remove_prec_peak:
                            fname_base += "_np"
                        fname = f"{fname_base}"
                        metric_d[fname] = batched_metric[fname]
            if compute_inst_tol_match_mzs and (
                "cos_hun_inst_tol" in self.hparams.auxiliary_scores
                or "jss_hun_inst_tol" in self.hparams.auxiliary_scores
            ):
                spec_inst_type = kwargs.get("spec_inst_type")
                if spec_inst_type is not None and self.inst_tol_lookup is not None:
                    inst_idxs = spec_inst_type.reshape(-1)
                    device = true_ints.device
                    dtype = true_ints.dtype
                    tol_per_sample = self.inst_tol_lookup[inst_idxs].to(dtype=dtype)
                    min_mz_per_sample = self.inst_min_mz_lookup[inst_idxs].to(dtype=dtype)
                    # Build per-peak effective absolute tolerance: (B, N_true).
                    # tol_rel * clamp(mz, min=min_mz) converts relative ppm to absolute Da.
                    safe_mzs = batch_true_mzs_r.clamp(max=self.hparams.mz_max)
                    tol_per_peak = tol_per_sample[:, None] * th.maximum(
                        safe_mzs, min_mz_per_sample[:, None]
                    )
                    batch_match_mask_it = calculate_match_mzs(
                        batch_true_mzs_r,
                        batch_pred_mzs_r,
                        tolerance=0.0,
                        relative=False,
                        tol_per_true=tol_per_peak,
                    )
                    batch_true_match_mask_it = batch_match_mask_it.max(-1).values
                    batch_pred_match_mask_it = batch_match_mask_it.max(-2).values
                    if "cos_hun_inst_tol" in self.hparams.auxiliary_scores:
                        for remove_prec_peak in hun_remove_prec_peak_flags:
                            for sqrt in hun_sqrt_flags:
                                inp_true = sqrt_batched_true_ints_r if sqrt else batched_true_ints_r
                                inp_pred = sqrt_batched_pred_ints_r if sqrt else batched_pred_ints_r
                                fname_base = "cos_hun_inst_tol"
                                if sqrt:
                                    fname_base += "_sqrt"
                                if remove_prec_peak:
                                    fname_base += "_np"
                                metric_d[fname_base] = batch_cos_hun_helper(
                                    inp_true,
                                    inp_pred,
                                    batch_match_mask_it,
                                    batch_true_match_mask_it,
                                    batch_pred_match_mask_it,
                                    remove_prec_peak,
                                    batch_true_prec_mask_r,
                                    batch_pred_prec_mask_r,
                                )
                    if "jss_hun_inst_tol" in self.hparams.auxiliary_scores:
                        for remove_prec_peak in hun_remove_prec_peak_flags:
                            for sqrt in hun_sqrt_flags:
                                inp_true = sqrt_batched_true_ints_r if sqrt else batched_true_ints_r
                                inp_pred = sqrt_batched_pred_ints_r if sqrt else batched_pred_ints_r
                                fname_base = "jss_hun_inst_tol"
                                if sqrt:
                                    fname_base += "_sqrt"
                                if remove_prec_peak:
                                    fname_base += "_np"
                                metric_d[fname_base] = batch_jss_hun_helper(
                                    inp_true,
                                    inp_pred,
                                    batch_match_mask_it,
                                    batch_true_match_mask_it,
                                    batch_pred_match_mask_it,
                                    remove_prec_peak,
                                    batch_true_prec_mask_r,
                                    batch_pred_prec_mask_r,
                                    log_min=self.hparams.log_min,
                                )
            if (
                "ndcg" in self.hparams.auxiliary_scores
                or "true_oos_prob" in self.hparams.auxiliary_scores
                or "true_oos_e" in self.hparams.auxiliary_scores
            ):
                pred_split_fn = get_split_fn(pred_batch_idxs)
                true_split_fn = get_split_fn(true_batch_idxs)
                split_true_ints = true_split_fn(true_ints)
                split_pred_ints = pred_split_fn(pred_ints)
                for b_idx in range(batch_size):  # the followings are not speed up yet
                    b_true_ints = split_true_ints[b_idx]
                    b_pred_ints = split_pred_ints[b_idx]

                    if compute_match_mzs:
                        b_true_match_mask = batch_true_match_mask[b_idx][: b_true_ints.shape[0]]
                        b_pred_match_mask = batch_pred_match_mask[b_idx][: b_pred_ints.shape[0]]

                    if "ndcg" in self.hparams.auxiliary_scores:
                        for union in [True, False]:
                            if union:
                                fname_base = "ndcg_un"
                                tie_break_flags = [True, False]
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
                                    batch_match_mask[b_idx][
                                        : b_true_ints.shape[0], : b_pred_ints.shape[0]
                                    ],
                                    b_true_match_mask,
                                    b_pred_match_mask,
                                    optimistic,
                                    union,
                                )
                                fname = f"{fname_base}{suffix}"
                                metric_d[fname][b_idx] = b_ndcg

                    if "true_oos_prob" in self.hparams.auxiliary_scores:
                        b_true_oos_prob = th.sum(b_true_ints[~b_true_match_mask]) / th.sum(
                            b_true_ints
                        )
                        metric_d["true_oos_prob"][b_idx] = b_true_oos_prob

                    if "true_oos_e" in self.hparams.auxiliary_scores:
                        b_true_oos_probs = b_true_ints[~b_true_match_mask] / th.sum(
                            b_true_ints[~b_true_match_mask]
                        )
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
                        dim=0,
                    )
                    assert th.min(b_pred_node_count) > 0, b_pred_node_count
                    metric_d[f"pred_{key}_count"] = b_pred_node_count

            if "pred_formula_count" in self.hparams.auxiliary_scores:
                b_pred_formula_count = scatter_reduce(
                    th.ones_like(kwargs["pred_formula_batch_idxs"]),
                    kwargs["pred_formula_batch_idxs"],
                    reduce="sum",
                    dim=0,
                )
                b_pred_formula_count = b_pred_formula_count - 1  # -1 for OOS!
                assert th.min(b_pred_formula_count) > 0, b_pred_formula_count
                metric_d["pred_formula_count"] = b_pred_formula_count

            if "pred_edge_count" in self.hparams.auxiliary_scores:
                b_pred_edge_count = scatter_reduce(
                    th.ones_like(kwargs["pred_edge_batch_idxs"]),
                    kwargs["pred_edge_batch_idxs"],
                    reduce="sum",
                    dim=0,
                )
                assert th.min(b_pred_edge_count) > 0, b_pred_edge_count
                metric_d["pred_edge_count"] = b_pred_edge_count

            return metric_d

        self.metric_fn = calculate_all_auxiliary_metrics

    def _get_batch_metric_reduce_fn(self, sample_weight):
        if sample_weight == "none":
            calc_sample_weights = lambda spec_per_group, spec_per_mol, group_per_mol: th.ones_like(
                spec_per_group, dtype=th.float32
            )
        elif sample_weight == "group":
            calc_sample_weights = (
                lambda spec_per_group, spec_per_mol, group_per_mol: 1.0 / spec_per_group
            )
        elif sample_weight == "mol":
            calc_sample_weights = (
                lambda spec_per_group, spec_per_mol, group_per_mol: 1.0 / spec_per_mol
            )
        elif sample_weight == "group_mol":
            calc_sample_weights = lambda spec_per_group, spec_per_mol, group_per_mol: 1.0 / (
                spec_per_group * group_per_mol
            )

        def _batch_metric_reduce(
            b_metric,
            b_spec_per_group,
            b_spec_per_mol,
            b_group_per_mol,
            reduce,
            return_weights=False,
        ):
            b_sample_weight = calc_sample_weights(b_spec_per_group, b_spec_per_mol, b_group_per_mol)
            b_total_weight = th.sum(b_sample_weight, dim=0)
            if reduce == "w_mean":
                b_reduce_metric = th.sum(b_sample_weight * b_metric, dim=0) / b_total_weight
            elif reduce == "w_std":
                b_reduce_metric = th.sqrt(
                    th.sum(
                        b_sample_weight
                        * (b_metric - th.sum(b_sample_weight * b_metric, dim=0) / b_total_weight)
                        ** 2,
                        dim=0,
                    )
                    / b_total_weight
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
        self.train_batch_metric_reduce_fn = self._get_batch_metric_reduce_fn(
            self.hparams.train_sample_weight
        )
        self.eval_batch_metric_reduce_fn = self._get_batch_metric_reduce_fn(
            self.hparams.eval_sample_weight
        )

        def _batch_metric_reduce_fn(split, **kwargs):
            if split == "train":
                return self.train_batch_metric_reduce_fn(**kwargs)
            else:
                return self.eval_batch_metric_reduce_fn(**kwargs)

        self.batch_metric_reduce_fn = _batch_metric_reduce_fn

    def _setup_result_trackers(self):
        if self.hparams.spec_params["merge"]:
            self.max_num_datapoints = 22000
        else:
            self.max_num_datapoints = 270000
        report_metric_std = getattr(self.hparams, "report_metric_std", False)
        for split in ["train", "val", "test"]:
            setattr(self, f"{split}_results", None)
            setattr(self, f"{split}_counter", 0)
            setattr(self, f"{split}_metric_sum_w", {})
            setattr(self, f"{split}_metric_sum_wx", {})
            setattr(self, f"{split}_metric_sum_wx2", {} if report_metric_std else None)
            setattr(self, f"{split}_mean_metrics", {})
            setattr(self, f"{split}_std_metrics", {} if report_metric_std else None)
            if self.hparams.track_datapoint_metrics:
                setattr(self, f"{split}_datapoint_metrics", {})
                setattr(self, f"{split}_num_datapoints", -th.ones([1], dtype=th.int64))
            for name in self.metric_names:
                _name = name.replace(".", "-")
                mean_metrics = getattr(self, f"{split}_mean_metrics")
                if split != "test":
                    mean_metrics[_name] = th.full(
                        [self.hparams.max_epochs], float("nan"), dtype=th.float32
                    )
                else:
                    mean_metrics[_name] = th.full([1], float("nan"), dtype=th.float32)
                if report_metric_std:
                    std_metrics = getattr(self, f"{split}_std_metrics")
                    if split != "test":
                        std_metrics[_name] = th.full(
                            [self.hparams.max_epochs], float("nan"), dtype=th.float32
                        )
                    else:
                        std_metrics[_name] = th.full([1], float("nan"), dtype=th.float32)

    def _setup_sampler(self):
        self._cur_batch_size = 0
        self._cur_batch_weight = 0.0
        self._max_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches
        self.automatic_optimization = self.hparams.automatic_optimization
        train_dl_generator = th.Generator()
        train_dl_generator.manual_seed(self.hparams.seed)
        self.train_dl_seeds = th.randint(
            low=0, high=2**32 - 1, size=[self.hparams.max_epochs + 1], generator=train_dl_generator
        )

    def _preproc_spec(
        self,
        spec_mzs,
        spec_ints,
        spec_batch_idxs,
        filter_spec=False,
        target_rank_filter_spec=False,
        bin_spec=False,
        transform_spec=False,
        normalize_spec=False,
    ):
        """
        Preprocess a spectrum: filter, bin, transform, and normalize intensities as needed.
        """
        # assumes spec_ints are not logged, or transformed in any way
        # does not assume any particular kind of normalization

        if filter_spec:
            # filter
            spec_mzs, spec_ints, spec_batch_idxs = self.filter_func(
                spec_mzs, spec_ints, spec_batch_idxs
            )
        if target_rank_filter_spec:
            spec_mzs, spec_ints, spec_batch_idxs = self.target_rank_filter_func(
                spec_mzs, spec_ints, spec_batch_idxs
            )
        if bin_spec:
            # bin
            spec_mzs, spec_ints, spec_batch_idxs = self.bin_func(
                spec_mzs, spec_ints, spec_batch_idxs
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
        log_out: bool = False,
    ):
        """
        Convenience wrapper for spectrum preprocessing, handling different modes for train/pred.
        """
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
                normalize_spec=False,
            )
        elif train and (not pred):
            spec_mzs, spec_ints, spec_batch_idxs = self._preproc_spec(
                spec_mzs,
                spec_ints,
                spec_batch_idxs,
                filter_spec=True,
                target_rank_filter_spec=True,
                bin_spec=self.binned_loss,
                transform_spec=True,
                normalize_spec=True,
            )
        elif (not train) and pred:
            # untransform
            spec_ints = self.ints_untransform_func(spec_ints, spec_batch_idxs)
            # normalize (note that this messes up OOS stuff, which is fine for eval...)
            spec_ints = self.ints_normalize_func(spec_ints, spec_batch_idxs)
        elif (not train) and (not pred):
            spec_mzs, spec_ints, spec_batch_idxs = self._preproc_spec(
                spec_mzs,
                spec_ints,
                spec_batch_idxs,
                filter_spec=True,
                target_rank_filter_spec=self.hparams.get(
                    "eval_target_apply_train_rank_filters",
                    self.hparams.get("eval_target_apply_train_peak_filters", False),
                ),
                bin_spec=False,
                transform_spec=False,
                normalize_spec=True,
            )
        if log_out:
            spec_ints = safelog(spec_ints, eps=self.hparams.log_min)
        return spec_mzs, spec_ints, spec_batch_idxs

    def predict_step(self, **batch_kwargs):
        """
        PyTorch Lightning predict_step override.
        """
        return self.forward(**batch_kwargs)

    def forward(self, **batch_kwargs):
        """
        Forward pass through the model, with optional activation checkpointing.
        """
        # get predictions
        if self.hparams.activation_checkpointing:
            forward_keys = list(inspect.signature(self.model.forward).parameters.keys())
            forward_keys.remove("kwargs")
            forward_args = [batch_kwargs.get(k) for k in forward_keys]
            pred = th.utils.checkpoint.checkpoint(
                self.model.forward, *forward_args, use_reentrant=False
            )
        else:
            pred = self.model.forward(**batch_kwargs)
        return pred

    def _common_step(self, batch, split="train", log=True):
        """
        Common logic for train/val/test/inference steps: preprocess, forward, loss, metrics.
        """
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
        # prepare ground truth
        train_true_mzs, train_true_logprobs, train_true_batch_idxs = self.preproc_spec(
            true_mzs, true_ints, true_batch_idxs, train=True, pred=False, log_in=False, log_out=True
        )
        # forward pass and get predictions
        pred_d = self.forward(**batch)
        pred_mzs = pred_d.pop("pred_mzs")
        pred_logprobs = pred_d.pop("pred_logprobs")
        pred_batch_idxs = pred_d.pop("pred_batch_idxs")
        pred_oos_logprobs = pred_d.pop("pred_oos_logprobs", None)
        train_pred_mzs, train_pred_logprobs, train_pred_batch_idxs = self.preproc_spec(
            pred_mzs,
            pred_logprobs,
            pred_batch_idxs,
            train=True,
            pred=True,
            log_in=True,
            log_out=True,
        )

        # Build per-sample tolerance tensors when inst_type_loss_tol is configured.
        # Each sample gets the (rel, min_mz) pair for its instrument type, looked up from
        # inst_tol_map (keyed by inst_type index). This lets FT use tight 10 ppm matching
        # while QTOF uses a wider floor to cover floor-truncated m/z artifacts (≤0.009 Da).
        # When inst_type_loss_tol is null in config, inst_tol_map is None and both tensors
        # stay None, falling back to the global scalar tolerance in all loss functions.
        tol_per_sample = None
        min_mz_per_sample = None
        if self.inst_tol_lookup is not None and "spec_inst_type" in batch:
            # reshape(-1) instead of squeeze(-1): squeeze would collapse [1] → 0-d scalar
            # when batch_size=1. reshape(-1) always produces a flat 1-D tensor.
            inst_type_idxs = batch["spec_inst_type"].reshape(-1)  # [batch_size]
            dtype, device = train_true_logprobs.dtype, train_true_logprobs.device
            # Tensor indexing (no Python loop) — lookup tensors built once in _setup_tolerance
            tol_per_sample = self.inst_tol_lookup[inst_type_idxs].to(dtype=dtype)
            min_mz_per_sample = self.inst_min_mz_lookup[inst_type_idxs].to(dtype=dtype)

        # prepare training dict and compute loss
        train_d = {
            "true_mzs": train_true_mzs,
            "true_logprobs": train_true_logprobs,
            "true_batch_idxs": train_true_batch_idxs,
            "pred_mzs": train_pred_mzs,
            "pred_logprobs": train_pred_logprobs,
            "pred_batch_idxs": train_pred_batch_idxs,
            "pred_oos_logprobs": pred_oos_logprobs,
            "batch_size": batch_size,
            **pred_d,
        }
        if tol_per_sample is not None:
            train_d["tol_per_sample"] = tol_per_sample
            train_d["min_mz_per_sample"] = min_mz_per_sample

        loss_d = self.loss_fn(**train_d)
        loss = loss_d["loss"]
        mean_loss = self.batch_metric_reduce_fn(
            b_metric=loss,
            b_spec_per_group=spec_per_group,
            b_spec_per_mol=spec_per_mol,
            b_group_per_mol=group_per_mol,
            reduce="w_mean",
            split=split,
        )
        total_loss, total_weight = self.batch_metric_reduce_fn(
            b_metric=loss,
            b_spec_per_group=spec_per_group,
            b_spec_per_mol=spec_per_mol,
            b_group_per_mol=group_per_mol,
            reduce="w_sum",
            return_weights=True,
            split=split,
        )

        active_metric_names = self._get_active_metric_names(split)
        need_aux_metrics = len(active_metric_names.intersection(self.auxiliary_metric_names)) > 0
        need_eval_tensors = self._needs_eval_tensors(split)
        both_d = {}
        if need_eval_tensors:
            # Prepare eval tensors outside autograd: they are only used for metrics/logging.
            with th.inference_mode():
                # Note we need to compare spectra in normal space, so no log here.
                eval_true_mzs, eval_true_probs, eval_true_batch_idxs = self.preproc_spec(
                    true_mzs,
                    true_ints,
                    true_batch_idxs,
                    train=False,
                    pred=False,
                    log_in=False,
                    log_out=False,
                )
                eval_true_logprobs = safelog(eval_true_probs, eps=self.hparams.log_min)
                eval_pred_mzs, eval_pred_probs, eval_pred_batch_idxs = self.preproc_spec(
                    pred_mzs.detach(),
                    pred_logprobs.detach(),
                    pred_batch_idxs.detach(),
                    train=False,
                    pred=True,
                    log_in=True,
                    log_out=False,
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
                    **pred_d,
                }
        # just log loss
        if log:
            self.log(f"{split}_batch_loss", mean_loss, batch_size=batch_size, on_epoch=True)
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
            **loss_d,
        }
        # running inference and report metrics
        if need_aux_metrics:
            assert true_mzs.shape[0] > 0, true_mzs.shape[0]
            with th.inference_mode():
                metric_input_d = {
                    "true_mzs": eval_true_mzs,
                    "true_ints": eval_true_probs,
                    "true_logprobs": eval_true_logprobs,
                    "true_batch_idxs": eval_true_batch_idxs,
                    "pred_mzs": eval_pred_mzs,
                    "pred_ints": eval_pred_probs,
                    "pred_logprobs": eval_pred_logprobs,
                    "pred_batch_idxs": eval_pred_batch_idxs,
                    "true_prec_mzs": true_prec_mzs,
                    "split": split,
                }
                for arg in self.extra_metric_args:
                    metric_input_d[arg] = both_d[arg]
                for arg in self.batch_metric_args:
                    if arg in batch:
                        metric_input_d[arg] = batch[arg]
                metric_output_d = self.metric_fn(**metric_input_d)
                for k, v in metric_output_d.items():
                    assert k not in results, k
                    results[k] = v
        for metric_name in active_metric_names:
            assert metric_name in results, metric_name
        if self._should_store_ce_metadata(split):
            idx_to_inst, idx_to_prec, _ = self._get_dataset_maps()
            if "spec_inst_type" in batch and idx_to_inst:
                inst_idxs = batch["spec_inst_type"].reshape(-1).tolist()
                results["inst_type_str"] = [idx_to_inst.get(i, "unknown") for i in inst_idxs]
            else:
                results["inst_type_str"] = ["unknown"] * batch_size
            if "spec_prec_type" in batch and idx_to_prec:
                prec_idxs = batch["spec_prec_type"].reshape(-1).tolist()
                results["prec_type_str"] = [idx_to_prec.get(i, "unknown") for i in prec_idxs]
            else:
                results["prec_type_str"] = ["unknown"] * batch_size
            if "spec_ce" in batch:
                results["ce_primary"] = primary_ce_to_batch(
                    batch["spec_ce"], batch.get("spec_ce_batch_idxs"), batch_size
                )
            else:
                results["ce_primary"] = th.full((batch_size,), float("nan"))
            if "spec_nce" in batch:
                results["ce_nce"] = primary_ce_to_batch(
                    batch["spec_nce"], batch.get("spec_nce_batch_idxs"), batch_size
                )
            else:
                results["ce_nce"] = th.full((batch_size,), float("nan"))
            if "spec_ace" in batch:
                results["ce_ace"] = primary_ce_to_batch(
                    batch["spec_ace"], batch.get("spec_ace_batch_idxs"), batch_size
                )
            else:
                results["ce_ace"] = th.full((batch_size,), float("nan"))
        return results

    def training_step(self, batch, batch_idx):
        """
        PyTorch Lightning training_step override.
        Handles automatic optimization only.
        """
        if self.hparams.automatic_optimization:
            batch_results = self._common_step(batch, split="train")
            mean_loss = batch_results["mean_loss"]
            self._update_results(batch_results, "train")
            return mean_loss
        else:
            raise NotImplementedError("Manual optimization not implemented")

    def validation_step(self, batch, batch_idx):
        """
        PyTorch Lightning validation_step override.
        """
        assert not self.training
        batch_results = self._common_step(batch, split="val")
        mean_loss = batch_results["mean_loss"]
        self._update_results(batch_results, "val")
        return mean_loss

    def test_step(self, batch, batch_idx):
        """
        PyTorch Lightning test_step override.
        """
        assert not self.training
        batch_results = self._common_step(batch, split="test")
        mean_loss = batch_results["mean_loss"]
        self._update_results(batch_results, "test")
        return mean_loss

    def inference_with_ground_truth_step(self, batch, split="test"):
        """Run inference and compute evaluation metrics against ground truth without loss.

        Performs a forward pass and computes auxiliary metrics (e.g. cos_sim, cos_hun)
        against the ground-truth spectrum. Loss functions are never called, making this
        cheaper than ``_common_step`` for pure inference evaluation.

        Args:
            batch: Input batch dict containing ground-truth spectrum fields
                (``spec_mzs``, ``spec_ints``, ``spec_batch_idxs``, etc.).
            split: Metric split name used to select which auxiliary metrics to compute.

        Returns:
            Dict with predicted spectrum tensors (``pred_mzs``, ``pred_logprobs``,
            ``pred_batch_idxs``), ``oos_prob`` if available, and all auxiliary metric
            values (e.g. ``cos_sim``, ``cos_hun``).
        """
        if self.training:
            warnings.warn("Model is in training mode")

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
        pred_oos_logprobs = pred_d.pop("pred_oos_logprobs", None)

        with th.inference_mode():
            eval_true_mzs, eval_true_probs, eval_true_batch_idxs = self.preproc_spec(
                true_mzs,
                true_ints,
                true_batch_idxs,
                train=False,
                pred=False,
                log_in=False,
                log_out=False,
            )
            eval_true_logprobs = safelog(eval_true_probs, eps=self.hparams.log_min)
            eval_pred_mzs, eval_pred_probs, eval_pred_batch_idxs = self.preproc_spec(
                pred_mzs.detach(),
                pred_logprobs.detach(),
                pred_batch_idxs.detach(),
                train=False,
                pred=True,
                log_in=True,
                log_out=False,
            )
            eval_pred_logprobs = safelog(eval_pred_probs, eps=self.hparams.log_min)

        results = {
            "unique_id": batch["spec_unique_id"],
            "smiles": batch["mol_smiles"],
            "spec_per_group": spec_per_group,
            "spec_per_mol": spec_per_mol,
            "group_per_mol": group_per_mol,
            "true_mzs": eval_true_mzs,
            "true_logprobs": eval_true_logprobs,
            "true_batch_idxs": eval_true_batch_idxs,
            "true_prec_mzs": true_prec_mzs,
            "pred_mzs": eval_pred_mzs,
            "pred_logprobs": eval_pred_logprobs,
            "pred_batch_idxs": eval_pred_batch_idxs,
            **pred_d,
        }
        if pred_oos_logprobs is not None:
            results["oos_prob"] = th.exp(pred_oos_logprobs)

        active_metric_names = self._get_active_metric_names(split)
        need_aux_metrics = len(active_metric_names.intersection(self.auxiliary_metric_names)) > 0
        if need_aux_metrics:
            with th.inference_mode():
                metric_input_d = {
                    "true_mzs": eval_true_mzs,
                    "true_ints": eval_true_probs,
                    "true_logprobs": eval_true_logprobs,
                    "true_batch_idxs": eval_true_batch_idxs,
                    "pred_mzs": eval_pred_mzs,
                    "pred_ints": eval_pred_probs,
                    "pred_logprobs": eval_pred_logprobs,
                    "pred_batch_idxs": eval_pred_batch_idxs,
                    "true_prec_mzs": true_prec_mzs,
                    "split": split,
                }
                for arg in self.extra_metric_args:
                    metric_input_d[arg] = results[arg]
                for arg in self.batch_metric_args:
                    if arg in batch:
                        metric_input_d[arg] = batch[arg]
                metric_output_d = self.metric_fn(**metric_input_d)
            for k, v in metric_output_d.items():
                results[k] = v

        return results

    def inference_step(self, batch):
        """
        Custom inference step for evaluation outside of Lightning's test/val loop.
        This will assume that the model is in eval mode, but will not switch modes.
        """
        if self.training:
            warnings.warn("Model is in training mode")

        # forward pass and get predictions
        pred_d = self.forward(**batch)
        pred_mzs = pred_d.pop("pred_mzs")
        pred_logprobs = pred_d.pop("pred_logprobs")
        pred_batch_idxs = pred_d.pop("pred_batch_idxs")
        pred_oos_logprobs = pred_d.pop("pred_oos_logprobs", None)
        eval_pred_mzs, eval_pred_probs, eval_pred_batch_idxs = self.preproc_spec(
            pred_mzs,
            pred_logprobs,
            pred_batch_idxs,
            train=False,
            pred=True,
            log_in=True,
            log_out=False,
        )

        results = {
            "pred_mzs": eval_pred_mzs,
            "pred_ints": eval_pred_probs,
            "pred_batch_idxs": eval_pred_batch_idxs,
            "pred_oos_logprobs": pred_oos_logprobs,
        }
        # get rest of pred_d
        for k, v in pred_d.items():
            results[k] = v
        return results

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler for Lightning.
        """
        optimizer_cls = get_optimizer_class(self.hparams.optimizer)
        optimizer = optimizer_cls(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
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
        """
        Helper to get the WandbLogger from the list of loggers.
        """
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                return logger
        return None

    def _get_dataset_maps(self) -> tuple[dict[int, str], dict[int, str], dict[str, int]]:
        """Return (idx_to_inst_type, idx_to_prec_type, inst_type_to_idx) from hparams.

        Built deterministically from spec_params, matching BaseDataset logic.

        Returns:
            Tuple of (idx_to_inst_type, idx_to_prec_type, inst_type_to_idx) dicts.
        """
        if hasattr(self, "_cached_ds_maps"):
            return self._cached_ds_maps
        inst_types = sorted(self.hparams.spec_params.get("inst_types", []))
        prec_types = sorted(self.hparams.spec_params.get("prec_types", []))
        idx_to_inst = dict(enumerate(inst_types))
        idx_to_prec = dict(enumerate(prec_types))
        inst_to_idx = {t: idx for idx, t in enumerate(inst_types)}
        object.__setattr__(self, "_cached_ds_maps", (idx_to_inst, idx_to_prec, inst_to_idx))
        return self._cached_ds_maps

    def _should_store_raw_epoch_results(self, split: str) -> bool:
        """Return whether raw spectra/IDs must be retained for epoch-end logging."""
        num_log_images = getattr(self.hparams, f"num_log_{split}_images", 0)
        return num_log_images > 0

    def _should_store_ce_metadata(self, split: str) -> bool:
        """Return whether CE/adduct/instrument metadata must be retained."""
        return bool(self.hparams["log_adduct_inst_ce_metrics"] and split in ("val", "test"))

    def _should_store_metric_vectors(self, split: str) -> bool:
        """Return whether per-example metric tensors must be retained for a split."""
        return bool(
            self._should_store_raw_epoch_results(split)
            or self._should_store_ce_metadata(split)
            or self.hparams.log_hist_metrics
            or self.hparams.track_datapoint_metrics
        )

    def _should_report_metric_std(self) -> bool:
        """Return whether epoch std metrics should be computed and logged."""
        return bool(getattr(self.hparams, "report_metric_std", True))

    def _needs_eval_tensors(self, split: str) -> bool:
        """Return whether eval spectra must be materialized for this split."""
        active_metric_names = self._get_active_metric_names(split)
        need_aux_metrics = len(active_metric_names.intersection(self.auxiliary_metric_names)) > 0
        return bool(need_aux_metrics or self._should_store_raw_epoch_results(split))

    def _get_active_metric_names(self, split: str) -> set[str]:
        """Return the metric names that should be computed and reduced for a split."""
        if split == "train":
            return self.metric_names.difference(self.auxiliary_metric_names)
        return self.metric_names

    def _get_result_keys(self, split: str) -> list[str]:
        """Return the minimal set of batch-result keys required for a split."""
        keys = []
        if self._should_store_raw_epoch_results(split):
            keys.extend(
                [
                    "unique_ids",
                    "smiles",
                    "true_mzs",
                    "true_logprobs",
                    "true_unique_ids",
                    "pred_mzs",
                    "pred_logprobs",
                    "pred_unique_ids",
                ]
            )
        if self._should_store_ce_metadata(split):
            keys.extend(
                [
                    "inst_type_str",
                    "prec_type_str",
                    "ce_primary",
                    "ce_nce",
                    "ce_ace",
                ]
            )
        if self._should_store_metric_vectors(split):
            keys.extend(list(self._get_active_metric_names(split)))
        return keys

    def _get_sample_weights(
        self,
        split: str,
        spec_per_group: th.Tensor,
        spec_per_mol: th.Tensor,
        group_per_mol: th.Tensor,
    ) -> th.Tensor:
        """Return per-sample weights for the configured reduction mode."""
        sample_weight = (
            self.hparams.train_sample_weight
            if split == "train"
            else self.hparams.eval_sample_weight
        )
        if sample_weight == "none":
            return th.ones_like(spec_per_group, dtype=th.float32)
        if sample_weight == "group":
            return 1.0 / spec_per_group
        if sample_weight == "mol":
            return 1.0 / spec_per_mol
        assert sample_weight == "group_mol", sample_weight
        return 1.0 / (spec_per_group * group_per_mol)

    def _accumulate_running_metrics(self, batch_results: dict, split: str) -> None:
        """Accumulate weighted metric moments for online epoch reduction."""
        active_metric_names = self._get_active_metric_names(split)
        report_metric_std = self._should_report_metric_std()
        sample_weights = self._get_sample_weights(
            split,
            batch_results["spec_per_group"],
            batch_results["spec_per_mol"],
            batch_results["group_per_mol"],
        )
        sum_w_d = getattr(self, f"{split}_metric_sum_w")
        sum_wx_d = getattr(self, f"{split}_metric_sum_wx")
        sum_wx2_d = getattr(self, f"{split}_metric_sum_wx2")
        for metric_name in active_metric_names:
            metric = batch_results[metric_name]
            weights = sample_weights.to(dtype=metric.dtype, device=metric.device)
            metric_sum_w = weights.sum().detach()
            metric_sum_wx = (weights * metric).sum().detach()
            if metric_name in sum_w_d:
                sum_w_d[metric_name] = sum_w_d[metric_name] + metric_sum_w
                sum_wx_d[metric_name] = sum_wx_d[metric_name] + metric_sum_wx
                if report_metric_std:
                    metric_sum_wx2 = (weights * metric.square()).sum().detach()
                    sum_wx2_d[metric_name] = sum_wx2_d[metric_name] + metric_sum_wx2
            else:
                sum_w_d[metric_name] = metric_sum_w
                sum_wx_d[metric_name] = metric_sum_wx
                if report_metric_std:
                    metric_sum_wx2 = (weights * metric.square()).sum().detach()
                    sum_wx2_d[metric_name] = metric_sum_wx2

    def _reduce_running_metrics(
        self, split: str
    ) -> tuple[dict[str, th.Tensor], dict[str, th.Tensor]]:
        """Reduce online metric moments into epoch mean/std."""
        mean_metrics, std_metrics = {}, {}
        report_metric_std = self._should_report_metric_std()
        sum_w_d = getattr(self, f"{split}_metric_sum_w")
        sum_wx_d = getattr(self, f"{split}_metric_sum_wx")
        sum_wx2_d = getattr(self, f"{split}_metric_sum_wx2")
        for metric_name in self._get_active_metric_names(split):
            if metric_name not in sum_w_d:
                continue
            sum_w = sum_w_d[metric_name]
            mean = sum_wx_d[metric_name] / sum_w
            mean_metrics[metric_name] = mean
            if report_metric_std:
                var = sum_wx2_d[metric_name] / sum_w - mean.square()
                std_metrics[metric_name] = th.sqrt(th.clamp(var, min=0.0))
        return mean_metrics, std_metrics

    def _reset_running_metrics(self, split: str) -> None:
        """Reset online metric accumulators for a split."""
        setattr(self, f"{split}_metric_sum_w", {})
        setattr(self, f"{split}_metric_sum_wx", {})
        setattr(
            self,
            f"{split}_metric_sum_wx2",
            {} if self._should_report_metric_std() else None,
        )

    def _ensure_datapoint_metric_buffers(self, split: str, metric_names) -> None:
        """Lazily allocate per-datapoint metric buffers for the requested metric names."""
        if not self.hparams.track_datapoint_metrics:
            return
        datapoint_metrics = getattr(self, f"{split}_datapoint_metrics")
        for name in metric_names:
            key = name.replace(".", "-")
            if key not in datapoint_metrics:
                datapoint_metrics[key] = -th.ones([self.max_num_datapoints], dtype=th.float32)

    def _update_results(self, batch_results, split):
        """
        Update the results tracker with new batch results for the given split.
        """
        results_attr = f"{split}_results"
        counter_attr = f"{split}_counter"
        active_metric_names = self._get_active_metric_names(split)
        self._accumulate_running_metrics(batch_results, split)
        # filter keys (filtering first to save time/memory)
        keys = self._get_result_keys(split)
        unique_ids = batch_results.pop("unique_id")
        true_batch_idxs = batch_results.pop("true_batch_idxs", None)
        pred_batch_idxs = batch_results.pop("pred_batch_idxs", None)
        batch_results = {k: v for k, v in batch_results.items() if k in keys}
        # update unique IDs only when raw spectra are retained for image logging
        if (
            self._should_store_raw_epoch_results(split)
            and true_batch_idxs is not None
            and pred_batch_idxs is not None
        ):
            true_unique_ids = unique_ids[true_batch_idxs]
            pred_unique_ids = unique_ids[pred_batch_idxs]
            batch_results["unique_ids"] = unique_ids
            batch_results["true_unique_ids"] = true_unique_ids
            batch_results["pred_unique_ids"] = pred_unique_ids
        # check all metrics
        if self._should_store_metric_vectors(split):
            assert all([metric_name in batch_results for metric_name in active_metric_names])
        if keys:
            # transfer to cpu
            batch_results = to_cpu(batch_results, detach=True)
            # setup results dict
            if getattr(self, results_attr) is None:
                setattr(self, results_attr, {k: list() for k in keys})
            else:
                assert set(keys) == set(getattr(self, results_attr).keys())
            results_dict = getattr(self, results_attr)
            # add to results dict
            for k, v in batch_results.items():
                results_dict[k].append(v)
        # increment counter
        setattr(self, counter_attr, getattr(self, counter_attr) + unique_ids.shape[0])

    def _consolidate_results(self, split):
        """
        Consolidate all batch results for a split, log metrics, and update best metrics.
        """
        results = getattr(self, f"{split}_results")
        active_metric_names = self._get_active_metric_names(split)
        mean_metrics, std_metrics = self._reduce_running_metrics(split)
        if not mean_metrics:
            return
        if results is not None:
            keys = results.keys()
            for k in keys:
                elems = results[k]
                # If elements are tensors, make sure they're all at least 1-d before concat
                if all(isinstance(x, th.Tensor) for x in elems):
                    norm = []
                    for x in elems:
                        if x.dim() == 0:
                            # convert scalar tensor to 1-d so it can be concatenated
                            norm.append(x.unsqueeze(0))
                        else:
                            norm.append(x)
                    # concatenate along first dimension
                    results[k] = th.cat(norm, dim=0)
                else:
                    # expected lists-of-lists for non-tensor entries
                    assert isinstance(elems[0], list)
                    results[k] = flatten_lol(elems)

        if (
            self.hparams["log_adduct_inst_ce_metrics"]
            and split in ("val", "test")
            and results is not None
        ):
            self._log_adduct_inst_ce_table(results, split)
        for k in mean_metrics.keys():
            self.log(
                f"{split}_{k}_epoch/mean",
                mean_metrics[k],
            )
            if self._should_report_metric_std():
                self.log(
                    f"{split}_{k}_epoch/std",
                    std_metrics[k],
                )
        if split != "test":
            # update epoch stats
            mean_metrics_epochs = getattr(self, f"{split}_mean_metrics")
            for k in mean_metrics.keys():
                mean_metrics_epochs[k.replace(".", "-")][self.current_epoch] = mean_metrics[k]
            if self._should_report_metric_std():
                std_metrics_epochs = getattr(self, f"{split}_std_metrics")
                for k in mean_metrics.keys():
                    std_metrics_epochs[k.replace(".", "-")][self.current_epoch] = std_metrics[k]
            # log histograms
            if self.hparams.log_hist_metrics and results is not None:
                wandb_logger = self._get_wandb_logger()
                import wandb

                for k, v in results.items():
                    if k in active_metric_names:
                        log_d = {
                            f"{split}_{k}_hist": wandb.Histogram(v.cpu()),
                            "epoch": self.current_epoch,
                        }
                        if wandb_logger is not None:
                            wandb_logger.experiment.log(log_d)
            # log best metric
            update_datapoint_metrics = False
            checkpoint_metric = (
                self.hparams.checkpoint_metric.removeprefix("train_")
                .removeprefix("val_")
                .removesuffix("/mean")
                .removesuffix("/std")
                .removesuffix("_epoch")
            )
            for k in self.metric_bests:
                if k not in active_metric_names:
                    continue
                mean_metric_epochs = mean_metrics_epochs[k.replace(".", "-")][
                    : self.current_epoch + 1
                ]
                valid_mask = ~th.isnan(mean_metric_epochs)
                if not valid_mask.any():
                    continue
                if self.metric_min_max[k] == "min":
                    argbest_metric = th.argmin(
                        mean_metric_epochs.masked_fill(~valid_mask, float("inf"))
                    )
                elif self.metric_min_max[k] == "max":
                    argbest_metric = th.argmax(
                        mean_metric_epochs.masked_fill(~valid_mask, float("-inf"))
                    )
                else:
                    assert self.metric_min_max[k] is None, self.metric_min_max[k]
                    continue
                mean_metric_best = mean_metric_epochs[argbest_metric]
                self.log(f"{split}_{k}_best/mean", mean_metric_best)
                if self._should_report_metric_std():
                    std_metric_epochs = std_metrics_epochs[k.replace(".", "-")][
                        : self.current_epoch + 1
                    ]
                    std_metric_best = std_metric_epochs[argbest_metric]
                    self.log(f"{split}_{k}_best/std", std_metric_best)
                # if it's the best, update the datapoint metrics
                if k == checkpoint_metric and argbest_metric == self.current_epoch:
                    update_datapoint_metrics = True
            if checkpoint_metric == "epoch":
                assert not update_datapoint_metrics
                update_datapoint_metrics = True
        else:
            update_datapoint_metrics = True

        if (
            self.hparams.track_datapoint_metrics
            and update_datapoint_metrics
            and results is not None
        ):
            datapoint_metrics = getattr(self, f"{split}_datapoint_metrics")
            num_datapoints_p = getattr(self, f"{split}_num_datapoints")
            self._ensure_datapoint_metric_buffers(split, mean_metrics.keys())
            example_key = list(mean_metrics.keys())[0]
            num_datapoints = num_datapoints_p.item()
            if num_datapoints == -1:
                num_datapoints_p[0] = len(results[example_key])
                num_datapoints = num_datapoints_p.item()
            assert num_datapoints <= datapoint_metrics[example_key.replace(".", "-")].shape[0], (
                num_datapoints,
                datapoint_metrics[example_key.replace(".", "-")].shape[0],
            )
            for k in mean_metrics.keys():
                assert len(results[k]) == num_datapoints, (k, len(results[k]), num_datapoints)
                datapoint_metrics[k.replace(".", "-")][:num_datapoints] = results[k]

    def _log_adduct_inst_ce_table(self, results: dict, split: str) -> None:
        """Log per-(adduct×inst, CE-bin) median metrics to wandb.

        Groups spectra by the cross-product of precursor type and instrument type,
        then bins collision energy in steps of 10 (NCE % for FT/IT, ACE eV for
        QTOF).  Each row reports the median, mean, std, 95% CI, min, and max of
        every auxiliary metric together with the per-instrument m/z tolerance.

        - ``val``: logs ``wandb.plot.bar`` — one panel per primary metric, updated
          each epoch, no history accumulation.
        - ``test``: logs a full ``wandb.Table`` for detailed inspection.

        Silently returns if no WandbLogger is present, ``spec_inst_type`` is absent
        from the batch, or CE values are not loaded (``nce``/``ace`` both False).

        Args:
            results: Consolidated results dict from ``_consolidate_results``.
            split: One of ``"val"`` or ``"test"``.
        """
        import pandas as pd

        wandb_logger = self._get_wandb_logger()
        if wandb_logger is None:
            return
        import wandb

        inst_types = results["inst_type_str"]  # list[str], length N
        prec_types = results["prec_type_str"]  # list[str], length N

        if all(it == "unknown" for it in inst_types):
            return  # spec_inst_type not in batch — nothing to group on

        # Use ACE for QTOF (native unit) and NCE for FT so CE bins reflect the values
        # used in split_filters and match human-readable experiment conventions.
        # Falls back to ce_primary when the per-instrument CE is unavailable (all NaN).
        ce_nce = results["ce_nce"].numpy()
        ce_ace = results["ce_ace"].numpy()
        ce_primary = results["ce_primary"].numpy()
        ce_vals = np.where(
            np.array(inst_types) == "QTOF",
            np.where(np.isnan(ce_ace), ce_primary, ce_ace),
            np.where(np.isnan(ce_nce), ce_primary, ce_nce),
        )

        df = pd.DataFrame(
            {
                "inst_type": inst_types,
                "prec_type": prec_types,
                "ce_val": ce_vals,
            }
        )
        for m in self.auxiliary_metric_names:
            if m in results and isinstance(results[m], th.Tensor):
                df[m] = results[m].numpy()

        df = df.dropna(subset=["ce_val"])  # CE not loaded → skip
        if df.empty:
            return

        df["ce_bin"] = (df["ce_val"] // 10 * 10).astype(int)
        df["adduct_inst"] = df["prec_type"] + "_" + df["inst_type"]

        metric_cols = [m for m in self.auxiliary_metric_names if m in df.columns]
        _, _, inst_to_idx = self._get_dataset_maps()
        min_count = self.hparams["log_adduct_inst_ce_min_count"]
        rows = []
        for (adduct_inst, ce_bin), grp in df.groupby(["adduct_inst", "ce_bin"], observed=True):
            if len(grp) < min_count:
                continue

            inst_type = grp["inst_type"].iloc[0]
            if self.inst_tol_lookup is not None and inst_type in inst_to_idx:
                tol = self.inst_tol_lookup[inst_to_idx[inst_type]].item()
            else:
                tol = self.tolerance
            tol_label = f"{tol * 1e6:.0f}ppm" if self.relative else f"{tol:.4f}Da"

            row: dict = {
                "adduct_inst": adduct_inst,
                "ce_bin": f"{ce_bin}-{ce_bin + 10}",
                "inst_tol": tol_label,
                "count": len(grp),
            }
            for m in metric_cols:
                col = grp[m]
                n = len(col)
                mean = col.mean()
                std = col.std(ddof=1) if n > 1 else 0.0
                ci95 = 1.96 * std / (n**0.5) if n > 1 else 0.0
                row[f"{m}_median"] = round(float(col.median()), 4)
                row[f"{m}_mean"] = round(float(mean), 4)
                row[f"{m}_std"] = round(float(std), 4)
                row[f"{m}_ci95"] = round(float(ci95), 4)
                row[f"{m}_min"] = round(float(col.min()), 4)
                row[f"{m}_max"] = round(float(col.max()), 4)
            rows.append(row)

        if not rows:
            return

        columns = list(rows[0].keys())
        table = wandb.Table(columns=columns, data=[list(r.values()) for r in rows])

        if split == "val":
            # One bar-chart panel per metric, updated in-place each epoch.
            # Uses the first metric that has a _median column as the primary view.
            primary = f"{metric_cols[0]}_median" if metric_cols else None
            if primary:
                wandb_logger.experiment.log(
                    {
                        f"val_adduct_inst_ce/{metric_cols[0]}": wandb.plot.bar(
                            table,
                            label="adduct_inst",
                            value=primary,
                            title=f"val {metric_cols[0]} by adduct×inst×CE (ep {self.current_epoch})",
                        ),
                        "epoch": self.current_epoch,
                    }
                )
        else:
            # Full table at test time for detailed inspection.
            wandb_logger.experiment.log({f"{split}_adduct_inst_ce_table": table})

    def _log_images(self, split):
        """
        Log example spectra as images to WandB for the given split.
        """
        results = getattr(self, f"{split}_results")
        num_log_images = getattr(self.hparams, f"num_log_{split}_images")
        if num_log_images <= 0 or results is None:
            return
        counter = getattr(self, f"{split}_counter")
        num_log_images = min(num_log_images, counter)
        wandb_logger = self._get_wandb_logger()
        if wandb_logger is None:
            return
        # randomly sample unique_ids
        unique_ids = th.unique(results["unique_ids"], sorted=True)
        if num_log_images == unique_ids.shape[0]:
            sample_idxs = th.arange(num_log_images)
            sample_unique_ids = unique_ids
        else:
            # use a non-global torch Generator for deterministic selection
            with th_temp_generator(420) as gen:
                sample_idxs = th.randperm(unique_ids.shape[0], generator=gen)[:num_log_images]
            sample_unique_ids = unique_ids[sample_idxs]
        # plot images
        for i in range(num_log_images):
            unique_id = sample_unique_ids[i].item()
            unique_idx = th.nonzero(results["unique_ids"] == unique_id, as_tuple=False).item()
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
            assert np.isclose(np.sum(true_ints), 1.0), np.sum(true_ints)
            assert np.isclose(np.sum(pred_ints), 1.0), np.sum(pred_ints)
            # plot
            data = plot_spectra_sparse(
                true_mzs, true_ints, pred_mzs, pred_ints, smiles, return_data=True
            )
            wandb_logger.log_image(
                key=f"{split}_example_{i}",
                caption=[
                    f"unique_id = {unique_id}, epoch = {self.current_epoch:03d}, loss = {loss:.3f}, cos_sim = {cos_sim:.3f}, wrecall = {wrecall:.3f}"
                ],
                images=[data],
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
        self._reset_running_metrics("train")
        self.train_counter = 0
        # seed dataloader
        # self._seed_dataloader(self.current_epoch+1)

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            # consolidate results
            self._consolidate_results("val")
            # log images
            self._log_images("val")
        # reset
        self.val_results = None
        self._reset_running_metrics("val")
        self.val_counter = 0

    def on_test_epoch_end(self):
        # consolidate results
        self._consolidate_results("test")
        # reset
        self.test_results = None
        self._reset_running_metrics("test")
        self.test_counter = 0

    def _print_grad_norm(self, prefix=None, total_num_params=5):
        if prefix is None:
            prefix = "no_prefix"
        print(f">> {prefix}, {self._cur_batch_size}, {self._max_batch_size}")
        opt = self.optimizers()
        num_params = 0
        for pg in opt.param_groups:
            for p in pg["params"]:
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
        self._cur_batch_weight = 0.0

    def _check_ce_params(self):
        # check merge params
        if self.hparams.spec_params["merge"]:
            assert (not self.hparams.spec_params["nce"]) or self.hparams.spec_params[
                "merge_keep_ces"
            ]
        elif self.hparams.spec_params["nce"] or self.hparams.spec_params["ace"]:
            assert not self.hparams.spec_params["merge_keep_ces"]

    def _seed_dataloader(self, seed):
        generator = th.Generator()
        generator.manual_seed(self.train_dl_seeds[seed].item())
        train_dataloader = self.trainer.train_dataloader
        batch_sampler = train_dataloader.batch_sampler

        # Seed generator and recompute batches on whichever object(s) own them.
        # Walk inner sampler first (so nested pre_compute runs before outer),
        # then the batch_sampler itself. Works for SpecMolFragDynamicBatchSampler
        # (nested sampler owns generator), DualGroupDynamicBatchSampler (owns its
        # own generator), and standard BatchSampler (nested sampler may own generator).
        inner = getattr(batch_sampler, "sampler", None)
        for obj in [inner, batch_sampler]:
            if obj is None:
                continue
            if hasattr(obj, "generator"):
                obj.generator = generator
            if hasattr(obj, "_pre_compute_batches"):
                obj._pre_compute_batches()

    def on_save_checkpoint(self, checkpoint):
        # rng
        checkpoint["rng_states"] = _collect_rng_states(include_cuda=th.cuda.is_available())
        # metrics
        checkpoint["train_mean_metrics"] = self.train_mean_metrics
        checkpoint["val_mean_metrics"] = self.val_mean_metrics
        if self._should_report_metric_std():
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
            checkpoint["rng_states"]["torch.cuda"][0] = checkpoint["rng_states"]["torch.cuda"][
                0
            ].cpu()
        _set_rng_states(checkpoint["rng_states"])
        # metrics
        self.train_mean_metrics = checkpoint["train_mean_metrics"]
        self.val_mean_metrics = checkpoint["val_mean_metrics"]
        if self._should_report_metric_std():
            self.train_std_metrics = checkpoint.get("train_std_metrics", self.train_std_metrics)
            self.val_std_metrics = checkpoint.get("val_std_metrics", self.val_std_metrics)
        if self.hparams.track_datapoint_metrics:
            self.train_num_datapoints = checkpoint["train_num_datapoints"]
            self.val_num_datapoints = checkpoint["val_num_datapoints"]
            self.train_datapoint_metrics = checkpoint["train_datapoint_metrics"]
            self.val_datapoint_metrics = checkpoint["val_datapoint_metrics"]
