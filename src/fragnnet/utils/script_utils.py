import gc
import json
import multiprocessing
import os
import tempfile
import warnings
from dataclasses import dataclass
from itertools import chain
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch as th

from fragnnet.iceberg.pl_model import IcebergIntenPL
from fragnnet.pl_model import FraGNNetPL
from fragnnet.runner import load_config
from fragnnet.utils.formula_utils import (
    formula_delta_to_signed_str,
    formula_row_prec_deltas,
)
from fragnnet.utils.frag_utils import PACKED_DAG_INDEX_DATASET, get_node_feats
from fragnnet.utils.misc_utils import (
    LOG_ZERO,
    ProgressLogger,
    flatten_lol,
    get_best_ckpt_from_wandb,
    scatter_reduce,
    to_device,
)
from fragnnet.utils.nn_utils import decompile_jit_ckpt
from fragnnet.utils.pl_utils import get_pl_hparams
from fragnnet.utils.setup_utils import get_model_cls
from fragnnet.utils.spec_utils import batched_mf1000_normalize


def init_wandb_ckpt(
    run_id: str,
    last_ckpt: bool,
    model_cls: type,
    config_d: dict[str, Any],
) -> th.nn.Module:
    """Initialize model from W&B checkpoint.

    Args:
        run_id: W&B run ID to download from
        last_ckpt: Whether to load the last checkpoint (True) or best checkpoint (False)
        model_cls: Model class to instantiate
        config_d: Configuration dictionary for model initialization

    Returns:
        Initialized and loaded model
    """
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
                    epoch = int(
                        os.path.basename(file.name)
                        .removeprefix("model-epoch=")
                        .removesuffix(".ckpt")
                    )
                    if best_ckpt_wandb_fp is None or epoch > best_epoch:
                        best_ckpt_wandb_fp = ckpt_wandb_fp
                        best_epoch = epoch
            ckpt_wandb_fp = best_ckpt_wandb_fp  # bugfix: assign winning path after loop
        assert ckpt_wandb_fp is not None
        ckpt_file_local = run.file(ckpt_wandb_fp).download(root=temp_dp, replace=False)
        ckpt_fp = ckpt_file_local.name
        ckpt = th.load(ckpt_fp, map_location="cpu", weights_only=False)
        # decompile
        assert not config_d["compile"]
        ckpt = decompile_jit_ckpt(ckpt)
        # init
        model = model_cls(**config_d)
        try:
            model.load_state_dict(ckpt["state_dict"], strict=True)
        except RuntimeError:
            print(f"> error when loading from checkpoint id {run_id}, will try with strict=False")
            model.load_state_dict(ckpt["state_dict"], strict=False)
    return model


def select_model_vals(
    model_to_vals: dict[str, dict[str, th.Tensor]],
    keys: list[str],
    stack_dim: int = 0,
) -> dict[str, th.Tensor]:
    """Select and stack specific keys from model value dictionary.

    Args:
        model_to_vals: Dictionary mapping model names to value dictionaries
        keys: List of keys to extract from each value dictionary
        stack_dim: Dimension along which to stack values (default: 0)

    Returns:
        Dictionary with selected keys stacked into tensors
    """
    select_d = {k: [] for k in keys}
    for _, vals in model_to_vals.items():
        for k in keys:
            select_d[k].append(vals[k])
    select_d = {k: th.stack(v, dim=stack_dim) for k, v in select_d.items()}
    return select_d


def log_mean(
    log_p: th.Tensor,
    dim: int = 0,
) -> th.Tensor:
    """Compute logarithm of mean in log-probability space.

    Computes: log(mean(exp(log_p))) = logsumexp(log_p) - log(size).

    Args:
        log_p: Log-probability tensor
        dim: Dimension along which to compute mean (default: 0)

    Returns:
        Log of mean along specified dimension
    """
    return th.logsumexp(log_p, dim=dim) - np.log(log_p.shape[dim])


def log_std(
    log_p: th.Tensor,
    dim: int = 0,
    correction: int = 1,
) -> th.Tensor:
    """Compute logarithm of standard deviation in log-probability space.

    Computes standard deviation using Bessel's correction.

    Args:
        log_p: Log-probability tensor
        dim: Dimension along which to compute standard deviation (default: 0)
        correction: Bessel's correction factor (default: 1 for unbiased estimate)

    Returns:
        Log of standard deviation along specified dimension
    """
    support_size = log_p.shape[dim]
    log_mean_p = log_mean(log_p, dim=dim)
    log_mean_p_sq = log_mean(2 * log_p, dim=dim)
    log_var = th.log(th.exp(log_mean_p_sq) - th.exp(2 * log_mean_p))
    return 0.5 * (log_var + np.log(support_size) - np.log(support_size - correction))


def sparse_kl(
    log_p: th.Tensor,
    log_q: th.Tensor,
    b_idx: th.Tensor,
    dim_size: int,
) -> th.Tensor:
    """Compute sparse Kullback-Leibler divergence.

    Computes KL(P || Q) using scatter reduction for sparse operations.
    KL is clamped to [0, inf) to handle numerical precision issues.

    Args:
        log_p: Log-probability tensor for distribution P
        log_q: Log-probability tensor for distribution Q
        b_idx: Batch indices for scatter reduction
        dim_size: Size of dimension for scatter reduction

    Returns:
        KL divergence per batch element
    """
    p = th.exp(log_p)
    kl = scatter_reduce(
        p * (log_p - th.clamp(log_q, min=LOG_ZERO(log_p.dtype))),
        b_idx,
        reduce="sum",
        dim_size=dim_size,
    )
    kl = th.clamp(kl, min=0.0)
    return kl


def pearson_r(
    x: th.Tensor,
    y: th.Tensor,
    dim: int = 0,
) -> th.Tensor:
    """Compute Pearson correlation coefficient.

    Args:
        x: First tensor
        y: Second tensor
        dim: Dimension along which to compute correlation (default: 0)

    Returns:
        Pearson correlation coefficient
    """
    x_mean = th.mean(x, dim=dim)
    y_mean = th.mean(y, dim=dim)
    x_std = th.std(x, dim=dim)
    y_std = th.std(y, dim=dim)
    r = th.mean((x - x_mean) * (y - y_mean), dim=dim) / (x_std * y_std)
    return r


def setup_device(
    config_d: dict[str, Any],
    device: str | None = None,
    enable_cpu_optimizations: bool = True,
) -> th.device:
    """
    Prepare and configure the compute device for inference.

    Args:
            config_d: Configuration dictionary (should contain 'accelerator' key)
            device: Device string (e.g., "cpu", "cuda:0"). Auto-detects if None.
            enable_cpu_optimizations: Whether to enable CPU-specific optimizations

    Returns:
            Configured torch device
    """
    if device is None:
        if config_d.get("accelerator") == "gpu" and th.cuda.is_available():
            n_devices = th.cuda.device_count()
            if n_devices != 1:
                print(
                    f"Warning: {n_devices} CUDA devices visible; using cuda. "
                    "Set CUDA_VISIBLE_DEVICES to a single GPU or MIG UUID to be explicit."
                )
            th_device = th.device("cuda")
        else:
            th_device = th.device("cpu")
    elif not th.cuda.is_available() and device != "cpu":
        raise RuntimeError(f"Requested device '{device}' but CUDA is not available.")
    else:
        th_device = th.device(device)

    print(f"> Using device: {th_device}")

    # CPU-specific optimizations
    if th_device.type == "cpu" and enable_cpu_optimizations:
        th.set_num_threads(multiprocessing.cpu_count())
        th.multiprocessing.set_sharing_strategy("file_system")
        th.use_deterministic_algorithms(True)
        print(f"  - CPU threads: {multiprocessing.cpu_count()}")
        print("  - Deterministic mode: enabled")

    return th_device


def get_ckpt_path(
    wandb_run_id: str | None,
    model_ckpt: str | None,
    model_save_dp: str,
    use_cached: bool = True,
) -> str:
    """
    Get checkpoint path either from local file or download from W&B.

    Args:
            wandb_run_id: W&B run ID to download from
            model_ckpt: Local checkpoint path
            model_save_dp: Directory to save downloaded checkpoints
            use_cached: Whether to use cached W&B downloads

    Returns:
            Path to checkpoint file
    """
    if model_ckpt is None or not os.path.isfile(model_ckpt):
        # Download from W&B
        os.makedirs(model_save_dp, exist_ok=True)
        print(f">> Downloading model to {model_save_dp} from W&B")
        print(f">> W&B run id: {wandb_run_id}")
        ckpt_fp = get_best_ckpt_from_wandb(model_save_dp, wandb_run_id, use_cached=use_cached)
    else:
        # Use local checkpoint
        ckpt_fp = model_ckpt
        print(f">> Using local checkpoint: {ckpt_fp}")

    return ckpt_fp


def init_model_from_ckpt(
    ckpt_fp: str,
    config_d: dict[str, Any],
    device: th.device,
    map_location: str | None = None,
) -> th.nn.Module:
    """
    Initialize and load model from checkpoint file.
    Args:
            ckpt_fp: Path to checkpoint file
            config_d: Configuration dictionary
            device: Device to load model on
            map_location: Location to map checkpoint tensors (defaults to device)

    Returns:
            Loaded model
    """
    if map_location is None:
        map_location = str(device)

    # Load checkpoint
    print(f">> Loading checkpoint from: {ckpt_fp}")
    ckpt = th.load(ckpt_fp, map_location=map_location, weights_only=False)

    # Use saved compilation metadata: inference setup already disables compilation
    # in config_d, but compiled checkpoints still contain _orig_mod state keys.
    ckpt = decompile_jit_ckpt(ckpt)
    config_d["compile"] = False
    # Get model class and initialize
    model_type = config_d["model_type"]
    print(f">> Model type: {model_type}")
    model_cls = get_model_cls(model_type)

    model = model_cls(**config_d)

    # Inference requires complete trained weights; partial loads can leave random layers.
    model.load_state_dict(ckpt["state_dict"], strict=True)
    print(">> Model loaded successfully (strict=True)")

    model.to(device)
    model.eval()

    return model


def get_config_from_ckpt(
    ckpt_fp: str,
    custom_fp: str | None = None,
    template_fp: str | None = None,
) -> dict[str, Any]:
    """
    Load model and config from checkpoint file (using checkpoint hparams).

    This is the recommended approach for MS2C prediction scripts or when you
    want to use the exact configuration that was used during training.

    Args:
            ckpt_fp: Path to checkpoint file
            custom_fp: Path to custom config YAML to override checkpoint config
            template_fp: Path to template config YAML (required if custom_fp provided)

    Returns:
            Configuration dictionary
    """
    # Load checkpoint to extract config
    print(">> Loading checkpoint to extract config")
    ckpt = th.load(ckpt_fp, map_location="cpu", weights_only=False)

    print(">> Extracting config from checkpoint hparams")
    config_d = get_pl_hparams(ckpt)

    # Override with custom config if provided
    if custom_fp is not None:
        assert template_fp is not None, "template_fp required when custom_fp is provided"
    # Load and merge template and custom config
    if template_fp is not None and custom_fp is not None:
        print(f">> Loading custom config from {custom_fp}")
        config_d_2 = load_config(template_fp, custom_fp, override=None)
        for k, v in config_d_2.items():
            if k in config_d and v != config_d[k]:
                print(f"  - Overwriting: {k} = {v} (was {config_d[k]})")
            elif k not in config_d:
                print(f"  - Adding: {k} = {v}")
            config_d[k] = v
    return config_d


# ============================================================================
# Inference Configuration and Setup Utilities
# ============================================================================
#
# These functions provide modular utilities for inference evaluation scripts.
# They complement load_model_and_init_config() but serve different use cases:
#
# ============================================================================


def apply_inference_config(
    config_d: dict[str, Any],
    auxiliary_scores: list | None = None,
    skip_extra_losses: bool | None = None,
    disable_preproc: bool | None = False,
    eval_mz_bin_res: list | None = None,
    dynamic_batch_sampler: bool | None = None,
    num_workers: int | None = None,
    output_formula_str: bool = False,
    train_batch_size: int | None = None,
    eval_batch_size: int | None = None,
    drop_empty_spectra: bool | None = None,
    # Set dataframes in config for dataset initialization
    mol_src: pd.DataFrame | str | None = None,
    spec_src: pd.DataFrame | str | None = None,
) -> dict[str, Any]:
    """Apply inference-specific configuration modifications.

    Modifies configuration dictionary for inference evaluation, including batch sizes,
    preprocessing settings, output options, and model-specific parameters.

    Args:
        config_d: Configuration dictionary to modify
        auxiliary_scores: List of auxiliary score types to compute
        skip_extra_losses: Whether to skip extra loss terms
        disable_preproc: Whether to disable preprocessing in all modules
        eval_mz_bin_res: M/z bin resolutions for evaluation scoring
        dynamic_batch_sampler: Whether to use dynamic batch sampling
        num_workers: Number of data loading workers
        output_formula_str: Whether to output formula strings
        train_batch_size: Training batch size override
        eval_batch_size: Evaluation batch size override
        drop_empty_spectra: Override for spec_params.drop_empty_spectra. Pass False for
            prediction-only runs, whose spec_df carries no ground-truth peaks. None leaves
            the checkpoint's setting alone (evaluation runs score against real spectra).
        mol_src: DataFrame or path to molecule source data
        spec_src: DataFrame or path to spectrum source data

    Returns:
        Modified configuration dictionary
    """
    # Common configuration modifications
    if auxiliary_scores is not None:
        config_d["auxiliary_scores"] = auxiliary_scores
    if skip_extra_losses is not None:
        config_d["skip_extra_losses"] = skip_extra_losses
    config_d["compile"] = False  # Disable compilation for evaluation
    # Disable preprocessing if requested
    if disable_preproc:
        for k in ["spec", "mol", "frag", "magma", "ann"]:
            key = f"{k}_params"
            config_d[key]["preprocess"] = False
            if k in ["frag"]:
                config_d[key]["preload"] = False

    # Handle formula string outputs
    config_d["output_formula_str"] = output_formula_str

    # Model-specific configurations
    # for fragnnet
    config_d["spec_params"]["prec_type_str"] = True
    config_d["frag_params"]["formula_str"] = True
    # for iceberg
    config_d["magma_params"]["adduct_form_deltas"] = True
    # for graff
    config_d["ann_params"]["formula_str"] = output_formula_str
    # Set batch sizes if provided
    if train_batch_size is not None:
        config_d["train_batch_size"] = train_batch_size
    if eval_batch_size is not None:
        config_d["eval_batch_size"] = eval_batch_size

    # config_d['auxiliary_scores'] = auxiliary_scores if auxiliary_scores is not None else []
    if eval_mz_bin_res is not None:
        config_d["eval_mz_bin_res"] = eval_mz_bin_res

    # Ground-truth peak cleanup is meaningless without ground-truth peaks: an inference
    # spec_df has `peaks == []` by construction, so drop_empty_spectra (True in
    # config/template.yml) removes every spectrum and the run silently predicts nothing.
    # Only override when the checkpoint config carries the cleanup group - writing the key
    # into a legacy config that omits it would trip the missing-cleanup-params check in
    # BaseDataset._setup_dfs.
    if drop_empty_spectra is not None and "drop_empty_spectra" in config_d["spec_params"]:
        config_d["spec_params"]["drop_empty_spectra"] = drop_empty_spectra

    # Apply common inference settings
    config_d["track_datapoint_metrics"] = False

    # Disable preprocessing to save memory
    config_d["frag_params"]["preload"] = False
    config_d["frag_params"]["preprocess"] = False
    config_d["magma_params"]["preprocess"] = False

    # Disable samplers for deterministic inference
    config_d["dynamic_batch_sampler"] = (
        dynamic_batch_sampler if dynamic_batch_sampler is not None else False
    )
    config_d["group_sampler"] = False
    config_d["simple_group_sampler"] = False

    if num_workers is not None:
        config_d["num_workers"] = num_workers

    # Set dataframes in config for dataset initialization
    if mol_src is not None:
        config_d["mol_fp"] = mol_src
    if spec_src is not None:
        config_d["spec_fp"] = spec_src
    return config_d


def save_inference_results(
    vals: Any,
    save_dp: str,
    model_type: str,
    seed: int,
    eval_split: str,
    force_overwrite: bool = False,
    split_type: str | None = None,
) -> None:
    """
    Save inference results to disk with proper directory structure.

    Args:
            vals: Results to save (typically a dictionary or DataFrame)
            save_dp: Base directory to save results
            model_type: Model type identifier
            seed: Random seed number
            eval_split: Dataset split name
            force_overwrite: Whether to overwrite existing files
            split_type: Optional split type (e.g., "inchikey", "scaffold")
    """
    # Build save path
    if split_type is not None:
        seed_save_dp = os.path.join(save_dp, split_type, model_type, f"s{seed}")
    else:
        seed_save_dp = os.path.join(save_dp, model_type, f"s{seed}")

    os.makedirs(seed_save_dp, exist_ok=True)
    seed_vals_fp = os.path.join(seed_save_dp, f"{eval_split}.pkl")

    # Save with overwrite logic
    if os.path.isfile(seed_vals_fp):
        print(f"> Info: {seed_vals_fp} exists already")
        if force_overwrite:
            print(f"> Warning: overwriting {seed_vals_fp}")
            th.save(vals, seed_vals_fp)
        else:
            print("> Skipping save (use --force_overwrite to overwrite)")
    else:
        th.save(vals, seed_vals_fp)
        print(f"> Saved results to {seed_vals_fp}")


def cleanup_model(model: th.nn.Module, use_gpu: bool = True) -> None:
    """
    Clean up model and free memory. This is useful after inference crashed to avoid memory leaks.

    Args:
            model: Model to clean up
            use_gpu: Whether GPU memory should be cleared
    """
    model.cpu()
    del model
    gc.collect()

    if use_gpu:
        with th.no_grad():
            th.cuda.empty_cache()


# -----------------------------------------------------------------------------
# The export schema
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnSpec:
    """One column of the inference export.

    Attributes:
        name: Column name in the DataFrame and dataset name in HDF5.
        family: Which axis the column is ragged along, or "" for one value per spectrum.
            A ragged family's members all carry a per-spectrum block and share one index
            pair; the four families' block lengths are unrelated.
        token: The `output_subset` token that requests this column, or None for a
            structural column (identifiers, and the index arrays that make a family's
            probabilities interpretable) that is always emitted.
        flag: Name of the `process_batch_to_tabular_row` boolean that gates this column
            instead of a token, for the two annotation columns.
    """

    name: str
    family: str = ""
    token: str | None = None
    flag: str | None = None


# Rows per HDF5 chunk. Compression works per chunk, so a chunk large enough to hold long
# runs of similar values is what turns the index columns from 8 bytes per small integer
# into ~0.5. Measured on a real index column: h5py's auto-chunk (1024 rows) gives 9.3 MB,
# this gives 5.1 MB, uncompressed 80 MB.
_HDF5_CHUNK_LEN = 1 << 16

# HDF5 index-pair prefix per ragged family. Readers already expect these exact spellings
# (start_points, node_start_points, formula_start_points, joint_start_points).
RAGGED_FAMILY_PREFIX = {"peak": "", "node": "node_", "formula": "formula_", "joint": "joint_"}

# The single source of truth for the export: column order, HDF5 family grouping, and the
# `output_subset` vocabulary are all read off this table, so they cannot drift apart.
# Order here is the column order of the emitted DataFrame, ahead of the eval-score columns.
COLUMN_SPECS = (
    ColumnSpec("spec_id"),
    ColumnSpec("mol_id"),
    ColumnSpec("pred_mzs", "peak", token="pred_mzs"),
    ColumnSpec("pred_ints", "peak", token="pred_ints"),
    ColumnSpec("pred_oos_prob", token="pred_oos_prob"),
    ColumnSpec("frag_formula_peak_idxs", "peak"),
    ColumnSpec("frag_formula_peak_mzs", "peak", token="frag_formula_peak_mzs"),
    ColumnSpec("pred_formula_formula_idxs", "formula"),
    ColumnSpec("pred_formula_probs", "formula", token="pred_formula_probs"),
    ColumnSpec("pred_joint_formula_idxs", "joint", token="pred_joint_formula_idxs"),
    ColumnSpec("pred_formula_node_probs", "joint", token="pred_formula_node_probs"),
    ColumnSpec("pred_joint_node_idxs", "joint", token="pred_joint_node_idxs"),
    ColumnSpec("pred_node_formula_probs", "joint", token="pred_node_formula_probs"),
    ColumnSpec("pred_node_probs", "node", token="pred_node_probs"),
    ColumnSpec("pred_node_node_idxs", "node"),
    ColumnSpec("index_rebased"),
    ColumnSpec("pred_formula_strs", "formula", flag="return_formula_str"),
    ColumnSpec("pred_formula_adducts", "formula", flag="return_annotation"),
    ColumnSpec("pred_node_ccs", "node", flag="return_cc_hash"),
)

# Valid tokens a user can request in `output_subset` for inference outputs.
OUTPUT_SUBSET_OPTIONS = [c.token for c in COLUMN_SPECS if c.token is not None]


def selected_column_specs(return_formula_str: bool, return_cc_hash: bool) -> list[ColumnSpec]:
    """The columns this run emits, in order.

    `output_subset` deliberately does NOT narrow this list: a column outside the subset is
    still emitted, holding NaN, and `--drop_none_cols` is what removes it. Both the column
    names and the row values are built from this one function, so a row can never be
    misaligned against its header.
    """
    flags = {
        "return_formula_str": return_formula_str,
        "return_cc_hash": return_cc_hash,
        # `pred_formula_adducts` is what makes either annotation resolvable: it turns a
        # neutral `pred_formula_strs` entry into the observed ion, and tells a `pred_node_ccs`
        # bitmask which adduct that fragment carries on the CRF/CMF row it was reached by.
        "return_annotation": return_formula_str or return_cc_hash,
    }
    return [c for c in COLUMN_SPECS if c.flag is None or flags[c.flag]]


# Probability tokens are exp() of the corresponding raw log-prob model output key.
# The raw FraGNNet output (fragnnet_model out_d) is emitted in log space, while the
# `output_subset` vocabulary above uses probability names. `pred_ints` is handled
# separately because it also needs the intensity untransform. This mirrors the
# derivations in `process_batch_to_tabular_row`.
_PROB_TOKEN_TO_LOGPROB_KEY = {
    "pred_oos_prob": "pred_oos_logprobs",
    "pred_formula_probs": "pred_formula_logprobs",
    "pred_formula_node_probs": "pred_formula_node_logprobs",
    "pred_node_formula_probs": "pred_node_formula_logprobs",
    "pred_node_probs": "pred_node_logprobs",
}


def list_output_subset_options() -> list[str]:
    """List all valid output subset option tokens.

    Returns:
        List of valid output subset tokens that can be used in inference
    """
    return OUTPUT_SUBSET_OPTIONS


def prepare_output_subset(
    output_everything: bool,
    output_subset: set[str] | None,
    model_metric_names: set[str] | None = None,
) -> set[str] | None:
    """
    Prepare output subset for inference results.

    Args:
            output_everything: Whether to output all metrics
            output_subset: Initial output subset
            model_metric_names: Model-specific metric names to include

    Returns:
            Final output subset (None if output_everything is True)
    """
    if output_everything:
        return None

    if output_subset is None:
        output_subset = set()
    else:
        output_subset = set(output_subset)

    if model_metric_names is not None:
        output_subset = output_subset.union(model_metric_names)

    return output_subset


def required_index_vars_for_outputs(
    output_subset: set[str] | None,
    model,
    nb_iso: bool = False,
    return_formula_str: bool = False,
    return_cc_hash: bool = False,
) -> tuple[set[str], set[str], set[str], set[str]]:
    """
    Given desired `output_subset` tokens, determine which index variable arrays are
    required to be present in batch/model outputs.

    `return_formula_str` / `return_cc_hash` are the annotation columns. They are
    addressed by the same index arrays as the P(f) / P(n) tokens but carry no
    probability of their own, so they are requested separately - otherwise
    "annotation without probabilities" is unreachable and the annotation columns
    come out as NaN.

    Returns sets of required names for: batch, node, formula, nb_node indices.
    """
    # If no subset specified, assume all indices may be needed
    if output_subset is None:
        output_subset = set(OUTPUT_SUBSET_OPTIONS)

    # Base mapping from output tokens -> required index vars
    # Note: These names correspond to keys produced by model steps and consumed
    # in `process_batch_to_tabular_row`.
    token_to_idx = {
        "pred_mzs": {"batch": {"pred_batch_idxs"}},
        "pred_ints": {"batch": {"pred_batch_idxs"}},
        "pred_oos_prob": {},
        "frag_formula_peak_mzs": {"batch": {"pred_batch_idxs"}},
        # P(f)
        "pred_formula_probs": {
            "batch": {"pred_formula_batch_idxs"},
            "formula": {"pred_formula_formula_idxs"},
        },
        # P(n|f) on joint grid, requires joint idxs and joint batch.
        # The P(f) idxs come along because `batch_filter_by_peak_intensity` builds the
        # joint keep-mask from `pred_formula_formula_idxs`; without them it raises
        # NameError as soon as `min_peak_int > 0`.
        "pred_formula_node_probs": {
            "batch": {"pred_joint_batch_idxs", "pred_formula_batch_idxs"},
            "formula": {"pred_joint_formula_idxs", "pred_formula_formula_idxs"},
            "node": {"pred_joint_node_idxs"},
        },
        # P(f|n) on joint grid
        "pred_node_formula_probs": {
            "batch": {"pred_joint_batch_idxs", "pred_formula_batch_idxs"},
            "formula": {"pred_joint_formula_idxs", "pred_formula_formula_idxs"},
            "node": {"pred_joint_node_idxs"},
        },
        # Joint (formula, node) support with no probability attached: which DAG nodes
        # could explain each surviving formula/peak.
        "pred_joint_formula_idxs": {
            "batch": {"pred_joint_batch_idxs", "pred_formula_batch_idxs"},
            "formula": {"pred_joint_formula_idxs", "pred_formula_formula_idxs"},
            "node": {"pred_joint_node_idxs"},
        },
        "pred_joint_node_idxs": {
            "batch": {"pred_joint_batch_idxs", "pred_formula_batch_idxs"},
            "formula": {"pred_joint_formula_idxs", "pred_formula_formula_idxs"},
            "node": {"pred_joint_node_idxs"},
        },
        # P(n)
        "pred_node_probs": {
            "batch": {"pred_node_batch_idxs"},
            "node": {"pred_node_node_idxs"},
        },
    }

    # Aggregate required idx names across tokens
    req_batch, req_node, req_formula, req_nb_node = set(), set(), set(), set()
    for tok in output_subset:
        m = token_to_idx.get(tok, {})
        req_batch |= m.get("batch", set())
        req_node |= m.get("node", set())
        req_formula |= m.get("formula", set())
        req_nb_node |= m.get("nb_node", set())

    # Annotation columns. `pred_formula_strs` is looked up with
    # `pred_formula_formula_idxs`, `pred_node_ccs` with `pred_node_node_idxs`; both are
    # otherwise only requested by their probability token.
    if return_formula_str:
        req_batch |= {"pred_formula_batch_idxs"}
        req_formula |= {"pred_formula_formula_idxs"}
    if return_cc_hash:
        # cc hashes are per DAG node. Pruning them to the nodes behind a surviving peak
        # needs the joint (formula, node) grid, and the joint keep-mask is in turn built
        # from the P(f) indices - so all three come along. The joint index columns are
        # also what links a node back to a peak downstream.
        req_batch |= {
            "pred_node_batch_idxs",
            "pred_joint_batch_idxs",
            "pred_formula_batch_idxs",
        }
        req_node |= {"pred_node_node_idxs", "pred_joint_node_idxs"}
        req_formula |= {"pred_joint_formula_idxs", "pred_formula_formula_idxs"}

    # Model-specific adjustments
    is_fragnnet = isinstance(model, FraGNNetPL)
    is_iceberg = isinstance(model, IcebergIntenPL)

    if is_iceberg:
        # Iceberg does not use node/formula idxs in inference outputs
        req_node.clear()
        req_formula.clear()
        req_nb_node.clear()

    if is_fragnnet and nb_iso:
        # If neighbor isotope predictions are enabled, include NB index vars
        req_batch |= {
            "pred_nb_node_batch_idxs",
            "pred_nb_joint_batch_idxs",
            "pred_nb_node_node_batch_idxs",
        }
        req_formula |= {"pred_nb_joint_formula_idxs"}
        req_nb_node |= {
            "pred_nb_node_node_idxs",
            "pred_nb_joint_node_idxs",
            "pred_nb_node_node_node_idxs",
        }

    return req_batch, req_node, req_formula, req_nb_node


def get_required_config_indices(
    output_subset: set[str] | None,
    model,
    nb_iso: bool = False,
    return_formula_str: bool = False,
    return_cc_hash: bool = False,
) -> tuple[set[str], set[str], set[str], set[str]]:
    """
    Public API: Given `output_subset` tokens, return required index variable names
    grouped by category (batch, node, formula, nb_node). Useful for configuring
    dataset/model outputs to ensure all needed indices are present.
    """
    return required_index_vars_for_outputs(
        output_subset, model, nb_iso, return_formula_str, return_cc_hash
    )


def get_idx_vars(
    output_subset: set[str] | None,
    model,
    nb_iso: bool = False,
    return_formula_str: bool = False,
    return_cc_hash: bool = False,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Get final output keys based on subset selection.

    Args:
            output_subset: Subset of output keys to include (None = all)
            default_output_keys: Default set of output keys

    Returns:
            Final set of output keys
    """
    # Determine which index variables to track
    fraggnn_model = isinstance(model, FraGNNetPL)
    iceberg_model = isinstance(model, IcebergIntenPL)
    if not (fraggnn_model or iceberg_model):
        assert not nb_iso
        BATCH_IDX_VARS = ["pred_batch_idxs", "true_batch_idxs"]
        NODE_IDX_VARS = []
        FORMULA_IDX_VARS = []
        NB_NODE_IDX_VARS = []
    elif iceberg_model:
        assert not nb_iso
        BATCH_IDX_VARS = ["pred_batch_idxs", "true_batch_idxs"]
        if model.model.output_formula_str:
            BATCH_IDX_VARS.extend(["pred_formula_batch_idxs"])
        NODE_IDX_VARS = []
        FORMULA_IDX_VARS = []
        NB_NODE_IDX_VARS = []
    else:  # fraggnn_model
        BATCH_IDX_VARS = [
            "pred_batch_idxs",
            "true_batch_idxs",
            "pred_formula_batch_idxs",
            "pred_node_batch_idxs",
            "pred_joint_batch_idxs",
        ]
        NODE_IDX_VARS = ["pred_node_node_idxs", "pred_joint_node_idxs"]
        FORMULA_IDX_VARS = ["pred_formula_formula_idxs", "pred_joint_formula_idxs"]
        NB_NODE_IDX_VARS = []
        if nb_iso:
            BATCH_IDX_VARS.extend(
                [
                    "pred_nb_node_batch_idxs",
                    "pred_nb_joint_batch_idxs",
                    "pred_nb_node_node_batch_idxs",
                ]
            )
            FORMULA_IDX_VARS.extend(["pred_nb_joint_formula_idxs"])
            NB_NODE_IDX_VARS.extend(
                [
                    "pred_nb_node_node_idxs",
                    "pred_nb_joint_node_idxs",
                    "pred_nb_node_node_node_idxs",
                ]
            )

    if output_subset is not None:
        req_batch, req_node, req_formula, req_nb_node = required_index_vars_for_outputs(
            output_subset, model, nb_iso, return_formula_str, return_cc_hash
        )
        BATCH_IDX_VARS = [k for k in BATCH_IDX_VARS if k in req_batch]
        NODE_IDX_VARS = [k for k in NODE_IDX_VARS if k in req_node]
        FORMULA_IDX_VARS = [k for k in FORMULA_IDX_VARS if k in req_formula]
        NB_NODE_IDX_VARS = [k for k in NB_NODE_IDX_VARS if k in req_nb_node]

    return BATCH_IDX_VARS, NODE_IDX_VARS, FORMULA_IDX_VARS, NB_NODE_IDX_VARS


def get_score_types(config_d: dict[str, Any]) -> list[str]:
    """Generate a list of evaluation score types based on the configuration dictionary.

    Args:
        config_d: Configuration dictionary containing 'auxiliary_scores' and 'eval_mz_bin_res'

    Returns:
        List of evaluation score types
    """
    eval_score_types = []
    for score_type in config_d["auxiliary_scores"]:
        if "_hun" in score_type:
            # Hungarian algorithm scores don't need binning
            eval_score_types.append(score_type)
            if config_d["eval_hun_sqrt"]:
                eval_score_types.append(score_type + "_sqrt")
        else:
            # Other scores need binning by m/z resolution
            for bin_res in config_d["eval_mz_bin_res"]:
                eval_score_types.append(f"{score_type}_{bin_res}")
                if config_d["eval_bin_sqrt"]:
                    eval_score_types.append(f"{score_type}_sqrt_{bin_res}")
    return eval_score_types


def get_tabular_columns(
    config_d: dict[str, Any],
    return_formula_str: bool = False,
    return_cc_hash: bool = False,
    output_subset: set[str] | None = None,
) -> list[str]:
    """
    Get column names for tabular inference output.

    Args:
            config_d: Configuration dictionary
            return_formula_str: Whether output includes the formula-string annotation column
            return_cc_hash: Whether output includes the node cc-hash annotation column
            output_subset: Optional subset of output tokens. Checked for unknown tokens but
                    NOT used to narrow the column list - see `selected_column_specs`.

    Returns:
            List of column names for DataFrame
    """
    columns_name = [c.name for c in selected_column_specs(return_formula_str, return_cc_hash)]
    columns_name.extend(get_score_types(config_d))

    if output_subset is not None:
        missing_cols = [c for c in output_subset if c not in columns_name]
        if len(missing_cols) > 0:
            print(
                f"Warning: {len(missing_cols)} columns in output_subset are not in available columns: {missing_cols}"
            )

    return columns_name


def run_inference_pkl(
    model: th.nn.Module,
    dataloader: th.utils.data.DataLoader,
    device: th.device,
    batch_cutoff: int = -1,
    nb_iso: bool = False,
    output_set: set[str] | None = None,
    evaluate: bool = True,
) -> dict[str, Any]:
    """Run inference and return pickled dictionary of results.

    Executes inference on all (or a subset of) batches, with optional index rebatching
    to consolidate results across batches into a single output dictionary.

    Args:
        model: Model instance for inference
        dataloader: DataLoader providing batches
        device: Device to run inference on
        batch_cutoff: Maximum number of batches to process (-1 for all)
        nb_iso: Whether model uses neighbor isotope predictions
        output_set: Subset of output keys to include (None for all)
        evaluate: Whether to compute evaluation metrics using ground truth

    Returns:
        Dictionary with concatenated inference results from all batches
    """
    # setup accumulators and index vars
    cum_num_datapoints = 0
    cum_num_nodes = 0
    cum_num_formulas = 0
    cum_num_nb_nodes = 0

    is_fragnnet_model = isinstance(model, FraGNNetPL)
    BATCH_IDX_VARS, NODE_IDX_VARS, FORMULA_IDX_VARS, NB_NODE_IDX_VARS = get_idx_vars(
        output_set,
        model,
        nb_iso,
    )

    # Setup for evaluation mode (return dict)
    vals = {}
    # Common setup
    print(f"\n>>> Starting inference on {len(dataloader)} batches...")
    if model.training:
        raise RuntimeError(f"> Warning: model {model} is in training mode!")

    # Run inference loops
    total_batches = len(dataloader) if batch_cutoff < 0 else min(len(dataloader), batch_cutoff)
    progress = ProgressLogger(total=total_batches, desc="Running inference")
    with th.inference_mode():
        for batch_idx, batch_input in enumerate(dataloader):
            # Early exit for batch cutoff
            if batch_idx >= total_batches:
                break
            progress.update()

            # Move batch to device
            batch_input = to_device(batch_input, device, non_blocking=False)

            # Enable explainability tensors only when formula/node idx vars are needed
            _need_expl = bool(
                {"pred_formula_formula_idxs"} & set(FORMULA_IDX_VARS)
                or {"pred_node_node_idxs"} & set(NODE_IDX_VARS)
            )
            if is_fragnnet_model:
                # Single switch: tell the model whether this export will read the marginals.
                # Passing it either way is the point - the model defaults to True, so only
                # an explicit False buys the skip.
                batch_input["return_explainability_tensors"] = _need_expl

            # Run inference. `evaluate` only adds ground-truth scoring; the predicted
            # intensities have the same meaning either way.
            b_output = model.inference_step(
                batch_input, metrics="test" if evaluate else None
            )

            b_output = to_device(b_output, "cpu", non_blocking=False)
            if output_set is not None:
                # Translate user-facing `output_subset` tokens into concrete model
                # output columns. The raw model output uses log-prob keys
                # (e.g. `pred_node_logprobs`), while the token vocabulary uses
                # probability names (e.g. `pred_node_probs`). Probability tokens are
                # exp() of their logprob key; `pred_ints` also applies the intensity
                # untransform (mirrors `process_batch_to_tabular_row`). Index variables
                # are always retained so the flat concatenated columns can be regrouped
                # per-spectrum / per-node downstream.
                all_idx_vars = set(
                    BATCH_IDX_VARS + NODE_IDX_VARS + FORMULA_IDX_VARS + NB_NODE_IDX_VARS
                )
                derived = {k: b_output[k] for k in all_idx_vars if b_output.get(k) is not None}
                for token in output_set:
                    if token == "pred_ints":
                        # `inference_step` always returns L1-normalized intensities, so
                        # the base-peak rescale is all that is owed.
                        derived["pred_ints"] = batched_mf1000_normalize(
                            b_output["pred_ints"], b_output["pred_batch_idxs"]
                        )
                    elif token in _PROB_TOKEN_TO_LOGPROB_KEY:
                        key = _PROB_TOKEN_TO_LOGPROB_KEY[token]
                        if b_output.get(key) is None:
                            raise KeyError(f"output token {token!r} needs model output "
                                           f"{key!r}, which is absent/None")
                        derived[token] = th.exp(b_output[key])
                    elif b_output.get(token) is not None:
                        derived[token] = b_output[token]
                    else:
                        raise KeyError(
                            f"output token {token!r} is not produced by run_inference_pkl; "
                            "use output_format=hdf5 for this token"
                        )
                b_output = derived

            # ============================================================
            # EVALUATION MODE: Collect all outputs with index rebatching
            # ============================================================
            output_keys = b_output.keys()

            # Calculate batch counts for index rebatching. Any per-batch index that
            # ranges over spectra gives the datapoint count, so fall back through them
            # (e.g. a node-only output_subset has no `pred_batch_idxs` but still needs
            # `pred_node_batch_idxs` rebatched across batches).
            b_num_datapoints = 0
            for _batch_key in (
                "pred_batch_idxs",
                "true_batch_idxs",
                "pred_node_batch_idxs",
                "pred_formula_batch_idxs",
                "pred_joint_batch_idxs",
            ):
                v = b_output.get(_batch_key)
                if v is not None and v.numel() > 0:
                    b_num_datapoints = int(v.max()) + 1
                    break

            if is_fragnnet_model and "pred_node_node_idxs" in output_keys:
                b_num_nodes = b_output["pred_node_node_idxs"].max() + 1
            else:
                b_num_nodes = 0

            if is_fragnnet_model and "pred_formula_formula_idxs" in output_keys:
                b_num_formulas = b_output["pred_formula_formula_idxs"].max() + 1
            else:
                b_num_formulas = 0

            if is_fragnnet_model and nb_iso and "pred_nb_node_node_idxs" in output_keys:
                b_num_nb_nodes = b_output["pred_nb_node_node_idxs"].max() + 1
            else:
                b_num_nb_nodes = 0

            # Rebatch indices
            for k in BATCH_IDX_VARS:
                if k in b_output:
                    b_output[k] = b_output[k] + cum_num_datapoints
            for k in NODE_IDX_VARS:
                if k in b_output:
                    b_output[k] = b_output[k] + cum_num_nodes
            for k in FORMULA_IDX_VARS:
                if k in b_output:
                    b_output[k] = b_output[k] + cum_num_formulas
            for k in NB_NODE_IDX_VARS:
                if k in b_output:
                    b_output[k] = b_output[k] + cum_num_nb_nodes

            # Update cumulative counts
            cum_num_datapoints += b_num_datapoints
            cum_num_nodes += b_num_nodes
            cum_num_formulas += b_num_formulas
            cum_num_nb_nodes += b_num_nb_nodes

            # Collect outputs
            for k in b_output.keys():
                if b_output[k] is None:
                    continue
                if k not in vals:
                    assert batch_idx == 0
                    vals[k] = [b_output[k]]
                else:
                    assert batch_idx > 0
                    vals[k].append(b_output[k])
    progress.finish()

    # Consolidate collected outputs
    print(f"> num_datapoints = {cum_num_datapoints}")
    print(f"> num_nodes = {cum_num_nodes}")
    print(f"> num_formulas = {cum_num_formulas}")
    print(f"> num_nb_nodes = {cum_num_nb_nodes}")

    for k in vals:
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


def run_inference_tabular(
    model: th.nn.Module,
    dataloader: th.utils.data.DataLoader,
    device: th.device,
    batch_cutoff: int = -1,
    evaluate: bool = False,
    output_set: set[str] | None = None,
    nb_iso: bool = False,
    min_peak_int: float = 0.0,
    return_formula_str: bool = False,
    return_cc_hash: bool = False,
    rebase_indices: bool = True,
) -> pd.DataFrame:
    """
    Run inference and return all results as a single DataFrame.

    This is a convenience wrapper around run_inference_tabular_iter that accumulates
    all batch results into a single DataFrame.

    Args:
            model: Model instance
            dataloader: DataLoader with batches
            device: Device to run inference on
            batch_cutoff: Max batches to process (-1 = all)
            evaluate: Whether to run evaluation step
            output_set: Subset of outputs to include
            nb_iso: Whether model uses neighbor isotope prediction
            min_peak_int: Minimum peak intensity threshold
            return_formula_and_cc: Whether to return formula strings and CCs
            rebase_indices: Whether to rebase indices

    Returns:
            DataFrame with all inference results
    """
    # Get column names from config
    columns_name = get_tabular_columns(
        model.hparams, return_formula_str, return_cc_hash, output_subset=output_set
    )

    # Accumulate all batch results into a column-keyed dict
    preds_l: dict[str, list] = {}
    for batch_rows in run_inference_tabular_iter(
        model=model,
        dataloader=dataloader,
        device=device,
        batch_cutoff=batch_cutoff,
        evaluate=evaluate,
        output_set=output_set,
        nb_iso=nb_iso,
        min_peak_int=min_peak_int,
        return_formula_str=return_formula_str,
        return_cc_hash=return_cc_hash,
        rebase_indices=rebase_indices,
    ):
        assert len(batch_rows) == len(columns_name), (
            f"batch_rows has {len(batch_rows)} columns, expected {len(columns_name)}"
        )
        for k, v in zip(columns_name, batch_rows):
            preds_l.setdefault(k, []).extend(v)

    pred_df = pd.DataFrame(preds_l)
    return pred_df


def run_inference_hdf5(
    model: th.nn.Module,
    dataloader: th.utils.data.DataLoader,
    device: th.device,
    batch_cutoff: int = -1,
    evaluate: bool = False,
    output_set: set[str] | None = None,
    nb_iso: bool = False,
    min_peak_int: float = 0.0,
    return_formula_str: bool = False,
    return_cc_hash: bool = False,
    rebase_indices: bool = True,
    chunk_size=10000,
    save_path: str = "",
    drop_none_cols: bool = True,
    compression: str | None = "gzip",
    compression_opts: int | None = 4,
) -> pd.DataFrame:
    """
    Stream inference results to an HDF5 file and return the number of spectra written.

    Unlike run_inference_pkl, results are flushed every `chunk_size` spectra rather than
    held in memory, so this is the writer to use for large eval splits.

    Layout: scalar columns (spec_id, mol_id, scores, ...) hold one value per spectrum and
    are indexed directly by spectrum. Ragged columns are stored flat, concatenated across
    spectra, and sliced with a per-spectrum index pair:

        family    index pair                                 columns
        peak      start_points     / end_points              pred_mzs, pred_ints,
                                                             frag_formula_peak_{idxs,mzs}
        node      node_start_points    / node_end_points     pred_node_probs,
                                                             pred_node_node_idxs,
                                                             pred_node_ccs (2-D: hash pair)
        formula   formula_start_points / formula_end_points  pred_formula_probs,
                                                             pred_formula_formula_idxs,
                                                             pred_formula_strs (vlen str)
        joint     joint_start_points   / joint_end_points    pred_joint_{formula,node}_idxs,
                                                             pred_formula_node_probs,
                                                             pred_node_formula_probs

    So spectrum i's P(n) is pred_node_probs[node_start_points[i]:node_end_points[i]]. The
    four families have independent lengths — a spectrum's peak, node, formula and joint
    counts are unrelated — so each carries its own index pair. A family whose columns were
    not requested writes no datasets at all.

    Args:
            model: Model instance
            dataloader: DataLoader with batches
            device: Device to run inference on
            batch_cutoff: Max batches to process (-1 = all)
            evaluate: Whether to run evaluation step
            output_set: Subset of outputs to include
            nb_iso: Whether model uses neighbor isotope prediction
            min_peak_int: Minimum peak intensity threshold
            return_formula_and_cc: Whether to return formula strings and CCs
            rebase_indices: Whether to rebase indices
            drop_none_cols: Skip scalar columns that hold nothing but NaN placeholders
            compression: HDF5 filter for every dataset, or None to store raw. gzip is a
                    standard HDF5 filter, so the file stays readable outside h5py; lzf is
                    faster but h5py-only.
            compression_opts: Level passed to the filter (gzip 0-9); None for filters that
                    take no level.

    Returns:
            DataFrame with all inference results
    """
    # Get column names from config
    columns_name = get_tabular_columns(
        model.hparams, return_formula_str, return_cc_hash, output_subset=output_set
    )
    # Ragged per-spectrum column families, read off COLUMN_SPECS so they cannot drift from
    # the DataFrame schema. Each spectrum contributes a variable-length block to every
    # column in a family, so each family needs its own running offset plus per-spectrum
    # <prefix>start_points/<prefix>end_points. Writing these at the per-spectrum offset
    # (the way scalar columns are written) makes every chunk flush clobber the blocks the
    # previous flush wrote.
    # Families are independent: a spectrum's peak count, node count, formula count and
    # joint-grid size are all different numbers, so no family's offset can substitute for
    # another's.
    ragged_groups = tuple(
        (prefix, tuple(c.name for c in COLUMN_SPECS if c.family == family))
        for family, prefix in RAGGED_FAMILY_PREFIX.items()
    )
    ragged_columns = {k for _, members in ragged_groups for k in members}
    group_offsets = {prefix: 0 for prefix, _ in ragged_groups}

    def is_blocked(values) -> bool:
        """True when a column holds per-spectrum blocks rather than scalar placeholders.

        A column that was not requested comes through as one NaN per spectrum, which must
        keep going down the scalar path.
        """
        return bool(values) and isinstance(values[0], (th.Tensor, np.ndarray, list))

    def is_all_none(save_v) -> bool:
        """True when a scalar column carries no value at all, only NaN/None placeholders.

        An unrequested column arrives as one NaN per spectrum (see `is_blocked`), which is
        what `drop_none_cols` removes. Non-float columns are never placeholders.
        """
        if len(save_v) == 0:
            return False
        if save_v.dtype.kind == "f":
            return bool(np.isnan(save_v).all())
        if save_v.dtype == object:
            return all(v is None or (isinstance(v, float) and np.isnan(v)) for v in save_v)
        return False

    def concat_blocks(values, column_name):
        """Flatten per-spectrum blocks into one array h5py can store."""
        if isinstance(values[0], th.Tensor):
            return th.cat(values, 0).detach().cpu().numpy()
        flat = [item for block in values for item in block]
        if column_name == "pred_formula_adducts":
            # A missing precursor type is NaN in the tabular export. HDF5 needs one
            # string dtype across chunks, including when the first chunk is all missing.
            return np.array(
                ["NaN" if isinstance(v, float) and np.isnan(v) else v for v in flat],
                dtype=h5py.string_dtype(),
            )
        if flat and isinstance(flat[0], str):
            # Formula strings are ragged text, so they need a variable-length string dtype.
            return np.array(flat, dtype=h5py.string_dtype())
        return np.array(flat)

    # Accumulate all batch results
    preds_l = {}
    filter_stats: dict = {}

    # Columns dropped as all-NaN placeholders. The decision is made once, on the first
    # flush, and enforced on every later one: this writer appends to a single file, so a
    # column skipped early and written late would be short by every earlier spectrum.
    placeholder_cols: set[str] = set()

    start_point = 0
    with h5py.File(save_path, "w") as save_file:

        def create_kwargs(dtype, trailing):
            """Chunking/compression for one dataset.

            Compression is what makes this file small: the index columns are small integers
            stored in 8 bytes, so byte-shuffling exposes long runs of zeros and gzip takes
            them ~15-26x. The float probability columns are high-entropy and barely move
            (~1.2x) - they are already float32, and neither a dtype change nor a filter
            helps them. Measured on a real 96k-spectrum eval: an 80 MB index column becomes
            5.1 MB.

            The large chunk needs compression to be safe. HDF5 allocates a whole chunk once
            any of it is touched, so a 65536-row chunk holding a 378-row column costs 528 KB
            raw - but 7 KB once the unwritten tail is compressed.
            """
            if not compression:
                # Let h5py size the chunks; a big one would be pure waste uncompressed.
                return {"chunks": True}
            kwargs = {
                "chunks": (_HDF5_CHUNK_LEN,) + trailing,
                "compression": compression,
            }
            if compression_opts is not None:
                kwargs["compression_opts"] = compression_opts
            if dtype.kind in "biufc":
                # Byte-shuffle only applies to fixed-width numbers; it is a no-op on the
                # variable-length strings in pred_formula_strs.
                kwargs["shuffle"] = True
            return kwargs

        def resize_and_save(k, save_v, start_point):
            if k not in save_file:
                # Trailing dims are kept fixed and only axis 0 grows, so 2-D columns such as
                # pred_node_ccs (one hash pair per DAG node) store as (n_nodes, 2).
                save_file.create_dataset(
                    k,
                    shape=(0,) + save_v.shape[1:],
                    dtype=save_v.dtype,
                    maxshape=(None,) + save_v.shape[1:],
                    **create_kwargs(save_v.dtype, save_v.shape[1:]),
                )  # important to avoid bug in large number
            save_file[k].resize(start_point + len(save_v), axis=0)
            save_file[k][start_point : start_point + len(save_v)] = save_v

        save_file.create_dataset(
            "start_points",
            shape=(0,),
            dtype=np.int64,
            maxshape=(None,),
            **create_kwargs(np.dtype(np.int64), ()),
        )
        save_file.create_dataset(
            "end_points",
            shape=(0,),
            dtype=np.int64,
            maxshape=(None,),
            **create_kwargs(np.dtype(np.int64), ()),
        )

        batch_iterator = run_inference_tabular_iter(
            model=model,
            dataloader=dataloader,
            device=device,
            batch_cutoff=batch_cutoff,
            evaluate=evaluate,
            output_set=output_set,
            nb_iso=nb_iso,
            min_peak_int=min_peak_int,
            return_formula_str=return_formula_str,
            return_cc_hash=return_cc_hash,
            rebase_indices=rebase_indices,
            filter_stats=filter_stats if min_peak_int > 0 else None,
        )
        # The final sentinel flushes a partial chunk on actual iterator exhaustion,
        # including cutoffs and samplers whose reported length is only an estimate.
        for batch_rows in chain(batch_iterator, (None,)):
            if batch_rows is not None:
                assert len(batch_rows) == len(columns_name)
                for k, v in zip(columns_name, batch_rows):
                    if k not in preds_l:
                        preds_l[k] = []
                    preds_l[k].extend(v)
            if preds_l and (batch_rows is None or len(preds_l["pred_mzs"]) >= chunk_size):
                print("saving ...", start_point)
                # Scalar columns: one value per spectrum, so the per-spectrum offset is
                # right for them. Ragged families are written below at their own offsets.
                for k in preds_l:
                    if k in ragged_columns and is_blocked(preds_l[k]):
                        continue
                    # check whether is tensor
                    if len(preds_l[k]) > 0 and hasattr(preds_l[k][0], "detach"):
                        save_v = th.cat(preds_l[k], 0).detach().cpu().numpy()
                    else:
                        save_v = np.array(preds_l[k])
                    if drop_none_cols:
                        if start_point == 0:
                            if is_all_none(save_v):
                                placeholder_cols.add(k)
                                continue
                        elif k in placeholder_cols:
                            # A metric that is NaN for the first chunk's spectra and real
                            # later would silently lose those later values, so stop instead.
                            if not is_all_none(save_v):
                                raise RuntimeError(
                                    f"Column '{k}' was dropped as an all-NaN placeholder on "
                                    f"the first chunk but holds values at spectrum "
                                    f"{start_point}. Rerun with drop_none_cols=False."
                                )
                            continue
                    resize_and_save(k, save_v, start_point)

                for prefix, members in ragged_groups:
                    present = [k for k in members if k in preds_l and is_blocked(preds_l[k])]
                    if not present:
                        continue  # family not requested for this run
                    # Every column in a family is indexed by the same per-spectrum block,
                    # so their block lengths must agree spectrum by spectrum. Checking only
                    # the concatenated totals would let a per-spectrum misalignment through.
                    block_lengths = [len(block) for block in preds_l[present[0]]]
                    for k in present[1:]:
                        assert [len(b) for b in preds_l[k]] == block_lengths, (
                            f"{k} block lengths disagree with {present[0]} in family '{prefix}'"
                        )
                    group_start = group_offsets[prefix]
                    offset = group_start
                    starts, ends = [], []
                    for n in block_lengths:
                        starts.append(offset)
                        offset += n
                        ends.append(offset)
                    for k in present:
                        resize_and_save(k, concat_blocks(preds_l[k], k), group_start)
                    # start/end points are per spectrum, so they go at the spectrum offset.
                    resize_and_save(f"{prefix}start_points", np.array(starts), start_point)
                    resize_and_save(f"{prefix}end_points", np.array(ends), start_point)
                    group_offsets[prefix] = offset

                start_point += len(preds_l["pred_mzs"])
                del preds_l
                preds_l = {}
    if filter_stats:
        n_before = filter_stats.get("n_peaks_before", 0)
        n_after = filter_stats.get("n_peaks_after", 0)
        n_inst_filt = filter_stats.get("n_inst_filtered", 0)
        n_inst_total = filter_stats.get("n_inst_total", 0)
        pct_peaks = 100.0 * (n_before - n_after) / n_before if n_before > 0 else 0.0
        pct_inst = 100.0 * n_inst_filt / n_inst_total if n_inst_total > 0 else 0.0
        print(
            f">>> Peak filter stats (min_peak_int={min_peak_int}):\n"
            f"    Peaks before: {n_before:,}  after: {n_after:,}  removed: {n_before - n_after:,} ({pct_peaks:.1f}%)\n"
            f"    Instances filtered: {n_inst_filt:,} / {n_inst_total:,} ({pct_inst:.1f}%)"
        )
    return start_point


def run_inference_tabular_iter(
    model: th.nn.Module,
    dataloader: th.utils.data.DataLoader,
    device: th.device,
    batch_cutoff: int = -1,
    evaluate: bool = False,
    output_set: set[str] | None = None,
    nb_iso: bool = False,
    min_peak_int: float = 0.0,
    return_formula_str: bool = False,
    return_cc_hash: bool = False,
    rebase_indices: bool = True,
    filter_stats: dict | None = None,
):
    """
    Generator that yields batch results for streaming inference processing.

    Yields batch_rows for each batch processed. Caller is responsible for accumulating,
    saving, or processing results as needed.

    batch_rows is COLUMN-major: a list of ``len(columns_name)`` lists, each holding one
    value per spectrum in the batch (NOT a list of per-spectrum rows). Transpose with
    ``zip(*batch_rows)`` to get rows, or zip it against the column names to accumulate
    per column.

    Args:
            model: Model instance
            dataloader: DataLoader with batches
            device: Device to run inference on
            batch_cutoff: Max batches to process (-1 = all)
            evaluate: Whether to run evaluation step
            output_set: Subset of outputs to include
            nb_iso: Whether model uses neighbor isotope prediction
            min_peak_int: Minimum peak intensity threshold
            return_formula_and_cc: Whether to return formula strings and CCs
            rebase_indices: Whether to rebase indices
            output_set: Subset of output keys; probabilities outside it are dropped
            filter_stats: Optional mutable dict to accumulate peak-filtering statistics
                (n_peaks_before, n_peaks_after, n_inst_filtered, n_inst_total).

    Yields:
            One column-major batch_rows per batch

    Example:
            columns_name = [...]  # Define column names once
            for batch_rows in run_inference_tabular_iter(model, dataloader, device):
                    df = pd.DataFrame(list(zip(*batch_rows)), columns=columns_name)
                    df.to_pickle(f"batch_{batch_num}.pkl")
    """
    # A DataLoader over an IterableDataset has no length: batches are formed lazily, so
    # the count is unknown until the stream is exhausted.
    try:
        n_batches = len(dataloader)
    except TypeError:
        n_batches = None

    if n_batches is None:
        total_batches = batch_cutoff if batch_cutoff > 0 else None
    else:
        total_batches = n_batches if batch_cutoff < 0 else min(n_batches, batch_cutoff)
    n_desc = "an unknown number of" if total_batches is None else total_batches
    print(f"\n>>> Starting inference on {n_desc} batches...")
    if model.training:
        raise RuntimeError(f"> Warning: model {model} is in training mode!")

    BATCH_IDX_VARS, NODE_IDX_VARS, FORMULA_IDX_VARS, _ = get_idx_vars(
        output_set,
        model,
        nb_iso,
        return_formula_str=return_formula_str,
        return_cc_hash=return_cc_hash,
    )

    eval_score_types = get_score_types(model.hparams)
    model.eval()
    progress = ProgressLogger(total=total_batches, desc="Running inference")
    with th.inference_mode():
        for batch_idx, batch_input in enumerate(dataloader):
            # Early exit for batch cutoff
            if batch_cutoff >= 0 and batch_idx >= batch_cutoff:
                print(f"> reached batch cutoff of {batch_cutoff}, stopping inference")
                break
            progress.update()
            # Process batch and get data rows
            batch_rows = process_batch_to_tabular_row(
                batch_input=batch_input,
                model=model,
                device=device,
                evaluate=evaluate,
                eval_score_types=eval_score_types,
                BATCH_IDX_VARS=BATCH_IDX_VARS,
                NODE_IDX_VARS=NODE_IDX_VARS,
                FORMULA_IDX_VARS=FORMULA_IDX_VARS,
                min_peak_int=min_peak_int,
                rebase_indices=rebase_indices,
                return_formula_str=return_formula_str,
                return_cc_hash=return_cc_hash,
                output_set=output_set,
                filter_stats=filter_stats,
            )
            yield batch_rows
    progress.finish()


def get_split_fn(bs_idx, batch_size: int | None = None):
    """Build a splitter for tensors grouped by sorted batch indices.

    Args:
        bs_idx: Sorted batch index tensor aligned with the tensor to split.
        batch_size: Expected number of batch entries. When provided, empty groups
            are preserved with zero-length splits, which keeps per-spectrum outputs
            aligned after filtering removes all entries for some batch indices.

    Returns:
        A function that splits any tensor aligned with ``bs_idx`` into per-batch
        chunks.
    """
    # check the validity of bs idx
    if len(bs_idx) > 1:
        assert (bs_idx[1:] - bs_idx[:-1]).min() >= 0
    if batch_size is None:
        _, batch_counts = th.unique(bs_idx, return_counts=True)
    else:
        batch_counts = th.bincount(bs_idx.long(), minlength=batch_size)
    batch_counts = batch_counts.detach().cpu().tolist()

    def split_fn(inp_tensor):
        return th.split(inp_tensor, batch_counts)

    return split_fn


def process_batch_to_tabular_row(
    batch_input: dict,
    model: th.nn.Module,
    device: th.device,
    evaluate: bool,
    eval_score_types: list,
    BATCH_IDX_VARS: list,
    NODE_IDX_VARS: list,
    FORMULA_IDX_VARS: list,
    min_peak_int: float,
    rebase_indices: bool,
    return_formula_str: bool,
    return_cc_hash: bool,
    output_set: set[str] | None = None,
    filter_stats: dict | None = None,
) -> list:
    """Process a single batch and return list of data rows.

    Converts raw model outputs and batch inputs into tabular format, handling
    index rebasing, formula/node probability extraction, and optional supplementary
    fields (formula strings, node canonical SMILES).

    Args:
        batch_input: Input batch data dictionary
        model: Model instance for inference
        device: Device to run on
        evaluate: Whether to run evaluation step with ground truth
        eval_score_types: List of evaluation score types to compute
        BATCH_IDX_VARS: Names of batch index variables to include in output
        NODE_IDX_VARS: Names of node index variables to include in output
        FORMULA_IDX_VARS: Names of formula index variables to include in output
        min_peak_int: Minimum peak intensity threshold for filtering
        rebase_indices: Whether to rebase indices to 0-based per-molecule
        return_formula_str: Whether to include formula strings
        return_cc_hash: Whether to include node canonical SMILES

    Returns:
        List of data rows for this batch, with one row per spectrum-molecule pair
    """
    # Emitted column order; the same call `get_tabular_columns` makes for the header.
    selected_columns = selected_column_specs(return_formula_str, return_cc_hash)
    spec_ids = batch_input["spec_unique_id"]
    mol_ids = batch_input["mol_id"]
    # Extract string/graph data before moving to device (if they are not tensors)
    frag_formula_str = batch_input.get("frag_formula_str")
    frag_pyg = batch_input.get("frag_pyg")
    if "mol_pyg" in batch_input:
        batch_input["mol_pyg"].mol_ids = mol_ids

    batch_input = to_device(batch_input, device)
    _need_expl = bool(
        {"pred_formula_formula_idxs"} & set(FORMULA_IDX_VARS)
        or {"pred_node_node_idxs"} & set(NODE_IDX_VARS)
    )
    if isinstance(model, FraGNNetPL):
        # Single switch, passed either way; see the note in run_inference_pkl.
        batch_input["return_explainability_tensors"] = _need_expl
    with th.inference_mode():
        # `evaluate` only adds ground-truth scoring; the predicted intensities have the
        # same meaning either way.
        batch_result = model.inference_step(
            batch_input, metrics="test" if evaluate else None
        )

    # batch_input['batch_size'] is a tensor -> convert to Python int
    batch_size = int(batch_input["batch_size"].detach().cpu().item())
    pred_formula_batch_idxs = (
        batch_result["pred_formula_batch_idxs"]
        if "pred_formula_batch_idxs" in BATCH_IDX_VARS
        else None
    )
    pred_node_batch_idxs = (
        batch_result["pred_node_batch_idxs"] if "pred_node_batch_idxs" in BATCH_IDX_VARS else None
    )
    # split peaks into batchs
    pred_batch_idxs_batched = batch_result["pred_batch_idxs"]
    pred_mzs_batched = batch_result["pred_mzs"]
    # `inference_step` always returns L1-normalized intensities, so the base-peak rescale
    # is all that is owed.
    pred_ints_batched = batched_mf1000_normalize(
        batch_result["pred_ints"], pred_batch_idxs_batched
    )

    # prepare
    # Index arrays can be requested by an annotation column alone (formula strings, cc
    # hashes). When that happens the matching probability is still cheap to derive, but
    # the caller did not ask for it - dropping it here is what makes "annotation without
    # probabilities" actually smaller on disk instead of just differently named.
    def _want(token: str) -> bool:
        return output_set is None or token in output_set

    pred_oos_probs = (
        batch_result["oos_prob"].detach().cpu()
        if "oos_prob" in batch_result and _want("pred_oos_prob")
        else None
    )
    # values for eval scores
    # this will be np.nan if not evaluating
    eval_scores_d = {}
    if evaluate:
        for eval_score_type in eval_score_types:
            eval_scores_d[eval_score_type] = batch_result[eval_score_type].detach().cpu()

    # joint_batch_idxs maps each entry in the joint prediction arrays (like pred_formula_node_logprobs)
    # which are always None for NEIMS since it doesn't output formula/node predictions.
    frag_formula_cumsizes = batch_input.get("frag_formula_cumsizes")
    frag_num_nodes = batch_input.get("frag_num_nodes")
    joint_batch_idxs = (
        batch_result["pred_joint_batch_idxs"].detach().cpu()
        if "pred_joint_batch_idxs" in BATCH_IDX_VARS
        else None
    )

    # =============================
    # extract formula strings and PyG object if needed
    # =============================
    # Extract formula strings and PyG object if needed
    frag_formula_str = batch_input.get("frag_formula_str", None)
    frag_pyg = batch_input.get("frag_pyg", None)

    # ============================s
    # per-spectrum processing
    # ============================
    # Get cumulative counts for rebasing from batch_input
    # These are tensors of shape (batch_size + 1) containing offsets

    def _prob(token: str, logprob_key: str, axis_present) -> th.Tensor | None:
        """exp() a marginal, unless its axis is absent or the caller did not ask for it.

        An axis can be requested by an annotation column alone (`pred_formula_strs`,
        `pred_node_ccs`), which needs the index arrays but no probability. Dropping the
        probability here is what makes "annotation without probabilities" smaller on disk
        rather than merely differently named.
        """
        if axis_present is None or not _want(token):
            return None
        return th.exp(batch_result[logprob_key])

    arrays = batch_filter_by_peak_intensity(
        PredArrays(
            pred_batch_idxs=batch_result["pred_batch_idxs"],
            pred_ints=pred_ints_batched,
            pred_mzs=pred_mzs_batched,
            frag_formula_peak_mzs=batch_input.get("frag_formula_peak_mzs"),
            frag_formula_peak_idxs=batch_input.get("frag_formula_peak_idxs"),
            # The batch-idx arrays come in raw: the gated locals above decide which
            # *payload* arrays are requested, but the filter needs the raw axis to rebase
            # and the post-call rebind restores the same values the tuple unpack did.
            pred_formula_batch_idxs=batch_result.get("pred_formula_batch_idxs"),
            pred_formula_formula_idxs=(
                None
                if pred_formula_batch_idxs is None
                else batch_result.get("pred_formula_formula_idxs")
            ),
            pred_formula_probs=_prob(
                "pred_formula_probs", "pred_formula_logprobs", pred_formula_batch_idxs
            ),
            pred_joint_batch_idxs=batch_result.get("pred_joint_batch_idxs"),
            pred_joint_formula_idxs=(
                None if joint_batch_idxs is None else batch_result.get("pred_joint_formula_idxs")
            ),
            pred_joint_node_idxs=(
                None if joint_batch_idxs is None else batch_result.get("pred_joint_node_idxs")
            ),
            pred_formula_node_probs=_prob(
                "pred_formula_node_probs", "pred_formula_node_logprobs", joint_batch_idxs
            ),
            pred_node_formula_probs=_prob(
                "pred_node_formula_probs", "pred_node_formula_logprobs", joint_batch_idxs
            ),
            pred_node_batch_idxs=batch_result.get("pred_node_batch_idxs"),
            pred_node_probs=_prob(
                "pred_node_probs", "pred_node_logprobs", pred_node_batch_idxs
            ),
            pred_node_node_idxs=(
                None if pred_node_batch_idxs is None else batch_result.get("pred_node_node_idxs")
            ),
        ),
        min_peak_int,
        frag_formula_cumsizes,
        frag_num_nodes,
        filter_stats=filter_stats,
    )
    pred_batch_idx = arrays.pred_batch_idxs
    spec_ints = arrays.pred_ints
    spec_mzs = arrays.pred_mzs
    frag_formula_peak_mzs = arrays.frag_formula_peak_mzs
    frag_formula_peak_idxs = arrays.frag_formula_peak_idxs
    pred_formula_batch_idxs = arrays.pred_formula_batch_idxs
    pred_formula_formula_idxs = arrays.pred_formula_formula_idxs
    pred_formula_probs = arrays.pred_formula_probs
    joint_batch_idxs = arrays.pred_joint_batch_idxs
    pred_joint_formula_idxs = arrays.pred_joint_formula_idxs
    pred_joint_node_idxs = arrays.pred_joint_node_idxs
    pred_formula_node_probs = arrays.pred_formula_node_probs
    pred_node_formula_probs = arrays.pred_node_formula_probs
    pred_node_batch_idxs = arrays.pred_node_batch_idxs
    pred_node_probs = arrays.pred_node_probs
    pred_node_node_idxs = arrays.pred_node_node_idxs
    numpy_pred_oos_probs = (
        pred_oos_probs.detach().cpu().numpy() if pred_oos_probs is not None else None
    )

    # split based on cleaned results
    if joint_batch_idxs is not None:
        joint_split_fn = get_split_fn(joint_batch_idxs, batch_size=batch_size)
        joint_split_values = [
            None if _ is None else joint_split_fn(_)
            for _ in [
                pred_joint_formula_idxs,
                pred_formula_node_probs,
                pred_joint_node_idxs,
                pred_node_formula_probs,
            ]
        ]

    # formula prob batch idxs, maps each entry in pred_formula_logprobs to its batch idx
    if pred_formula_batch_idxs is not None:
        formula_split_fn = get_split_fn(pred_formula_batch_idxs, batch_size=batch_size)
        formula_split_values = [
            None if _ is None else formula_split_fn(_)
            for _ in [pred_formula_formula_idxs, pred_formula_probs]
        ]

    # node prob batch idxs, maps each entry in pred_node_logprobs to its batch idx
    if pred_node_batch_idxs is not None:
        node_split_fn = get_split_fn(pred_node_batch_idxs, batch_size=batch_size)
        node_split_values = [
            None if _ is None else node_split_fn(_) for _ in [pred_node_probs, pred_node_node_idxs]
        ]
    # specs
    batch_split_fn = get_split_fn(pred_batch_idx, batch_size=batch_size)
    pred_mzs_split = batch_split_fn(spec_mzs)
    pred_ints_split = batch_split_fn(spec_ints)
    frag_formula_peak_idxs_split = (
        None if frag_formula_peak_idxs is None else batch_split_fn(frag_formula_peak_idxs)
    )
    frag_formula_peak_mzs_split = (
        None if frag_formula_peak_mzs is None else batch_split_fn(frag_formula_peak_mzs)
    )
    assert len(pred_mzs_split) == len(pred_ints_split)

    # Per-formula-row precursor delta for the annotation columns. The dataset lays formula
    # rows out as NULL, CRF[1:F], CMF[1:F], so the delta a row carries depends on which half
    # it is in; `formula_row_prec_deltas` is the same split the model applies when it builds
    # ion formula strings. Indexed exactly like `frag_formula_str`, so a row's neutral
    # formula and its adduct are always read from the same place. FraGNNet only: no other
    # model here has a CRF/CMF formula axis.
    formula_row_deltas = None
    if (return_formula_str or return_cc_hash) and isinstance(model, FraGNNetPL):
        # Degrade to NaN like every sibling field rather than raising: `spec_prec_type_str`
        # exists only when spec_params.prec_type_str is on (False in config/template.yml and
        # in nearly every experiment config), and it is `apply_inference_config` that forces
        # it on. A caller reaching this function another way - a notebook, a future script, a
        # unit-level call - would otherwise get a KeyError mid-export instead of a column of
        # NaN. Warned rather than silent, because a missing adduct makes the neutral
        # `pred_formula_strs` ambiguous, which is the whole reason this column exists.
        # `frag_formula_cumsizes` needs no such guard: an annotation flag forces the
        # formula axis into the index vars, and batch_filter_by_peak_intensity above
        # rebases with it unconditionally, so a batch missing it never reaches this line.
        prec_type_strs = batch_input.get("spec_prec_type_str")
        if prec_type_strs is None:
            warnings.warn(
                "formula annotation was requested but the batch has no "
                "'spec_prec_type_str' (spec_params.prec_type_str is off); "
                "pred_formula_adducts will be NaN. Build the config with "
                "apply_inference_config, which turns that key on.",
                stacklevel=2,
            )
        else:
            formula_row_deltas = formula_row_prec_deltas(
                prec_type_strs,
                frag_formula_cumsizes.cpu().tolist(),
                include_cmf="cmf_h_formulae_idx" in model.model.frag_node_feats,
                max_h_transfer=model.model.num_hs,
            )

    batch_rows = None
    for b_idx in range(batch_size):
        # per-spectrum masks and basic fields
        # b_peak_mask = pred_batch_idxs == b_idx
        b_pred_oos_prob = numpy_pred_oos_probs[b_idx] if pred_oos_probs is not None else np.nan
        b_spec_id = spec_ids[b_idx]
        b_mol_id = mol_ids[b_idx]
        b_spec_mzs = pred_mzs_split[b_idx].detach().cpu()
        b_spec_ints = pred_ints_split[b_idx].detach().cpu()
        b_frag_formula_peak_idxs = (
            frag_formula_peak_idxs_split[b_idx]
            if frag_formula_peak_idxs_split is not None
            else None
        )
        b_frag_formula_peak_mzs = (
            frag_formula_peak_mzs_split[b_idx] if frag_formula_peak_mzs_split is not None else None
        )
        if joint_batch_idxs is not None:
            (
                b_pred_joint_formula_idxs,
                b_pred_formula_node_probs,
                b_pred_joint_node_idxs,
                b_pred_node_formula_probs,
            ) = [_[b_idx] if _ is not None else None for _ in joint_split_values]
        else:
            (
                b_pred_joint_formula_idxs,
                b_pred_formula_node_probs,
                b_pred_joint_node_idxs,
                b_pred_node_formula_probs,
            ) = [None for _ in range(4)]
        # Filter formula probability data (P(f)) for the current spectrum/molecule (b_idx)
        # and sort formulas by probability descending
        if pred_formula_batch_idxs is not None:
            b_pred_formula_formula_idxs, b_pred_formula_probs = [
                _[b_idx] if _ is not None else None for _ in formula_split_values
            ]
        else:
            b_pred_formula_formula_idxs, b_pred_formula_probs = [None for _ in range(2)]

        # node probabilities P(n) for the current spectrum/molecule (b_idx)
        if pred_node_batch_idxs is not None:
            b_pred_node_probs, b_pred_node_node_idxs = [
                _[b_idx] if _ is not None else None for _ in node_split_values
            ]
        else:
            b_pred_node_probs, b_pred_node_node_idxs = [None for _ in range(2)]

        # Store original indices before rebasing (for lookups)
        orig_formula_idxs_for_lookup = (
            b_pred_formula_formula_idxs.clone() if b_pred_formula_formula_idxs is not None else None
        )
        orig_node_idxs_for_lookup = (
            b_pred_node_node_idxs.clone() if b_pred_node_node_idxs is not None else None
        )

        if rebase_indices:  # possibly can be put into batch, but Idk what's the popurse here.
            # Rebase formula indices to 0-based contiguous
            if b_pred_formula_formula_idxs is not None and len(b_pred_formula_formula_idxs) > 0:
                unique_f_idxs, inverse = th.unique(
                    b_pred_formula_formula_idxs, sorted=True, return_inverse=True
                )

                # Apply rebasing to formula arrays
                b_pred_formula_formula_idxs = inverse
                if b_pred_joint_formula_idxs is not None:
                    b_pred_joint_formula_idxs = th.searchsorted(
                        unique_f_idxs, b_pred_joint_formula_idxs
                    )
                if b_frag_formula_peak_idxs is not None:
                    b_frag_formula_peak_idxs = th.searchsorted(
                        unique_f_idxs, b_frag_formula_peak_idxs
                    )

            # Rebase node indices to 0-based contiguous
            if b_pred_node_node_idxs is not None and len(b_pred_node_node_idxs) > 0:
                unique_n_idxs, inverse = th.unique(
                    b_pred_node_node_idxs, sorted=True, return_inverse=True
                )
                b_pred_node_node_idxs = inverse
                if b_pred_joint_node_idxs is not None:
                    b_pred_joint_node_idxs = th.searchsorted(unique_n_idxs, b_pred_joint_node_idxs)

        # assemble row, keyed by column name; ordered against COLUMN_SPECS below so the
        # row can never drift out of step with `get_tabular_columns`'s header.
        # An absent array becomes NaN rather than being dropped - see `selected_column_specs`.
        row_d = {
            "spec_id": b_spec_id.item(),
            "mol_id": b_mol_id,
            "pred_mzs": b_spec_mzs,
            "pred_ints": b_spec_ints,
            "pred_oos_prob": b_pred_oos_prob,
            "frag_formula_peak_idxs": b_frag_formula_peak_idxs,
            "frag_formula_peak_mzs": b_frag_formula_peak_mzs,
            "pred_formula_formula_idxs": b_pred_formula_formula_idxs,
            "pred_formula_probs": b_pred_formula_probs,
            "pred_joint_formula_idxs": b_pred_joint_formula_idxs,
            "pred_formula_node_probs": b_pred_formula_node_probs,
            "pred_joint_node_idxs": b_pred_joint_node_idxs,
            "pred_node_formula_probs": b_pred_node_formula_probs,
            "pred_node_probs": b_pred_node_probs,
            "pred_node_node_idxs": b_pred_node_node_idxs,
            # Add rebase flag: True if indices were rebased, False otherwise
            "index_rebased": bool(rebase_indices),
        }

        # Formula-axis annotation. `pred_formula_strs` and `pred_formula_adducts` address the
        # batch-wide formula arrays with the same index, so it is resolved once for both.
        global_f_idxs = None
        if (return_formula_str or return_cc_hash) and frag_formula_cumsizes is not None:
            f_offset = frag_formula_cumsizes[b_idx].item()
            if rebase_indices and orig_formula_idxs_for_lookup is not None:
                # Use original (pre-rebase) indices
                global_f_idxs = orig_formula_idxs_for_lookup + f_offset
            elif not rebase_indices and b_pred_formula_formula_idxs is not None:
                # Use current indices
                global_f_idxs = b_pred_formula_formula_idxs + f_offset
        np_f_idxs = None if global_f_idxs is None else global_f_idxs.cpu().numpy()

        if return_formula_str:
            row_d["pred_formula_strs"] = (
                frag_formula_str[np_f_idxs].tolist()
                if np_f_idxs is not None and frag_formula_str is not None
                else np.nan
            )

        if return_formula_str or return_cc_hash:
            # Signed rendering ("-H", "+Na"), so a peak reads as "C6H4O2 -H" against the
            # neutral formula string, and a cc bitmask as that fragment plus this adduct.
            if np_f_idxs is not None:
                row_d["pred_formula_adducts"] = (
                    [formula_delta_to_signed_str(d) for d in formula_row_deltas[np_f_idxs]]
                    if formula_row_deltas is not None
                    else [np.nan] * len(np_f_idxs)
                )
            else:
                row_d["pred_formula_adducts"] = np.nan

        if return_cc_hash:
            b_pred_node_ccs = np.nan
            if frag_pyg is not None:
                i_frag_pyg = frag_pyg[b_idx]
                node_feat_idxs = i_frag_pyg.node_feat_idxs.view(-1)
                all_node_ccs = get_node_feats(i_frag_pyg.x, node_feat_idxs, "cc")
                if rebase_indices and orig_node_idxs_for_lookup is not None:
                    # Use original (pre-rebase) indices
                    b_pred_node_ccs = all_node_ccs[orig_node_idxs_for_lookup.long()].detach().cpu()
                elif not rebase_indices and b_pred_node_node_idxs is not None:
                    # Use current indices
                    b_pred_node_ccs = all_node_ccs[b_pred_node_node_idxs.long()].detach().cpu()
            row_d["pred_node_ccs"] = b_pred_node_ccs

        # Move every exported tensor to the CPU here. Only some of the arrays above are
        # explicitly moved, so without this the saved predictions mix CPU and CUDA tensors
        # and the pickle cannot be loaded on a host without that CUDA device.
        data_row = []
        for c in selected_columns:
            value = row_d[c.name]
            if value is None:
                value = np.nan
            elif th.is_tensor(value):
                value = value.detach().cpu()
            data_row.append(value)
        # Append evaluation scores
        for eval_score_type in eval_score_types:
            if evaluate:
                data_row.append(eval_scores_d[eval_score_type][b_idx].item())
            else:
                data_row.append(np.nan)

        # Append the completed data row for this spectrum/molecule
        if batch_rows is None:
            batch_rows = [[_] for _ in data_row]
        else:
            for d, v in zip(batch_rows, data_row):
                d.append(v)
    return batch_rows


@dataclass
class PredArrays:
    """The batch-level prediction arrays the exporters filter, split and write.

    Grouped by the axis each array is indexed along, because that is what decides how it
    gets filtered and how it is written out:

    - **peak** (P): one entry per predicted spectrum peak.
    - **formula** (F): one entry per predicted formula.
    - **joint** (J): one entry per (formula, node) cell of the joint grid.
    - **node** (N): one entry per DAG node.

    Every field except the peak axis is optional: an array is None when the run did not
    request the output that needs it, and each consumer must tolerate that. The four
    lengths are unrelated, so no family's offsets or masks transfer to another.
    """

    # peak axis (P)
    pred_batch_idxs: th.Tensor
    pred_ints: th.Tensor
    pred_mzs: th.Tensor
    frag_formula_peak_mzs: th.Tensor | None = None
    frag_formula_peak_idxs: th.Tensor | None = None
    # formula axis (F)
    pred_formula_batch_idxs: th.Tensor | None = None
    pred_formula_formula_idxs: th.Tensor | None = None
    pred_formula_probs: th.Tensor | None = None
    # joint axis (J)
    pred_joint_batch_idxs: th.Tensor | None = None
    pred_joint_formula_idxs: th.Tensor | None = None
    pred_joint_node_idxs: th.Tensor | None = None
    pred_formula_node_probs: th.Tensor | None = None
    pred_node_formula_probs: th.Tensor | None = None
    # node axis (N)
    pred_node_batch_idxs: th.Tensor | None = None
    pred_node_probs: th.Tensor | None = None
    pred_node_node_idxs: th.Tensor | None = None


def batch_filter_by_peak_intensity(
    arrays: PredArrays,
    min_peak_int: float,
    frag_formula_cumsizes,
    frag_num_nodes,
    filter_stats: dict | None = None,
) -> PredArrays:
    """Drop low-intensity peaks, cascade to formulas/nodes, and rebase indices per molecule.

    Filtering cascades along the DAG: peaks below `min_peak_int` go first, then the
    formulas only those peaks explained, then the joint cells of those formulas, then the
    nodes left with no joint cell. Rebasing runs unconditionally afterwards, including
    when no filtering happened, because the caller's `pred_formula_strs` / `pred_node_ccs`
    lookups index a single molecule's table and so need per-molecule indices.

    Args:
        arrays: Batch-level prediction arrays, with batch-global indices.
        min_peak_int: Intensity threshold on the same 0-1000 scale the exporter emits;
            <= 0 skips filtering but still rebases.
        frag_formula_cumsizes: Per-spectrum formula offsets, shape (batch_size + 1,).
        frag_num_nodes: Per-spectrum node offsets, shape (batch_size + 1,).
        filter_stats: Optional mutable dict accumulating n_peaks_before/n_peaks_after/
            n_inst_filtered/n_inst_total across batches.

    Returns:
        A new `PredArrays` with the surviving entries and per-molecule indices. Fields
        that came in None stay None.
    """
    pred_batch_idxs = arrays.pred_batch_idxs
    pred_ints = arrays.pred_ints
    pred_mzs = arrays.pred_mzs
    frag_formula_peak_mzs = arrays.frag_formula_peak_mzs
    frag_formula_peak_idxs = arrays.frag_formula_peak_idxs
    pred_formula_batch_idxs = arrays.pred_formula_batch_idxs
    pred_formula_formula_idxs = arrays.pred_formula_formula_idxs
    pred_formula_probs = arrays.pred_formula_probs
    pred_joint_batch_idxs = arrays.pred_joint_batch_idxs
    pred_joint_formula_idxs = arrays.pred_joint_formula_idxs
    pred_joint_node_idxs = arrays.pred_joint_node_idxs
    pred_formula_node_probs = arrays.pred_formula_node_probs
    pred_node_formula_probs = arrays.pred_node_formula_probs
    pred_node_batch_idxs = arrays.pred_node_batch_idxs
    pred_node_probs = arrays.pred_node_probs
    pred_node_node_idxs = arrays.pred_node_node_idxs

    # No early return when there is nothing to filter: the index arrays still have to be
    # rebased from batch-global to per-molecule below, or the `pred_formula_strs` /
    # `pred_node_ccs` lookups index a single molecule's table with a batch-wide index
    # (out of bounds for every spectrum after the first).
    valid_pred_batch_idx = pred_batch_idxs
    valid_f_idxs = None
    # Stay None unless the peak filter actually builds them, so a missing P(f) index
    # array can never leave them dangling for the joint/node steps below.
    keep_mask_f = keep_mask_j = None

    # 1. Filter by peak intensity
    # This step identifies which formulas are associated with peaks that have intensity > min_peak_int.
    # It uses frag_formula_peak_idxs which maps peaks to local formula indices.
    if min_peak_int > 0 and frag_formula_peak_idxs is not None:
        pred_peak_in_each_batch = th.bincount(pred_batch_idxs)
        pred_peak_in_each_batch_cumsum = th.cumsum(pred_peak_in_each_batch, 0)
        pred_peak_in_each_batch_cumsum_pad0 = th.cat(
            [pred_peak_in_each_batch_cumsum.new_zeros(1), pred_peak_in_each_batch_cumsum], 0
        )
        n_peaks_before = len(pred_ints)
        valid_peak_mask = pred_ints > min_peak_int
        # Filter peaks themselves
        pred_ints = pred_ints[valid_peak_mask]
        pred_mzs = pred_mzs[valid_peak_mask]
        frag_formula_peak_idxs = frag_formula_peak_idxs[valid_peak_mask]
        valid_local_f_idxs = frag_formula_peak_idxs  # here the idx is local not global

        valid_pred_batch_idx = pred_batch_idxs[valid_peak_mask]
        if frag_formula_peak_mzs is not None:  # Only if frag_formula_peak_mzs is available
            frag_formula_peak_mzs = frag_formula_peak_mzs[valid_peak_mask]

        if filter_stats is not None:
            n_peaks_after = int(valid_peak_mask.sum().item())
            batch_size_local = (
                int(pred_batch_idxs.max().item()) + 1 if len(pred_batch_idxs) > 0 else 0
            )
            before_counts = th.bincount(pred_batch_idxs, minlength=batch_size_local)
            after_counts = th.bincount(valid_pred_batch_idx, minlength=batch_size_local)
            filter_stats["n_peaks_before"] = filter_stats.get("n_peaks_before", 0) + n_peaks_before
            filter_stats["n_peaks_after"] = filter_stats.get("n_peaks_after", 0) + n_peaks_after
            filter_stats["n_inst_filtered"] = filter_stats.get("n_inst_filtered", 0) + int(
                (before_counts != after_counts).sum().item()
            )
            filter_stats["n_inst_total"] = filter_stats.get("n_inst_total", 0) + batch_size_local

        # Map local formula indices (0..N_formulas_in_mol) to global formula indices (if available)
        # b_pred_formula_formula_idxs contains the global indices for the formulas in this molecule
        if pred_formula_formula_idxs is not None:
            # Ensure indices are valid (should be within bounds if data is consistent)
            # We use .long() to ensure they are treated as indices
            # Note: valid_local_f_idxs are indices into the formula array for this molecule
            #
            # The offset turning a per-molecule formula index into a row of the
            # batch-global formula arrays must count FORMULA ROWS per spectrum, not peaks.
            # The model trims one null formula per molecule, so peaks-per-spectrum and
            # formula-rows-per-spectrum drift apart by one for every spectrum; using the
            # peak offset makes each spectrum after the first read another molecule's
            # formulas. Spectrum 0 has offset 0 either way, which is why batch_size=1
            # hid this.
            formula_rows_in_each_batch = th.bincount(
                pred_formula_batch_idxs, minlength=len(pred_peak_in_each_batch)
            )
            formula_row_cumsum_pad0 = th.cat(
                [
                    formula_rows_in_each_batch.new_zeros(1),
                    th.cumsum(formula_rows_in_each_batch, 0),
                ],
                0,
            )
            valid_f_idxs = pred_formula_formula_idxs[
                valid_local_f_idxs.long() + formula_row_cumsum_pad0[valid_pred_batch_idx]
            ]
            valid_f_idxs = th.unique(valid_f_idxs)
            mask = th.zeros(
                pred_formula_formula_idxs.max().item() + 1,
                dtype=th.bool,
                device=valid_f_idxs.device,
            )
            mask[valid_f_idxs] = True
            keep_mask_f = mask[pred_formula_formula_idxs]
            # Guarded explicitly: `mask[None]` would not raise, it would add an axis and
            # hand back a (1, F) tensor that reads as "a mask exists" everywhere below.
            if pred_joint_formula_idxs is not None:
                keep_mask_j = mask[pred_joint_formula_idxs]
        else:
            # Fallback if formula indices are missing (unlikely to be useful but prevents crash)
            valid_f_idxs = th.unique(
                valid_local_f_idxs.long()
                + pred_peak_in_each_batch_cumsum_pad0[valid_pred_batch_idx]
            )
    # 3. Apply filtering to all formula-related tensors.
    # Skipped wholesale when no peak filtering ran (min_peak_int <= 0, or no
    # frag_formula_peak_idxs to map peaks to formulas with); the rebase below still runs.

    # Filter formula predictions (P(f))
    if keep_mask_f is not None and pred_formula_formula_idxs is not None:
        # keep_mask_f = th.isin(pred_formula_formula_idxs, valid_f_idxs) # speed up
        pred_formula_batch_idxs = pred_formula_batch_idxs[keep_mask_f]
        pred_formula_formula_idxs = pred_formula_formula_idxs[keep_mask_f]
        if pred_formula_probs is not None:
            pred_formula_probs = pred_formula_probs[keep_mask_f]

    # Filter joint predictions (P(f,n), P(n|f), P(f|n)) down to the surviving formulas.
    # There used to be a second, node-based term on this mask, but it was vacuous:
    # `pred_node_node_idxs` is `arange(num_nodes)`, so every joint entry passed it. (It
    # was also an IndexError waiting to happen, since the mask was sized by the joint's
    # largest node index but written at every node index.) Node pruning now runs the
    # other way round, below.
    if keep_mask_j is not None and pred_joint_formula_idxs is not None:
        pred_joint_batch_idxs = pred_joint_batch_idxs[keep_mask_j]
        pred_joint_formula_idxs = pred_joint_formula_idxs[keep_mask_j]
        if pred_formula_node_probs is not None:
            pred_formula_node_probs = pred_formula_node_probs[keep_mask_j]
        if pred_joint_node_idxs is not None:
            pred_joint_node_idxs = pred_joint_node_idxs[keep_mask_j]
        if pred_node_formula_probs is not None:
            pred_node_formula_probs = pred_node_formula_probs[keep_mask_j]

    # Filter node predictions (P(n)) to the nodes that can still explain a surviving
    # peak, i.e. the ones left in the filtered joint grid. Before this, P(f) was pruned
    # by intensity but P(n) was not, so `pred_node_probs` / `pred_node_node_idxs` - and
    # the `pred_node_ccs` annotation indexed by them - covered every node of the DAG
    # however aggressive `min_peak_int` was.
    #
    # Without the joint there is no node -> formula map available here, so the node
    # arrays pass through whole. `required_index_vars_for_outputs` requests the joint
    # whenever cc-hash annotation is asked for, which keeps that fallback off the
    # inference path.
    if (
        keep_mask_j is not None
        and pred_node_node_idxs is not None
        and pred_joint_node_idxs is not None
        and len(pred_node_node_idxs) > 0
    ):
        keep_mask_n = th.zeros(
            int(pred_node_node_idxs.max().item()) + 1,
            dtype=th.bool,
            device=pred_node_node_idxs.device,
        )
        keep_mask_n[pred_joint_node_idxs] = True
        keep_mask_n = keep_mask_n[pred_node_node_idxs]
        pred_node_batch_idxs = pred_node_batch_idxs[keep_mask_n]
        pred_node_node_idxs = pred_node_node_idxs[keep_mask_n]
        if pred_node_probs is not None:
            pred_node_probs = pred_node_probs[keep_mask_n]
    # rebase
    if pred_joint_batch_idxs is not None:
        if pred_joint_formula_idxs is not None:
            pred_joint_formula_idxs = (
                pred_joint_formula_idxs - frag_formula_cumsizes[pred_joint_batch_idxs]
            )
        if pred_joint_node_idxs is not None:
            pred_joint_node_idxs = pred_joint_node_idxs - frag_num_nodes[pred_joint_batch_idxs]
    if pred_node_batch_idxs is not None:
        if pred_node_node_idxs is not None:
            pred_node_node_idxs = pred_node_node_idxs - frag_num_nodes[pred_node_batch_idxs]
    if pred_formula_batch_idxs is not None:
        if pred_formula_formula_idxs is not None:
            pred_formula_formula_idxs = (
                pred_formula_formula_idxs - frag_formula_cumsizes[pred_formula_batch_idxs]
            )

    return PredArrays(
        pred_batch_idxs=valid_pred_batch_idx,
        pred_ints=pred_ints,
        pred_mzs=pred_mzs,
        frag_formula_peak_mzs=frag_formula_peak_mzs,
        frag_formula_peak_idxs=frag_formula_peak_idxs,
        pred_formula_batch_idxs=pred_formula_batch_idxs,
        pred_formula_formula_idxs=pred_formula_formula_idxs,
        pred_formula_probs=pred_formula_probs,
        pred_joint_batch_idxs=pred_joint_batch_idxs,
        pred_joint_formula_idxs=pred_joint_formula_idxs,
        pred_joint_node_idxs=pred_joint_node_idxs,
        pred_formula_node_probs=pred_formula_node_probs,
        pred_node_formula_probs=pred_node_formula_probs,
        pred_node_batch_idxs=pred_node_batch_idxs,
        pred_node_probs=pred_node_probs,
        pred_node_node_idxs=pred_node_node_idxs,
    )


def available_frag_mol_ids(frag_dp: str) -> set[str]:
    """Return the mol_ids (as strings) that have a DAG in ``frag_dp``.

    ``frag_dp`` is either a directory of per-molecule pickles (the folder backend,
    ``<name>/dags/``) or a packed ``dags.h5`` (the HDF5 backend). Both layouts skip
    molecules whose fragmentation failed or timed out, so the caller must drop those
    rows from mol_df / spec_df before building the dataset — the HDF5 backend does no
    missing-file check of its own and raises a KeyError on the first absent mol_id.

    For the HDF5 backend the ids come from the ``meta_info.json`` written next to the
    .h5 when available (its keys are exactly the successfully fragmented molecules),
    and otherwise from the packed index: it is dense in mol_id, so an entry with
    ``length == 0`` means "no DAG stored".

    Args:
        frag_dp: Directory of DAG files, or path to a packed ``dags.h5``.

    Returns:
        Set of mol_ids, as strings, for which a DAG can be loaded.
    """
    if not frag_dp.endswith(".h5"):
        return {f.split(".")[0] for f in os.listdir(frag_dp)}

    meta_info_fp = os.path.join(os.path.dirname(os.path.abspath(frag_dp)), "meta_info.json")
    if os.path.exists(meta_info_fp):
        with open(meta_info_fp) as f:
            return {str(mol_id) for mol_id in json.load(f)}

    with h5py.File(frag_dp, "r") as h5_file:
        index = h5_file[PACKED_DAG_INDEX_DATASET][:]
    return {str(mol_id) for mol_id in np.nonzero(index["length"])[0]}
