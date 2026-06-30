import gc
import multiprocessing
import os
import tempfile
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch as th
from tqdm import tqdm

from fragnnet.iceberg.pl_model import IcebergIntenPL
from fragnnet.pl_model import FraGNNetPL
from fragnnet.runner import load_config
from fragnnet.utils.frag_utils import get_node_feats
from fragnnet.utils.misc_utils import (
    LOG_ZERO,
    flatten_lol,
    get_best_ckpt_from_wandb,
    scatter_reduce,
    to_device,
)
from fragnnet.utils.nn_utils import decompile_jit_ckpt
from fragnnet.utils.pl_utils import get_pl_hparams
from fragnnet.utils.setup_utils import get_model_cls


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

    # Handle compiled checkpoints: only decompile when the config says the
    # model was compiled (avoids unnecessary decompilation overhead)
    if config_d.get("compile", False):
        print("> Checkpoint was compiled, decompiling...")
        ckpt = decompile_jit_ckpt(ckpt)
        config_d["compile"] = False  # prevent re-compilation during eval
    # Get model class and initialize
    model_type = config_d["model_type"]
    print(f">> Model type: {model_type}")
    model_cls = get_model_cls(model_type)

    model = model_cls(**config_d)

    # Load state dict
    try:
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print(">> Model loaded successfully (strict=True)")
    except RuntimeError as e:
        print("> Warning: Error loading with strict=True, retrying with strict=False")
        print(f"  Error: {e}")
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(">> Model loaded with strict=False")

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
# Output subset tokens and index requirements
# -----------------------------------------------------------------------------
# Valid tokens a user can request in `output_subset` for inference outputs.
OUTPUT_SUBSET_OPTIONS = [
    "pred_mzs",
    "pred_ints",
    "pred_oos_prob",
    "frag_formula_peak_mzs",
    "pred_formula_probs",
    "pred_formula_node_probs",
    "pred_node_formula_probs",
    "pred_node_probs",
]


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
) -> tuple[set[str], set[str], set[str], set[str]]:
    """
    Given desired `output_subset` tokens, determine which index variable arrays are
    required to be present in batch/model outputs.

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
        # P(n|f) on joint grid, requires joint idxs and joint batch
        "pred_formula_node_probs": {
            "batch": {"pred_joint_batch_idxs"},
            "formula": {"pred_joint_formula_idxs"},
            "node": {"pred_joint_node_idxs"},
        },
        # P(f|n) on joint grid
        "pred_node_formula_probs": {
            "batch": {"pred_joint_batch_idxs"},
            "formula": {"pred_joint_formula_idxs"},
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
) -> tuple[set[str], set[str], set[str], set[str]]:
    """
    Public API: Given `output_subset` tokens, return required index variable names
    grouped by category (batch, node, formula, nb_node). Useful for configuring
    dataset/model outputs to ensure all needed indices are present.
    """
    return required_index_vars_for_outputs(output_subset, model, nb_iso)


def get_idx_vars(
    output_subset: set[str] | None,
    model,
    nb_iso: bool = False,
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
            output_subset, model, nb_iso
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
            return_formula_and_cc: Whether output includes formula strings and node CCs
            output_subset: Optional subset of output keys to include

    Returns:
            List of column names for DataFrame
    """
    eval_score_types = get_score_types(config_d)
    columns_name = [
        "spec_id",
        "mol_id",
        "pred_mzs",
        "pred_ints",
        "pred_oos_prob",
        "frag_formula_peak_idxs",
        "frag_formula_peak_mzs",
        "pred_formula_formula_idxs",
        "pred_formula_probs",
        "pred_joint_formula_idxs",
        "pred_formula_node_probs",
        "pred_joint_node_idxs",
        "pred_node_formula_probs",
        "pred_node_probs",
        "pred_node_node_idxs",
        "index_rebased",
    ]
    if return_formula_str:
        columns_name.extend(["pred_formula_strs"])
    if return_cc_hash:
        columns_name.extend(["pred_node_ccs"])
    columns_name.extend(eval_score_types)

    if output_subset is not None:
        missing_cols = [c for c in output_subset if c not in columns_name]
        if len(missing_cols) > 0:
            print(
                f"Warning: {len(missing_cols)} columns in output_subset are not in available columns: {missing_cols}"
            )
        # columns_name = [c for c in columns_name if c in output_subset]

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
    with th.inference_mode():
        for batch_idx, batch_input in tqdm(
            enumerate(dataloader), total=total_batches, desc="Running inference"
        ):
            # Early exit for batch cutoff
            if batch_idx >= total_batches:
                break

            # Move batch to device
            batch_input = to_device(batch_input, device, non_blocking=False)

            # Enable explainability tensors only when formula/node idx vars are needed
            _need_expl = bool(
                {"pred_formula_formula_idxs"} & set(FORMULA_IDX_VARS)
                or {"pred_node_node_idxs"} & set(NODE_IDX_VARS)
            )
            if _need_expl and is_fragnnet_model:
                batch_input["return_explainability_tensors"] = True

            # Run inference (choose method based on mode)
            if evaluate:
                # Use ground truth evaluation method
                b_output = model.inference_with_ground_truth_step(batch_input, split="test")
            else:
                # Use prediction-only method
                b_output = model.inference_step(batch_input)

            b_output = to_device(b_output, "cpu", non_blocking=False)
            if output_set is not None:
                b_output = {k: b_output[k] for k in b_output if k in output_set}

            # ============================================================
            # EVALUATION MODE: Collect all outputs with index rebatching
            # ============================================================
            output_keys = b_output.keys()

            # Calculate batch counts for index rebatching
            if "pred_batch_idxs" in output_keys:
                b_num_datapoints = b_output["pred_batch_idxs"].max() + 1
            elif "true_batch_idxs" in output_keys:
                b_num_datapoints = b_output["true_batch_idxs"].max() + 1
            else:
                b_num_datapoints = 0

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
    sequential_name = set(
        ["pred_mzs", "pred_ints", "frag_formula_peak_idxs", "frag_formula_peak_mzs"]
    )  # can update this vec
    # Accumulate all batch results
    num_pred = 0
    num_batch = len(dataloader)
    preds_l = {}
    filter_stats: dict = {}

    spec_start_point = 0
    start_point = 0
    with h5py.File(save_path, "w") as save_file:

        def resize_and_save(k, save_v, start_point):
            if k not in save_file:
                save_file.create_dataset(
                    k,
                    shape=(0,),
                    dtype=save_v.dtype,
                    maxshape=(None,),
                )  # important to avoid bug in large number
            save_file[k].resize(start_point + len(save_v), axis=0)
            save_file[k][start_point : start_point + len(save_v)] = save_v

        save_file.create_dataset(
            "start_points",
            shape=(0,),
            dtype=np.int64,
            maxshape=(None,),
        )
        save_file.create_dataset(
            "end_points",
            shape=(0,),
            dtype=np.int64,
            maxshape=(None,),
        )

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
            filter_stats=filter_stats if min_peak_int > 0 else None,
        ):
            num_pred += 1
            assert len(batch_rows) == len(columns_name)
            for k, v in zip(columns_name, batch_rows):
                if k not in preds_l:
                    preds_l[k] = []
                preds_l[k].extend(v)
            # use pred_mzs length as the canonical flush trigger (always present)
            if len(preds_l["pred_mzs"]) >= chunk_size or num_pred == num_batch:
                print("saving ...", start_point)
                for k in preds_l:
                    if k not in sequential_name:
                        # check whether is tensor
                        if len(preds_l[k]) > 0 and hasattr(preds_l[k][0], "detach"):
                            save_v = th.cat(preds_l[k], 0).detach().cpu().numpy()
                        else:
                            save_v = np.array(preds_l[k])
                        resize_and_save(k, save_v, start_point)
                # save mz and intensity
                spec_start_points = []
                spec_end_points = []

                for mz in preds_l["pred_mzs"]:
                    spec_start_points.append(spec_start_point)
                    spec_start_point += len(mz)
                    spec_end_points.append(spec_start_point)

                save_length = None
                for k in sequential_name:
                    if k not in preds_l:
                        continue  # key absent when model doesn't output this field; skip safely
                    if not preds_l[k] or not isinstance(preds_l[k][0], th.Tensor):
                        continue  # all-nan / None (e.g. formula peaks absent for NEIMS); skip
                    save_v = th.cat(preds_l[k], 0).detach().cpu().numpy()
                    if save_length is None:
                        save_length = len(save_v)
                    else:
                        assert len(save_v) == save_length, k
                    resize_and_save(k, save_v, spec_start_point - save_length)
                resize_and_save("start_points", np.array(spec_start_points), start_point)
                resize_and_save("end_points", np.array(spec_end_points), start_point)
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

    Yields batch_rows (list of data rows) for each batch processed. Caller is responsible
    for accumulating, saving, or processing results as needed.

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
            filter_stats: Optional mutable dict to accumulate peak-filtering statistics
                (n_peaks_before, n_peaks_after, n_inst_filtered, n_inst_total).

    Yields:
            List of data rows for each batch

    Example:
            columns_name = [...]  # Define column names once
            for batch_rows in run_inference_tabular_iter(model, dataloader, device):
                    df = pd.DataFrame(batch_rows, columns=columns_name)
                    df.to_parquet(f"batch_{batch_num}.parquet")
    """
    print(
        f"\n>>> Starting inference on {min(len(dataloader), batch_cutoff if batch_cutoff > 0 else len(dataloader))} batches..."
    )
    if model.training:
        raise RuntimeError(f"> Warning: model {model} is in training mode!")

    BATCH_IDX_VARS, NODE_IDX_VARS, FORMULA_IDX_VARS, _ = get_idx_vars(
        output_set,
        model,
        nb_iso,
    )

    eval_score_types = get_score_types(model.hparams)
    total_batches = len(dataloader) if batch_cutoff < 0 else min(len(dataloader), batch_cutoff)
    model.eval()
    with th.inference_mode():
        for batch_idx, batch_input in tqdm(
            enumerate(dataloader), total=total_batches, desc="Running inference"
        ):
            # Early exit for batch cutoff
            if batch_idx >= batch_cutoff and batch_cutoff > 0:
                print(f"> reached batch cutoff of {batch_cutoff}, stopping inference")
                break
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
                filter_stats=filter_stats,
            )
            yield batch_rows


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
    if _need_expl and isinstance(model, FraGNNetPL):
        batch_input["return_explainability_tensors"] = True
    with th.inference_mode():
        if evaluate:
            batch_result = model.inference_with_ground_truth_step(batch_input, split="test")
        else:
            batch_result = model.predict_step(**batch_input)

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
    pred_ints_batched = model.ints_untransform_func(
        th.exp(batch_result["pred_logprobs"]), pred_batch_idxs_batched
    )

    # prepare
    pred_oos_probs = batch_result["oos_prob"].detach().cpu() if "oos_prob" in batch_result else None
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

    (
        pred_batch_idx,
        joint_batch_idxs,
        pred_formula_batch_idxs,
        pred_node_batch_idxs,
        spec_ints,
        spec_mzs,
        frag_formula_peak_mzs,
        frag_formula_peak_idxs,
        pred_formula_formula_idxs,
        pred_formula_probs,
        pred_joint_formula_idxs,
        pred_formula_node_probs,
        pred_joint_node_idxs,
        pred_node_formula_probs,
        pred_node_probs,
        pred_node_node_idxs,
    ) = batch_filter_by_peak_intensity(
        batch_result["pred_batch_idxs"],
        batch_result.get("pred_joint_batch_idxs"),
        batch_result.get("pred_formula_batch_idxs"),
        batch_result.get("pred_node_batch_idxs"),
        frag_formula_cumsizes,
        frag_num_nodes,
        min_peak_int,
        pred_ints_batched,
        pred_mzs_batched,
        batch_input.get("frag_formula_peak_mzs"),
        batch_input.get("frag_formula_peak_idxs"),
        None if pred_formula_batch_idxs is None else batch_result.get("pred_formula_formula_idxs"),
        None
        if pred_formula_batch_idxs is None
        else th.exp(batch_result.get("pred_formula_logprobs")),
        None if joint_batch_idxs is None else batch_result.get("pred_joint_formula_idxs"),
        None
        if joint_batch_idxs is None
        else th.exp(batch_result.get("pred_formula_node_logprobs")),
        None if joint_batch_idxs is None else batch_result.get("pred_joint_node_idxs"),
        None
        if joint_batch_idxs is None
        else th.exp(batch_result.get("pred_node_formula_logprobs")),
        None if pred_node_batch_idxs is None else th.exp(batch_result.get("pred_node_logprobs")),
        None if pred_node_batch_idxs is None else batch_result.get("pred_node_node_idxs"),
        verbose=False,
        filter_stats=filter_stats,
    )
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

        # assemble row and append
        # Add rebase flag: True if indices were rebased, False otherwise
        rebased = bool(rebase_indices)
        data_row = [
            b_spec_id.item(),
            b_mol_id,
            b_spec_mzs,
            b_spec_ints,
            b_pred_oos_prob,
            b_frag_formula_peak_idxs if b_frag_formula_peak_idxs is not None else np.nan,
            b_frag_formula_peak_mzs if b_frag_formula_peak_mzs is not None else np.nan,
            b_pred_formula_formula_idxs if b_pred_formula_formula_idxs is not None else np.nan,
            b_pred_formula_probs if b_pred_formula_probs is not None else np.nan,
            b_pred_joint_formula_idxs if b_pred_joint_formula_idxs is not None else np.nan,
            b_pred_formula_node_probs if b_pred_formula_node_probs is not None else np.nan,
            b_pred_joint_node_idxs if b_pred_joint_node_idxs is not None else np.nan,
            b_pred_node_formula_probs if b_pred_node_formula_probs is not None else np.nan,
            b_pred_node_probs if b_pred_node_probs is not None else np.nan,
            b_pred_node_node_idxs if b_pred_node_node_idxs is not None else np.nan,
            rebased,
        ]

        if return_formula_str:
            b_pred_formula_strs = np.nan
            if frag_formula_str is not None:
                f_offset = frag_formula_cumsizes[b_idx].item()
                if rebase_indices and orig_formula_idxs_for_lookup is not None:
                    # Use original (pre-rebase) indices
                    global_f_idxs = orig_formula_idxs_for_lookup + f_offset
                elif not rebase_indices and b_pred_formula_formula_idxs is not None:
                    # Use current indices
                    global_f_idxs = b_pred_formula_formula_idxs + f_offset
                else:
                    global_f_idxs = None

                if global_f_idxs is not None:
                    b_pred_formula_strs = frag_formula_str[global_f_idxs.cpu().numpy()].tolist()
            data_row.append(b_pred_formula_strs)

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
            data_row.append(b_pred_node_ccs)

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


def filter_by_peak_intensity(
    min_peak_int: float,
    pred_ints: th.Tensor,
    pred_mzs: th.Tensor,
    frag_formula_peak_mzs: th.Tensor | None,
    frag_formula_peak_idxs: th.Tensor | None,
    b_pred_formula_formula_idxs: th.Tensor | None,
    b_pred_formula_probs: th.Tensor | None,
    b_pred_joint_formula_idxs: th.Tensor | None = None,
    b_pred_formula_node_probs: th.Tensor | None = None,
    b_pred_joint_node_idxs: th.Tensor | None = None,
    b_pred_node_formula_probs: th.Tensor | None = None,
    b_pred_node_probs: th.Tensor | None = None,
    b_pred_node_node_idxs: th.Tensor | None = None,
    verbose: bool = False,
) -> tuple[
    th.Tensor,
    th.Tensor,
    th.Tensor | None,
    th.Tensor | None,
    th.Tensor | None,
    th.Tensor | None,
    th.Tensor | None,
    th.Tensor | None,
    th.Tensor | None,
    th.Tensor | None,
    th.Tensor | None,
    th.Tensor | None,
]:
    """Filter predictions based on peak intensity threshold.

    Removes low-intensity peaks and cascades the filtering to associated formulas and nodes.
    Preserves global index mappings during filtering.

    Args:
        min_peak_int: Minimum peak intensity threshold for filtering
        pred_ints: Predicted intensities for peaks
        pred_mzs: Predicted m/z values for peaks
        frag_formula_peak_mzs: M/z values per peak aligned with frag_formula_peak_idxs
        frag_formula_peak_idxs: Mapping from peaks to formula indices
        b_pred_formula_formula_idxs: Global formula indices for predicted formulas
        b_pred_formula_probs: Predicted formula probabilities P(f)
        b_pred_joint_formula_idxs: Joint formula indices for P(f,n), P(n|f), P(f|n)
        b_pred_formula_node_probs: Predicted P(n|f) probabilities
        b_pred_joint_node_idxs: Joint node indices for P(f,n), P(n|f), P(f|n)
        b_pred_node_formula_probs: Predicted P(f|n) probabilities
        b_pred_node_probs: Predicted node probabilities P(n)
        b_pred_node_node_idxs: Node indices for predicted nodes
        verbose: Whether to print filtering statistics

    Returns:
        Tuple of filtered tensors: (pred_ints, pred_mzs, frag_formula_peak_mzs,
        frag_formula_peak_idxs, b_pred_formula_formula_idxs, b_pred_formula_probs,
        b_pred_joint_formula_idxs, b_pred_formula_node_probs, b_pred_joint_node_idxs,
        b_pred_node_formula_probs, b_pred_node_probs, b_pred_node_node_idxs)
    """
    if min_peak_int <= 0:
        return (
            pred_ints,
            pred_mzs,
            frag_formula_peak_mzs,
            frag_formula_peak_idxs,
            b_pred_formula_formula_idxs,
            b_pred_formula_probs,
            b_pred_joint_formula_idxs,
            b_pred_formula_node_probs,
            b_pred_joint_node_idxs,
            b_pred_node_formula_probs,
            b_pred_node_probs,
            b_pred_node_node_idxs,
        )

    # Track initial sizes for stats
    if verbose:
        init_n_formulas = (
            b_pred_formula_formula_idxs.shape[0] if b_pred_formula_formula_idxs is not None else 0
        )
        init_n_nodes = b_pred_node_node_idxs.shape[0] if b_pred_node_node_idxs is not None else 0
        init_n_peaks = pred_ints.shape[0] if pred_ints is not None else 0
        final_n_peaks = init_n_peaks

    valid_f_idxs = None
    valid_n_idxs = None

    # 1. Filter by peak intensity
    # This step identifies which formulas are associated with peaks that have intensity > min_peak_int.
    # It uses frag_formula_peak_idxs which maps peaks to local formula indices.
    if min_peak_int > 0 and frag_formula_peak_idxs is not None:
        valid_peak_mask = pred_ints > min_peak_int
        final_n_peaks = valid_peak_mask.sum().item()
        valid_local_f_idxs = frag_formula_peak_idxs[valid_peak_mask]

        # Filter peaks themselves
        pred_ints = pred_ints[valid_peak_mask]
        pred_mzs = pred_mzs[valid_peak_mask]
        frag_formula_peak_idxs = frag_formula_peak_idxs[valid_peak_mask]
        if frag_formula_peak_mzs is not None:  # Only if frag_formula_peak_mzs is available
            frag_formula_peak_mzs = frag_formula_peak_mzs[valid_peak_mask]

        # Map local formula indices (0..N_formulas_in_mol) to global formula indices (if available)
        # b_pred_formula_formula_idxs contains the global indices for the formulas in this molecule
        if b_pred_formula_formula_idxs is not None:
            # Ensure indices are valid (should be within bounds if data is consistent)
            # We use .long() to ensure they are treated as indices
            # Note: valid_local_f_idxs are indices into the formula array for this molecule
            valid_f_idxs = b_pred_formula_formula_idxs[valid_local_f_idxs.long()]
            valid_f_idxs = th.unique(valid_f_idxs)
        else:
            # Fallback if formula indices are missing (unlikely to be useful but prevents crash)
            valid_f_idxs = th.unique(valid_local_f_idxs)

    # If no filtering was applied (valid_f_idxs is None), return original tensors
    if valid_f_idxs is None:
        return (
            pred_ints,
            pred_mzs,
            frag_formula_peak_mzs,
            frag_formula_peak_idxs,
            b_pred_formula_formula_idxs,
            b_pred_formula_probs,
            b_pred_joint_formula_idxs,
            b_pred_formula_node_probs,
            b_pred_joint_node_idxs,
            b_pred_node_formula_probs,
            b_pred_node_probs,
            b_pred_node_node_idxs,
        )

    # 3. Apply filtering to all formula-related tensors

    # Filter formula predictions (P(f))
    if b_pred_formula_formula_idxs is not None:
        keep_mask_f = th.isin(b_pred_formula_formula_idxs, valid_f_idxs)
        b_pred_formula_formula_idxs = b_pred_formula_formula_idxs[keep_mask_f]
        if b_pred_formula_probs is not None:
            b_pred_formula_probs = b_pred_formula_probs[keep_mask_f]

    # Filter node predictions (P(n)) first
    # We keep nodes that appear in the filtered joint distribution
    if b_pred_node_probs is not None and b_pred_node_node_idxs is not None:
        # Get valid node indices after filtering
        valid_n_idxs = b_pred_node_node_idxs
    else:
        valid_n_idxs = None

    # Filter joint predictions (P(f,n), P(n|f), P(f|n))
    # Filter by BOTH formula (must be in valid_f_idxs) AND node (must be in valid_n_idxs if available)
    if b_pred_joint_formula_idxs is not None:
        keep_mask_j = th.isin(b_pred_joint_formula_idxs, valid_f_idxs)

        # Also filter by node if we have valid node indices
        if valid_n_idxs is not None and b_pred_joint_node_idxs is not None:
            keep_mask_j = keep_mask_j & th.isin(b_pred_joint_node_idxs, valid_n_idxs)

        b_pred_joint_formula_idxs = b_pred_joint_formula_idxs[keep_mask_j]
        if b_pred_formula_node_probs is not None:
            b_pred_formula_node_probs = b_pred_formula_node_probs[keep_mask_j]
        if b_pred_joint_node_idxs is not None:
            b_pred_joint_node_idxs = b_pred_joint_node_idxs[keep_mask_j]
        if b_pred_node_formula_probs is not None:
            b_pred_node_formula_probs = b_pred_node_formula_probs[keep_mask_j]

    # Track final sizes and print stats
    if verbose:
        final_n_formulas = (
            b_pred_formula_formula_idxs.shape[0] if b_pred_formula_formula_idxs is not None else 0
        )
        final_n_nodes = b_pred_node_node_idxs.shape[0] if b_pred_node_node_idxs is not None else 0
        print(
            f"  > Filtered stats: Peaks {init_n_peaks}->{final_n_peaks}, Formulas {init_n_formulas}->{final_n_formulas}, Nodes {init_n_nodes}->{final_n_nodes}"
        )

    return (
        pred_ints,
        pred_mzs,
        frag_formula_peak_mzs,
        frag_formula_peak_idxs,
        b_pred_formula_formula_idxs,
        b_pred_formula_probs,
        b_pred_joint_formula_idxs,
        b_pred_formula_node_probs,
        b_pred_joint_node_idxs,
        b_pred_node_formula_probs,
        b_pred_node_probs,
        b_pred_node_node_idxs,
    )


def batch_filter_by_peak_intensity(
    pred_batch_idxs: th.Tensor,
    pred_joint_batch_idxs: th.Tensor,
    pred_formula_batch_idxs: th.Tensor,
    pred_node_batch_idxs: th.Tensor,
    frag_formula_cumsizes,
    frag_num_nodes,
    min_peak_int: float,
    pred_ints: th.Tensor,
    pred_mzs: th.Tensor,
    frag_formula_peak_mzs: th.Tensor | None,
    frag_formula_peak_idxs: th.Tensor | None,
    pred_formula_formula_idxs: th.Tensor | None,
    pred_formula_probs: th.Tensor | None,
    pred_joint_formula_idxs: th.Tensor | None = None,
    pred_formula_node_probs: th.Tensor | None = None,
    pred_joint_node_idxs: th.Tensor | None = None,
    pred_node_formula_probs: th.Tensor | None = None,
    pred_node_probs: th.Tensor | None = None,
    pred_node_node_idxs: th.Tensor | None = None,
    verbose: bool = False,
    filter_stats: dict | None = None,
) -> tuple[
    th.Tensor,
    th.Tensor,
    th.Tensor | None,
    th.Tensor | None,
    th.Tensor | None,
    th.Tensor | None,
    th.Tensor | None,
    th.Tensor | None,
    th.Tensor | None,
    th.Tensor | None,
    th.Tensor | None,
    th.Tensor | None,
]:
    """Filter predictions based on peak intensity threshold.

    Removes low-intensity peaks and cascades the filtering to associated formulas and nodes.
    Preserves global index mappings during filtering.

    Args:
        min_peak_int: Minimum peak intensity threshold for filtering
        pred_ints: Predicted intensities for peaks
        pred_mzs: Predicted m/z values for peaks
        frag_formula_peak_mzs: M/z values per peak aligned with frag_formula_peak_idxs
        frag_formula_peak_idxs: Mapping from peaks to formula indices
        b_pred_formula_formula_idxs: Global formula indices for predicted formulas
        b_pred_formula_probs: Predicted formula probabilities P(f)
        b_pred_joint_formula_idxs: Joint formula indices for P(f,n), P(n|f), P(f|n)
        b_pred_formula_node_probs: Predicted P(n|f) probabilities
        b_pred_joint_node_idxs: Joint node indices for P(f,n), P(n|f), P(f|n)
        b_pred_node_formula_probs: Predicted P(f|n) probabilities
        b_pred_node_probs: Predicted node probabilities P(n)
        b_pred_node_node_idxs: Node indices for predicted nodes
        verbose: Whether to print filtering statistics

    Returns:
        Tuple of filtered tensors: (pred_ints, pred_mzs, frag_formula_peak_mzs,
        frag_formula_peak_idxs, b_pred_formula_formula_idxs, b_pred_formula_probs,
        b_pred_joint_formula_idxs, b_pred_formula_node_probs, b_pred_joint_node_idxs,
        b_pred_node_formula_probs, b_pred_node_probs, b_pred_node_node_idxs)
    """
    if min_peak_int <= 0:
        return (
            pred_batch_idxs,
            pred_joint_batch_idxs,
            pred_formula_batch_idxs,
            pred_node_batch_idxs,
            pred_ints,
            pred_mzs,
            frag_formula_peak_mzs,
            frag_formula_peak_idxs,
            pred_formula_formula_idxs,
            pred_formula_probs,
            pred_joint_formula_idxs,
            pred_formula_node_probs,
            pred_joint_node_idxs,
            pred_node_formula_probs,
            pred_node_probs,
            pred_node_node_idxs,
        )

    valid_f_idxs = None
    valid_n_idxs = None

    # 1. Filter by peak intensity
    # This step identifies which formulas are associated with peaks that have intensity > min_peak_int.
    # It uses frag_formula_peak_idxs which maps peaks to local formula indices.
    pred_peak_in_each_batch = th.bincount(pred_batch_idxs)
    pred_peak_in_each_batch_cumsum = th.cumsum(pred_peak_in_each_batch, 0)
    pred_peak_in_each_batch_cumsum_pad0 = th.cat(
        [pred_peak_in_each_batch_cumsum.new_zeros(1), pred_peak_in_each_batch_cumsum], 0
    )
    if min_peak_int > 0 and frag_formula_peak_idxs is not None:
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
            valid_f_idxs = pred_formula_formula_idxs[
                valid_local_f_idxs.long()
                + pred_peak_in_each_batch_cumsum_pad0[valid_pred_batch_idx]
            ]
            valid_f_idxs = th.unique(valid_f_idxs)
            mask = th.zeros(
                pred_formula_formula_idxs.max().item() + 1,
                dtype=th.bool,
                device=valid_f_idxs.device,
            )
            mask[valid_f_idxs] = True
            keep_mask_f = mask[pred_formula_formula_idxs]
            keep_mask_j = mask[pred_joint_formula_idxs]
        else:
            # Fallback if formula indices are missing (unlikely to be useful but prevents crash)
            valid_f_idxs = th.unique(
                valid_local_f_idxs.long()
                + pred_peak_in_each_batch_cumsum_pad0[valid_pred_batch_idx]
            )
    # If no filtering was applied (valid_f_idxs is None), return original tensors
    if valid_f_idxs is None:
        return (
            pred_batch_idxs,
            pred_joint_batch_idxs,
            pred_formula_batch_idxs,
            pred_node_batch_idxs,
            pred_ints,
            pred_mzs,
            frag_formula_peak_mzs,
            frag_formula_peak_idxs,
            pred_formula_formula_idxs,
            pred_formula_probs,
            pred_joint_formula_idxs,
            pred_formula_node_probs,
            pred_joint_node_idxs,
            pred_node_formula_probs,
            pred_node_probs,
            pred_node_node_idxs,
        )

    # 3. Apply filtering to all formula-related tensors

    # Filter formula predictions (P(f))
    if pred_formula_formula_idxs is not None:
        # keep_mask_f = th.isin(pred_formula_formula_idxs, valid_f_idxs) # speed up
        pred_formula_batch_idxs = pred_formula_batch_idxs[keep_mask_f]
        pred_formula_formula_idxs = pred_formula_formula_idxs[keep_mask_f]
        if pred_formula_probs is not None:
            pred_formula_probs = pred_formula_probs[keep_mask_f]

    # Filter node predictions (P(n)) first
    # We keep nodes that appear in the filtered joint distribution

    if pred_node_probs is not None and pred_node_node_idxs is not None:
        # Get valid node indices after filtering
        valid_n_idxs = pred_node_node_idxs
        if pred_joint_node_idxs is not None:
            valid_n_mask = th.zeros(
                pred_joint_node_idxs.max().item() + 1,
                dtype=th.bool,
                device=pred_joint_node_idxs.device,
            )
            valid_n_mask[valid_n_idxs] = True
    else:
        valid_n_idxs = None

    # Filter joint predictions (P(f,n), P(n|f), P(f|n))
    # Filter by BOTH formula (must be in valid_f_idxs) AND node (must be in valid_n_idxs if available)
    if pred_joint_formula_idxs is not None:
        # keep_mask_j = th.isin(pred_joint_formula_idxs, valid_f_idxs)

        # Also filter by node if we have valid node indices
        if valid_n_idxs is not None and pred_joint_node_idxs is not None:
            keep_mask_j = (
                keep_mask_j & valid_n_mask[pred_joint_node_idxs]
            )  # th.isin(pred_joint_node_idxs, valid_n_idxs)
        pred_joint_batch_idxs = pred_joint_batch_idxs[keep_mask_j]
        pred_joint_formula_idxs = pred_joint_formula_idxs[keep_mask_j]
        if pred_formula_node_probs is not None:
            pred_formula_node_probs = pred_formula_node_probs[keep_mask_j]
        if pred_joint_node_idxs is not None:
            pred_joint_node_idxs = pred_joint_node_idxs[keep_mask_j]
        if pred_node_formula_probs is not None:
            pred_node_formula_probs = pred_node_formula_probs[keep_mask_j]
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

    return (
        valid_pred_batch_idx,
        pred_joint_batch_idxs,
        pred_formula_batch_idxs,
        pred_node_batch_idxs,
        pred_ints,
        pred_mzs,
        frag_formula_peak_mzs,
        frag_formula_peak_idxs,
        pred_formula_formula_idxs,
        pred_formula_probs,
        pred_joint_formula_idxs,
        pred_formula_node_probs,
        pred_joint_node_idxs,
        pred_node_formula_probs,
        pred_node_probs,
        pred_node_node_idxs,
    )
