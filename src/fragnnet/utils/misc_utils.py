"""Utility helpers shared across FraGNNet training and inference.

This module bundles lightweight helpers for seeding, type conversions,
progress/reporting, scatter reductions, memory checks, SLURM info, and
W&B checkpoint management. Functions are designed to be side-effect light
and torch/pandas friendly for use in data loading, model code, and scripts.
"""

import glob
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from collections.abc import (
    Callable,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from contextlib import contextmanager
from distutils.util import strtobool
from functools import wraps
from typing import Any, TypeVar

import joblib
import numpy as np
import pandas as pd
import torch as th
import torch_geometric as pyg
import tqdm

from fragnnet.utils.dgl_compat_utils import DGLGraph

T = TypeVar("T")
R = TypeVar("R")

TQDM_DISABLE = False
PPM = 1 / 1000000
EPS = 1e-7
LOG_HALF = float(np.log(0.5))
LOG_TWO = float(np.log(2.0))
LOG_ZERO_FP32 = float(th.finfo(th.float32).min)
LOG_ZERO_FP16 = float(th.finfo(th.float16).min)
LOG_ZERO_BP16 = float(th.finfo(th.bfloat16).min)
MAX_CROSS_ENTROPY = 1e19
TOLERANCE_MIN_MZ = 200.0


def LOG_ZERO(dtype: th.dtype) -> float:
    """Return a tiny log value for the given torch dtype."""

    if dtype == th.float32:
        return LOG_ZERO_FP32
    if dtype == th.float16:
        return LOG_ZERO_FP16
    if dtype == th.bfloat16:
        return LOG_ZERO_BP16
    raise ValueError(dtype)


def timeit(func: Callable[..., R]) -> Callable[..., R]:
    """Decorator to print execution time for the wrapped function."""

    # adapted from https://dev.to/kcdchennai/python-decorator-to-measure-execution-time-54hk
    @wraps(func)
    def timeit_wrapper(*args: Any, **kwargs: Any) -> R:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} Took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


def booltype(x: str) -> bool:
    """Convert a string to boolean using distutils.strtobool."""

    return bool(strtobool(x))


def none_or_nan(thing: Any) -> bool:
    """Return True when the value is effectively missing.

    Treats common null-like cases as missing: None, empty string, scalar NaN,
    empty containers, pandas objects that are all null, and NumPy arrays that
    are empty or all-NaN. Falls back to ``pd.isnull`` for other scalar-ish
    inputs.
    """

    # Check for None first
    if thing is None:
        return True

    # Check for empty string
    if isinstance(thing, str) and thing == "":
        return True

    # Check for NaN in floats
    if isinstance(thing, float) and np.isnan(thing):
        return True

    # Handle different data types safely
    try:
        # For scalar values, use pd.isnull which handles NaN properly
        if np.isscalar(thing):
            return bool(pd.isnull(thing))

        # For empty containers (lists, tuples, arrays)
        if hasattr(thing, "__len__") and len(thing) == 0:
            return True

        # For pandas Series/DataFrame
        if isinstance(thing, (pd.Series, pd.DataFrame)):
            return (
                thing.isnull().all().all()
                if isinstance(thing, pd.DataFrame)
                else bool(thing.isnull().all())
            )

        # For numpy arrays
        if isinstance(thing, np.ndarray):
            return thing.size == 0 or (thing.dtype.kind in "fc" and bool(np.isnan(thing).all()))

        # For other iterables, check if they're effectively empty
        if hasattr(thing, "__iter__") and not isinstance(thing, str):
            try:
                return len(thing) == 0
            except TypeError:
                # Some iterables don't support len()
                return False

        # Try pd.isnull as fallback for other types
        return bool(pd.isnull(thing))

    except (TypeError, ValueError, AttributeError):
        # If all checks fail, assume it's not null/nan
        return False


@contextmanager
def np_temp_seed(seed: int) -> Generator[None, None, None]:
    """Temporarily set the NumPy random seed within the context."""

    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


@contextmanager
def th_temp_seed(seed: int) -> Generator[None, None, None]:
    """Temporarily set the torch random seed within the context."""

    state = th.get_rng_state()
    th.manual_seed(seed)
    try:
        yield
    finally:
        th.set_rng_state(state)


@contextmanager
def th_temp_generator(seed: int) -> Generator[th.Generator, None, None]:
    """Yield an independent torch Generator seeded with ``seed`` without touching global RNG."""
    gen = th.Generator()
    gen.manual_seed(seed)
    try:
        yield gen
    finally:
        pass


def flatten_lol(lol: Iterable[Iterable[T]]) -> list[T]:
    """Flatten a list of iterables into a single list."""

    return [item for sublist in lol for item in sublist]


def wandb_symlink(run_dir: str, wandb_symlink_dp: str, job_id: str | int) -> None:
    """Create or refresh a symlink for a W&B run directory."""

    symlink_dst = os.path.join(wandb_symlink_dp, str(job_id))
    symlink_src = os.path.split(os.path.abspath(run_dir))[0]
    if os.path.islink(symlink_dst):
        os.unlink(symlink_dst)
    os.symlink(symlink_src, symlink_dst)


def list_str2float(str_list: Iterable[str]) -> list[float]:
    """Convert an iterable of strings to a list of floats."""

    return [float(str_item) for str_item in str_list]


# https://stackoverflow.com/a/58936697/6937913
@contextmanager
def tqdm_joblib(tqdm_object: tqdm.tqdm) -> Iterator[tqdm.tqdm]:
    """Patch joblib to report progress into the provided tqdm instance."""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


# some utils for function timeout
# adapted from https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call


class TimeoutException(Exception):
    """Raised when a function exceeds a specified timeout."""


# @contextmanager
# def time_limit(seconds):
#   if seconds is None:
#       yield
#   def signal_handler(signum, frame):
#       raise TimeoutException(f"Timed out! ({seconds} seconds)")
#   signal.signal(signal.SIGALRM, signal_handler)
#   signal.alarm(seconds)
#   try:
#       yield
#   finally:
#       signal.alarm(0)


def timeout_func(
    func: Callable[..., R],
    args: Sequence[Any] | None = None,
    kwargs: Mapping[str, Any] | None = None,
    timeout: int = 30,
    default: R | None = None,
) -> R:
    """Run ``func`` with a timeout, raising if exceeded."""

    class InterruptableThread(threading.Thread):
        def __init__(self) -> None:
            threading.Thread.__init__(self)
            self.result: R = default  # type: ignore[assignment]
            self.exc_info: tuple[type | None, BaseException | None, Any | None] = (
                None,
                None,
                None,
            )

        def run(self) -> None:
            try:
                self.result = func(*(args or ()), **(kwargs or {}))
            except Exception:
                self.exc_info = sys.exc_info()

        def suicide(self) -> None:
            raise TimeoutException(f"{func.__name__} timeout (taking more than {timeout} sec)")

    it = InterruptableThread()
    it.start()
    it.join(timeout)
    if it.exc_info[0] is not None:
        exc_type, exc_value, exc_tb = it.exc_info
        raise Exception(exc_type, exc_value, exc_tb)
    if it.is_alive():
        it.suicide()
        raise RuntimeError
    return it.result


def my_tqdm(*args: Any, **kwargs: Any) -> tqdm.tqdm:
    """Wrapper around tqdm respecting the module-level disable flag."""

    return tqdm.tqdm(*args, **kwargs, disable=TQDM_DISABLE)


def get_tensor_memory_usage(tensor: th.Tensor) -> int:
    """Calculate the byte size of a tensor."""

    return tensor.nelement() * tensor.element_size()


def get_tensor_dict_memory_usage(**tensor_dict: Any) -> int:
    """Compute total memory usage for tensor values in a dictionary."""

    total_memory = 0
    for k, v in tensor_dict.items():
        if isinstance(v, th.Tensor):
            total_memory += get_tensor_memory_usage(v)
    return total_memory


def get_pyg_memory_usage(pyg_graph: pyg.data.Data) -> int:
    """Return the approximate memory usage of a PyG graph."""

    return pyg.profile.get_data_size(pyg_graph)


def get_mol_graph_size(mol_data: Mapping[str, Any], mol_params: Mapping[str, Any]) -> int:
    """Compute memory usage for molecular graph data in bytes."""

    if mol_params["pyg"]:
        mol_pyg = mol_data["mol_pyg"]
        mol_graph_size = get_pyg_memory_usage(mol_pyg)
    else:
        mol_graph_size = 0
    return mol_graph_size


def scatter_masked_softmax(
    logits: th.Tensor,
    mask: th.Tensor,
    subset_idxs: th.Tensor,
    mask_logprob: float | None = None,
    log: bool = True,
) -> th.Tensor:
    """Apply log-softmax over grouped logits with masking support."""

    if mask_logprob is None:
        mask_logprob = LOG_ZERO(logits.dtype)
    # calculate appropriate mask value
    with th.no_grad():
        c = scatter_masked_logsumexp(logits, mask, subset_idxs)
        lm = th.gather(input=c, index=subset_idxs, dim=0)
        mask_value = mask_logprob + lm
    # apply mask
    masked_logits = mask * logits + (1 - mask) * mask_value
    # normalize
    masked_logits = scatter_logsoftmax(masked_logits, subset_idxs)
    if not log:
        # exponentiate
        return th.exp(masked_logits)
    return masked_logits


def scatter_masked_logsumexp(
    logits: th.Tensor, mask: th.Tensor, subset_idxs: th.Tensor, mask_value: float | None = None
) -> th.Tensor:
    """Compute logsumexp over groups after applying a mask."""

    if mask_value is None:
        mask_value = LOG_ZERO(logits.dtype)
    masked_logits = mask * logits + (1 - mask) * mask_value
    masked_logsumexp = scatter_logsumexp(masked_logits, subset_idxs)
    return masked_logsumexp


def scatter_logsumexp(
    logits: th.Tensor, subset_idxs: th.Tensor, eps: float = EPS, dim_size: int | None = None
) -> th.Tensor:
    """Numerically stable logsumexp grouped by ``subset_idxs``."""

    # numel() is metadata — no GPU sync. Return LOG_ZERO for every group when input is empty.
    if logits.numel() == 0:
        k = dim_size if dim_size is not None else 0
        return th.full((k,), LOG_ZERO(logits.dtype), device=logits.device, dtype=logits.dtype)

    if dim_size is None:
        k = th.max(subset_idxs) + 1
    else:
        assert dim_size >= th.max(subset_idxs) + 1
        k = dim_size

    sm = scatter_reduce(
        src=logits,
        index=subset_idxs,
        reduce="amax",
        dim_size=k,
        default=LOG_ZERO(logits.dtype),
    )

    lm = th.gather(input=sm, index=subset_idxs, dim=0)
    logits = logits - lm

    se = scatter_reduce(
        src=th.exp(logits), index=subset_idxs, reduce="sum", dim_size=k, default=0.0
    )
    return sm + th.log(se + eps)


def scatter_logmeanexp(
    logits: th.Tensor, subset_idxs: th.Tensor, eps: float = EPS, dim_size: int | None = None
) -> th.Tensor:
    """Compute log-mean-exp across groups defined by ``subset_idxs``."""

    den = scatter_reduce(
        src=th.ones_like(logits),
        index=subset_idxs,
        reduce="sum",
        dim_size=dim_size,
        default=0.0,
    )
    log_num = scatter_logsumexp(logits, subset_idxs, eps=eps, dim_size=dim_size)
    return log_num - safelog(den)


def scatter_logsoftmax(logits: th.Tensor, subset_idxs: th.Tensor) -> th.Tensor:
    """Compute a log-softmax over groups defined by ``subset_idxs``."""

    c = scatter_logsumexp(logits, subset_idxs)
    logits = logits - c[subset_idxs]
    return logits


def scatter_softmax(logits: th.Tensor, subset_idxs: th.Tensor) -> th.Tensor:
    """Softmax over grouped logits defined by ``subset_idxs``."""

    return th.exp(scatter_logsoftmax(logits, subset_idxs))


def scatter_l1normalize(vals: th.Tensor, subset_idxs: th.Tensor) -> th.Tensor:
    """Apply L1 normalization within each group of ``subset_idxs``."""

    c = scatter_reduce(src=vals, index=subset_idxs, reduce="sum", dim_size=th.max(subset_idxs) + 1)
    c = th.clamp(c, min=EPS)
    vals = vals / c[subset_idxs]
    return vals


def scatter_l2normalize(vals: th.Tensor, subset_idxs: th.Tensor) -> th.Tensor:
    """Apply L2 normalization within each group of ``subset_idxs``."""

    c = scatter_reduce(
        src=vals**2, index=subset_idxs, reduce="sum", dim_size=th.max(subset_idxs) + 1
    )
    c = th.clamp(th.sqrt(c), min=EPS)
    vals = vals / c[subset_idxs]
    return vals


def scatter_logl2normalize(logits: th.Tensor, subset_idxs: th.Tensor) -> th.Tensor:
    """Apply log-space L2 normalization grouped by ``subset_idxs``."""

    c = scatter_logsumexp(2 * logits, subset_idxs)
    logits = logits - 0.5 * c[subset_idxs]
    return logits


def scatter_var(
    src: th.Tensor,
    index: th.Tensor,
    dim_size: int | None = None,
    correction: int = 1,
    sqrt: bool = False,
) -> th.Tensor:
    """Compute grouped variance (optionally sqrt) across ``index`` groups."""

    if dim_size is None:
        dim_size = th.max(index) + 1
    else:
        assert dim_size >= th.max(index) + 1
    m = scatter_reduce(src=src, index=index, reduce="mean", dim_size=dim_size, include_self=False)
    v_num = scatter_reduce(src=(src - m[index]) ** 2, index=index, reduce="sum", dim_size=dim_size)
    v_den = scatter_reduce(src=th.ones_like(src), index=index, reduce="sum", dim_size=dim_size)
    v = v_num / th.clamp(v_den - correction, min=EPS)
    if sqrt:
        v = th.sqrt(v)
    return v


def scatter_argmax(
    src: th.Tensor,
    index: th.Tensor,
    other_index: th.Tensor,
    dim_size: int | None = None,
    return_max: bool = False,
) -> th.Tensor | tuple[th.Tensor, th.Tensor]:
    """Argmax grouped by ``index`` returning positions from ``other_index``."""

    if dim_size is None:
        dim_size = th.max(index) + 1
    else:
        assert dim_size >= th.max(index) + 1
    mx = scatter_reduce(src=src, index=index, reduce="amax", dim_size=dim_size, include_self=False)
    ma = src == mx[index]
    ma_idx = other_index * ma + (-1) * (~ma)
    amx = scatter_reduce(
        src=ma_idx,
        index=index,
        reduce="amax",
        dim_size=dim_size,
        include_self=True,
        default=-1,
    )
    if return_max:
        return amx, mx
    return amx


def scatter_argtopk(
    src: th.Tensor,
    index: th.Tensor,
    other_index: th.Tensor,
    k: int,
    dim_size: int | None = None,
    return_max: bool = False,
) -> th.Tensor | tuple[th.Tensor, th.Tensor]:
    """Return top-k indices per group defined by ``index``."""

    assert k > 0
    assert th.is_floating_point(src)
    src = src.detach().clone()
    counts = scatter_reduce(
        src=th.ones_like(src, dtype=th.long),
        index=index,
        reduce="sum",
        dim_size=th.max(index) + 1,
    )
    amxs, mxs = [], []
    for i in range(k):
        amx, mx = scatter_argmax(src, index, other_index, dim_size=dim_size, return_max=True)
        mask = (th.arange(amx.shape[0], device=index.device)[index] == index) & (
            amx[index] == other_index
        )
        src[mask] = -float("inf")
        amx[counts <= i] = -1
        amxs.append(amx)
        mxs.append(mx)
    amxs = th.stack(amxs, dim=1)
    mxs = th.stack(mxs, dim=1)
    if return_max:
        return amxs, mxs
    return amxs


def scatter_reduce(
    src: th.Tensor,
    index: th.Tensor,
    reduce: str,
    dim: int = 0,
    dim_size: int | None = None,
    default: float = 0.0,
    include_self: bool = True,
) -> th.Tensor:
    """Thin wrapper over torch.scatter_reduce_ with validation."""

    if reduce == "mean" and include_self:
        print("scatter_reduce: mean reduce with include_self=True is not recommended")
    index = index.to(th.int64)
    if index.numel() == 0:
        if dim_size is None:
            dim_size = 0
        result_shape = src.shape[:dim] + (dim_size,) + src.shape[dim + 1 :]
        return th.full(result_shape, default, dtype=src.dtype, device=src.device)
    max_index = int(th.max(index))
    if max_index < 0:
        raise ValueError("scatter_reduce: index contains negative entries")
    if dim_size is None:
        dim_size = max_index + 1
    elif dim_size <= max_index:
        raise ValueError(
            f"scatter_reduce: dim_size {dim_size} is too small for max index {max_index}"
        )
    result_shape = src.shape[:dim] + (dim_size,) + src.shape[dim + 1 :]
    results = th.full(result_shape, default, dtype=src.dtype, device=src.device)
    results.scatter_reduce_(dim=dim, index=index, src=src, reduce=reduce, include_self=include_self)
    return results


def safelog(x: th.Tensor, eps: float = EPS) -> th.Tensor:
    """Safe log with epsilon clamp."""

    return th.log(th.clamp(x, min=eps))


def batchwise_max(xs: th.Tensor, batch_idxs: th.Tensor) -> th.Tensor:
    """Compute max per batch index (debug helper)."""

    batch_size = th.max(batch_idxs) + 1
    maxs = th.zeros([batch_size], device=xs.device, dtype=xs.dtype)
    for b in range(batch_size):
        maxs[b] = th.max(xs[batch_idxs == b])
    return maxs


def batchwise_lse(xs: th.Tensor, batch_idxs: th.Tensor) -> th.Tensor:
    """Compute logsumexp per batch index (debug helper)."""

    batch_size = th.max(batch_idxs) + 1
    lses = th.zeros([batch_size], device=xs.device, dtype=xs.dtype)
    for b in range(batch_size):
        lses[b] = th.logsumexp(xs[batch_idxs == b], 0)
    return lses


def dedup_peaks(
    mzs: th.Tensor, logprobs: th.Tensor, batch_idxs: th.Tensor
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Deduplicate peaks by ``mzs`` and aggregate log probabilities."""

    b_mzs = th.stack([batch_idxs.type(mzs.dtype), mzs], dim=1)
    dd_b_mzs, dd_logprobs, dd_batch_idxs = dedup(b_mzs, *[("lse", logprobs), ("amax", batch_idxs)])
    dd_b_mzs = dd_b_mzs[:, 1]
    return dd_b_mzs, dd_logprobs, dd_batch_idxs


def dedup(
    keys: th.Tensor, *agg_vals_tups: tuple[str, th.Tensor], dim: int = 0
) -> tuple[th.Tensor, ...]:
    """Deduplicate rows in ``keys`` and aggregate associated values."""

    un_keys, inv_keys = th.unique(keys, dim=dim, return_inverse=True)
    res: list[th.Tensor] = [un_keys]
    for agg_vals_tup in agg_vals_tups:
        agg, vals = agg_vals_tup
        assert agg in ["lse", "sum", "min", "mean", "amax"]
        assert vals.shape[0] == keys.shape[0]
        if agg == "lse":
            un_vals = scatter_logsumexp(vals, inv_keys, dim_size=un_keys.shape[0])
        else:
            un_vals = scatter_reduce(
                vals,
                inv_keys,
                reduce=agg,
                dim_size=un_keys.shape[0],
                include_self=False,
            )
        res.append(un_vals)
    return tuple(res)


def to_cpu(
    data_d: dict[str, Any], non_blocking: bool = True, detach: bool = False
) -> dict[str, Any]:
    """Move tensor-like entries in a dict to CPU."""

    for k in data_d:
        if isinstance(data_d[k], th.Tensor):
            data = data_d[k]
            if detach:
                data = data.detach()
            data = data.to("cpu", non_blocking=non_blocking)
            data_d[k] = data
    return data_d


def to_device(
    data_d: dict[str, Any], device: str | th.device, non_blocking: bool = True
) -> dict[str, Any]:
    """Move tensor or graph entries in a dict to the specified device."""

    for k in data_d:
        v = data_d[k]
        # Support optional DGL: use DGLGraph placeholder for isinstance checks
        if isinstance(v, th.Tensor) or isinstance(v, DGLGraph) or isinstance(v, pyg.data.Data):
            v = v.to(device, non_blocking=non_blocking)
            data_d[k] = v
    return data_d


def deep_update(
    mapping: MutableMapping[str, Any], *updating_mappings: Mapping[str, Any]
) -> MutableMapping[str, Any]:
    """Recursively merge dictionaries, giving precedence to later mappings."""

    updated_mapping: MutableMapping[str, Any] = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if (
                k in updated_mapping
                and isinstance(updated_mapping[k], dict)
                and isinstance(v, dict)
            ):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping


def print_shapes(input_dict: Mapping[str, Any]) -> None:
    """Print shapes/types for entries in ``input_dict`` (debug helper)."""

    for k, v in input_dict.items():
        if isinstance(v, th.Tensor) or isinstance(v, np.ndarray):
            print(k, "/", tuple(v.shape), "/", type(v))
        elif isinstance(v, list) or isinstance(v, tuple):
            print(k, "/", len(v), "/", type(v))
        elif isinstance(v, pyg.data.Data):
            print(k, "/", (v.num_nodes, v.num_edges), "/", type(v))
        elif isinstance(v, DGLGraph):
            print(k, "/", (v.number_of_nodes(), v.number_of_edges()), "/", type(v))
        else:
            print(k, "/", None, "/", type(v))


def th_setdiff1d(t1: th.Tensor, t2: th.Tensor) -> th.Tensor:
    """Set difference of two 1D torch tensors."""

    t1 = th.unique(t1)
    t2 = th.unique(t2)
    return t1[(t1[:, None] != t2).all(dim=1)]


def get_package_version(package: Any) -> tuple[int, int, int]:
    """Return a (major, minor, patch) tuple for a package."""

    version = package.__version__.split("+")[0]
    major, minor, patch = version.split(".")
    return (int(major), int(minor), int(patch))


def check_pyg_compile() -> bool:
    """Verify PyTorch Geometric is compiled against torch >=2.1."""

    th_major_version, th_minor_version = get_package_version(th)[:2]
    pyg_major_version, pyg_minor_version = get_package_version(pyg)[:2]
    assert th_major_version >= 2, th_major_version
    assert pyg_major_version >= 2, pyg_major_version
    return th_minor_version >= 1 and pyg_minor_version >= 4


def check_pyg_full_compile() -> bool:
    """Check for full PyG compatibility (torch>=2.x, pyg>=2.5)."""

    th_major_version, th_minor_version = get_package_version(th)[:2]
    pyg_major_version, pyg_minor_version = get_package_version(pyg)[:2]
    assert th_major_version >= 2, th_major_version
    return pyg_major_version >= 2 and pyg_minor_version >= 5


# wandb stuff
def check_import_wandb() -> bool:
    """Check for wandb installation and warn if missing."""

    try:
        import wandb
    except ImportError:
        print("wandb is not installed. Please install it using 'pip install wandb' if needed.")
    return False


def get_best_ckpt_from_wandb(
    saved_dp: str,
    run_id: str,
    entity: str = "frag-gnn",
    project: str = "frag-gnn",
    use_cached: bool = False,
) -> str:
    """Fetch the best (latest) checkpoint from a W&B run and cache it."""

    cached_ckpts = glob.glob(os.path.join(saved_dp, f"*_{run_id}.ckpt"))
    if use_cached and len(cached_ckpts) == 1:
        print(f"> found cached ckpt {cached_ckpts[0]} for {run_id}")
        return cached_ckpts[0]

    import wandb

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    print(f"> Processing model files for run {run.id} {run.name}")
    model_tag = f"{run.name}_{run_id}"
    os.makedirs(saved_dp, exist_ok=True)
    ckpt_file, ckpt_epoch_num = None, None
    for file in run.files():
        if file.name.startswith("ckpt/model-epoch="):
            epoch_num = int(file.name.removeprefix("ckpt/model-epoch=").removesuffix(".ckpt"))
            if ckpt_file is None or epoch_num > ckpt_epoch_num:
                ckpt_epoch_num = epoch_num
                ckpt_file = file.name

    if ckpt_file is None or ckpt_epoch_num is None:
        print("> Skip, ckpt_file is None or ckpt_epoch_num is None")

    ckpt_fp = f"{saved_dp}/{model_tag}.ckpt"
    if os.path.isfile(ckpt_fp) and use_cached:
        return ckpt_fp

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"> Downloading ckpt {ckpt_file}")
        run.file(ckpt_file).download(root=tmp_dir, replace=False)
        print(f"> Save ckpt {ckpt_fp}")
        shutil.copy(f"{tmp_dir}/{ckpt_file}", ckpt_fp)
    return ckpt_fp


def get_wandb_runs_by_grp(
    group_name: str, entity: str = "frag-gnn", project: str = "frag-gnn"
) -> list[Any]:
    """Return all runs in a W&B group."""

    import wandb

    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", include_sweeps=False, filters={"group": group_name})
    return list(runs)


def get_wandb_runids_by_grp(
    group_name: str, entity: str = "frag-gnn", project: str = "frag-gnn"
) -> list[str]:
    """Return run IDs for all runs in a W&B group."""

    runs = get_wandb_runs_by_grp(group_name, entity=entity, project=project)
    run_ids = [run.id for run in runs]
    return run_ids


def delete_ckpt_from_wandb(
    run_id: str, entity: str = "frag-gnn", project: str = "frag-gnn"
) -> None:
    """Delete checkpoint files for a given W&B run."""

    import wandb

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    for file in run.files():
        if file.name.startswith("ckpt/model-epoch="):
            file.delete()


class NestedDefaultDict(defaultdict):
    """Nested defaultdict that auto-creates deeper levels."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(NestedDefaultDict, self).__init__(NestedDefaultDict, *args, **kwargs)

    def __repr__(self) -> str:
        return repr(dict(self))


def kl_div(x_logprobs: th.Tensor, y_logprobs: th.Tensor) -> th.Tensor:
    """Compute Kullback-Leibler divergence between two log-prob vectors."""

    return th.sum(th.exp(x_logprobs) * (x_logprobs - y_logprobs), dim=0)


def js_div(
    x_ids: th.Tensor, x_logprobs: th.Tensor, y_ids: th.Tensor, y_logprobs: th.Tensor
) -> th.Tensor:
    """Compute Jensen-Shannon divergence for sparse log-prob vectors."""

    assert th.unique(x_ids).shape[0] == x_ids.shape[0]
    assert th.unique(y_ids).shape[0] == y_ids.shape[0]

    z_ids, z_inv_idxs = th.unique(th.cat([x_ids, y_ids], dim=0), return_inverse=True)

    x_logprobs_p = th.full_like(
        z_ids, fill_value=LOG_ZERO(x_logprobs.dtype), dtype=x_logprobs.dtype
    )
    y_logprobs_p = th.full_like(
        z_ids, fill_value=LOG_ZERO(y_logprobs.dtype), dtype=y_logprobs.dtype
    )

    x_logprobs_p[z_inv_idxs[: x_ids.shape[0]]] = x_logprobs
    y_logprobs_p[z_inv_idxs[x_ids.shape[0] :]] = y_logprobs

    z_logprobs = th.logsumexp(th.stack([x_logprobs_p, y_logprobs_p], dim=0), dim=0) - LOG_TWO
    jsd = 0.5 * kl_div(x_logprobs_p, z_logprobs) + 0.5 * kl_div(y_logprobs_p, z_logprobs)
    jsd_n = jsd / LOG_TWO

    return jsd_n


def get_slurm_job_id() -> str | None:
    """Return SLURM job id from environment if available."""

    return os.getenv("SLURM_JOB_ID", default=None)


def get_slurm_allocated_cores(job_id: str) -> int | None:
    """Query SLURM for number of CPUs allocated to a job.

    Prefers the ``SLURM_CPUS_PER_TASK`` environment variable (set by SLURM
    for every task, including array tasks) and falls back to ``scontrol``
    when that variable is absent.  The ``scontrol`` path takes the first
    match so it works correctly for array jobs, which emit one line per task.
    """
    cpus_env = os.getenv("SLURM_CPUS_PER_TASK")
    if cpus_env is not None:
        try:
            return int(cpus_env)
        except ValueError:
            pass  # fall back to scontrol

    # fallback: use scontrol and take only the first line
    try:
        command = f"scontrol show job {job_id} | grep -oP 'NumCPUs=\\K\\d+'"
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,  # get str directly
        )
        lines = result.stdout.strip().splitlines()
        if not lines:
            return None
        return int(lines[0])  # first line only; array jobs may have multiple
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Error querying SLURM for job {job_id}: {e}")
        return None


def get_core_count() -> int:
    """Return CPU core count, preferring SLURM allocation when present."""

    slurm_id = get_slurm_job_id()
    if slurm_id is not None:
        num_core = get_slurm_allocated_cores(slurm_id)
    else:
        num_core = multiprocessing.cpu_count()
    return int(num_core) if num_core is not None else multiprocessing.cpu_count()


def validate_bin_geometry(mz_max: float, mz_bin_res: float) -> int:
    """Validate and return the bin count for a given m/z grid configuration.

    Enforces that mz_max and mz_bin_res define a consistent, integer bin grid:
    ``num_bins = mz_max / mz_bin_res`` must be an integer.

    Args:
        mz_max: Maximum m/z value covered by the grid.
        mz_bin_res: m/z bin resolution.

    Returns:
        Number of bins in the grid.

    Raises:
        ValueError: If ``mz_max`` or ``mz_bin_res`` is non-positive, or if
            ``mz_max / mz_bin_res`` is not an integer (within floating-point tolerance).

    Example:
        >>> validate_bin_geometry(1500.0, 0.01)  # Default: 150,000 bins
        150000
        >>> validate_bin_geometry(200.0, 0.5)    # Non-default: 400 bins
        400
        >>> validate_bin_geometry(200.0, 0.3)    # ValueError: 666.666... not integer
        Traceback: ValueError: Incompatible bin geometry...
    """
    import math

    if mz_max <= 0:
        raise ValueError(f"mz_max must be > 0, got {mz_max}")
    if mz_bin_res <= 0:
        raise ValueError(f"mz_bin_res must be > 0, got {mz_bin_res}")

    num_bins_float = mz_max / mz_bin_res
    num_bins = int(round(num_bins_float))
    if not math.isclose(num_bins_float, num_bins, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "Incompatible bin geometry: mz_max / mz_bin_res must be an integer. "
            f"Got mz_max={mz_max}, mz_bin_res={mz_bin_res}, ratio={num_bins_float}."
        )
    return num_bins


def progress_wrapper(
    iterable: Iterable[T],
    total: int | None = None,
    desc: str | None = None,
    disable_tqdm: bool = False,
) -> Iterator[T]:
    """Wrap an iterable with tqdm or fallback logging-based progress."""

    import logging

    if disable_tqdm:
        log_interval = max(1, total // 20) if total else 100
        for idx, item in enumerate(iterable):
            if idx % log_interval == 0:
                progress = f"{idx}/{total}" if total else f"{idx}"
                logging.info(f"{desc}: {progress}")
            yield item
        if total:
            logging.info(f"{desc}: {total}/{total} - Complete")
    else:
        from tqdm import tqdm

        for item in tqdm(iterable, total=total, desc=desc):
            yield item
