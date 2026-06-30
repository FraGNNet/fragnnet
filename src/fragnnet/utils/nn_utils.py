import copy
import math
from typing import Any

import torch as th
import torch.nn as nn


def get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    """Create N deep copies of a PyTorch module.

    Args:
        module: PyTorch module to clone
        N: Number of clones to create

    Returns:
        ModuleList containing N deep copies of the input module
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_lr_scheduler(
    optimizer: th.optim.Optimizer,
    decay_rate: float,
    decay_steps: int = 5000,
    warmup_steps: int = 1000,
    schedule_type: str = "exponential",
    total_steps: int | None = None,
    min_lr_ratio: float = 0.0,
) -> th.optim.lr_scheduler.LambdaLR:
    """Build learning rate scheduler with warmup followed by decay.

    Supports two schedule types after the linear warmup phase:
    - ``"exponential"``: stepwise exponential decay (original behavior).
    - ``"cosine"``: cosine annealing from peak LR down to ``min_lr_ratio * peak``.

    Args:
        optimizer: PyTorch optimizer to schedule.
        decay_rate: Base for exponential decay; ignored when ``schedule_type="cosine"``.
        decay_steps: Steps between exponential decay applications; ignored for cosine.
        warmup_steps: Number of steps for linear warmup.
        schedule_type: One of ``"exponential"`` or ``"cosine"``.
        total_steps: Total training steps (warmup + decay); required for cosine.
        min_lr_ratio: Minimum LR as a fraction of peak LR; only used for cosine.

    Returns:
        LambdaLR scheduler object.

    Raises:
        ValueError: If ``schedule_type="cosine"`` and ``total_steps`` is not provided.
        ValueError: If ``schedule_type`` is not recognised.
    """
    if schedule_type == "exponential":

        def lr_lambda(step):
            if step >= warmup_steps:
                s = step - warmup_steps
                return decay_rate ** (s // decay_steps)
            else:
                return (step / warmup_steps) if warmup_steps > 0 else 1.0

    elif schedule_type == "cosine":
        if total_steps is None:
            raise ValueError("total_steps must be provided for cosine schedule")
        decay_steps_cosine = max(total_steps - warmup_steps, 1)

        def lr_lambda(step):
            if step < warmup_steps:
                return (step / warmup_steps) if warmup_steps > 0 else 1.0
            progress = (step - warmup_steps) / decay_steps_cosine
            progress = min(progress, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type!r}. Choose 'exponential' or 'cosine'.")

    return th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def nan_forward_hook(self: nn.Module, input: Any, output: Any) -> None:
    """Forward hook to detect NaN values in module outputs during training.

    Args:
        self: The module being hooked
        input: Input to the module (unused)
        output: Output from the module (checked for NaN values)

    Raises:
        RuntimeError: If NaN values are detected in the output
    """
    if isinstance(output, tuple):
        outputs = list(output)
    elif isinstance(output, dict):
        outputs = list(output.values())
    else:
        outputs = [output]
    for i, val in enumerate(outputs):
        nan_mask = th.isnan(val)
        if nan_mask.any():
            print(">> In", self.__class__.__name__)
            raise RuntimeError(
                f"Found NAN in output {i} at indices: ",
                nan_mask.nonzero(),
                "where:",
                val[nan_mask.nonzero()[:, 0].unique(sorted=True)],
            )


def nan_backward_hook(self: nn.Module, grad_input: Any, grad_output: Any) -> None:
    """Backward hook to detect NaN values in gradients during backpropagation.

    Args:
        self: The module being hooked
        grad_input: Gradients with respect to module inputs (checked for NaN)
        grad_output: Gradients with respect to module outputs (checked for NaN)

    Raises:
        RuntimeError: If NaN values are detected in gradients
    """
    for i, val in enumerate(grad_input):
        if val is None:
            continue
        nan_mask = th.isnan(val)
        if nan_mask.any():
            print(">> In", self.__class__.__name__)
            raise RuntimeError(
                f"Found NAN in grad_input {i} at indices: ",
                nan_mask.nonzero(),
                "where:",
                val[nan_mask.nonzero()[:, 0].unique(sorted=True)],
            )
    for i, val in enumerate(grad_output):
        if val is None:
            continue
        nan_mask = th.isnan(val)
        if nan_mask.any():
            print(">> In", self.__class__.__name__)
            raise RuntimeError(
                f"Found NAN in grad_output {i} at indices: ",
                nan_mask.nonzero(),
                "where:",
                val[nan_mask.nonzero()[:, 0].unique(sorted=True)],
            )


def decompile_jit_ckpt(ckpt: dict[str, Any]) -> dict[str, Any]:
    """Convert compiled PyTorch checkpoint to normal uncompiled version.

    Removes `_orig_mod` prefix from compiled model state dict keys and updates
    the compile flag. This is useful for loading compiled models in non-compiled
    environments.

    Args:
        ckpt: Checkpoint dictionary containing 'state_dict' and 'hyper_parameters'

    Returns:
        Patched checkpoint dictionary with normalized state dict keys
    """

    # inspired by
    # https://github.com/pytorch/pytorch/issues/101107#issuecomment-1801128683
    # print(f"> src: {path}")

    if ckpt["hyper_parameters"]["compile"] == False:
        return ckpt

    in_state_dict = ckpt["state_dict"]
    pairings = [(src_key, src_key.replace("._orig_mod", "")) for src_key in in_state_dict.keys()]
    # if all(src_key == dest_key for src_key, dest_key in pairings):
    #   return  # Do not write checkpoint if no need to repair!
    out_state_dict = {}
    for src_key, dest_key in pairings:
        if src_key != dest_key:
            print(f"{src_key}  ==>  {dest_key}")
        out_state_dict[dest_key] = in_state_dict[src_key]

    ckpt["state_dict"] = out_state_dict
    ckpt["hyper_parameters"]["compile"] = False
    return ckpt


def is_ckpt_compiled(ckpt: dict[str, Any]) -> bool:
    """Check if a checkpoint is from a compiled PyTorch model.

    Args:
        ckpt: Checkpoint dictionary to check

    Returns:
        True if checkpoint contains compiled model keys (with '_orig_mod'), False otherwise
    """
    state_dict = ckpt["state_dict"]
    return any("_orig_mod" in key for key in state_dict)


def get_optimizer_class(optimizer_name: str) -> type:
    """Get PyTorch optimizer class by name.

    Args:
        optimizer_name: Name of optimizer ('adam', 'adamw', or 'sgd')

    Returns:
        PyTorch optimizer class corresponding to the name

    Raises:
        ValueError: If optimizer name is not recognized
    """
    if optimizer_name == "adam":
        optimizer_cls = th.optim.Adam
    elif optimizer_name == "adamw":
        optimizer_cls = th.optim.AdamW
    elif optimizer_name == "sgd":
        optimizer_cls = th.optim.SGD
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")
    return optimizer_cls
