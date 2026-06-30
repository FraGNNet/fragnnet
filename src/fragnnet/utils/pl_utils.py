import copy
import logging

import numpy as np
from lightning.fabric.loggers.logger import rank_zero_experiment
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only


class ConsoleLogger(Logger):
    """Custom console logger that logs metrics to console via logging module.

    This logger is useful for displaying training metrics to stdout without
    writing to files or external services.
    """

    def __init__(self):
        super().__init__()

    @property
    @rank_zero_experiment
    def name(self) -> str:
        """Return logger name."""
        pass

    @property
    @rank_zero_experiment
    def experiment(self):
        """Return experiment object."""
        pass

    @property
    @rank_zero_experiment
    def version(self) -> str:
        """Return logger version."""
        pass

    @rank_zero_only
    def log_hyperparams(self, params: dict) -> None:
        """Log hyperparameters. Not implemented for console logger.

        Args:
            params: Dictionary of hyperparameters.
        """
        # No need to log hparams
        pass

    @rank_zero_only
    def log_metrics(self, metrics: dict, step: int) -> None:
        """Log metrics to console via logging module.

        Args:
            metrics: Dictionary of metric name to value.
            step: Current training step.
        """
        metrics = copy.deepcopy(metrics)

        epoch_num = "??"
        if "epoch" in metrics:
            epoch_num = metrics.pop("epoch")

        for k, v in metrics.items():
            logging.info(f"Epoch {epoch_num}, step {step}-- {k} : {v}")

    @rank_zero_only
    def finalize(self, status: str) -> None:
        """Finalize the logger.

        Args:
            status: Final status message.
        """
        pass


class PrintGradCallback(Callback):
    """Callback that logs model parameter and gradient norms during training.

    Useful for debugging training dynamics and detecting vanishing/exploding
    gradients.
    """

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx: int) -> None:
        """Log parameter norms at the start of each training batch.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: PyTorch Lightning module.
            batch: Current training batch.
            batch_idx: Index of current batch.
        """
        ps = []
        model_params = pl_module.parameters()
        for p in model_params:
            ps.append(p.norm().item())
        logging.info(ps[:10])
        logging.info("param_norm: %s", np.mean(ps))

    def on_after_backward(self, trainer, pl_module) -> None:
        """Log gradient norms after backward pass.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: PyTorch Lightning module.
        """
        p_grads = []
        model_params = pl_module.parameters()
        for p in model_params:
            if p.grad is not None:
                p_grads.append(p.grad.norm().item())
        logging.info(p_grads[:10])
        logging.info("grad_norm: %s", np.mean(p_grads))


def get_pl_hparams(ckpt: dict) -> dict:
    """Extract and return a deep copy of hyperparameters from a checkpoint.

    Args:
        ckpt: Checkpoint dictionary containing 'hyper_parameters' key.

    Returns:
        Deep copy of hyperparameters dictionary.
    """
    hyper_parameters = copy.deepcopy(ckpt["hyper_parameters"])
    return hyper_parameters
