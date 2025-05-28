try:
    from lightning.pytorch.utilities import rank_zero_only
    from lightning.pytorch.loggers.logger import rank_zero_experiment
    from lightning.pytorch.loggers import Logger
    from lightning.pytorch.callbacks import Callback
except ModuleNotFoundError:
    from pytorch_lightning.utilities import rank_zero_only
    from pytorch_lightning.loggers.logger import rank_zero_experiment
    from pytorch_lightning.loggers import Logger
    from pytorch_lightning.callbacks import Callback

import copy
import logging


class ConsoleLogger(Logger):
    """Custom console logger class"""

    def __init__(self):
        super().__init__()

    @property
    @rank_zero_experiment
    def name(self):
        pass

    @property
    @rank_zero_experiment
    def experiment(self):
        pass

    @property
    @rank_zero_experiment
    def version(self):
        pass

    @rank_zero_only
    def log_hyperparams(self, params):
        ## No need to log hparams
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):

        metrics = copy.deepcopy(metrics)

        epoch_num = "??"
        if "epoch" in metrics:
            epoch_num = metrics.pop("epoch")

        for k, v in metrics.items():
            logging.info(f"Epoch {epoch_num}, step {step}-- {k} : {v}")

    @rank_zero_only
    def finalize(self, status):
        pass


class PrintGradCallback(Callback):

	def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):

		ps = []
		model_params = pl_module.parameters()
		for p in model_params:
			ps.append(p.norm().item())
		logging.info(ps[:10])
		logging.info("param_norm",np.mean(ps))
	
	def on_after_backward(self, trainer, pl_module):

		p_grads = []
		model_params = pl_module.parameters()
		for p in model_params:
			if p.grad is not None:
				p_grads.append(p.grad.norm().item())
		logging.info(p_grads[:10])
		logging.info("grad_norm",np.mean(p_grads))

