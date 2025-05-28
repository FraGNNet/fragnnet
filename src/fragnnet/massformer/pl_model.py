import torch as th
import torch._dynamo as th_dynamo

from fragnnet.pl_model import BinnedPL
from fragnnet.utils.nn_utils import build_lr_scheduler
from fragnnet.massformer.model import MassFormerModel
from fragnnet.massformer.nn_utils import flag_bounded, PolynomialDecayLR


class MassFormerPL(BinnedPL):

	def _setup_model(self):

		self.model = MassFormerModel(
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
			mf_fix_num_pt_layers=self.hparams.mf_fix_num_pt_layers,
			mf_reinit_num_pt_layers=self.hparams.mf_reinit_num_pt_layers,
			mf_reinit_layernorm=self.hparams.mf_reinit_layernorm,
			int_embedder=self.hparams.int_embedder,
			ce_insert_type=self.hparams.ce_insert_type,
			ce_insert_location=self.hparams.ce_insert_location,
			ce_insert_merge=self.hparams.ce_insert_merge,
			ce_insert_size=self.hparams.ce_insert_size,
			ce_max=self.hparams.ce_max,
			ce_mean=self.hparams.ce_mean,
			ce_std=self.hparams.ce_std,
			prec_insert_location=self.hparams.prec_insert_location,
			prec_insert_size=self.hparams.prec_insert_size,
			prec_types=self.hparams.spec_params["prec_types"],
			inst_insert_location=self.hparams.inst_insert_location,
			inst_insert_size=self.hparams.inst_insert_size,
			inst_types=self.hparams.spec_params["inst_types"],
			log_min=self.hparams.log_min,
		)

		assert self.hparams.automatic_optimization

		self._check_ce_params()

		# compile
		if self.hparams.compile:
			th_dynamo.reset()
			self.dynamo_prof = th_dynamo.utils.CompileProfiler()
			self.model = th.compile(self.model,backend=self.dynamo_prof,dynamic=True)

		self.mf_flag = self.hparams.mf_flag
		if self.mf_flag:
			raise NotImplementedError
			assert self.hparams.precision == 32, self.hparams.precision
			assert not self.hparams.automatic_optimization
		
	def configure_optimizers(self):
		
		if self.hparams.optimizer == "adam":
			optimizer_cls = th.optim.Adam
		elif self.hparams.optimizer == "adamw":
			optimizer_cls = th.optim.AdamW
		elif self.hparams.optimizer == "sgd":
			optimizer_cls = th.optim.SGD
		else:
			raise ValueError(f"Unknown optimizer {self.optimizer}")
		nopt_params, pt_params = self.model.get_split_params()
		optimizer = optimizer_cls(
			[
				{"params": nopt_params, "weight_decay": self.hparams.weight_decay},
				{"params": pt_params, "weight_decay": self.hparams.mf_pt_weight_decay},
			], 
			lr=self.hparams.lr
		)
		ret = {
			"optimizer": optimizer,
		}
		assert not (self.hparams.lr_schedule and self.hparams.mf_lr_schedule)
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
		if self.hparams.mf_lr_schedule:
			tot_updates = self.hparams.mf_tot_updates
			warmup_updates = int(0.1 * tot_updates)
			scheduler = PolynomialDecayLR(
				optimizer,
				warmup_updates=warmup_updates,
				tot_updates=tot_updates,
				lr=self.hparams.mf_high_lr,
				end_lr=self.hparams.mf_low_lr,
				power=1.0
			)
			ret["lr_scheduler"] = {
				"scheduler": scheduler,
				"frequency": 1,
				"interval": "step",
			}
		return ret
