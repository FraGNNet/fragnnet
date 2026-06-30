import torch as th
from torch import _dynamo as th_dynamo

try:
    # this is available in PyTorch 2.0+ till 2.4
    from torch._dynamo.utils import CompileProfiler

    profiler_available = True
except ImportError:
    profiler_available = False

from fragnnet.graff.data_utils import e_map, x_map
from fragnnet.graff.model import GrAFFModel
from fragnnet.model.loss import get_sparse_cross_entropy_fn
from fragnnet.pl_model import SpectrumPL
from fragnnet.utils.misc_utils import LOG_ZERO


class GrAFFPL(SpectrumPL):
    def _setup_model(self):
        node_feats = {**x_map, **{"is_virtual_node": {False: 0, True: 1}}}
        edge_feats = {
            **e_map,
            **{
                "is_virtual_in_edge": {False: 0, True: 1},
                "is_virtual_out_edge": {False: 0, True: 1},
            },
        }

        self.model = GrAFFModel(
            vocab_size=self.hparams.ann_params["library_size"],
            num_isotope_types=self.hparams.ann_params["max_isotope"] + 1,
            encoder_dim=self.hparams.graff_encoder_dim,
            decoder_dim=self.hparams.graff_decoder_dim,
            encoder_depth=self.hparams.graff_encoder_depth,
            decoder_depth=self.hparams.graff_decoder_depth,
            num_eigs=self.hparams.ann_params["num_eigs"],
            eig_dim=self.hparams.graff_eig_dim,
            eig_depth=self.hparams.graff_eig_depth,
            dropout=self.hparams.graff_dropout,
            min_probability=self.hparams.graff_min_probability,
            min_mz=self.hparams.graff_min_mz,
            node_feats=node_feats,
            edge_feats=edge_feats,
            output_formula_str=self.hparams.output_formula_str,
        )

        # compile
        if self.hparams.compile:
            th_dynamo.reset()
            if profiler_available:
                self.dynamo_prof = CompileProfiler()
                self.model = self.model.get_compile(backend=self.dynamo_prof, dynamic=True)
            else:
                self.model = th.compile(self.model, dynamic=True)

    def _setup_loss_names(self):
        # flag losses for tracking
        loss_names = [
            "loss",
            "primary_loss",
        ]
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
            loss_batch_size=self.hparams.loss_batch_size,
        )

        assert self.hparams.loss_type == "cross_entropy", self.hparams.loss_type
        # assert self.hparams.output_distribution == "peak_marginal", self.hparams.output_distribution
        self.binned_loss = False

        self._setup_loss_names()

        def _loss_fn(
            true_mzs,
            true_logprobs,
            true_batch_idxs,
            pred_mzs,
            pred_logprobs,
            pred_batch_idxs,
            **kwargs,
        ):
            batch_size = th.max(true_batch_idxs) + 1
            ios_ce, oos_ce, true_oos_logprob, true_oos_e = sparse_ce_fn(
                true_mzs,
                true_logprobs,
                true_batch_idxs,
                pred_mzs,
                pred_logprobs,
                pred_batch_idxs,
                th.full(
                    (batch_size,),
                    LOG_ZERO(true_logprobs.dtype),
                    device=pred_logprobs.device,
                    dtype=pred_logprobs.dtype,
                ),
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
                # "spec_ce": spec_ce
            }
            return loss_d

        self.loss_fn = _loss_fn
