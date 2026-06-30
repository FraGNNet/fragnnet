import torch as th

from fragnnet.model import PrecursorModel
from fragnnet.model.loss import get_sparse_cross_entropy_fn
from fragnnet.pl_model import SpectrumPL
from fragnnet.utils.misc_utils import LOG_ZERO


class PrecursorPL(SpectrumPL):
    """PyTorch Lightning module for precursor prediction with sparse CE loss."""

    def _setup_model(self) -> None:
        """Initialize the precursor classification model (no compilation supported)."""
        assert not self.hparams.compile
        self.model = PrecursorModel()

    def _setup_loss_names(self) -> None:
        """Register loss names for metric tracking."""
        # flag losses for tracking
        loss_names = [
            "loss",
            "primary_loss",
        ]
        self.loss_names = loss_names
        self.metric_names.update(loss_names)

    def _setup_loss_fn(self) -> None:
        """Configure sparse cross-entropy loss for precursor prediction."""
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
            loss_batch_size=self.hparams.loss_batch_size,
        )

        self._setup_loss_names()

        def loss_fn(
            true_mzs: th.Tensor,
            true_logprobs: th.Tensor,
            true_batch_idxs: th.Tensor,
            pred_mzs: th.Tensor,
            pred_logprobs: th.Tensor,
            pred_batch_idxs: th.Tensor,
            **kwargs,
        ) -> dict[str, th.Tensor]:
            batch_size = th.max(true_batch_idxs).item() + 1
            assert pred_mzs.shape[0] == pred_logprobs.shape[0] == batch_size
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
            }
            return loss_d

        self.loss_fn = loss_fn
        self.binned_loss = False
