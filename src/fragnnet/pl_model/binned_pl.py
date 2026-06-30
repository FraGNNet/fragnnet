import torch as th

from fragnnet.model.loss import sparse_cosine_distance_binned
from fragnnet.pl_model import SpectrumPL


class BinnedPL(SpectrumPL):
    """
    PyTorch Lightning module for binned spectrum prediction using cosine distance loss.

    Inherits from SpectrumPL and overrides the loss function to use binned sparse
    cosine distance for matching predicted and ground truth spectra.
    """

    def _setup_loss_fn(self) -> None:
        """
        Setup loss function using binned sparse cosine distance.

        Configures the loss function to compute cosine distance between predicted
        and ground truth spectra in binned m/z space. Sets up loss tracking metrics.
        """

        def cos_dist_fn(
            true_mzs: th.Tensor,
            true_logprobs: th.Tensor,
            true_batch_idxs: th.Tensor,
            pred_mzs: th.Tensor,
            pred_logprobs: th.Tensor,
            pred_batch_idxs: th.Tensor,
        ) -> th.Tensor:
            """
            Compute binned sparse cosine distance between true and predicted spectra.

            Args:
                true_mzs: Ground truth m/z values (sparse format)
                true_logprobs: Ground truth log probabilities (sparse format)
                true_batch_idxs: Batch indices for true spectra
                pred_mzs: Predicted m/z values (binned format)
                pred_logprobs: Predicted log probabilities (binned format)
                pred_batch_idxs: Batch indices for predicted spectra

            Returns:
                Cosine distance between true and predicted spectra
            """
            return sparse_cosine_distance_binned(
                true_mzs,
                true_logprobs,
                true_batch_idxs,
                pred_mzs,
                pred_logprobs,
                pred_batch_idxs,
                log_distance=(self.hparams.loss_type == "log_cosine_distance"),
            )

        assert self.hparams.loss_type == "cosine_distance", self.hparams.loss_type
        assert self.hparams.sparse_cosine_similarity
        self.binned_loss = True

        def _loss_fn(
            true_mzs: th.Tensor,
            true_logprobs: th.Tensor,
            true_batch_idxs: th.Tensor,
            pred_mzs: th.Tensor,
            pred_logprobs: th.Tensor,
            pred_batch_idxs: th.Tensor,
            **kwargs,
        ) -> dict[str, th.Tensor]:
            """
            Compute loss and related metrics.

            Args:
                true_mzs: Ground truth m/z values
                true_logprobs: Ground truth log probabilities
                true_batch_idxs: Batch indices for true spectra
                pred_mzs: Predicted m/z values
                pred_logprobs: Predicted log probabilities
                pred_batch_idxs: Batch indices for predicted spectra
                **kwargs: Additional unused arguments

            Returns:
                Dictionary with 'loss' and 'spec_cd' keys
            """
            spec_cd = cos_dist_fn(
                true_mzs, true_logprobs, true_batch_idxs, pred_mzs, pred_logprobs, pred_batch_idxs
            )
            loss = spec_cd
            loss_d = {"loss": loss, "spec_cd": spec_cd}
            return loss_d

        self.loss_fn = _loss_fn
        loss_names = ["loss", "spec_cd"]
        self.metric_names.update(loss_names)
