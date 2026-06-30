"""
NEIMS Model for MS/MS Spectrum Prediction
This module based on NEIMS from https://pubs.acs.org/doi/10.1021/acscentsci.9b00085 and adapted to predicte MS/MS

The model uses molecular fingerprints and optional embeddings for collision energy (CE),
precursor type, and instrument type, and predicts spectra using a feedforward neural network.
"""

import torch as th
import torch.nn as nn

from fragnnet.model.base_model import CEModel, InstModel, PrecModel
from fragnnet.model.nn_blocks import SpecFFN
from fragnnet.utils.feat_utils import get_mol_fp_size


class NeimsModel(nn.Module, CEModel, PrecModel, InstModel):
    """
    NEIMS model for spectrum prediction.

    Supports molecular fingerprint input, optional CE/precursor/instrument embeddings,
    and spectrum prediction via a feedforward neural network.
    """

    def __init__(
        self,
        mol_fingerprint_morgan: bool,
        mol_fingerprint_rdkit: bool,
        mol_fingerprint_maccs: bool,
        mlp_hidden_size: int,
        mlp_dropout: float,
        mlp_num_layers: int,
        mlp_use_residuals: bool,
        mz_max: int,
        mz_bin_res: float,
        ff_prec_mz_offset: int,
        ff_bidirectional: bool,
        ff_output_map_size: int,
        ff_output_activation: str,
        int_embedder: str,
        ce_insert_type: str,
        ce_insert_location: str,
        ce_insert_merge: bool,
        ce_insert_size: int,
        use_nce: bool,
        nce_max: float,
        nce_mean: float,
        nce_std: float,
        use_ace: bool,
        ace_max: float,
        ace_mean: float,
        ace_std: float,
        prec_insert_location: str,
        prec_insert_size: int,
        prec_types: list[str],
        inst_insert_location: str,
        inst_insert_size: int,
        inst_types: list[str],
        log_min: float,
        morgan_radius: int = 3,
        morgan_nbits: int = 2048,
        rdkit_nbits: int = 2048,
        neims_bottleneck_factor: float = 0.5,
    ):
        # Initialize nn.Module
        super().__init__()

        # Store fingerprint configuration
        self.mol_fingerprint_morgan = mol_fingerprint_morgan
        self.mol_fingerprint_rdkit = mol_fingerprint_rdkit
        self.mol_fingerprint_maccs = mol_fingerprint_maccs

        # Compute input size from fingerprint types
        self.mol_fp_dim = get_mol_fp_size(
            self.mol_fingerprint_morgan,
            self.mol_fingerprint_rdkit,
            self.mol_fingerprint_maccs,
            morgan_radius=morgan_radius,
            morgan_nbits=morgan_nbits,
            rdkit_nbits=rdkit_nbits,
        )
        self.mlp_input_dim = self.mol_fp_dim

        # Initialize collision energy embedding (if used)
        self._ce_init(
            int_embedder=int_embedder,
            ce_insert_type=ce_insert_type,
            ce_insert_location=ce_insert_location,
            ce_insert_merge=ce_insert_merge,
            ce_insert_size=ce_insert_size,
            nce_max=nce_max,
            nce_mean=nce_mean,
            nce_std=nce_std,
            use_nce=use_nce,
            use_ace=use_ace,
            ace_max=ace_max,
            ace_mean=ace_mean,
            ace_std=ace_std,
        )
        self.mlp_input_dim += self.ce_mlp_input_dim

        # Initialize precursor type embedding (if used)
        self._prec_init(
            prec_insert_location=prec_insert_location,
            prec_insert_size=prec_insert_size,
            prec_num_types=len(prec_types),
        )
        self.mlp_input_dim += self.prec_mlp_input_dim

        # Initialize instrument type embedding (if used)
        self._inst_init(
            inst_insert_location=inst_insert_location,
            inst_insert_size=inst_insert_size,
            inst_num_types=len(inst_types),
        )
        self.mlp_input_dim += self.inst_mlp_input_dim

        # Feedforward network for spectrum prediction
        self.ffn = SpecFFN(
            input_size=self.mlp_input_dim,
            hidden_size=mlp_hidden_size,
            mz_max=mz_max,
            mz_bin_res=mz_bin_res,
            num_layers=mlp_num_layers,
            dropout=mlp_dropout,
            use_residuals=mlp_use_residuals,
            bidirectional=ff_bidirectional,
            prec_mz_offset=ff_prec_mz_offset,
            output_map_size=ff_output_map_size,
            output_activation=ff_output_activation,
            log_min=log_min,
            bottleneck_factor=neims_bottleneck_factor,
        )

    def _ce_location_check(self):
        assert self.ce_insert_location in ["mlp", "none"], (
            f"ce_insert_location={self.ce_insert_location} not supported"
        )

    def _prec_location_check(self):
        assert self.prec_insert_location in ["mlp", "none"], (
            f"prec_insert_location={self.prec_insert_location} not supported"
        )

    def _inst_location_check(self):
        assert self.inst_insert_location in ["mlp", "none"], (
            f"prec_insert_location={self.inst_insert_location} not supported"
        )

    def forward(
        self,
        mol_fingerprint: th.Tensor,
        spec_prec_mz: th.Tensor,
        spec_ce: th.Tensor = None,
        spec_ce_batch_idxs: th.Tensor = None,
        spec_prec_type: th.Tensor = None,
        spec_inst_type: th.Tensor = None,
        **kwargs,
    ):
        """
        Forward pass for NEIMS model.

        Args:
            mol_fingerprint: Molecular fingerprint tensor.
            spec_prec_mz: Precursor m/z tensor.
            spec_ce: Collision energy tensor (optional).
            spec_ce_batch_idxs: Batch indices for CE (optional).
            spec_prec_type: Precursor type tensor (optional).
            spec_inst_type: Instrument type tensor (optional).

        Returns:
            Dictionary with predicted m/z, log-probabilities, batch indices, and spectra.
        """
        fh = mol_fingerprint.reshape(-1, self.mol_fp_dim)
        batch_size = fh.shape[0]

        # Embed collision energy, precursor type, and instrument type if used
        ce = spec_ce
        ce_batch_idxs = spec_ce_batch_idxs
        ce_embed = self.embed_ce(ce, ce_batch_idxs, batch_size)
        prec_embed = self.embed_prec(spec_prec_type)
        inst_embed = self.embed_inst(spec_inst_type)

        # Concatenate embeddings to input if configured
        if self.ce_insert_location == "mlp":
            fh = th.cat([fh, ce_embed], dim=1)
        if self.prec_insert_location == "mlp":
            fh = th.cat([fh, prec_embed], dim=1)
        if self.inst_insert_location == "mlp":
            fh = th.cat([fh, inst_embed], dim=1)

        # Predict spectrum using feedforward network
        pred_mzs, pred_logprobs, pred_batch_idxs, pred_specs = self.ffn(fh, spec_prec_mz)
        out_d = {
            "pred_mzs": pred_mzs,
            "pred_logprobs": pred_logprobs,
            "pred_batch_idxs": pred_batch_idxs,
            "pred_specs": pred_specs,
        }
        return out_d
