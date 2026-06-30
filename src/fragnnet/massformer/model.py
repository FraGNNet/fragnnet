import torch as th
import torch.nn as nn

from fragnnet.massformer.nn_utils import GFv2Embedder
from fragnnet.model import CEModel, InstModel, PrecModel
from fragnnet.model.nn_blocks import SpecFFN
from fragnnet.utils.nn_utils import *


class MassFormerModel(nn.Module, CEModel, PrecModel, InstModel):
    def __init__(
        self,
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
        mf_fix_num_pt_layers: int,
        mf_reinit_num_pt_layers: int,
        mf_reinit_layernorm: bool,
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
    ):
        # nn.Module init
        super().__init__()

        self.gf = GFv2Embedder(
            fix_num_pt_layers=mf_fix_num_pt_layers,
            reinit_num_pt_layers=mf_reinit_num_pt_layers,
            reinit_layernorm=mf_reinit_layernorm,
        )
        self.mlp_input_dim = self.gf.get_embed_dim()

        # ce stuff
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

        # prec stuff
        self._prec_init(
            prec_insert_location=prec_insert_location,
            prec_insert_size=prec_insert_size,
            prec_num_types=len(prec_types),
        )
        self.mlp_input_dim += self.prec_mlp_input_dim

        # inst stuff
        self._inst_init(
            inst_insert_location=inst_insert_location,
            inst_insert_size=inst_insert_size,
            inst_num_types=len(inst_types),
        )
        self.mlp_input_dim += self.inst_mlp_input_dim

        self.ffn = SpecFFN(
            input_size=self.mlp_input_dim,
            hidden_size=mlp_hidden_size,
            mz_max=mz_max,
            mz_bin_res=mz_bin_res,
            num_layers=mlp_num_layers,
            dropout=mlp_dropout,
            prec_mz_offset=ff_prec_mz_offset,
            bidirectional=ff_bidirectional,
            use_residuals=mlp_use_residuals,
            output_map_size=ff_output_map_size,
            output_activation=ff_output_activation,
            log_min=log_min,
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
        assert self.prec_insert_location in ["mlp", "none"], (
            f"prec_insert_location={self.prec_insert_location} not supported"
        )

    def get_split_params(self):
        nopt_params, pt_params = self.gf.get_split_params()
        pt_params.extend(list(self.ffn.parameters()))
        return nopt_params, pt_params

    def forward(
        self,
        mol_mf_idx: th.Tensor,
        mol_mf_attn_bias: th.Tensor,
        mol_mf_attn_edge_type: th.Tensor,
        mol_mf_spatial_pos: th.Tensor,
        mol_mf_in_degree: th.Tensor,
        mol_mf_out_degree: th.Tensor,
        mol_mf_x: th.Tensor,
        mol_mf_edge_input: th.Tensor,
        mol_mf_y: th.Tensor,
        spec_prec_mz: th.Tensor,
        spec_ce: th.Tensor = None,
        spec_ce_batch_idxs: th.Tensor = None,
        spec_prec_type: th.Tensor = None,
        spec_inst_type: th.Tensor = None,
        mf_perturb: th.Tensor = None,
        **kwargs,
    ):
        gf_d = {
            "idx": mol_mf_idx,
            "attn_bias": mol_mf_attn_bias,
            "attn_edge_type": mol_mf_attn_edge_type,
            "spatial_pos": mol_mf_spatial_pos,
            "in_degree": mol_mf_in_degree,
            "out_degree": mol_mf_out_degree,
            "x": mol_mf_x,
            "edge_input": mol_mf_edge_input,
            "y": mol_mf_y,
        }
        fh = self.gf(gf_d, perturb=mf_perturb)
        batch_size = fh.shape[0]
        # get ce
        ce = spec_ce
        ce_batch_idxs = spec_ce_batch_idxs
        ce_embed = self.embed_ce(ce, ce_batch_idxs, batch_size)
        prec_embed = self.embed_prec(spec_prec_type)
        inst_embed = self.embed_inst(spec_inst_type)
        if self.ce_insert_location == "mlp":
            fh = th.cat([fh, ce_embed], dim=1)
        if self.prec_insert_location == "mlp":
            fh = th.cat([fh, prec_embed], dim=1)
        if self.inst_insert_location == "mlp":
            fh = th.cat([fh, inst_embed], dim=1)

        # apply ffn
        pred_mzs, pred_logprobs, pred_batch_idxs, pred_specs = self.ffn(fh, spec_prec_mz)
        out_d = {
            "pred_mzs": pred_mzs,
            "pred_logprobs": pred_logprobs,
            "pred_batch_idxs": pred_batch_idxs,
            "pred_specs": pred_specs,
        }
        return out_d
