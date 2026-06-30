"""
FragGNNMet Model for MS/MS Spectrum Prediction
"""

import logging

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg

from fragnnet.model.base_model import (
    CEModel,
    CEScaler,
    FragModeModel,
    FragModeScaler,
    InstModel,
    PrecModel,
)
from fragnnet.model.form_embedder import get_embedder
from fragnnet.model.nn_blocks import GNN, MLPBlocks, build_pool_module
from fragnnet.utils.data_utils import combine_formulae
from fragnnet.utils.feat_utils import get_mol_feats_sizes
from fragnnet.utils.formula_utils import PREC_TYPE_TO_FORMULA_DIFF
from fragnnet.utils.frag_utils import (
    compute_boundary_mask,
    get_edge_feats,
    get_node_feats,
    th_long_to_mask,
)
from fragnnet.utils.misc_utils import (
    LOG_ZERO,
    check_pyg_compile,
    check_pyg_full_compile,
    scatter_logsoftmax,
    scatter_logsumexp,
    scatter_masked_softmax,
    scatter_reduce,
)
from fragnnet.utils.spec_utils import batched_bin_func

logger = logging.getLogger(__name__)


class FraGNNetModel(nn.Module, CEModel, PrecModel, InstModel, FragModeModel):
    def __init__(
        self,
        num_depth: int,
        num_hs: int,
        num_elements: int,
        int_embedder: str,
        int_embedder_tight: bool,
        mol_node_feats: list[str],
        mol_edge_feats: list[str],
        mol_pe_embed_k: int,
        mol_hidden_size: int,
        mol_num_layers: int,
        mol_gnn_type: str,
        mol_normalization: str,
        mol_dropout: float,
        mol_num_heads: int,
        mol_pool_type: str,
        frag_node_feats: list[str],
        frag_edge_feats: list[str],
        frag_hidden_size: int,
        frag_num_layers: int,
        frag_gnn_type: str,
        frag_normalization: str,
        frag_dropout: float,
        frag_pool_type: str,
        frag_embed_combine: str,
        frag_pool_combine: str,
        mlp_output_format: str,
        mlp_hidden_size: int,
        mlp_normalization: str,
        mlp_dropout: float,
        mlp_num_layers: int,
        mlp_use_residuals: bool,
        cc_interstage_type: str,
        cc_interstage_use_rest: bool,
        nb_iso: bool,
        skip_edge_loss: bool,
        mask_null_formula: bool,
        predict_oos: bool,
        bin_output: bool,
        mz_bin_res: float,
        mz_max: float,
        ce_insert_location: str,
        ce_insert_type: str,
        ce_insert_merge: bool,
        ce_insert_size: int,
        nce_max: float,
        nce_mean: float,
        nce_std: float,
        prec_insert_location: str,
        prec_insert_size: int,
        prec_types: list[str],
        inst_insert_location: str,
        inst_insert_size: int,
        inst_types: list[str],
        frag_mode_insert_location: str,
        frag_mode_insert_size: int,
        frag_modes: list[str],
        frag_mode_scale: bool,
        ce_scale: bool,
        ce_scaler_hidden_dim: int,
        output_formula_str: bool,
        use_nce: bool,
        use_ace: bool,
        ace_max: float,
        ace_mean: float,
        ace_std: float,
        cc_feature_dropout: float,
        debug_validate_outputs: bool = False,
    ):
        # nn.Module init
        super().__init__()

        self._ce_init(
            int_embedder=int_embedder,
            ce_insert_location=ce_insert_location,
            ce_insert_type=ce_insert_type,
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

        self._prec_init(
            prec_insert_location=prec_insert_location,
            prec_insert_size=prec_insert_size,
            prec_num_types=len(prec_types),
        )

        self._inst_init(
            inst_insert_location=inst_insert_location,
            inst_insert_size=inst_insert_size,
            inst_num_types=len(inst_types),
        )

        self._frag_mode_init(
            frag_mode_insert_location=frag_mode_insert_location,
            frag_mode_insert_size=frag_mode_insert_size,
            frag_mode_num_types=len(frag_modes),
        )
        self.frag_modes = frag_modes

        self.num_depth = num_depth
        self.num_hs = num_hs
        self.num_elements = num_elements
        self.debug_validate_outputs = debug_validate_outputs
        if not 0.0 <= cc_feature_dropout <= 1.0:
            raise ValueError(f"cc_feature_dropout must be in [0, 1], got {cc_feature_dropout}")
        self.cc_feature_dropout = float(cc_feature_dropout)

        # calculate node/edge feats sizes
        self.mol_node_feats = mol_node_feats
        self.mol_edge_feats = mol_edge_feats
        self.mol_pe_embed_k = mol_pe_embed_k
        self._compute_mol_feats_sizes()

        # setup mol gnn
        self.mol_node_feats_size += (
            self.ce_mol_input_dim
            + self.prec_mol_input_dim
            + self.inst_mol_input_dim
            + self.frag_mode_mol_input_dim
        )
        mol_kwargs = {
            "node_feats_size": self.mol_node_feats_size,
            "edge_feats_size": self.mol_edge_feats_size,
            "hidden_size": mol_hidden_size,
            "num_layers": mol_num_layers,
            "gnn_type": mol_gnn_type,
            "dropout": mol_dropout,
            "normalization": mol_normalization,
            "num_heads": mol_num_heads,
        }

        # Mol GNN
        self.mol_embedder = GNN(**mol_kwargs)
        self.mol_pool_type = mol_pool_type
        self.mol_pool = build_pool_module(mol_pool_type, mol_hidden_size)
        if mol_pool_type in ["mean_max_sum", "mean_std_softmax"]:
            mol_pool_out_size = mol_hidden_size * 3
        else:
            mol_pool_out_size = mol_hidden_size

        if int_embedder_tight:
            formula_d = {"max_count_int": 255}
            depth_d = {"max_count_int": num_depth + 1}
            complement_d = {"max_count_int": 2}
        else:
            formula_d = depth_d = complement_d = {}
        self.formula_embedder = get_embedder(int_embedder, **formula_d)
        self.depth_embedder = get_embedder(int_embedder, **depth_d)
        self.complement_embedder = get_embedder(int_embedder, **complement_d)
        self.frag_node_feats = frag_node_feats
        self.frag_edge_feats = frag_edge_feats
        self._compute_frag_feats_sizes()
        self.mlp_output_dim = 2 * self.num_hs + 1
        if "cmf_h_formulae_idx" in self.frag_node_feats:
            # if cmf, we need to double the output dim
            self.mlp_output_dim *= 2

        # define interstage
        assert cc_interstage_type in [
            "add",
            "sub",
            "linear",
            "direct",
            "gated",
            "mlp",
            "bilinear",
        ]
        self.cc_interstage_type = cc_interstage_type
        if self.cc_interstage_type == "linear":
            self.cc_interstage = nn.Linear(mol_pool_out_size * 2, mol_pool_out_size)
        elif self.cc_interstage_type == "gated":
            # Gated fusion: learns to mix frag and mol embeddings
            self.cc_interstage = nn.Sequential(
                nn.Linear(mol_pool_out_size * 2, mol_pool_out_size), nn.Sigmoid()
            )
        elif self.cc_interstage_type == "mlp":
            # Deep fusion using MLPBlocks (more flexible, parameter-efficient with residuals)
            self.cc_interstage = MLPBlocks(
                input_size=mol_pool_out_size * 2,
                output_size=mol_pool_out_size,
                hidden_size=mlp_hidden_size,
                num_layers=2,
                dropout=mlp_dropout,
                use_residuals=mlp_use_residuals,
                normalization=mlp_normalization,
            )
        elif self.cc_interstage_type == "bilinear":
            # Bilinear interaction
            self.cc_interstage_bilinear = nn.Bilinear(
                mol_pool_out_size, mol_pool_out_size, mol_pool_out_size
            )
            self.cc_interstage_proj = nn.Linear(mol_pool_out_size, mol_pool_out_size)

        if "boundary_pair" in self.frag_node_feats:
            # Projects cat(inside_atom_embed, outside_atom_embed) → mol_hidden_size.
            # Pooling over pairs then yields mol_pool_out_size (same as cc / boundary_cc).
            # TODO(next version): remove the legacy boundary_pair feature path entirely.
            self.boundary_pair_proj = nn.Linear(mol_hidden_size * 2, mol_hidden_size)

        frag_kwargs = {
            "node_feats_size": self.frag_node_feats_size,
            "edge_feats_size": self.frag_edge_feats_size,
            "hidden_size": frag_hidden_size,
            "num_layers": frag_num_layers,
            "gnn_type": frag_gnn_type,
            "dropout": frag_dropout,
            "normalization": frag_normalization,
        }

        self.frag_embedder = GNN(**frag_kwargs)
        self.frag_pool_type = frag_pool_type
        self.frag_pool = build_pool_module(frag_pool_type, frag_hidden_size)
        self.frag_embed_combine = frag_embed_combine
        self.frag_pool_combine = frag_pool_combine
        self.frag_pool_preserves_hidden_size = self.frag_pool_type not in [
            "mean_max_sum",
            "mean_std_softmax",
        ]
        if (
            self.frag_pool_combine in ["subtract", "add"]
            and not self.frag_pool_preserves_hidden_size
        ):
            raise ValueError(
                "frag_pool_combine must be 'none' when frag_pool_type expands feature width. "
                f"Received frag_pool_type={self.frag_pool_type!r}, "
                f"frag_pool_combine={self.frag_pool_combine!r}."
            )
        self.mlp_output_format = mlp_output_format

        # determine mlp input dims
        if self.frag_embed_combine == "cat":
            mlp_input_dim = 2 * self.frag_embedder.hidden_size
        else:
            assert self.frag_embed_combine == "avg", self.frag_embed_combine
            mlp_input_dim = self.frag_embedder.hidden_size
        mlp_input_dim += (
            self.ce_mlp_input_dim
            + self.prec_mlp_input_dim
            + self.inst_mlp_input_dim
            + self.frag_mode_mlp_input_dim
        )

        #  mlp for formula
        if self.mlp_output_format in ["formula", "node_formula"]:
            formula_mlp_kwargs = {
                "input_size": mlp_input_dim,
                "output_size": self.mlp_output_dim,
                "hidden_size": mlp_hidden_size,
                "num_layers": mlp_num_layers,
                "dropout": mlp_dropout,
                "use_residuals": mlp_use_residuals,
                "normalization": mlp_normalization,
            }
            self.formula_module = MLPBlocks(**formula_mlp_kwargs)

        # mode-conditioned output scaling (applied after formula_module)
        self.frag_mode_scale = frag_mode_scale
        if self.frag_mode_scale:
            self.frag_mode_scaler = FragModeScaler(
                num_frag_modes=len(frag_modes),
                output_dim=self.mlp_output_dim,
            )
        else:
            self.frag_mode_scaler = None

        # CE-conditioned FiLM output scaling (applied after frag_mode_scaler)
        self.ce_scale = ce_scale
        if self.ce_scale:
            self.ce_scaler = CEScaler(
                nce_mean=nce_mean,
                nce_std=nce_std,
                use_nce=use_nce,
                ace_mean=ace_mean,
                ace_std=ace_std,
                use_ace=use_ace,
                output_dim=self.mlp_output_dim,
                hidden_dim=ce_scaler_hidden_dim,
            )
        else:
            self.ce_scaler = None

        # use extra mlp for node formula
        if self.mlp_output_format in ["node_formula"]:
            node_mlp_kwargs = {
                "input_size": 2 * self.frag_embedder.hidden_size,
                "output_size": 1,
                "hidden_size": mlp_hidden_size,
                "num_layers": mlp_num_layers,
                "dropout": mlp_dropout,
                "use_residuals": mlp_use_residuals,
                "normalization": mlp_normalization,
            }
            self.node_module = MLPBlocks(**node_mlp_kwargs)
        else:
            self.node_module = None

        self.predict_oos = predict_oos
        if self.predict_oos:
            # Account for pooling output sizes
            if self.frag_pool_type in ["mean_max_sum", "mean_std_softmax"]:
                frag_pool_out_size = self.frag_embedder.hidden_size * 3
            else:
                frag_pool_out_size = self.frag_embedder.hidden_size

            oos_mlp_kwargs = {
                "input_size": mol_pool_out_size + frag_pool_out_size,
                "output_size": 1,
                "hidden_size": mlp_hidden_size,
                "num_layers": mlp_num_layers,
                "dropout": mlp_dropout,
                "use_residuals": mlp_use_residuals,
                "normalization": "none",  # never normalized
            }
            self.oos_module = MLPBlocks(**oos_mlp_kwargs)
        else:
            self.oos_module = None

        self.skip_edge_loss = skip_edge_loss
        self.mask_null_formula = mask_null_formula
        self.nb_iso = nb_iso
        self.bin_output = bin_output
        self.mz_bin_res = mz_bin_res
        self.mz_max = mz_max
        self.output_formula_str = output_formula_str
        self.cc_interstage_use_rest = cc_interstage_use_rest

        if self.bin_output:
            self.register_buffer("mz_bins", th.arange(mz_bin_res, mz_max + mz_bin_res, mz_bin_res), persistent=False)

        # this is required
        assert "h_formulae_idx" in self.frag_node_feats

        # Pre-compute h_counts as a buffer — fixed for the lifetime of the model.
        # Layout: [0, -1, +1, -2, +2, ..., -num_hs, +num_hs]
        _h_counts = th.zeros([2 * num_hs + 1], dtype=th.long)
        _h_counts[1 + 2 * th.arange(num_hs)] = -th.arange(1, num_hs + 1)
        _h_counts[2 + 2 * th.arange(num_hs)] = th.arange(1, num_hs + 1)
        include_cmf = "cmf_h_formulae_idx" in frag_node_feats
        if include_cmf:
            _h_counts = th.cat([_h_counts, _h_counts], dim=0)
        self.register_buffer("_h_counts_buf", _h_counts, persistent=False)
        _h_counts_range = th.arange(-num_hs, num_hs + 1, dtype=th.long)
        if include_cmf:
            _h_counts_range = th.cat([_h_counts_range, _h_counts_range])
        self.register_buffer("_h_counts_range_buf", _h_counts_range, persistent=False)

    def _maybe_apply_cc_feature_dropout(self, frag_node_mask_embed: th.Tensor) -> th.Tensor:
        """Drop the full-fragment cc feature group for a whole training batch."""
        if not self.training or self.cc_feature_dropout <= 0.0 or "cc" not in self.frag_node_feats:
            return frag_node_mask_embed
        keep = th.rand((), device=frag_node_mask_embed.device) >= self.cc_feature_dropout
        return frag_node_mask_embed * keep.to(dtype=frag_node_mask_embed.dtype)

    def _ce_location_check(self):
        assert self.ce_insert_location != "frag", "ce_insert_location=frag not supported"

    def _prec_location_check(self):
        assert self.prec_insert_location != "frag", "prec_insert_location=frag not supported"

    def _inst_location_check(self):
        assert self.inst_insert_location != "frag", "inst_insert_location=frag not supported"

    def _frag_mode_location_check(self):
        assert self.frag_mode_insert_location != "frag", (
            "frag_mode_insert_location=frag not supported"
        )

    def _compute_mol_feats_sizes(self):
        """method compute mol feature size
        these features don't rely on any model parameters
        """
        self.mol_node_feats_size, self.mol_edge_feats_size = get_mol_feats_sizes(
            self.mol_node_feats, self.mol_edge_feats, self.mol_pe_embed_k
        )

    def _compute_frag_feats_sizes(self):
        """method compute frag-graph feature size
        these features do depend on model parameters
        """
        # Determine the actual output size after pooling
        if self.mol_pool_type in ["mean_max_sum", "mean_std_softmax"]:
            mol_pool_out_size = self.mol_embedder.hidden_size * 3
        else:
            mol_pool_out_size = self.mol_embedder.hidden_size

        # nodes
        self.frag_node_feats_size = 0
        if "cc" in self.frag_node_feats:
            self.frag_node_feats_size += mol_pool_out_size
        if "boundary_cc" in self.frag_node_feats:
            self.frag_node_feats_size += mol_pool_out_size
        if "boundary_pair" in self.frag_node_feats:
            self.frag_node_feats_size += mol_pool_out_size
        if "base_formula" in self.frag_node_feats:
            self.frag_node_feats_size += self.num_elements * self.formula_embedder.num_dim
        if "neutral_loss_formula" in self.frag_node_feats:
            self.frag_node_feats_size += self.num_elements * self.formula_embedder.num_dim
        if "depth" in self.frag_node_feats:
            self.frag_node_feats_size += self.num_depth * self.depth_embedder.num_dim
        # edges
        self.frag_edge_feats_size = 0
        if "cc" in self.frag_edge_feats:
            self.frag_edge_feats_size += mol_pool_out_size
        if "base_formula" in self.frag_edge_feats:
            self.frag_edge_feats_size += self.num_elements * self.formula_embedder.num_dim
        if "complement" in self.frag_edge_feats:
            self.frag_edge_feats_size += self.complement_embedder.num_dim

    def get_compile(self, **kwargs):
        if check_pyg_full_compile():
            return th.compile(self, **kwargs)
        else:
            self.compile_submodules(**kwargs)
            return self

    def compile_submodules(self, **kwargs):
        """pyg does not support dynamic shape compiling"""
        self.formula_embedder = th.compile(self.formula_embedder, **kwargs)
        self.depth_embedder = th.compile(self.depth_embedder, **kwargs)
        self.complement_embedder = th.compile(self.complement_embedder, **kwargs)
        if hasattr(self, "ce_embedder"):
            self.ce_embedder = th.compile(self.ce_embedder, **kwargs)
        if hasattr(self, "m_ce_embedder"):
            self.m_ce_embedder = th.compile(self.m_ce_embedder, **kwargs)
        if hasattr(self, "boundary_pair_proj"):
            self.boundary_pair_proj = th.compile(self.boundary_pair_proj, **kwargs)
        if check_pyg_compile():
            self.mol_embedder = pyg.compile(self.mol_embedder, **kwargs)  # type: ignore[assignment]
            self.frag_embedder = pyg.compile(self.frag_embedder, **kwargs)  # type: ignore[assignment]

    def forward(
        self,
        mol_pyg: pyg.data.Data,
        frag_pyg: pyg.data.Data,
        mol_num_nodes: th.Tensor,
        frag_num_nodes: th.Tensor,
        frag_formula_peak_idxs: th.Tensor,
        frag_formula_peak_mzs: th.Tensor,
        frag_formula_peak_probs: th.Tensor,
        frag_formula_sizes: th.Tensor,
        frag_formula_cumsizes: th.Tensor,
        frag_formula_peak_sizes: th.Tensor,
        frag_formula_str: np.ndarray | None = None,
        spec_ce: th.Tensor | None = None,
        spec_ce_batch_idxs: th.Tensor | None = None,
        spec_prec_type: th.Tensor | None = None,
        spec_inst_type: th.Tensor | None = None,
        spec_frag_mode: th.Tensor | None = None,
        spec_prec_type_str: np.ndarray | None = None,
        return_explainability_tensors: bool = False,
        **kwargs,
    ):
        """forward methods for joint predictor

        Args:
            mol_pyg (pyg.data.Data): molecule pyg data object
            frag_pyg (pyg.data.Data): fragmentation graph pyg data object
            mol_num_nodes (th.Tensor): number of nodes in molecule graph
            frag_num_nodes (th.Tensor): number of nodes in fragmentation graph
            frag_formula_peak_idxs (th.Tensor): _description_
            frag_formula_peak_mzs (th.Tensor): _description_
            frag_formula_peak_probs (th.Tensor): _description_
            frag_formula_sizes (th.Tensor): _description_
            frag_formula_cumsizes (th.Tensor): _description_
            frag_formula_peak_sizes (th.Tensor): _description_

        Returns:
            _type_: _description_
        """

        # Fix: Explicitly cast PyG batch tensors for autocast compatibility
        # if th.is_autocast_enabled():
        #   mol_pyg.x = mol_pyg.x.to(th.bfloat16)
        #   if mol_pyg.edge_attr is not None:
        #       mol_pyg.edge_attr = mol_pyg.edge_attr.to(th.bfloat16)
        #   frag_pyg.x = frag_pyg.x.to(th.bfloat16)
        #   if frag_pyg.edge_attr is not None:
        #       frag_pyg.edge_attr = frag_pyg.edge_attr.to(th.bfloat16)

        # mol_x: mol level node feature matrix
        # mol_edge_index: mol graph connectivity in COO format with shape [2, num_edges]
        # edge_attr: mol graph edge feature matrix with shape [num_edges, num_edge_features]
        # batch: sample idx repsect to current batch
        mol_x, mol_edge_index, mol_edge_attr, mol_batch = (
            mol_pyg.x,
            mol_pyg.edge_index,
            mol_pyg.edge_attr,
            mol_pyg.batch,
        )

        # frag_x: frag-graph level node feature matrix
        # frag_edge_index: frag graph connectivity in COO format with shape [2, num_edges]
        # frag_edge_attr: frag graph edge feature matrix with shape [num_edges, num_edge_features]
        # batch: sample idx repsect to current batch
        frag_x, frag_edge_index, frag_edge_attr, frag_batch = (
            frag_pyg.x,
            frag_pyg.edge_index,
            frag_pyg.edge_attr,
            frag_pyg.batch,
        )

        # Type assertions for batch indices (PyG types them as Any | None)
        assert mol_batch is not None, "mol_batch must not be None"
        assert frag_batch is not None, "frag_batch must not be None"

        device = mol_num_nodes.device
        # int_dtype = mol_edge_index.dtype
        float_dtype = mol_edge_attr.dtype
        frag_node_feat_idxs = frag_pyg.node_feat_idxs[0]
        frag_edge_feat_idxs = frag_pyg.edge_feat_idxs[0]
        batch_frag_num_nodes = frag_x.shape[0]
        batch_frag_num_edges = frag_edge_index.shape[1]
        batch_frag_num_formulae = frag_formula_cumsizes[-1]
        batch_size = frag_batch[-1] + 1

        # get ce value
        if self.use_ace:
            ace = kwargs["spec_ace"]
            ace_batch_idxs = kwargs["spec_ace_batch_idxs"]
        else:
            ace = None
            ace_batch_idxs = None
        ce_embed = self.embed_ce(
            spec_ce, spec_ce_batch_idxs, batch_size, ace=ace, ace_batch_idxs=ace_batch_idxs
        )
        # get prec value
        prec_embed = self.embed_prec(spec_prec_type)
        # get inst value
        inst_embed = self.embed_inst(spec_inst_type)
        # get frag_mode value
        frag_mode_embed = self.embed_frag_mode(spec_frag_mode)

        # Type assertions for embedding tensors (narrow types from Any | None to Tensor)

        # Precompute per-mol node counts once from the cumulative prefix sum (avoids up to 4
        # redundant th.unique scans over mol_batch when multiple embeddings insert at "mol").
        mol_batch_counts = mol_num_nodes[1:] - mol_num_nodes[:-1]

        _mol_extras = []
        if self.ce_insert_location == "mol":
            assert ce_embed is not None, "ce_embed must not be None"
            _mol_extras.append(ce_embed)
        if self.prec_insert_location == "mol":
            assert prec_embed is not None, "prec_embed must not be None"
            _mol_extras.append(prec_embed)
        if self.inst_insert_location == "mol":
            assert inst_embed is not None, "inst_embed must not be None"
            _mol_extras.append(inst_embed)
        if self.frag_mode_insert_location == "mol":
            assert frag_mode_embed is not None, "frag_mode_embed must not be None"
            _mol_extras.append(frag_mode_embed)
        if _mol_extras:
            mol_x = th.cat(
                [mol_x, th.repeat_interleave(th.cat(_mol_extras, dim=1), mol_batch_counts, dim=0)],
                dim=1,
            )

        # get per-atom embeddings
        mol_embed_gnn = self.mol_embedder(mol_x, mol_batch, mol_edge_index, mol_edge_attr)

        # pool mol embeddings
        # For sum/mean, use segment_csr (ptr-only path) — same logic as frag_pool below.
        # mol_num_nodes is already the CSR ptr (cumulative atom counts, shape [B+1]).
        # Max is excluded: segment_csr max allocates ~3× peak memory vs scatter_max.
        if self.mol_pool_type in ("sum", "mean"):
            mol_embed_gnn_pool = self.mol_pool(
                mol_embed_gnn, index=None, ptr=mol_num_nodes, dim_size=batch_size
            )
        else:
            mol_embed_gnn_pool = self.mol_pool(mol_embed_gnn, mol_batch)

        # process dag
        # create interstage
        frag_ndata, frag_edata = [], []
        # node atom embeddings
        # frag_node_mask is needed by cc, boundary_cc, and boundary_pair features
        _needs_frag_node_mask = any(
            f in self.frag_node_feats for f in ("cc", "boundary_cc", "boundary_pair")
        )
        if _needs_frag_node_mask:
            frag_node_mask = th_long_to_mask(
                get_node_feats(frag_x, frag_node_feat_idxs, "cc").to(device)
            )
        if "cc" in self.frag_node_feats:
            frag_node_mask_idxs = th.nonzero(frag_node_mask).long()
            # Compute offset to convert local atom indices to global batch indices
            frag_node_offsets = mol_num_nodes[
                th.bucketize(frag_node_mask_idxs[:, 0], frag_num_nodes, right=True) - 1
            ]
            frag_node_mask_idxs[:, 1] = frag_node_mask_idxs[:, 1] + frag_node_offsets

            # Validate frag_node_mask_idxs are within bounds
            valid_frag_mask = frag_node_mask_idxs[:, 1] < mol_embed_gnn.shape[0]
            if not th.all(valid_frag_mask):
                logger.warning(
                    f"Filtered {(~valid_frag_mask).sum()} invalid frag node indices out of {len(frag_node_mask_idxs)}"
                )
                frag_node_mask_idxs = frag_node_mask_idxs[valid_frag_mask]

            # Aggregate CC atom embeddings per fragment using pooling
            if frag_node_mask_idxs.shape[0] == 0:
                frag_node_mask_embed = th.zeros(
                    batch_frag_num_nodes, mol_embed_gnn.shape[1], device=device, dtype=float_dtype
                )
            elif self.mol_pool_type in ("sum", "mean"):
                M_cc = th.sparse_coo_tensor(
                    th.stack([frag_node_mask_idxs[:, 0], frag_node_mask_idxs[:, 1]]),
                    th.ones(frag_node_mask_idxs.shape[0], device=device, dtype=float_dtype),
                    (batch_frag_num_nodes, mol_embed_gnn.shape[0]),
                )
                frag_node_mask_embed = th.sparse.mm(M_cc, mol_embed_gnn)
                if self.mol_pool_type == "mean":
                    degree = (
                        th.bincount(frag_node_mask_idxs[:, 0], minlength=batch_frag_num_nodes)
                        .to(float_dtype)
                        .clamp(min=1)
                        .unsqueeze(1)
                    )
                    frag_node_mask_embed = frag_node_mask_embed / degree
            else:
                frag_node_embed = th.index_select(mol_embed_gnn, 0, frag_node_mask_idxs[:, 1])
                frag_node_mask_embed = self.mol_pool(
                    frag_node_embed, frag_node_mask_idxs[:, 0], dim_size=batch_frag_num_nodes
                )

            # frag_node_mask_embed can be one of following:
            # 1. add/sub: Simple arithmetic operations
            # 2. linear: Linear projection of concatenation
            # 3. gated: Learnable gate to mix embeddings
            # 4. mlp: Deep nonlinear fusion
            # 5. bilinear: Bilinear interaction
            # 6. direct: No fusion, use fragment embedding only

            if self.cc_interstage_use_rest:
                # compute rest node idxs (non-CC atoms per fragment)
                # NOTE: ~frag_node_mask includes padding zeros (bits beyond mol size) so
                # bounds-filtering is required to exclude them.
                rest_node_mask_idx = th.nonzero(~frag_node_mask).long()
                rest_node_offsets = mol_num_nodes[
                    th.bucketize(rest_node_mask_idx[:, 0], frag_num_nodes, right=True) - 1
                ]
                rest_node_mask_idx[:, 1] = rest_node_mask_idx[:, 1] + rest_node_offsets
                valid_rest_mask = rest_node_mask_idx[:, 1] < mol_embed_gnn.shape[0]
                rest_node_mask_idx = rest_node_mask_idx[valid_rest_mask]
                rest_node_embed = th.index_select(mol_embed_gnn, 0, rest_node_mask_idx[:, 1])
                mol_pool_embed = self.mol_pool(rest_node_embed, rest_node_mask_idx[:, 0])
            else:
                mol_pool_embed = mol_embed_gnn_pool[frag_batch]

            if self.cc_interstage_type == "add":
                frag_node_mask_embed = frag_node_mask_embed + mol_pool_embed
            elif self.cc_interstage_type == "sub":
                frag_node_mask_embed = frag_node_mask_embed - mol_pool_embed
            elif self.cc_interstage_type == "linear":
                frag_node_mask_embed = self.cc_interstage(
                    th.cat([frag_node_mask_embed, mol_pool_embed], dim=1)
                )
            elif self.cc_interstage_type == "gated":
                # Gate controls mixing: g * frag + (1-g) * mol
                gate = self.cc_interstage(th.cat([frag_node_mask_embed, mol_pool_embed], dim=1))
                frag_node_mask_embed = gate * frag_node_mask_embed + (1 - gate) * mol_pool_embed
            elif self.cc_interstage_type == "mlp":
                # MLPBlocks fusion (handles residuals internally if use_residuals=True)
                frag_node_mask_embed = self.cc_interstage(
                    th.cat([frag_node_mask_embed, mol_pool_embed], dim=1)
                )
            elif self.cc_interstage_type == "bilinear":
                # Bilinear interaction: captures pairwise interactions
                interaction = self.cc_interstage_bilinear(frag_node_mask_embed, mol_pool_embed)
                frag_node_mask_embed = frag_node_mask_embed + self.cc_interstage_proj(interaction)
            else:
                assert self.cc_interstage_type == "direct", self.cc_interstage_type
                frag_node_mask_embed = frag_node_mask_embed

            frag_node_mask_embed = self._maybe_apply_cc_feature_dropout(frag_node_mask_embed)
            frag_ndata.append(frag_node_mask_embed)
        if "boundary_cc" in self.frag_node_feats:
            num_frags_total = frag_x.shape[0]
            boundary_mask = compute_boundary_mask(
                frag_node_mask, mol_edge_index, mol_num_nodes, frag_num_nodes
            )
            boundary_idxs = th.nonzero(boundary_mask).long()
            if boundary_idxs.numel() > 0:
                boundary_offsets = mol_num_nodes[
                    th.bucketize(boundary_idxs[:, 0], frag_num_nodes, right=True) - 1
                ]
                boundary_idxs[:, 1] = boundary_idxs[:, 1] + boundary_offsets
                valid = boundary_idxs[:, 1] < mol_embed_gnn.shape[0]
                boundary_idxs = boundary_idxs[valid]
                if self.mol_pool_type in ("sum", "mean"):
                    M_bcc = th.sparse_coo_tensor(
                        th.stack([boundary_idxs[:, 0], boundary_idxs[:, 1]]),
                        th.ones(boundary_idxs.shape[0], device=device, dtype=float_dtype),
                        (num_frags_total, mol_embed_gnn.shape[0]),
                    )
                    boundary_embed = th.sparse.mm(M_bcc, mol_embed_gnn)
                    if self.mol_pool_type == "mean":
                        degree = (
                            th.bincount(boundary_idxs[:, 0], minlength=num_frags_total)
                            .to(float_dtype)
                            .clamp(min=1)
                            .unsqueeze(1)
                        )
                        boundary_embed = boundary_embed / degree
                else:
                    boundary_embed = th.index_select(mol_embed_gnn, 0, boundary_idxs[:, 1])
                    boundary_embed = self.mol_pool(
                        boundary_embed, boundary_idxs[:, 0], dim_size=num_frags_total
                    )
            else:
                boundary_embed = th.zeros(
                    num_frags_total,
                    mol_embed_gnn.shape[1],
                    device=device,
                    dtype=mol_embed_gnn.dtype,
                )
            frag_ndata.append(boundary_embed)
        if "boundary_pair" in self.frag_node_feats:
            # boundary_pair requires cc to be in frag_node_feats (frag_node_mask must exist)
            # TODO(next version): delete this block when boundary_pair is removed from configs/features.
            num_frags_total = frag_x.shape[0]
            pair_frag_idxs = kwargs["boundary_pair_frag_idxs"]
            pair_in_global = kwargs["boundary_pair_in_idxs"]
            pair_out_global = kwargs["boundary_pair_out_idxs"]
            if pair_frag_idxs.numel() > 0:
                valid = (pair_in_global < mol_embed_gnn.shape[0]) & (
                    pair_out_global < mol_embed_gnn.shape[0]
                )
                pair_frag_idxs = pair_frag_idxs[valid]
                pair_in_global = pair_in_global[valid]
                pair_out_global = pair_out_global[valid]
            if pair_frag_idxs.numel() > 0:
                in_embed = th.index_select(mol_embed_gnn, 0, pair_in_global)
                out_embed = th.index_select(mol_embed_gnn, 0, pair_out_global)
                pair_embed = self.boundary_pair_proj(th.cat([in_embed, out_embed], dim=-1))
                boundary_pair_out = self.mol_pool(
                    pair_embed, pair_frag_idxs, dim_size=num_frags_total
                )
            else:
                boundary_pair_out = th.zeros(
                    num_frags_total,
                    mol_embed_gnn.shape[1],
                    device=device,
                    dtype=mol_embed_gnn.dtype,
                )
            frag_ndata.append(boundary_pair_out)
        # edge atom embeddings
        if "cc" in self.frag_edge_feats:
            # connected competents
            frag_edge_mask = th_long_to_mask(
                get_edge_feats(frag_edge_attr, frag_edge_feat_idxs, "cc").to(device)
            )
            frag_edge_mask_idxs = th.nonzero(frag_edge_mask).long()
            frag_edge_node_idxs = frag_edge_index[0][frag_edge_mask_idxs[:, 0]]
            frag_edge_offsets = mol_num_nodes[
                th.bucketize(frag_edge_node_idxs, frag_num_nodes, right=True) - 1
            ]
            frag_edge_mask_idxs[:, 1] = frag_edge_mask_idxs[:, 1] + frag_edge_offsets
            if frag_edge_mask_idxs.shape[0] == 0:
                frag_edge_mask_embed = th.zeros(
                    batch_frag_num_edges, mol_embed_gnn.shape[1], device=device, dtype=float_dtype
                )
            else:
                M_ecc = th.sparse_coo_tensor(
                    th.stack([frag_edge_mask_idxs[:, 0], frag_edge_mask_idxs[:, 1]]),
                    th.ones(frag_edge_mask_idxs.shape[0], device=device, dtype=float_dtype),
                    (batch_frag_num_edges, mol_embed_gnn.shape[0]),
                )
                frag_edge_mask_embed = th.sparse.mm(M_ecc, mol_embed_gnn)
            frag_edata.append(frag_edge_mask_embed)
        # node formulae
        if "base_formula" in self.frag_node_feats:
            frag_base_formula_raw = get_node_feats(
                frag_x, frag_node_feat_idxs, "base_formula"
            ).reshape(batch_frag_num_nodes, -1)
            frag_node_formula = self.formula_embedder(frag_base_formula_raw)
            frag_ndata.append(frag_node_formula)
        # neutral loss formula: precursor_formula - fragment_formula
        # precursor is the root node (depth bit 0 set); broadcast its base_formula to all nodes
        if "neutral_loss_formula" in self.frag_node_feats:
            assert "base_formula" in self.frag_node_feats, (
                "neutral_loss_formula requires base_formula in frag_node_feats"
            )
            assert "depth" in self.frag_node_feats, (
                "neutral_loss_formula requires depth in frag_node_feats"
            )
            depth_raw = get_node_feats(frag_x, frag_node_feat_idxs, "depth").reshape(
                batch_frag_num_nodes, -1
            )
            root_mask = depth_raw[:, 0].bool()
            prec_formula_by_batch = th.zeros(
                batch_size,
                frag_base_formula_raw.shape[1],
                dtype=frag_base_formula_raw.dtype,
                device=device,
            )
            prec_formula_by_batch[frag_batch[root_mask]] = frag_base_formula_raw[root_mask]
            neutral_loss_raw = prec_formula_by_batch[frag_batch] - frag_base_formula_raw
            frag_ndata.append(self.formula_embedder(neutral_loss_raw))
        # edge formulae
        if "base_formula" in self.frag_edge_feats:
            frag_edge_formula = self.formula_embedder(
                get_edge_feats(frag_edge_attr, frag_edge_feat_idxs, "base_formula").reshape(
                    batch_frag_num_edges, -1
                )
            )
            frag_edata.append(frag_edge_formula)
        # node depth
        if "depth" in self.frag_node_feats:
            frag_depth = self.depth_embedder(
                get_node_feats(frag_x, frag_node_feat_idxs, "depth").reshape(
                    batch_frag_num_nodes, -1
                )
            )
            frag_ndata.append(frag_depth)
        # edge complement
        if "complement" in self.frag_edge_feats:
            frag_edge_complement = self.complement_embedder(
                get_edge_feats(frag_edge_attr, frag_edge_feat_idxs, "complement").reshape(
                    batch_frag_num_edges, -1
                )
            )
            frag_edata.append(frag_edge_complement)
        # empty feats check
        if len(frag_ndata) == 0:
            assert self.frag_node_feats_size == 0, self.frag_node_feats_size
            frag_ndata.append(th.zeros([batch_frag_num_nodes, 0], dtype=float_dtype, device=device))
        if len(frag_edata) == 0:
            assert self.frag_edge_feats_size == 0, self.frag_edge_feats_size
            frag_edata.append(th.zeros([batch_frag_num_edges, 0], dtype=float_dtype, device=device))

        # charge migration fragmentation flag check
        include_cmf = "cmf_h_formulae_idx" in self.frag_node_feats
        # get output formula aggregation
        frag_node_batch_idxs = frag_batch
        frag_formula_batch_idxs = th.repeat_interleave(
            th.arange(batch_size, device=device),
            frag_formula_sizes,  # equivalent to cumsizes[1:] - cumsizes[:-1], already computed
        )
        frag_node_offsets = frag_formula_cumsizes[frag_node_batch_idxs]

        h_formulae_idx_feat = get_node_feats(frag_x, frag_node_feat_idxs, "h_formulae_idx")
        frag_joint_formula_idxs = h_formulae_idx_feat + frag_node_offsets.unsqueeze(-1)
        # When CMF features are present the node feature layout may include extra columns
        expected_h_width = 2 * self.num_hs + 1
        if not include_cmf:
            assert h_formulae_idx_feat.shape[1] >= expected_h_width, (
                f"h_formulae_idx feature size {h_formulae_idx_feat.shape[1]} is smaller than expected {expected_h_width} "
                f"(2*num_hs+1); DAG was built with fewer h-transfers than num_hs={self.num_hs}"
            )
        else:
            # allow extra columns when CMF is included but ensure minimum expected width
            if h_formulae_idx_feat.shape[1] < expected_h_width:
                raise AssertionError(
                    f"h_formulae_idx feature size {h_formulae_idx_feat.shape[1]} smaller than expected {expected_h_width} (2*num_hs+1) even with CMF enabled"
                )
        if include_cmf:
            # if we use cmf, we need to extend the formula indices
            frag_joint_formula_idxs = th.cat(
                [
                    frag_joint_formula_idxs,
                    get_node_feats(frag_x, frag_node_feat_idxs, "cmf_h_formulae_idx")
                    + frag_node_offsets.unsqueeze(-1),
                ],
                dim=1,
            )

        # Trim h-columns if the DAG was built with more h-transfers than model's num_hs.
        # This allows loading h4 DAGs while training/inferring with num_hs=3 (or lower).
        # Layout: [Δ0, Δ-1, Δ+1, Δ-2, Δ+2, ..., Δ-num_hs, Δ+num_hs] — trimming from
        # the end removes the largest-magnitude shifts first.
        frag_joint_formula_idxs = frag_joint_formula_idxs.reshape(batch_frag_num_nodes, -1)
        h_width = frag_joint_formula_idxs.shape[1] // (2 if include_cmf else 1)
        num_hs_diff = (h_width - 1) // 2 - self.num_hs
        assert num_hs_diff >= 0, (
            f"DAG max_h_transfer ({(h_width - 1) // 2}) < model num_hs ({self.num_hs}); "
            "regenerate DAGs with at least num_hs h-transfers"
        )
        if num_hs_diff > 0 and include_cmf:
            raise NotImplementedError(
                "Trimming h-shifts on CMF DAGs is not yet supported. "
                "Regenerate DAGs with max_h_transfer matching num_hs."
            )
        if num_hs_diff > 0:
            # th.unique deferred here — only needed when DAG max_h_transfer > model num_hs
            frag_formula_idxs_pretrim = th.unique(frag_joint_formula_idxs)
            # Remove last 2*num_hs_diff columns (drops Δ±(num_hs+1) … Δ±dag_max_h)
            frag_joint_formula_idxs = frag_joint_formula_idxs[:, : -2 * num_hs_diff].flatten()

            # Find unique formula indices still referenced after trimming (+ NULL per batch)
            frag_joint_formula_idxs_un, frag_joint_formula_idxs_inv = th.unique(
                th.cat([frag_joint_formula_idxs, frag_formula_cumsizes[:-1]], dim=0),
                return_inverse=True,
            )
            frag_joint_formula_idxs_inv = frag_joint_formula_idxs_inv[
                : frag_joint_formula_idxs.shape[0]
            ]

            # Mask for which pre-trim peaks survive into the trimmed set
            frag_formula_peak_mask = th.isin(
                frag_formula_idxs_pretrim[
                    ~th.isin(frag_formula_idxs_pretrim, frag_formula_cumsizes[:-1])
                ],
                frag_joint_formula_idxs_un,
            )

            # Update formula counts and batch mapping
            batch_frag_num_formulae = frag_joint_formula_idxs_un.shape[0]
            frag_formula_batch_idxs = frag_formula_batch_idxs[frag_joint_formula_idxs_un]
            frag_joint_formula_idxs = th.arange(batch_frag_num_formulae, device=device)[
                frag_joint_formula_idxs_inv
            ]

            frag_formula_idxs = th.arange(batch_frag_num_formulae, device=device)
            frag_formula_sizes = scatter_reduce(
                th.ones_like(frag_formula_batch_idxs),
                frag_formula_batch_idxs,
                reduce="sum",
                dim_size=batch_size,
            )
            assert not th.any(frag_formula_sizes <= 1), frag_formula_sizes

            frag_formula_cumsizes = th.cumsum(
                th.cat(
                    [
                        th.zeros([1], device=device, dtype=frag_formula_sizes.dtype),
                        frag_formula_sizes,
                    ],
                    dim=0,
                ),
                dim=0,
            )
            frag_node_offsets = frag_formula_cumsizes[frag_node_batch_idxs]

            # Update peak-related tensors
            non_null_mask = ~th.isin(frag_formula_idxs, frag_formula_cumsizes[:-1])
            frag_formula_peak_idxs = frag_formula_idxs[non_null_mask]
            frag_formula_peak_idxs = (
                frag_formula_peak_idxs
                - frag_formula_cumsizes[:-1][frag_formula_batch_idxs[non_null_mask]]
            )
            frag_formula_peak_probs = frag_formula_peak_probs[frag_formula_peak_mask]
            frag_formula_peak_mzs = frag_formula_peak_mzs[frag_formula_peak_mask]
            frag_formula_peak_sizes = frag_formula_sizes - 1
        else:
            frag_joint_formula_idxs = frag_joint_formula_idxs.flatten()
        # get isomorphism aggregation
        if self.nb_iso:
            frag_nb_idxs = get_node_feats(frag_x, frag_node_feat_idxs, "nb_iso_idx").flatten()
            frag_nb_offsets = scatter_reduce(
                frag_nb_idxs, frag_node_batch_idxs, reduce="amax", dim_size=batch_size
            )
            frag_nb_offsets = th.cat(
                [
                    th.zeros([1], dtype=frag_nb_offsets.dtype, device=device),
                    frag_nb_offsets + 1,
                ],
                dim=0,
            )
            frag_nb_offsets = th.cumsum(frag_nb_offsets, dim=0)
            batch_frag_nb_num_nodes = frag_nb_offsets[-1].item()
            frag_nb_offsets = th.gather(
                input=frag_nb_offsets[:-1], index=frag_node_batch_idxs, dim=0
            )
            frag_nb_idxs = frag_nb_idxs + frag_nb_offsets
            assert th.max(frag_nb_idxs) < batch_frag_nb_num_nodes, (
                th.max(frag_nb_idxs),
                batch_frag_nb_num_nodes,
            )
            frag_nb_un_idxs, frag_nb_inv_idxs = th.unique(frag_nb_idxs, return_inverse=True)

        # assemble all features for dag
        # concatenate everything
        frag_x_embed = th.cat(frag_ndata, dim=-1)
        # concatenate everything
        frag_edge_attr_embed = th.cat(frag_edata, dim=-1)

        # define frag network
        frag_embed_gnn = self.frag_embedder(
            frag_x_embed, frag_node_batch_idxs, frag_edge_index, frag_edge_attr_embed
        )
        frag_embed_node = self.frag_embedder.input_project(frag_x_embed)
        _need_gnn_pool = self.frag_pool_combine != "none" or self.predict_oos
        # For sum/mean, precompute CSR ptr and call frag_pool with index=None so PyG routes
        # through segment_csr instead of scatter. segment_csr requires index=None (or
        # _deterministic=True) to activate; passing both index and ptr keeps scatter active.
        # Benefits: contiguous memory reads, ptr is (B+1,) vs scatter index (N,); mean also
        # saves ~50% peak memory (no internal count buffer). frag_batch is sorted in PyG
        # batches so ptr is always valid.
        # NOTE: "max" is intentionally excluded — segment_csr for max allocates ~3× the peak
        # memory of scatter_max (temporary workspace for backward), which outweighs the 8–9×
        # speed gain when memory is the primary constraint.
        if self.frag_pool_type in ("sum", "mean") and _need_gnn_pool:
            _frag_counts = th.bincount(frag_batch, minlength=batch_size)
            frag_ptr = th.cat([_frag_counts.new_zeros(1), _frag_counts.cumsum(0)])
        else:
            frag_ptr = None
        # _pool_idx: pass None when using ptr (segment_csr path), frag_batch otherwise.
        _pool_idx = None if frag_ptr is not None else frag_batch
        frag_embed_gnn_pool = (
            self.frag_pool(frag_embed_gnn, _pool_idx, ptr=frag_ptr, dim_size=batch_size)
            if _need_gnn_pool
            else None
        )
        if self.frag_pool_combine == "subtract":
            frag_embed_node_pool = self.frag_pool(
                frag_embed_node, _pool_idx, ptr=frag_ptr, dim_size=batch_size
            )
            frag_embed_gnn = frag_embed_gnn - frag_embed_gnn_pool[frag_batch]
            frag_embed_node = frag_embed_node - frag_embed_node_pool[frag_batch]
        elif self.frag_pool_combine == "add":
            frag_embed_node_pool = self.frag_pool(
                frag_embed_node, _pool_idx, ptr=frag_ptr, dim_size=batch_size
            )
            frag_embed_gnn = frag_embed_gnn + frag_embed_gnn_pool[frag_batch]
            frag_embed_node = frag_embed_node + frag_embed_node_pool[frag_batch]
        else:
            assert self.frag_pool_combine == "none", self.frag_pool_combine

        # get frag dag embedding
        if self.frag_embed_combine == "cat":
            frag_embed_parts = [frag_embed_gnn, frag_embed_node]
        else:
            assert self.frag_embed_combine == "avg", self.frag_embed_combine
            frag_embed_parts = [0.5 * frag_embed_gnn + 0.5 * frag_embed_node]

        # Cache unique batch counts to avoid redundant computation (Memory Opt #1)
        if (
            self.ce_insert_location == "mlp"
            or self.prec_insert_location == "mlp"
            or self.inst_insert_location == "mlp"
            or self.frag_mode_insert_location == "mlp"
        ):
            # frag_num_nodes is a cumulative prefix sum → diff gives per-sample node counts,
            # equivalent to th.unique(frag_node_batch_idxs, return_counts=True)[1] but O(batch)
            unique_batch_counts = frag_num_nodes[1:] - frag_num_nodes[:-1]
            _mlp_extras = []
            if self.ce_insert_location == "mlp":
                _mlp_extras.append(ce_embed)
            if self.prec_insert_location == "mlp":
                _mlp_extras.append(prec_embed)
            if self.inst_insert_location == "mlp":
                _mlp_extras.append(inst_embed)
            if self.frag_mode_insert_location == "mlp":
                _mlp_extras.append(frag_mode_embed)
            frag_embed_parts.append(
                th.repeat_interleave(th.cat(_mlp_extras, dim=1), unique_batch_counts, dim=0)
            )

        frag_embed = th.cat(frag_embed_parts, dim=1)
        frag_joint_batch_idxs = th.repeat_interleave(frag_node_batch_idxs, self.mlp_output_dim)
        # Broadcast comparison via reshape — reuses frag_node_offsets (already equals
        # frag_formula_cumsizes[frag_node_batch_idxs], the null-formula index per node) and
        # avoids materializing a full (N×H) null_tiled tensor.
        frag_joint_mask = (
            (
                frag_joint_formula_idxs.reshape(batch_frag_num_nodes, self.mlp_output_dim)
                != frag_node_offsets.unsqueeze(1)
            )
            .float()
            .flatten()
        )

        if self.mlp_output_format == "formula":
            #  predicts joint logits for each (node, formula) pair:
            # log p(f,n)
            frag_joint_logits = self.formula_module(frag_embed)
            # apply mode-conditioned affine scaling if enabled
            if self.frag_mode_scaler is not None:
                assert spec_frag_mode is not None, "spec_frag_mode required for frag_mode_scaler"
                frag_joint_logits = self.frag_mode_scaler(
                    frag_joint_logits, spec_frag_mode, frag_node_batch_idxs
                )
            # apply CE-conditioned FiLM scaling if enabled
            if self.ce_scaler is not None:
                frag_joint_logits = self.ce_scaler(
                    frag_joint_logits,
                    frag_node_batch_idxs,
                    nce_stats=kwargs.get("spec_nce_stats"),
                    ace_stats=kwargs.get("spec_ace_stats"),
                )
            # if include_cmf:
            #   frag_joint_logits[:, 2 * self.num_hs + 1:] = LOG_ZERO(frag_joint_logits.dtype)
            assert frag_joint_logits.shape[1] == self.mlp_output_dim, (
                frag_joint_logits.shape[1],
                2 * self.num_hs + 1,
            )
            # print("> frag_joint_logits shape:", frag_joint_logits.shape)
            frag_joint_logits = frag_joint_logits.flatten()

            # compute all joint log probabilits
            frag_joint_logprobs = scatter_logsoftmax(frag_joint_logits, frag_joint_batch_idxs)
            # compute total NULL probability (before renormalization)
            frag_null_formula_logprob = scatter_logsumexp(
                (1.0 - frag_joint_mask) * frag_joint_logprobs
                + frag_joint_mask * LOG_ZERO(frag_joint_logprobs.dtype),
                frag_joint_batch_idxs,
            )
            frag_null_formula_logprob = th.clamp(frag_null_formula_logprob, max=0.0)

            if self.mask_null_formula:
                # compute non-NULL renormalized intensity
                frag_joint_logprobs = scatter_masked_softmax(
                    frag_joint_logits, frag_joint_mask, frag_joint_batch_idxs, log=True
                )

            # reshape
            # Each row corresponds to a node, and each column to a possible formula for that node.
            frag_joint_logprobs = frag_joint_logprobs.reshape(-1, self.mlp_output_dim)

            # aggregate formula logits by node, then normalize, this will give us probilities of each node
            # log p(n) = logsumexp_f log p(f,n)
            frag_node_logits = th.logsumexp(frag_joint_logprobs, dim=1)
            frag_node_logprobs = scatter_logsoftmax(frag_node_logits, frag_node_batch_idxs)

            # calculate conditional probability
            # log p(f|n) = log p(f,n) - log p(n)
            frag_node_formula_logprobs = frag_joint_logprobs - frag_node_logprobs.unsqueeze(-1)
            frag_node_formula_logprobs = th.log_softmax(frag_node_formula_logprobs, dim=1)

        elif self.mlp_output_format == "node_formula":
            # predicts log p(f|n) (formula given node) and log p(n) (node probability) separately.
            # log p(f|n) formula given node
            frag_node_formula_logits = self.formula_module(frag_embed)
            # apply mode-conditioned affine scaling if enabled
            if self.frag_mode_scaler is not None:
                assert spec_frag_mode is not None, "spec_frag_mode required for frag_mode_scaler"
                frag_node_formula_logits = self.frag_mode_scaler(
                    frag_node_formula_logits, spec_frag_mode, frag_node_batch_idxs
                )
            assert frag_node_formula_logits.shape[1] == self.mlp_output_dim, (
                frag_node_formula_logits.shape[1],
                self.mlp_output_dim,
            )
            frag_node_formula_logprobs = th.log_softmax(frag_node_formula_logits, dim=1)

            # log p(n) node probability
            frag_node_logits = self.node_module(frag_embed).squeeze(1)
            frag_node_logprobs = scatter_logsoftmax(frag_node_logits, frag_node_batch_idxs)

            # joint logits = formula_logits + node_logits (broadcast); exposed for post-branch
            # scatter_logsoftmax over real positions (mirrors formula path's frag_joint_logits)
            frag_joint_logits = (
                frag_node_formula_logits + frag_node_logits.unsqueeze(-1)
            ).flatten()

            # log p(f,n) = log p(f|n) + log p(n)
            frag_joint_logprobs = frag_node_formula_logprobs + frag_node_logprobs.unsqueeze(-1)
            frag_joint_logprobs = frag_joint_logprobs.flatten()

            # compute total NULL probability (before renormalization)
            frag_null_formula_logprob = scatter_logsumexp(
                (1.0 - frag_joint_mask) * frag_joint_logprobs
                + frag_joint_mask * LOG_ZERO(frag_joint_logprobs.dtype),
                frag_joint_batch_idxs,
            )
            frag_null_formula_logprob = th.clamp(frag_null_formula_logprob, max=0.0)

            if self.mask_null_formula:
                # compute non-NULL renormalized intensity
                frag_joint_logprobs = scatter_masked_softmax(
                    frag_joint_logprobs,
                    frag_joint_mask,
                    frag_joint_batch_idxs,
                    log=True,
                )
        else:
            raise ValueError(f"Unknown mlp_output_format: {self.mlp_output_format}")

        # aggregate by formula
        # log p(f) = logsumexp_n log p(f,n)
        frag_formula_mask = th.ones_like(frag_formula_batch_idxs, dtype=float_dtype)
        frag_formula_mask[frag_formula_cumsizes[:-1]] = 0.0
        # aggregate by formula — use explicit dim_size so dedup'd batches where some high-index
        # formula slots are unreferenced still produce a tensor of size batch_frag_num_formulae,
        # matching frag_formula_mask / frag_formula_batch_idxs.
        frag_formula_logprobs = scatter_logsumexp(
            frag_joint_logprobs.flatten(),
            frag_joint_formula_idxs,
            dim_size=int(batch_frag_num_formulae),
        )
        # softmax over formulae
        if self.mask_null_formula:
            frag_formula_logprobs = scatter_masked_softmax(
                frag_formula_logprobs,
                frag_formula_mask,
                frag_formula_batch_idxs,
                log=True,
            )
        else:
            frag_formula_logprobs = scatter_masked_softmax(
                frag_formula_logprobs,
                th.ones_like(frag_formula_logprobs),
                frag_formula_batch_idxs,
                log=True,
            )

        # OOS Stuff
        if self.predict_oos:
            # get OOS logits and logprobs
            oos_logits = self.oos_module(th.cat([mol_embed_gnn_pool, frag_embed_gnn_pool], dim=1))
            oos_logits = oos_logits.flatten()
            oos_logprobs = F.logsigmoid(oos_logits)
            not_oos_logprobs = F.logsigmoid(-oos_logits)
        else:
            oos_logprobs = th.full(
                [batch_size],
                LOG_ZERO(frag_formula_logprobs.dtype),
                device=device,
                dtype=frag_formula_logprobs.dtype,
            )
            not_oos_logprobs = th.zeros(
                [batch_size], device=device, dtype=frag_formula_logprobs.dtype
            )

        # adjust frag_formula_logprobs
        frag_formula_oos_logprobs = frag_formula_logprobs + th.repeat_interleave(
            not_oos_logprobs, frag_formula_sizes, dim=0
        )

        # convert to spectrum
        frag_formula_offsets = th.repeat_interleave(
            frag_formula_cumsizes[:-1],
            frag_formula_sizes - 1,  # -1 is for NULL formulae
        )
        spec_mzs = frag_formula_peak_mzs
        spec_logprobs = frag_formula_oos_logprobs[
            frag_formula_peak_idxs + frag_formula_offsets
        ] + th.log(frag_formula_peak_probs)

        # get batch idxs
        spec_batch_idxs = th.repeat_interleave(
            th.arange(frag_formula_peak_sizes.shape[0], device=device),
            frag_formula_peak_sizes,
        )

        if not self.skip_edge_loss:
            frag_edge_batch_idxs = frag_batch[frag_edge_index[0]]
            frag_edge_logits = (
                frag_node_logprobs[frag_edge_index[0]] + frag_node_logprobs[frag_edge_index[1]]
            )
            frag_edge_logprobs = scatter_logsoftmax(frag_edge_logits, frag_edge_batch_idxs)
            # print(scatter_logsumexp(frag_edge_logprobs,frag_edge_batch_idxs))
            frag_node_h_counts = get_node_feats(frag_x, frag_node_feat_idxs, "h_counts")
            frag_edge_h_ranges = get_edge_feats(frag_edge_attr, frag_edge_feat_idxs, "h_range")
            # Memory Opt #2: Use einsum to compute h_diffs without Cartesian product expansion
            src_h_counts = frag_node_h_counts[frag_edge_index[0]]  # (num_edges,)
            dst_h_counts = frag_node_h_counts[frag_edge_index[1]]  # (num_edges,)
            frag_edge_h_diffs = src_h_counts.unsqueeze(1) - dst_h_counts.unsqueeze(1)
            frag_edge_h_diffs = frag_edge_h_diffs.reshape(frag_edge_h_diffs.shape[0], -1)
            frag_edge_h_range_masks = th.logical_or(
                frag_edge_h_diffs < frag_edge_h_ranges[:, 0].unsqueeze(-1),
                frag_edge_h_diffs > frag_edge_h_ranges[:, 1].unsqueeze(-1),
            )
            src_formula_logprobs = frag_node_formula_logprobs[
                frag_edge_index[0]
            ]  # (num_edges, mlp_output_dim)
            dst_formula_logprobs = frag_node_formula_logprobs[
                frag_edge_index[1]
            ]  # (num_edges, mlp_output_dim)
            # Outer product of per-node formula log-probs across each edge: (E, H*H)
            frag_edge_h_logprobs = th.einsum(
                "ni,nj->nij", src_formula_logprobs, dst_formula_logprobs
            ).reshape(len(src_formula_logprobs), -1)

        else:
            frag_edge_logprobs = None
            frag_edge_h_diffs = None
            frag_edge_h_range_masks = None
            frag_edge_h_logprobs = None
            frag_edge_batch_idxs = None

        # select (will remove all NULL formula idxs, potentially some node idxs too if they contain only NULLs)
        _real_mask = frag_joint_mask.bool()
        # Compute real positions once; derive node/h-slot indices via integer div/mod to avoid
        # materializing three separate (N×H) long tensors (node_idxs, h_counts, h_offsets).
        real_pos = th.nonzero(_real_mask, as_tuple=False).squeeze(1)  # (J,)
        H = self.mlp_output_dim
        frag_real_joint_logits = frag_joint_logits[_real_mask]
        frag_real_joint_node_idxs = real_pos // H
        _real_h_pos = real_pos % H
        frag_real_joint_h_counts = self._h_counts_buf[_real_h_pos]
        frag_real_joint_h_offsets = th.zeros_like(frag_real_joint_h_counts)
        frag_real_joint_formula_idxs = frag_joint_formula_idxs[_real_mask]
        frag_real_joint_batch_idxs = frag_node_batch_idxs[frag_real_joint_node_idxs]
        # P(f,n)
        frag_real_joint_logprobs = scatter_logsoftmax(
            frag_real_joint_logits, frag_real_joint_batch_idxs
        )
        # P(n) - sum, renormalize, but keep all-NULL nodes (as zeros)
        _need_expl_idxs = return_explainability_tensors or self.debug_validate_outputs
        frag_real_node_node_idxs = (
            th.arange(batch_frag_num_nodes, device=device) if _need_expl_idxs else None
        )
        frag_real_node_logprobs = scatter_logsumexp(
            frag_real_joint_logprobs,
            frag_real_joint_node_idxs,
            dim_size=batch_frag_num_nodes,
        )
        frag_real_node_logprobs = scatter_logsoftmax(frag_real_node_logprobs, frag_node_batch_idxs)
        # P(f) - sum, renormalize, but keep NULL formulae (as zeros)
        frag_real_formula_logprobs = scatter_logsumexp(
            frag_real_joint_logprobs,
            frag_real_joint_formula_idxs,
            dim_size=batch_frag_num_formulae,
        )
        frag_real_formula_logprobs = scatter_logsoftmax(
            frag_real_formula_logprobs,
            frag_formula_batch_idxs,
        )
        frag_real_formula_formula_idxs = (
            th.arange(batch_frag_num_formulae, device=device) if _need_expl_idxs else None
        )
        # P(f|n) - remove all NULLs from conditionals
        frag_real_node_formula_logprobs = (
            frag_real_joint_logprobs - frag_real_node_logprobs[frag_real_joint_node_idxs]
        )
        frag_real_node_formula_logprobs = scatter_logsoftmax(
            frag_real_node_formula_logprobs, frag_real_joint_node_idxs
        )
        # P(n|f) - remove all NULLs from conditionals
        frag_real_formula_node_logprobs = (
            frag_real_joint_logprobs - frag_real_formula_logprobs[frag_real_joint_formula_idxs]
        )
        frag_real_formula_node_logprobs = scatter_logsoftmax(
            frag_real_formula_node_logprobs, frag_real_joint_formula_idxs
        )
        # hydrogens
        frag_real_joint_h_idxs = (
            frag_real_joint_h_counts + self.num_hs + frag_real_joint_h_offsets
        ) + frag_real_joint_batch_idxs * self.mlp_output_dim

        frag_real_h_logprobs = scatter_logsumexp(
            frag_real_joint_logprobs,
            frag_real_joint_h_idxs,
            dim_size=batch_size * self.mlp_output_dim,
        )

        frag_real_h_logprobs = th.clamp(frag_real_h_logprobs, max=0.0)
        frag_real_h_counts = self._h_counts_range_buf.repeat(batch_size)
        frag_real_h_batch_idxs = th.repeat_interleave(
            th.arange(batch_size, device=device), self.mlp_output_dim
        )

        # calculate isomorphic distributions
        if self.nb_iso:
            # P(n')
            frag_nb_node_logprobs = scatter_logsumexp(
                frag_real_node_logprobs, frag_nb_idxs, dim_size=batch_frag_nb_num_nodes
            )
            frag_nb_node_batch_idxs = scatter_reduce(
                frag_node_batch_idxs,
                frag_nb_idxs,
                reduce="amax",
                dim_size=batch_frag_nb_num_nodes,
            )
            frag_nb_node_node_idxs = frag_nb_un_idxs
            # P(n'|f)
            frag_nb_joint_idxs = frag_nb_idxs[frag_real_joint_node_idxs]
            frag_nb_joint_both_idxs = th.stack(
                [frag_nb_joint_idxs, frag_real_joint_formula_idxs], dim=1
            )
            frag_nb_joint_both_un_idxs, frag_nb_joint_both_inv_idxs = th.unique(
                frag_nb_joint_both_idxs, return_inverse=True, dim=0
            )
            frag_nb_joint_node_idxs = frag_nb_joint_both_un_idxs[:, 0]
            frag_nb_joint_formula_idxs = frag_nb_joint_both_un_idxs[:, 1]
            frag_nb_joint_batch_idxs = frag_nb_node_batch_idxs[frag_nb_joint_node_idxs]
            frag_nb_formula_node_logprobs = scatter_logsumexp(
                frag_real_formula_node_logprobs,
                frag_nb_joint_both_inv_idxs,
                dim_size=frag_nb_joint_both_un_idxs.shape[0],
            )
            frag_nb_formula_node_logprobs = th.clamp(frag_nb_formula_node_logprobs, max=0.0)
            # P(n|n')
            frag_nb_node_node_logprobs = scatter_logsoftmax(frag_real_node_logprobs, frag_nb_idxs)
            frag_nb_node_node_node_idxs = frag_nb_idxs  # frag_nb_node_node_idxs[frag_nb_inv_idxs]
            frag_nb_node_node_batch_idxs = frag_nb_node_batch_idxs[frag_nb_inv_idxs]
            assert th.all(frag_nb_node_node_logprobs <= 0.0)
            # P(f|n')
            frag_nb_node_formula_logprobs = (
                frag_real_node_formula_logprobs
                + frag_nb_node_node_logprobs[frag_real_joint_node_idxs]
            )
            frag_nb_node_formula_logprobs = scatter_logsumexp(
                frag_nb_node_formula_logprobs,
                frag_nb_joint_both_inv_idxs,
                dim_size=frag_nb_joint_both_un_idxs.shape[0],
            )
            frag_nb_node_formula_logprobs = th.clamp(frag_nb_node_formula_logprobs, max=0.0)
            # P(f,n')
            frag_nb_joint_logprobs = scatter_logsumexp(
                frag_real_joint_logprobs,
                frag_nb_joint_both_inv_idxs,
                dim_size=frag_nb_joint_both_un_idxs.shape[0],
            )
        else:
            frag_nb_node_logprobs = None
            frag_nb_node_formula_logprobs = None
            frag_nb_formula_node_logprobs = None
            frag_nb_node_node_logprobs = None
            frag_nb_node_node_idxs = None
            frag_nb_node_batch_idxs = None
            frag_nb_joint_node_idxs = None
            frag_nb_joint_formula_idxs = None
            frag_nb_joint_batch_idxs = None
            frag_nb_joint_logprobs = None
            frag_nb_node_node_node_idxs = None
            frag_nb_node_node_batch_idxs = None

        if self.debug_validate_outputs:
            assert (
                th.unique(frag_real_node_node_idxs).shape[0]
                == frag_real_node_node_idxs.max() + 1
                == batch_frag_num_nodes
            )
            assert (
                th.unique(frag_real_formula_formula_idxs).shape[0]
                == frag_real_formula_formula_idxs.max() + 1
                == batch_frag_num_formulae
            )
            if self.nb_iso:
                assert (
                    th.unique(frag_nb_node_node_idxs).shape[0]
                    == frag_nb_node_node_idxs.max() + 1
                    == batch_frag_nb_num_nodes
                )

        if self.bin_output:
            # import pdb; pdb.set_trace()
            spec_bin_mzs, spec_bin_logprobs, spec_bin_batch_idxs = batched_bin_func(
                mzs=spec_mzs,
                ints=spec_logprobs,
                batch_idxs=spec_batch_idxs,
                mz_max=self.mz_max,
                mz_bin_res=self.mz_bin_res,
                agg="lse",
                sparse=True,
                return_mzs=True,
            )
            spec_mzs = spec_bin_mzs
            spec_logprobs = spec_bin_logprobs
            spec_batch_idxs = spec_bin_batch_idxs

        if self.debug_validate_outputs:
            term1 = scatter_logsumexp(spec_logprobs, spec_batch_idxs).exp()
            term2 = oos_logprobs.exp()
            term3 = (not_oos_logprobs + frag_null_formula_logprob).exp()
            sum_probs = term1 + term2 + term3
            assert th.all(th.isclose(sum_probs, th.ones_like(sum_probs), rtol=0.0, atol=5e-2)), (
                term1,
                term2,
                term3,
            )
        # size definitions:
        # B: batch size
        # P: total number of predicted spectrum peaks across the batch
        # F: total number of predicted formulas across the batch
        # N: total number of predicted nodes (fragments) across the batch
        # J: total number of predicted joint distribution entries across the batch
        # E: total number of edges across the batch
        # H: number of hydrogen counts predicted (num_hs*2+1)
        # N': total number of predicted neighborhood (isomorphic) nodes across the batch
        # J': total number of predicted joint distribution entries for isomorphic nodes across the batch
        out_d = {
            "pred_mzs": spec_mzs,  # Predicted m/z values for the spectrum peaks [P]
            "pred_logprobs": spec_logprobs,  # Predicted log probabilities (intensities) for the spectrum peaks [P]
            "pred_batch_idxs": spec_batch_idxs,  # Batch indices mapping each peak to a spectrum in the batch [P]
            "pred_formula_logprobs": frag_real_formula_logprobs,  # Log probabilities of formulas P(f) [F]
            "pred_formula_batch_idxs": frag_formula_batch_idxs,  # Batch indices for the formulas [F]
            "pred_node_logprobs": frag_real_node_logprobs,  # Log probabilities of nodes (fragments) P(n) [N]
            "pred_node_batch_idxs": frag_node_batch_idxs,  # Batch indices for the nodes [N]
            "pred_node_formula_logprobs": frag_real_node_formula_logprobs,  # Conditional log probabilities P(f|n) [J]
            "pred_formula_node_logprobs": frag_real_formula_node_logprobs,  # Conditional log probabilities P(n|f) [J]
            "pred_joint_logprobs": frag_real_joint_logprobs,  # Joint log probabilities P(f,n) [J]
            "pred_joint_node_idxs": frag_real_joint_node_idxs,  # Node indices for the joint distribution entries [J]
            "pred_joint_formula_idxs": frag_real_joint_formula_idxs,  # Formula indices for the joint distribution entries [J]
            "pred_joint_batch_idxs": frag_real_joint_batch_idxs,  # Batch indices for the joint distribution entries [J]
            "pred_null_formula_logprob": frag_null_formula_logprob,  # Log probability of the NULL formula (unexplained intensity) [B]
            "pred_edge_logprobs": frag_edge_logprobs,  # Log probabilities for edges (if edge loss is used) [E]
            "pred_edge_h_diffs": frag_edge_h_diffs,  # Hydrogen count differences across edges [E, H*H]
            "pred_edge_h_range_masks": frag_edge_h_range_masks,  # Masks for valid hydrogen ranges on edges [E, H*H]
            "pred_edge_h_logprobs": frag_edge_h_logprobs,  # Log probabilities for hydrogen transfers on edges [E, H*H]
            "pred_edge_batch_idxs": frag_edge_batch_idxs,  # Batch indices for edges [E]
            "pred_oos_logprobs": oos_logprobs,  # Log probability of Out-Of-Scope (OOS) intensity [B]
            "pred_h_counts": frag_real_h_counts,  # Hydrogen counts for the hydrogen distribution [B*H]
            "pred_h_batch_idxs": frag_real_h_batch_idxs,  # Batch indices for the hydrogen distribution [B*H]
            "pred_h_logprobs": frag_real_h_logprobs,  # Log probabilities of hydrogen counts [B*H]
            "pred_nb_node_logprobs": frag_nb_node_logprobs,  # (Isomorphism) Log probabilities of neighborhood nodes P(n') [N']
            "pred_nb_node_formula_logprobs": frag_nb_node_formula_logprobs,  # (Isomorphism) Conditional log probabilities P(f|n') [J']
            "pred_nb_formula_node_logprobs": frag_nb_formula_node_logprobs,  # (Isomorphism) Conditional log probabilities P(n'|f) [J']
            "pred_nb_node_node_logprobs": frag_nb_node_node_logprobs,  # (Isomorphism) Conditional log probabilities P(n|n') [N]
            "pred_nb_node_node_idxs": frag_nb_node_node_idxs,  # (Isomorphism) Indices for node-node mapping [N]
            "pred_nb_node_batch_idxs": frag_nb_node_batch_idxs,  # (Isomorphism) Batch indices for neighborhood nodes [N']
            "pred_nb_joint_logprobs": frag_nb_joint_logprobs,  # (Isomorphism) Joint log probabilities P(f,n') [J']
            "pred_nb_joint_node_idxs": frag_nb_joint_node_idxs,  # (Isomorphism) Node indices for joint distribution [J']
            "pred_nb_joint_formula_idxs": frag_nb_joint_formula_idxs,  # (Isomorphism) Formula indices for joint distribution [J']
            "pred_nb_joint_batch_idxs": frag_nb_joint_batch_idxs,  # (Isomorphism) Batch indices for joint distribution [J']
            "pred_nb_node_node_node_idxs": frag_nb_node_node_node_idxs,  # (Isomorphism) Node indices for node-node mapping [N]
            "pred_nb_node_node_batch_idxs": frag_nb_node_node_batch_idxs,  # (Isomorphism) Batch indices for node-node mapping [N]
        }
        if return_explainability_tensors:
            out_d["pred_formula_formula_idxs"] = frag_real_formula_formula_idxs  # [F]
            out_d["pred_node_node_idxs"] = frag_real_node_node_idxs  # [N]
            out_d["pred_joint_h_counts"] = frag_real_joint_h_counts  # [J]
            out_d["pred_joint_h_idxs"] = frag_real_joint_h_idxs  # [J]

        if self.output_formula_str:
            assert frag_formula_str is not None, (
                "frag_formula_strs must be provided if output_formula_str=True"
            )
            assert spec_prec_type_str is not None, (
                "spec_prec_type_strs must be provided if output_formula_str=True"
            )
            assert frag_formula_str.shape[0] == frag_real_formula_logprobs.shape[0]
            prec_type_str = spec_prec_type_str
            prec_type_delta_comp = np.array(
                [PREC_TYPE_TO_FORMULA_DIFF[prec_type_str[i]] for i in range(len(prec_type_str))]
            )
            prec_type_delta_comp = prec_type_delta_comp[frag_formula_batch_idxs.cpu().numpy()]
            pred_formula_str = [
                combine_formulae(frag_formula_str[i], prec_type_delta_comp[i])
                if frag_formula_str[i] != ""
                else ""
                for i in range(len(frag_formula_str))
            ]
            pred_formula_str = np.array(pred_formula_str)
            out_d["pred_formula_str"] = pred_formula_str

        if self.debug_validate_outputs:
            for k, v in out_d.items():
                if "logprob" in k and v is not None and v.max() > 0.0:
                    logger.warning(f"{k} has value {v.max()} > 0")
                if "batch_idx" in k and v is not None:
                    if v.numel() == 0:
                        raise ValueError(f"Empty batch index: {k}")
                    elif th.unique(v).shape[0] != batch_size:
                        raise ValueError(
                            f"Missing items in {k} of batch: {th.unique(v).shape[0]} expeacting {batch_size}"
                        )
        return out_d
