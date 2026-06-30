"""
Graph Neural Network model for spectrum prediction.

This class implements a graph neural network (GNN) model for MS/MS spectrum prediction.
It supports flexible node/edge features, positional encoding, and modular metadata embedding
for collision energy (CE), precursor type, and instrument type. The model uses a GNN for
molecular graph encoding, followed by pooling and a feedforward network for spectrum prediction.

"""

import torch as th
import torch.nn as nn
import torch_geometric as pyg

from fragnnet.model.base_model import CEModel, InstModel, PrecModel
from fragnnet.model.nn_blocks import GNN, SpecFFN, build_pool_module
from fragnnet.utils.feat_utils import get_mol_feats_sizes


class GNNModel(nn.Module, CEModel, PrecModel, InstModel):
    """
    Graph Neural Network model for spectrum prediction.

    - Supports configurable node/edge features and positional encoding.
    - Modular metadata embedding for CE, precursor, and instrument type.
    - Uses a GNN for molecular graph encoding, pooling, and a feedforward network for prediction.
    """

    def __init__(
        self,
        mol_node_feats: list[str],
        mol_edge_feats: list[str],
        mol_pe_embed_k: int,
        mol_hidden_size: int,
        mol_num_layers: int,
        mol_gnn_type: str,
        mol_normalization: str,
        mol_dropout: float,
        mol_pool_type: str,
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
    ):
        super().__init__()
        # collision energy embedding
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
        # precursor embedding
        self._prec_init(
            prec_insert_location=prec_insert_location,
            prec_insert_size=prec_insert_size,
            prec_num_types=len(prec_types),
        )
        # instrument embedding
        self._inst_init(
            inst_insert_location=inst_insert_location,
            inst_insert_size=inst_insert_size,
            inst_num_types=len(inst_types),
        )

        # calculate node/edge feats sizes
        self.mol_node_feats = mol_node_feats
        self.mol_edge_feats = mol_edge_feats
        self.mol_pe_embed_k = mol_pe_embed_k
        self._compute_mol_feats_sizes()

        # setup mol gnn
        self.mol_node_feats_size += (
            self.ce_mol_input_dim + self.prec_mol_input_dim + self.inst_mol_input_dim
        )
        mol_kwargs = {
            "node_feats_size": self.mol_node_feats_size,
            "edge_feats_size": self.mol_edge_feats_size,
            "hidden_size": mol_hidden_size,
            "num_layers": mol_num_layers,
            "gnn_type": mol_gnn_type,
            "dropout": mol_dropout,
            "normalization": mol_normalization,
        }
        # Mol GNN
        self.mol_embedder = GNN(**mol_kwargs)
        self.mol_pool_type = mol_pool_type
        self.mol_pool = build_pool_module(mol_pool_type, mol_hidden_size)

        # MLP input = GNN output + metadata
        self.mlp_input_dim = mol_hidden_size
        self.mlp_input_dim += (
            self.ce_mlp_input_dim + self.prec_mlp_input_dim + self.inst_mlp_input_dim
        )

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
        )

    def forward(
        self,
        mol_pyg: pyg.data.Data,
        spec_prec_mz: th.Tensor,
        spec_ce: th.Tensor | None = None,
        spec_ce_batch_idxs: th.Tensor | None = None,
        spec_prec_type: th.Tensor | None = None,
        spec_inst_type: th.Tensor | None = None,
        **kwargs,
    ):
        """
        Forward pass for GNNModel.

        Args:
            mol_pyg: PyG Data object with molecular graph (x, edge_index, edge_attr, batch).
            spec_prec_mz: Precursor m/z tensor.
            spec_ce: Collision energy tensor (optional).
            spec_ce_batch_idxs: Batch indices for CE (optional).
            spec_prec_type: Precursor type tensor (optional).
            spec_inst_type: Instrument type tensor (optional).

        Returns:
            Dictionary with predicted m/z, log-probabilities, batch indices, and spectra.
        """
        # Extract molecular graph features
        # mol_x: node feature matrix
        # mol_edge_index: graph connectivity (COO format)
        # mol_edge_attr: edge feature matrix
        # mol_batch: sample index for each node
        mol_x, mol_edge_index, mol_edge_attr, mol_batch = (
            mol_pyg.x,
            mol_pyg.edge_index,
            mol_pyg.edge_attr,
            mol_pyg.batch,
        )

        batch_size = mol_batch[-1] + 1

        # Metadata embedders
        ce = spec_ce
        ce_batch_idxs = spec_ce_batch_idxs
        ce_embed = self.embed_ce(ce, ce_batch_idxs, batch_size)
        prec_embed = self.embed_prec(spec_prec_type)
        inst_embed = self.embed_inst(spec_inst_type)

        # Metadata embeddings at the node feature level
        if self.ce_insert_location == "mol":
            mol_ce_embed = th.repeat_interleave(
                ce_embed, th.unique(mol_batch, return_counts=True)[1], dim=0
            )
            mol_x = th.cat([mol_x, mol_ce_embed], dim=1)
        if self.prec_insert_location == "mol":
            mol_prec_embed = th.repeat_interleave(
                prec_embed, th.unique(mol_batch, return_counts=True)[1], dim=0
            )
            mol_x = th.cat([mol_x, mol_prec_embed], dim=1)
        if self.inst_insert_location == "mol":
            mol_inst_embed = th.repeat_interleave(
                inst_embed, th.unique(mol_batch, return_counts=True)[1], dim=0
            )
            mol_x = th.cat([mol_x, mol_inst_embed], dim=1)

        # GNN encoding
        mol_embed_gnn = self.mol_embedder(mol_x, mol_batch, mol_edge_index, mol_edge_attr)
        # Pooling to get graph-level embedding
        mol_embed_gnn_pool = self.mol_pool(mol_embed_gnn, mol_batch)
        ffn_input = mol_embed_gnn_pool

        # Metadata embeddings at the MLP level
        if self.ce_insert_location == "mlp":
            ffn_input = th.cat([ffn_input, ce_embed], dim=1)
        if self.prec_insert_location == "mlp":
            ffn_input = th.cat([ffn_input, prec_embed], dim=1)
        if self.inst_insert_location == "mlp":
            ffn_input = th.cat([ffn_input, inst_embed], dim=1)

        # Spectrum prediction
        pred_mzs, pred_logprobs, pred_batch_idxs, pred_specs = self.ffn(ffn_input, spec_prec_mz)
        out_d = {
            "pred_mzs": pred_mzs,
            "pred_logprobs": pred_logprobs,
            "pred_batch_idxs": pred_batch_idxs,
            "pred_specs": pred_specs,
        }
        return out_d

    def _compute_mol_feats_sizes(self):
        """
        Compute molecular node and edge feature sizes.
        """
        self.mol_node_feats_size, self.mol_edge_feats_size = get_mol_feats_sizes(
            self.mol_node_feats, self.mol_edge_feats, self.mol_pe_embed_k
        )

    def _ce_location_check(self):
        assert self.ce_insert_location in ["mlp", "mol", "none"], (
            f"ce_insert_location={self.ce_insert_location} not supported"
        )

    def _prec_location_check(self):
        assert self.prec_insert_location in ["mlp", "mol", "none"], (
            f"prec_insert_location={self.prec_insert_location} not supported"
        )

    def _inst_location_check(self):
        assert self.inst_insert_location in ["mlp", "mol", "none"], (
            f"prec_insert_location={self.inst_insert_location} not supported"
        )
