"""
Molecular Graph Encoder for embedding molecules.
"""

import torch as th
import torch.nn as nn
import torch_geometric as pyg

from fragnnet.model.base_model import CEModel, FragModeModel, InstModel, PrecModel
from fragnnet.model.nn_blocks import GNN, MLPBlocks, build_pool_module
from fragnnet.utils.feat_utils import get_mol_feats_sizes


class MolEncoder(nn.Module, CEModel, PrecModel, InstModel, FragModeModel):
    """
    Graph Neural Network model for molecular graph embedding.
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
        output_dim: int,
        mlp_hidden_size: int,
        mlp_dropout: float,
        mlp_num_layers: int,
        mlp_use_residuals: bool,
        int_embedder: str,
        ce_insert_type: str,
        ce_insert_location: str,
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
        mol_num_heads: int,
        use_nce: bool,
        use_ace: bool,
        ace_max: float,
        ace_mean: float,
        ace_std: float,
    ):
        super().__init__()

        # Save params
        self.mol_node_feats = mol_node_feats
        self.mol_edge_feats = mol_edge_feats
        self.mol_pe_embed_k = mol_pe_embed_k
        self.mol_hidden_size = mol_hidden_size
        self.mol_num_layers = mol_num_layers
        self.mol_gnn_type = mol_gnn_type
        self.mol_normalization = mol_normalization
        self.mol_dropout = mol_dropout
        self.mol_pool_type = mol_pool_type
        self.output_dim = output_dim
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_dropout = mlp_dropout
        self.mlp_num_layers = mlp_num_layers
        self.mlp_use_residuals = mlp_use_residuals

        # CE params
        self.int_embedder = int_embedder
        self.ce_insert_type = ce_insert_type
        self.ce_insert_location = ce_insert_location
        self.ce_insert_merge = ce_insert_merge
        self.ce_insert_size = ce_insert_size
        self.nce_max = nce_max
        self.nce_mean = nce_mean
        self.nce_std = nce_std
        self.use_nce = use_nce
        self.use_ace = use_ace
        self.ace_max = ace_max
        self.ace_mean = ace_mean
        self.ace_std = ace_std

        # Prec params
        self.prec_insert_location = prec_insert_location
        self.prec_insert_size = prec_insert_size
        self.prec_types = prec_types
        self.prec_num_types = len(prec_types)

        # Inst params
        self.inst_insert_location = inst_insert_location
        self.inst_insert_size = inst_insert_size
        self.inst_types = inst_types
        self.inst_num_types = len(inst_types)

        # Frag-mode params
        self.frag_mode_insert_location = frag_mode_insert_location
        self.frag_mode_insert_size = frag_mode_insert_size
        self.frag_modes = frag_modes
        self.frag_mode_num_types = len(frag_modes)

        # Initialize components
        self._compute_mol_feats_sizes()
        self._ce_location_check()
        self._prec_location_check()
        self._inst_location_check()
        self._frag_mode_location_check()

        self._ce_init(
            int_embedder=self.int_embedder,
            ce_insert_location=self.ce_insert_location,
            ce_insert_type=self.ce_insert_type,
            ce_insert_merge=self.ce_insert_merge,
            ce_insert_size=self.ce_insert_size,
            nce_mean=self.nce_mean,
            nce_std=self.nce_std,
            nce_max=self.nce_max,
            use_nce=self.use_nce,
            use_ace=self.use_ace,
            ace_max=self.ace_max,
            ace_mean=self.ace_mean,
            ace_std=self.ace_std,
        )
        self._prec_init(
            prec_insert_location=self.prec_insert_location,
            prec_insert_size=self.prec_insert_size,
            prec_num_types=self.prec_num_types,
        )
        self._inst_init(
            inst_insert_location=self.inst_insert_location,
            inst_insert_size=self.inst_insert_size,
            inst_num_types=self.inst_num_types,
        )
        self._frag_mode_init(
            frag_mode_insert_location=self.frag_mode_insert_location,
            frag_mode_insert_size=self.frag_mode_insert_size,
            frag_mode_num_types=self.frag_mode_num_types,
        )

        # GNN
        # Adjust input size if metadata is inserted at 'mol' level
        mol_input_size = self.mol_node_feats_size
        if self.ce_insert_location == "mol":
            mol_input_size += self.ce_mol_input_dim
        if self.prec_insert_location == "mol":
            mol_input_size += self.prec_insert_size
        if self.inst_insert_location == "mol":
            mol_input_size += self.inst_insert_size
        if self.frag_mode_insert_location == "mol":
            mol_input_size += self.frag_mode_mol_input_dim

        self.mol_embedder = GNN(
            node_feats_size=mol_input_size,
            edge_feats_size=self.mol_edge_feats_size,
            hidden_size=self.mol_hidden_size,
            num_layers=self.mol_num_layers,
            gnn_type=self.mol_gnn_type,
            normalization=self.mol_normalization,
            dropout=self.mol_dropout,
            num_heads=mol_num_heads,
        )

        # Pooling
        self.mol_pool = build_pool_module(self.mol_pool_type, self.mol_hidden_size)

        # MLP Projection Head
        # Adjust input size if metadata is inserted at 'mlp' level
        mlp_input_size = self.mol_hidden_size
        if self.ce_insert_location == "mlp":
            mlp_input_size += self.ce_mlp_input_dim
        if self.prec_insert_location == "mlp":
            mlp_input_size += self.prec_insert_size
        if self.inst_insert_location == "mlp":
            mlp_input_size += self.inst_insert_size
        if self.frag_mode_insert_location == "mlp":
            mlp_input_size += self.frag_mode_mlp_input_dim

        self.projector = MLPBlocks(
            input_size=mlp_input_size,
            hidden_size=self.mlp_hidden_size,
            output_size=self.output_dim,
            num_layers=self.mlp_num_layers,
            dropout=self.mlp_dropout,
            use_residuals=self.mlp_use_residuals,
            normalization="layer",  # Usually good for projection heads
        )

    def forward(
        self,
        mol_pyg: pyg.data.Batch,
        ce: th.Tensor = None,
        ce_batch_idxs: th.Tensor = None,
        ace: th.Tensor = None,
        ace_batch_idxs: th.Tensor = None,
        prec_type: th.Tensor = None,
        inst_type: th.Tensor = None,
        frag_mode: th.Tensor = None,
    ):
        """
        Forward pass for molecular graph embedding.
        """
        mol_x, mol_edge_index, mol_edge_attr, mol_batch = (
            mol_pyg.x,
            mol_pyg.edge_index,
            mol_pyg.edge_attr,
            mol_pyg.batch,
        )
        batch_size = mol_batch[-1] + 1

        # Metadata embedders
        ce_embed = self.embed_ce(
            ce,
            ce_batch_idxs,
            batch_size,
            ace=ace,
            ace_batch_idxs=ace_batch_idxs,
        )
        prec_embed = self.embed_prec(prec_type)
        inst_embed = self.embed_inst(inst_type)
        frag_mode_embed = self.embed_frag_mode(frag_mode)

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
        if self.frag_mode_insert_location == "mol":
            mol_frag_mode_embed = th.repeat_interleave(
                frag_mode_embed, th.unique(mol_batch, return_counts=True)[1], dim=0
            )
            mol_x = th.cat([mol_x, mol_frag_mode_embed], dim=1)

        # GNN encoding
        mol_embed_gnn = self.mol_embedder(mol_x, mol_batch, mol_edge_index, mol_edge_attr)
        # Pooling to get graph-level embedding
        mol_embed_gnn_pool = self.mol_pool(mol_embed_gnn, mol_batch)

        # Metadata embeddings at the MLP level
        mlp_input = mol_embed_gnn_pool
        if self.ce_insert_location == "mlp":
            mlp_input = th.cat([mlp_input, ce_embed], dim=1)
        if self.prec_insert_location == "mlp":
            mlp_input = th.cat([mlp_input, prec_embed], dim=1)
        if self.inst_insert_location == "mlp":
            mlp_input = th.cat([mlp_input, inst_embed], dim=1)
        if self.frag_mode_insert_location == "mlp":
            mlp_input = th.cat([mlp_input, frag_mode_embed], dim=1)

        # Projection
        mol_embed = self.projector(mlp_input)

        return mol_embed

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

    def _frag_mode_location_check(self):
        assert self.frag_mode_insert_location in ["mlp", "mol", "none"], (
            f"frag_mode_insert_location={self.frag_mode_insert_location} not supported"
        )
