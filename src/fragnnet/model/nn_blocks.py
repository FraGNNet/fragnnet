"""
This module implements various neural network building blocks for graph-based and feed-forward models.
It includes implementations of Graph Neural Networks (GNNs), Multilayer Perceptrons (MLPs), and other
specialized layers for molecular and spectra.

- ZerosAggregation: Dummy aggregation module that outputs zero tensors.
- GNN: Generic wrapper for different types of Graph Neural Networks (e.g., GINE, MPNN, GAT, GGNN).
- GINE: Graph Isomorphism Network with Edge Features.
- GINE_EFA: GINE augmented with per-graph EFA linear global attention.
- MPNN: Neural Message Passing for Quantum Chemistry.
- GAT: Graph Attention Networks (v1 and v2).
- GGNN: Gated Graph Neural Networks.
- NodeMLP: Multilayer Perceptron for node-level ms/ms prediction.
- MLPBlocks: Multilayer Perceptron with optional residual connections.
- NeimsBlock: Specialized block from the NEIMS paper.
- LowRankDense: Low-rank approximation of dense layers.
- SpecFFN: Feed-forward network for spectrum prediction.
"""

import logging

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch.utils.checkpoint import checkpoint as th_checkpoint
from torch_geometric.nn import aggr

from fragnnet.utils.misc_utils import safelog
from fragnnet.utils.nn_utils import get_clones

logger = logging.getLogger(__name__)


class PoolWithDimSizeAdapter(nn.Module):
    """Normalize pooling modules to a common interface with optional ``dim_size``."""

    def __init__(self, pool_module: nn.Module):
        super().__init__()
        self.pool_module = pool_module

    def forward(
        self,
        x: th.Tensor,
        index: th.Tensor | None = None,
        ptr: th.Tensor | None = None,
        dim_size: int | None = None,
        dim: int = -2,
        **kwargs,
    ) -> th.Tensor:
        try:
            pooled = self.pool_module(
                x,
                index=index,
                ptr=ptr,
                dim_size=dim_size,
                dim=dim,
                **kwargs,
            )
        except TypeError as exc:
            if "dim_size" not in str(exc):
                raise
            pooled = self.pool_module(x, index)

        if dim_size is None or pooled.shape[0] == dim_size:
            return pooled

        out = x.new_zeros((dim_size, pooled.shape[1]))
        if pooled.shape[0] > 0:
            out[: pooled.shape[0]] = pooled
        return out


class ZerosAggregation(nn.Module):
    """
    Zero Aggregation module.
    This module performs a dummy aggregation by summing up features across all elements
    and then multiplying the result by zero. Effectively, it outputs a tensor of zeros
    with the same shape as the aggregated result.

    Args:
        nn (torch.nn.Module): Base class for PyTorch modules.
    """

    def __init__(self):
        super().__init__()
        self.dummy_agg = aggr.SumAggregation()

    def forward(
        self,
        x: th.Tensor,
        index: th.Tensor | None = None,
        ptr: th.Tensor | None = None,
        dim_size: int | None = None,
        dim: int = -2,
        **kwargs,
    ) -> th.Tensor:
        """
        Forward pass for zero aggregation.

        Args:
            x (torch.Tensor): Input features.
            index (torch.Tensor | None): Group indices for aggregation.
            ptr (torch.Tensor | None): CSR-style segment pointers.
            dim_size (int | None): Explicit number of output groups.
            dim (int): Aggregation dimension.

        Returns:
            torch.Tensor: Zero tensor with the same shape as the aggregated result.
        """
        agg = self.dummy_agg(x, index=index, ptr=ptr, dim_size=dim_size, dim=dim, **kwargs)
        return th.zeros_like(agg)


class GNN(nn.Module):
    """
    Generic GNN Class.
    This class serves as a wrapper for different types of Graph Neural Networks (GNNs),
    such as GINE, NodeMLP, MPNN, GAT, and GGNN.

    Args:
        hidden_size (int): Hidden layer size.
        num_layers (int): Number of GNN layers.
        node_feats_size (int): Size of node features.
        edge_feats_size (int): Size of edge features.
        gnn_type (str): Type of GNN to use (e.g., "GINE", "MPNN", "GAT").
        dropout (float): Dropout rate.
        normalization (str): Type of normalization to use ("batch", "layer", etc.).
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        node_feats_size: int,
        edge_feats_size: int,
        gnn_type: str,
        dropout: float,
        normalization: str,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.edge_feats_size = edge_feats_size
        self.node_feats_size = node_feats_size
        self.dropout = dropout
        self.normalization = normalization
        self.gnn_type = gnn_type
        self.num_layers = num_layers

        # Input projection layer to map node features to hidden size
        self.input_project = nn.Linear(self.node_feats_size, self.hidden_size)

        # Initialize the appropriate GNN type
        if self.gnn_type == "GINE":
            self.gnn = GINE(
                hidden_size=self.hidden_size,
                edge_feats_size=self.edge_feats_size,
                node_feats_size=self.node_feats_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                normalization=self.normalization,
            )
        elif self.gnn_type == "GINE_EFA":
            self.gnn = GINE_EFA(
                hidden_size=self.hidden_size,
                edge_feats_size=self.edge_feats_size,
                node_feats_size=self.node_feats_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                normalization=self.normalization,
                **kwargs,
            )
        elif self.gnn_type == "NodeMLP":
            self.gnn = NodeMLP(
                hidden_size=self.hidden_size,
                edge_feats_size=self.edge_feats_size,
                node_feats_size=self.node_feats_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                normalization=self.normalization,
            )
        elif self.gnn_type == "MPNN":
            self.gnn = MPNN(
                node_feats_size=self.node_feats_size,
                edge_feats_size=self.edge_feats_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                normalization=self.normalization,
            )
        elif self.gnn_type == "GAT" or self.gnn_type == "GATv2":
            self.gnn = GAT(
                node_feats_size=self.node_feats_size,
                edge_feats_size=self.edge_feats_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                normalization=self.normalization,
                is_v2=(self.gnn_type == "GATv2"),
            )
        elif self.gnn_type == "GGNN":
            self.gnn = GGNN(
                node_feats_size=self.node_feats_size,
                edge_feats_size=self.edge_feats_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
            )
        else:
            raise NotImplementedError(self.gnn_type)

    def forward(self, x, batch_index, edge_index, edge_attr):
        """
        Forward pass for the GNN.

        Args:
            x (torch.Tensor): Node features.
            batch_index (torch.Tensor): Batch indices for nodes.
            edge_index (torch.Tensor): Edge indices.
            edge_attr (torch.Tensor): Edge features.

        Returns:
            torch.Tensor: Encoded node features.
        """
        z = self.input_project(x)
        z = self.gnn(z, batch_index, edge_index, edge_attr)
        return z


def get_norm(normalization, hidden_size):
    # norm_fn may or may not depend on the batch index
    if normalization == "batch":
        norm_mod = nn.BatchNorm1d(hidden_size)

        def norm_fn(x, b, n):
            return n(x)

    elif normalization == "layer":
        norm_mod = nn.LayerNorm(hidden_size)

        def norm_fn(x, b, n):
            return n(x)

    elif normalization == "graph":
        norm_mod = pyg.nn.GraphNorm(hidden_size)

        def norm_fn(x, b, n):
            return n(x, b)

    else:
        assert normalization == "none", normalization
        norm_mod = nn.Identity()

        def norm_fn(x, b, n):
            return n(x)

    return norm_mod, norm_fn


class GINE(nn.Module):
    """
    Graph Isomorphism Network with Edge Features (GINE).
    Implements the GINEConv layer as described in the paper:
    "Strategies for Pre-training Graph Neural Networks" (https://arxiv.org/abs/1905.12265).

    Args:
        hidden_size (int): Hidden layer size.
        node_feats_size (int): Size of node features.
        edge_feats_size (int): Size of edge features.
        num_layers (int): Number of GNN layers.
        dropout (float): Dropout rate.
        normalization (str): Type of normalization to use ("batch", "layer", etc.).
    """

    def __init__(
        self,
        hidden_size: int,
        node_feats_size: int,
        edge_feats_size: int,
        num_layers: int,
        dropout: float,
        normalization: str,
        **kwargs,
    ):
        super().__init__()
        assert edge_feats_size >= 0, edge_feats_size

        # Linear transformation for edge features
        self.edge_transform = nn.Linear(edge_feats_size, hidden_size)

        # Initialize GINE layers
        self.layers = []
        for i in range(num_layers):
            apply_fn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            temp_layer = pyg.nn.conv.GINEConv(nn=apply_fn, eps=0.0, edge_dim=None)
            self.layers.append(temp_layer)

        self.layers = nn.ModuleList(self.layers)

        # Setup normalization and dropout
        norm_mod, norm_fn = get_norm(normalization, hidden_size)
        self.norm_fn = norm_fn
        self.norms = get_clones(norm_mod, num_layers)
        self.dropouts = get_clones(nn.Dropout(dropout), num_layers)

    def forward(self, x, batch_index, edge_index, edge_attr):
        """
        Forward pass for GINE.

        Args:
            x (torch.Tensor): Node features.
            batch_index (torch.Tensor): Batch indices for nodes.
            edge_index (torch.Tensor): Edge indices.
            edge_attr (torch.Tensor): Edge features.

        Returns:
            torch.Tensor: Encoded node features.
        """
        edge_attr = self.edge_transform(edge_attr)
        for dropout, layer, norm in zip(self.dropouts, self.layers, self.norms):
            layer_out = layer(x, edge_index, edge_attr)
            layer_out = self.norm_fn(layer_out, batch_index, norm)
            x = F.relu(dropout(layer_out)) + x  # Residual connection
        return x


def _efa_linear_attn(Q: th.Tensor, K: th.Tensor, V: th.Tensor, batch_index: th.Tensor) -> th.Tensor:
    """Per-graph EFA linear attention on flat PyG node tensors.

    Uses the ELU+1 feature map, giving O(N·H·D²) complexity vs O(N²·H·D) for softmax.
    Context is computed separately per graph via scatter_add so variable-size graphs
    are handled correctly without padding.

    Args:
        Q: Queries of shape (N, num_heads, head_dim).
        K: Keys of shape (N, num_heads, head_dim).
        V: Values of shape (N, num_heads, head_dim).
        batch_index: Graph assignments of shape (N,).

    Returns:
        Attended values of shape (N, num_heads, head_dim).
    """
    phi_Q = F.elu(Q) + 1  # (N, H, D)
    phi_K = F.elu(K) + 1  # (N, H, D)
    N, H, D = phi_Q.shape
    num_graphs = int(batch_index.max().item()) + 1

    # Accumulate outer-product KV context per graph: (G, H, D, D)
    KV = Q.new_zeros(num_graphs, H, D, D)
    KV_nodes = th.einsum("nhd,nhe->nhde", phi_K, V)  # (N, H, D, D)
    KV.scatter_add_(0, batch_index.view(N, 1, 1, 1).expand_as(KV_nodes), KV_nodes)

    # Accumulate key sums per graph for denominator: (G, H, D)
    K_sum = Q.new_zeros(num_graphs, H, D)
    K_sum.scatter_add_(0, batch_index.view(N, 1, 1).expand_as(phi_K), phi_K)

    # Gather per-node context and compute normalized output
    KV_node = KV[batch_index]  # (N, H, D, D)
    K_sum_node = K_sum[batch_index]  # (N, H, D)
    out = th.einsum("nhd,nhde->nhe", phi_Q, KV_node)  # (N, H, D)
    Z = (phi_Q * K_sum_node).sum(dim=-1, keepdim=True).clamp(min=1e-6)  # (N, H, 1)
    return out / Z


class GINE_EFA(nn.Module):
    """GINE augmented with per-graph EFA linear global attention.

    After each GINEConv message-passing step, a graph-scoped linear attention pass
    (Euclidean Fast Attention with ELU+1 feature map) mixes information across all
    nodes in the same molecule, capturing long-range interactions that local MP misses.
    Complexity is O(N·H·D²) instead of O(N²·H·D) for standard softmax attention.

    Args:
        hidden_size (int): Hidden layer size.
        node_feats_size (int): Size of node features.
        edge_feats_size (int): Size of edge features.
        num_layers (int): Number of GNN + attention layers.
        dropout (float): Dropout rate.
        normalization (str): Normalization type ("batch", "layer", "graph", "none").
        num_heads (int): Number of attention heads. Must divide hidden_size. Defaults to 4.
    """

    def __init__(
        self,
        hidden_size: int,
        node_feats_size: int,
        edge_feats_size: int,
        num_layers: int,
        dropout: float,
        normalization: str,
        num_heads: int = 4,
        **kwargs,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        assert edge_feats_size >= 0, edge_feats_size

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.edge_transform = nn.Linear(edge_feats_size, hidden_size)

        gine_layers = []
        for _ in range(num_layers):
            apply_fn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            gine_layers.append(pyg.nn.conv.GINEConv(nn=apply_fn, eps=0.0, edge_dim=None))
        self.layers = nn.ModuleList(gine_layers)

        self.efa_q = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.efa_k = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.efa_v = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.efa_out = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )

        norm_mod, norm_fn = get_norm(normalization, hidden_size)
        self.norm_fn = norm_fn
        self.norms = get_clones(norm_mod, num_layers)
        self.efa_norms = get_clones(norm_mod, num_layers)
        self.dropouts = get_clones(nn.Dropout(dropout), num_layers)

    def forward(
        self,
        x: th.Tensor,
        batch_index: th.Tensor,
        edge_index: th.Tensor,
        edge_attr: th.Tensor,
    ) -> th.Tensor:
        """Forward pass: local GINEConv → global EFA → residual sum, repeated per layer.

        Args:
            x: Node features of shape (N, hidden_size).
            batch_index: Graph assignments of shape (N,).
            edge_index: Edge connectivity of shape (2, E).
            edge_attr: Edge features of shape (E, edge_feats_size).

        Returns:
            Updated node features of shape (N, hidden_size).
        """
        edge_attr = self.edge_transform(edge_attr)
        for dropout, layer, norm, efa_norm, q_proj, k_proj, v_proj, out_proj in zip(
            self.dropouts,
            self.layers,
            self.norms,
            self.efa_norms,
            self.efa_q,
            self.efa_k,
            self.efa_v,
            self.efa_out,
        ):
            # Local message passing (same residual pattern as GINE)
            local_out = layer(x, edge_index, edge_attr)
            local_out = self.norm_fn(local_out, batch_index, norm)
            local_out = F.relu(dropout(local_out)) + x

            # Global linear attention within each graph
            N, C = local_out.shape
            Q = q_proj(local_out).view(N, self.num_heads, self.head_dim)
            K = k_proj(local_out).view(N, self.num_heads, self.head_dim)
            V = v_proj(local_out).view(N, self.num_heads, self.head_dim)
            efa_out = th_checkpoint(
                _efa_linear_attn, Q, K, V, batch_index, use_reentrant=False
            ).reshape(N, C)
            efa_out = dropout(out_proj(F.relu(efa_out)))
            efa_out = self.norm_fn(efa_out, batch_index, efa_norm)

            x = local_out + efa_out
        return x


class MPNN(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        node_feats_size: int,
        edge_feats_size: int,
        num_layers: int,
        dropout: float,
        normalization: str,
        **kwargs,
    ):
        """Neural Message Passing for Quantum Chemistry
        https://arxiv.org/abs/1704.01212
        """
        super().__init__()
        assert edge_feats_size >= 0, edge_feats_size
        self.edge_transform = nn.Linear(edge_feats_size, hidden_size)
        # MPNN
        self.layers = []
        for i in range(num_layers):
            edge_network = edge_network = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size * hidden_size),
            )
            layer = pyg.nn.NNConv(
                hidden_size, hidden_size, edge_network, aggr="mean", root_weight=False
            )
            self.layers.append(layer)
        self.layers = nn.ModuleList(self.layers)
        # setup norms and dropout
        norm_mod, norm_fn = get_norm(normalization, hidden_size)
        self.norm_fn = norm_fn
        self.norms = get_clones(norm_mod, num_layers)
        self.dropouts = get_clones(nn.Dropout(dropout), num_layers)

    def forward(self, x, batch_index, edge_index, edge_attr):
        """forward."""
        edge_attr = self.edge_transform(edge_attr)
        for dropout, layer, norm in zip(self.dropouts, self.layers, self.norms):
            layer_out = layer(x, edge_index, edge_attr)
            layer_out = self.norm_fn(layer_out, batch_index, norm)
            x = F.relu(dropout(layer_out)) + x
        return x


class GAT(nn.Module):
    """Graph attention networks. https://arxiv.org/abs/1710.10903
    Graph attention networks v2. (How Attentive are Graph Attention Networks?  https://arxiv.org/abs/2105.14491)
    """

    def __init__(
        self,
        node_feats_size: int,
        edge_feats_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        normalization: str,
        num_gat_heads: int = 8,
        gat_dropout: float = 0.0,
        is_v2: bool = True,
        is_concat: bool = True,
        **kwargs,
    ):
        """_summary_
        Args:
            node_feats_size (int): _description_
            edge_feats_size (int): _description_
            node_hidden_dim (int): _description_
            num_step_message_passing (int): _description_
            num_step_set2set (int): _description_
            output_dim (int): _description_
            gat_heads (int, optional): _description_. Defaults to 1.
            gat_dropout (float, optional): _description_. Defaults to 0.0.
            is_v2 (bool, optional). use v2 GATConv, Defaults to True
            is_concat (bool, optional). concatenated multihead attetion, Defaults to True
        """
        super().__init__()
        assert edge_feats_size >= 0, edge_feats_size
        self.edge_transform = nn.Linear(edge_feats_size, hidden_size)
        gat_output_size = hidden_size // num_gat_heads if is_concat else hidden_size
        if is_concat and hidden_size % num_gat_heads != 0:
            raise ValueError(
                f"Ensure that the number of output channels of "
                f"'GAT' (got '{hidden_size}') is divisible "
                f"by the number of heads (got '{num_gat_heads}')"
            )
        GATConv_fn = pyg.nn.conv.GATConv if not is_v2 else pyg.nn.conv.GATv2Conv
        self.layers = []
        for i in range(num_layers):
            layer = GATConv_fn(
                in_channels=hidden_size,
                out_channels=gat_output_size,
                heads=num_gat_heads,
                edge_dim=hidden_size,
                dropout=gat_dropout,
                concat=is_concat,
            )
            self.layers.append(layer)
        self.layers = nn.ModuleList(self.layers)
        # setup norms and dropout
        norm_mod, norm_fn = get_norm(normalization, hidden_size)
        self.norm_fn = norm_fn
        self.norms = get_clones(norm_mod, num_layers)
        self.dropouts = get_clones(nn.Dropout(dropout), num_layers)

    def forward(self, x, batch_index, edge_index, edge_attr):
        edge_attr = self.edge_transform(edge_attr)
        for dropout, layer, norm in zip(self.dropouts, self.layers, self.norms):
            layer_out = layer(x, edge_index, edge_attr)
            layer_out = self.norm_fn(layer_out, batch_index, norm)
            x = F.relu(dropout(layer_out)) + x
        return x


class GGNN(nn.Module):
    def __init__(
        self,
        node_feats_size: int,
        edge_feats_size: int,
        hidden_size: int,
        num_layers: int,
        **kwargs,
    ):
        """ """
        super().__init__()
        assert edge_feats_size >= 0, edge_feats_size
        self.edge_transform = nn.Linear(edge_feats_size, 1)
        self.model = pyg.nn.conv.GatedGraphConv(out_channels=hidden_size, num_layers=num_layers)

    def forward(self, x, batch_index, edge_index, edge_attr):
        """forward."""
        edge_weight = self.edge_transform(edge_attr).squeeze(1)
        x = self.model(x=x, edge_index=edge_index, edge_weight=edge_weight)
        return x


class NodeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        node_feats_size: int,
        edge_feats_size: int,
        num_layers: int,
        dropout: float,
        normalization: str,
        **kwargs,
    ):
        """NodeMLP"""
        super().__init__()

        assert edge_feats_size >= 0, edge_feats_size
        if edge_feats_size > 0:
            logger.warning(
                "NodeMLP does not use edge features, but got edge_feats_size > 0, this most likely indicates a misconfiguration and wasted VRAM."
            )

        self.layers = []
        for i in range(num_layers):
            apply_fn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            temp_layer = apply_fn
            self.layers.append(temp_layer)

        self.layers = nn.ModuleList(self.layers)
        # setup norm and norm_fn
        # norm_fn may or may not depend on the batch index
        if normalization == "batch":
            norm_mod = nn.BatchNorm1d(hidden_size)
            self.norm_fn = lambda x, b, n: n(x)
        elif normalization == "layer":
            norm_mod = nn.LayerNorm(hidden_size)
            self.norm_fn = lambda x, b, n: n(x)
        elif normalization == "graph":
            norm_mod = pyg.nn.GraphNorm(hidden_size)
            self.norm_fn = lambda x, b, n: n(x, b)
        else:
            assert normalization == "none", normalization
            norm_mod = nn.Identity()
            self.norm_fn = lambda x, b, n: n(x)
        self.norms = get_clones(norm_mod, num_layers)
        self.dropouts = get_clones(nn.Dropout(dropout), num_layers)

    def forward(self, x, batch_index, edge_index, edge_attr):
        """forward."""
        for dropout, layer, norm in zip(self.dropouts, self.layers, self.norms):
            layer_out = layer(x)
            layer_out = self.norm_fn(layer_out, batch_index, norm)
            x = F.relu(dropout(layer_out)) + x
        return x


class MLPBlocks(nn.Module):
    """
     Just Good Old Multilayer perceptron with residuals
    layer->dropout->activation(Relu)
    if residuals is True, add a skip connection between block
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        normalization: str,
        dropout: float,
        num_layers: int,
        output_size: int = None,
        use_residuals: bool = False,
    ):
        """MLP with optional residual connections.

        Args:
            input_size (int): Input feature dimension.
            hidden_size (int): Hidden layer dimension.
            normalization (str): Type of normalization ('batch', 'layer', or 'none').
            dropout (float): Dropout probability.
            num_layers (int): Number of hidden layers (excluding input/output).
            output_size (int, optional): Output feature dimension. If None, no output layer is used. If not None, use a linear layer to project to output_size.
            use_residuals (bool, optional): If True, adds skip connections between hidden layers.
        """

        super().__init__()
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.input_layer = nn.Linear(input_size, hidden_size)
        middle_layer = nn.Linear(hidden_size, hidden_size)
        self.layers = get_clones(middle_layer, num_layers - 1)

        self.output_layer = None
        self.output_size = output_size
        if self.output_size is not None:
            self.output_layer = nn.Linear(hidden_size, self.output_size)

        self.use_residuals = use_residuals

        # setup norm
        if normalization == "batch":
            norm_mod = nn.BatchNorm1d(hidden_size)
        elif normalization == "layer":
            norm_mod = nn.LayerNorm(hidden_size)
        else:
            assert normalization == "none", normalization
            norm_mod = nn.Identity()
        self.norms = get_clones(norm_mod, num_layers - 1)

    def forward(self, x):
        output = x
        output = self.input_layer(x)
        output = self.dropout_layer(output)
        output = self.activation(output)
        old_op = output
        for layer_index, layer in enumerate(self.layers):
            output = layer(output)
            output = self.dropout_layer(output)
            output = self.activation(output)
            output = self.norms[layer_index](output)
            if self.use_residuals:
                output = output + old_op
                old_op = output

        if self.output_layer is not None:
            output = self.output_layer(output)
        return output


class NeimsBlock(nn.Module):
    """from the NEIMS paper (uses LeakyReLU instead of ReLU)"""

    def __init__(self, in_dim, out_dim, dropout, bottleneck_factor=0.5):
        super().__init__()
        bottleneck_size = int(round(bottleneck_factor * out_dim))
        self.in_batch_norm = nn.BatchNorm1d(in_dim)
        self.in_activation = nn.LeakyReLU()
        self.in_linear = nn.Linear(in_dim, bottleneck_size)
        self.out_batch_norm = nn.BatchNorm1d(bottleneck_size)
        self.out_linear = nn.Linear(bottleneck_size, out_dim)
        self.out_activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        h = x
        h = self.in_batch_norm(h)
        h = self.in_activation(h)
        h = self.dropout(h)
        h = self.in_linear(h)
        h = self.out_batch_norm(h)
        h = self.out_activation(h)
        h = self.out_linear(h)
        return h


class LowRankDense(nn.Module):
    """
    Low-Rank Dense Layer. Ref: https://arxiv.org/pdf/2001.08885
    This class implements a low-rank approximation of a dense (fully connected) layer.
    Instead of a single weight matrix, it factorizes the weight matrix into two smaller
    matrices, reducing the number of parameters and computational cost.

    Args:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
        rank (int): Rank of the factorization (controls the size of the intermediate layer).
        bias (bool): Whether to include a bias term in the layer.
    """

    def __init__(self, input_dim, output_dim, rank, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank

        # First projection: input_dim -> rank
        self.proj1 = nn.Linear(input_dim, rank, bias=False)

        # Second projection: rank -> output_dim
        self.proj2 = nn.Linear(rank, output_dim, bias=bias)

    def forward(self, x):
        """
        Forward pass for the Low-Rank Dense layer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim].
        """
        # First projection: Reduce input dimension to rank
        x = self.proj1(x)

        # Second projection: Map rank dimension to output dimension
        x = self.proj2(x)

        return x


class SpecFFN(nn.Module):
    """Spectral Feed-Forward Network.

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        mz_max,
        mz_bin_res,
        num_layers,
        dropout,
        prec_mz_offset,
        bidirectional,
        use_residuals,
        output_map_size,
        output_activation,
        log_min,
        bottleneck_factor=0.5,
    ):
        super().__init__()

        self.input_size = input_size
        self.mz_max = mz_max
        self.mz_bin_res = mz_bin_res
        self.prec_mz_offset = prec_mz_offset
        self.bidirectional = bidirectional
        self.use_residuals = use_residuals
        self._compute_output_size()

        self.in_layer = nn.Linear(input_size, hidden_size)
        self.ff_layers = nn.ModuleList([])
        # self.ff_layers.append(nn.Linear(mlp_hidden_size, mlp_hidden_size))
        for i in range(num_layers):
            self.ff_layers.append(NeimsBlock(hidden_size, hidden_size, dropout, bottleneck_factor))
        if output_map_size == -1:
            if self.bidirectional:
                # assumes gating, mass masking
                self.forw_out_layer = nn.Linear(hidden_size, self.output_size)
                self.rev_out_layer = nn.Linear(hidden_size, self.output_size)
                self.out_gate = nn.Sequential(
                    *[nn.Linear(hidden_size, self.output_size), nn.Sigmoid()]
                )
            else:
                self.out_layer = nn.Linear(hidden_size, self.output_size)
                self.out_gate = nn.Sequential(
                    *[nn.Linear(hidden_size, self.output_size), nn.Sigmoid()]
                )
        else:
            if self.bidirectional:
                # assumes gating, mass masking
                self.forw_out_layer = LowRankDense(hidden_size, self.output_size, output_map_size)
                self.rev_out_layer = LowRankDense(hidden_size, self.output_size, output_map_size)
                self.out_gate = nn.Sequential(
                    *[LowRankDense(hidden_size, self.output_size, output_map_size), nn.Sigmoid()]
                )
            else:
                self.out_layer = LowRankDense(hidden_size, self.output_size, output_map_size)
                self.out_gate = nn.Sequential(
                    *[LowRankDense(hidden_size, self.output_size, output_map_size), nn.Sigmoid()]
                )
        assert output_activation in ["relu", "sigmoid"], output_activation
        if output_activation == "relu":
            self.out_activation = nn.ReLU()
        else:
            self.out_activation = nn.Sigmoid()
        self.out_normalization = lambda x: F.normalize(x, p=1, dim=1)
        self.log_min = log_min

    def _compute_output_size(self):
        mz_bins = th.arange(
            self.mz_bin_res, self.mz_max + self.mz_bin_res, self.mz_bin_res, dtype=th.float32
        )
        self.register_buffer("mz_bins", mz_bins, persistent=False)
        self.register_buffer("mzs", mz_bins - 0.5 * self.mz_bin_res, persistent=False)
        self.output_size = mz_bins.shape[0]

    def _prec_mz_to_idx(self, prec_mz):
        prec_mz_idx = th.bucketize(prec_mz, self.mz_bins.to(prec_mz.device), right=True)
        assert th.max(prec_mz_idx) < self.output_size, (prec_mz_idx, self.output_size)
        return prec_mz_idx

    def forward(self, input_h, prec_mz):
        # get prec_mz_idxs
        prec_mz_idxs = self._prec_mz_to_idx(prec_mz)
        # process inputs
        fh = self.in_layer(input_h)
        # big MLP
        for layer in self.ff_layers:
            if self.use_residuals:
                fh = fh + layer(fh)
            else:
                fh = layer(fh)
        # bidirectional prediction
        if self.bidirectional:
            ff = self.forw_out_layer(fh)
            fr = reverse_prediction(self.rev_out_layer(fh), prec_mz_idxs, self.prec_mz_offset)
            fg = self.out_gate(fh)
            fo = ff * fg + fr * (1.0 - fg)
        else:
            # apply output layer
            fo = self.out_layer(fh)
            # apply gating
            fg = self.out_gate(fh)
            fo = fg * fo
        fo = self.out_activation(fo)
        fo = mask_prediction_by_mass(fo, prec_mz_idxs, self.prec_mz_offset)
        spec = self.out_normalization(fo)
        # handle all zeroes (set to first bin by default)
        all_zero_mask = th.max(spec, dim=1)[0] <= 0.0
        all_zero_bonus = th.zeros_like(spec)
        all_zero_bonus[all_zero_mask, 0] = 1.0
        spec = spec + all_zero_bonus
        # convert dense spectrum to sparse
        mask = spec > 0.0
        pred_mzs = (self.mzs.unsqueeze(0).expand(spec.shape[0], -1))[mask]
        pred_logprobs = safelog(spec, eps=self.log_min)[mask]
        pred_batch_idxs = (
            th.arange(spec.shape[0], device=spec.device)
            .unsqueeze(1)
            .expand(-1, spec.shape[1])[mask]
        )
        pred_specs = spec
        return pred_mzs, pred_logprobs, pred_batch_idxs, pred_specs


def _pack_variable_length(
    x: th.Tensor, batch_idx: th.Tensor
) -> tuple[th.Tensor, th.Tensor]:
    """Pack flat (total, dim) into padded (batch, max_len, dim) with boolean mask.

    Args:
        x: Flat feature tensor of shape (total_elements, dim).
        batch_idx: Batch index per element of shape (total_elements,).

    Returns:
        x_padded: Padded tensor of shape (batch_size, max_len, dim).
        mask: Boolean mask of shape (batch_size, max_len), True = padding position.
    """
    batch_size = int(batch_idx.max().item()) + 1
    set_sizes = batch_idx.bincount(minlength=batch_size)  # (batch_size,)
    max_size = int(set_sizes.max().item())

    cum_starts = th.cat(
        [th.zeros(1, dtype=th.long, device=x.device), set_sizes.cumsum(0)[:-1]]
    )
    local_pos = th.arange(x.shape[0], device=x.device) - cum_starts[batch_idx]

    x_padded = x.new_zeros(batch_size, max_size, x.shape[-1])
    x_padded[batch_idx, local_pos] = x

    arange = th.arange(max_size, device=x.device).unsqueeze(0)
    mask = arange >= set_sizes.unsqueeze(1)  # True = padding
    return x_padded, mask


def _unpack_variable_length(x_padded: th.Tensor, batch_idx: th.Tensor) -> th.Tensor:
    """Inverse of _pack_variable_length: strips padding, returns (total, dim).

    Args:
        x_padded: Padded tensor of shape (batch_size, max_len, dim).
        batch_idx: Original batch indices of shape (total_elements,).

    Returns:
        Flat tensor of shape (total_elements, dim).
    """
    batch_size = x_padded.shape[0]
    set_sizes = batch_idx.bincount(minlength=batch_size)
    return th.cat([x_padded[i, : int(set_sizes[i].item())] for i in range(batch_size)], dim=0)


class _MAB(nn.Module):
    """Multi-head Attention Block: Q attends to KV with residual + LayerNorm + FFN.

    Args:
        dim: Feature dimension (Q, K, V all share the same dim).
        num_heads: Number of attention heads.
        dropout: Dropout applied inside attention and after the attention sublayer.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        Q: th.Tensor,
        KV: th.Tensor,
        kv_padding_mask: th.Tensor | None = None,
    ) -> th.Tensor:
        """Args:
            Q: Query tensor of shape (B, Nq, dim).
            KV: Key/value tensor of shape (B, Nkv, dim).
            kv_padding_mask: Boolean mask (B, Nkv), True = padding (ignored).

        Returns:
            Updated query tensor of shape (B, Nq, dim).
        """
        H, _ = self.attn(Q, KV, KV, key_padding_mask=kv_padding_mask)
        Q = self.norm1(Q + self.drop(H))
        Q = self.norm2(Q + self.ffn(Q))
        return Q


class _ISAB(nn.Module):
    """Induced Set Attention Block (ISAB) from the Set Transformer paper.

    Each block runs two MABs:
      H = MAB(I, X)     — inducing points compress the input set
      out = MAB(X, H)   — input reads back from the compressed representation

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        num_inds: Number of learnable inducing points.
        dropout: Dropout rate.
    """

    def __init__(self, dim: int, num_heads: int, num_inds: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.inducing_points = nn.Parameter(th.randn(1, num_inds, dim))
        self.mab_i = _MAB(dim, num_heads, dropout)  # I <- X
        self.mab_x = _MAB(dim, num_heads, dropout)  # X <- H

    def forward(
        self, X: th.Tensor, x_padding_mask: th.Tensor | None = None
    ) -> th.Tensor:
        """Args:
            X: Input of shape (B, N, dim).
            x_padding_mask: Boolean mask (B, N), True = padding.

        Returns:
            Updated X of shape (B, N, dim).
        """
        B = X.shape[0]
        I = self.inducing_points.expand(B, -1, -1)
        H = self.mab_i(I, X, kv_padding_mask=x_padding_mask)  # (B, num_inds, dim)
        return self.mab_x(X, H)  # (B, N, dim) — H has no padding


class _PMA(nn.Module):
    """Pooling by Multi-head Attention (PMA) with k learned seed vectors.

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        num_seeds: Number of output vectors (1 produces a single set embedding).
        dropout: Dropout rate.
    """

    def __init__(
        self, dim: int, num_heads: int, num_seeds: int = 1, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.seeds = nn.Parameter(th.randn(1, num_seeds, dim))
        self.mab = _MAB(dim, num_heads, dropout)

    def forward(self, X: th.Tensor) -> th.Tensor:
        """Args:
            X: Input of shape (B, N, dim).

        Returns:
            Pooled representation of shape (B, num_seeds, dim).
        """
        return self.mab(self.seeds.expand(X.shape[0], -1, -1), X)


class SetTransformer(nn.Module):
    """Set Transformer with ISAB encoder blocks and PMA pooling.

    Implements the architecture from "Set Transformer: A Framework for
    Attention-based Permutation-Invariant Neural Networks" (Lee et al., 2019).
    Each ISAB block runs two cross-attention MABs and achieves O(N·m) complexity
    where m = num_inds. Final pooling uses PMA (learned seed vectors) instead of
    mean pooling.

    Args:
        input_dim: Feature dimension entering the ISAB blocks (after any external
            input projection).
        hidden_dim: Hidden dimension used in the output MLP projector.
        output_dim: Dimension of the output embedding.
        num_heads: Number of attention heads (must divide input_dim).
        num_inds: Number of inducing points per ISAB block.
        num_blocks: Number of ISAB blocks.
        dropout: Dropout rate applied inside attention and FFN sublayers.
        return_nodes: If True, return per-element outputs of shape
            (total_elements, output_dim); if False, return per-set embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int = 4,
        num_inds: int = 32,
        num_blocks: int = 2,
        dropout: float = 0.1,
        return_nodes: bool = False,
    ) -> None:
        super().__init__()
        self.return_nodes = return_nodes
        self.isab_blocks = nn.ModuleList(
            [_ISAB(input_dim, num_heads, num_inds, dropout) for _ in range(num_blocks)]
        )
        if return_nodes:
            self.out_proj = nn.Linear(input_dim, output_dim)
        else:
            self.pma = _PMA(input_dim, num_heads, num_seeds=1, dropout=dropout)
            self.out_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x: th.Tensor, batch_idx: th.Tensor | None = None) -> th.Tensor:
        """Forward pass for the Set Transformer.

        Args:
            x: Input tensor.
                - Fixed-size path (batch_idx is None): shape (B, N, input_dim).
                - Variable-size path (batch_idx provided): shape (total, input_dim).
            batch_idx: Batch index per element of shape (total,). Required for
                variable-size sets; omit for uniformly-sized batches.

        Returns:
            - return_nodes=False: set embeddings of shape (B, output_dim).
            - return_nodes=True: per-element outputs of shape (total, output_dim).
        """
        if batch_idx is None:
            for isab in self.isab_blocks:
                x = isab(x)
            if self.return_nodes:
                return self.out_proj(x)
            return self.out_proj(self.pma(x).squeeze(1))

        x_padded, mask = _pack_variable_length(x, batch_idx)
        for isab in self.isab_blocks:
            x_padded = isab(x_padded, x_padding_mask=mask)

        if self.return_nodes:
            return _unpack_variable_length(self.out_proj(x_padded), batch_idx)
        return self.out_proj(self.pma(x_padded).squeeze(1))


def reverse_prediction(raw_prediction, prec_mass_idx, prec_mass_offset):
    # adapted from NEIMS
    # raw_prediction is [B,D], prec_mass_idx is [B]

    batch_size = raw_prediction.shape[0]
    max_idx = raw_prediction.shape[1]
    assert th.all(prec_mass_idx < max_idx)
    rev_prediction = th.flip(raw_prediction, dims=(1,))
    # convention is to shift right, so we express as negative to go left
    offset_idx = th.minimum(
        max_idx * th.ones_like(prec_mass_idx), prec_mass_idx + prec_mass_offset + 1
    )
    shifts = -(max_idx - offset_idx)
    gather_idx = (
        th.arange(max_idx, device=raw_prediction.device).unsqueeze(0).expand(batch_size, max_idx)
    )
    gather_idx = (gather_idx - shifts.unsqueeze(1)) % max_idx
    offset_rev_prediction = th.gather(rev_prediction, 1, gather_idx)
    # you could mask_prediction_by_mass here but it's unnecessary
    return offset_rev_prediction


class Set2SetPooling(nn.Module):
    """Wrapper for Set2Set that handles batch argument correctly."""

    def __init__(self, node_dim: int, processing_steps: int = 3):
        super().__init__()
        self.set2set = pyg.nn.Set2Set(node_dim, processing_steps=processing_steps)
        self.linear = nn.Linear(2 * node_dim, node_dim)

    def forward(self, x, batch):
        """Forward pass that accepts batch argument like other PyG aggregations."""
        x = self.set2set(x, batch)
        x = self.linear(x)
        return x


def build_pool_module(pool_type: str, node_dim: int):
    """Method to build pooling layer for given pool_type and dim
        node dim is required for attention layer.
    Args:
        pool_type (str): pooling type in ['sum','mean','max','attention','set2set','set_transformer','softmax','none']
        node_dim (int): node_dim, used for attention and set_transformer
    Returns:
        nn.Module: pooling module
    """

    if pool_type == "sum":
        pool_module = aggr.SumAggregation()
    elif pool_type == "mean":
        pool_module = aggr.MeanAggregation()
    elif pool_type == "max":
        pool_module = aggr.MaxAggregation()
    elif pool_type == "attention":
        pool_module = aggr.AttentionalAggregation(
            gate_nn=nn.Sequential(nn.Linear(node_dim, node_dim), nn.ReLU(), nn.Linear(node_dim, 1))
        )
    elif pool_type == "set_transformer":
        # use SetTransformerAggregation from pyg
        # TODO pass channels, num_seed_points, num_encoder_blocks, num_decoder_blocks, heads, concat, layer_norm, dropout
        raise NotImplementedError
        # pool_module = aggr.SetTransformerAggregation(
        #   channels=node_dim,
        #   num_seed_points=16,
        #   num_encoder_blocks=1,
        #   num_decoder_blocks=1,
        #   heads=4,
        #   concat=True,
        #   layer_norm=False,
        #   dropout= 0.1
        # )
    elif pool_type == "equilibrium":
        raise NotImplementedError
        # TODO pass in_channels and out_channels
        # pool_module = aggr.EquilibriumAggregation(
        #   in_channels=node_dim,
        #   out_channels=node_dim
        # )
    elif pool_type == "softmax":
        pool_module = aggr.SoftmaxAggregation(learn=True)
    elif pool_type == "mean_max_sum":
        # raise NotImplementedError
        pool_module = aggr.MultiAggregation(["mean", "max", "sum"])
    elif pool_type == "mean_std_softmax":
        # raise NotImplementedError
        pool_module = aggr.MultiAggregation(["mean", "std", aggr.SoftmaxAggregation(learn=True)])
    else:
        assert pool_type == "none", pool_type
        # just return 0s
        pool_module = ZerosAggregation()
    return PoolWithDimSizeAdapter(pool_module)


def mask_prediction_by_mass(raw_prediction, prec_mass_idx, prec_mass_offset):
    # adapted from NEIMS
    # raw_prediction is [B,D], prec_mass_idx is [B]

    max_idx = raw_prediction.shape[1]
    assert th.all(prec_mass_idx < max_idx)
    idx = th.arange(max_idx, device=prec_mass_idx.device)
    mask = (idx.unsqueeze(0) <= (prec_mass_idx.unsqueeze(1) + prec_mass_offset)).float()
    return mask * raw_prediction
