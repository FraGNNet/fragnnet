import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch import Tensor
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
)


class ResBlock(nn.Module):
    def __init__(self, ch, dr):
        super().__init__()
        self.linear = nn.Linear(ch, ch)
        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(dr)
        self.norm = nn.LayerNorm(ch)

    def forward(self, x):
        x = x + self.dropout(self.act(self.linear(x)))
        x = self.norm(x)
        return x


class GINEConv(pyg.nn.conv.GINEConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.act = nn.SiLU(inplace=False)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError(
                "Node and edge feature dimensionalities do not "
                "match. Consider setting the 'edge_dim' "
                "attribute of 'GINEConv'"
            )

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        return self.act(x_j + edge_attr)

    def forward(
        self,
        x: Tensor | OptPairTensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)


class GINELayer(nn.Module):
    def __init__(self, model_dim, dropout=0, bottleneck=1):
        super().__init__()
        self.model_dim = model_dim
        self.dropout = dropout
        identity = nn.Sequential(nn.Identity())
        identity[0].in_features = model_dim // bottleneck
        self.conv = GINEConv(nn=identity, edge_dim=model_dim // bottleneck, eps=0, train_eps=True)
        self.edge_lin = nn.Linear(model_dim, model_dim // bottleneck)
        self.lin1 = nn.Linear(model_dim, model_dim // bottleneck)
        self.lin2 = (
            nn.Linear(model_dim // bottleneck, model_dim) if bottleneck > 1 else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.norm = pyg.nn.GraphNorm(model_dim)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x, e, batch, edge_index):
        dx = self.lin1(x)
        dx = self.conv(x=dx, edge_index=edge_index, edge_attr=e)
        dx = self.dropout(dx)
        dx = self.lin2(dx)
        dx = self.act(dx)
        x = self.norm(x + dx, batch)
        return x


class GINEEdgeLayer(nn.Module):
    def __init__(self, model_dim, dropout=0, bottleneck=1):
        super().__init__()
        self.model_dim = model_dim
        self.dropout = dropout
        self.lin1 = nn.Linear(3 * model_dim, model_dim // bottleneck)
        self.lin2 = (
            nn.Linear(model_dim // bottleneck, model_dim) if bottleneck > 1 else nn.Identity()
        )
        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.norm = pyg.nn.GraphNorm(model_dim)

    def forward(self, x, e, batch, edge_index):
        de = torch.cat([e, x[edge_index[0]], x[edge_index[1]]], 1)
        de = self.lin1(de)
        de = self.act(de)
        de = self.dropout(de)
        de = self.lin2(de)
        e = self.norm(e + de, batch[edge_index[0]])
        return e


class GINE(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        model_dim,
        model_depth,
        jumping_knowledge=True,
        bottleneck=1,
        dropout=0,
        **kwargs,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.model_dim = model_dim
        self.model_depth = model_depth
        self.dropout = dropout
        self.bottleneck = bottleneck
        self.jumping_knowledge = jumping_knowledge

        if node_dim == model_dim:
            self.node_emb = nn.Identity()
        else:
            self.node_emb = nn.Sequential(
                nn.Linear(node_dim, model_dim),
            )
        if edge_dim == model_dim:
            self.edge_emb = nn.Identity()
        else:
            self.edge_emb = nn.Sequential(
                nn.Linear(edge_dim, model_dim),
            )

        layers = []
        edge_layers = []
        for _ in range(model_depth):
            layers.append(GINELayer(model_dim, dropout, bottleneck))
            edge_layers.append(GINEEdgeLayer(model_dim, dropout, bottleneck))
        self.layers = nn.ModuleList(layers)
        self.edge_layers = nn.ModuleList(edge_layers)

        self.linear = nn.Linear(model_dim, model_dim)

    def forward(self, g, x, e, e_mask=None):
        x = self.node_emb(x)
        e = self.edge_emb(e)
        for layer, edge_layer in zip(self.layers, self.edge_layers):
            # for layer in self.layers:
            x = layer(x, e, g.batch, g.edge_index)
            if e_mask is not None:
                e = torch.where(
                    e_mask, edge_layer(x, e[e_mask], g.batch, g.edge_index[:, e_mask]), e
                )
            else:
                e = edge_layer(x, e, g.batch, g.edge_index)
        x = self.linear(x)
        return x


class CanonicalOneHot(nn.Module):
    def __init__(self, node_feats={}, edge_feats={}, mask_value=-1, use_bool=True):
        super().__init__()
        self.node_feats = node_feats  # {**x_map, **node_feats}
        self.edge_feats = edge_feats  # {**e_map, **edge_feats}
        self.node_dim = sum(map(len, self.node_feats.values()))
        self.edge_dim = sum(map(len, self.edge_feats.values()))
        self.mask_value = mask_value
        self.use_bool = use_bool

    def forward(self, x, e):
        x_onehot = torch.zeros(x.shape[0], self.node_dim, device=x.device)
        j = 0
        for i, (feat, levels) in enumerate(self.node_feats.items()):
            if self.use_bool and [*levels] == [False, True]:
                # set masks to False
                mask = x[:, i] == self.mask_value
                x_onehot[~mask, j] = x[~mask, i].float()
                j += 1
            else:
                d = len(levels)
                mask = x[:, i] == self.mask_value
                try:
                    x_onehot[~mask, j : j + d] = F.one_hot(x[~mask, i].long(), d).float()
                except RuntimeError as e:
                    print(e)
                    print(x[~mask, i].long())
                    print(d)
                    print(feat)
                    print(levels)
                    import pdb

                    pdb.set_trace()
                j += d

        e_onehot = torch.zeros(e.shape[0], self.edge_dim, device=x.device)
        j = 0
        for i, (feat, levels) in enumerate(self.edge_feats.items()):
            if self.use_bool and [*levels] == [False, True]:
                # set masks to False
                mask = e[:, i] == self.mask_value
                e_onehot[~mask, j] = e[~mask, i].float()
                j += 1
            else:
                d = len(levels)
                mask = e[:, i] == self.mask_value
                e_onehot[~mask, j : j + d] = F.one_hot(e[~mask, i].long(), d).float()
                j += d

        return x_onehot, e_onehot
