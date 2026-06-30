import numpy as np
import torch as th
import torch.nn as nn
import torch_geometric as pyg

from fragnnet.graff.nn_utils import *
from fragnnet.utils.misc_utils import (
    check_pyg_compile,
    check_pyg_full_compile,
    scatter_logmeanexp,
    scatter_logsoftmax,
    scatter_reduce,
)


class GrAFFModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_isotope_types,
        encoder_dim,
        decoder_dim,
        encoder_depth,
        decoder_depth,
        num_eigs,
        eig_dim,
        eig_depth,
        dropout,
        min_probability,
        min_mz,
        node_feats,
        edge_feats,
        output_formula_str,
        **kwargs,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.num_eigs = num_eigs
        self.eig_dim = eig_dim
        self.eig_depth = eig_depth
        self.dropout = dropout

        self.vocab_size = vocab_size
        self.num_isotope_types = num_isotope_types
        self.covariates_dim = 2

        # inference time only
        self.min_probability = min_probability
        self.min_mz = min_mz
        self.output_formula_str = output_formula_str

        # embedding layers
        self.onehot = CanonicalOneHot(node_feats=node_feats, edge_feats=edge_feats)
        self.node_emb = nn.Sequential(
            nn.Linear(self.onehot.node_dim, encoder_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim),
        )
        self.edge_emb = nn.Sequential(
            nn.Linear(self.onehot.edge_dim, encoder_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim),
        )

        if num_eigs > 0:
            self.signnet = SignNet(
                num_eigs=num_eigs,
                embed_dim=encoder_dim,
                rho_dim=encoder_dim,
                rho_depth=eig_depth,
                phi_dim=eig_dim,
                phi_depth=eig_depth,
                dropout=dropout,
            )
        else:
            self.signnet = None

        self.encoder = GINE(
            node_dim=encoder_dim,
            edge_dim=encoder_dim,
            model_dim=encoder_dim,
            model_depth=encoder_depth,
            dropout=dropout,
        )

        self.cov_emb = nn.Sequential(
            nn.Linear(self.covariates_dim, encoder_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim),
        )

        self.attn = nn.Linear(encoder_dim, 1)

        layers = []
        layers.append(nn.Linear(encoder_dim, decoder_dim))
        for _ in range(decoder_depth):
            layers.append(ResBlock(decoder_dim, dropout))
        self.decoder = nn.Sequential(*layers)

        self.isotope_shift = nn.Linear(decoder_dim, self.num_isotope_types)
        self.clf = nn.Linear(decoder_dim, self.vocab_size)

    def get_compile(self, **kwargs):
        if check_pyg_full_compile():
            return th.compile(self, **kwargs)
        else:
            self.compile_submodules(**kwargs)
            return self

    def compile_submodules(self, **kwargs):
        self.onehot = th.compile(self.onehot, **kwargs)
        self.node_emb = th.compile(self.node_emb, **kwargs)
        self.edge_emb = th.compile(self.edge_emb, **kwargs)
        if check_pyg_compile():
            if self.signnet is not None:
                self.signnet = pyg.compile(self.signnet, **kwargs)
            self.encoder = pyg.compile(self.encoder, **kwargs)
        self.cov_emb = th.compile(self.cov_emb, **kwargs)
        self.decoder = th.compile(self.decoder, **kwargs)

    def forward(
        self,
        mol_graff: pyg.data.Data,
        ann_library_mzs: th.Tensor,
        ann_isotope_mask: th.Tensor,
        ann_formula_idxs: th.Tensor,
        ann_isotope_idxs: th.Tensor,
        # ann_dup_idxs: th.Tensor,
        ann_batch_idxs: th.Tensor,
        ann_library_formulae: np.array = None,
        spec_ce: th.Tensor = None,
        spec_ce_batch_idxs: th.Tensor = None,
        **kwargs,
    ):
        batch_size = len(mol_graff.ptr) - 1
        device = mol_graff.x.device

        # embed node, edge, eigenfeatures
        x_atom, x_bond = self.onehot(mol_graff.x, mol_graff.edge_attr)
        x_atom = self.node_emb(x_atom)
        x_bond = self.edge_emb(x_bond)
        if self.num_eigs > 0:
            x_eig = self.signnet(
                mol_graff.eigvecs[:, : self.num_eigs],
                mol_graff.eigvals[:, : self.num_eigs][mol_graff.batch],
            )
        else:
            x_eig = 0

        # run message passing
        x_mol = self.encoder(mol_graff, x_atom + x_eig, x_bond)
        # and attention-pool across atoms
        w = pyg.utils.softmax(self.attn(x_mol), mol_graff.batch)
        z = pyg.nn.global_add_pool(x_mol * w, mol_graff.batch)

        # condition molecule representation on covariates
        isotope_mask = ann_isotope_mask.reshape(batch_size, 1)
        ce = scatter_reduce(
            spec_ce, spec_ce_batch_idxs, reduce="mean", dim_size=batch_size, include_self=False
        )
        ce = ce.reshape(batch_size, 1)
        covariates = th.cat([isotope_mask, ce], dim=1)
        z = z + self.cov_emb(covariates)
        # transform to spectrum representation
        z = self.decoder(z)
        # and predict logits
        log_y_pred = self.clf(z)

        # add the (approximated) isotopic envelope
        log_y_pred = log_y_pred.view(batch_size, self.vocab_size, 1)
        isotope_shift = self.isotope_shift(z).view(batch_size, 1, self.num_isotope_types)
        log_y_pred = log_y_pred + isotope_shift

        # # don't double count probabilities
        # log_y_pred = log_y_pred.flatten()
        # log_y_pred[ann_dup_idxs] = log_y_pred[ann_dup_idxs] - LOG_TWO

        # softmax
        # log_y_pred = log_y_pred.reshape(batch_size,-1)
        # log_y_pred = th.log_softmax(log_y_pred, dim=1)
        log_y_pred = log_y_pred.flatten()
        # aggregate same formula (and isotope)
        # import pdb; pdb.set_trace()
        log_y_pred = scatter_logmeanexp(log_y_pred, ann_formula_idxs)
        # softmax over peaks in the spectrum
        log_y_pred = scatter_logsoftmax(log_y_pred, ann_batch_idxs)

        pred_mzs = ann_library_mzs
        pred_logprobs = log_y_pred
        pred_batch_idxs = ann_batch_idxs
        pred_isotope_idxs = ann_isotope_idxs

        assert pred_mzs.shape[0] == pred_logprobs.shape[0] == pred_isotope_idxs.shape[0], (
            pred_mzs.shape,
            pred_logprobs.shape,
            pred_isotope_idxs.shape,
        )

        # # aggregate logprobs in the same spectrum based on mz
        # pred_mzs_batch_idxs = th.stack([pred_mzs, pred_batch_idxs.float()], dim=1)
        # pred_mzs_batch_idxs, pred_logprobs, pred_formula_idxs = dedup(
        #   pred_mzs_batch_idxs,
        #   *[("lse", pred_logprobs), ("amax", pred_formula_idxs)],
        #   dim=0
        # )
        # pred_mzs = pred_mzs_batch_idxs[:,0]
        # pred_batch_idxs = pred_mzs_batch_idxs[:,1].long()

        assert not th.isinf(th.exp(pred_logprobs)).any(), th.isinf(th.exp(pred_logprobs))

        out_d = {
            "pred_mzs": pred_mzs,
            "pred_logprobs": pred_logprobs,
            "pred_batch_idxs": pred_batch_idxs,
        }

        if self.output_formula_str:
            # convert idxs to formulae
            pred_formula_str = ann_library_formulae
            out_d["pred_isotope_idxs"] = pred_isotope_idxs
            out_d["pred_formula_str"] = pred_formula_str

        return out_d


class SignNet(nn.Module):
    def __init__(self, num_eigs, embed_dim, phi_dim, phi_depth, rho_dim, rho_depth, dropout=0):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_eigs = num_eigs
        self.phi_dim = phi_dim
        self.phi_depth = phi_depth
        self.rho_dim = rho_dim
        self.rho_depth = rho_depth
        self.dropout = dropout

        layers = []
        layers.append(nn.Linear(2, phi_dim))
        for _ in range(phi_depth):
            layers.extend(
                [
                    nn.Linear(phi_dim, phi_dim),
                    nn.SiLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.LayerNorm(phi_dim),
                ]
            )
        self.phi = nn.Sequential(*layers)

        layers = []
        if phi_dim != rho_dim:
            layers.append(nn.Linear(phi_dim, rho_dim))
        for _ in range(rho_depth):
            layers.extend(
                [
                    nn.Linear(rho_dim, rho_dim),
                    nn.SiLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.LayerNorm(rho_dim),
                ]
            )
        if embed_dim != rho_dim:
            layers.append(nn.Linear(rho_dim, embed_dim))
        self.rho = nn.Sequential(*layers)

    def forward(self, v, l):
        x = self.phi(th.stack([v, l], -1)) + self.phi(th.stack([-v, l], -1))
        x = self.rho(x.sum(1))
        return x
