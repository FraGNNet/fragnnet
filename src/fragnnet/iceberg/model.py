import copy

import numpy as np
import torch as th
import torch.nn as nn

# Removed: import torch_scatter as ts  # Now using scatter functions from misc_utils instead
from pyteomics.mass import Composition

import fragnnet.iceberg.common as common
import fragnnet.iceberg.fragmentation as fragmentation
import fragnnet.iceberg.nn_utils as nn_utils
from fragnnet.iceberg.common.chem_utils import vec_to_formula
from fragnnet.utils.data_utils import composition_to_string
from fragnnet.utils.dgl_compat_utils import DGL_AVAILABLE, dgl_nn
from fragnnet.utils.misc_utils import dedup, scatter_logsoftmax, scatter_reduce, scatter_softmax


class IcebergGenModel(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        layers: int = 2,
        set_layers: int = 2,
        dropout: float = 0,
        mpnn_type: str = "GGNN",
        pool_op: str = "avg",
        pe_embed_k: int = 0,
        max_broken: int = fragmentation.FRAGMENT_ENGINE_PARAMS["max_broken_bonds"],
        root_encode: str = "gnn",
        inject_early: bool = False,
        embed_adduct: bool = False,
        encode_forms: bool = False,
        add_hs: bool = False,
    ):
        super().__init__()

        # Check DGL is available
        if not DGL_AVAILABLE:
            raise ImportError(
                "DGL (Deep Graph Library) is required to use Iceberg models. "
                "Please install it with: pip install dgl"
            )
        self.hidden_size = hidden_size
        self.root_encode = root_encode
        self.pe_embed_k = pe_embed_k
        self.embed_adduct = embed_adduct
        self.encode_forms = encode_forms
        self.add_hs = add_hs

        self.formula_in_dim = 0
        if self.encode_forms:
            self.embedder = nn_utils.get_embedder("abs-sines")
            self.formula_dim = common.NORM_VEC.shape[0]
            # Calculate formula dim
            self.formula_in_dim = self.formula_dim * self.embedder.num_dim
            # Account for diffs
            self.formula_in_dim *= 2

        self.pool_op = pool_op
        self.inject_early = inject_early

        self.layers = layers
        self.mpnn_type = mpnn_type
        self.set_layers = set_layers
        self.dropout = dropout

        self.max_broken = max_broken + 1
        self.broken_onehot = th.nn.Parameter(th.eye(self.max_broken))
        self.broken_onehot.requires_grad = False
        self.broken_clamp = max_broken

        edge_feats = fragmentation.MAX_BONDS

        node_feats = pe_embed_k + common.ELEMENT_DIM + common.MAX_H
        orig_node_feats = node_feats
        if self.inject_early:
            node_feats = node_feats + self.hidden_size

        adduct_shift = 0
        if self.embed_adduct:
            adduct_types = len(common.ion2onehot_pos)
            onehot = th.eye(adduct_types)
            self.adduct_embedder = nn.Parameter(onehot.float())
            self.adduct_embedder.requires_grad = False
            adduct_shift = adduct_types

        # Define network
        self.gnn = nn_utils.MoleculeGNN(
            hidden_size=self.hidden_size,
            num_step_message_passing=self.layers,
            set_transform_layers=self.set_layers,
            mpnn_type=self.mpnn_type,
            gnn_node_feats=node_feats + adduct_shift,
            gnn_edge_feats=edge_feats,
            dropout=self.dropout,
        )

        if self.root_encode == "gnn":
            self.root_module = self.gnn

            # if inject early, need separate root and child GNN's
            if self.inject_early:
                self.root_module = nn_utils.MoleculeGNN(
                    hidden_size=self.hidden_size,
                    num_step_message_passing=self.layers,
                    set_transform_layers=self.set_layers,
                    mpnn_type=self.mpnn_type,
                    gnn_node_feats=orig_node_feats + adduct_shift,
                    gnn_edge_feats=edge_feats,
                    dropout=self.dropout,
                )
        elif self.root_encode == "fp":
            self.root_module = nn_utils.MLPBlocks(
                input_size=2048,
                hidden_size=self.hidden_size,
                output_size=None,
                dropout=self.dropout,
                use_residuals=True,
                num_layers=1,
            )
        else:
            raise ValueError()

        # MLP layer to take representations from the pooling layer
        # And predict a single scalar value at each of them
        # I.e., Go from size B x 2h -> B x 1
        self.output_map = nn_utils.MLPBlocks(
            input_size=self.hidden_size * 3 + self.max_broken + self.formula_in_dim,
            hidden_size=self.hidden_size,
            output_size=1,
            dropout=self.dropout,
            num_layers=1,
            use_residuals=True,
        )

        if self.pool_op == "avg":
            self.pool = dgl_nn.AvgPooling()
        elif self.pool_op == "attn":
            self.pool = dgl_nn.GlobalAttentionPooling(nn.Linear(hidden_size, 1))
        else:
            raise NotImplementedError()

        self.sigmoid = nn.Sigmoid()

    def forward(
        self, graphs, root_repr, ind_maps, broken, adducts, root_forms=None, frag_forms=None
    ):
        """forward _summary_

        Args:
            graphs (_type_): _description_
            root_repr (_type_): _description_
            ind_maps (_type_): _description_
            broken (_type_): _description_
            adducts (_type_): _description_
            root_forms (_type_, optional): _description_. Defaults to None.
            frag_forms (_type_, optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        if self.embed_adduct:
            embed_adducts = self.adduct_embedder[adducts.long()]
        if self.root_encode == "fp":
            root_embeddings = self.root_module(root_repr)
            raise NotImplementedError()
        elif self.root_encode == "gnn":
            with root_repr.local_scope():
                if self.embed_adduct:
                    embed_adducts_expand = embed_adducts.repeat_interleave(
                        root_repr.batch_num_nodes(), 0
                    )
                    ndata = root_repr.ndata["h"]
                    ndata = th.cat([ndata, embed_adducts_expand], -1)
                    root_repr.ndata["h"] = ndata
                root_embeddings = self.root_module(root_repr)
                root_embeddings = self.pool(root_repr, root_embeddings)
        else:
            pass

        # Line up the features to be parallel between fragment avgs and root
        # graphs
        ext_root = root_embeddings[ind_maps]

        # Extend the root further to cover each individual atom
        ext_root_atoms = th.repeat_interleave(ext_root, graphs.batch_num_nodes(), dim=0)
        concat_list = [graphs.ndata["h"]]

        if self.inject_early:
            concat_list.append(ext_root_atoms)

        if self.embed_adduct:
            adducts_mapped = embed_adducts[ind_maps]
            adducts_exp = th.repeat_interleave(adducts_mapped, graphs.batch_num_nodes(), dim=0)
            concat_list.append(adducts_exp)

        with graphs.local_scope():
            graphs.ndata["h"] = th.cat(concat_list, -1).float()

            frag_embeddings = self.gnn(graphs)

            # Average embed the full root molecules and fragments
            avg_frags = self.pool(graphs, frag_embeddings)

        # Extend the avg of each fragment
        ext_frag_atoms = th.repeat_interleave(avg_frags, graphs.batch_num_nodes(), dim=0)

        exp_num = graphs.batch_num_nodes()
        # Do the same with the avg fragments

        broken = th.clamp(broken, max=self.broken_clamp)
        ext_frag_broken = th.repeat_interleave(broken, exp_num, dim=0)
        broken_onehots = self.broken_onehot[ext_frag_broken.long()]

        mlp_cat_vec = [
            ext_root_atoms,
            ext_root_atoms - ext_frag_atoms,
            frag_embeddings,
            broken_onehots,
        ]
        if self.encode_forms:
            root_exp = root_forms[ind_maps]
            diffs = root_exp - frag_forms
            form_encodings = self.embedder(frag_forms)
            diff_encodings = self.embedder(diffs)
            form_atom_exp = th.repeat_interleave(form_encodings, exp_num, dim=0)
            diff_atom_exp = th.repeat_interleave(diff_encodings, exp_num, dim=0)

            mlp_cat_vec.extend([form_atom_exp, diff_atom_exp])

        hidden = th.cat(
            mlp_cat_vec,
            dim=1,
        )

        output = self.output_map(hidden)
        output = self.sigmoid(output)
        padded_out = nn_utils.pad_packed_tensor(output, graphs.batch_num_nodes(), 0)
        padded_out = th.squeeze(padded_out, -1)
        return padded_out


class IcebergIntenModel(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        gnn_layers: int = 2,
        mlp_layers: int = 0,
        set_layers: int = 2,
        dropout: float = 0,
        mpnn_type: str = "PNA",
        pool_op: str = "avg",
        node_feats: int = common.ELEMENT_DIM + common.MAX_H,
        pe_embed_k: int = 0,
        max_broken: int = fragmentation.FRAGMENT_ENGINE_PARAMS["max_broken_bonds"],
        frag_set_layers: int = 0,
        root_encode: str = "gnn",
        inject_early: bool = False,
        embed_adduct: bool = False,
        binned_targs: bool = True,
        encode_forms: bool = False,
        add_hs: bool = False,
        mz_bin_res: float = 0.1,
        mz_max: float = 1500.0,
        sum_ints: bool = True,
        output_formula_str: bool = False,
        fix_batch_bug: bool = False,
        **kwargs,
    ):
        super().__init__()

        # Check DGL is available
        if not DGL_AVAILABLE:
            raise ImportError(
                "DGL (Deep Graph Library) is required to use Iceberg models. "
                "Please install it with: pip install dgl"
            )

        self.hidden_size = hidden_size
        self.pe_embed_k = pe_embed_k
        self.root_encode = root_encode
        self.pool_op = pool_op
        self.inject_early = inject_early
        self.embed_adduct = embed_adduct
        self.binned_targs = binned_targs
        self.encode_forms = encode_forms
        self.add_hs = add_hs
        self.output_formula_str = output_formula_str
        self.fix_batch_bug = fix_batch_bug

        self.formula_in_dim = 0
        if self.encode_forms:
            self.embedder = nn_utils.get_embedder("abs-sines")
            self.formula_dim = common.NORM_VEC.shape[0]

            # Calculate formula dim
            self.formula_in_dim = self.formula_dim * self.embedder.num_dim

            # Account for diffs
            self.formula_in_dim *= 2

        self.gnn_layers = gnn_layers
        self.set_layers = set_layers
        self.frag_set_layers = frag_set_layers
        self.mpnn_type = mpnn_type
        self.mlp_layers = mlp_layers
        self.dropout = dropout

        self.max_broken = max_broken + 1
        self.broken_onehot = th.nn.Parameter(th.eye(self.max_broken))
        self.broken_onehot.requires_grad = False
        self.broken_clamp = max_broken

        edge_feats = fragmentation.MAX_BONDS

        orig_node_feats = node_feats
        if self.inject_early:
            node_feats = node_feats + self.hidden_size

        adduct_shift = 0
        if self.embed_adduct:
            adduct_types = len(common.ion2onehot_pos)
            onehot = th.eye(adduct_types)
            self.adduct_embedder = nn.Parameter(onehot.float())
            self.adduct_embedder.requires_grad = False
            adduct_shift = adduct_types

        # Define network
        self.gnn = nn_utils.MoleculeGNN(
            hidden_size=self.hidden_size,
            num_step_message_passing=self.gnn_layers,
            set_transform_layers=self.set_layers,
            mpnn_type=self.mpnn_type,
            gnn_node_feats=node_feats + adduct_shift,
            gnn_edge_feats=edge_feats,
            dropout=self.dropout,
        )

        if self.root_encode == "gnn":
            self.root_module = self.gnn

            # if inject early, need separate root and child GNN's
            if self.inject_early:
                self.root_module = nn_utils.MoleculeGNN(
                    hidden_size=self.hidden_size,
                    num_step_message_passing=self.gnn_layers,
                    set_transform_layers=self.set_layers,
                    mpnn_type=self.mpnn_type,
                    gnn_node_feats=node_feats + adduct_shift,
                    gnn_edge_feats=edge_feats,
                    dropout=self.dropout,
                )
        elif self.root_encode == "fp":
            self.root_module = nn_utils.MLPBlocks(
                input_size=2048,
                hidden_size=self.hidden_size,
                output_size=None,
                dropout=self.dropout,
                num_layers=1,
                use_residuals=True,
            )
        else:
            raise ValueError()

        # MLP layer to take representations from the pooling layer
        # And predict a single scalar value at each of them
        # I.e., Go from size B x 2h -> B x 1
        self.intermediate_out = nn_utils.MLPBlocks(
            input_size=self.hidden_size * 3 + self.max_broken + self.formula_in_dim,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
            num_layers=self.mlp_layers,
            use_residuals=True,
        )

        trans_layer = nn_utils.TransformerEncoderLayer(
            self.hidden_size,
            nhead=8,
            batch_first=True,
            norm_first=False,
            dim_feedforward=self.hidden_size * 4,
        )
        self.trans_layers = nn_utils.get_clones(trans_layer, self.frag_set_layers)

        self.num_outputs = 1
        self.output_activations = [nn.Sigmoid()]
        self.output_size = fragmentation.FRAGMENT_ENGINE_PARAMS["max_broken_bonds"] * 2 + 1
        self.output_map = nn.Linear(self.hidden_size, self.num_outputs * self.output_size)

        # Define map from output layer to attn
        self.isomer_attn_out = copy.deepcopy(self.output_map)

        # Define buckets
        self.mz_bin_res = mz_bin_res
        self.mz_max = mz_max
        buckets = th.DoubleTensor(np.arange(mz_bin_res, mz_max + mz_bin_res, mz_bin_res))
        self.inten_buckets = nn.Parameter(buckets)
        self.inten_buckets.requires_grad = False

        if self.pool_op == "avg":
            self.pool = dgl_nn.AvgPooling()
        elif self.pool_op == "attn":
            self.pool = dgl_nn.GlobalAttentionPooling(nn.Linear(hidden_size, 1))
        else:
            raise NotImplementedError()

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        graphs,
        root_repr,
        ind_maps,
        num_frags,
        broken,
        adducts,
        adduct_form_deltas=None,
        max_add_hs=None,
        max_remove_hs=None,
        masses=None,
        root_forms=None,
        frag_forms=None,
    ):
        """forward _summary_

        Args:
            graphs (_type_): _description_
            root_repr (_type_): _description_
            ind_maps (_type_): _description_
            num_frags (_type_): _description_
            broken (_type_): _description_
            adducts (_type_): _description_
            max_add_hs (_type_, optional): _description_. Defaults to None.
            max_remove_hs (_type_, optional): _description_. Defaults to None.
            masses (_type_, optional): _description_. Defaults to None.
            root_forms (_type_, optional): _description_. Defaults to None.
            frag_forms (_type_, optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        device = num_frags.device

        # if root fingerprints:
        embed_adducts = self.adduct_embedder[adducts.long()]
        if self.root_encode == "fp":
            root_embeddings = self.root_module(root_repr)
            raise NotImplementedError()
        elif self.root_encode == "gnn":
            with root_repr.local_scope():
                if self.embed_adduct:
                    embed_adducts_expand = embed_adducts.repeat_interleave(
                        root_repr.batch_num_nodes(), 0
                    )
                    ndata = root_repr.ndata["h"]
                    ndata = th.cat([ndata, embed_adducts_expand], -1)
                    root_repr.ndata["h"] = ndata
                root_embeddings = self.root_module(root_repr)
                root_embeddings = self.pool(root_repr, root_embeddings)
        else:
            pass

        # Line up the features to be parallel between fragment avgs and root
        # graphs
        ext_root = root_embeddings[ind_maps]
        # Extend the root further to cover each individual atom
        ext_root_atoms = th.repeat_interleave(ext_root, graphs.batch_num_nodes(), dim=0)
        concat_list = [graphs.ndata["h"]]

        if self.inject_early:
            concat_list.append(ext_root_atoms)

        if self.embed_adduct:
            adducts_mapped = embed_adducts[ind_maps]
            adducts_exp = th.repeat_interleave(adducts_mapped, graphs.batch_num_nodes(), dim=0)
            concat_list.append(adducts_exp)

        with graphs.local_scope():
            graphs.ndata["h"] = th.cat(concat_list, -1).float()

            frag_embeddings = self.gnn(graphs)

            # Average embed the full root molecules and fragments
            avg_frags = self.pool(graphs, frag_embeddings)

        # expand broken and map it to each fragment
        broken_arange = th.arange(broken.shape[-1]).to(device)
        broken_mask = broken_arange[None, :] < num_frags[:, None]

        broken = th.clamp(broken[broken_mask], max=self.broken_clamp)
        broken_onehots = self.broken_onehot[broken.long()]

        ### Build hidden with forms
        mlp_cat_list = [ext_root, ext_root - avg_frags, avg_frags, broken_onehots]

        hidden = th.cat(mlp_cat_list, dim=1)

        # Pack so we can use interpeak attn
        padded_hidden = nn_utils.pad_packed_tensor(hidden, num_frags, 0)

        if self.encode_forms:
            diffs = root_forms[:, None, :] - frag_forms
            form_encodings = self.embedder(frag_forms)
            diff_encodings = self.embedder(diffs)
            new_hidden = th.cat([padded_hidden, form_encodings, diff_encodings], dim=-1)
            padded_hidden = new_hidden

        padded_hidden = self.intermediate_out(padded_hidden)
        batch_size, max_frags, hidden_dim = padded_hidden.shape

        # Build up a mask
        arange_frags = th.arange(padded_hidden.shape[1]).to(device)
        attn_mask = ~(arange_frags[None, :] < num_frags[:, None])

        hidden = padded_hidden
        for trans_layer in self.trans_layers:
            hidden, _ = trans_layer(hidden, src_key_padding_mask=attn_mask)

        # hidden: B x L x h
        # attn_mask: B x L

        # Build mask
        max_inten_shift = (self.output_size - 1) / 2
        max_break_ar = th.arange(self.output_size, device=device)[None, None, :].to(device)
        max_breaks_ub = max_add_hs + max_inten_shift
        max_breaks_lb = -max_remove_hs + max_inten_shift

        ub_mask = max_break_ar <= max_breaks_ub[:, :, None]
        lb_mask = max_break_ar >= max_breaks_lb[:, :, None]

        # B x Length x Mass shifts
        valid_pos = th.logical_and(ub_mask, lb_mask)

        # B x L x Output
        output = self.output_map(hidden)
        attn_weights = self.isomer_attn_out(hidden)

        # Mask attn weights
        attn_weights = attn_weights.masked_fill(~valid_pos, -99999)

        # Calc inverse indices => B x Out x L x shift
        inverse_indices = th.bucketize(masses, self.inten_buckets, right=True)
        # inverse_indices = inverse_indices[:, None, :, :].expand(attn_weights.shape)

        # B x Out x (L * Mass shifts)
        attn_weights = attn_weights.reshape(batch_size, -1)
        output = output.reshape(batch_size, -1)
        inverse_indices = inverse_indices.reshape(batch_size, -1)
        valid_pos_binned = valid_pos.reshape(batch_size, -1)

        # B x Outs x ( L * mass shifts )
        # Use scatter_softmax from misc_utils instead of torch_scatter
        # Old: pool_weights = ts.scatter_softmax(attn_weights, index=inverse_indices, dim=-1)
        pool_weights = scatter_softmax(attn_weights, inverse_indices)
        weighted_out = pool_weights * output

        # B x Outs x (UNIQUE(L * mass shifts))
        # Use scatter_reduce with "sum" instead of ts.scatter_add
        # Old: output_binned = ts.scatter_add(weighted_out, index=inverse_indices, dim=-1, dim_size=self.inten_buckets.shape[-1])
        output_binned = scatter_reduce(
            src=weighted_out,
            index=inverse_indices,
            reduce="sum",
            dim=-1,
            dim_size=self.inten_buckets.shape[-1],
            include_self=False,
        )

        output = output.reshape(batch_size, max_frags, -1)
        # pool_weights_reshaped = pool_weights.reshape(
        #   batch_size, max_frags, -1
        # )
        inverse_indices_reshaped = inverse_indices.reshape(batch_size, max_frags, -1)

        # B x Outs x binned
        # Use scatter_reduce with "amax" instead of ts.scatter_max
        # Old: valid_pos_binned = ts.scatter_max((valid_pos_binned).long(), index=inverse_indices, dim_size=self.inten_buckets.shape[-1], dim=-1)[0].bool()
        valid_pos_binned = scatter_reduce(
            src=(valid_pos_binned).long(),
            index=inverse_indices,
            reduce="amax",
            dim=-1,
            dim_size=self.inten_buckets.shape[-1],
            include_self=False,
        ).bool()

        # Activate each dim with its respective output activation
        assert len(self.output_activations) == 1
        output_binned = self.output_activations[0](output_binned)
        # output_binned = th.sigmoid(output_binned)
        output_binned = output_binned.masked_fill(~valid_pos_binned, 0)

        # # Index into output binned using inverse_indices_reshaped
        # # Revert the binned output back to frags for attribution
        # # B x Out x L x Mass shifts
        inverse_indices_reshaped_temp = inverse_indices_reshaped.reshape(batch_size, -1)
        output_unbinned = th.take_along_dim(output_binned, inverse_indices_reshaped_temp, dim=-1)
        output_unbinned = output_unbinned.reshape(batch_size, max_frags, -1)
        # output_unbinned_alpha = output_unbinned * pool_weights_reshaped

        if self.fix_batch_bug:
            # note output_unbinned_alpha won't be correct anymore
            output_binned[:, 0] = 0.0 * output_binned[:, 0]

        # do the stupid inverse
        output_binned_mask = output_binned > 0
        pred_mzs = (
            self.inten_buckets.unsqueeze(0).expand_as(output_binned) - self.mz_bin_res / 2.0
        )[output_binned_mask]
        pred_logprobs = th.log(output_binned[output_binned_mask])
        pred_batch_idxs = th.repeat_interleave(
            th.arange(output_binned.shape[0], device=output_binned.device),
            th.sum(output_binned_mask, dim=1),
            dim=0,
        )
        # normalize
        pred_logprobs = scatter_logsoftmax(pred_logprobs, pred_batch_idxs)

        if self.fix_batch_bug:
            assert th.min(pred_mzs) > 1.0, th.min(pred_mzs)

        out_d = {
            "pred_mzs": pred_mzs,
            "pred_logprobs": pred_logprobs,
            "pred_batch_idxs": pred_batch_idxs,
        }

        if self.binned_targs:
            out_d["pred_spec"] = output_binned

        if self.output_formula_str:
            # get formula stuff
            # note: does not include precursor adduct stuff
            # import pdb; pdb.set_trace()
            assert adduct_form_deltas is not None
            assert th.all((masses == 0).all(2) == (masses == 0).any(2))
            valid_nz_pos = valid_pos & (masses > 0)
            pred_formula_mzs = masses[valid_nz_pos]
            pred_formula_counts = th.sum(valid_nz_pos.long(), dim=2)
            pred_formula_vecs = th.repeat_interleave(
                frag_forms.reshape(-1, frag_forms.shape[2]), pred_formula_counts.flatten(), dim=0
            )
            _, pred_h_h_idxs = th.nonzero(
                valid_nz_pos.reshape(-1, valid_nz_pos.shape[2]), as_tuple=True
            )
            pred_h_deltas = (
                th.arange(valid_nz_pos.shape[2], device=valid_nz_pos.device)
                - valid_nz_pos.shape[2] // 2
            )
            pred_formula_vecs[:, common.element_to_ind["H"]] += pred_h_deltas[pred_h_h_idxs]
            assert th.all(pred_formula_vecs >= 0)
            assert th.all(pred_formula_vecs.any(1))
            pred_formula_batch_idxs = th.repeat_interleave(
                th.arange(frag_forms.shape[0], device=frag_forms.device),
                pred_formula_counts.sum(dim=1),
                dim=0,
            )
            # dedup
            pred_b_formula_vecs = th.cat(
                [pred_formula_batch_idxs.unsqueeze(1), pred_formula_vecs], dim=1
            )
            pred_b_formula_vecs, pred_formula_mzs, pred_formula_batch_idxs = dedup(
                pred_b_formula_vecs,
                *[("amax", pred_formula_mzs), ("amax", pred_formula_batch_idxs)],
                dim=0,
            )
            pred_formula_vecs = pred_b_formula_vecs[:, 1:]
            # account for adducts
            pred_formula_vecs += adduct_form_deltas[pred_formula_batch_idxs]
            # convert formula vecs to strings (this is slow!)
            pred_formula_str = [
                composition_to_string(
                    Composition(formula=vec_to_formula(pred_formula_vec.squeeze(0)))
                )
                for pred_formula_vec in pred_formula_vecs.cpu().split(1, dim=0)
            ]
            pred_formula_str = np.array(pred_formula_str)

            out_d["pred_formula_mzs"] = pred_formula_mzs
            out_d["pred_formula_str"] = pred_formula_str
            out_d["pred_formula_batch_idxs"] = pred_formula_batch_idxs

        return out_d
