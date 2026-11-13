"""Tests for cc_atom_dropout: training-only per-atom dropout on the fragment cc mask.

Each atom is independently dropped from a fragment's atom-membership set during training, so the
pooled ``cc`` embedding sees a random subset of the fragment's atoms. This regularizes against
memorizing exact atom sets (aimed at the train->val gap), and is much finer-grained than
``cc_feature_dropout``, which zeroes the whole cc group for the whole batch on one scalar draw.

Invariants under test:
- eval() is unaffected (dropout is training-only) and deterministic;
- train() with p>0 perturbs the prediction, and different draws give different predictions;
- p=0 is a no-op (train output == eval output);
- p=1.0 does not produce NaN/inf -- every fragment is restored rather than emptied.
"""

from __future__ import annotations

import pytest
import torch as th
import torch_geometric as pyg

from fragnnet.model.fragnnet_model import FraGNNetModel
from fragnnet.utils.data_utils import mol_from_smiles
from fragnnet.utils.feat_utils import get_frag_graph, get_mol_graph
from fragnnet.utils.frag_utils import compute_dags, extract_mol_info

MOL_NODE_FEATS = ["a_onehot", "a_degree", "a_ring", "a_mass"]
MOL_EDGE_FEATS = ["b_degree"]
FRAG_NODE_FEATS = ["cc", "base_formula", "depth", "h_formulae_idx"]


def _build_model(
    cc_atom_dropout: float,
    cc_channel_dropout: float = 0.0,
    frag_node_feats: list[str] | None = None,
    candidate_ion_head: bool = False,
    ion_channel_factorized: bool = False,
    ion_channel_adduct_gated: bool = False,
    frag_competition: bool = False,
) -> FraGNNetModel:
    if frag_node_feats is None:
        frag_node_feats = FRAG_NODE_FEATS
    return FraGNNetModel(
        num_depth=4,
        num_hs=2,
        num_elements=17,
        int_embedder="fourier",
        int_embedder_tight=True,
        mol_node_feats=MOL_NODE_FEATS,
        mol_edge_feats=MOL_EDGE_FEATS,
        mol_pe_embed_k=0,
        mol_hidden_size=32,
        mol_num_layers=2,
        mol_gnn_type="MPNN",
        mol_normalization="layer",
        mol_dropout=0.0,
        mol_num_heads=1,
        mol_pool_type="mean",
        frag_node_feats=frag_node_feats,
        frag_edge_feats=[],
        frag_hidden_size=32,
        frag_num_layers=2,
        frag_gnn_type="NodeMLP",
        frag_normalization="layer",
        frag_dropout=0.0,
        frag_pool_type="max",
        frag_embed_combine="cat",
        frag_pool_combine="none",
        mlp_output_format="formula",
        mlp_hidden_size=32,
        mlp_normalization="layer",
        mlp_dropout=0.0,
        mlp_num_layers=2,
        mlp_use_residuals=True,
        cc_interstage_type="add",
        cc_interstage_use_rest=False,
        nb_iso=False,
        skip_edge_loss=True,
        mask_null_formula=False,
        predict_oos=False,
        bin_output=False,
        mz_bin_res=0.01,
        mz_max=1500.0,
        ce_insert_location="mol",
        ce_insert_type="embed_clamped",
        ce_insert_merge=True,
        ce_insert_size=8,
        nce_max=200.0,
        nce_mean=30.0,
        nce_std=20.0,
        prec_insert_location="mol",
        prec_insert_size=8,
        prec_types=["[M+H]+"],
        inst_insert_location="mol",
        inst_insert_size=8,
        inst_types=["FT"],
        frag_mode_insert_location="mol",
        frag_mode_insert_size=8,
        frag_modes=["HCD"],
        frag_mode_scale=False,
        ce_scale=False,
        ce_scaler_hidden_dim=16,
        adduct_scale=False,
        stratum_scale=False,
        stratum_scaler_embed_dim=8,
        output_formula_str=False,
        use_nce=True,
        use_ace=False,
        ace_max=50.0,
        ace_mean=20.0,
        ace_std=10.0,
        cc_feature_dropout=0.0,
        cc_atom_dropout=cc_atom_dropout,
        cc_channel_dropout=cc_channel_dropout,
        candidate_ion_head=candidate_ion_head,
        ion_channel_factorized=ion_channel_factorized,
        ion_channel_adduct_gated=ion_channel_adduct_gated,
        ion_channel_rank=4,
        frag_competition=frag_competition,
        frag_competition_top_k=8,
        frag_competition_hidden_dim=16,
        frag_competition_layers=1,
        frag_competition_heads=4,
        frag_competition_dropout=0.0,
        frag_competition_detach_encoder=True,
        frag_competition_max_delta=5.0,
    )


def _forward_kwargs(
    smiles: str = "CCOC(=O)c1ccccc1",
    frag_node_feats: list[str] | None = None,
) -> dict:
    if frag_node_feats is None:
        frag_node_feats = FRAG_NODE_FEATS
    mol = mol_from_smiles(smiles)
    mol_pyg = get_mol_graph(
        mol, atom_feats=MOL_NODE_FEATS, bond_feats=MOL_EDGE_FEATS, pe_embed_k=0, bigraph=True
    )
    mol_d = extract_mol_info(smiles)
    dag_d = compute_dags(
        mol_d,
        max_depth=3,  # must match model num_depth=4 (depth-3 frags + root)
        h_prior=True,
        max_h_transfer=2,
        frag_max_time=int(1e6),
        isotopes=False,
        nb_isomorphic=False,
        wl_max_iterations=-1,
        multi_cut_bfs=False,
    )
    frag_pyg = get_frag_graph(
        dag_d["dag"],
        frag_node_feats=frag_node_feats,
        frag_edge_feats=[],
        edges=False,
        bigraph=False,
    )
    num_formulae = len(dag_d["idx_to_formula"])
    nz = th.nonzero(dag_d["formula_peak_probs"])
    peak_idxs = nz[:, 0]
    return {
        "mol_pyg": pyg.data.Batch.from_data_list([mol_pyg]),
        "frag_pyg": pyg.data.Batch.from_data_list([frag_pyg]),
        "mol_num_nodes": th.tensor([0, mol_pyg.num_nodes]),
        "frag_num_nodes": th.tensor([0, frag_pyg.num_nodes]),
        "frag_formula_peak_idxs": peak_idxs,
        "frag_formula_peak_mzs": dag_d["formula_peak_mzs"][nz[:, 0], nz[:, 1]],
        "frag_formula_peak_probs": dag_d["formula_peak_probs"][nz[:, 0], nz[:, 1]],
        "frag_formula_sizes": th.tensor([num_formulae]),
        "frag_formula_cumsizes": th.tensor([0, num_formulae]),
        "frag_formula_peak_sizes": th.tensor([peak_idxs.shape[0]]),
        "spec_ce": th.tensor([30.0]),
        "spec_ce_batch_idxs": th.tensor([0]),
        "spec_prec_type": th.tensor([0]),
        "spec_inst_type": th.tensor([0]),
        "spec_frag_mode": th.tensor([0]),
    }

# NOTE: only the shared builders are carried over to this branch. The cc_atom_dropout,
# candidate-ion, ion-channel and fragment-competition tests that lived here belong to
# feiw_v2 model work this branch does not have; tests/test_fragnnet_audit_fixes.py
# imports `_forward_kwargs` from this module, so the builders stay.
