"""Tests for multi-location CE injection (ce_insert_location = "mol,mlp").

The model dispatches CE insertion on membership in a parsed set of locations. The key
regression guarded here is the silent-drop bug: before multi-location support, the code used
exact string matches (== "mol" / == "mlp"), so a comma-separated "mol,mlp" matched NEITHER
branch and dropped CE entirely (ce_mol_input_dim == ce_mlp_input_dim == 0). These tests assert
that each location string produces the correct nonzero input dims at the mol encoder and/or the
per-fragment scoring MLP, so a future regression that stops parsing the combo fails loudly.
"""

from __future__ import annotations

import pytest

from fragnnet.model.fragnnet_model import FraGNNetModel


def _build(ce_insert_location: str = "mol", **overrides) -> FraGNNetModel:
    """Smallest valid FraGNNetModel with the given ce_insert_location.

    Extra keyword args override the default FraGNNetModel kwargs (e.g. context_embedding=True),
    so sibling test modules can reuse this builder.
    """
    kwargs = dict(
        num_depth=4,
        num_hs=2,
        num_elements=17,
        int_embedder="fourier",
        int_embedder_tight=True,
        mol_node_feats=["a_onehot", "a_degree", "a_ring", "a_mass"],
        mol_edge_feats=["b_degree"],
        mol_pe_embed_k=0,
        mol_hidden_size=32,
        mol_num_layers=2,
        mol_gnn_type="MPNN",
        mol_normalization="layer",
        mol_dropout=0.0,
        mol_num_heads=1,
        mol_pool_type="mean",
        frag_node_feats=["cc", "base_formula", "depth", "h_formulae_idx"],
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
        ce_insert_location=ce_insert_location,
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
    )
    kwargs["ce_insert_location"] = ce_insert_location
    kwargs.update(overrides)
    return FraGNNetModel(**kwargs)

# NOTE: only the shared `_build` helper is carried over to this branch. The CE-location,
# candidate-ion, ion-channel, fragment-competition and siamese tests that lived here
# belong to feiw_v2 model work this branch does not have;
# tests/test_fragnnet_audit_fixes.py imports `_build` from this module.
