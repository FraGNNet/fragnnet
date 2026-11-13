"""Tests for out_degree, in_degree, and num_heavy_atoms fragment DAG node features.

These three features are computed on-the-fly in get_frag_graph from stored
edge_index / base_formula — they are NOT stored in dag.x. Tests therefore route
through get_frag_graph for any of these three features.

Covers:
1. NODE_FEAT_TO_IDX contains the new entries at expected indices.
2. Degree values match edge_index from the stored DAG.
3. Root always has in_degree 0; leaves always have out_degree 0.
4. num_heavy_atoms matches sum of non-H element counts from base_formula.
5. Feature shapes, dtypes, and non-negativity.
6. Model forward pass runs with the new features enabled.
"""

from __future__ import annotations

import pytest
import torch as th

from fragnnet.utils.frag_utils import (
    ELEMENT_TO_IDX,
    NODE_FEAT_TO_IDX,
    compute_dags,
    extract_mol_info,
    get_node_feats,
)
from fragnnet.utils.feat_utils import get_frag_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_dag(smiles: str, max_depth: int = 3) -> dict:
    mol_d = extract_mol_info(smiles)
    return compute_dags(
        mol_d,
        max_depth=max_depth,
        h_prior=True,
        max_h_transfer=2,
        frag_max_time=int(1e6),
        isotopes=False,
        nb_isomorphic=False,
        wl_max_iterations=-1,
        multi_cut_bfs=False,
    )


def _get_stored_feat(dag_d: dict, feat: str) -> th.Tensor:
    """Read a feature that IS stored in dag.x."""
    dag = dag_d["dag"]
    node_feat_idxs = dag.node_feat_idxs[0]
    return get_node_feats(dag.x, node_feat_idxs, feat)


def _get_otf_feat(dag_d: dict, feat: str) -> th.Tensor:
    """Read an on-the-fly feature via get_frag_graph (out_degree / in_degree / num_heavy_atoms)."""
    frag_pyg = get_frag_graph(
        dag_d["dag"],
        frag_node_feats=[feat],
        frag_edge_feats=[],
        edges=False,
        bigraph=False,
    )
    return get_node_feats(frag_pyg.x, frag_pyg.node_feat_idxs[0], feat)


# ---------------------------------------------------------------------------
# 1. NODE_FEAT_TO_IDX contains the new entries at expected indices
# ---------------------------------------------------------------------------


def test_node_feat_to_idx_has_new_entries() -> None:
    assert NODE_FEAT_TO_IDX["out_degree"] == 7
    assert NODE_FEAT_TO_IDX["in_degree"] == 8
    assert NODE_FEAT_TO_IDX["num_heavy_atoms"] == 9


# ---------------------------------------------------------------------------
# 2. Degree computation from a known topology
# ---------------------------------------------------------------------------


def test_degrees_linear_chain() -> None:
    """propanoic acid: CC(=O)O — small molecule with a clear chain DAG."""
    dag_d = _run_dag("CC(=O)O")
    dag = dag_d["dag"]
    edge_index = dag.edge_index  # shape (2, num_edges): (src, dst)

    num_nodes = dag_d["dag_num_nodes"]
    out_feat = _get_otf_feat(dag_d, "out_degree").reshape(-1)
    in_feat = _get_otf_feat(dag_d, "in_degree").reshape(-1)

    # recompute expected degrees from edge_index
    expected_out = th.zeros(num_nodes, dtype=th.long)
    expected_in = th.zeros(num_nodes, dtype=th.long)
    if edge_index.shape[1] > 0:
        for i in range(edge_index.shape[1]):
            expected_out[edge_index[0, i]] += 1
            expected_in[edge_index[1, i]] += 1

    assert th.all(out_feat == expected_out), f"out_degree mismatch: {out_feat} vs {expected_out}"
    assert th.all(in_feat == expected_in), f"in_degree mismatch: {in_feat} vs {expected_in}"


def test_root_has_zero_in_degree() -> None:
    """Root node (precursor) always has in_degree == 0."""
    dag_d = _run_dag("c1ccccc1")  # benzene
    in_feat = _get_otf_feat(dag_d, "in_degree").reshape(-1)
    depth_feat = _get_stored_feat(dag_d, "depth")
    root_mask = depth_feat[:, 0].bool()
    assert root_mask.sum() == 1, "Expected exactly one root node"
    root_idx = root_mask.nonzero(as_tuple=True)[0][0]
    assert in_feat[root_idx].item() == 0, "Root node must have in_degree == 0"


def test_leaves_have_zero_out_degree() -> None:
    """Leaf nodes (terminal fragments) always have out_degree == 0."""
    dag_d = _run_dag("CC(=O)O")
    out_feat = _get_otf_feat(dag_d, "out_degree").reshape(-1)
    edge_index = dag_d["dag"].edge_index
    num_nodes = dag_d["dag_num_nodes"]

    nodes_with_children = set()
    if edge_index.shape[1] > 0:
        nodes_with_children = set(edge_index[0].tolist())
    leaf_mask = th.tensor(
        [i not in nodes_with_children for i in range(num_nodes)], dtype=th.bool
    )
    assert th.all(out_feat[leaf_mask] == 0), "All leaf nodes must have out_degree == 0"


# ---------------------------------------------------------------------------
# 3. Single-bond molecule (root degrees)
# ---------------------------------------------------------------------------


def test_single_bond_molecule_root_degrees() -> None:
    """Methanol (CO) has one fragmentable bond; root in_degree must be 0."""
    dag_d = _run_dag("CO", max_depth=1)
    out_feat = _get_otf_feat(dag_d, "out_degree").reshape(-1)
    in_feat = _get_otf_feat(dag_d, "in_degree").reshape(-1)
    depth_feat = _get_stored_feat(dag_d, "depth")
    root_idx = depth_feat[:, 0].bool().nonzero(as_tuple=True)[0][0]
    assert in_feat[root_idx].item() == 0, "Root in_degree must be 0"
    assert out_feat[root_idx].item() >= 0


# ---------------------------------------------------------------------------
# 4. num_heavy_atoms excludes H and matches element sum
# ---------------------------------------------------------------------------


def test_num_heavy_atoms_excludes_hydrogen() -> None:
    """num_heavy_atoms is the sum of all element counts except H."""
    dag_d = _run_dag("CCO")  # ethanol: C2H6O → root has 3 heavy atoms
    dag = dag_d["dag"]
    node_feat_idxs = dag.node_feat_idxs[0]

    heavy_feat = _get_otf_feat(dag_d, "num_heavy_atoms").reshape(-1)
    base_formula_feat = get_node_feats(dag.x, node_feat_idxs, "base_formula")

    h_col = ELEMENT_TO_IDX["H"]
    expected_heavy = base_formula_feat.sum(dim=1) - base_formula_feat[:, h_col]
    assert th.all(heavy_feat == expected_heavy), (
        f"num_heavy_atoms mismatch:\n{heavy_feat}\nvs\n{expected_heavy}"
    )


def test_num_heavy_atoms_root_known_value() -> None:
    """Root node of ethanol (CCO) should have num_heavy_atoms == 3 (C, C, O)."""
    dag_d = _run_dag("CCO")
    heavy_feat = _get_otf_feat(dag_d, "num_heavy_atoms").reshape(-1)
    depth_feat = _get_stored_feat(dag_d, "depth")
    root_idx = depth_feat[:, 0].bool().nonzero(as_tuple=True)[0][0]
    assert heavy_feat[root_idx].item() == 3


def test_num_heavy_atoms_all_positive() -> None:
    """Every fragment node must have at least 1 heavy atom."""
    dag_d = _run_dag("c1ccccc1C(=O)O")  # benzoic acid
    heavy_feat = _get_otf_feat(dag_d, "num_heavy_atoms").reshape(-1)
    assert th.all(heavy_feat >= 1), f"Found fragment with 0 heavy atoms: {heavy_feat}"


# ---------------------------------------------------------------------------
# 5. Feature shape, dtype, and sign
# ---------------------------------------------------------------------------


def test_feature_shapes_via_frag_graph() -> None:
    """out_degree, in_degree, num_heavy_atoms each produce shape (num_nodes, 1)."""
    dag_d = _run_dag("CC(=O)O")
    num_nodes = dag_d["dag_num_nodes"]
    for feat in ("out_degree", "in_degree", "num_heavy_atoms"):
        t = _get_otf_feat(dag_d, feat)
        assert t.shape == (num_nodes, 1), f"{feat} shape {t.shape} != ({num_nodes}, 1)"


def test_feature_dtype() -> None:
    """New features should be int64."""
    dag_d = _run_dag("CCO")
    for feat in ("out_degree", "in_degree", "num_heavy_atoms"):
        t = _get_otf_feat(dag_d, feat)
        assert t.dtype == th.int64, f"{feat} dtype {t.dtype} != int64"


def test_feature_non_negative() -> None:
    """All three scalar features must be >= 0."""
    dag_d = _run_dag("c1ccccc1")
    for feat in ("out_degree", "in_degree", "num_heavy_atoms"):
        t = _get_otf_feat(dag_d, feat)
        assert th.all(t >= 0), f"{feat} has negative values: {t}"


# ---------------------------------------------------------------------------
# 6. Model forward pass with new features
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_model():
    """Instantiate the smallest valid FraGNNetModel with the 3 new frag features."""
    from fragnnet.model.fragnnet_model import FraGNNetModel

    return FraGNNetModel(
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
        frag_node_feats=["cc", "base_formula", "depth", "h_formulae_idx", "out_degree", "in_degree", "num_heavy_atoms"],
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
    )


def test_model_forward_with_new_frag_feats(minimal_model) -> None:
    """Forward pass must complete without error and produce finite output."""
    import torch_geometric as pyg

    from fragnnet.utils.data_utils import mol_from_smiles
    from fragnnet.utils.feat_utils import get_mol_graph

    smiles = "CCO"
    mol = mol_from_smiles(smiles)
    mol_pyg = get_mol_graph(
        mol,
        atom_feats=["a_onehot", "a_degree", "a_ring", "a_mass"],
        bond_feats=["b_degree"],
        pe_embed_k=0,
        bigraph=True,
    )

    dag_d = _run_dag(smiles)
    frag_pyg = get_frag_graph(
        dag_d["dag"],
        frag_node_feats=["cc", "base_formula", "depth", "h_formulae_idx", "out_degree", "in_degree", "num_heavy_atoms"],
        frag_edge_feats=[],
        edges=False,
        bigraph=False,
    )

    num_formulae = len(dag_d["idx_to_formula"])
    formula_peak_mzs = dag_d["formula_peak_mzs"]
    formula_peak_probs = dag_d["formula_peak_probs"]

    nz = th.nonzero(formula_peak_probs)
    peak_idxs = nz[:, 0]
    peak_mzs = formula_peak_mzs[nz[:, 0], nz[:, 1]]
    peak_probs = formula_peak_probs[nz[:, 0], nz[:, 1]]

    minimal_model.eval()
    with th.no_grad():
        out = minimal_model(
            mol_pyg=pyg.data.Batch.from_data_list([mol_pyg]),
            frag_pyg=pyg.data.Batch.from_data_list([frag_pyg]),
            mol_num_nodes=th.tensor([0, mol_pyg.num_nodes]),
            frag_num_nodes=th.tensor([0, frag_pyg.num_nodes]),
            frag_formula_peak_idxs=peak_idxs,
            frag_formula_peak_mzs=peak_mzs,
            frag_formula_peak_probs=peak_probs,
            frag_formula_sizes=th.tensor([num_formulae]),
            frag_formula_cumsizes=th.tensor([0, num_formulae]),
            frag_formula_peak_sizes=th.tensor([peak_idxs.shape[0]]),
            spec_ce=th.tensor([30.0]),
            spec_ce_batch_idxs=th.tensor([0]),
            spec_prec_type=th.tensor([0]),
            spec_inst_type=th.tensor([0]),
            spec_frag_mode=th.tensor([0]),
            boundary_pair_frag_idxs=dag_d["dag"].boundary_pair_frag_idxs,
            boundary_pair_in_idxs=dag_d["dag"].boundary_pair_in_local,
            boundary_pair_out_idxs=dag_d["dag"].boundary_pair_out_local,
        )

    assert out is not None
    logprobs = out if isinstance(out, th.Tensor) else out.get("logprobs", next(iter(out.values())))
    assert th.isfinite(logprobs).all(), "Model output contains NaN or Inf"
