"""Behavioral regressions for FraGNNet output heads and CMF labels."""

import random

import numpy as np
import pytest
import torch as th
from pyteomics.mass import Composition
from test_cc_atom_dropout import _forward_kwargs
from test_ce_multi_location import _build

from fragnnet.dataset.spec_mol_frag_dataset import SpecMolFragDataset
from fragnnet.utils.feat_utils import batch_mols_frags, get_mol_graph
from fragnnet.utils.formula_utils import (
    ELECTRON_MASS,
    PERIODIC_TABLE,
    PREC_TYPE_TO_CMF_FORMULA_DIFF,
    PREC_TYPE_TO_CMF_MASS_DIFF,
    parse_formula,
)
from fragnnet.utils.frag_utils import compute_dags, extract_mol_info


@pytest.fixture(autouse=True)
def seeded_model() -> None:
    random.seed(0)
    np.random.seed(0)  # noqa: NPY002 -- seed the global RNG used by library code.
    th.manual_seed(0)


@pytest.mark.parametrize("prec", list(PREC_TYPE_TO_CMF_MASS_DIFF))
def test_cmf_composition_changes_match_existing_mass_changes(prec: str) -> None:
    composition = Composition(PREC_TYPE_TO_CMF_FORMULA_DIFF[prec])
    mass = sum(PERIODIC_TABLE.GetMostCommonIsotopeMass(e) * n for e, n in composition.items())
    charge = -1 if prec.endswith("-") else 1
    assert mass - charge * ELECTRON_MASS == pytest.approx(PREC_TYPE_TO_CMF_MASS_DIFF[prec])


# NOTE: the "mol,mlp" (multi-location CE) and context_embedding cases from feiw_v2 are
# omitted; neither model option exists on this branch.
@pytest.mark.parametrize(
    "options",
    [
        {"frag_embed_combine": "avg"},
    ],
)
def test_node_formula_readout_width(options: dict) -> None:
    model = _build(mlp_output_format="node_formula", **options)
    data = _forward_kwargs("CCCO", frag_node_feats=model.frag_node_feats)
    out = model(**data)
    assert th.isfinite(out["pred_logprobs"]).all()
    loss = -out["pred_logprobs"].mean()
    loss.backward()
    assert model.node_module.input_layer.weight.grad is not None
    assert th.isfinite(model.node_module.input_layer.weight.grad).all()


# NOTE: the depth_ce_bias case from feiw_v2 is omitted; that model option does not
# exist on this branch.
@pytest.mark.parametrize("feature", ["ce_scale"])
def test_node_formula_conditioning_changes_predictions_and_receives_gradients(feature: str) -> None:
    model = _build(mlp_output_format="node_formula", **{feature: True}).eval()
    data = _forward_kwargs("CCCO", frag_node_feats=model.frag_node_feats)
    data["spec_nce_stats"] = th.tensor([[30.0, 0.0, 1.0]])
    before = model(**data)["pred_logprobs"].detach()
    layer = model.ce_scaler.net[-1] if feature == "ce_scale" else model.depth_ce_bias_layer
    with th.no_grad():
        # Shift one channel or depth, so normalization cannot cancel the intervention.
        index = model.mlp_output_dim if feature == "ce_scale" else 1
        layer.bias[index] = 2.0
    after = model(**data)["pred_logprobs"]
    assert not th.allclose(before, after)
    (-after.mean()).backward()
    assert layer.weight.grad is not None
    assert th.isfinite(layer.weight.grad).all()
    assert layer.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("include_cmf", [False, True])
def test_formula_labels_match_masses_in_mixed_adduct_batch(include_cmf: bool) -> None:
    prec_types = ["[M+H]+", "[M-H]-", "[M+Na]+", "[M]+"]
    feats = ["cc", "base_formula", "depth", "h_formulae_idx"]
    if include_cmf:
        feats.append("cmf_h_formulae_idx")
    model = _build(prec_types=prec_types, frag_node_feats=feats, output_formula_str=True).eval()
    md = extract_mol_info("CCCO")
    dag = compute_dags(md, 3, True, 2, 100)
    dataset = SpecMolFragDataset.__new__(SpecMolFragDataset)
    dataset.frag_params = {
        "pyg": True,
        "pyg_node_feats": feats,
        "pyg_edge_feats": [],
        "pyg_edges": False,
        "pyg_bigraph": False,
        "formula_peak_probs": True,
        "formula_peak_mzs": True,
        "num_isotopes": 1,
        "formula_str": True,
        "include_cmf": include_cmf,
    }
    fragments = []
    for prec in prec_types:
        frag = dataset._process_neutral_frag(dag)
        fragments.append(
            dataset._apply_fragment_mz_change(
                frag,
                prec,
                frag["frag_formula_peak_mzs"],
                clone_pyg=True,
            )
        )
    mol = get_mol_graph(md["mol"], model.mol_node_feats, model.mol_edge_feats, 0, True)
    data = batch_mols_frags(
        [mol] * len(prec_types),
        [f["frag_pyg"] for f in fragments],
        [f["frag_formula_peak_mzs"] for f in fragments],
        [f["frag_formula_peak_probs"] for f in fragments],
    )
    data.update(
        spec_ce=th.full((4,), 30.0),
        spec_ce_batch_idxs=th.arange(4),
        spec_prec_type=th.arange(4),
        spec_inst_type=th.zeros(4, dtype=th.long),
        spec_frag_mode=th.zeros(4, dtype=th.long),
        spec_prec_type_str=np.array(prec_types),
        frag_formula_str=np.concatenate([f["frag_formula_str"] for f in fragments]),
    )
    with th.no_grad():
        out = model(**data)
        model.output_formula_str = False
        without_labels = model(**data)
    assert th.equal(out["pred_logprobs"], without_labels["pred_logprobs"])
    assert th.equal(out["pred_mzs"], without_labels["pred_mzs"])
    offsets = data["frag_formula_cumsizes"][:-1][out["pred_batch_idxs"]]
    formula_idxs = data["frag_formula_peak_idxs"] + offsets
    for peak, formula_idx, batch_idx in zip(out["pred_mzs"], formula_idxs, out["pred_batch_idxs"]):
        formula = out["pred_formula_str"][formula_idx]
        mass = sum(
            PERIODIC_TABLE.GetMostCommonIsotopeMass(e) * n
            for e, n in parse_formula(formula).items()
        )
        charge = -1 if prec_types[batch_idx].endswith("-") else 1
        assert float(peak) == pytest.approx(mass - charge * ELECTRON_MASS, abs=1e-5)
    assert all(out["pred_formula_str"][i] == "" for i in data["frag_formula_cumsizes"][:-1])
