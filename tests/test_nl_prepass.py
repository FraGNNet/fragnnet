"""Tests for NeutralLossRule prepass (SO₂ extrusion, SO₃ loss, N₂O extrusion, CO loss).

Verifies that:
- NL SMARTS patterns compile and match expected molecules.
- _apply_nl_prepass returns product masks with the correct atom count.
- _inject_nl_masks_into_dag adds new nodes only when the formula is absent from BFS.
- smarts_prepass=True in compute_dags correctly includes NL products.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from fragnnet.frag.smarts_prepass import NL_RULES, _apply_nl_prepass
from fragnnet.utils.frag_utils import (
    _inject_nl_masks_into_dag,
    compute_dags,
    extract_mol_info,
    get_fraggen_input_arrays,
    get_node_feats,
    th_long_to_mask,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nl_rules_by_name() -> dict:
    return {r.name: r for r in NL_RULES}


@pytest.mark.parametrize("multi_cut_bfs", [False, True])
@pytest.mark.parametrize("smarts_prepass", [False, True])
def test_so2_product_in_both_bfs_modes(multi_cut_bfs: bool, smarts_prepass: bool) -> None:
    mol_d = extract_mol_info("O=[N+]([O-])c1ccc(S(=O)(=O)Nc2ncccn2)cc1")
    mol = mol_d["mol"]
    product = next(
        mask for mask, idx in _apply_nl_prepass(NL_RULES, mol, mol.GetNumAtoms())
        if NL_RULES[idx].name == "aryl_sulfonamide_so2"
    )
    result = compute_dags(
        mol_d, 3, False, 4, 100,
        multi_cut_bfs=multi_cut_bfs, smarts_prepass=smarts_prepass,
    )
    dag = result["dag"]
    masks = th_long_to_mask(get_node_feats(dag.x, dag.node_feat_idxs[0], "cc"))
    found = np.flatnonzero((masks[:, :mol.GetNumAtoms()].numpy() == product).all(axis=1))
    # Multi-cut BFS reaches this atom mask without the rule: cutting both S bonds and
    # merging the two resulting components drops exactly S+2O (a depth-3 node with two
    # parents). The rule also forms the new Ar-N bond, which the cc mask does not record,
    # so the masks are indistinguishable and the prepass dedups onto the existing node.
    # Only single-cut BFS needs the rule to reach this product at all.
    assert len(found) == int(smarts_prepass or multi_cut_bfs)
    if smarts_prepass:
        assert ((dag.edge_index[0] == 0) & (dag.edge_index[1] == found[0])).any()


def _run_dag(smiles: str, smarts_prepass: bool) -> dict:
    mol_d = extract_mol_info(smiles)
    return compute_dags(
        mol_d,
        max_depth=3,
        h_prior=False,
        max_h_transfer=4,
        frag_max_time=int(1e6),
        multi_cut_bfs=False,
        smarts_prepass=smarts_prepass,
    )


# ---------------------------------------------------------------------------
# 1. SMARTS compilation and basic matching
# ---------------------------------------------------------------------------


class TestNLRulesCompile:
    def test_all_rules_compile(self):
        """All NL_RULES SMARTS patterns must parse without error."""
        for rule in NL_RULES:
            patt = Chem.MolFromSmarts(rule.reactant_smarts)
            assert patt is not None, f"Rule {rule.name!r} SMARTS failed to compile"

    def test_aryl_sulfonamide_matches_sulfadiazine(self):
        """Aromatic sulfonamide pattern matches N-aryl sulfonamide."""
        mol = Chem.MolFromSmiles("O=[N+]([O-])c1ccc(S(=O)(=O)Nc2ncccn2)cc1")
        results = _apply_nl_prepass(NL_RULES, mol, mol.GetNumAtoms())
        names = [NL_RULES[ridx].name for _, ridx in results]
        assert "aryl_sulfonamide_so2" in names

    def test_aryl_sulfonamide_no_match_aliphatic(self):
        """Aliphatic sulfonamide must NOT match aryl_sulfonamide_so2 (requires [c])."""
        mol = Chem.MolFromSmiles("CS(=O)(=O)NC")  # methanesulfonamide, no aromatic C–S
        results = _apply_nl_prepass(NL_RULES, mol, mol.GetNumAtoms())
        names = [NL_RULES[ridx].name for _, ridx in results]
        assert "aryl_sulfonamide_so2" not in names

    def test_aryl_sulfonate_matches_pyrene_sulfonic_acid(self):
        """Sulfonate pattern matches aromatic sulfonic acid."""
        mol = Chem.MolFromSmiles("O=S(=O)(O)c1ccc2ccc3cccc4ccc1c2c34")
        results = _apply_nl_prepass(NL_RULES, mol, mol.GetNumAtoms())
        names = [NL_RULES[ridx].name for _, ridx in results]
        assert "aryl_sulfonate_so2" in names
        assert "aryl_sulfonate_so3" in names

    def test_n_nitramine_matches_nitramine_ring(self):
        """N-nitramine pattern matches cyclic nitramine."""
        mol = Chem.MolFromSmiles("O=[N+]([O-])NC1=NCCN1")
        results = _apply_nl_prepass(NL_RULES, mol, mol.GetNumAtoms())
        names = [NL_RULES[ridx].name for _, ridx in results]
        assert "n_nitramine_n2o" in names

    def test_n_nitramine_no_match_nitrobenzene(self):
        """Nitrobenzene (no N-H adjacent) must not match n_nitramine_n2o."""
        mol = Chem.MolFromSmiles("O=[N+]([O-])c1ccccc1")
        results = _apply_nl_prepass(NL_RULES, mol, mol.GetNumAtoms())
        names = [NL_RULES[ridx].name for _, ridx in results]
        assert "n_nitramine_n2o" not in names


# ---------------------------------------------------------------------------
# 2. Product mask atom counts
# ---------------------------------------------------------------------------


class TestNLProductMasks:
    def test_sulfonamide_so2_removes_s_and_two_o(self):
        """SO₂ extrusion removes exactly S + 2 O atoms from the product mask."""
        smi = "O=[N+]([O-])c1ccc(S(=O)(=O)Nc2ncccn2)cc1"
        mol = Chem.MolFromSmiles(smi)
        n_total = mol.GetNumAtoms()  # 19
        results = _apply_nl_prepass(NL_RULES, mol, n_total)
        so2_results = [(m, r) for m, r in results if NL_RULES[r].name == "aryl_sulfonamide_so2"]
        assert len(so2_results) == 1
        product_mask, _ = so2_results[0]
        # 3 atoms removed (S + 2×O = SO₂)
        assert int(product_mask.sum()) == n_total - 3

    def test_sulfonate_so3_removes_s_and_three_o(self):
        """SO₃ loss removes exactly S + 3 O atoms."""
        smi = "O=S(=O)(O)c1ccc2ccc3cccc4ccc1c2c34"  # pyrene sulfonic acid
        mol = Chem.MolFromSmiles(smi)
        n_total = mol.GetNumAtoms()  # 20
        results = _apply_nl_prepass(NL_RULES, mol, n_total)
        so3_results = [(m, r) for m, r in results if NL_RULES[r].name == "aryl_sulfonate_so3"]
        assert len(so3_results) == 1
        product_mask, _ = so3_results[0]
        assert int(product_mask.sum()) == n_total - 4  # S + 3×O

    def test_nitramine_n2o_removes_two_n_and_one_o(self):
        """N₂O extrusion removes exactly 2 N + 1 O atoms (N₂O)."""
        smi = "O=[N+]([O-])NC1=NCCN1"
        mol = Chem.MolFromSmiles(smi)
        n_total = mol.GetNumAtoms()  # 9
        results = _apply_nl_prepass(NL_RULES, mol, n_total)
        n2o_results = [(m, r) for m, r in results if NL_RULES[r].name == "n_nitramine_n2o"]
        assert len(n2o_results) == 1
        product_mask, _ = n2o_results[0]
        assert int(product_mask.sum()) == n_total - 3  # N + O + N = N₂O

    def test_deduplication_same_nl_set(self):
        """Multiple SMARTS embeddings on the same atoms must be deduplicated."""
        # Symmetric molecule where both orientations of the NL match hit the same atoms
        smi = "O=S(=O)(O)c1ccccc1"  # benzenesulfonic acid
        mol = Chem.MolFromSmiles(smi)
        results = _apply_nl_prepass(NL_RULES, mol, mol.GetNumAtoms())
        # Only 2 unique NL products (SO₂ and SO₃), not 4 (not doubled by embedding symmetry)
        names = [NL_RULES[r].name for _, r in results]
        assert names.count("aryl_sulfonate_so2") == 1
        assert names.count("aryl_sulfonate_so3") == 1


# ---------------------------------------------------------------------------
# 3. DAG injection
# ---------------------------------------------------------------------------


class TestInjectNLMasks:
    def test_new_formula_adds_node_and_edge(self):
        """A genuinely new NL product mask adds 1 node and 1 edge to the DAG."""
        smi = "O=[N+]([O-])c1ccc(S(=O)(=O)Nc2ncccn2)cc1"
        dag_base = _run_dag(smi, smarts_prepass=False)
        dag_nl = _run_dag(smi, smarts_prepass=True)
        # At least 1 new node from the NL prepass (sulfonamide SO₂ product)
        assert dag_nl["dag_num_nodes"] > dag_base["dag_num_nodes"]

    def test_so3_product_not_duplicated_when_already_in_bfs(self):
        """SO₃ loss (simple bridge cut) already in BFS should not add an extra node."""
        smi = "O=S(=O)(O)c1ccc2ccc3cccc4ccc1c2c34"  # pyrene sulfonic acid
        dag_base = _run_dag(smi, smarts_prepass=False)
        dag_nl = _run_dag(smi, smarts_prepass=True)
        # SO₃ product (Ar fragment from direct S–C cut) is in BFS; SO₂ product is new.
        # Net new nodes = 1 (only SO₂ product).
        delta = dag_nl["dag_num_nodes"] - dag_base["dag_num_nodes"]
        assert delta == 1

    def test_nl_node_has_depth2(self):
        """Injected NL product nodes must have min_depth=2."""
        from fragnnet.frag.compute_frags import MASK_DTYPE

        smi = "O=[N+]([O-])c1ccc(S(=O)(=O)Nc2ncccn2)cc1"
        mol_d = extract_mol_info(smi)
        mol = mol_d["mol"]
        num_nodes, num_edges, node_mask, edges, edge_mask, _ = get_fraggen_input_arrays(mol_d)

        nl_results = _apply_nl_prepass(NL_RULES, mol, num_nodes)
        nl_masks = [m for m, _ in nl_results]

        # Build minimal DAG scaffold (root only)
        root_mask = np.ones((1, num_nodes), dtype=MASK_DTYPE)
        depth_matrix = np.zeros((1, 4), dtype=MASK_DTYPE)
        depth_matrix[0, 0] = 1
        dag_edges = np.empty((0, 2), dtype=np.int64)
        dag_meta = {
            "nodes_min_depth": np.array([0], dtype=MASK_DTYPE),
            "edges_min_depth": np.array([], dtype=MASK_DTYPE),
            "reached_depth": 0,
            "force_stopped": False,
        }

        _, _, _, out_meta = _inject_nl_masks_into_dag(
            nl_masks, root_mask, depth_matrix, dag_edges, dag_meta, num_nodes, 1
        )
        assert all(d == 2 for d in out_meta["nodes_min_depth"][1:])


# ---------------------------------------------------------------------------
# 4. Diarylethylene C₂H₂ extrusion (diarylvinyl_c2h2_loss)
# ---------------------------------------------------------------------------


class TestDiarylvinylC2H2Loss:
    # trans-stilbene: C6H5-CH=CH-C6H5 (14 heavy atoms; loses 2 vinyl C → 12 remain)
    TARGET_STILBENE = "C(=C/c1ccccc1)\\c1ccccc1"
    # resveratrol: hydroxylated stilbene derivative
    TARGET_RESVERATROL = "Oc1ccc(C=Cc2cc(O)cc(O)c2)cc1"
    # pyridyl-vinyl-phenyl: heteroaromatic variant
    TARGET_PYRIDYL = "c1ccnc(c1)/C=C/c1ccccc1"
    # Negative: indene (vinyl CH=CH in a ring → excluded by !R)
    NEG_INDENE = "C1=Cc2ccccc21"
    # Negative: chalcone (vinyl bridge not between two aromatic C)
    NEG_CHALCONE = "O=C(C=Cc1ccccc1)c1ccccc1"
    # Negative: styrene (terminal vinyl, one CH₂ end)
    NEG_STYRENE = "C(=C)c1ccccc1"

    def test_rule_exists_in_nl_rules(self):
        assert any(r.name == "diarylvinyl_c2h2_loss" for r in NL_RULES)

    def test_smarts_compiles(self):
        rule = next(r for r in NL_RULES if r.name == "diarylvinyl_c2h2_loss")
        assert Chem.MolFromSmarts(rule.reactant_smarts) is not None

    def test_matches_stilbene(self):
        rule = next(r for r in NL_RULES if r.name == "diarylvinyl_c2h2_loss")
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.TARGET_STILBENE)
        assert mol.HasSubstructMatch(patt)

    def test_matches_resveratrol(self):
        rule = next(r for r in NL_RULES if r.name == "diarylvinyl_c2h2_loss")
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.TARGET_RESVERATROL)
        assert mol.HasSubstructMatch(patt)

    def test_matches_pyridyl_variant(self):
        rule = next(r for r in NL_RULES if r.name == "diarylvinyl_c2h2_loss")
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.TARGET_PYRIDYL)
        assert mol.HasSubstructMatch(patt)

    def test_no_match_indene(self):
        """Vinyl in a fused ring (indene) must not match due to !R guard."""
        rule = next(r for r in NL_RULES if r.name == "diarylvinyl_c2h2_loss")
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.NEG_INDENE)
        assert not mol.HasSubstructMatch(patt)

    def test_no_match_chalcone(self):
        """Chalcone vinyl (bonded to C=O, not aryl, on one side) must not match."""
        rule = next(r for r in NL_RULES if r.name == "diarylvinyl_c2h2_loss")
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.NEG_CHALCONE)
        assert not mol.HasSubstructMatch(patt)

    def test_no_match_styrene(self):
        """Terminal vinyl (=CH₂, two H) must not match [CH;!R] (exactly one H)."""
        rule = next(r for r in NL_RULES if r.name == "diarylvinyl_c2h2_loss")
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.NEG_STYRENE)
        assert not mol.HasSubstructMatch(patt)

    def test_product_mask_loses_two_vinyl_carbons(self):
        """C₂H₂ extrusion removes exactly 2 vinyl C atoms from stilbene (14→12)."""
        mol = Chem.MolFromSmiles(self.TARGET_STILBENE)
        n_total = mol.GetNumAtoms()  # 14
        results = _apply_nl_prepass(NL_RULES, mol, n_total)
        rule_results = [
            (m, ri) for m, ri in results if NL_RULES[ri].name == "diarylvinyl_c2h2_loss"
        ]
        assert len(rule_results) == 1
        prod_mask, _ = rule_results[0]
        assert int(prod_mask.sum()) == n_total - 2  # lose 2 vinyl C atoms

    def test_lost_atoms_are_carbon(self):
        """Both atoms removed from stilbene are carbon (the vinyl CH=CH bridge)."""
        mol = Chem.MolFromSmiles(self.TARGET_STILBENE)
        results = _apply_nl_prepass(NL_RULES, mol, mol.GetNumAtoms())
        rule_results = [
            (m, ri) for m, ri in results if NL_RULES[ri].name == "diarylvinyl_c2h2_loss"
        ]
        assert rule_results
        prod_mask, _ = rule_results[0]
        lost_syms = [mol.GetAtomWithIdx(i).GetSymbol() for i, v in enumerate(prod_mask) if not v]
        assert lost_syms == ["C", "C"]

    def test_deduplication_on_symmetric_stilbene(self):
        """Both SMARTS embeddings (forward and reverse) must deduplicate to 1 result."""
        mol = Chem.MolFromSmiles(self.TARGET_STILBENE)
        results = _apply_nl_prepass(NL_RULES, mol, mol.GetNumAtoms())
        names = [NL_RULES[ri].name for _, ri in results]
        assert names.count("diarylvinyl_c2h2_loss") == 1

    def test_nl_mass_is_c2h2(self):
        """Rule nl_mass must equal exact monoisotopic mass of C₂H₂ (26.0157 Da)."""
        rule = next(r for r in NL_RULES if r.name == "diarylvinyl_c2h2_loss")
        assert abs(rule.nl_mass - 26.0157) < 1e-4

    def test_dag_adds_nodes_for_stilbene(self):
        """DAG with smarts_prepass=True must have more nodes than without for stilbene."""
        dag_base = _run_dag(self.TARGET_STILBENE, smarts_prepass=False)
        dag_nl = _run_dag(self.TARGET_STILBENE, smarts_prepass=True)
        assert dag_nl["dag_num_nodes"] > dag_base["dag_num_nodes"]


# ---------------------------------------------------------------------------
# 4. Sulfoxide SO loss rule
# ---------------------------------------------------------------------------


class TestSulfoxideSoLoss:
    DMSO = "CS(=O)C"
    METHYL_PHENYL_SULFOXIDE = "c1ccccc1S(=O)C"
    DIPHENYL_SULFOXIDE = "c1ccccc1S(=O)c1ccccc1"
    NEG_SULFONE = "CS(=O)(=O)C"
    NEG_THIOETHER = "CSC"
    NEG_SULFONAMIDE = "CS(=O)(=O)NC"

    def test_rule_exists_in_nl_rules(self):
        assert any(r.name == "sulfoxide_so_loss" for r in NL_RULES)

    def test_smarts_compiles(self):
        rule = next(r for r in NL_RULES if r.name == "sulfoxide_so_loss")
        assert Chem.MolFromSmarts(rule.reactant_smarts) is not None

    def test_matches_dmso(self):
        rule = next(r for r in NL_RULES if r.name == "sulfoxide_so_loss")
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.DMSO)
        assert mol.HasSubstructMatch(patt)

    def test_matches_methyl_phenyl_sulfoxide(self):
        rule = next(r for r in NL_RULES if r.name == "sulfoxide_so_loss")
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.METHYL_PHENYL_SULFOXIDE)
        assert mol.HasSubstructMatch(patt)

    def test_matches_diphenyl_sulfoxide(self):
        rule = next(r for r in NL_RULES if r.name == "sulfoxide_so_loss")
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.DIPHENYL_SULFOXIDE)
        assert mol.HasSubstructMatch(patt)

    def test_no_match_sulfone(self):
        rule = next(r for r in NL_RULES if r.name == "sulfoxide_so_loss")
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.NEG_SULFONE)
        assert not mol.HasSubstructMatch(patt)

    def test_no_match_thioether(self):
        rule = next(r for r in NL_RULES if r.name == "sulfoxide_so_loss")
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.NEG_THIOETHER)
        assert not mol.HasSubstructMatch(patt)

    def test_no_match_sulfonamide(self):
        rule = next(r for r in NL_RULES if r.name == "sulfoxide_so_loss")
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.NEG_SULFONAMIDE)
        assert not mol.HasSubstructMatch(patt)

    def test_product_mask_loses_s_and_o(self):
        mol = Chem.MolFromSmiles(self.DMSO)
        results = _apply_nl_prepass(NL_RULES, mol, mol.GetNumAtoms())
        rule_results = [(m, ri) for m, ri in results if NL_RULES[ri].name == "sulfoxide_so_loss"]
        assert len(rule_results) >= 1
        prod_mask, _ = rule_results[0]
        # DMSO: 4 atoms total (C, S, O, C); SO loss removes 2 → 2 retained
        assert int(prod_mask.sum()) == mol.GetNumAtoms() - 2

    def test_product_retains_only_carbons_for_dmso(self):
        mol = Chem.MolFromSmiles(self.DMSO)
        results = _apply_nl_prepass(NL_RULES, mol, mol.GetNumAtoms())
        rule_results = [(m, ri) for m, ri in results if NL_RULES[ri].name == "sulfoxide_so_loss"]
        prod_mask, _ = rule_results[0]
        retained = [mol.GetAtomWithIdx(i).GetSymbol() for i, v in enumerate(prod_mask) if v]
        lost = [mol.GetAtomWithIdx(i).GetSymbol() for i, v in enumerate(prod_mask) if not v]
        assert sorted(retained) == ["C", "C"]
        assert sorted(lost) == ["O", "S"]

    def test_product_mask_diphenyl_sulfoxide(self):
        mol = Chem.MolFromSmiles(self.DIPHENYL_SULFOXIDE)
        n_total = mol.GetNumAtoms()  # 13: 12 C + 1 S + 1 O = 14 heavy atoms
        results = _apply_nl_prepass(NL_RULES, mol, n_total)
        rule_results = [(m, ri) for m, ri in results if NL_RULES[ri].name == "sulfoxide_so_loss"]
        assert len(rule_results) >= 1
        prod_mask, _ = rule_results[0]
        # 14 atoms - 2 (S+O) = 12 retained (both phenyl rings)
        assert int(prod_mask.sum()) == n_total - 2

    def test_dag_adds_nodes_for_sulfinyl_amide(self):
        # CS(=O)CCNC(=O)c1ccccc1: the C-methyl and C-CCNC flanking groups combine
        # into a 12-atom fragment not reachable by BFS; SO-loss injects it as new.
        smi = "CS(=O)CCNC(=O)c1ccccc1"
        dag_base = _run_dag(smi, smarts_prepass=False)
        dag_nl = _run_dag(smi, smarts_prepass=True)
        assert dag_nl["dag_num_nodes"] > dag_base["dag_num_nodes"]


# ---------------------------------------------------------------------------
# 5. Diaryl ketone CO loss rule
# ---------------------------------------------------------------------------


class TestDiarylKetoneCOLoss:
    # Benzophenone: Ph-CO-Ph (canonical diaryl ketone, 14 heavy atoms)
    BENZOPHENONE = "O=C(c1ccccc1)c1ccccc1"
    # Heteroaromatic: phenyl 2-furyl ketone (c1 and c4 are aromatic, keto not in ring)
    FURYL_PHENYL_KETONE = "O=C(c1ccccc1)c1ccco1"
    # Negative: chromone (keto C in lactone ring → !R excludes it)
    CHROMONE = "O=c1ccoc2ccccc12"
    # Negative: fluorenone (keto C in 5-membered ring fused to two benzene rings)
    FLUORENONE = "O=C1c2ccccc2-c2ccccc21"
    # Negative: acetophenone (only one aryl group; methyl C is not [c])
    ACETOPHENONE = "CC(=O)c1ccccc1"
    # Negative: chalcone (vinyl bridge between aryl and keto; keto not between two [c])
    CHALCONE = "O=C(C=Cc1ccccc1)c1ccccc1"

    def test_rule_exists_in_nl_rules(self):
        assert any(r.name == "diaryl_ketone_co_loss" for r in NL_RULES)

    def test_smarts_compiles(self):
        rule = next(r for r in NL_RULES if r.name == "diaryl_ketone_co_loss")
        assert Chem.MolFromSmarts(rule.reactant_smarts) is not None

    def test_matches_benzophenone(self):
        rule = next(r for r in NL_RULES if r.name == "diaryl_ketone_co_loss")
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.BENZOPHENONE)
        assert mol.HasSubstructMatch(patt)

    def test_matches_furyl_phenyl_ketone(self):
        """Heteroaromatic diaryl ketone must match (furan ring counts as [c])."""
        rule = next(r for r in NL_RULES if r.name == "diaryl_ketone_co_loss")
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.FURYL_PHENYL_KETONE)
        assert mol.HasSubstructMatch(patt)

    def test_no_match_chromone(self):
        """Chromone keto C is in a lactone ring; !R guard must exclude it."""
        rule = next(r for r in NL_RULES if r.name == "diaryl_ketone_co_loss")
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.CHROMONE)
        assert not mol.HasSubstructMatch(patt)

    def test_no_match_fluorenone(self):
        """Fluorenone keto C is in a ring; !R guard must exclude it."""
        rule = next(r for r in NL_RULES if r.name == "diaryl_ketone_co_loss")
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.FLUORENONE)
        assert not mol.HasSubstructMatch(patt)

    def test_no_match_acetophenone(self):
        """Acetophenone has only one aryl group; [c:4] cannot match the methyl C."""
        rule = next(r for r in NL_RULES if r.name == "diaryl_ketone_co_loss")
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.ACETOPHENONE)
        assert not mol.HasSubstructMatch(patt)

    def test_product_mask_loses_c_and_o(self):
        """CO extrusion removes exactly 1 C + 1 O (2 atoms) from benzophenone (14→12)."""
        mol = Chem.MolFromSmiles(self.BENZOPHENONE)
        n_total = mol.GetNumAtoms()  # 14
        results = _apply_nl_prepass(NL_RULES, mol, n_total)
        rule_results = [
            (m, ri) for m, ri in results if NL_RULES[ri].name == "diaryl_ketone_co_loss"
        ]
        assert len(rule_results) == 1
        prod_mask, _ = rule_results[0]
        assert int(prod_mask.sum()) == n_total - 2  # lose C + O

    def test_lost_atoms_are_carbon_and_oxygen(self):
        """The 2 atoms removed from benzophenone must be C (keto) and O."""
        mol = Chem.MolFromSmiles(self.BENZOPHENONE)
        results = _apply_nl_prepass(NL_RULES, mol, mol.GetNumAtoms())
        rule_results = [
            (m, ri) for m, ri in results if NL_RULES[ri].name == "diaryl_ketone_co_loss"
        ]
        assert rule_results
        prod_mask, _ = rule_results[0]
        lost_syms = sorted(
            mol.GetAtomWithIdx(i).GetSymbol() for i, v in enumerate(prod_mask) if not v
        )
        assert lost_syms == ["C", "O"]

    def test_nl_mass_is_co(self):
        """Rule nl_mass must equal exact monoisotopic mass of CO (27.9949 Da)."""
        rule = next(r for r in NL_RULES if r.name == "diaryl_ketone_co_loss")
        assert abs(rule.nl_mass - 27.9949) < 1e-4

    def test_dag_adds_nodes_for_benzophenone(self):
        """Benzophenone DAG must gain at least 1 new node from CO loss prepass."""
        dag_base = _run_dag(self.BENZOPHENONE, smarts_prepass=False)
        dag_nl = _run_dag(self.BENZOPHENONE, smarts_prepass=True)
        assert dag_nl["dag_num_nodes"] > dag_base["dag_num_nodes"]

    def test_product_mask_not_in_bfs(self):
        """The CO loss product atom mask must be absent from raw BFS (3-CC path blocked)."""
        from fragnnet.frag import compute_frags

        mol_d = extract_mol_info(self.BENZOPHENONE)
        mol = Chem.MolFromSmiles(self.BENZOPHENONE)
        num_nodes, num_edges, node_mask, edges, edge_mask, n2e = get_fraggen_input_arrays(mol_d)
        bfs_masks, _, _, _ = compute_frags.compute_ccs(
            num_nodes, num_edges, node_mask, edges, edge_mask, n2e, 3, int(1e6)
        )
        bfs_keys = {bfs_masks[i].tobytes() for i in range(len(bfs_masks))}

        # Apply CO loss rule to find the product mask
        results = _apply_nl_prepass(NL_RULES, mol, num_nodes)
        co_results = [(m, ri) for m, ri in results if NL_RULES[ri].name == "diaryl_ketone_co_loss"]
        assert co_results, "diaryl_ketone_co_loss produced no results for benzophenone"
        prod_mask, _ = co_results[0]
        assert prod_mask.tobytes() not in bfs_keys


class TestAminobenzothiazoleHcnLoss:
    # 6-chlorobenzothiazol-2-amine (NIST20 mol_id 27791; NL=27.0113 Da confirmed)
    TARGET_1 = "Nc1nc2ccc(Cl)cc2s1"
    # 4-bromo-6-fluorobenzothiazol-2-amine (NIST20 mol_id 27768; two halogens)
    TARGET_2 = "Nc1nc2c(Br)cc(F)cc2s1"
    # Unsubstituted benzothiazol-2-amine (NIST20 mol_id 27800; bare scaffold)
    BARE_SCAFFOLD = "Nc1nc2ccccc2s1"
    # Negative: N-acetylated amine, no free NH2 (should not match)
    NEG_N_ACYL = "CC(=O)Nc1nc2ccccc2s1"
    # Negative: 2-aminobenzimidazole, ring N instead of ring S (should not match)
    NEG_BENZIMIDAZOLE = "Nc1nc2ccccc2[nH]1"
    # Negative: 2-aminothiazole, non-fused (no benzo ring; should not match)
    NEG_NONFUSED_THIAZOLE = "Nc1nccs1"
    # Negative: plain benzene
    NEG_BENZENE = "c1ccccc1"

    def test_rule_exists_in_nl_rules(self):
        assert any(r.name == "aminobenzothiazole_hcn_loss" for r in NL_RULES)

    def test_smarts_compiles(self):
        rule = _nl_rules_by_name()["aminobenzothiazole_hcn_loss"]
        assert Chem.MolFromSmarts(rule.reactant_smarts) is not None

    def test_matches_target_1(self):
        rule = _nl_rules_by_name()["aminobenzothiazole_hcn_loss"]
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.TARGET_1)
        assert mol.HasSubstructMatch(patt)

    def test_matches_target_2_dihalogen(self):
        """Different substituent positions/halogens must still match."""
        rule = _nl_rules_by_name()["aminobenzothiazole_hcn_loss"]
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.TARGET_2)
        assert mol.HasSubstructMatch(patt)

    def test_matches_bare_scaffold(self):
        rule = _nl_rules_by_name()["aminobenzothiazole_hcn_loss"]
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.BARE_SCAFFOLD)
        assert mol.HasSubstructMatch(patt)

    def test_no_match_n_acyl(self):
        """N-acylated amine has no free NH2; SMARTS [NX3H2] must exclude it."""
        rule = _nl_rules_by_name()["aminobenzothiazole_hcn_loss"]
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.NEG_N_ACYL)
        assert not mol.HasSubstructMatch(patt)

    def test_no_match_benzimidazole(self):
        """Ring S is required; benzimidazole's ring N must not satisfy 's1'."""
        rule = _nl_rules_by_name()["aminobenzothiazole_hcn_loss"]
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.NEG_BENZIMIDAZOLE)
        assert not mol.HasSubstructMatch(patt)

    def test_no_match_nonfused_thiazole(self):
        """Benzo-fusion is required; a bare thiazole ring must not match."""
        rule = _nl_rules_by_name()["aminobenzothiazole_hcn_loss"]
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.NEG_NONFUSED_THIAZOLE)
        assert not mol.HasSubstructMatch(patt)

    def test_no_match_benzene(self):
        rule = _nl_rules_by_name()["aminobenzothiazole_hcn_loss"]
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.NEG_BENZENE)
        assert not mol.HasSubstructMatch(patt)

    def test_product_mask_loses_c_and_n(self):
        """HCN extrusion removes exactly 1 C + 1 N (2 atoms)."""
        mol = Chem.MolFromSmiles(self.TARGET_1)
        n_total = mol.GetNumAtoms()
        results = _apply_nl_prepass(NL_RULES, mol, n_total)
        rule_results = [
            (m, ri) for m, ri in results if NL_RULES[ri].name == "aminobenzothiazole_hcn_loss"
        ]
        assert len(rule_results) == 1
        prod_mask, _ = rule_results[0]
        assert int(prod_mask.sum()) == n_total - 2

    def test_lost_atoms_are_ring_c_and_ring_n(self):
        """The 2 removed atoms must be the ring C2 and ring N3 (not S1 or exocyclic N)."""
        mol = Chem.MolFromSmiles(self.TARGET_1)
        results = _apply_nl_prepass(NL_RULES, mol, mol.GetNumAtoms())
        rule_results = [
            (m, ri) for m, ri in results if NL_RULES[ri].name == "aminobenzothiazole_hcn_loss"
        ]
        assert rule_results
        prod_mask, _ = rule_results[0]
        lost_syms = sorted(
            mol.GetAtomWithIdx(i).GetSymbol() for i, v in enumerate(prod_mask) if not v
        )
        assert lost_syms == ["C", "N"]
        # exocyclic amine N and ring S1 must both survive in the product
        retained_syms = [mol.GetAtomWithIdx(i).GetSymbol() for i, v in enumerate(prod_mask) if v]
        assert retained_syms.count("N") == 1  # only the exocyclic amine N remains
        assert retained_syms.count("S") == 1  # ring S1 remains

    def test_nl_mass_is_hcn(self):
        """Rule nl_mass must equal exact monoisotopic mass of HCN (27.0109 Da)."""
        rule = _nl_rules_by_name()["aminobenzothiazole_hcn_loss"]
        assert abs(rule.nl_mass - 27.0109) < 1e-4

    def test_dag_adds_nodes_for_target(self):
        """DAG for the target molecule must gain at least 1 new node from HCN loss prepass."""
        dag_base = _run_dag(self.TARGET_1, smarts_prepass=False)
        dag_nl = _run_dag(self.TARGET_1, smarts_prepass=True)
        assert dag_nl["dag_num_nodes"] > dag_base["dag_num_nodes"]

    def test_product_mask_not_in_bfs(self):
        """The HCN loss product atom mask must be absent from raw BFS (3-CC path blocked)."""
        from fragnnet.frag import compute_frags

        mol_d = extract_mol_info(self.TARGET_1)
        mol = Chem.MolFromSmiles(self.TARGET_1)
        num_nodes, num_edges, node_mask, edges, edge_mask, n2e = get_fraggen_input_arrays(mol_d)
        bfs_masks, _, _, _ = compute_frags.compute_ccs(
            num_nodes, num_edges, node_mask, edges, edge_mask, n2e, 3, int(1e6)
        )
        bfs_keys = {bfs_masks[i].tobytes() for i in range(len(bfs_masks))}

        results = _apply_nl_prepass(NL_RULES, mol, num_nodes)
        hcn_results = [
            (m, ri) for m, ri in results if NL_RULES[ri].name == "aminobenzothiazole_hcn_loss"
        ]
        assert hcn_results, "aminobenzothiazole_hcn_loss produced no results for target"
        prod_mask, _ = hcn_results[0]
        assert prod_mask.tobytes() not in bfs_keys
