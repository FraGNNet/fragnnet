"""SMARTS-based rearrangement prepass for multi-bond BFS fragmentation.

Contains :class:`SmartsFragRule`, :data:`FRAG_RULES`, :func:`_apply_smarts_prepass`,
:class:`NeutralLossRule`, :data:`NL_RULES`, and :func:`_apply_nl_prepass` — the
pure-Python prepass for reactions that require new bond formation (1,3-elimination,
acyclic ester CO₂ loss) and neutral-loss rearrangements (SO₂ extrusion, SO₃ loss,
N₂O extrusion).

Kept in a separate module so the Cython ``.so`` can shadow ``multi_cut_bfs``
for the BFS entry-point while this prepass remains importable without any
``importlib`` hacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem

MASK_DTYPE = np.uint8


@dataclass
class SmartsFragRule:
    """SMARTS-based fragmentation rule for the rearrangement prepass.

    Encodes a 2-fragment reaction that cannot be described as a simple bond cut
    (e.g., 1,3-elimination, ester CO₂ loss) by specifying which bonds to cut and
    how to reassemble 3-component results into 2 fragments.

    All atom map numbers must be **1-based** (never 0).  RDKit treats map number 0
    as "no map", so :0 atoms cannot be retrieved by ``GetAtomMapNum()``.

    Attributes:
        name: Rule identifier for logging and debugging.
        reactant_smarts: SMARTS pattern for the reactant substructure only
            (no ">>" product part).  All mapped atoms use 1-based map numbers.
        cut_bond_map_pairs: Pairs of 1-based atom map numbers defining bonds to
            sever.  Each pair ``(a, b)`` identifies the bond between the matched
            atoms carrying map numbers *a* and *b*.
        merge_outer_map_nums: When cutting produces 3 CCs (n_ccs==3), the map
            numbers of the two "outer" atoms whose CCs are merged into one
            fragment.  Empty list discards n_ccs==3 results.
        filter_bond_map_nums: Optional pair of map numbers.  If the matched atoms
            are directly bonded in the original graph the rule is skipped — used
            to exclude cyclic cases already covered by BFS cut=2.
        min_num_atoms: Minimum number of atoms required for this rule to apply.
            Rules are skipped early if the molecule has fewer atoms, avoiding
            unnecessary SMARTS matching on small molecules.

    Example:
        >>> rule = SmartsFragRule(
        ...     name="crf12_0",
        ...     reactant_smarts="[*+0:1]-[*+0:2]-[*+0:3]-[*+0:4]",
        ...     cut_bond_map_pairs=[(1, 2), (3, 4)],
        ...     merge_outer_map_nums=[1, 4],
        ...     filter_bond_map_nums=(1, 4),
        ...     min_num_atoms=4,
        ... )
    """

    name: str
    reactant_smarts: str
    cut_bond_map_pairs: list[tuple[int, int]]
    merge_outer_map_nums: list[int] = field(default_factory=list)
    filter_bond_map_nums: tuple[int, int] | None = None
    min_num_atoms: int = 0


FRAG_RULES: list[SmartsFragRule] = [
    # crf12_0: acyclic 1,3-elimination (single bonds on both sides)
    # A–B–C–D  →  A·D (new bond) + B=C
    # Cut bonds A-B and C-D; outer atoms A,D merge; inner B,C stay together.
    # Skip if A and D are already directly bonded (4-membered ring → BFS cut=2).
    SmartsFragRule(
        name="crf12_0",
        reactant_smarts="[*+0:1]-[*+0:2]-[*+0:3]-[*+0:4]",
        cut_bond_map_pairs=[(1, 2), (3, 4)],
        merge_outer_map_nums=[1, 4],
        filter_bond_map_nums=(1, 4),
        min_num_atoms=4,
    ),
    # crf12_1: acyclic 1,3-elimination (double bond between inner atoms)
    # A–B=C–D  →  A·D (new bond) + B≡C
    SmartsFragRule(
        name="crf12_1",
        reactant_smarts="[*+0:1]-[*+0:2]=[*+0:3]-[*+0:4]",
        cut_bond_map_pairs=[(1, 2), (3, 4)],
        merge_outer_map_nums=[1, 4],
        filter_bond_map_nums=(1, 4),
        min_num_atoms=4,
    ),
    # crf13_2: CO2 loss from acyclic ester
    # R–C(=O)–O–R'  →  CO2  +  R·R' (new R-R' bond)
    # Cut bonds R-C and O-R'; outer R,R' merge; inner C,=O,O stay together (CO2).
    # Skip if R and R' are already directly bonded (lactone → BFS cut=2).
    SmartsFragRule(
        name="crf13_2",
        reactant_smarts="[*+0:1][C+0:2](=[O+0:3])[O+0:4][*+0:5]",
        cut_bond_map_pairs=[(1, 2), (4, 5)],
        merge_outer_map_nums=[1, 5],
        filter_bond_map_nums=(1, 5),
        min_num_atoms=5,
    ),
]

# Pre-compiled SMARTS patterns (keyed by rule name) — avoid recompiling per molecule.
_COMPILED_PATTERNS: dict[str, Chem.Mol] = {}


def _get_compiled_pattern(rule: SmartsFragRule) -> Chem.Mol | None:
    """Return a cached compiled SMARTS pattern for *rule*, compiling on first call."""
    if rule.name not in _COMPILED_PATTERNS:
        _COMPILED_PATTERNS[rule.name] = Chem.MolFromSmarts(rule.reactant_smarts)
    return _COMPILED_PATTERNS[rule.name]


def _union_find_ccs_all(
    num_nodes: int,
    node_mask: np.ndarray,
    edges: np.ndarray,
    edge_mask: np.ndarray,
    num_edges: int,
) -> list[np.ndarray]:
    """Compute all connected components without a cap on their count.

    Used by the SMARTS prepass where reactions may produce 3 components
    (e.g., 1,3-elimination splits A-B-C-D into {A}, {B,C}, {D}).

    Args:
        num_nodes: Total atom count.
        node_mask: Shape ``(num_nodes,)`` uint8 — active atoms.
        edges: Shape ``(MAX_NUM_EDGES, 2)`` int32 — bond endpoints.
        edge_mask: Shape ``(MAX_NUM_EDGES,)`` uint8 — active bonds.
        num_edges: Number of real bonds (un-padded prefix length).

    Returns:
        List of node-mask arrays (shape ``(num_nodes,)`` uint8), one per CC.
        Empty list if no active atoms.
    """
    # Use a Python list for parent to avoid numpy indexing overhead in the hot loop.
    parent = list(range(num_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    live_atoms = [i for i in range(num_nodes) if node_mask[i] == 1]
    if not live_atoms:
        return []

    for idx in range(num_edges):
        if edge_mask[idx] == 0:
            continue
        u, v = int(edges[idx, 0]), int(edges[idx, 1])
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pu] = pv

    roots: dict[int, int] = {}
    for a in live_atoms:
        r = find(a)
        if r not in roots:
            roots[r] = len(roots)

    cc_masks = [np.zeros(num_nodes, dtype=MASK_DTYPE) for _ in range(len(roots))]
    for a in live_atoms:
        cc_masks[roots[find(a)]][a] = 1

    return cc_masks


def _build_edge_dict(edges: np.ndarray, num_edges: int) -> dict[tuple[int, int], int]:
    """Build a reverse lookup from (atom_u, atom_v) → edge index.

    Both orderings (u, v) and (v, u) are stored so lookups are O(1) regardless
    of the direction the edge was stored in the array.

    Args:
        edges: Shape ``(MAX_NUM_EDGES, 2)`` int32 — bond endpoints.
        num_edges: Number of real bonds (un-padded prefix length).

    Returns:
        Dict mapping ``(atom_a, atom_b)`` to its 0-based edge index.
    """
    d: dict[tuple[int, int], int] = {}
    for i in range(num_edges):
        u, v = int(edges[i, 0]), int(edges[i, 1])
        d[(u, v)] = i
        d[(v, u)] = i
    return d


def _apply_smarts_prepass(
    rules: list[SmartsFragRule],
    mol: Chem.Mol,
    base_edge_mask: np.ndarray,
    edges: np.ndarray,
    num_nodes: int,
    num_edges: int,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Apply SMARTS fragmentation rules to the root molecule.

    For each rule, matches the reactant SMARTS against *mol*, cuts the specified
    bonds, and decomposes the result into exactly two fragment node-masks.
    When a cut produces 3 components (e.g., 1,3-elimination), the two "outer"
    components (containing ``merge_outer_map_nums`` atoms) are merged into one
    fragment.

    Only the root (un-fragmented) molecule is processed; the resulting fragment
    pairs are injected as depth-1 children in the DAG.

    Args:
        rules: List of :class:`SmartsFragRule` to apply.
        mol: RDKit molecule (same atoms/indices used to build the edge arrays).
        base_edge_mask: Shape ``(MAX_NUM_EDGES,)`` uint8 — root edge mask (all
            real bonds active).
        edges: Shape ``(MAX_NUM_EDGES, 2)`` int32 — bond endpoints.
        num_nodes: Number of atoms.
        num_edges: Number of real bonds.

    Returns:
        List of ``(node_mask_a, node_mask_b, rule_idx)`` tuples, one per valid
        match.  ``rule_idx`` is the 0-based index into *rules* that produced the
        pair.  Duplicate fragment pairs are **not** deduplicated here; the BFS
        ``css_to_id_dict`` handles deduplication by node-mask key.
    """
    root_node_mask = np.ones(num_nodes, dtype=MASK_DTYPE)
    results: list[tuple[np.ndarray, np.ndarray, int]] = []

    # Fix 1: build (atom_a, atom_b) → edge_idx lookup once per molecule call.
    edge_dict = _build_edge_dict(edges, num_edges)

    for rule_idx, rule in enumerate(rules):
        # Fix 3: skip rules that can't match given the molecule size.
        if rule.min_num_atoms > 0 and num_nodes < rule.min_num_atoms:
            continue

        # Use pre-compiled SMARTS pattern (avoids recompiling on every molecule).
        patt = _get_compiled_pattern(rule)
        if patt is None:
            continue

        # Build: query atom index → atom map number (skip unmapped atoms, i.e. map==0)
        q_idx_to_map: dict[int, int] = {}
        for q_idx in range(patt.GetNumAtoms()):
            map_num = patt.GetAtomWithIdx(q_idx).GetAtomMapNum()
            if map_num != 0:
                q_idx_to_map[q_idx] = map_num

        matches = mol.GetSubstructMatches(patt)
        for match in matches:
            # match[q_idx] = mol atom index for query atom q_idx
            map_num_to_mol: dict[int, int] = {
                map_num: match[q_idx] for q_idx, map_num in q_idx_to_map.items()
            }

            # Skip cyclic cases already covered by BFS cut=2.
            # Fix 1: O(1) dict lookup instead of O(E) np.where scan.
            if rule.filter_bond_map_nums is not None:
                fa, fb = rule.filter_bond_map_nums
                mol_a = map_num_to_mol.get(fa)
                mol_b = map_num_to_mol.get(fb)
                if mol_a is not None and mol_b is not None:
                    bond_idx = edge_dict.get((mol_a, mol_b), -1)
                    if bond_idx != -1 and base_edge_mask[bond_idx] == 1:
                        continue  # atoms directly bonded → cyclic; covered by cut=2

            # Cut the specified bonds in a temporary mask.
            work_mask = base_edge_mask.copy()
            valid = True
            for map_a, map_b in rule.cut_bond_map_pairs:
                mol_a = map_num_to_mol.get(map_a)
                mol_b = map_num_to_mol.get(map_b)
                if mol_a is None or mol_b is None:
                    valid = False
                    break
                bond_idx = edge_dict.get((mol_a, mol_b), -1)  # Fix 1: O(1) lookup
                if bond_idx == -1:
                    valid = False
                    break
                work_mask[bond_idx] = 0
            if not valid:
                continue

            # Fix 2: _union_find_ccs_all now uses Python list for parent array
            # (avoids numpy element-wise indexing overhead in the union-find loop).
            cc_masks = _union_find_ccs_all(
                num_nodes, root_node_mask, edges, work_mask, num_edges
            )
            n_ccs = len(cc_masks)

            if n_ccs == 2:
                results.append((cc_masks[0], cc_masks[1], rule_idx))
            elif n_ccs == 3 and rule.merge_outer_map_nums:
                # Identify which CCs contain the "outer" atoms
                outer_cc_set: set[int] = set()
                for map_num in rule.merge_outer_map_nums:
                    mol_idx = map_num_to_mol.get(map_num)
                    if mol_idx is not None:
                        for cc_idx, cc_mask in enumerate(cc_masks):
                            if cc_mask[mol_idx] == 1:
                                outer_cc_set.add(cc_idx)
                                break
                inner_cc_set = set(range(n_ccs)) - outer_cc_set
                if len(outer_cc_set) == 2 and len(inner_cc_set) == 1:
                    outer_mask = np.zeros(num_nodes, dtype=MASK_DTYPE)
                    for cc_idx in outer_cc_set:
                        outer_mask |= cc_masks[cc_idx]
                    inner_mask = cc_masks[next(iter(inner_cc_set))]
                    results.append((outer_mask, inner_mask, rule_idx))

    return results


@dataclass
class NeutralLossRule:
    """Rule for rearrangements that expel a neutral fragment and form a new bond.

    Unlike :class:`SmartsFragRule`, these reactions cannot be represented as
    simple bond cuts because the product requires bond formation (e.g. SO₂
    extrusion creates a new Ar–N bond).  The product is represented as an atom
    mask covering all precursor atoms *except* the neutral-loss atoms.  Because
    the new bond is not in the original graph, the product atoms may be
    disconnected in the original connectivity — H-count estimation will add
    excess H's at the free valences; the correct formula lands at a negative
    H-shift in the formula graph (covered by max_h_transfer).

    Attributes:
        name: Rule identifier for logging.
        reactant_smarts: SMARTS pattern.  All mapped atoms use 1-based map numbers.
        nl_map_nums: Map numbers of the atoms that leave as the neutral loss.
            The product mask = all atoms minus these.
        nl_mass: Exact monoisotopic neutral-loss mass (Da), for documentation.
        min_num_atoms: Minimum molecule heavy-atom count for early exit.
    """

    name: str
    reactant_smarts: str
    nl_map_nums: list[int]
    nl_mass: float
    min_num_atoms: int = 0


NL_RULES: list[NeutralLossRule] = [
    # Aromatic sulfonamide SO₂ extrusion (HIGHEST PRIORITY).
    # Ar–SO₂–NH–Het → [Ar–NH–Het]⁻ + SO₂  (NL = 63.962 Da).
    # A new Ar–N bond forms, so the product atoms are disconnected in the original
    # graph.  H-cap adds +2H (one at Ar–C, one at N); correct formula at H_delta=-2.
    # Confirmed in spec_id=610784 and 856552; affects thousands of NIST23 spectra.
    NeutralLossRule(
        name="aryl_sulfonamide_so2",
        reactant_smarts="[c:1]-[SX4:2](=[O:3])(=[O:4])-[NH:5]",
        nl_map_nums=[2, 3, 4],
        nl_mass=63.9619,
        min_num_atoms=8,
    ),
    # Aromatic sulfonate SO₂ loss.
    # Ar–SO₃⁻ → [Ar–O]⁻ + SO₂  (NL = 63.962 Da).
    # A new Ar–O bond forms; +2H excess at Ar–C and O-; correct formula at H_delta=-2.
    # Root cause of miss in pyrenes (82.8%) and phenanthrenes (50.9%) in [M-H]-.
    NeutralLossRule(
        name="aryl_sulfonate_so2",
        reactant_smarts="[c:1]-[SX4:2](=[O:3])(=[O:4])[O-,OH:5]",
        nl_map_nums=[2, 3, 4],
        nl_mass=63.9619,
        min_num_atoms=7,
    ),
    # Aromatic sulfonate SO₃ loss.
    # Ar–SO₃⁻ → [Ar]⁻ + SO₃  (NL = 79.957 Da).
    # Direct S–C bond cleavage; product is a connected Ar anion.
    # +1H excess at Ar–C; correct formula at H_delta=-1.
    NeutralLossRule(
        name="aryl_sulfonate_so3",
        reactant_smarts="[c:1]-[SX4:2](=[O:3])(=[O:4])[O-,OH:5]",
        nl_map_nums=[2, 3, 4, 5],
        nl_mass=79.9568,
        min_num_atoms=7,
    ),
    # N-nitramine N₂O extrusion.
    # R–N(NO₂)–NH–R' → [R–N=R'] + N₂O  (NL = 44.001 Da).
    # Confirmed in spec_id=550113 (0.1 mDa).  The O⁻ from NO₂ (:3) stays in the
    # product; +2H excess at O⁻ and at R' atom; correct formula at H_delta=-2.
    NeutralLossRule(
        name="n_nitramine_n2o",
        reactant_smarts="[N+:1](=[O:2])([O-:3])-[NH:4]-[#6,#7]",
        nl_map_nums=[1, 2, 4],
        nl_mass=44.0010,
        min_num_atoms=5,
    ),
    # Sulfoxide SO loss.
    # R–S(=O)–R' → [R–R'] + SO  (NL = 47.967 Da).
    # The C–S bonds break and a new C–C bond forms in the product; retained atoms
    # may be disconnected in the original graph (+2H excess; correct formula at H_delta=-2).
    # SX3 matches only sulfoxides (3-connected S), excluding sulfones (SX4) and thioethers.
    NeutralLossRule(
        name="sulfoxide_so_loss",
        reactant_smarts="[#6:1]-[SX3:2](=[O:3])-[#6:4]",
        nl_map_nums=[2, 3],
        nl_mass=47.9669,
        min_num_atoms=3,
    ),
    # Diarylethylene (stilbene-type) C₂H₂ extrusion.
    # Ar–CH=CH–Ar' → [Ar–Ar'] + HC≡CH  (NL = 26.016 Da).
    # The two vinyl CH atoms (:2, :3), each bearing one H, leave as acetylene (C₂H₂).
    # A new Ar–Ar' bond forms in the product (not present in the precursor), making
    # this a rearrangement outside BFS DAG coverage. Product atoms are disconnected
    # in the original graph; H-cap adds +1H at each free Ar valence, so the correct
    # biaryl formula lands at H_delta=-2 (two excess H from the two cut free valences).
    # `!R` on :2/:3 excludes bridged vinyl units in fused rings (indene, acenaphthylene)
    # where the vinyl CH=CH is part of a ring and BFS already covers the 2-bond cut.
    # min_num_atoms=8: minimum diarylethylene is two 5-membered aromatics + 2 vinyl C.
    NeutralLossRule(
        name="diarylvinyl_c2h2_loss",
        reactant_smarts="[c:1]-[CH;!R:2]=[CH;!R:3]-[c:4]",
        nl_map_nums=[2, 3],
        nl_mass=26.0157,
        min_num_atoms=8,
    ),
    # Diaryl ketone CO extrusion (negative mode).
    # Ar–C(=O)–Ar' → [Ar–Ar']⁻ + CO  (NL = 27.995 Da).
    # The keto carbon (:2) and its oxygen (:3) leave as CO; a new Ar–Ar' bond forms
    # in the product.  Cutting both Ar–CO bonds would give 3 CCs (BFS-invalid), so the
    # merged {Ar+Ar'} atom mask is outside BFS coverage.  H-cap adds +1H at each free
    # Ar valence; correct biaryl formula at H_delta=-2.
    # `!R` on :2 excludes ring ketones (chromone, fluorenone) where a 2-CC BFS cut
    # already covers the product; `[c:1]` and `[c:4]` restrict to diaryl ketones
    # (excludes acetophenone, chalcone, aldehyde, carboxylic acid).
    # min_num_atoms=12: minimum is two 5-membered aromatic rings + keto CO (5+5+2).
    NeutralLossRule(
        name="diaryl_ketone_co_loss",
        reactant_smarts="[c:1]-[CX3;!R:2](=[O:3])-[c:4]",
        nl_map_nums=[2, 3],
        nl_mass=27.9949,
        min_num_atoms=12,
    ),
    # 2-Aminobenzothiazole HCN extrusion (direct-cation ring fragmentation).
    # The thiazole ring's C2=N3 unit expels HCN (using one H from the exocyclic
    # amine), and the remaining fragment retains ring S1 and the exocyclic amine
    # attached to the benzo ring (aminothiophenol-type cation).  A new Ar–NH2 bond
    # forms, so the product atoms are disconnected in the original graph; H-cap
    # adds excess H at C3a (broken N3 bond), S1 (thiol H), and the amine (lost its
    # only bond) — correct formula lands at a negative H-shift (max_h_transfer).
    # NL confirmed empirically: mol_id 27770 (spec) shows NL=27.0113 Da at
    # intensity 217/999 (2nd-most-intense peak); consistent NL≈27.011 across
    # 13/14 halogenated NIST20 2-aminobenzothiazoles checked.
    # min_num_atoms=10: bare 2-aminobenzothiazole (C7H6N2S) has 10 heavy atoms.
    NeutralLossRule(
        name="aminobenzothiazole_hcn_loss",
        reactant_smarts="[NX3H2]-[c:1]1[n:2]c2ccccc2s1",
        nl_map_nums=[1, 2],
        nl_mass=27.0109,
        min_num_atoms=10,
    ),
]

# Pre-compiled SMARTS patterns (keyed by rule name) — avoid recompiling per molecule.
_COMPILED_NL_PATTERNS: dict[str, Chem.Mol] = {}


def _get_compiled_nl_pattern(rule: NeutralLossRule) -> Chem.Mol | None:
    """Return a cached compiled SMARTS pattern for *rule*, compiling on first call."""
    if rule.name not in _COMPILED_NL_PATTERNS:
        _COMPILED_NL_PATTERNS[rule.name] = Chem.MolFromSmarts(rule.reactant_smarts)
    return _COMPILED_NL_PATTERNS[rule.name]


def _apply_nl_prepass(
    rules: list[NeutralLossRule],
    mol: Chem.Mol,
    num_nodes: int,
) -> list[tuple[np.ndarray, int]]:
    """Apply neutral-loss rules to the root molecule.

    For each rule, matches the reactant SMARTS against *mol* and returns a
    product atom mask that excludes the neutral-loss atoms.  Because the
    product requires new bond formation, the retained atoms may be disconnected
    in the original graph; H-count estimation adds excess H's at the free
    valences.  The correct formula therefore lands at a negative H-shift
    (see :class:`NeutralLossRule` docstring).

    Args:
        rules: List of :class:`NeutralLossRule` to apply.
        mol: RDKit molecule.
        num_nodes: Number of heavy atoms.

    Returns:
        List of ``(product_mask, rule_idx)`` tuples.  *product_mask* is a
        ``(num_nodes,)`` uint8 array with 1 for retained atoms and 0 for
        neutral-loss atoms.  Duplicate NL atom sets (same atoms lost via
        different SMARTS embeddings) are deduplicated.
    """
    results: list[tuple[np.ndarray, int]] = []
    for rule_idx, rule in enumerate(rules):
        if rule.min_num_atoms > 0 and num_nodes < rule.min_num_atoms:
            continue
        patt = _get_compiled_nl_pattern(rule)
        if patt is None:
            continue
        q_idx_to_map: dict[int, int] = {}
        for q_idx in range(patt.GetNumAtoms()):
            map_num = patt.GetAtomWithIdx(q_idx).GetAtomMapNum()
            if map_num != 0:
                q_idx_to_map[q_idx] = map_num
        seen_nl_sets: set[frozenset] = set()
        for match in mol.GetSubstructMatches(patt):
            map_to_mol: dict[int, int] = {m: match[q] for q, m in q_idx_to_map.items()}
            nl_atoms: set[int] = set()
            valid = True
            for nl_map in rule.nl_map_nums:
                mol_idx = map_to_mol.get(nl_map)
                if mol_idx is None:
                    valid = False
                    break
                nl_atoms.add(mol_idx)
            if not valid:
                continue
            nl_key = frozenset(nl_atoms)
            if nl_key in seen_nl_sets:
                continue
            seen_nl_sets.add(nl_key)
            product_mask = np.ones(num_nodes, dtype=MASK_DTYPE)
            for idx in nl_atoms:
                product_mask[idx] = 0
            results.append((product_mask, rule_idx))
    return results
