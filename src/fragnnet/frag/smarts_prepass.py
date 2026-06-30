"""SMARTS-based rearrangement prepass for multi-bond BFS fragmentation.

Contains :class:`SmartsFragRule`, :data:`FRAG_RULES`, and
:func:`_apply_smarts_prepass` — the pure-Python prepass for reactions that
require new bond formation (1,3-elimination, acyclic ester CO₂ loss).

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
