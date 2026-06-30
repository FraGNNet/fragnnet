# from copy import deepcopy
import bz2
import gzip
import json
import logging
import os
import pickle
import sys
import tarfile
import traceback
from collections import Counter
from hashlib import blake2b
from typing import TYPE_CHECKING
from zipfile import ZipFile

if TYPE_CHECKING:
    import h5py as _h5py

import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import rdkit.Chem.rdmolops as rdmolops
import torch as th
import torch.nn.functional as F
import torch_geometric as pyg

import fragnnet.frag.compute_frags as compute_frags
from fragnnet.frag.compute_frags import (
    MASK_DTYPE,
    MAX_NUM_EDGES,
    MAX_NUM_NODES,
)
from fragnnet.frag.multi_cut_bfs import compute_ccs_multi_cut as _compute_ccs_multi_cut_cy
from fragnnet.frag.multi_cut_bfs import get_ring_edge_mask
from fragnnet.frag.smarts_prepass import FRAG_RULES, _apply_smarts_prepass
from fragnnet.utils.data_utils import mol_from_smiles, par_apply
from fragnnet.utils.formula_utils import (
    PREC_TYPE_TO_CMF_MASS_DIFF,
    PREC_TYPE_TO_MASS_DIFF,
    formula_to_peak_mzs,
)
from fragnnet.utils.misc_utils import PPM, progress_wrapper

# Fragment index dtype (change here to update everywhere)
# uint32 supports up to ~4B unique formulas per molecule (uint16 max is 65535)
FRAG_INDEX_DTYPE = np.uint32

# full list of common elements
# NOTE: this maybe a bad idea to not cover all the element
# we should handle all the elemnet
ELEMENT_TO_VE = {
    "C": 4,
    "O": 2,
    "N": 3,
    "P": 3,  # up to 5
    "S": 2,  # up to 6
    "F": 1,
    "Cl": 1,
    "Br": 1,
    "I": 1,
    "Se": 2,  # up to 6, same as S
    "Si": 4,
    "As": 3,  # up to 5
    "B": 3,
    # Tier 1 metals — covalent organometallics only (C–metal single bonds)
    "Hg": 2,  # methylmercury, ethylmercury
    "Sn": 4,  # organotin (tributyltin, trimethyltin)
    "Pb": 4,  # organolead (tetraethyllead)
}
HEAVY_ELEMENTS = list(ELEMENT_TO_VE.keys())
NUM_HEAVY_ELEMENTS = len(HEAVY_ELEMENTS)
ELEMENTS = HEAVY_ELEMENTS + ["H"]
NUM_ELEMENTS = len(ELEMENTS)

CANONICAL_ELEMENT_ORDER = ["C", "H"] + sorted([elem for elem in HEAVY_ELEMENTS if elem != "C"])
CANONICAL_H_IDX = CANONICAL_ELEMENT_ORDER.index("H")

ELEMENT_TO_IDX = {elem: idx for idx, elem in enumerate(ELEMENT_TO_VE.keys())}
ELEMENT_TO_IDX["H"] = len(ELEMENT_TO_VE.keys())

IDX_TO_ELEMENT = {idx: elem for elem, idx in ELEMENT_TO_IDX.items()}

MAX_H_TRANSFER = 4
MAX_NUM_MZS_PER_FORMULA = 5

NODE_FEAT_DTYPE = th.int64
EDGE_FEAT_DTYPE = th.int64
META_DATA_DTYPE = th.float32

MASK_SIZE = 128
assert MASK_SIZE >= MAX_NUM_NODES, "MASK_SIZE should be larger than MAX_NUM_NODES"
assert MASK_SIZE % 64 == 0, "MASK_SIZE should be a multiple of 64"

BOND_TYPE_TO_IDX = {
    Chem.rdchem.BondType.names["AROMATIC"]: 2,
    Chem.rdchem.BondType.names["DOUBLE"]: 2,
    Chem.rdchem.BondType.names["TRIPLE"]: 3,
    Chem.rdchem.BondType.names["SINGLE"]: 1,
}

NODE_FEAT_TO_IDX = {
    "depth": 0,
    "cc": 1,
    "base_formula": 2,
    "h_formulae_idx": 3,
    "h_counts": 4,
    "nb_iso_idx": 5,
    "cmf_h_formulae_idx": 6,
}

EDGE_FEAT_TO_IDX = {"cc": 0, "base_formula": 1, "h_range": 2, "complement": 3}


def convert_cc_mask_to_int(cc: list | np.ndarray) -> int:
    """covert a cc mask to an int

    Args: a list of 0 and 1s

    Returns:
        int: int presentation of input mask
    """

    return sum([2**i for i in cc])


def convert_cc_int_to_mask(num_nodes: int, cc_int: int, bitmask=True) -> tuple[int]:
    """_summary_

    Args:
        num_nodes (int): number of nodes
        cc_int (int): int version of cc mask
        bitmask (bool, optional): _description_. Defaults to True.

    Returns:
        list|np.ndarray: a list of 0 and 1s
    """

    cc = []
    quot = cc_int
    for i in range(num_nodes):
        quot, rem = divmod(quot, 2)
        if bitmask:
            cc.append(int(rem))
        else:
            if rem == 1:
                cc.append(i)
    return cc


def convert_cc_int_to_np_mask(num_nodes: int, cc_int: int, bitmask=True) -> np.ndarray:
    """_summary_

    Args:
        num_nodes (int): number of nodes
        cc_int (int): int version of cc mask
        bitmask (bool, optional): _description_. Defaults to True.

    Returns:
        list|np.ndarray: a list of 0 and 1s
    """

    cc = convert_cc_int_to_mask(num_nodes, cc_int, bitmask)
    np_mask = np.array(cc, dtype=MASK_DTYPE)
    return np_mask


def cc_bit_mask_to_atom_idx(cc: list | np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        cc (list | np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    cc_np = np.array(cc) if isinstance(cc, list) else cc
    atom_ids = np.where(cc_np == 1)[0].astype(np.int32)
    return atom_ids


def get_fraggen_input_arrays(mol_d: dict):
    """_summary_

    Args:
        mol_d (_type_): _description_

    Returns:
        _type_: _description_
    """
    num_nodes = mol_d["atom_mask_arr"].shape[0]
    num_edges = mol_d["bond_mask_arr"].shape[0]
    assert num_nodes <= MAX_NUM_NODES, num_nodes
    assert num_edges <= MAX_NUM_EDGES, num_edges
    edges = np.zeros((MAX_NUM_EDGES, 2), dtype=np.intc)
    for bond_idx, bond in enumerate(mol_d["bonds"]):
        edges[bond_idx, 0] = bond[0]
        edges[bond_idx, 1] = bond[1]
    node_to_edge_idx = compute_frags.py_compute_node_to_edge_idx(num_nodes, num_edges, edges)
    # print(node_to_edge_idx)
    node_mask = np.zeros((num_nodes,), dtype=MASK_DTYPE)
    node_mask[:num_nodes] = 1
    edge_mask = np.zeros((num_edges,), dtype=MASK_DTYPE)
    edge_mask[:num_edges] = 1
    return num_nodes, num_edges, node_mask, edges, edge_mask, node_to_edge_idx


def extract_mol_info(smiles_or_mol, use_default_valence=False) -> dict:
    """Method to exatract mol infomation in to dict

    Args:
        smiles_or_mol (_type_): _description_
        use_default_valence (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_

    Returns:
        dict :
        mol_d["mol"]: mol object
        mol_d["num_hs"]: number of totoal Hs
        mol_d["sbond_arr"]: single bond array
        mol_d["ve_arr"]: max velance array
        mol_d["hs_arr"]: max Hs per atom arary
        mol_d["elems"]: elemns per atom
        mol_d["bonds"]: a list of from atom - to atom pairs
        mol_d["bond_mask_arr"]: defualt bond mask, this should be just 1s
        mol_d["atom_mask_arr"] : defualt atom  mask, this should be just 1s
        mol_d["atoms_to_bonds"]:
        mol_d["element_counts"] : atom count per element
    """
    if isinstance(smiles_or_mol, str):
        mol = mol_from_smiles(smiles_or_mol)
    else:
        assert isinstance(smiles_or_mol, Chem.rdchem.Mol), type(smiles_or_mol)
        mol = smiles_or_mol

    pt = Chem.GetPeriodicTable()
    # GetValenceList
    # some checks
    charge = rdmolops.GetFormalCharge(mol)
    if charge not in {0, 1}:
        raise ValueError(
            f"unsupported molecular charge {charge}; only 0 (neutral) or +1 ([M]+ cations) supported"
        )
    # enumerate atoms
    sbond_arr = []
    ve_arr = []
    hs_arr = []
    elems = []
    elem_idxs = []
    num_hs = 0
    num_atoms = 0
    num_radicals = 0
    element_counts = dict.fromkeys(ELEMENT_TO_VE.keys(), 0)

    for atom in mol.GetAtoms():
        cur_idx = atom.GetIdx()
        cur_num_hs = atom.GetTotalNumHs()
        cur_deg = atom.GetTotalDegree()
        # cur_num_bonds = atom.GetNumBonds()
        cur_element = atom.GetSymbol()

        if cur_element not in element_counts:
            raise ValueError(f"Molecules with {cur_element} atom(s) currently not supported")

        element_counts[cur_element] += 1
        elems.append(cur_element)
        elem_idxs.append(ELEMENT_TO_IDX[cur_element])
        # number of single bond need to attach to this atom to keep atom connected
        # this equals to replace all all bond to single, and count how many single bond
        sbond_arr.append(cur_deg - cur_num_hs)
        # ve_arr.append(ELEMENT_TO_VE[cur_element])

        # set valence value for each atom
        # assumption we will use current valance unless
        # use default valence flag is set to true
        # not defualt valence can be -1 for transition metals
        # on paper we should not encounter them at all
        ve_value = atom.GetTotalValence()
        # For cationic atoms with no H (e.g. [N+] in nitro, N-oxide), the
        # formal charge funds an extra heavy-atom bond rather than an H slot.
        # Subtracting FC prevents the H-cap from being inflated by 1 per such
        # atom.  The guard cur_num_hs == 0 preserves correctness for protonated
        # amines ([NH3+], [NH2+]) where the extra valence IS used for H.
        if atom.GetFormalCharge() > 0 and cur_num_hs == 0:
            ve_value -= atom.GetFormalCharge()
        if use_default_valence:
            default_valence = pt.GetDefaultValence(cur_element)
            if default_valence != -1:
                ve_value = min(ve_value, default_valence)

        ve_arr.append(ve_value)
        hs_arr.append(cur_num_hs)
        num_hs += cur_num_hs
        num_atoms += 1
        num_radicals += atom.GetNumRadicalElectrons()

    element_counts["H"] = num_hs
    assert num_radicals == 0, num_radicals
    sbond_arr = np.array(sbond_arr, dtype=np.int32)
    ve_arr = np.array(ve_arr, dtype=np.int32)
    hs_arr = np.array(hs_arr, dtype=np.int32)
    # enumerate bonds
    atoms_to_bonds = {}
    bonds, bond_type_idxs, ring_bond_mask = [], [], []
    num_bonds = 0
    adj = np.zeros((num_atoms, num_atoms), dtype=np.int32)
    # adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
    for bond in mol.GetBonds():
        cur_idx = bond.GetIdx()
        from_idx = bond.GetBeginAtomIdx()
        to_idx = bond.GetEndAtomIdx()
        cur_type_idx = BOND_TYPE_TO_IDX[bond.GetBondType()]
        assert from_idx != to_idx
        adj[from_idx, to_idx] = 1

        bonds.append((from_idx, to_idx))
        bond_type_idxs.append(cur_type_idx)
        ring_bond_mask.append(1 if bond.IsInRing() else 0)

        if from_idx not in atoms_to_bonds:
            atoms_to_bonds[from_idx] = [cur_idx]
        else:
            atoms_to_bonds[from_idx].append(cur_idx)

        if to_idx not in atoms_to_bonds:
            atoms_to_bonds[to_idx] = [cur_idx]
        else:
            atoms_to_bonds[to_idx].append(cur_idx)
        num_bonds += 1
    bonds = np.array(bonds, dtype=np.int32)
    mol_d = {}
    mol_d["mol"] = mol
    mol_d["num_hs"] = num_hs
    mol_d["sbond_arr"] = sbond_arr
    mol_d["ve_arr"] = ve_arr
    mol_d["hs_arr"] = hs_arr
    mol_d["elems"] = elems
    mol_d["elem_idxs"] = elem_idxs
    mol_d["bonds"] = bonds
    mol_d["bond_type_idxs"] = bond_type_idxs
    mol_d["ring_bond_mask"] = np.array(ring_bond_mask, dtype=np.uint8)
    mol_d["bond_mask_arr"] = np.ones((num_bonds,), dtype=bool)
    mol_d["atom_mask_arr"] = np.ones((num_atoms,), dtype=bool)  # can be computed
    mol_d["atoms_to_bonds"] = atoms_to_bonds  # can be computed
    mol_d["element_counts"] = element_counts
    return mol_d


def compute_cc_h_cap(
    cc_atom_ids: np.ndarray, ve_arr: np.ndarray, sbond_arr: np.ndarray, num_radicals: int
):
    """compute max amount of Hs a cc can have.
        For any ccs the max amount of Hs it can have is the congifcation where all the bond are single
        And all the atom has max amount of Hs
    Args:
        cc (list|np.ndarray): cc mask in list form
        ve_arr (list|np.ndarray): max velance each atom can have
        sbond_arr (_type_): _description_
        num_radicals (_type_): _description_

    Returns:
        _type_: _description_
    """

    assert num_radicals == 0
    if not isinstance(cc_atom_ids, np.ndarray):
        cc_atom_ids = np.array(list(cc_atom_ids))
    cap_sbond_arr = sbond_arr[cc_atom_ids]
    cap_ve_arr = ve_arr[cc_atom_ids]
    cap_ve_mask = cap_ve_arr < cap_sbond_arr
    cap_ve_arr[cap_ve_mask] = cap_sbond_arr[cap_ve_mask]
    return np.sum(cap_ve_arr) - np.sum(cap_sbond_arr) - num_radicals


def compute_cc_h_floor(
    cc_atom_ids: np.ndarray,
    ve_arr: np.ndarray,
    sbond_arr: np.ndarray,
    num_radicals: int,
    bonds: np.ndarray,
    atoms_to_bonds: dict,
    bond_mask_arr: np.ndarray,
):
    """compute min amount of Hs a cc can have.

    Args:
        cc_atom_ids (np.ndarray): atom ids in the cc
        ve_arr (np.ndarray): _description_
        sbond_arr (np.ndarray): _description_
        bonds (np.ndarray): _description_
        atoms_to_bonds (dict): _description_
        bond_mask_arr (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    assert num_radicals == 0
    # if we could use update from single bond to double bond to get an electron pair
    diff_arr = np.maximum(ve_arr - sbond_arr, 0)
    # cc_atoms = list(cc_atom_ids)
    # print(diff_arr)
    # this computes a lower bound
    h_arr = np.copy(diff_arr)

    for _, atom in enumerate(cc_atom_ids):
        bond_idxs = atoms_to_bonds[atom]
        for bond_idx in bond_idxs:
            if h_arr[atom] == 0:
                break
            if not bond_mask_arr[bond_idx]:
                continue
            bond = bonds[bond_idx]
            if bond[0] == atom:
                other = bond[1]
            else:
                other = bond[0]
            # dont't form more than 3 bonds with anything!
            h_arr[atom] = max(0, h_arr[atom] - min(diff_arr[other], 2))
    # print(h_arr)
    cc_floor = sum(h_arr[atom] for atom in cc_atom_ids)
    cc_floor -= num_radicals
    cc_floor = max(cc_floor, 0)  # why can cc_floor be negative?
    return cc_floor


def compute_approximate_formula(
    cc: list | np.ndarray,
    mol_d: dict,
    max_h_transfer: int,
    formula_strs: bool = False,
    bitmask: bool = True,
    base_formula=None,
) -> tuple[dict[int, str] | dict[int, np.ndarray], dict[int, int]]:
    """
    Given a connected component (fragment) of a molecule, compute all possible formulas
    within a specified hydrogen transfer range (delta H), and return either formula strings
    or hydrogen counts for each possible transfer.

    Args:
        cc (list | np.ndarray): Connected component, either as a bitmask or atom indices.
        mol_d (dict): Molecule dictionary containing atom/bond info and arrays.
        max_h_transfer (int): Maximum allowed hydrogen transfer (delta H) for enumeration.
        formula_strs (bool, optional): If True, return formula strings; if False, return formula arrays. Defaults to False.
        bitmask (bool, optional): If True, interpret cc as bitmask; if False, as atom indices. Defaults to True.
        base_formula (np.ndarray, optional): Precomputed base formula array for the fragment. If None, computed internally. Defaults to None.

    Returns:
        tuple[dict[int, str] | dict[int, np.ndarray], dict[int, int]]:
            - delta_h_to_formula: Maps delta H (hydrogen transfer) to formula string (if formula_strs=True)
              or formula array (if formula_strs=False) for each possible configuration.
            - delta_h_to_h_count: Maps delta H to the corresponding hydrogen count, or -1 for invalid configurations.
    """

    # Extract molecule arrays and info
    bonds = mol_d["bonds"]  # Bond definitions (atom id pairs)
    atoms_to_bonds = mol_d["atoms_to_bonds"]  # Atom to bond mapping
    ve_arr = np.copy(mol_d["ve_arr"])
    sbond_arr = np.copy(mol_d["sbond_arr"])
    num_hs = mol_d["num_hs"]  # Total number of hydrogens in molecule
    hs_arr = mol_d["hs_arr"]  # Hydrogens per atom
    elem_idxs = mol_d["elem_idxs"]  # Element indices per atom
    bond_mask_arr = mol_d["bond_mask_arr"]

    # Compute base formula for fragment if not provided
    if base_formula is None and bitmask:
        base_formula = cc_bitmask_to_formula_arr(cc, elem_idxs)
    elif base_formula is None and not bitmask:
        base_formula = cc_to_formula_arr(cc, elem_idxs, bitmask=False)

    # Get atom indices for fragment
    atom_ids = cc_bit_mask_to_atom_idx(cc) if bitmask else cc

    # Update single bond and bond mask arrays for fragment
    sbond_arr, bond_mask_arr = compute_frags.update_bonds(
        atom_ids, sbond_arr, bond_mask_arr, bonds, atoms_to_bonds
    )
    # Compute hydrogen cap and floor for fragment
    cap = compute_cc_h_cap(atom_ids, ve_arr, sbond_arr, 0)
    floor = compute_frags.compute_cc_h_floor(
        atom_ids, ve_arr, sbond_arr, 0, bonds, atoms_to_bonds, bond_mask_arr
    )

    # Defensive checks on hydrogen bounds
    assert floor <= cap, (floor, cap, atom_ids)
    assert floor >= 0, floor

    # Count hydrogens present in fragment before transfer
    num_hs_prior = sum(hs_arr[atom] for atom in atom_ids)

    # Restrict hydrogen range to allowed transfer window
    floor = max(floor, num_hs_prior - max_h_transfer)
    cap = min(cap, num_hs_prior + max_h_transfer, num_hs)
    # Guard: the actual H count in the molecule is always a valid configuration.
    # compute_cc_h_floor can overcount for fragments containing nitro groups because
    # update_bonds treats all bonds as single bonds — the double bond O= in [N+](=O)[O-]
    # appears to have 1 free valence slot (ve=2, sbond=1), but its only neighbor N+ has
    # diff=0 (after formal-charge correction), so the floor can't be reduced and ends up
    # > num_hs_prior, making delta_h=0 invalid and assigning "" to the node formula.
    floor = min(floor, num_hs_prior)

    delta_h_to_formula = {}  # Maps delta H to formula string/array
    delta_h_to_h_count = {}  # Maps delta H to hydrogen count (or -1 if invalid)

    # Prepare formula string templates if requested
    if formula_strs:
        formula_template, formual_no_h = formula_arr_to_str(base_formula, get_h_template=True)

    # Enumerate all possible hydrogen transfers
    for delta_h in range(-max_h_transfer, max_h_transfer + 1):
        h = num_hs_prior + delta_h  # Proposed hydrogen count for fragment
        if h < floor or h > cap:
            # Invalid configuration: assign zero formula and -1 count
            formula = np.zeros_like(base_formula)
            formula_str = ""
            delta_h_to_h_count[delta_h] = -1
        else:
            # Valid configuration: update formula with new hydrogen count
            formula = np.copy(base_formula)
            formula[ELEMENT_TO_IDX["H"]] = h
            formula = tuple(formula)
            if h == 0:
                formula_str = formual_no_h
            else:
                formula_str = formula_template.format(h)
            delta_h_to_h_count[delta_h] = h
        # Store formula string if requested, else array/tuple
        if formula_strs:
            formula = formula_str
        delta_h_to_formula[delta_h] = formula
    return delta_h_to_formula, delta_h_to_h_count


def update_bonds(
    cc_atom_ids: np.ndarray,
    sbond_arr: np.ndarray,
    bond_mask_arr: np.ndarray,
    bonds: np.ndarray,
    atoms_to_bonds: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Update the single bond array and bond mask array for a given fragment (connected component).
    Only bonds where both atoms are in the fragment are considered.

    Args:
        cc_atom_ids (np.ndarray): Atom indices in the fragment.
        sbond_arr (np.ndarray): Array of single bond counts per atom (will be overwritten).
        bond_mask_arr (np.ndarray): Boolean mask for bonds (will be overwritten).
        bonds (np.ndarray): Array of bond pairs (atom indices).
        atoms_to_bonds (dict): Mapping from atom index to list of bond indices.

    Returns:
        tuple:
            - sbond_arr (np.ndarray): Updated single bond counts for fragment atoms.
            - bond_mask_arr (np.ndarray): Updated mask indicating which bonds are within the fragment.
    """
    # Reset arrays for fragment
    sbond_arr = np.zeros_like(sbond_arr, dtype=sbond_arr.dtype)
    bond_mask_arr = np.zeros_like(bond_mask_arr, dtype=bool)
    # For each atom, count bonds to other atoms in fragment
    for atom in cc_atom_ids:
        for bond_idx in atoms_to_bonds[atom]:
            bond = bonds[bond_idx]
            if bond[0] in cc_atom_ids and bond[1] in cc_atom_ids:
                sbond_arr[atom] += 1
                bond_mask_arr[bond_idx] = True
    return sbond_arr, bond_mask_arr


def compute_approximate_cchs(ccs, mol_d, h_prior=False, max_h_transfer=MAX_H_TRANSFER):
    """
    Enumerate all valid (fragment, hydrogen count) pairs for a set of fragments (connected components).
    For each fragment, compute the allowed hydrogen count range and return all valid combinations.

    Args:
        ccs (iterable): List or set of fragments (bitmask or atom indices).
        mol_d (dict): Molecule dictionary with atom/bond info and arrays.
        h_prior (bool, optional): If True, restrict hydrogen range based on prior hydrogens. Defaults to False.
        max_h_transfer (int, optional): Maximum allowed hydrogen transfer. Defaults to MAX_H_TRANSFER.

    Returns:
        set: Set of (fragment, hydrogen count) tuples for all valid configurations.
    """
    bonds = mol_d["bonds"]  # Bond definitions
    atoms_to_bonds = mol_d["atoms_to_bonds"]
    ve_arr = np.copy(mol_d["ve_arr"])
    sbond_arr = np.copy(mol_d["sbond_arr"])
    num_hs = mol_d["num_hs"]
    hs_arr = mol_d["hs_arr"]
    bond_mask_arr = mol_d["bond_mask_arr"]
    all_cchs = set()
    # For each fragment, enumerate valid hydrogen counts
    for cc in list(set(ccs)):
        sbond_arr, bond_mask_arr = compute_frags.update_bonds(
            cc, sbond_arr, bond_mask_arr, bonds, atoms_to_bonds
        )
        cc_atom_ids = cc_bit_mask_to_atom_idx(cc)
        cap = compute_cc_h_cap(cc_atom_ids, ve_arr, sbond_arr, 0)
        floor = compute_frags.compute_cc_h_floor(
            cc_atom_ids, ve_arr, sbond_arr, 0, bonds, atoms_to_bonds, bond_mask_arr
        )
        assert floor <= cap, (floor, cap, cc)
        assert floor >= 0, floor
        if h_prior:
            num_hs_prior = sum(hs_arr[atom] for atom in cc)
            floor = max(floor, num_hs_prior - max_h_transfer)
            cap = min(cap, num_hs_prior + max_h_transfer, num_hs)
        else:
            cap = min(cap, num_hs)
        for h in range(floor, cap + 1):
            all_cchs.add((cc, h))
    return all_cchs


def cc_to_formula_arr(cc, elems, bitmask=False):
    """
    Convert a fragment (connected component) to a formula array (element counts, excluding hydrogens).

    Args:
        cc (list | np.ndarray): Fragment as bitmask or atom indices.
        elems (list): List of element symbols for each atom in molecule.
        bitmask (bool, optional): If True, interpret cc as bitmask; else as atom indices. Defaults to False.

    Returns:
        np.ndarray: Array of element counts (excluding hydrogens).
    """
    formula_arr = np.zeros([len(ELEMENT_TO_IDX)], dtype=int)
    if bitmask:
        for i in range(len(cc)):
            if cc[i]:
                elem = elems[i]
                formula_arr[ELEMENT_TO_IDX[elem]] += 1
    else:
        for atom in cc:
            elem = elems[atom]
            formula_arr[ELEMENT_TO_IDX[elem]] += 1
    return formula_arr


def cc_bitmask_to_formula_arr(cc, elem_idxs) -> np.ndarray:
    """
    Fast conversion of a fragment bitmask to a formula array (element counts, excluding hydrogens).
    Uses element indices for efficient counting.

    Args:
        cc (list | np.ndarray): Fragment bitmask (0/1 per atom).
        elem_idxs (list | np.ndarray): Element indices for each atom in molecule.

    Returns:
        np.ndarray: Array of element counts (excluding hydrogens).
    """
    formula_arr = np.zeros([len(ELEMENT_TO_IDX)], dtype=MASK_DTYPE)
    elem_idxs_np = np.array(elem_idxs) + 1  # Offset by 1 for masking
    elem_idx_np = np.multiply(cc, elem_idxs_np)
    unique, counts = np.unique(elem_idx_np, return_counts=True)
    for elem_idx, count in zip(unique, counts):
        if elem_idx == 0:
            continue
        else:
            formula_arr[elem_idx - 1] = count
    return formula_arr


def formula_arr_to_str(
    formula_arr: np.ndarray, get_h_template: bool = False
) -> str | tuple[str, str]:
    """
    Convert a formula array (element counts) to a string in canonical element order.
    Optionally returns a template for hydrogen count substitution.

    Args:
        formula_arr (np.ndarray): Array of element counts (including hydrogen).
        get_h_template (bool, optional): If True, return template string for hydrogen count. Defaults to False.

    Returns:
        str | tuple[str, str]:
            - If get_h_template is False: Formula string (e.g., 'C6H12O6').
            - If get_h_template is True: Tuple of (template string with 'H{:d}' placeholder, string without hydrogen).
    """
    elem_d = {}
    for idx, count in enumerate(formula_arr):
        elem = IDX_TO_ELEMENT[idx]
        elem_d[elem] = count

    if not get_h_template:
        formula_str = ""
        for elem in CANONICAL_ELEMENT_ORDER:
            if elem in elem_d:
                count = elem_d[elem]
                if count > 0:
                    formula_str += elem
                if count > 1:
                    formula_str += str(count)
        return formula_str
    else:
        formula_str = ""
        formula_str_no_h = ""
        for elem in CANONICAL_ELEMENT_ORDER:
            if elem in elem_d:
                if elem == "H":
                    formula_str += "H{:d}"
                else:
                    count = elem_d[elem]
                    if count > 0:
                        formula_str += elem
                        formula_str_no_h += elem
                    if count > 1:
                        formula_str += str(count)
                        formula_str_no_h += str(count)
        return formula_str, formula_str_no_h


def compute_frag_peak_stats(
    peaks,
    formula_peak_mzs,
    formula_peak_probs,
    idx_by_h_delta,
    prec_mz,
    allowed_h_transfer,
    tolerance=0.01,
    prec_type="[M+H]+",
    is_ppm=False,
):
    """_summary_

    Args:
        peaks (_type_): _description_
        formula_peak_mzs (_type_): _description_
        formula_peak_probs (_type_): _description_
        idx_by_h_delta (_type_): _description_
        allowed_h_transfer (_type_): _description_
        tolerance (float, optional): _description_. Defaults to 0.01.
        is_ppm (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    true_mzs, true_ints = list(zip(*peaks))
    theoretical_mzs = formula_peak_mzs
    theoretical_probs = formula_peak_probs

    allowed_idx = list(idx_by_h_delta[0])
    for h in range(1, allowed_h_transfer):
        allowed_idx += list(idx_by_h_delta[2 * h - 1])
        allowed_idx += list(idx_by_h_delta[2 * h])
    allowed_idx = list(set(allowed_idx))
    # print(allowed_idx)
    indices = th.tensor(allowed_idx)
    theoretical_mzs = th.index_select(theoretical_mzs, 0, indices)
    theoretical_probs = th.index_select(theoretical_probs, 0, indices)

    prec_mask = (theoretical_probs > 0.0).type(th.float32)
    # Compute two sets of theoretical m/z values:
    #   - CMF: charge migrates to H+ before fragmentation (correct for [M+NH4]+, [M+Cl]-, etc.)
    #   - std: adduct mass added directly (correct for [M+H]+, [M-H]- where CMF == std)
    # A true peak is matched if it falls within tolerance of EITHER set.
    cmf_diff = PREC_TYPE_TO_CMF_MASS_DIFF.get(prec_type, PREC_TYPE_TO_MASS_DIFF[prec_type])
    std_diff = PREC_TYPE_TO_MASS_DIFF[prec_type]
    theoretical_mzs_cmf = theoretical_mzs + prec_mask * cmf_diff
    theoretical_mzs_std = theoretical_mzs + prec_mask * std_diff
    # compute overlap
    overlap_true_idxs = []
    overlap_true_ints = []
    overlap_pred_idxs = []
    overlap_pred_peak_counts = []
    overlap_pred_formula_counts = []

    # check true_mzs against union of CMF and standard theoretical peaks
    for true_idx, true_mz in enumerate(true_mzs):
        if not is_ppm:
            mz_close_cmf = th.abs(theoretical_mzs_cmf - true_mz) < tolerance
            mz_close_std = th.abs(theoretical_mzs_std - true_mz) < tolerance
        else:
            mz_close_cmf = th.abs(theoretical_mzs_cmf - true_mz) < (true_mz * tolerance * PPM)
            mz_close_std = th.abs(theoretical_mzs_std - true_mz) < (true_mz * tolerance * PPM)
        mz_close = mz_close_cmf | mz_close_std
        if th.any(mz_close):
            pred_idx = th.nonzero(mz_close, as_tuple=False)
            num_formula_match = th.sum(th.any(mz_close, dim=1).type(th.int32)).item()
            num_peak_match = pred_idx.shape[0]
            overlap_true_idxs.append(true_idx)
            overlap_true_ints.append(true_ints[true_idx])
            overlap_pred_idxs.append(pred_idx)
            overlap_pred_peak_counts.append(num_peak_match)
            overlap_pred_formula_counts.append(num_formula_match)
    # remove duplicates
    if len(overlap_pred_idxs) > 0:
        overlap_pred_idxs = th.unique(th.cat(overlap_pred_idxs, dim=0), dim=0)
    else:
        overlap_pred_idxs = th.zeros((0, 2))
    recall = len(overlap_true_idxs) / len(true_mzs)
    w_recall = np.sum(overlap_true_ints) / np.sum(true_ints)
    prec = len(overlap_pred_idxs) / th.sum((theoretical_probs > 0.0).type(th.int32)).item()
    if len(overlap_pred_peak_counts) > 0:
        ppt_peak = np.mean(overlap_pred_peak_counts)
        ppt_formula = np.mean(overlap_pred_formula_counts)
    else:
        ppt_peak = np.nan
        ppt_formula = np.nan
    # check prec_mz stuff
    prec_recalls = []
    for comp_mzs in [theoretical_mzs_std, th.tensor(true_mzs)]:
        prec_mz_diffs = th.abs(comp_mzs - prec_mz)
        if not is_ppm:
            prec_mz_close = prec_mz_diffs < tolerance
        else:
            prec_mz_close = prec_mz_diffs < (prec_mz * tolerance * PPM)
        if th.any(prec_mz_close):
            prec_recalls.append(1.0)
        else:
            prec_recalls.append(0.0)
    prec_recall, prec_spec_recall = prec_recalls
    return pd.Series([recall, w_recall, prec, ppt_peak, ppt_formula, prec_recall, prec_spec_recall])


def th_long_to_mask(long: th.Tensor) -> th.Tensor:
    """convert long to mask, this can be more then one 64 bit longs in given row

    Args:
        long (_type_): N x MASK_SIZE//64 th.Tensor of int
    Returns:
        _type_: _description_
    """
    # long is N x MASK_SIZE//64
    num_dims = long.shape[1]
    long = long.reshape(long.shape[0], num_dims, 1)
    mask = 2 ** th.arange(64 - 1, -1, -1, device=long.device)
    mask = mask.reshape(1, 1, 64)
    return long.bitwise_and(mask).ne(0).reshape(long.shape[0], -1)


def th_mask_to_long(mask):
    """convert torch binary mask to 64 bits ints, MASK_SIZE//64 number of int are returned per row
    Args:
        long (_type_): N x MASK_SIZE//64 th.Tensor of int
    Returns:
        _type_: _description_
    """
    # mask is N x MASK_SIZE
    num_dims = mask.shape[1] // 64
    mask = mask.reshape(mask.shape[0], num_dims, 64)
    long = 2 ** th.arange(64 - 1, -1, -1, device=mask.device).expand(1, num_dims, 64)
    return th.sum(long * mask, dim=2).long()


def _inject_smarts_pairs_into_dag(
    smarts_pairs: list[tuple[np.ndarray, np.ndarray]],
    nodes_mask_matrix: np.ndarray,
    nodes_depth_matrix: np.ndarray,
    dag_edges_matrix: np.ndarray,
    dag_frag_meta: dict,
    num_nodes: int,
    min_frag_atoms: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Inject SMARTS prepass fragment pairs into a DAG produced by compute_frags.

    Appends fragment pairs from the SMARTS rearrangement prepass as direct
    children of the root node (index 0).  Prepass rules require at least 2 bond
    cuts plus 1 new bond formation, so injected fragments receive **node depth 2**
    (not 1) to reflect the two-cut distance from the root.  The root→fragment
    edge retains ``min_depth=1`` since it is introduced in the same expansion
    pass as depth-1 BFS edges.  See ``docs/frag_gen_prepass.md`` for the full
    depth-semantics rationale.

    Fragments already present in the DAG are deduplicated via their byte-string
    node-mask key.  Fragments smaller than ``min_frag_atoms`` heavy atoms are
    silently dropped, consistent with the ``min_frag_atoms`` filter applied
    inside the BFS.

    Args:
        smarts_pairs: List of ``(node_mask_a, node_mask_b)`` uint8 array pairs
            returned by ``_apply_smarts_prepass``.
        nodes_mask_matrix: Shape ``(n_frags, num_nodes)`` uint8 — existing DAG
            fragment node masks, root at index 0.
        nodes_depth_matrix: Shape ``(n_frags, max_depth+1)`` uint8 — depth
            one-hot for each fragment.
        dag_edges_matrix: Shape ``(n_edges, 2)`` int64 — (parent, child) pairs.
        dag_frag_meta: Dict with keys ``reached_depth``, ``edges_min_depth``,
            ``nodes_min_depth``, ``force_stopped``.
        num_nodes: Number of atoms in the root molecule.
        min_frag_atoms: Minimum heavy-atom count for a fragment to be kept.

    Returns:
        Updated ``(nodes_mask_matrix, nodes_depth_matrix, dag_edges_matrix,
        dag_frag_meta)`` tuple with the same format as the input.
    """
    if not smarts_pairs:
        return nodes_mask_matrix, nodes_depth_matrix, dag_edges_matrix, dag_frag_meta

    max_depth_cols = nodes_depth_matrix.shape[1]

    # Build bytes-key → fragment index lookup from existing rows.
    key_to_idx: dict[bytes, int] = {
        nodes_mask_matrix[i].tobytes(): i for i in range(len(nodes_mask_matrix))
    }
    # Build set of existing (parent, child) edges for O(1) dedup.
    existing_edges: set[tuple[int, int]] = {
        (int(dag_edges_matrix[i, 0]), int(dag_edges_matrix[i, 1]))
        for i in range(len(dag_edges_matrix))
    }

    new_masks: list[np.ndarray] = []
    new_depth_rows: list[np.ndarray] = []
    new_min_depths: list[int] = []
    new_edges: list[tuple[int, int]] = []
    new_edge_min_depths: list[int] = []

    for mask_a, mask_b in smarts_pairs:
        for mask in (mask_a, mask_b):
            if int(mask.sum()) < min_frag_atoms:
                continue
            key = mask.tobytes()
            if key not in key_to_idx:
                new_idx = len(nodes_mask_matrix) + len(new_masks)
                key_to_idx[key] = new_idx
                new_masks.append(mask.astype(MASK_DTYPE))
                depth_row = np.zeros(max_depth_cols, dtype=MASK_DTYPE)
                # Prepass rules require ≥2 bond cuts + 1 new bond from the root,
                # so the node depth label is 2 (not 1).  See docs/frag_gen_prepass.md.
                depth_row[min(2, max_depth_cols - 1)] = 1
                new_depth_rows.append(depth_row)
                new_min_depths.append(2)
            frag_idx = key_to_idx[key]
            edge = (0, frag_idx)
            if edge not in existing_edges:
                existing_edges.add(edge)
                new_edges.append(edge)
                # Edge min_depth stays 1: the root→prepass edge is introduced
                # in the same expansion pass as depth-1 BFS edges.
                new_edge_min_depths.append(1)

    if not new_masks and not new_edges:
        return nodes_mask_matrix, nodes_depth_matrix, dag_edges_matrix, dag_frag_meta

    if new_masks:
        nodes_mask_matrix = np.vstack(
            [nodes_mask_matrix] + [m.reshape(1, num_nodes) for m in new_masks]
        )
        nodes_depth_matrix = np.vstack([nodes_depth_matrix] + new_depth_rows)
        dag_frag_meta["nodes_min_depth"] = np.concatenate(
            [dag_frag_meta["nodes_min_depth"], np.array(new_min_depths, dtype=MASK_DTYPE)]
        )

    if new_edges:
        dag_edges_matrix = np.vstack([dag_edges_matrix, np.array(new_edges, dtype=np.int64)])
        dag_frag_meta["edges_min_depth"] = np.concatenate(
            [dag_frag_meta["edges_min_depth"], np.array(new_edge_min_depths, dtype=MASK_DTYPE)]
        )

    return nodes_mask_matrix, nodes_depth_matrix, dag_edges_matrix, dag_frag_meta


def compute_dags(
    mol_d: dict,
    max_depth: int,
    h_prior: bool,
    max_h_transfer: int,
    frag_max_time: int,
    isotopes: bool = False,
    nb_isomorphic: bool = False,
    # b_isomorphic:bool = False, # not used
    wl_max_iterations: int = -1,
    multi_cut_bfs: bool = False,
    max_cut_size: int = 2,
    smarts_prepass: bool = False,
    min_frag_atoms: int = 0,
) -> dict:
    """_summary_

    Args:
        mol_d (dict): _description_
        max_depth (int): _description_
        h_prior (bool): _description_
        max_h_transfer (int): _description_
        frag_max_time (int): _description_
        isotopes (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_

    Returns:
        dict: _description_
    """

    assert h_prior in [True, False], h_prior
    num_nodes, num_edges, node_mask, edges, edge_mask, node_to_edge_idx = get_fraggen_input_arrays(
        mol_d
    )
    # time the recursive part
    # effectively this is infinite amount of time if None or <= 0
    if frag_max_time is None or frag_max_time <= 0:
        frag_max_time = int(1e6)

    node_mask = node_mask.astype(MASK_DTYPE)
    edge_mask = edge_mask.astype(MASK_DTYPE)
    if multi_cut_bfs:
        ring_edge_mask = get_ring_edge_mask(mol_d["mol"], num_edges)
        # Per-SSSR-ring bond groups: one list per ring, containing bond indices.
        # Used by cut=2/3 to pair bonds within the same individual ring rather
        # than the entire ring system, avoiding O(r²) cross-ring pairs that
        # can never disconnect a fused ring system.
        ring_bond_groups = [list(r) for r in mol_d["mol"].GetRingInfo().BondRings()]
        # SMARTS prepass: compute fragment pairs in Python (RDKit API), then pass
        # as smarts_seed_pairs to the Cython BFS for injection as depth-1 children.
        smarts_seed_pairs = None
        if smarts_prepass:
            smarts_seed_pairs_tagged = _apply_smarts_prepass(
                FRAG_RULES,
                mol_d["mol"],
                edge_mask,
                edges,
                num_nodes,
                num_edges,
            )
            # Pass 3-tuples directly; the caller only needs the fragment masks.
            smarts_seed_pairs = smarts_seed_pairs_tagged
        nodes_mask_matrix, nodes_depth_matrix, dag_edges_matrix, dag_frag_meta = (
            _compute_ccs_multi_cut_cy(
                num_nodes,
                num_edges,
                node_mask,
                edges,
                edge_mask,
                node_to_edge_idx,
                max_depth,
                frag_max_time,
                ring_edge_mask=ring_edge_mask,
                max_cut_size=max_cut_size,
                smarts_seed_pairs=smarts_seed_pairs,
                ring_bond_groups=ring_bond_groups,
                min_frag_atoms=min_frag_atoms,
            )
        )
    else:
        nodes_mask_matrix, nodes_depth_matrix, dag_edges_matrix, dag_frag_meta = (
            compute_frags.compute_ccs(
                num_nodes,
                num_edges,
                node_mask,
                edges,
                edge_mask,
                node_to_edge_idx,
                max_depth,
                frag_max_time,
                min_frag_atoms,
            )
        )
        if smarts_prepass:
            # Post-process: inject SMARTS rearrangement fragments as depth-1
            # children of the root.  compute_frags.compute_ccs runs unchanged
            # (preserving ring-opening via its n_ccs=1 path), and the SMARTS
            # pairs are merged in afterwards.  No further BFS expansion of the
            # SMARTS fragments is needed because their sub-fragments are already
            # covered by the BFS from the root molecule.
            smarts_pairs_tagged = _apply_smarts_prepass(
                FRAG_RULES,
                mol_d["mol"],
                edge_mask,
                edges,
                num_nodes,
                num_edges,
            )
            smarts_pairs = [(a, b) for a, b, _ in smarts_pairs_tagged]
            nodes_mask_matrix, nodes_depth_matrix, dag_edges_matrix, dag_frag_meta = (
                _inject_smarts_pairs_into_dag(
                    smarts_pairs,
                    nodes_mask_matrix,
                    nodes_depth_matrix,
                    dag_edges_matrix,
                    dag_frag_meta,
                    num_nodes,
                    min_frag_atoms,
                )
            )

    if nb_isomorphic:
        node_nb_hashes = get_subgraph_hashes(
            nodes_mask_matrix=nodes_mask_matrix,
            elems=mol_d["elems"],
            bond_type_idxs=mol_d["bond_type_idxs"],
            edges=edges[:num_edges],
            node_to_edge_idx=node_to_edge_idx[:num_nodes],
            include_bond_type=False,
            max_iterations=wl_max_iterations,
        )
    else:
        node_nb_hashes = None

    # if b_isomorphic:
    #   node_b_hashes = get_subgraph_hashes(
    #       nodes_mask_matrix=nodes_mask_matrix,
    #       elems=mol_d["elems"],
    #       bond_type_idxs=mol_d["bond_type_idxs"],
    #       edges=edges[:num_edges],
    #       node_to_edge_idx=node_to_edge_idx[:num_nodes],
    #       include_bond_type=True)
    # else:
    #   node_b_hashes = None

    # get meta
    reached_depth = dag_frag_meta["reached_depth"]
    edges_min_depth = dag_frag_meta["edges_min_depth"]
    nodes_min_depth = dag_frag_meta["nodes_min_depth"]
    force_stop = dag_frag_meta["force_stopped"]

    # add node depth information
    # convert to a one hot encoding
    # node feature size
    depth_node_feat_size = max_depth + 1  # depth
    cc_node_feat_size = MASK_SIZE // 64  # mask
    base_formula_node_feat_size = len(ELEMENT_TO_IDX)  # base_formula
    formula_node_feat_size = (
        1 + 2 * max_h_transfer
    )  # h_formulae_idx, at max we can have 1 + 2 * max_h_transfer different formula
    h_count_node_feat_size = formula_node_feat_size
    nb_iso_node_feat_size = 1 if nb_isomorphic else 0
    cmf_formula_node_feat_size = formula_node_feat_size  # 1+2*max_h_transfer # h_formulae_idx, at max we can have 1 + 2 * max_h_transfer different formula
    # edge feature sizes
    cc_edge_feat_size = cc_node_feat_size
    base_formula_edge_feat_size = base_formula_node_feat_size
    h_range_edge_feat_size = 2
    node_feat_shapes = [
        depth_node_feat_size,
        cc_node_feat_size,
        base_formula_node_feat_size,
        formula_node_feat_size,
        h_count_node_feat_size,
        nb_iso_node_feat_size,
        cmf_formula_node_feat_size,
    ]
    edge_feat_shapes = [cc_edge_feat_size, base_formula_edge_feat_size, h_range_edge_feat_size]

    # add node h count and element information
    hs_arr = mol_d["hs_arr"]
    elem_idxs = mol_d["elem_idxs"]
    cc_formula_list = []
    hs_arr_np = np.array(hs_arr)
    for cc_mask in nodes_mask_matrix:
        cc_h_count = np.sum(np.multiply(cc_mask, hs_arr_np))
        # Cast to uint16 before setting H count: uint8 overflows for fragments with >255 H atoms
        # (possible when MAX_NUM_NODES=128 and ~4 H per heavy atom → up to 512 H)
        cc_formula = cc_bitmask_to_formula_arr(cc_mask, elem_idxs).astype(np.uint16)
        cc_formula[ELEMENT_TO_IDX["H"]] = cc_h_count
        cc_formula_list.append(cc_formula)
    node_base_formula_matrix = np.stack(cc_formula_list, dtype=np.uint16)

    # map nodes to formulae
    formula_d_list, h_count_d_list = [], []
    formula_counts = {}
    for idx, cc_mask in enumerate(nodes_mask_matrix):
        base_formula = node_base_formula_matrix[idx]
        base_formula[ELEMENT_TO_IDX["H"]] = 0  # remove Hs
        delta_h_to_formula, delta_h_to_h_count = compute_approximate_formula(
            cc_mask, mol_d, max_h_transfer, formula_strs=True, base_formula=base_formula
        )
        formula_d_list.append(delta_h_to_formula)
        h_count_d_list.append(delta_h_to_h_count)
        formulae = list(delta_h_to_formula.values())
        for formula in formulae:
            formula_counts[formula] = formula_counts.get(formula, 0) + 1

    # map formulae indices
    formula_to_idx = {
        formula: idx for idx, formula in enumerate(sorted(list(formula_counts.keys())))
    }
    idx_to_formula = {idx: formula for formula, idx in formula_to_idx.items()}
    formula_idx_by_h_delta = [set() for _ in range(1 + 2 * max_h_transfer)]

    formula_idx_list, h_count_list = [], []
    for formulae_dict, h_count_dict in zip(formula_d_list, h_count_d_list):
        # idxs have value range 0 - uint32 max
        formulae_idxs = np.zeros(formula_node_feat_size, dtype=FRAG_INDEX_DTYPE)
        # h_counts have value range -1 - int16 max, we use int16 here, -1 is for invalid
        # we are not expecting h_counts to be that high anyway
        h_counts = np.zeros(h_count_node_feat_size, dtype=np.int16)
        for h_delta in formulae_dict:
            formula = formulae_dict[h_delta]
            h_count = h_count_dict[h_delta]
            formula_idx = formula_to_idx[formula]
            # h_delta [0,-1,1,-2,2,-3,3,-4,4]
            h_delta_idx = h_delta * 2 if h_delta >= 0 else (-h_delta * 2) - 1
            # sanity checks: indices/counts must be non-negative and fit uint16
            if h_delta_idx < 0 or h_delta_idx >= formula_node_feat_size:
                raise ValueError(
                    f"h_delta_idx out of bounds: {h_delta_idx}, size={formula_node_feat_size}"
                )
            if formula_idx < 0 or formula_idx > np.iinfo(FRAG_INDEX_DTYPE).max:
                raise ValueError(f"formula_idx out of {FRAG_INDEX_DTYPE} range: {formula_idx}")
            formulae_idxs[h_delta_idx] = formula_idx
            h_counts[h_delta_idx] = h_count
            formula_idx_by_h_delta[h_delta_idx].add(formula_idx)
        formula_idx_list.append(formulae_idxs)
        h_count_list.append(h_counts)

    # before stacking, ensure no negative or out-of-range values exist
    max_frag_index = np.iinfo(FRAG_INDEX_DTYPE).max
    for arr in formula_idx_list:
        if np.any(np.asarray(arr) < 0):
            raise ValueError("Negative value found in formula index list before stacking")
        if np.any(np.asarray(arr) > max_frag_index):
            raise ValueError(
                f"Value exceeds {FRAG_INDEX_DTYPE} max in formula index list before stacking"
            )

    # for arr in h_count_list:
    #   if np.any(np.asarray(arr) < 0):
    #       raise ValueError("Negative value found in h_count list before stacking")
    #   if np.any(np.asarray(arr) > max_uint16):
    #       raise ValueError("Value exceeds uint16 max in h_count list before stacking")

    node_formulae_matrix = np.stack(formula_idx_list, dtype=FRAG_INDEX_DTYPE)
    node_h_count_matrix = np.stack(h_count_list, dtype=np.int16)

    # map nodes to nb_isomorphism indices
    if nb_isomorphic and node_nb_hashes is not None:
        nb_iso_map = {hash: idx for idx, hash in enumerate(sorted(list(set(node_nb_hashes))))}
        node_nb_iso_idx = np.array(
            [nb_iso_map[hash] for hash in node_nb_hashes], dtype=FRAG_INDEX_DTYPE
        ).reshape(-1, 1)
        # validate nb iso indices
        if node_nb_iso_idx.size > 0:
            if np.any(node_nb_iso_idx < 0) or np.any(
                node_nb_iso_idx > np.iinfo(FRAG_INDEX_DTYPE).max
            ):
                raise ValueError(
                    f"node_nb_iso_idx contains invalid values outside {FRAG_INDEX_DTYPE} range"
                )
        dag_num_nodes_nb = len(set(node_nb_hashes))
    else:
        node_nb_iso_idx = np.zeros((nodes_mask_matrix.shape[0], 0), dtype=FRAG_INDEX_DTYPE)
        dag_num_nodes_nb = -1

    assert nodes_mask_matrix.shape[1] <= MASK_SIZE
    node_cc_mask = th.as_tensor(nodes_mask_matrix, dtype=th.bool)
    node_cc_mask = F.pad(node_cc_mask, (0, MASK_SIZE - node_cc_mask.shape[1]), "constant", 0)
    node_cc_long = th_mask_to_long(node_cc_mask).type(NODE_FEAT_DTYPE)
    mol_edge_index = th.as_tensor(mol_d["bonds"].T, dtype=th.long)
    if mol_edge_index.numel() > 0:
        mol_edge_index = th.cat([mol_edge_index, mol_edge_index.flip(0)], dim=1)
    boundary_pair_frag_idxs, boundary_pair_in_local, boundary_pair_out_local = (
        compute_boundary_pair_idxs(
            node_cc_mask,
            mol_edge_index,
            th.tensor([0, num_nodes], dtype=th.long),
            th.tensor([0, node_cc_mask.shape[0]], dtype=th.long),
        )
    )
    # the order is important!
    pyg_node_feats = th.cat(
        [
            th.as_tensor(nodes_depth_matrix, dtype=NODE_FEAT_DTYPE),
            node_cc_long,
            th.as_tensor(node_base_formula_matrix, dtype=NODE_FEAT_DTYPE),
            th.as_tensor(node_formulae_matrix, dtype=NODE_FEAT_DTYPE),
            th.as_tensor(node_h_count_matrix, dtype=NODE_FEAT_DTYPE),
            th.as_tensor(node_nb_iso_idx, dtype=NODE_FEAT_DTYPE),
            # cmf features bit, not set here, this should set in dataloader, use 0 for NULL forumla
            th.zeros((node_cc_mask.shape[0], cmf_formula_node_feat_size), dtype=NODE_FEAT_DTYPE),
        ],
        dim=1,
    )

    # peak mzs array
    formula_peak_mzs = []
    formula_peak_probs = []
    peaks_for_element_cache = {}
    for formula, idx in formula_to_idx.items():
        if formula == "":
            peak_mzs = np.zeros(MAX_NUM_MZS_PER_FORMULA, dtype=np.float32)
            peak_probs = np.zeros(MAX_NUM_MZS_PER_FORMULA, dtype=np.float32)
        else:
            peak_mzs, peak_probs = formula_to_peak_mzs(
                formula,
                "",
                isotopes=isotopes,
                return_probs=True,
                peaks_for_element_cache=peaks_for_element_cache,
            )
            peak_mzs, peak_probs = zip(
                *sorted(zip(peak_mzs, peak_probs), key=lambda x: x[1], reverse=True)
            )
            peak_mzs = np.array(peak_mzs[:MAX_NUM_MZS_PER_FORMULA], dtype=np.float32)
            peak_mzs = np.pad(
                peak_mzs,
                (0, MAX_NUM_MZS_PER_FORMULA - len(peak_mzs)),
                "constant",
                constant_values=0,
            )
            peak_probs = np.array(peak_probs[:MAX_NUM_MZS_PER_FORMULA], dtype=np.float32)
            peak_probs = np.pad(
                peak_probs,
                (0, MAX_NUM_MZS_PER_FORMULA - len(peak_probs)),
                "constant",
                constant_values=0,
            )
        formula_peak_mzs.append(peak_mzs)
        formula_peak_probs.append(peak_probs)

    # save as float32 for speed and lower ram usage
    # forumla and peak mz it can produce
    formula_peak_mzs = th.as_tensor(np.stack(formula_peak_mzs, axis=0), dtype=META_DATA_DTYPE)
    # forumla and peak intensity it can produce, respect to isotopes labels
    formula_peak_probs = th.as_tensor(np.stack(formula_peak_probs, axis=0), dtype=META_DATA_DTYPE)

    # add edges info
    edge_diff_cc_mask, edge_diff_formula_mask, edge_diff_h_range = [], [], []

    for edge in dag_edges_matrix:
        from_idx, to_idx = edge
        from_cc_mask = nodes_mask_matrix[from_idx]
        to_cc_mask = nodes_mask_matrix[to_idx]
        # Use signed subtraction to avoid uint8 underflow (0 - 1 = 255 in uint8)
        diff_cc_mask = np.clip(
            from_cc_mask.astype(np.int16) - to_cc_mask.astype(np.int16), 0, 1
        ).astype(MASK_DTYPE)
        # compute_cc_h_floor/cap expect atom indices, not a bitmask
        diff_atom_ids = cc_bit_mask_to_atom_idx(diff_cc_mask)

        from_formula_mask = node_base_formula_matrix[from_idx]
        to_formula_mask = node_base_formula_matrix[to_idx]
        # Use signed subtraction to avoid uint16 underflow (parent always >= child for heavy atoms,
        # but H is zeroed below, so sign safety is needed)
        diff_formula_mask = from_formula_mask.astype(np.int32) - to_formula_mask.astype(np.int32)
        diff_formula_mask[CANONICAL_H_IDX] = 0  # we don't care Hs for this

        diff_h_floor = compute_cc_h_floor(
            diff_atom_ids,
            mol_d["ve_arr"],
            mol_d["sbond_arr"],
            0,
            mol_d["bonds"],
            mol_d["atoms_to_bonds"],
            mol_d["bond_mask_arr"],
        )
        diff_h_cap = compute_cc_h_cap(diff_atom_ids, mol_d["ve_arr"], mol_d["sbond_arr"], 0)
        assert diff_h_floor <= diff_h_cap, (diff_h_floor, diff_h_cap)
        assert diff_h_floor >= 0, diff_h_floor
        diff_h_range = [diff_h_floor, diff_h_cap]

        edge_diff_cc_mask.append(diff_cc_mask)
        edge_diff_formula_mask.append(diff_formula_mask)
        edge_diff_h_range.append(diff_h_range)

    assert len(edge_diff_cc_mask) == len(edge_diff_formula_mask)
    assert len(edge_diff_cc_mask) > 0 or pyg_node_feats.shape[0] == 1, (
        f"DAG has no edges, with {pyg_node_feats.shape[0]} nodes"
    )

    # Handle case when there are no edges
    if len(edge_diff_cc_mask) == 0:
        # Create empty edge feature tensor with correct shape
        cc_edge_feat_size = MASK_SIZE // 64
        base_formula_edge_feat_size = len(ELEMENT_TO_IDX)
        h_range_edge_feat_size = 2
        total_edge_feat_size = (
            cc_edge_feat_size + base_formula_edge_feat_size + h_range_edge_feat_size
        )
        pyg_edge_feats = th.zeros((0, total_edge_feat_size), dtype=EDGE_FEAT_DTYPE)
    else:
        edge_diff_cc_mask = th.as_tensor(np.stack(edge_diff_cc_mask, axis=0), dtype=th.bool)
        edge_diff_cc_mask = F.pad(
            edge_diff_cc_mask, (0, MASK_SIZE - edge_diff_cc_mask.shape[1]), "constant", 0
        )
        edge_diff_cc_long = th_mask_to_long(edge_diff_cc_mask).type(EDGE_FEAT_DTYPE)
        edge_diff_formula_mask = th.as_tensor(
            np.stack(edge_diff_formula_mask, axis=0), dtype=EDGE_FEAT_DTYPE
        )
        edge_diff_h_range = th.as_tensor(np.stack(edge_diff_h_range, axis=0), dtype=EDGE_FEAT_DTYPE)
        pyg_edge_feats = th.cat(
            [edge_diff_cc_long, edge_diff_formula_mask, edge_diff_h_range], dim=1
        )

    # edge index need to be int64 or it will throw error where compute degree
    pyg_edge_index = th.tensor(dag_edges_matrix.T, dtype=th.int64)

    pyg_cc_g = pyg.data.Data(pyg_node_feats, pyg_edge_index, pyg_edge_feats)

    pyg_cc_g.node_feat_idxs = th.cumsum(
        th.tensor([0] + node_feat_shapes, dtype=th.long), 0
    ).reshape(1, -1)
    pyg_cc_g.edge_feat_idxs = th.cumsum(
        th.tensor([0] + edge_feat_shapes, dtype=th.long), 0
    ).reshape(1, -1)
    pyg_cc_g.boundary_pair_frag_idxs = boundary_pair_frag_idxs
    pyg_cc_g.boundary_pair_in_local = boundary_pair_in_local
    pyg_cc_g.boundary_pair_out_local = boundary_pair_out_local

    # convert to pyg
    frag_d = {}
    frag_d["max_depth"] = max_depth
    frag_d["reached_depth"] = reached_depth
    frag_d["h_prior"] = h_prior
    frag_d["max_h_transfer"] = max_h_transfer  # max number of Hs transfers
    frag_d["formula_peak_mzs"] = formula_peak_mzs  # forumla and peak mz it can produce
    frag_d["formula_peak_probs"] = (
        formula_peak_probs  # forumla and peak int it can produce, respect to isotopes labels
    )
    frag_d["idx_to_formula"] = idx_to_formula  # idx to formula str, useful for annotation
    frag_d["idx_by_h_delta"] = (
        formula_idx_by_h_delta  # formula idx for each h_delta value, used in compute_frag_peak_stats
    )
    frag_d["dag"] = pyg_cc_g

    frag_d["edges_min_depth"] = edges_min_depth
    frag_d["nodes_min_depth"] = nodes_min_depth

    # add stats here
    # we need change data type again
    # we just need to change this one place
    frag_d["dag_num_edges"] = pyg_cc_g.num_edges
    frag_d["dag_num_nodes"] = pyg_cc_g.num_nodes
    frag_d["dag_num_nodes_nb"] = dag_num_nodes_nb
    frag_d["dag_sparsity"] = (
        2 * pyg_cc_g.num_edges / (pyg_cc_g.num_nodes * (pyg_cc_g.num_nodes - 1))
        if pyg_cc_g.num_nodes > 1
        else 0.0
    )
    frag_d["formula_redundancy"] = sum([v for k, v in formula_counts.items() if k != ""]) / len(
        [k for k in formula_counts if k != ""]
    )
    frag_d["node_feature_size"] = pyg_cc_g.num_features
    frag_d["edge_feature_size"] = pyg_cc_g.num_edge_features
    frag_d["is_directed"] = pyg_cc_g.is_directed() if pyg_cc_g.num_edges > 0 else False
    frag_d["dag_num_edges_by_depth"] = {
        k: np.count_nonzero(edges_min_depth == k) for k in range(reached_depth + 1)
    }
    frag_d["dag_num_nodes_by_depth"] = {
        k: np.count_nonzero(nodes_min_depth == k) for k in range(reached_depth + 1)
    }
    frag_d["force_stopped"] = force_stop

    return frag_d


def get_node_feats(node_feats: th.Tensor, node_feat_idxs: th.Tensor, key: str):
    """get node features by key used for pyg

    Args:
        node_feats (_type_): _description_
        node_feat_idxs (_type_): _description_
        key (_type_): _description_

    Returns:
        _type_: _description_
    """

    node_feat_idx = NODE_FEAT_TO_IDX[key]
    node_feats = node_feats[:, node_feat_idxs[node_feat_idx] : node_feat_idxs[node_feat_idx + 1]]
    # print(f"get_node_feats, node_feats tensor shape: {node_feats.shape}, num nodes: {len(node_feats)}, feature name: {key}" )
    return node_feats


def get_edge_feats(edge_feats: th.Tensor, edge_feat_idxs: int, key: str):
    """get edege feats used for pyg

    Args:
        edge_feats (_type_): _description_
        edge_feat_idxs (_type_): _description_
        key (_type_): _description_

    Returns:
        _type_: _description_
    """

    edge_feat_idx = EDGE_FEAT_TO_IDX[key]
    edge_feats = edge_feats[:, edge_feat_idxs[edge_feat_idx] : edge_feat_idxs[edge_feat_idx + 1]]
    return edge_feats


def get_frag_name(mol_id: str | int, is_compressed: bool) -> str:
    """Return the canonical filename for a fragment DAG pickle file.

    Args:
        mol_id: Molecule identifier (int or numeric string).
        is_compressed: Whether the file is bz2-compressed.

    Returns:
        Filename of the form ``{mol_id}.pkl[.bz2]``.
    """
    name = f"{int(mol_id)}.pkl"
    if is_compressed:
        name += ".bz2"
    return name


def get_frag_fp(mol_id: str | int, frag_dp: str, is_compressed: bool) -> str:
    """Return the full path for a fragment DAG pickle file.

    Args:
        mol_id: Molecule identifier (int or numeric string).
        frag_dp: Directory containing fragment files.
        is_compressed: Whether the file is bz2-compressed.

    Returns:
        Full file path using the canonical naming convention.
    """
    return os.path.join(frag_dp, get_frag_name(mol_id, is_compressed))


def _dump_pickle_file(fp: str, payload: dict) -> None:
    """Write a pickled payload using compression inferred from file suffix."""
    if fp.endswith((".pkl", ".pickle")):
        with open(fp, "wb") as pf:
            pickle.dump(payload, pf, protocol=pickle.HIGHEST_PROTOCOL)
        return

    if fp.endswith((".pkl.gz", ".pickle.gz")):
        with gzip.open(fp, "wb") as pf:
            pickle.dump(payload, pf, protocol=pickle.HIGHEST_PROTOCOL)
        return

    if fp.endswith((".pkl.bz2", ".pickle.bz2")):
        with bz2.open(fp, "wb") as pf:
            pickle.dump(payload, pf, protocol=pickle.HIGHEST_PROTOCOL)
        return

    raise ValueError(
        f"Unsupported pickle output extension for '{fp}'. "
        "Expected .pkl/.pickle with optional .gz or .bz2"
    )


def _load_pickle_file(fp: str):
    """Read a pickled payload using compression inferred from file suffix."""
    if fp.endswith((".pkl", ".pickle")):
        with open(fp, "rb") as pf:
            return pickle.load(pf)

    if fp.endswith((".pkl.gz", ".pickle.gz")):
        with gzip.open(fp, "rb") as pf:
            return pickle.load(pf)

    if fp.endswith((".pkl.bz2", ".pickle.bz2")):
        with bz2.open(fp, "rb") as pf:
            return pickle.load(pf)

    raise ValueError(
        f"Unsupported pickle input extension for '{fp}'. "
        "Expected .pkl/.pickle with optional .gz or .bz2"
    )


def get_dag_output_path(
    dag_dp: str,
    mol_id: str | int,
    compress_dags: bool,
    compress_format: str,
) -> str:
    """Build output path for one DAG artifact.

    The base filename is determined by :func:`get_frag_name` (canonical format
    ``{mol_id}.pkl``).  Compression with ``"gz"`` appends ``.gz``; ``"bz2"``
    appends ``.bz2`` (equivalent to passing ``is_compressed=True`` to
    :func:`get_frag_name`).

    Args:
        dag_dp: DAG output directory.
        mol_id: Molecule identifier.
        compress_dags: Whether per-file DAG outputs should be compressed.
        compress_format: Compression format, either ``"gz"`` or ``"bz2"``.

    Returns:
        Full output path for the DAG file.

    Raises:
        ValueError: If ``compress_format`` is invalid when compression is enabled.
    """
    if not compress_dags:
        return get_frag_fp(mol_id, dag_dp, is_compressed=False)

    if compress_format == "gz":
        return os.path.join(dag_dp, f"{int(mol_id)}.pkl.gz")
    if compress_format == "bz2":
        return get_frag_fp(mol_id, dag_dp, is_compressed=True)

    raise ValueError(
        f"Invalid compression format '{compress_format}'. Expected one of: 'gz', 'bz2'."
    )


def dump_dag_pickle(fp: str, dag_d: dict) -> None:
    """Write a DAG dictionary to disk with extension-aware compression.

    Args:
        fp: Output path ending with ``.pkl``, ``.pkl.gz``, or ``.pkl.bz2``.
        dag_d: DAG payload dictionary.

    Raises:
        ValueError: If file suffix is unsupported.
    """
    _dump_pickle_file(fp, dag_d)


def dump_dag_hdf5(h5_group, dag_d: dict) -> None:
    """Write one DAG dictionary into an open h5py group.

    All PyTorch tensors are converted to numpy arrays before writing.
    Dict-typed fields (``idx_to_formula``, ``idx_by_h_delta``, depth-by-depth
    counts) are JSON-encoded and stored as group attributes.

    Args:
        h5_group: An open, writable ``h5py.Group`` for this molecule.
        dag_d: DAG payload dictionary as returned by ``compute_dags``.
    """

    dag = dag_d["dag"]
    g = h5_group.create_group("dag")
    g.create_dataset("x", data=dag.x.numpy(), compression="gzip", compression_opts=4)
    g.create_dataset(
        "edge_index", data=dag.edge_index.numpy(), compression="gzip", compression_opts=4
    )
    g.create_dataset(
        "edge_attr", data=dag.edge_attr.numpy(), compression="gzip", compression_opts=4
    )
    g.create_dataset("node_feat_idxs", data=dag.node_feat_idxs.numpy())
    g.create_dataset("edge_feat_idxs", data=dag.edge_feat_idxs.numpy())
    for name in ("boundary_pair_frag_idxs", "boundary_pair_in_local", "boundary_pair_out_local"):
        if hasattr(dag, name):
            g.create_dataset(
                name, data=getattr(dag, name).numpy(), compression="gzip", compression_opts=4
            )

    h5_group.create_dataset(
        "formula_peak_mzs",
        data=dag_d["formula_peak_mzs"].numpy(),
        compression="gzip",
        compression_opts=4,
    )
    h5_group.create_dataset(
        "formula_peak_probs",
        data=dag_d["formula_peak_probs"].numpy(),
        compression="gzip",
        compression_opts=4,
    )

    edges_min_depth = dag_d["edges_min_depth"]
    nodes_min_depth = dag_d["nodes_min_depth"]
    h5_group.create_dataset("edges_min_depth", data=edges_min_depth)
    h5_group.create_dataset("nodes_min_depth", data=nodes_min_depth)

    # JSON-encoded variable-structure fields
    h5_group.attrs["idx_to_formula"] = json.dumps(
        {str(k): v for k, v in dag_d["idx_to_formula"].items()}
    )
    h5_group.attrs["idx_by_h_delta"] = json.dumps(
        [sorted(map(int, s)) for s in dag_d["idx_by_h_delta"]]
    )
    h5_group.attrs["dag_num_edges_by_depth"] = json.dumps(
        {str(k): int(v) for k, v in dag_d["dag_num_edges_by_depth"].items()}
    )
    h5_group.attrs["dag_num_nodes_by_depth"] = json.dumps(
        {str(k): int(v) for k, v in dag_d["dag_num_nodes_by_depth"].items()}
    )

    # Integer scalars
    for key in [
        "max_depth",
        "reached_depth",
        "max_h_transfer",
        "dag_num_edges",
        "dag_num_nodes",
        "dag_num_nodes_nb",
        "node_feature_size",
        "edge_feature_size",
    ]:
        h5_group.attrs[key] = int(dag_d[key])

    # Float scalars
    for key in ["dag_sparsity", "formula_redundancy"]:
        h5_group.attrs[key] = float(dag_d[key])

    # Bool scalars
    for key in ["h_prior", "is_directed", "force_stopped"]:
        h5_group.attrs[key] = bool(dag_d[key])


def load_dag_hdf5(h5_group) -> dict:
    """Read one DAG dictionary from an open h5py group.

    Numpy arrays are converted back to PyTorch tensors with their original
    dtypes.  JSON-encoded attributes are decoded to their original Python types.

    Args:
        h5_group: An open ``h5py.Group`` previously written by
            :func:`dump_dag_hdf5`.

    Returns:
        DAG payload dictionary compatible with the format returned by
        ``compute_dags``.
    """

    g = h5_group["dag"]
    dag = pyg.data.Data(
        x=th.from_numpy(g["x"][:]),
        edge_index=th.from_numpy(g["edge_index"][:]),
        edge_attr=th.from_numpy(g["edge_attr"][:]),
    )
    dag.node_feat_idxs = th.from_numpy(g["node_feat_idxs"][:])
    dag.edge_feat_idxs = th.from_numpy(g["edge_feat_idxs"][:])
    for name in ("boundary_pair_frag_idxs", "boundary_pair_in_local", "boundary_pair_out_local"):
        if name in g:
            setattr(dag, name, th.from_numpy(g[name][:]))

    dag_d: dict = {
        "dag": dag,
        "formula_peak_mzs": th.from_numpy(h5_group["formula_peak_mzs"][:]),
        "formula_peak_probs": th.from_numpy(h5_group["formula_peak_probs"][:]),
        "edges_min_depth": h5_group["edges_min_depth"][:],
        "nodes_min_depth": h5_group["nodes_min_depth"][:],
    }

    # JSON-encoded variable-structure fields
    dag_d["idx_to_formula"] = {
        int(k): v for k, v in json.loads(h5_group.attrs["idx_to_formula"]).items()
    }
    # Stored as list-of-lists; restore as list-of-sets to match original type
    dag_d["idx_by_h_delta"] = [set(s) for s in json.loads(h5_group.attrs["idx_by_h_delta"])]
    dag_d["dag_num_edges_by_depth"] = {
        int(k): v for k, v in json.loads(h5_group.attrs["dag_num_edges_by_depth"]).items()
    }
    dag_d["dag_num_nodes_by_depth"] = {
        int(k): v for k, v in json.loads(h5_group.attrs["dag_num_nodes_by_depth"]).items()
    }

    # Integer scalars
    for key in [
        "max_depth",
        "reached_depth",
        "max_h_transfer",
        "dag_num_edges",
        "dag_num_nodes",
        "dag_num_nodes_nb",
        "node_feature_size",
        "edge_feature_size",
    ]:
        dag_d[key] = int(h5_group.attrs[key])

    # Float scalars
    for key in ["dag_sparsity", "formula_redundancy"]:
        dag_d[key] = float(h5_group.attrs[key])

    # Bool scalars
    for key in ["h_prior", "is_directed", "force_stopped"]:
        dag_d[key] = bool(h5_group.attrs[key])

    return dag_d


def save_frag_d(frag_d: dict, mol_id: str, frag_dp: str, is_compressed: bool = False):
    """save frag_d use pickle if is_compressed save as .pbz
    Args:
        frag_d (dict): frag_d to save
        mol_id (str): mol_id to save as
        frag_dp (str): directory to save to
        is_compressed (bool, optional): whether to compress the file. Defaults to False.
    """

    fp = get_frag_fp(mol_id, frag_dp, is_compressed)
    _dump_pickle_file(fp, frag_d)


def _legacy_frag_names(mol_id: str | int, is_compressed: bool) -> list[str]:
    """Return legacy filenames to try for backward compatibility.

    Covers two previous naming conventions:

    - ``{mol_id}.pickle[.bz2]`` — original format before unification
    - ``{mol_id:08d}.pkl[.bz2]`` — short-lived zero-padded format

    Args:
        mol_id: Molecule identifier.
        is_compressed: Whether the file is bz2-compressed.

    Returns:
        List of legacy filenames to probe, in preference order.
    """
    suffix = ".bz2" if is_compressed else ""
    return [
        f"{int(mol_id):08d}.pkl{suffix}",
        f"{mol_id}.pickle{suffix}",
    ]


def load_frag_d(mol_id: str | int, frag_dp: str, is_compressed: bool = False):
    """Load a fragment DAG dictionary from disk.

    Supports folder, tar archive, and zip archive backends.  The canonical
    filename format is ``{mol_id}.pkl[.bz2]`` (see :func:`get_frag_name`).
    For backward compatibility, the zero-padded ``{mol_id:08d}.pkl[.bz2]`` and
    the original ``{mol_id}.pickle[.bz2]`` formats are also tried when the
    canonical file is not found.

    Args:
        mol_id: Molecule identifier (int or numeric string).
        frag_dp: Directory path or archive file (``.tar`` / ``.zip``) containing
            fragment DAG files.
        is_compressed: Whether the files are bz2-compressed. Defaults to False.

    Returns:
        Loaded fragment dictionary, or ``None`` if not found inside a tar archive.
    """
    if os.path.isfile(frag_dp) and str(frag_dp).endswith(".tar"):
        canonical = get_frag_name(mol_id, is_compressed)
        accepted = {canonical} | set(_legacy_frag_names(mol_id, is_compressed))
        with tarfile.open(frag_dp, "r") as tar_read:
            for member in tar_read.getmembers():
                if member.name in accepted:
                    f = tar_read.extractfile(member)
                    if f is None:
                        raise ValueError(
                            f"Could not extract '{member.name}' from archive '{frag_dp}'"
                        )
                    content = f.read()
                    if member.name.endswith(".bz2"):
                        return pickle.loads(bz2.decompress(content))
                    return pickle.loads(content)
        return None

    if os.path.isfile(frag_dp) and str(frag_dp).endswith(".zip"):
        canonical = get_frag_name(mol_id, is_compressed)
        with ZipFile(frag_dp, "r") as zip_read:
            names = zip_read.namelist()
            member_name = canonical
            if member_name not in names:
                for legacy in _legacy_frag_names(mol_id, is_compressed):
                    if legacy in names:
                        member_name = legacy
                        break
            with zip_read.open(member_name) as f:
                content = f.read()
        if member_name.endswith(".bz2"):
            return pickle.loads(bz2.decompress(content))
        return pickle.loads(content)

    fp = get_frag_fp(mol_id, frag_dp, is_compressed)
    if not os.path.isfile(fp):
        for legacy in _legacy_frag_names(mol_id, is_compressed):
            legacy_fp = os.path.join(frag_dp, legacy)
            if os.path.isfile(legacy_fp):
                fp = legacy_fp
                break
    return _load_pickle_file(fp)


def _hash_label(label, digest_size=32):
    """
    Adapted from https://networkx.org/documentation/stable/_modules/networkx/algorithms/graph_hashing.html
    """
    return blake2b(label.encode("ascii"), digest_size=digest_size).hexdigest()


def wl_hash(
    elems: list,
    bond_type_idxs: list,
    node_mask: np.ndarray,
    edges: np.ndarray,
    node_to_edge_idx: np.ndarray,
    include_bond_type: bool = False,
    max_iterations: int = -1,
) -> int:
    """
    Adapted from https://networkx.org/documentation/stable/_modules/networkx/algorithms/graph_hashing.html
    """

    cur_hashes = []
    num_nodes = len(elems)
    for i in range(num_nodes):
        if node_mask[i]:
            cur_hashes.append(str(elems[i]))
        else:
            cur_hashes.append("")
    cur_counter = Counter(cur_hashes)
    cur_counter.pop("", None)
    graph_hash_counts = sorted(cur_counter.items(), key=lambda x: x[0])
    iterations = np.sum(node_mask)
    assert iterations <= num_nodes, (iterations, num_nodes)
    if max_iterations == -1:
        max_iterations = iterations
    else:
        assert max_iterations >= 0, max_iterations
    ct = 0
    while ct < iterations and ct < max_iterations:
        # print(cur_hashes)
        new_hashes = []
        temp_atoms = 0
        # Step 2: Update hashes with local neighborhoods
        for node_idx in range(num_nodes):
            cur_hash = cur_hashes[node_idx]
            if not node_mask[node_idx]:
                new_hashes.append(cur_hash)
                continue
            # Count num atoms in this loop
            temp_atoms += 1
            # Get local neighbors
            neighbor_labels = []
            for edge_idx in node_to_edge_idx[node_idx]:
                if edge_idx == -1:
                    break
                node_idx_1, node_idx_2 = edges[edge_idx]
                if node_idx_1 == node_idx:
                    targ_node_idx = node_idx_2
                else:
                    targ_node_idx = node_idx_1
                assert targ_node_idx != node_idx
                if not node_mask[targ_node_idx]:
                    continue
                targ_hash = cur_hashes[targ_node_idx]
                if include_bond_type:
                    bondtype = bond_type_idxs[edge_idx]
                    neighbor_label = f"_{bondtype}_{targ_hash}"
                else:
                    neighbor_label = f"_{targ_hash}"
                neighbor_labels.append(neighbor_label)
            new_hash = cur_hash + "".join(sorted(neighbor_labels))
            new_hash = _hash_label(new_hash)
            new_hashes.append(new_hash)
        assert temp_atoms == iterations, (temp_atoms, iterations)
        new_counter = Counter(new_hashes)
        new_counter.pop("", None)
        graph_hash_counts.extend(sorted(new_counter.items(), key=lambda x: x[0]))
        cur_hashes = new_hashes
        # print(f"> {ct}")
        # print(new_graph_hash)
        # print(cur_hashes)
        ct += 1
    graph_hash = _hash_label(str(tuple(graph_hash_counts)))
    return graph_hash


def get_subgraph_hashes(
    nodes_mask_matrix: np.ndarray,
    elems: list,
    bond_type_idxs: list,
    edges: np.ndarray,
    node_to_edge_idx: np.ndarray,
    include_bond_type: bool,
    max_iterations: int,
):
    subgraph_hashes = []
    num_subgraphs = nodes_mask_matrix.shape[0]
    for i in range(num_subgraphs):
        subgraph_mask = nodes_mask_matrix[i]
        subgraph_hash = wl_hash(
            elems,
            bond_type_idxs,
            subgraph_mask,
            edges,
            node_to_edge_idx,
            include_bond_type=include_bond_type,
            max_iterations=max_iterations,
        )
        subgraph_hashes.append(subgraph_hash)
    return subgraph_hashes


def timed_get_dags(
    mol,
    mol_id,
    max_depth,
    h_prior,
    max_h_transfer,
    max_time,
    isotopes: bool,
    nb_isomorphic: bool,
    wl_max_iterations: int,
    multi_cut_bfs: bool = False,
    max_cut_size: int = 2,
    smarts_prepass: bool = False,
    min_frag_atoms: int = 0,
) -> tuple:
    try:
        mol_d = extract_mol_info(mol)
        # this maybe dangers in multi processing because of scopes
        dag_d = compute_dags(
            mol_d,
            max_depth,
            h_prior,
            max_h_transfer,
            max_time,
            isotopes,
            nb_isomorphic,
            # False, b_isomorphic, this not used
            wl_max_iterations,
            multi_cut_bfs=multi_cut_bfs,
            max_cut_size=max_cut_size,
            smarts_prepass=smarts_prepass,
            min_frag_atoms=min_frag_atoms,
        )
        return mol_id, dag_d
    except KeyboardInterrupt as e:
        # let these through
        raise e
    except Exception as e:
        # don't retry, theres a bug
        if type(mol) is not str:
            mol = Chem.MolToSmiles(mol)
        print(f">> Non-timeout error, aborting: {type(e)} {repr(e)} Input {mol}", file=sys.stderr)
        print("> Traceback", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    return mol_id, {}


def run_frag_gen_hdf5(
    mol_df: "pd.DataFrame",
    h5_file: "_h5py.File",
    max_depth: int,
    max_h_transfer: int,
    max_time: int,
    isotopes: bool,
    nb_isomorphic: bool,
    wl_max_iterations: int,
    multi_cut_bfs: bool,
    max_cut_size: int,
    smarts_prepass: bool,
    min_frag_atoms: int,
    disable_tqdm: bool = False,
) -> dict[str, list[int]]:
    """Run fragment generation for all molecules and write DAGs to an open HDF5 file.

    Iterates over rows in ``mol_df``, calls :func:`timed_get_dags` in parallel via
    :func:`~fragnnet.utils.data_utils.par_apply`, and writes each successfully
    fragmented molecule's DAG into ``h5_file`` under the key ``str(mol_id)``.

    Args:
        mol_df: DataFrame with at least ``mol`` and ``mol_id`` columns.
        h5_file: Open writable ``h5py.File`` object. Groups are created directly
            inside the root; the caller is responsible for opening and closing the file.
        max_depth: Maximum fragmentation depth for DAG generation.
        max_h_transfer: Maximum hydrogen transfers per fragment.
        max_time: Per-molecule time budget in seconds.
        isotopes: Whether to compute isotopic peak distributions.
        nb_isomorphic: Whether to compute neighbourhood-isomorphism features.
        wl_max_iterations: Maximum Weisfeiler-Lehman iterations.
        multi_cut_bfs: Whether to use ring-aware multi-bond-cut BFS.
        max_cut_size: Maximum bonds cut per BFS step (only when ``multi_cut_bfs``).
        smarts_prepass: Whether to run the SMARTS rearrangement prepass.
        min_frag_atoms: Minimum heavy atoms per fragment.
        disable_tqdm: Suppress the tqdm progress bar.

    Returns:
        Dictionary mapping ``str(mol_id)`` to ``[num_nodes, num_edges]`` for every
        molecule that was successfully fragmented.
    """
    _log = logging.getLogger(__name__)

    mol_input_rows = (
        [
            row["mol"],
            row["mol_id"],
            max_depth,
            True,  # h_prior
            max_h_transfer,
            max_time,
            isotopes,
            nb_isomorphic,
            wl_max_iterations,
            multi_cut_bfs,
            max_cut_size,
            smarts_prepass,
            min_frag_atoms,
        ]
        for _, row in mol_df.iterrows()
    )

    from typing import cast

    frag_results_gen = par_apply(
        iter(mol_input_rows), timed_get_dags, True, return_as_generator=True
    )
    if frag_results_gen is None:
        raise RuntimeError("Parallel fragment generation returned no results")
    frag_results_iter = cast("list[tuple[str, dict]]", frag_results_gen)

    meta_info: dict[str, list[int]] = {}
    n_errors = 0

    for fr in progress_wrapper(
        frag_results_iter,
        total=mol_df.shape[0],
        desc="Compute Frags",
        disable_tqdm=disable_tqdm,
    ):
        mol_id, dag_d = fr
        if len(dag_d) == 0:
            _log.warning("Empty DAG for mol_id=%s", mol_id)
            n_errors += 1
            continue
        meta_info[str(int(mol_id))] = [dag_d["dag"].num_nodes, dag_d["dag"].num_edges]
        grp = h5_file.create_group(f"{int(mol_id)}")
        dump_dag_hdf5(grp, dag_d)

    _log.info("Fragment generation done: %d stored, %d errors", len(meta_info), n_errors)
    return meta_info


def compute_boundary_mask(
    frag_node_mask: th.Tensor,
    mol_edge_index: th.Tensor,
    mol_num_nodes: th.Tensor,
    frag_num_nodes: th.Tensor,
) -> th.Tensor:
    """Compute boundary atom mask for each fragment on-the-fly.

    A boundary atom is one that belongs to the fragment AND is adjacent to at
    least one atom that does not (i.e. it sits at a cut bond).  The root
    fragment (full precursor) always produces an all-zero row.

    Args:
        frag_node_mask: Bool tensor of shape ``(F, mask_size)`` with LOCAL atom
            indices per molecule (as returned by ``th_long_to_mask`` on the cc
            node feature).
        mol_edge_index: Long tensor of shape ``(2, E)`` with GLOBAL atom indices
            across the batch (standard PyG format).
        mol_num_nodes: Long tensor of shape ``(B+1,)`` — cumulative atom counts
            per molecule in the batch (prefix sum, starts at 0).
        frag_num_nodes: Long tensor of shape ``(B+1,)`` — cumulative fragment
            counts per molecule in the batch (prefix sum, starts at 0).

    Returns:
        Bool tensor of shape ``(F, mask_size)`` where ``True`` marks boundary
        atoms for each fragment.
    """
    num_frags_total = frag_node_mask.shape[0]
    mask_size = frag_node_mask.shape[1]
    device = frag_node_mask.device

    u_global = mol_edge_index[0]
    v_global = mol_edge_index[1]
    edge_mol_idx = th.bucketize(u_global, mol_num_nodes, right=True) - 1

    boundary_mask = th.zeros(num_frags_total, mask_size, dtype=th.bool, device=device)
    num_mols = int(mol_num_nodes.shape[0]) - 1
    for m in range(num_mols):
        f_start = int(frag_num_nodes[m])
        f_end = int(frag_num_nodes[m + 1])
        if f_start == f_end:
            continue
        edge_sel = edge_mol_idx == m
        if not edge_sel.any():
            continue
        atom_off = int(mol_num_nodes[m])
        u_loc = u_global[edge_sel] - atom_off
        v_loc = v_global[edge_sel] - atom_off
        valid_e = (u_loc >= 0) & (u_loc < mask_size) & (v_loc >= 0) & (v_loc < mask_size)
        u_loc, v_loc = u_loc[valid_e], v_loc[valid_e]

        frags_m = frag_node_mask[f_start:f_end]  # (F_m, mask_size)
        f_m = f_end - f_start
        u_in = frags_m[:, u_loc].float()
        v_in = frags_m[:, v_loc].float()
        bm = th.zeros(f_m, mask_size, device=device)
        bm.scatter_add_(1, u_loc.unsqueeze(0).expand(f_m, -1), u_in * (1 - v_in))
        bm.scatter_add_(1, v_loc.unsqueeze(0).expand(f_m, -1), v_in * (1 - u_in))
        boundary_mask[f_start:f_end] = bm > 0

    return boundary_mask
    return meta_info


def compute_boundary_pair_idxs(
    frag_node_mask: th.Tensor,
    mol_edge_index: th.Tensor,
    mol_num_nodes: th.Tensor,
    frag_num_nodes: th.Tensor,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Compute boundary pair indices for each fragment on-the-fly.

    For each cut bond ``(u_in, v_out)`` where ``u`` is inside the fragment and
    ``v`` is outside (or vice versa), returns the global fragment index and the
    global atom indices of both endpoints.  The root fragment (full precursor)
    always produces zero pairs.

    Args:
        frag_node_mask: Bool tensor of shape ``(F, mask_size)`` with LOCAL atom
            indices per molecule (as returned by ``th_long_to_mask`` on the cc
            node feature).
        mol_edge_index: Long tensor of shape ``(2, E)`` with GLOBAL atom indices
            across the batch (standard PyG format).
        mol_num_nodes: Long tensor of shape ``(B+1,)`` — cumulative atom counts
            per molecule in the batch (prefix sum, starts at 0).
        frag_num_nodes: Long tensor of shape ``(B+1,)`` — cumulative fragment
            counts per molecule in the batch (prefix sum, starts at 0).

    Returns:
        Tuple of three long tensors of shape ``(N_pairs,)``:

        - ``pair_frag_idxs``: global fragment index for each pair.
        - ``pair_in_global``: global atom index of the inside atom.
        - ``pair_out_global``: global atom index of the outside atom.

        All three tensors are empty (numel == 0) when no cut bonds exist.
    """
    device = frag_node_mask.device
    mask_size = frag_node_mask.shape[1]

    u_global = mol_edge_index[0]
    v_global = mol_edge_index[1]
    edge_mol_idx = th.bucketize(u_global, mol_num_nodes, right=True) - 1

    num_mols = int(mol_num_nodes.shape[0]) - 1

    all_pair_frag: list[th.Tensor] = []
    all_pair_in: list[th.Tensor] = []
    all_pair_out: list[th.Tensor] = []

    for m in range(num_mols):
        f_start = int(frag_num_nodes[m])
        f_end = int(frag_num_nodes[m + 1])
        if f_start == f_end:
            continue
        edge_sel = edge_mol_idx == m
        if not edge_sel.any():
            continue
        atom_off = int(mol_num_nodes[m])
        u_loc = u_global[edge_sel] - atom_off
        v_loc = v_global[edge_sel] - atom_off
        valid_e = (u_loc >= 0) & (u_loc < mask_size) & (v_loc >= 0) & (v_loc < mask_size)
        u_loc, v_loc = u_loc[valid_e], v_loc[valid_e]
        u_glob_m = u_global[edge_sel][valid_e]
        v_glob_m = v_global[edge_sel][valid_e]

        frags_m = frag_node_mask[f_start:f_end]  # (F_m, mask_size)

        u_in = frags_m[:, u_loc]  # (F_m, E_m) bool
        v_in = frags_m[:, v_loc]  # (F_m, E_m) bool

        # cut bonds: u inside, v outside
        cut_uv = u_in & ~v_in  # (F_m, E_m)
        if cut_uv.any():
            idxs = th.nonzero(cut_uv)  # (N, 2): [frag_local, edge_local]
            all_pair_frag.append(idxs[:, 0] + f_start)
            all_pair_in.append(u_glob_m[idxs[:, 1]])
            all_pair_out.append(v_glob_m[idxs[:, 1]])

        # cut bonds: v inside, u outside
        cut_vu = v_in & ~u_in  # (F_m, E_m)
        if cut_vu.any():
            idxs = th.nonzero(cut_vu)  # (N, 2): [frag_local, edge_local]
            all_pair_frag.append(idxs[:, 0] + f_start)
            all_pair_in.append(v_glob_m[idxs[:, 1]])
            all_pair_out.append(u_glob_m[idxs[:, 1]])

    if not all_pair_frag:
        empty = th.zeros(0, dtype=th.long, device=device)
        return empty, empty, empty

    return th.cat(all_pair_frag), th.cat(all_pair_in), th.cat(all_pair_out)
