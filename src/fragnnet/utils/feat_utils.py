"""mol_graph.py (adapted from SCARF)

Classes to featurize molecules into a graph with onehot concat feats on atoms
and bonds. Inspired by the dgllife library.

"""

from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np
import scipy
import torch as th
import torch_geometric as pyg
from rdkit import Chem
from torch_geometric.data import Data

from fragnnet.utils.data_utils import (
    make_maccs_fingerprint,
    make_morgan_fingerprint,
    make_rdkit_fingerprint,
)
from fragnnet.utils.frag_utils import (
    CANONICAL_ELEMENT_ORDER,
    EDGE_FEAT_TO_IDX,
    NODE_FEAT_TO_IDX,
    get_edge_feats,
    get_node_feats,
)

# TODO: remove this registry

atom_feat_registry: dict[str, dict[str, Any]] = {}
bond_feat_registry: dict[str, dict[str, Any]] = {}


def register_bond_feat(cls: type) -> type:
    """Register a bond featurizer class in the global registry."""

    bond_feat_registry[cls.name] = {"fn": cls.featurize, "feat_size": cls.feat_size}
    return cls


def register_atom_feat(cls: type) -> type:
    """Register an atom featurizer class in the global registry."""

    atom_feat_registry[cls.name] = {"fn": cls.featurize, "feat_size": cls.feat_size}
    return cls


def get_mol_feats_sizes(
    atom_feats: Sequence[str] | None, bond_feats: Sequence[str] | None, pe_embed_k: int
) -> tuple[int, int]:
    """Return the dimensions of atom and bond feature vectors for a MolGraph."""

    mg = MolGraph(atom_feats, bond_feats, pe_embed_k)
    return mg.num_atom_feats, mg.num_bond_feats


class MolGraph:
    def __init__(
        self,
        atom_feats: Sequence[str] | None = None,
        bond_feats: Sequence[str] | None = None,
        pe_embed_k: int = 0,
    ):
        """Create a molecular graph featurizer with atom/bond feature sets."""
        if atom_feats is None:
            atom_feats = [
                "a_onehot",
                "a_degree",
                "a_hybrid",
                "a_formal",
                "a_radical",
                "a_ring",
                "a_mass",
                "a_chiral",
            ]
        if bond_feats is None:
            bond_feats = ["b_degree"]

        self.pe_embed_k = pe_embed_k
        self.atom_feats = atom_feats
        self.bond_feats = bond_feats
        self.a_featurizers = []
        self.b_featurizers = []

        self.num_atom_feats = 0
        self.num_bond_feats = 0

        for i in self.atom_feats:
            if i not in atom_feat_registry:
                raise ValueError(f"Feat {i} not recognized")
            feat_obj = atom_feat_registry[i]
            self.num_atom_feats += feat_obj["feat_size"]
            self.a_featurizers.append(feat_obj["fn"])

        for i in self.bond_feats:
            if i not in bond_feat_registry:
                raise ValueError(f"Feat {i} not recognized")
            feat_obj = bond_feat_registry[i]
            self.num_bond_feats += feat_obj["feat_size"]
            self.b_featurizers.append(feat_obj["fn"])

        self.num_atom_feats += self.pe_embed_k

    def get_mol_graph(
        self,
        mol: Chem.rdchem.Mol,
        bigraph: bool = True,
    ) -> dict[str, np.ndarray]:
        """Build raw numpy feature tensors for a molecule.

        Args:
            mol: RDKit molecule.
            bigraph: If ``True``, create bidirectional edges.

        Returns:
            Mapping containing atom features (|N| x d_n), bond features (|E| x d_e),
            and bond index tuples (|E| x 2).
        """
        all_atoms = mol.GetAtoms()
        all_bonds = mol.GetBonds()
        bond_feats = []
        bond_tuples = []
        atom_feats = []
        for bond in all_bonds:
            strt = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            bond_tuples.append((strt, end))
            bond_feat = []
            for fn in self.b_featurizers:
                bond_feat.extend(fn(bond))
            bond_feats.append(bond_feat)

        for atom in all_atoms:
            atom_feat = []
            for fn in self.a_featurizers:
                atom_feat.extend(fn(atom))
            atom_feats.append(atom_feat)

        atom_feats = np.array(atom_feats)
        bond_feats = np.array(bond_feats)
        bond_tuples = np.array(bond_tuples)

        # Add doubles
        if bigraph:
            rev_bonds = np.vstack([bond_tuples[:, 1], bond_tuples[:, 0]]).transpose(1, 0)
            bond_tuples = np.vstack([bond_tuples, rev_bonds])
            bond_feats = np.vstack([bond_feats, bond_feats])
        return {
            "atom_feats": atom_feats,
            "bond_feats": bond_feats,
            "bond_tuples": bond_tuples,
        }

    def get_networkx_graph(self, mol: Chem.rdchem.Mol, bigraph: bool = True) -> nx.Graph:
        """Convert a molecule into a NetworkX graph with feature attributes.

        Args:
            mol: RDKit molecule object.
            bigraph: Whether to create bidirectional edges.

        Returns:
            NetworkX graph with node features 'h' and edge features 'e'.
        """
        mol_graph = self.get_mol_graph(mol, bigraph=bigraph)

        bond_inds = mol_graph["bond_tuples"]
        bond_feats = mol_graph["bond_feats"]
        atom_feats = mol_graph["atom_feats"]

        g = nx.Graph()
        g.add_nodes_from(range(atom_feats.shape[0]))
        g.add_edges_from(bond_inds)

        node_attr_dict = {i: atom_feats[i] for i in range(atom_feats.shape[0])}
        nx.set_node_attributes(g, values=node_attr_dict, name="h")
        edge_attr_dict = {
            (int(bond_inds[i, 0]), int(bond_inds[i, 1])): bond_feats[i]
            for i in range(bond_inds.shape[0])
        }
        nx.set_edge_attributes(g, values=edge_attr_dict, name="e")

        return g

    def get_pyg_graph(self, mol: Chem.rdchem.Mol, bigraph: bool = True) -> Data:
        """Convert molecule to PyTorch Geometric Data object with features.

        Args:
            mol: RDKit molecule object.
            bigraph: Whether to create bidirectional edges.

        Returns:
            PyTorch Geometric Data object with node and edge features.
        """
        mol_graph = self.get_mol_graph(mol, bigraph=bigraph)
        bond_inds = mol_graph["bond_tuples"]
        bond_feats = mol_graph["bond_feats"]
        atom_feats = mol_graph["atom_feats"]

        g = Data(
            x=th.from_numpy(atom_feats).float(),
            edge_index=th.from_numpy(bond_inds).long().transpose(1, 0),
            edge_attr=th.from_numpy(bond_feats).float(),
        )

        if self.pe_embed_k > 0:
            pe_embeds = random_walk_pe(
                g,
                k=self.pe_embed_k,
            )
            if g.x is None:
                raise ValueError("Node features (g.x) cannot be None when pe_embed_k > 0")
            g.x = th.cat([g.x, pe_embeds], dim=-1)

        return g


class FeatBase:
    """Base class for atom and bond featurizers.

    Extend this class to implement custom atom and bond featurization functions.
    """

    feat_size = 0
    name = "base"

    @classmethod
    def featurize(cls, x: Chem.rdchem.Atom | Chem.rdchem.Bond) -> list[int]:
        """Featurize an atom or bond.

        Args:
            x: RDKit atom or bond object to featurize.

        Returns:
            List of integer feature values.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError()


@register_atom_feat
class AtomOneHot(FeatBase):
    """Atom element type features are one-hot encoded."""

    name = "a_onehot"
    allowable_set = CANONICAL_ELEMENT_ORDER
    feat_size = len(allowable_set) + 1

    @classmethod
    def featurize(cls, x: Chem.rdchem.Atom) -> list[int]:
        """One-hot encode atom element type.

        Args:
            x: RDKit atom object.

        Returns:
            One-hot encoded element type vector.
        """
        return one_hot_encoding(x.GetSymbol(), cls.allowable_set, True)


@register_atom_feat
class AtomDegree(FeatBase):
    """Atom Degree features are one-hot encoded."""

    name = "a_degree"
    allowable_set = list(range(11))
    feat_size = len(allowable_set) + 1 + 2

    @classmethod
    def featurize(cls, x: Chem.rdchem.Atom) -> list[int]:
        """One-hot encode atom degree and total degree.

        Args:
            x: RDKit atom object.

        Returns:
            List containing degree values and one-hot encoded degree vector.
        """
        deg = [x.GetDegree(), x.GetTotalDegree()]
        onehot = one_hot_encoding(deg, cls.allowable_set, True)
        return deg + onehot


@register_atom_feat
class AtomHybrid(FeatBase):
    """Atom HybridizationType Charge features are one-hot encoded."""

    name = "a_hybrid"
    allowable_set = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    feat_size = len(allowable_set) + 1

    @classmethod
    def featurize(cls, x: Chem.rdchem.Atom) -> list[int]:
        """One-hot encode atom hybridization type.

        Args:
            x: RDKit atom object.

        Returns:
            One-hot encoded hybridization type vector.
        """
        onehot = one_hot_encoding(x.GetHybridization(), cls.allowable_set, True)
        return onehot


@register_atom_feat
class AtomFormal(FeatBase):
    """Atom Formal Charge features are one-hot encoded."""

    name = "a_formal"
    allowable_set = list(range(-2, 3))
    feat_size = len(allowable_set) + 1

    @classmethod
    def featurize(cls, x: Chem.rdchem.Atom) -> list[int]:
        """One-hot encode atom formal charge.

        Args:
            x: RDKit atom object.

        Returns:
            One-hot encoded formal charge vector.
        """
        onehot = one_hot_encoding(x.GetFormalCharge(), cls.allowable_set, True)
        return onehot


@register_atom_feat
class AtomRadical(FeatBase):
    """AtomRadical features are one-hot encoded."""

    name = "a_radical"
    allowable_set = list(range(5))
    feat_size = len(allowable_set) + 1

    @classmethod
    def featurize(cls, x: Chem.rdchem.Atom) -> list[int]:
        """One-hot encode atom radical electron count.

        Args:
            x: RDKit atom object.

        Returns:
            One-hot encoded radical electron count vector.
        """
        onehot = one_hot_encoding(x.GetNumRadicalElectrons(), cls.allowable_set, True)
        return onehot


@register_atom_feat
class AtomRing(FeatBase):
    """Atom Ring and Atom Aromatic features are one-hot encoded."""

    name = "a_ring"
    allowable_set = [True, False]
    feat_size = len(allowable_set) * 2

    @classmethod
    def featurize(cls, x: Chem.rdchem.Atom) -> list[int]:
        """One-hot encode atom ring and aromaticity status.

        Args:
            x: RDKit atom object.

        Returns:
            List containing one-hot encoded ring status and aromaticity vectors.
        """
        onehot_ring = one_hot_encoding(x.IsInRing(), cls.allowable_set, False)
        onehot_aromatic = one_hot_encoding(x.GetIsAromatic(), cls.allowable_set, False)
        return onehot_ring + onehot_aromatic


@register_atom_feat
class AtomChiral(FeatBase):
    """Atom Chiral features are one-hot encoded. The last feature is for unspecified chirality."""

    name = "a_chiral"
    allowable_set = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ]
    feat_size = len(allowable_set) + 1

    @classmethod
    def featurize(cls, x: Chem.rdchem.Atom) -> list[int]:
        """One-hot encode atom chiral type.

        Args:
            x: RDKit atom object.

        Returns:
            One-hot encoded chiral type vector.
        """
        chiral_onehot = one_hot_encoding(x.GetChiralTag(), cls.allowable_set, True)
        return chiral_onehot


@register_atom_feat
class AtomMass(FeatBase):
    """Atom Mass features normalized by a coefficient."""

    name = "a_mass"
    coef = 0.01
    feat_size = 1

    @classmethod
    def featurize(cls, x: Chem.rdchem.Atom) -> list[float]:
        """Compute normalized atomic mass.

        Args:
            x: RDKit atom object.

        Returns:
            List containing normalized atomic mass value.
        """
        return [x.GetMass() * cls.coef]


@register_bond_feat
class BondDegree(FeatBase):
    """Bond degree/type features are one-hot encoded."""

    name = "b_degree"
    allowable_set = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    feat_size = len(allowable_set) + 1

    @classmethod
    def featurize(cls, x: Chem.rdchem.Bond) -> list[int]:
        """One-hot encode bond type.

        Args:
            x: RDKit bond object.

        Returns:
            One-hot encoded bond type vector.
        """
        return one_hot_encoding(x.GetBondType(), cls.allowable_set, True)


@register_bond_feat
class BondStereo(FeatBase):
    """Bond Stereo features are one-hot encoded. The last feature is for unspecified stereochemistry."""

    name = "b_stereo"
    allowable_set = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
    ]
    feat_size = len(allowable_set) + 1

    @classmethod
    def featurize(cls, x: Chem.rdchem.Bond) -> list[int]:
        """One-hot encode bond stereochemistry.

        Args:
            x: RDKit bond object.

        Returns:
            One-hot encoded bond stereochemistry vector.
        """
        return one_hot_encoding(x.GetStereo(), cls.allowable_set, True)


@register_bond_feat
class BondRing(FeatBase):
    """Bond Ring features are one-hot encoded. The last feature is for unspecified ring status."""

    name = "b_ring"
    feat_size = 2

    @classmethod
    def featurize(cls, x: Chem.rdchem.Bond) -> list[int]:
        """One-hot encode bond ring membership.

        Args:
            x: RDKit bond object.

        Returns:
            One-hot encoded bond ring status vector.
        """
        return one_hot_encoding(x.IsInRing(), [False, True], False)


@register_bond_feat
class BondConj(FeatBase):
    """Bond Conjugation features are one-hot encoded. The last feature is for unspecified conjugation."""

    name = "b_conj"
    feat_size = 2

    @classmethod
    def featurize(cls, x: Chem.rdchem.Bond) -> list[int]:
        """One-hot encode bond conjugation status.

        Args:
            x: RDKit bond object.

        Returns:
            One-hot encoded bond conjugation vector.
        """
        return one_hot_encoding(x.GetIsConjugated(), [False, True], False)


def one_hot_encoding(x: Any, allowable_set: list[Any], encode_unknown: bool = False) -> list[int]:
    """One-hot encode ``x`` against an allowable set.

    Code adapted from the DGL LifeSci featurizers.

    Args:
        x: Value to encode.
        allowable_set: Enumerated set of allowable values.
        encode_unknown: If ``True``, append an unknown bucket and route missing values there.

    Returns:
        Binary indicator vector marking the matching allowable value.
    """

    if encode_unknown and (allowable_set[-1] is not None):
        allowable_set.append(None)

    if encode_unknown and (x not in allowable_set):
        x = None

    return [int(x == s) for s in allowable_set]


def batch_mols_frags(
    mol_pyg_list: Sequence[pyg.data.Data],
    frag_pyg_list: Sequence[pyg.data.Data],
    formula_peak_mzs_list: Sequence[th.Tensor],
    formula_peak_probs_list: Sequence[th.Tensor],
    debug_validate_outputs: bool = False,
) -> dict[str, Any]:
    """Batch molecule/fragment graphs and sparse formula peak tensors."""
    # batch size
    batch_size = len(mol_pyg_list)
    # Convert the input lists of molecules and fragments into PyTorch Geometric data batches
    mol_pyg = pyg.data.Batch.from_data_list(mol_pyg_list)
    frag_pyg = pyg.data.Batch.from_data_list(frag_pyg_list)

    # Assert that the number of graphs in both molecule and fragment batches matches the batch size
    assert mol_pyg.num_graphs == frag_pyg.num_graphs == batch_size
    # Assert that the node/edge feature indices are consistent. This is useful for
    # debugging data corruption, but expensive enough to keep off in normal training.
    if debug_validate_outputs:
        assert all(
            th.all(frag_pyg.node_feat_idxs[0] == frag_pyg.node_feat_idxs[i])
            for i in range(batch_size)
        )
        assert all(
            th.all(frag_pyg.edge_feat_idxs[0] == frag_pyg.edge_feat_idxs[i])
            for i in range(batch_size)
        )

    # Compute the cumulative sum of nodes for molecules and fragments, to track the total number of nodes per graph
    mol_num_nodes = [g.num_nodes for g in mol_pyg_list]
    frag_num_nodes = [g.num_nodes for g in frag_pyg_list]
    mol_num_nodes = th.cumsum(th.tensor([0] + mol_num_nodes), dim=0)
    frag_num_nodes = th.cumsum(th.tensor([0] + frag_num_nodes), dim=0)

    boundary_pair_frag_idxs = []
    boundary_pair_in_idxs = []
    boundary_pair_out_idxs = []
    for mol_idx, (mol_pyg_item, frag_pyg_item) in enumerate(zip(mol_pyg_list, frag_pyg_list)):
        if not hasattr(frag_pyg_item, "boundary_pair_frag_idxs"):
            raise ValueError(
                "frag_pyg is missing precomputed boundary_pair_* attributes; "
                "regenerate DAGs before using the boundary_pair feature"
            )
        frag_offsets = frag_num_nodes[mol_idx]
        mol_offsets = mol_num_nodes[mol_idx]
        boundary_pair_frag_idxs.append(frag_pyg_item.boundary_pair_frag_idxs + frag_offsets)
        boundary_pair_in_idxs.append(frag_pyg_item.boundary_pair_in_local + mol_offsets)
        boundary_pair_out_idxs.append(frag_pyg_item.boundary_pair_out_local + mol_offsets)

    # Initialize lists
    frag_formula_peak_idxs = []  # 1D tensor, indices of the formula_i of fragment_j created peak_k in fragments, concatenated across all fragments.
    frag_formula_peak_mzs = []
    frag_formula_peak_probs = []
    frag_formula_sizes = []  # 1D tensor, offset for each fragment's formula indices in the concatenated batch.
    frag_formula_peak_sizes = []  # 1D tensor, number of nonzero peaks per fragment. in case of isotopes, each formula can have more than one peak.

    # Loop through the mz and probs for each sample
    for mzs, probs in zip(formula_peak_mzs_list, formula_peak_probs_list):
        # sparsify: retain only non-zero probability peaks
        idx = th.nonzero(probs)
        # Extract mass-to-charge ratios where probability is non-zero
        mzs = mzs[idx[:, 0], idx[:, 1]]
        # Extract probabilities where probability is non-zero
        probs = probs[idx[:, 0], idx[:, 1]]
        # Append the filtered results for each fragment
        frag_formula_peak_idxs.append(idx[:, 0])
        frag_formula_peak_mzs.append(mzs)
        frag_formula_peak_probs.append(probs)

        # Calculate the number of unique formula peaks and the total peak sizes
        # In case of isotopes, each formula can have more then one peak
        frag_formula_sizes.append(th.unique(idx, sorted=True).shape[0])
        frag_formula_peak_sizes.append(idx.shape[0])

    # Flatten the lists to single dimension tensor
    frag_formula_peak_idxs = th.cat(frag_formula_peak_idxs, dim=0)
    frag_formula_peak_mzs = th.cat(frag_formula_peak_mzs, dim=0)
    frag_formula_peak_probs = th.cat(frag_formula_peak_probs, dim=0)
    frag_formula_cumsizes = th.cumsum(th.tensor([0] + frag_formula_sizes), dim=0)
    frag_formula_sizes = th.tensor(frag_formula_sizes)
    frag_formula_peak_sizes = th.tensor(frag_formula_peak_sizes)
    batched_mol_frag = {
        "mol_pyg": mol_pyg,  # PyG batch for molecules
        "frag_pyg": frag_pyg,  # PyG batch for fragments
        "mol_num_nodes": mol_num_nodes,  # Cumulative node count for molecules
        "frag_num_nodes": frag_num_nodes,  # Cumulative node count for fragments
        "frag_formula_peak_idxs": frag_formula_peak_idxs,  # Flattened tensor mapping each peak to its local formula index (0-based per molecule)
        "frag_formula_peak_mzs": frag_formula_peak_mzs,  # Flattened tensor of m/z values for each peak
        "frag_formula_peak_probs": frag_formula_peak_probs,  # Flattened tensor of theoretical isotopic probabilities for each peak
        "frag_formula_sizes": frag_formula_sizes,  # Tensor containing the number of unique formulas (fragments) with peaks for each molecule
        "frag_formula_cumsizes": frag_formula_cumsizes,  # Cumulative sum of frag_formula_sizes (offsets for formula indices in batch)
        "frag_formula_peak_sizes": frag_formula_peak_sizes,  # Tensor containing the total number of peaks (isotopes) for each molecule
    }
    batched_mol_frag["boundary_pair_frag_idxs"] = th.cat(boundary_pair_frag_idxs, dim=0)
    batched_mol_frag["boundary_pair_in_idxs"] = th.cat(boundary_pair_in_idxs, dim=0)
    batched_mol_frag["boundary_pair_out_idxs"] = th.cat(boundary_pair_out_idxs, dim=0)
    return batched_mol_frag


def get_mol_fp(
    mol: Chem.rdchem.Mol,
    morgan: bool,
    rdkit: bool,
    maccs: bool,
    morgan_radius: int = 3,
    morgan_nbits: int = 2048,
    rdkit_nbits: int = 2048,
) -> th.Tensor:
    """Compute concatenated molecular fingerprints as a float tensor.

    Args:
        mol: RDKit molecule object.
        morgan: Whether to include Morgan fingerprints.
        rdkit: Whether to include RDKit fingerprints.
        maccs: Whether to include MACCS fingerprints.
        morgan_radius: Radius for Morgan fingerprint (default: 3).
        morgan_nbits: Number of bits for Morgan fingerprint (default: 2048).
        rdkit_nbits: Number of bits for RDKit fingerprint (default: 2048).

    Returns:
        Float tensor containing concatenated fingerprints.

    Raises:
        AssertionError: If all fingerprint flags are False.
    """
    assert morgan or rdkit or maccs
    fps = []
    if morgan:
        fp = make_morgan_fingerprint(mol, radius=morgan_radius, nbits=morgan_nbits)
        fps.append(fp)
    if rdkit:
        fp = make_rdkit_fingerprint(mol, nbits=rdkit_nbits)
        fps.append(fp)
    if maccs:
        fp = make_maccs_fingerprint(mol)
        fps.append(fp)
    fp = th.as_tensor(np.concatenate(fps, axis=0), dtype=th.float)
    return fp


def get_mol_fp_size(
    morgan: bool,
    rdkit: bool,
    maccs: bool,
    morgan_radius: int = 3,
    morgan_nbits: int = 2048,
    rdkit_nbits: int = 2048,
) -> int:
    """Return the length of the concatenated fingerprint vector for a test molecule.

    Args:
        morgan: Whether to include Morgan fingerprints.
        rdkit: Whether to include RDKit fingerprints.
        maccs: Whether to include MACCS fingerprints.
        morgan_radius: Radius for Morgan fingerprint (default: 3).
        morgan_nbits: Number of bits for Morgan fingerprint (default: 2048).
        rdkit_nbits: Number of bits for RDKit fingerprint (default: 2048).

    Returns:
        Total length of concatenated fingerprint vector.
    """
    mol = Chem.MolFromSmiles("CCO")  # type: ignore[attr-defined]
    fp = get_mol_fp(mol, morgan, rdkit, maccs, morgan_radius, morgan_nbits, rdkit_nbits)
    return fp.shape[0]


def get_mol_graph(
    mol: Chem.rdchem.Mol,
    atom_feats: Sequence[str],
    bond_feats: Sequence[str],
    pe_embed_k: int,
    bigraph: bool,
) -> Data:
    """Create a PyG molecule graph with selected features and optional RWPE.

    Args:
        mol: RDKit molecule object.
        atom_feats: Sequence of atom feature names to include.
        bond_feats: Sequence of bond feature names to include.
        pe_embed_k: Number of random walk positional encoding steps (0 to disable).
        bigraph: Whether to create bidirectional edges.

    Returns:
        PyTorch Geometric Data object representing the molecule graph.
    """

    mg = MolGraph(atom_feats, bond_feats, pe_embed_k)
    mol_pyg = mg.get_pyg_graph(mol, bigraph=bigraph)
    return mol_pyg


def get_frag_graph(
    dag_pyg: Data,
    frag_node_feats: Sequence[str],
    frag_edge_feats: Sequence[str],
    edges: bool,
    bigraph: bool,
) -> Data:
    """Subselect DAG features and optionally symmetrize edges for fragments.

    Args:
        dag_pyg: Input PyTorch Geometric Data object with node and edge features.
        frag_node_feats: Sequence of node feature names to include.
        frag_edge_feats: Sequence of edge feature names to include.
        edges: Whether to include edges in the output graph.
        bigraph: Whether to symmetrize edges by adding reverse directions.

    Returns:
        PyTorch Geometric Data object with selected features and optionally symmetrized edges.

    Raises:
        ValueError: If dag_pyg.x (node features) is None.
    """

    source_dag_pyg = dag_pyg
    device = dag_pyg.x.device
    # cast
    x = dag_pyg.x.long()
    edge_attr = dag_pyg.edge_attr.long()
    edge_index = dag_pyg.edge_index.long()
    node_feat_idxs = dag_pyg.node_feat_idxs.long()
    edge_feat_idxs = dag_pyg.edge_feat_idxs.long()
    # select node features
    _x = []
    _node_feat_idxs = [0]
    _node_feat_size = 0
    for feat, _ in NODE_FEAT_TO_IDX.items():
        if feat in frag_node_feats:
            _x_cur = get_node_feats(x, node_feat_idxs[0], feat)
            _x.append(_x_cur)
            _node_feat_size += _x_cur.shape[1]
            # print(feat,feat_idx,_node_feat_size)
        _node_feat_idxs.append(_node_feat_size)
    _x = th.cat(_x, dim=1)
    _node_feat_idxs = th.tensor(_node_feat_idxs, device=device, dtype=th.int64).reshape(1, -1)
    # select edge features
    _edge_index = edge_index.clone()
    _edge_attr = []
    _edge_feat_idxs = [0]
    _edge_feat_size = 0
    for feat, _ in EDGE_FEAT_TO_IDX.items():
        if feat in frag_edge_feats:
            assert edges
            _edge_attr_cur = get_edge_feats(edge_attr, edge_feat_idxs[0], feat)
            _edge_attr.append(_edge_attr_cur)
            _edge_feat_size += _edge_attr_cur.shape[1]
            # print(feat,feat_idx,_edge_feat_size)
        _edge_feat_idxs.append(_edge_feat_size)
    if len(_edge_attr) == 0:
        _edge_attr = [th.zeros(edge_attr.shape[0], 1, device=device, dtype=th.int64)]
    _edge_attr = th.cat(_edge_attr, dim=1)
    _edge_feat_idxs = th.tensor(_edge_feat_idxs, device=device, dtype=th.int64).reshape(1, -1)
    if bigraph:
        assert edges
        _edge_index_c = th.stack([_edge_index[1], _edge_index[0]], dim=0)
        assert not th.any(th.all(_edge_index == _edge_index_c, dim=0), dim=0)
        _edge_attr_c = _edge_attr.clone()
        if "complement" in frag_edge_feats:
            _edge_attr_c[:, -1] = 1 - _edge_attr_c[:, -1]
        _edge_index = th.cat([_edge_index, _edge_index_c], dim=1)
        _edge_attr = th.cat([_edge_attr, _edge_attr_c], dim=0)
    if not edges:
        assert not bigraph
        _edge_index = _edge_index[:, :0]
        _edge_attr = _edge_attr[:0, :]
    # create pyg object
    dag_pyg = pyg.data.Data(
        x=_x,
        edge_index=_edge_index,
        edge_attr=_edge_attr,
        node_feat_idxs=_node_feat_idxs,
        edge_feat_idxs=_edge_feat_idxs,
    )
    for attr_name in ("boundary_pair_frag_idxs", "boundary_pair_in_local", "boundary_pair_out_local"):
        if hasattr(source_dag_pyg, attr_name):
            setattr(dag_pyg, attr_name, getattr(source_dag_pyg, attr_name))
    return dag_pyg


def random_walk_pe(g: Data, k: int) -> th.Tensor:
    """Random Walk Positional Encoding, as introduced in
    `Graph Neural Networks with Learnable Structural and Positional Representations
    <https://arxiv.org/abs/2110.07875>`__

    This function computes the random walk positional encodings as landing probabilities
    from 1-step to k-step, starting from each node to itself.

    Args:
        g: Homogeneous PyTorch Geometric Data object (graph).
        k: Number of random walk steps to include.

    Returns:
        Random walk positional encodings of shape (N, k) where N is the number of nodes.

    Example:
        >>> import dgl
        >>> g = dgl.graph(([0,1,1], [1,1,0]))
        >>> dgl.random_walk_pe(g, 2)
        tensor([[0.0000, 0.5000],
                [0.5000, 0.7500]])
    """
    # sparse adjacency matrix
    A = scipy.sparse.csr_matrix(
        (
            th.ones_like(g.edge_index[0]).numpy(),
            (g.edge_index[0].numpy(), g.edge_index[1].numpy()),
        )
    )
    RW = A / (A.sum(1).reshape(-1, 1) + 1e-30)  # 1-step transition probability

    # Iterate for k steps
    PE = [RW.diagonal()]
    RW_power = RW.copy()
    for _ in range(k - 1):
        RW_power = RW_power @ RW
        PE.append(RW_power.diagonal())
    PE = np.stack(PE, axis=-1)
    PE = th.as_tensor(PE, dtype=th.float32)
    return PE
