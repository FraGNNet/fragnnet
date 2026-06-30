"""Unit tests for fragnnet.utils.feat_utils module.

Tests for molecular graph featurization, fingerprints, and utility functions.
"""

import numpy as np
import pytest
import torch as th
from rdkit import Chem
from torch_geometric.data import Data

from fragnnet.utils.feat_utils import (
    AtomChiral,
    AtomDegree,
    AtomFormal,
    AtomHybrid,
    AtomMass,
    AtomOneHot,
    AtomRadical,
    AtomRing,
    BondConj,
    BondDegree,
    BondRing,
    BondStereo,
    MolGraph,
    batch_mols_frags,
    get_frag_graph,
    get_mol_feats_sizes,
    get_mol_fp,
    get_mol_fp_size,
    get_mol_graph,
    one_hot_encoding,
    random_walk_pe,
)


class TestOneHotEncoding:
    """Tests for one_hot_encoding function."""

    def test_basic_encoding_first_position(self):
        """Test encoding when value matches first position."""
        result = one_hot_encoding(0, [0, 1, 2, 3])
        assert result == [1, 0, 0, 0]
        assert sum(result) == 1

    def test_basic_encoding_middle_position(self):
        """Test encoding when value matches middle position."""
        result = one_hot_encoding(1, [0, 1, 2, 3])
        assert result == [0, 1, 0, 0]
        assert sum(result) == 1

    def test_basic_encoding_last_position(self):
        """Test encoding when value matches last position."""
        result = one_hot_encoding(3, [0, 1, 2, 3])
        assert result == [0, 0, 0, 1]
        assert sum(result) == 1

    def test_encode_unknown_false_unmatched(self):
        """When encode_unknown=False, unmatched values produce all zeros."""
        result = one_hot_encoding(99, [0, 1, 2], encode_unknown=False)
        assert result == [0, 0, 0]

    def test_encode_unknown_true_unmatched(self):
        """When encode_unknown=True, unmatched values go to unknown bucket (last position)."""
        result = one_hot_encoding(99, [0, 1, 2], encode_unknown=True)
        assert result == [0, 0, 0, 1]

    def test_numeric_values(self):
        result = one_hot_encoding(2.0, [1.0, 2.0, 3.0])
        assert result == [0, 1, 0]

    def test_string_values(self):
        """Test one-hot encoding with string values."""
        result = one_hot_encoding("C", ["C", "N", "O", "S"])
        assert result == [1, 0, 0, 0]

    def test_return_is_list(self):
        """Verify return type is always a list."""
        result = one_hot_encoding(1, [0, 1, 2])
        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)


class TestAtomFeaturizers:
    """Tests for atom featurizer classes."""

    @pytest.fixture
    def ethanol_mol(self):
        return Chem.MolFromSmiles("CCO")

    @pytest.fixture
    def benzene_mol(self):
        return Chem.MolFromSmiles("c1ccccc1")

    def test_atom_onehot_carbon(self, ethanol_mol):
        """Test AtomOneHot featurization for carbon atom."""
        atom = ethanol_mol.GetAtomWithIdx(0)  # Carbon
        feat = AtomOneHot.featurize(atom)
        assert len(feat) == AtomOneHot.feat_size
        assert sum(feat) == 1  # Exactly one position should be 1
        assert feat[0] == 1  # Carbon is at index 0 in CANONICAL_ELEMENT_ORDER

    def test_atom_onehot_oxygen(self, ethanol_mol):
        """Test AtomOneHot featurization for oxygen atom."""
        atom = ethanol_mol.GetAtomWithIdx(2)  # Oxygen
        feat = AtomOneHot.featurize(atom)
        assert len(feat) == AtomOneHot.feat_size
        assert sum(feat) == 1  # Exactly one position should be 1

    def test_atom_degree_carbon(self, ethanol_mol):
        """Test AtomDegree: first carbon has degree 1, total degree 4."""
        atom = ethanol_mol.GetAtomWithIdx(0)
        feat = AtomDegree.featurize(atom)
        assert len(feat) == AtomDegree.feat_size
        # First two values are degree and total_degree
        assert feat[0] == 1  # Degree (explicit bonds)
        assert feat[1] == 4  # Total degree (explicit + implicit H)
        # Rest should be one-hot encoding
        assert sum(feat[2:]) == 1

    def test_atom_hybrid_sp3(self, ethanol_mol):
        """Test AtomHybrid: aliphatic carbons are sp3."""
        atom = ethanol_mol.GetAtomWithIdx(0)
        feat = AtomHybrid.featurize(atom)
        assert len(feat) == AtomHybrid.feat_size
        assert sum(feat) == 1

    def test_atom_hybrid_sp2_aromatic(self, benzene_mol):
        """Test AtomHybrid: aromatic carbons are sp2."""
        atom = benzene_mol.GetAtomWithIdx(0)
        feat = AtomHybrid.featurize(atom)
        assert len(feat) == AtomHybrid.feat_size
        assert sum(feat) == 1

    def test_atom_formal_neutral(self, ethanol_mol):
        """Test AtomFormal: neutral atoms have formal charge 0."""
        atom = ethanol_mol.GetAtomWithIdx(0)
        feat = AtomFormal.featurize(atom)
        assert len(feat) == AtomFormal.feat_size
        assert sum(feat) == 1
        # Charge 0 is at index 2 in range(-2, 3)
        assert feat[2] == 1

    def test_atom_radical_zero(self, ethanol_mol):
        """Test AtomRadical: normal atoms have zero radical electrons."""
        atom = ethanol_mol.GetAtomWithIdx(0)
        feat = AtomRadical.featurize(atom)
        assert len(feat) == AtomRadical.feat_size
        assert sum(feat) == 1

    def test_atom_ring_not_in_ring(self, ethanol_mol):
        """Test AtomRing: ethanol atoms are not in rings and not aromatic."""
        atom = ethanol_mol.GetAtomWithIdx(0)
        feat = AtomRing.featurize(atom)
        assert len(feat) == AtomRing.feat_size
        # Format is [in_ring=False, in_ring=True, aromatic=False, aromatic=True]
        assert feat == [0, 1, 0, 1]

    def test_atom_ring_in_ring(self, benzene_mol):
        """Test AtomRing: benzene carbons are in rings and aromatic."""
        atom = benzene_mol.GetAtomWithIdx(0)
        feat = AtomRing.featurize(atom)
        assert len(feat) == AtomRing.feat_size
        # Format is [in_ring=False, in_ring=True, aromatic=False, aromatic=True]
        assert feat == [1, 0, 1, 0]

    def test_atom_chiral_unspecified(self, ethanol_mol):
        """Test AtomChiral: ethanol has no chiral centers."""
        atom = ethanol_mol.GetAtomWithIdx(0)
        feat = AtomChiral.featurize(atom)
        assert len(feat) == AtomChiral.feat_size
        assert sum(feat) == 1

    def test_atom_mass_positive(self, ethanol_mol):
        """Test AtomMass: mass should be positive and normalized."""
        atom = ethanol_mol.GetAtomWithIdx(0)  # Carbon
        feat = AtomMass.featurize(atom)
        assert len(feat) == AtomMass.feat_size
        assert len(feat) == 1
        assert feat[0] > 0
        # Carbon mass ~12 * 0.01 = 0.12
        assert 0.1 < feat[0] < 0.15


class TestBondFeaturizers:
    """Tests for bond featurizer classes."""

    @pytest.fixture
    def ethanol_mol(self):
        return Chem.MolFromSmiles("CCO")

    @pytest.fixture
    def acetylene_mol(self):
        return Chem.MolFromSmiles("C#C")

    @pytest.fixture
    def benzene_mol(self):
        return Chem.MolFromSmiles("c1ccccc1")

    def test_bond_degree_single(self, ethanol_mol):
        """Test BondDegree: C-C and C-O bonds in ethanol are single."""
        bond = ethanol_mol.GetBondWithIdx(0)  # C-C bond
        feat = BondDegree.featurize(bond)
        assert len(feat) == BondDegree.feat_size
        assert sum(feat) == 1
        # SINGLE bond type is first in allowable_set
        assert feat[0] == 1

    def test_bond_degree_double(self):
        """Test BondDegree: C=C bond has double bond type."""
        mol = Chem.MolFromSmiles("C=C")
        bond = mol.GetBondWithIdx(0)
        feat = BondDegree.featurize(bond)
        assert len(feat) == BondDegree.feat_size
        assert sum(feat) == 1
        # DOUBLE bond type is second in allowable_set
        assert feat[1] == 1

    def test_bond_degree_triple(self, acetylene_mol):
        """Test BondDegree: C#C bond has triple bond type."""
        bond = acetylene_mol.GetBondWithIdx(0)
        feat = BondDegree.featurize(bond)
        assert len(feat) == BondDegree.feat_size
        assert sum(feat) == 1
        # TRIPLE bond type is third in allowable_set
        assert feat[2] == 1

    def test_bond_degree_aromatic(self, benzene_mol):
        """Test BondDegree: aromatic bonds in benzene."""
        bond = benzene_mol.GetBondWithIdx(0)
        feat = BondDegree.featurize(bond)
        assert len(feat) == BondDegree.feat_size
        assert sum(feat) == 1
        # AROMATIC bond type is fourth in allowable_set
        assert feat[3] == 1

    def test_bond_stereo_none(self, ethanol_mol):
        """Test BondStereo: aliphatic bonds have no stereo."""
        bond = ethanol_mol.GetBondWithIdx(0)
        feat = BondStereo.featurize(bond)
        assert len(feat) == BondStereo.feat_size
        assert sum(feat) == 1

    def test_bond_ring_not_in_ring(self, ethanol_mol):
        """Test BondRing: ethanol bonds are not in rings."""
        bond = ethanol_mol.GetBondWithIdx(0)
        feat = BondRing.featurize(bond)
        assert len(feat) == BondRing.feat_size
        assert feat[0] == 1  # Not in ring
        assert feat[1] == 0

    def test_bond_ring_in_ring(self, benzene_mol):
        """Test BondRing: benzene bonds are in rings."""
        bond = benzene_mol.GetBondWithIdx(0)
        feat = BondRing.featurize(bond)
        assert len(feat) == BondRing.feat_size
        assert feat[0] == 0  # In ring
        assert feat[1] == 1

    def test_bond_conj_not_conjugated(self, ethanol_mol):
        """Test BondConj: aliphatic bonds are not conjugated."""
        bond = ethanol_mol.GetBondWithIdx(0)
        feat = BondConj.featurize(bond)
        assert len(feat) == BondConj.feat_size
        assert feat[0] == 1  # Not conjugated
        assert feat[1] == 0

    def test_bond_conj_conjugated(self):
        """Test BondConj: conjugated bonds in a conjugated system."""
        # Create a conjugated diene: C=C-C=C
        mol = Chem.MolFromSmiles("C=CC=C")
        # The middle C-C bond should be conjugated
        bond = mol.GetBondWithIdx(1)
        feat = BondConj.featurize(bond)
        assert len(feat) == BondConj.feat_size
        assert sum(feat) == 1


class TestMolGraph:
    """Tests for MolGraph class."""

    @pytest.fixture
    def ethanol_mol(self):
        return Chem.MolFromSmiles("CCO")

    @pytest.fixture
    def mol_graph(self):
        return MolGraph()

    def test_initialization_default_feats(self, mol_graph):
        """Test default feature initialization."""
        assert hasattr(mol_graph, "get_mol_graph")
        assert hasattr(mol_graph, "get_pyg_graph")
        assert mol_graph.num_atom_feats > 0
        assert mol_graph.num_bond_feats > 0
        assert mol_graph.pe_embed_k == 0

    def test_mol_graph_shape(self, mol_graph, ethanol_mol):
        """Test MolGraph produces correct feature shapes."""
        graph_dict = mol_graph.get_mol_graph(ethanol_mol, bigraph=False)
        assert "atom_feats" in graph_dict
        assert "bond_feats" in graph_dict
        assert "bond_tuples" in graph_dict

        atom_feats = graph_dict["atom_feats"]
        bond_feats = graph_dict["bond_feats"]
        bond_tuples = graph_dict["bond_tuples"]

        # Ethanol has 3 atoms (C, C, O)
        assert atom_feats.shape[0] == 3
        assert atom_feats.shape[1] == mol_graph.num_atom_feats

        # Ethanol has 2 bonds (C-C and C-O) without bigraph
        assert bond_feats.shape[0] == 2
        assert bond_feats.shape[1] == mol_graph.num_bond_feats
        assert bond_tuples.shape == (2, 2)

    def test_mol_graph_bigraph(self, mol_graph, ethanol_mol):
        """Test bigraph doubles the number of directed edges."""
        graph_no_bi = mol_graph.get_mol_graph(ethanol_mol, bigraph=False)
        graph_bi = mol_graph.get_mol_graph(ethanol_mol, bigraph=True)

        # With bigraph, edges should be doubled (reversed edges added)
        assert graph_bi["bond_tuples"].shape[0] == 2 * graph_no_bi["bond_tuples"].shape[0]
        assert graph_bi["bond_feats"].shape[0] == 2 * graph_no_bi["bond_feats"].shape[0]

    def test_pyg_graph_structure(self, mol_graph, ethanol_mol):
        """Test PyG Data object structure."""
        data = mol_graph.get_pyg_graph(ethanol_mol, bigraph=False)
        assert isinstance(data, Data)
        assert data.x is not None
        assert data.edge_index is not None
        assert data.edge_attr is not None

        # Check dtypes
        assert data.x.dtype == th.float32
        assert data.edge_index.dtype == th.int64
        assert data.edge_attr.dtype == th.float32

    def test_pyg_graph_dimensions(self, mol_graph, ethanol_mol):
        """Test PyG graph has correct dimensions."""
        data = mol_graph.get_pyg_graph(ethanol_mol, bigraph=False)

        # 3 atoms, each with num_atom_feats features
        assert data.x.shape == (3, mol_graph.num_atom_feats)

        # 2 edges in non-bigraph form, each with num_bond_feats features
        assert data.edge_attr.shape == (2, mol_graph.num_bond_feats)

        # Edge index has shape (2, num_edges)
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] == 2

    def test_pe_embed(self):
        """Test positional embedding is concatenated to atom features."""
        pe_k = 16
        g = MolGraph(pe_embed_k=pe_k)
        mol = Chem.MolFromSmiles("CCO")
        data = g.get_pyg_graph(mol)

        # Total feature size should include pe_embed_k
        assert data.x.shape[1] == g.num_atom_feats
        assert g.num_atom_feats > pe_k  # Should have atom features + PE

    def test_custom_feats(self):
        """Test MolGraph with custom feature set."""
        atom_feats = ["a_onehot"]
        bond_feats = ["b_degree"]
        g = MolGraph(atom_feats=atom_feats, bond_feats=bond_feats)

        assert g.num_atom_feats == AtomOneHot.feat_size
        assert g.num_bond_feats == BondDegree.feat_size

    def test_get_networkx_graph(self):
        """Test NetworkX graph creation."""
        mol_graph = MolGraph()
        mol = Chem.MolFromSmiles("CCO")
        g = mol_graph.get_networkx_graph(mol, bigraph=False)

        assert g.number_of_nodes() == 3
        assert g.number_of_edges() == 2


class TestGetMolFeaturesSizes:
    """Tests for get_mol_feats_sizes function."""

    def test_basic_sizes(self):
        atom_sizes, bond_sizes = get_mol_feats_sizes(None, None, 0)
        assert isinstance(atom_sizes, int) and atom_sizes > 0
        assert isinstance(bond_sizes, int) and bond_sizes > 0

    def test_with_pe(self):
        atom_sizes, bond_sizes = get_mol_feats_sizes(None, None, 16)
        assert isinstance(atom_sizes, int) and atom_sizes > 0
        assert isinstance(bond_sizes, int) and bond_sizes > 0


class TestGetMolFp:
    """Tests for get_mol_fp function."""

    @pytest.fixture
    def ethanol_mol(self):
        return Chem.MolFromSmiles("CCO")

    @pytest.fixture
    def benzene_mol(self):
        return Chem.MolFromSmiles("c1ccccc1")

    def test_morgan_only(self, ethanol_mol):
        """Test Morgan fingerprint generation."""
        fp = get_mol_fp(ethanol_mol, morgan=True, rdkit=False, maccs=False)
        assert fp is not None
        assert len(fp) > 0
        # Morgan is typically 2048 bits
        assert isinstance(fp, (list, th.Tensor, np.ndarray))

    def test_rdkit_only(self, ethanol_mol):
        """Test RDKit fingerprint generation."""
        fp = get_mol_fp(ethanol_mol, morgan=False, rdkit=True, maccs=False)
        assert fp is not None
        assert len(fp) > 0

    def test_maccs_only(self, ethanol_mol):
        """Test MACCS fingerprint generation."""
        fp = get_mol_fp(ethanol_mol, morgan=False, rdkit=False, maccs=True)
        assert fp is not None
        assert len(fp) > 0
        # MACCS has 167 bits
        assert len(fp) == 167

    def test_all_fingerprints(self, ethanol_mol):
        """Test combined fingerprints have correct concatenated length."""
        fp_all = get_mol_fp(ethanol_mol, morgan=True, rdkit=True, maccs=True)
        fp_morgan = get_mol_fp(ethanol_mol, morgan=True, rdkit=False, maccs=False)
        fp_rdkit = get_mol_fp(ethanol_mol, morgan=False, rdkit=True, maccs=False)
        fp_maccs = get_mol_fp(ethanol_mol, morgan=False, rdkit=False, maccs=True)

        assert fp_all is not None and len(fp_all) > 0
        # Combined should have approximately sum of individual lengths
        expected_len = len(fp_morgan) + len(fp_rdkit) + len(fp_maccs)
        assert len(fp_all) == expected_len

    def test_assertion_all_false(self, ethanol_mol):
        """Test that selecting no fingerprints raises AssertionError."""
        with pytest.raises(AssertionError):
            get_mol_fp(ethanol_mol, morgan=False, rdkit=False, maccs=False)

    def test_different_mols_different_fp(self, ethanol_mol, benzene_mol):
        """Test that different molecules produce different fingerprints."""
        fp_ethanol = get_mol_fp(ethanol_mol, morgan=True, rdkit=False, maccs=False)
        fp_benzene = get_mol_fp(benzene_mol, morgan=True, rdkit=False, maccs=False)

        # Fingerprints should be different for different molecules
        fp_arr_eth = np.array(fp_ethanol)
        fp_arr_ben = np.array(fp_benzene)
        assert not np.array_equal(fp_arr_eth, fp_arr_ben)

    @pytest.mark.parametrize("radius", [2, 3, 4])
    def test_morgan_radius_parameter(self, ethanol_mol, radius):
        """Test Morgan fingerprint with different radius values."""
        fp = get_mol_fp(
            ethanol_mol,
            morgan=True,
            rdkit=False,
            maccs=False,
            morgan_radius=radius,
        )
        assert fp is not None
        assert len(fp) > 0
        # Default nbits is 2048
        assert len(fp) == 2048

    @pytest.mark.parametrize("nbits", [1024, 2048, 4096, 1500])
    def test_morgan_nbits_parameter(self, ethanol_mol, nbits):
        """Test Morgan fingerprint with different nbits values."""
        fp = get_mol_fp(
            ethanol_mol,
            morgan=True,
            rdkit=False,
            maccs=False,
            morgan_nbits=nbits,
        )
        assert fp is not None
        assert len(fp) == nbits

    @pytest.mark.parametrize("nbits", [1024, 2048, 4096])
    def test_rdkit_nbits_parameter(self, ethanol_mol, nbits):
        """Test RDKit fingerprint with different nbits values."""
        fp = get_mol_fp(
            ethanol_mol,
            morgan=False,
            rdkit=True,
            maccs=False,
            rdkit_nbits=nbits,
        )
        assert fp is not None
        assert len(fp) == nbits

    def test_combined_custom_nbits(self, ethanol_mol):
        """Test combined fingerprints with custom nbits values."""
        morgan_nbits = 1024
        rdkit_nbits = 512
        fp = get_mol_fp(
            ethanol_mol,
            morgan=True,
            rdkit=True,
            maccs=True,
            morgan_nbits=morgan_nbits,
            rdkit_nbits=rdkit_nbits,
        )
        assert fp is not None
        # Total length should be morgan_nbits + rdkit_nbits + 167 (MACCS)
        expected_len = morgan_nbits + rdkit_nbits + 167
        assert len(fp) == expected_len

    def test_backwards_compatibility_defaults(self, ethanol_mol):
        """Test that default parameters match previous behavior."""
        # Old call (no explicit parameters)
        fp_old = get_mol_fp(ethanol_mol, morgan=True, rdkit=False, maccs=False)
        # New call with explicit defaults
        fp_new = get_mol_fp(
            ethanol_mol,
            morgan=True,
            rdkit=False,
            maccs=False,
            morgan_radius=3,
            morgan_nbits=2048,
        )
        assert len(fp_old) == len(fp_new)
        assert len(fp_new) == 2048


class TestGetMolFpParameterValidation:
    """Tests for fingerprint parameter validation."""

    @pytest.fixture
    def ethanol_mol(self):
        return Chem.MolFromSmiles("CCO")

    def test_morgan_radius_zero_raises_error(self, ethanol_mol):
        """Test that morgan_radius=0 raises ValueError."""
        with pytest.raises(ValueError, match="morgan_radius must be positive"):
            get_mol_fp(
                ethanol_mol,
                morgan=True,
                rdkit=False,
                maccs=False,
                morgan_radius=0,
            )

    def test_morgan_radius_negative_raises_error(self, ethanol_mol):
        """Test that negative morgan_radius raises ValueError."""
        with pytest.raises(ValueError, match="morgan_radius must be positive"):
            get_mol_fp(
                ethanol_mol,
                morgan=True,
                rdkit=False,
                maccs=False,
                morgan_radius=-1,
            )

    def test_morgan_nbits_zero_raises_error(self, ethanol_mol):
        """Test that morgan_nbits=0 raises ValueError."""
        with pytest.raises(ValueError, match="morgan_nbits must be positive"):
            get_mol_fp(
                ethanol_mol,
                morgan=True,
                rdkit=False,
                maccs=False,
                morgan_nbits=0,
            )

    def test_morgan_nbits_negative_raises_error(self, ethanol_mol):
        """Test that negative morgan_nbits raises ValueError."""
        with pytest.raises(ValueError, match="morgan_nbits must be positive"):
            get_mol_fp(
                ethanol_mol,
                morgan=True,
                rdkit=False,
                maccs=False,
                morgan_nbits=-1,
            )

    def test_rdkit_nbits_zero_raises_error(self, ethanol_mol):
        """Test that rdkit_nbits=0 raises ValueError."""
        with pytest.raises(ValueError, match="rdkit_nbits must be positive"):
            get_mol_fp(
                ethanol_mol,
                morgan=False,
                rdkit=True,
                maccs=False,
                rdkit_nbits=0,
            )

    def test_rdkit_nbits_negative_raises_error(self, ethanol_mol):
        """Test that negative rdkit_nbits raises ValueError."""
        with pytest.raises(ValueError, match="rdkit_nbits must be positive"):
            get_mol_fp(
                ethanol_mol,
                morgan=False,
                rdkit=True,
                maccs=False,
                rdkit_nbits=-1,
            )


class TestGetMolFpSize:
    """Tests for get_mol_fp_size function."""

    def test_morgan_size(self):
        """Test Morgan fingerprint size is consistent."""
        size = get_mol_fp_size(morgan=True, rdkit=False, maccs=False)
        assert size > 0
        assert isinstance(size, int)

    def test_rdkit_size(self):
        """Test RDKit fingerprint size is consistent."""
        size = get_mol_fp_size(morgan=False, rdkit=True, maccs=False)
        assert size > 0

    def test_maccs_size(self):
        """Test MACCS fingerprint size is exactly 167."""
        size = get_mol_fp_size(morgan=False, rdkit=False, maccs=True)
        assert size == 167

    def test_all_fingerprints_size(self):
        """Test combined fingerprint size."""
        size_all = get_mol_fp_size(morgan=True, rdkit=True, maccs=True)
        size_morgan = get_mol_fp_size(morgan=True, rdkit=False, maccs=False)
        size_rdkit = get_mol_fp_size(morgan=False, rdkit=True, maccs=False)
        size_maccs = get_mol_fp_size(morgan=False, rdkit=False, maccs=True)

        assert size_all == size_morgan + size_rdkit + size_maccs

    @pytest.mark.parametrize("nbits", [1024, 2048, 4096])
    def test_morgan_custom_nbits_size(self, nbits):
        """Test Morgan fingerprint size with custom nbits."""
        size = get_mol_fp_size(
            morgan=True,
            rdkit=False,
            maccs=False,
            morgan_nbits=nbits,
        )
        assert size == nbits

    @pytest.mark.parametrize("nbits", [512, 1024, 2048])
    def test_rdkit_custom_nbits_size(self, nbits):
        """Test RDKit fingerprint size with custom nbits."""
        size = get_mol_fp_size(
            morgan=False,
            rdkit=True,
            maccs=False,
            rdkit_nbits=nbits,
        )
        assert size == nbits

    def test_combined_custom_nbits_size(self):
        """Test combined fingerprint size with custom nbits."""
        morgan_nbits = 1024
        rdkit_nbits = 512
        size = get_mol_fp_size(
            morgan=True,
            rdkit=True,
            maccs=True,
            morgan_nbits=morgan_nbits,
            rdkit_nbits=rdkit_nbits,
        )
        expected_size = morgan_nbits + rdkit_nbits + 167  # MACCS is always 167
        assert size == expected_size


class TestGetMolGraphFunc:
    """Tests for get_mol_graph function."""

    def test_basic_graph(self):
        """Test basic graph creation with default parameters."""
        mol = Chem.MolFromSmiles("CCO")
        g = get_mol_graph(mol, None, None, 0, bigraph=True)
        assert g is not None
        assert isinstance(g, Data)
        assert g.x is not None
        assert g.edge_index is not None

    def test_graph_nodes_and_edges(self):
        """Test graph has correct number of nodes and edges."""
        mol = Chem.MolFromSmiles("CCO")  # 3 atoms, 2 bonds
        g = get_mol_graph(mol, None, None, 0, bigraph=False)

        assert g.x.shape[0] == 3  # 3 nodes
        assert g.edge_index.shape[1] == 2  # 2 edges (non-bigraph)

    def test_bigraph_doubling(self):
        """Test bigraph doubles the number of edges."""
        mol = Chem.MolFromSmiles("CCO")
        g_no_bi = get_mol_graph(mol, None, None, 0, bigraph=False)
        g_bi = get_mol_graph(mol, None, None, 0, bigraph=True)

        # Bigraph should have twice the edges
        assert g_bi.edge_index.shape[1] == 2 * g_no_bi.edge_index.shape[1]

    def test_with_pe(self):
        """Test graph with positional embeddings."""
        mol = Chem.MolFromSmiles("CCO")
        g = get_mol_graph(mol, None, None, pe_embed_k=8, bigraph=True)
        assert g is not None
        assert isinstance(g, Data)
        # PE should be added to node features
        assert g.x is not None

    def test_benzene_graph(self):
        """Test graph creation for aromatic molecule."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        g = get_mol_graph(mol, None, None, 0, bigraph=False)

        assert g.x.shape[0] == 6  # 6 carbons
        assert g.edge_index.shape[1] == 6  # 6 bonds in benzene ring


class TestBatchMolsFrags:
    """Tests for batch_mols_frags function.

    Note: This function requires properly formatted PyG Data objects
    with node_feat_idxs and edge_feat_idxs from preprocessing.
    """

    def test_batch_mols_frags_callable(self):
        """Test that batch_mols_frags is callable."""
        assert callable(batch_mols_frags)
        # Check it has the expected signature
        import inspect

        sig = inspect.signature(batch_mols_frags)
        assert len(sig.parameters) > 0

    def test_batch_mols_frags_sparse_peaks(self):
        """Test batching with sparse peak probabilities."""
        node_feat_idxs = th.tensor([[0, 1]], dtype=th.int64)
        edge_feat_idxs = th.tensor([[0, 1]], dtype=th.int64)

        mol_pyg_list = [
            Data(x=th.zeros((2, 1)), edge_index=th.tensor([[0], [1]]), edge_attr=th.zeros((1, 1))),
            Data(x=th.zeros((3, 1)), edge_index=th.tensor([[0], [2]]), edge_attr=th.zeros((1, 1))),
        ]
        frag_pyg_list = [
            Data(
                x=th.zeros((2, 1)),
                edge_index=th.tensor([[0], [1]]),
                edge_attr=th.zeros((1, 1)),
                node_feat_idxs=node_feat_idxs,
                edge_feat_idxs=edge_feat_idxs,
                boundary_pair_frag_idxs=th.zeros(0, dtype=th.long),
                boundary_pair_in_local=th.zeros(0, dtype=th.long),
                boundary_pair_out_local=th.zeros(0, dtype=th.long),
            ),
            Data(
                x=th.zeros((3, 1)),
                edge_index=th.tensor([[0], [2]]),
                edge_attr=th.zeros((1, 1)),
                node_feat_idxs=node_feat_idxs,
                edge_feat_idxs=edge_feat_idxs,
                boundary_pair_frag_idxs=th.zeros(0, dtype=th.long),
                boundary_pair_in_local=th.zeros(0, dtype=th.long),
                boundary_pair_out_local=th.zeros(0, dtype=th.long),
            ),
        ]

        mzs1 = th.tensor([[10.0, 11.0], [20.0, 21.0]])
        probs1 = th.tensor([[1.0, 0.0], [0.0, 0.0]])
        mzs2 = th.tensor([[30.0, 31.0], [40.0, 41.0]])
        probs2 = th.tensor([[0.1, 0.0], [0.2, 0.3]])

        out = batch_mols_frags(
            mol_pyg_list=mol_pyg_list,
            frag_pyg_list=frag_pyg_list,
            formula_peak_mzs_list=[mzs1, mzs2],
            formula_peak_probs_list=[probs1, probs2],
        )

        # Indices are per-molecule, per-formula: the first molecule's first formula (index 0)
        # contributes one non-zero isotope peak, and the second molecule's second formula
        # (index 1) contributes three non-zero peaks, yielding the expected [0, 0, 1, 1].
        assert out["frag_formula_peak_idxs"].tolist() == [0, 0, 1, 1]
        assert th.allclose(
            out["frag_formula_peak_mzs"],
            th.tensor([10.0, 30.0, 40.0, 41.0]),
            atol=1e-6,
            rtol=1e-6,
        )
        assert th.allclose(
            out["frag_formula_peak_probs"],
            th.tensor([1.0, 0.1, 0.2, 0.3]),
            atol=1e-6,
            rtol=1e-6,
        )
        assert out["frag_formula_sizes"].tolist() == [1, 2]
        assert out["frag_formula_peak_sizes"].tolist() == [1, 3]
        assert out["frag_formula_cumsizes"].tolist() == [0, 1, 3]

    def test_batch_mols_frags_offsets_boundary_pair_cache(self):
        """Boundary-pair caches should be offset into the global batched index space."""
        node_feat_idxs = th.tensor([[0, 1]], dtype=th.int64)
        edge_feat_idxs = th.tensor([[0, 1]], dtype=th.int64)

        mol_pyg_list = [
            Data(x=th.zeros((2, 1)), edge_index=th.tensor([[0], [1]]), edge_attr=th.zeros((1, 1))),
            Data(x=th.zeros((3, 1)), edge_index=th.tensor([[0], [2]]), edge_attr=th.zeros((1, 1))),
        ]
        frag_pyg_list = [
            Data(
                x=th.zeros((2, 1)),
                edge_index=th.tensor([[0], [1]]),
                edge_attr=th.zeros((1, 1)),
                node_feat_idxs=node_feat_idxs,
                edge_feat_idxs=edge_feat_idxs,
                boundary_pair_frag_idxs=th.tensor([1], dtype=th.long),
                boundary_pair_in_local=th.tensor([0], dtype=th.long),
                boundary_pair_out_local=th.tensor([1], dtype=th.long),
            ),
            Data(
                x=th.zeros((3, 1)),
                edge_index=th.tensor([[0], [2]]),
                edge_attr=th.zeros((1, 1)),
                node_feat_idxs=node_feat_idxs,
                edge_feat_idxs=edge_feat_idxs,
                boundary_pair_frag_idxs=th.tensor([2], dtype=th.long),
                boundary_pair_in_local=th.tensor([2], dtype=th.long),
                boundary_pair_out_local=th.tensor([1], dtype=th.long),
            ),
        ]

        out = batch_mols_frags(
            mol_pyg_list=mol_pyg_list,
            frag_pyg_list=frag_pyg_list,
            formula_peak_mzs_list=[th.tensor([[10.0]]), th.tensor([[20.0]])],
            formula_peak_probs_list=[th.tensor([[1.0]]), th.tensor([[1.0]])],
        )

        assert out["boundary_pair_frag_idxs"].tolist() == [1, 4]
        assert out["boundary_pair_in_idxs"].tolist() == [0, 4]
        assert out["boundary_pair_out_idxs"].tolist() == [1, 3]

    def test_batch_mols_frags_requires_precomputed_boundary_pair_attrs(self):
        """Batching should fail fast when DAG-derived boundary-pair attrs are missing."""
        node_feat_idxs = th.tensor([[0, 1]], dtype=th.int64)
        edge_feat_idxs = th.tensor([[0, 1]], dtype=th.int64)

        mol_pyg_list = [
            Data(x=th.zeros((2, 1)), edge_index=th.tensor([[0], [1]]), edge_attr=th.zeros((1, 1)))
        ]
        frag_pyg_list = [
            Data(
                x=th.zeros((2, 1)),
                edge_index=th.tensor([[0], [1]]),
                edge_attr=th.zeros((1, 1)),
                node_feat_idxs=node_feat_idxs,
                edge_feat_idxs=edge_feat_idxs,
            )
        ]

        with pytest.raises(ValueError, match="missing precomputed boundary_pair_"):
            batch_mols_frags(
                mol_pyg_list=mol_pyg_list,
                frag_pyg_list=frag_pyg_list,
                formula_peak_mzs_list=[th.tensor([[10.0]])],
                formula_peak_probs_list=[th.tensor([[1.0]])],
            )


class TestGetFragGraph:
    """Tests for get_frag_graph function.

    Note: These are smoke tests. Full feature testing requires proper
    node/edge feature organization from actual preprocessing.
    """

    def test_smoke_frag_graph(self):
        """Smoke test that get_frag_graph is callable."""
        assert callable(get_frag_graph)
        # Check it has the expected signature
        import inspect

        sig = inspect.signature(get_frag_graph)
        assert len(sig.parameters) > 0

    def test_frag_graph_selects_feats_and_bigraph(self):
        """Test feature selection and complement handling in bigraph mode."""
        node_feat_idxs = th.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=th.int64)
        edge_feat_idxs = th.tensor([[0, 1, 2, 3, 4]], dtype=th.int64)
        x = th.tensor([[10, 11, 12, 13, 14, 15, 16], [20, 21, 22, 23, 24, 25, 26]])
        edge_index = th.tensor([[0, 1], [1, 0]], dtype=th.int64)
        edge_attr = th.tensor([[1, 2, 3, 0], [4, 5, 6, 1]], dtype=th.int64)
        dag = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_feat_idxs=node_feat_idxs,
            edge_feat_idxs=edge_feat_idxs,
        )

        out = get_frag_graph(
            dag_pyg=dag,
            frag_node_feats=["depth", "h_formulae_idx"],
            frag_edge_feats=["cc", "complement"],
            edges=True,
            bigraph=True,
        )

        expected_x = th.stack([x[:, 0], x[:, 3]], dim=1)
        assert th.equal(out.x, expected_x)

        expected_edge = th.stack([edge_attr[:, 0], edge_attr[:, 3]], dim=1)
        assert out.edge_attr.shape == (edge_attr.shape[0] * 2, expected_edge.shape[1])
        assert th.equal(out.edge_attr[: edge_attr.shape[0]], expected_edge)
        reversed_expected = expected_edge.clone()
        reversed_expected[:, 1] = 1 - reversed_expected[:, 1]
        assert th.equal(out.edge_attr[edge_attr.shape[0] :], reversed_expected)

    def test_frag_graph_no_edges(self):
        """Test edge stripping when edges=False."""
        node_feat_idxs = th.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=th.int64)
        edge_feat_idxs = th.tensor([[0, 1, 2, 3, 4]], dtype=th.int64)
        x = th.tensor([[0, 1, 2, 3, 4, 5, 6]])
        edge_index = th.tensor([[0], [0]], dtype=th.int64)
        edge_attr = th.tensor([[1, 2, 3, 0]], dtype=th.int64)
        dag = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_feat_idxs=node_feat_idxs,
            edge_feat_idxs=edge_feat_idxs,
        )

        out = get_frag_graph(
            dag_pyg=dag,
            frag_node_feats=["depth"],
            frag_edge_feats=[],
            edges=False,
            bigraph=False,
        )

        assert out.edge_index.shape[1] == 0
        assert out.edge_attr.shape[0] == 0


class TestRandomWalkPE:
    """Tests for random_walk_pe function."""

    def test_basic_pe_callable(self):
        """Test that random_walk_pe is callable."""
        assert callable(random_walk_pe)

    def test_pe_with_simple_graph(self):
        """Test PE computation on a simple cyclic graph."""
        # Create a simple cyclic graph: 0-1-2-0
        edge_index = th.tensor([[0, 1, 2], [1, 2, 0]])
        data = Data(edge_index=edge_index)
        pe = random_walk_pe(data, k=2)

        assert pe is not None
        assert isinstance(pe, th.Tensor)
        assert pe.shape[1] == 2  # k=2
        assert pe.shape[0] == 3  # 3 nodes

    def test_pe_different_k_values(self):
        """Test PE with different k values."""
        edge_index = th.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        data = Data(edge_index=edge_index)

        for k in [1, 4, 8, 16]:
            pe = random_walk_pe(data, k=k)
            assert pe.shape[1] == k
            assert pe.shape[0] == 4

    def test_pe_linear_graph(self):
        """Test PE on a linear graph 0-1-2-3."""
        # Simple undirected graph: 0-1
        edge_index = th.tensor([[0, 1], [1, 0]], dtype=th.long)
        data = Data(edge_index=edge_index, num_nodes=2)
        pe = random_walk_pe(data, k=2)

        assert pe is not None
        assert isinstance(pe, th.Tensor)
        assert pe.shape == (2, 2)

    def test_pe_output_properties(self):
        """Test PE output has expected numerical properties."""
        edge_index = th.tensor([[0, 1], [1, 0]])
        data = Data(edge_index=edge_index)
        pe = random_walk_pe(data, k=4)

        # PE should be real valued
        assert pe.dtype in [th.float32, th.float64]
        # PE should be finite
        assert th.isfinite(pe).all()

    def test_pe_two_node_back_and_forth(self):
        """Test PE values for a 2-node undirected graph."""
        # 0 <-> 1 so the 1-step return prob is 0, 2-step return prob is 1
        edge_index = th.tensor([[0, 1], [1, 0]], dtype=th.long)
        data = Data(edge_index=edge_index, num_nodes=2)
        pe = random_walk_pe(data, k=2)

        expected = th.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=pe.dtype)
        assert th.allclose(pe, expected, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
