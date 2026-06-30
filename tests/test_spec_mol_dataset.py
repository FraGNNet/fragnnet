"""Integration tests for SpecMolDataset with fingerprint configuration."""

import torch as th
from rdkit import Chem

from fragnnet.utils.feat_utils import get_mol_fp


class TestSpecMolDatasetFingerprintConfig:
    """Tests for SpecMolDataset fingerprint configuration integration."""

    def test_fingerprint_size_with_default_params(self, mol_params_default):
        """Test that fingerprints have expected size with default parameters."""
        mol = Chem.MolFromSmiles("CCO")
        fp = get_mol_fp(
            mol,
            mol_params_default["fingerprint_morgan"],
            mol_params_default["fingerprint_rdkit"],
            mol_params_default["fingerprint_maccs"],
            morgan_radius=mol_params_default["morgan_radius"],
            morgan_nbits=mol_params_default["morgan_nbits"],
            rdkit_nbits=mol_params_default["rdkit_nbits"],
        )

        # Expected size: morgan_nbits + rdkit_nbits + MACCS (167)
        expected_size = 2048 + 2048 + 167
        assert len(fp) == expected_size

    def test_fingerprint_size_with_custom_radius(self, mol_params_custom_radius):
        """Test that custom morgan_radius still produces correct size."""
        mol = Chem.MolFromSmiles("CCO")
        fp = get_mol_fp(
            mol,
            mol_params_custom_radius["fingerprint_morgan"],
            mol_params_custom_radius["fingerprint_rdkit"],
            mol_params_custom_radius["fingerprint_maccs"],
            morgan_radius=mol_params_custom_radius["morgan_radius"],
            morgan_nbits=mol_params_custom_radius["morgan_nbits"],
            rdkit_nbits=mol_params_custom_radius["rdkit_nbits"],
        )

        # Only Morgan enabled, should match morgan_nbits
        assert len(fp) == 2048

    def test_fingerprint_size_with_custom_nbits(self, mol_params_custom_nbits):
        """Test that custom nbits parameters produce correct sizes."""
        mol = Chem.MolFromSmiles("CCO")
        fp = get_mol_fp(
            mol,
            mol_params_custom_nbits["fingerprint_morgan"],
            mol_params_custom_nbits["fingerprint_rdkit"],
            mol_params_custom_nbits["fingerprint_maccs"],
            morgan_radius=mol_params_custom_nbits["morgan_radius"],
            morgan_nbits=mol_params_custom_nbits["morgan_nbits"],
            rdkit_nbits=mol_params_custom_nbits["rdkit_nbits"],
        )

        # Expected size: 1024 (morgan) + 1024 (rdkit) + 167 (MACCS)
        expected_size = 1024 + 1024 + 167
        assert len(fp) == expected_size

    def test_backwards_compatibility_without_params(self):
        """Test that omitting new parameters uses defaults."""
        mol = Chem.MolFromSmiles("CCO")

        # Simulate old-style call (no new parameters in mol_params)
        mol_params = {
            "fingerprint": True,
            "fingerprint_morgan": True,
            "fingerprint_rdkit": True,
            "fingerprint_maccs": True,
            # Note: no morgan_radius, morgan_nbits, rdkit_nbits
        }

        # This should use defaults (3, 2048, 2048) via .get() in dataset
        fp = get_mol_fp(
            mol,
            mol_params["fingerprint_morgan"],
            mol_params["fingerprint_rdkit"],
            mol_params["fingerprint_maccs"],
            morgan_radius=mol_params.get("morgan_radius", 3),
            morgan_nbits=mol_params.get("morgan_nbits", 2048),
            rdkit_nbits=mol_params.get("rdkit_nbits", 2048),
        )

        # Should match default size
        expected_size = 2048 + 2048 + 167
        assert len(fp) == expected_size

    def test_different_molecules_same_config(self, mol_params_default):
        """Test that different molecules with same config produce same-sized fingerprints."""
        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("c1ccccc1")
        mol3 = Chem.MolFromSmiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")

        fp1 = get_mol_fp(
            mol1,
            mol_params_default["fingerprint_morgan"],
            mol_params_default["fingerprint_rdkit"],
            mol_params_default["fingerprint_maccs"],
            morgan_radius=mol_params_default["morgan_radius"],
            morgan_nbits=mol_params_default["morgan_nbits"],
            rdkit_nbits=mol_params_default["rdkit_nbits"],
        )

        fp2 = get_mol_fp(
            mol2,
            mol_params_default["fingerprint_morgan"],
            mol_params_default["fingerprint_rdkit"],
            mol_params_default["fingerprint_maccs"],
            morgan_radius=mol_params_default["morgan_radius"],
            morgan_nbits=mol_params_default["morgan_nbits"],
            rdkit_nbits=mol_params_default["rdkit_nbits"],
        )

        fp3 = get_mol_fp(
            mol3,
            mol_params_default["fingerprint_morgan"],
            mol_params_default["fingerprint_rdkit"],
            mol_params_default["fingerprint_maccs"],
            morgan_radius=mol_params_default["morgan_radius"],
            morgan_nbits=mol_params_default["morgan_nbits"],
            rdkit_nbits=mol_params_default["rdkit_nbits"],
        )

        # All should have same size
        assert len(fp1) == len(fp2) == len(fp3)

        # But different values (different molecules)
        assert not th.equal(fp1, fp2)
        assert not th.equal(fp1, fp3)
        assert not th.equal(fp2, fp3)

    def test_radius_affects_fingerprint_values(self):
        """Test that changing morgan_radius affects fingerprint values."""
        mol = Chem.MolFromSmiles("c1ccccc1")  # Benzene

        fp_r2 = get_mol_fp(
            mol,
            morgan=True,
            rdkit=False,
            maccs=False,
            morgan_radius=2,
            morgan_nbits=2048,
        )

        fp_r3 = get_mol_fp(
            mol,
            morgan=True,
            rdkit=False,
            maccs=False,
            morgan_radius=3,
            morgan_nbits=2048,
        )

        # Same size
        assert len(fp_r2) == len(fp_r3) == 2048

        # But different values (different radius captures different structure)
        assert not th.equal(fp_r2, fp_r3)

    def test_config_parameter_flow_consistency(self):
        """Test that parameters flow consistently through the pipeline."""
        mol = Chem.MolFromSmiles("CCO")

        # Simulate the flow: mol_params -> dataset -> get_mol_fp
        mol_params = {
            "fingerprint": True,
            "fingerprint_morgan": True,
            "fingerprint_rdkit": True,
            "fingerprint_maccs": False,
            "morgan_radius": 2,
            "morgan_nbits": 1024,
            "rdkit_nbits": 512,
        }

        fp = get_mol_fp(
            mol,
            mol_params["fingerprint_morgan"],
            mol_params["fingerprint_rdkit"],
            mol_params["fingerprint_maccs"],
            morgan_radius=mol_params.get("morgan_radius", 3),
            morgan_nbits=mol_params.get("morgan_nbits", 2048),
            rdkit_nbits=mol_params.get("rdkit_nbits", 2048),
        )

        # Expected: 1024 (morgan) + 512 (rdkit) + 0 (no MACCS)
        expected_size = 1024 + 512
        assert len(fp) == expected_size
