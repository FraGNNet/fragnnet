"""
Unit tests for 01_inf_prepare_mol_df.py.
"""

import importlib.util
import shutil
import tempfile
from pathlib import Path

import pytest


# Helper to import modules with numeric names
def _load_script_module(script_name):
    """Load a preprocessing script as a module."""
    script_path = Path(__file__).parent.parent / "preproc_scripts" / "inference" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Test01InfPrepareMolDf:
    """Tests for 01_inf_prepare_mol_df.py functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create and cleanup temporary directory."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def test_smiles_list(self):
        """Test SMILES strings."""
        return ["CCO", "CC(=O)O", "CCN", "C", "CC"]

    @pytest.fixture
    def test_json_data(self):
        """Test JSON candidates data."""
        return {"CCO": ["CC", "C", "CCN"], "CC(=O)O": ["CCO", "CC", "C"]}

    def test_smiles_list_input(self, temp_dir, test_smiles_list):
        """Test SMILES list input mode."""
        mol_df_module = _load_script_module("01_inf_prepare_mol_df.py")

        # Validate inputs
        mol_df_module.validate_inputs(test_smiles_list, max_heavy_atoms=128)

        # Create mol_df
        mol_df = mol_df_module.create_mol_df(test_smiles_list, ids_list=None, max_heavy_atoms=128)

        # Verify output
        assert len(mol_df) > 0
        assert "mol_id" in mol_df.columns
        assert "smiles" in mol_df.columns
        assert "mol" in mol_df.columns
        assert "formula" in mol_df.columns
        assert "inchikey_s" in mol_df.columns
        assert "exact_mw" in mol_df.columns
        assert mol_df["mol_id"].min() == 0
        assert all(mol_df["mol_id"] == range(len(mol_df)))

    def test_smiles_list_with_ids(self, temp_dir, test_smiles_list):
        """Test SMILES list with custom molecule IDs."""
        mol_df_module = _load_script_module("01_inf_prepare_mol_df.py")

        ids = [f"mol_{i}" for i in range(len(test_smiles_list))]
        mol_df = mol_df_module.create_mol_df(test_smiles_list, ids_list=ids, max_heavy_atoms=128)

        assert len(mol_df) > 0
        assert "input_mol_id" in mol_df.columns
        assert mol_df["input_mol_id"].notna().all()

    def test_charged_molecules_filtered(self, temp_dir):
        """Test that charged molecules are filtered out."""
        mol_df_module = _load_script_module("01_inf_prepare_mol_df.py")

        smiles_list = ["CCO", "[NH4+]", "CC(=O)O"]
        mol_df = mol_df_module.create_mol_df(smiles_list, ids_list=None, max_heavy_atoms=128)

        # Charged molecules should be filtered
        assert all(mol_df["charge"] == 0)

    def test_radical_molecules_filtered(self, temp_dir):
        """Test that molecules with radicals are filtered out."""
        mol_df_module = _load_script_module("01_inf_prepare_mol_df.py")

        smiles_list = ["CCO", "[CH3]", "CC(=O)O"]
        mol_df = mol_df_module.create_mol_df(smiles_list, ids_list=None, max_heavy_atoms=128)

        # Radical molecules should be filtered
        assert all(mol_df["num_radicals"] == 0)

    def test_heavy_atom_limit(self, temp_dir):
        """Test that molecules exceeding heavy atom limit are filtered."""
        mol_df_module = _load_script_module("01_inf_prepare_mol_df.py")

        # C60 has 60 heavy atoms
        smiles_list = ["CCO", "C1=CC=C2C=CC=CC2=C1"]  # naphthalene, 10 heavy atoms
        mol_df = mol_df_module.create_mol_df(smiles_list, ids_list=None, max_heavy_atoms=5)

        # Only CCO (3 heavy atoms) should remain
        assert len(mol_df) == 1
        assert mol_df.iloc[0]["formula"] == "C2H6O"

    def test_molecular_properties_computed(self, temp_dir):
        """Test that molecular properties are computed correctly."""
        mol_df_module = _load_script_module("01_inf_prepare_mol_df.py")

        mol_df = mol_df_module.create_mol_df(["CCO"], ids_list=None, max_heavy_atoms=128)
        assert len(mol_df) == 1
        row = mol_df.iloc[0]
        assert row["formula"] == "C2H6O"
        assert 45 < row["exact_mw"] < 47
        assert row["num_heavy_atoms"] == 3
        assert row["charge"] == 0
        assert row["num_radicals"] == 0

    def test_json_input_mode(self, temp_dir, test_json_data):
        """Test JSON input mode with candidate molecules."""
        mol_df_module = _load_script_module("01_inf_prepare_mol_df.py")

        # Extract unique SMILES from JSON
        unique_smiles = set()
        for query, candidates in test_json_data.items():
            unique_smiles.add(query)
            unique_smiles.update(candidates)

        smiles_list = list(unique_smiles)
        mol_df = mol_df_module.create_mol_df(smiles_list, ids_list=None, max_heavy_atoms=128)

        # Should have molecules created
        assert len(mol_df) > 0
        assert "mol_id" in mol_df.columns
