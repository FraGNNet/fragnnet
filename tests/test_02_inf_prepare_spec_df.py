"""
Unit tests for 02_inf_prepare_spec_df.py.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


# Helper to import modules with numeric names
def _load_script_module(script_name):
    """Load a preprocessing script as a module."""
    script_path = Path(__file__).parent.parent / "preproc_scripts" / "inference" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Test02InfPrepareSpecDf:
    """Tests for 02_inf_prepare_spec_df.py functionality."""

    @pytest.fixture
    def test_mol_df(self):
        """Create a test mol_df."""
        return pd.DataFrame(
            {
                "mol_id": [0, 1, 2],
                "smiles": ["CCO", "CC(=O)O", "CCN"],
                "formula": ["C2H6O", "C2H4O2", "C2H7N"],
                "inchikey_s": [
                    "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                    "QTBSBXVTEAMEQO-UHFFFAOYSA-N",
                    "XGZVUUAUJVXPNA-UHFFFAOYSA-N",
                ],
                "exact_mw": [46.04, 60.02, 45.06],
            }
        )

    def test_config_building(self, test_mol_df):
        """Test basic configuration building."""
        spec_df_module = _load_script_module("02_inf_prepare_spec_df.py")

        config = spec_df_module.get_config(
            prec_types=["[M+H]+"],
            qtof_ces=[],
            ft_ces=["25.0NCE"],
            inst_types=["FT"],
            output_dir="/tmp",
            merged_spectra=False,
        )

        assert "combinations" in config
        assert len(config["combinations"]) == 1
        assert config["combinations"][0]["ce"] == 25.0
        assert config["combinations"][0]["ce_unit"] == "NCE"

    def test_config_multiple_conditions(self, test_mol_df):
        """Test config with multiple experimental combinations."""
        spec_df_module = _load_script_module("02_inf_prepare_spec_df.py")

        config = spec_df_module.get_config(
            prec_types=["[M+H]+", "[M-H]-"],
            qtof_ces=[],
            ft_ces=["25.0NCE", "35.0NCE"],
            inst_types=["FT"],
            output_dir="/tmp",
            merged_spectra=False,
        )

        # Should have 2 prec_types * 2 CE values = 4 combinations
        assert len(config["combinations"]) == 4

    def test_spec_df_creation(self, test_mol_df):
        """Test basic spec_df creation."""
        spec_df_module = _load_script_module("02_inf_prepare_spec_df.py")

        config = spec_df_module.get_config(
            prec_types=["[M+H]+"],
            qtof_ces=[],
            ft_ces=["25.0NCE"],
            inst_types=["FT"],
            output_dir="/tmp",
            merged_spectra=False,
        )

        spec_df = spec_df_module.create_spec_df(test_mol_df, config)

        # Should have molecules * combinations entries
        assert len(spec_df) == len(test_mol_df) * len(config["combinations"])

        required_cols = [
            "spec_id",
            "mol_id",
            "prec_type",
            "nce",
            "ace",
            "inst_type",
            "group_id",
            "dset",
            "prec_mz",
            "peaks",
            "frag_mode",
            "formula",
            "inchikey",
            "exact_mass",
        ]
        for col in required_cols:
            assert col in spec_df.columns

    def test_spec_df_with_nce_values(self, test_mol_df):
        """Test spec_df with NCE collision energies."""
        spec_df_module = _load_script_module("02_inf_prepare_spec_df.py")

        config = spec_df_module.get_config(
            prec_types=["[M+H]+"],
            qtof_ces=[],
            ft_ces=["25.0NCE", "35.0NCE"],
            inst_types=["FT"],
            output_dir="/tmp",
            merged_spectra=False,
        )

        spec_df = spec_df_module.create_spec_df(test_mol_df, config)

        # Should have entries with NCE values (all populated for FT)
        assert spec_df["nce"].notna().all()  # All NCE values should be present for FT
        assert spec_df["ace"].isnull().all()  # All ACE should be NaN when using FT
        assert all(spec_df["nce"] > 0)

    def test_spec_df_with_ace_values(self, test_mol_df):
        """Test spec_df with ACE (eV) collision energies."""
        spec_df_module = _load_script_module("02_inf_prepare_spec_df.py")

        config = spec_df_module.get_config(
            prec_types=["[M+H]+"],
            qtof_ces=["10.0eV", "20.0eV"],
            ft_ces=[],
            inst_types=["QTOF"],
            output_dir="/tmp",
            merged_spectra=False,
        )

        spec_df = spec_df_module.create_spec_df(test_mol_df, config)

        # Should have ACE values for QTOF
        assert spec_df["ace"].notna().all()
        assert spec_df["nce"].isnull().all()
        assert all(spec_df["ace"] > 0)

    def test_spec_df_precursor_mz(self, test_mol_df):
        """Test that precursor m/z is computed correctly."""
        spec_df_module = _load_script_module("02_inf_prepare_spec_df.py")

        config = spec_df_module.get_config(
            prec_types=["[M+H]+"],
            qtof_ces=[],
            ft_ces=["25.0NCE"],
            inst_types=["FT"],
            output_dir="/tmp",
            merged_spectra=False,
        )

        spec_df = spec_df_module.create_spec_df(test_mol_df, config)

        # Precursor m/z should be computed and present
        assert spec_df["prec_mz"].notna().all()
        assert all(spec_df["prec_mz"] > 0)

    def test_spec_df_merged_spectra_mode(self, test_mol_df):
        """Test spec_df with merged spectra mode."""
        spec_df_module = _load_script_module("02_inf_prepare_spec_df.py")

        config = spec_df_module.get_config(
            prec_types=["[M+H]+", "[M-H]-"],
            qtof_ces=[],
            ft_ces=["25.0NCE", "35.0NCE"],
            inst_types=["FT"],
            output_dir="/tmp",
            merged_spectra=True,
        )

        spec_df = spec_df_module.create_spec_df(test_mol_df, config)

        # Group IDs should be grouped by prec_type and inst_type
        groups = spec_df.groupby(["prec_type", "inst_type"])
        for (prec, inst), group in groups:
            unique_group_ids = group["group_id"].unique()
            # All entries with same (prec, inst) should have same group_id
            assert len(unique_group_ids) == 1

    def test_collision_energy_parsing(self):
        """Test parsing of various collision energy formats."""
        spec_df_module = _load_script_module("02_inf_prepare_spec_df.py")

        # Test different NCE formats
        config = spec_df_module.get_config(
            prec_types=["[M+H]+"],
            qtof_ces=[],
            ft_ces=["25.0NCE", "25.0%", "30.0eV"],  # Different formats
            inst_types=["FT"],
            output_dir="/tmp",
            merged_spectra=False,
        )

        # Should parse all formats successfully
        assert len(config["combinations"]) == 3

    def test_validate_inputs_errors(self):
        """Test that validate_inputs raises errors for invalid inputs."""
        spec_df_module = _load_script_module("02_inf_prepare_spec_df.py")

        # Empty prec_types should raise
        with pytest.raises(ValueError):
            spec_df_module.validate_inputs([], [], [], [])

        # Empty inst_types should raise
        with pytest.raises(ValueError):
            spec_df_module.validate_inputs(["[M+H]+"], [], [], [])
