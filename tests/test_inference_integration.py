"""
Integration tests for inference preprocessing pipeline.
"""

import importlib.util
import json
from pathlib import Path


# Helper to import modules with numeric names
def _load_script_module(script_name):
    """Load a preprocessing script as a module."""
    script_path = Path(__file__).parent.parent / "preproc_scripts" / "inference" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestInferenceIntegration:
    """Integration tests across multiple preprocessing stages."""

    def test_mol_df_to_spec_df_pipeline(self):
        """Test full mol_df to spec_df pipeline."""
        mol_df_module = _load_script_module("01_inf_prepare_mol_df.py")
        spec_df_module = _load_script_module("02_inf_prepare_spec_df.py")

        # Create mol_df
        smiles_list = ["CCO", "CC(=O)O", "CCN"]
        mol_df = mol_df_module.create_mol_df(smiles_list, ids_list=None, max_heavy_atoms=128)

        # Create spec_df from mol_df
        config = spec_df_module.get_config(
            prec_types=["[M+H]+"],
            qtof_ces=[],
            ft_ces=["25.0NCE"],
            inst_types=["FT"],
            output_dir="/tmp",
            merged_spectra=False,
        )

        spec_df = spec_df_module.create_spec_df(mol_df, config)

        # Verify integration
        assert len(mol_df) > 0
        assert len(spec_df) == len(mol_df) * len(config["combinations"])
        assert spec_df["mol_id"].isin(mol_df["mol_id"]).all()

    def test_config_json_generation(self):
        """Test that config JSON can be properly serialized."""
        spec_df_module = _load_script_module("02_inf_prepare_spec_df.py")

        config = spec_df_module.get_config(
            prec_types=["[M+H]+", "[M-H]-"],
            qtof_ces=["10.0eV"],
            ft_ces=["25.0NCE"],
            inst_types=["FT", "QTOF"],
            output_dir="/tmp/test",
            merged_spectra=True,
        )

        # Should be JSON serializable
        config_json = json.dumps(config)
        config_reloaded = json.loads(config_json)

        assert len(config_reloaded["combinations"]) == len(config["combinations"])
        assert config_reloaded["input_parameters"]["merged_spectra"] == True

    def test_json_mode_end_to_end(self):
        """Test full JSON input mode from candidates to spectrum."""
        mol_df_module = _load_script_module("01_inf_prepare_mol_df.py")
        spec_df_module = _load_script_module("02_inf_prepare_spec_df.py")

        # Test data
        candidates = {
            "CCO": ["CC", "C", "CCN"],
            "CC(=O)O": ["CCO", "CC", "C"],
        }

        # Extract unique SMILES
        unique_smiles = set()
        for query, cands in candidates.items():
            unique_smiles.add(query)
            unique_smiles.update(cands)

        # Create mol_df
        mol_df = mol_df_module.create_mol_df(
            list(unique_smiles), ids_list=None, max_heavy_atoms=128
        )

        # Create spec_df
        config = spec_df_module.get_config(
            prec_types=["[M+H]+"],
            qtof_ces=[],
            ft_ces=["25.0NCE"],
            inst_types=["FT"],
            output_dir="/tmp",
            merged_spectra=False,
        )

        spec_df = spec_df_module.create_spec_df(mol_df, config)

        # Verify end-to-end
        assert len(mol_df) >= len(unique_smiles)
        assert len(spec_df) > 0
        assert "prec_mz" in spec_df.columns
