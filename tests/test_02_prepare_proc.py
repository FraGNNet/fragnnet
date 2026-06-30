"""Unit tests for preproc_scripts/02_prepare_proc.py module.

Tests validate:
- group_id construction uses documented key: mol_id + prec_type + inst_type
- group_id is identical across datasets and CE patterns for the same trio
- group_id differs when the documented trio changes
"""

import importlib.util
from pathlib import Path

import pandas as pd


def load_prepare_module():
    """Load the preprocessing script as a module via importlib."""
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "preproc_scripts" / "02_prepare_proc.py"
    spec = importlib.util.spec_from_file_location("prepare_proc", str(script_path))
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Unable to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _setup_stubs(monkeypatch, module):
    """Common stub setup for all tests."""
    monkeypatch.setattr(module.data_utils, "mol_from_smiles", lambda s, **kwargs: f"mol({s})")
    monkeypatch.setattr(
        module.data_utils, "mol_to_smiles", lambda m: str(m).removeprefix("mol(").removesuffix(")")
    )
    monkeypatch.setattr(module.data_utils, "parse_peaks_str", lambda s: [(100.0, 1.0)])
    monkeypatch.setattr(module.data_utils, "get_res", lambda peaks: None)
    monkeypatch.setattr(module.data_utils, "parse_inst_info", lambda row: ("LCMS", "HCD"))
    monkeypatch.setattr(module.data_utils, "parse_ace_str", lambda s: None)
    monkeypatch.setattr(module.data_utils, "parse_nce_str", lambda s: None)
    monkeypatch.setattr(module.data_utils, "parse_prec_type_str", lambda s: s)
    monkeypatch.setattr(module.data_utils, "parse_ion_mode_str", lambda s: s)
    monkeypatch.setattr(module.data_utils, "convert_peaks_to_float", lambda peaks: peaks)
    monkeypatch.setattr(module.data_utils, "parse_ri_str", lambda s: None)
    monkeypatch.setattr(module.data_utils, "mol_to_inchikey_s", lambda m: "AAAAAAAAAAAAAA")
    monkeypatch.setattr(module.data_utils, "get_murcko_scaffold", lambda m: "CC")
    monkeypatch.setattr(module.data_utils, "mol_to_formula", lambda m: "C2H6")
    monkeypatch.setattr(module.data_utils, "mol_to_inchi", lambda m: "InChI=1S/C2H6/h1H3,2H3")
    monkeypatch.setattr(
        module.data_utils, "mol_to_mol_weight", lambda m, exact: 30.0 if exact else 30.1
    )
    monkeypatch.setattr(module.data_utils, "mol_to_num_atoms", lambda m: 8)
    monkeypatch.setattr(module.data_utils, "mol_to_num_bonds", lambda m: 7)
    monkeypatch.setattr(module.data_utils, "mol_to_charge", lambda m: 0)
    monkeypatch.setattr(module.data_utils, "check_single_mol", lambda m: True)
    monkeypatch.setattr(module.data_utils, "mol_to_num_radicals", lambda m: 0)
    monkeypatch.setattr(module.data_utils, "infer_prec_mz", lambda row: 31.0)
    monkeypatch.setattr(module.data_utils, "parse_annotations", lambda row: ([], [], [], [], []))


def test_group_id_uses_documented_trio(monkeypatch):
    """Verify that group_id uses only mol_id + prec_type + inst_type.

    Two rows with same molecule/adduct/instrument but different datasets and CE
    patterns should share the same group_id, validating that dset and ce_type
    are excluded from grouping as documented.
    """
    module = load_prepare_module()
    _setup_stubs(monkeypatch, module)

    # Same smiles, prec_type, instrument; different datasets and CE patterns
    rows = [
        {
            "spec_id": 0,
            "smiles": "CC",
            "dset": "A",
            "dset_spec_id": "1",
            "col_energy": "10 eV",
            "col_energy_extra_1": None,
            "col_energy_extra_2": None,
            "peaks": "[(100,1.0)]",
            "prec_type": "M+H",
            "spec_type": "MS2",
            "ion_mode": "P",
            "col_gas": "N2",
            "ri": "",
            "formula": "",
            "inchikey": None,
            "exact_mass": None,
            "prec_mz": None,
            "notes": "",
        },
        {
            "spec_id": 1,
            "smiles": "CC",
            "dset": "B",
            "dset_spec_id": "2",
            "col_energy": "15 eV",
            "col_energy_extra_1": "20 eV",
            "col_energy_extra_2": None,
            "peaks": "[(100,1.0)]",
            "prec_type": "M+H",
            "spec_type": "MS2",
            "ion_mode": "P",
            "col_gas": "N2",
            "ri": "",
            "formula": "",
            "inchikey": None,
            "exact_mass": None,
            "prec_mz": None,
            "notes": "",
        },
    ]
    df = pd.DataFrame(rows)

    spec_df, mol_df, ann_df, _ = module.preprocess_spec(df)

    # Both spectra should map to the same group_id
    group_ids = spec_df["group_id"].unique()
    assert len(group_ids) == 1, (
        "Expected identical group_id across datasets/CE patterns for same trio"
    )

    # Verify grouping key values are identical for both rows
    trio_cols = ["mol_id", "prec_type", "inst_type"]
    assert spec_df[trio_cols].nunique().sum() == 3, "Grouping trio should match across rows"


def test_group_id_differs_when_precursor_changes(monkeypatch):
    """Verify that group_id differs when precursor type (part of trio) changes.

    Two rows with same molecule and instrument but different precursor types
    should have different group_ids, confirming that the trio is used correctly.
    """
    module = load_prepare_module()
    _setup_stubs(monkeypatch, module)

    # Same smiles and instrument; different precursor type
    rows = [
        {
            "spec_id": 0,
            "smiles": "CC",
            "dset": "A",
            "dset_spec_id": "1",
            "col_energy": "10 eV",
            "col_energy_extra_1": None,
            "col_energy_extra_2": None,
            "peaks": "[(100,1.0)]",
            "prec_type": "M+H",
            "spec_type": "MS2",
            "ion_mode": "P",
            "col_gas": "N2",
            "ri": "",
            "formula": "",
            "inchikey": None,
            "exact_mass": None,
            "prec_mz": None,
            "notes": "",
        },
        {
            "spec_id": 1,
            "smiles": "CC",
            "dset": "B",
            "dset_spec_id": "2",
            "col_energy": "15 eV",
            "col_energy_extra_1": None,
            "col_energy_extra_2": None,
            "peaks": "[(100,1.0)]",
            "prec_type": "M+Na",
            "spec_type": "MS2",
            "ion_mode": "P",
            "col_gas": "N2",
            "ri": "",
            "formula": "",
            "inchikey": None,
            "exact_mass": None,
            "prec_mz": None,
            "notes": "",
        },
    ]
    df = pd.DataFrame(rows)

    spec_df, mol_df, ann_df, _ = module.preprocess_spec(df)

    # Different precursor type should yield different group_ids
    group_ids = spec_df["group_id"].unique()
    assert len(group_ids) == 2, "Expected different group_id when precursor type differs"

    # mol_id and inst_type match, prec_type differs
    assert spec_df[["mol_id", "inst_type"]].nunique().sum() == 2
    assert spec_df[["prec_type"]].nunique().iloc[0] == 2
