"""Unit tests for preproc_scripts/01_prepare_df.py module.

Tests validate:
- MSP parsing and column renaming
- NIST EI CSV processing
- Comment field parsing
- Data dictionary validation
"""

import importlib.util
import tempfile
from pathlib import Path

import pandas as pd


def load_prepare_module():
    """Load the preprocessing script as a module via importlib."""
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "preproc_scripts" / "01_prepare_df.py"
    spec = importlib.util.spec_from_file_location("prepare_df", str(script_path))
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Unable to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_info_from_comments():
    """Test extraction of info from MSP comments field."""
    module = load_prepare_module()

    comments = 'computed SMILES="CC(C)O" MoNA Rating="10" fragmentation mode="CID"'
    # Function includes quotes in extraction; verify content is present
    extracted = module.extract_info_from_comments(comments, "computed SMILES")
    assert extracted is not None and "CC(C)O" in extracted
    assert module.extract_info_from_comments(comments, "nonexistent") is None


def test_preproc_msp_basic():
    """Test basic MSP parsing."""
    module = load_prepare_module()

    msp_content = """ID: 1
Name: Test Compound
PrecursorMZ: 100.5
Num peaks: 2
50 100
75 200

ID: 2
Name: Test Compound 2
PrecursorMZ: 150.5
Num peaks: 1
100 500

"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".msp", delete=False) as f:
        f.write(msp_content)
        msp_fp = f.name

    try:
        keys = {"ID", "Name", "PrecursorMZ", "Num peaks", "MS"}
        df = module.preproc_msp(msp_fp, keys, num_entries=-1)

        assert len(df) == 2, "Should have 2 spectra"
        assert df.iloc[0]["ID"] == "1"
        assert df.iloc[0]["Name"] == "Test Compound"
        assert df.iloc[0]["PrecursorMZ"] == "100.5"
        assert df.iloc[0]["Num peaks"] == 2
        assert "50 100" in df.iloc[0]["MS"]
        assert "75 200" in df.iloc[0]["MS"]
    finally:
        Path(msp_fp).unlink()


def test_preproc_msp_num_entries_limit():
    """Test MSP parsing with entry limit."""
    module = load_prepare_module()

    msp_content = """ID: 1
Name: Compound 1
Num peaks: 1
50 100

ID: 2
Name: Compound 2
Num peaks: 1
60 200

ID: 3
Name: Compound 3
Num peaks: 1
70 300

"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".msp", delete=False) as f:
        f.write(msp_content)
        msp_fp = f.name

    try:
        keys = {"ID", "Name", "Num peaks", "MS"}
        df = module.preproc_msp(msp_fp, keys, num_entries=2)
        assert len(df) == 2, "Should limit to 2 entries"
    finally:
        Path(msp_fp).unlink()


def test_merge_and_check_with_smiles():
    """Test merge_and_check when mol_df is None (SMILES in MSP)."""
    module = load_prepare_module()

    msp_df = pd.DataFrame(
        {
            "Name": ["Compound1", "Compound2"],
            "smiles": ["CC(C)O", "c1ccccc1"],
            "spec_id": [None, None],
            "extra_col": [1, 2],
        }
    )
    rename_dict = {"Name": "name", "smiles": "smiles", "spec_id": "spec_id"}

    spec_df = module.merge_and_check(msp_df, None, rename_dict)

    assert len(spec_df) == 2
    assert list(spec_df.columns) == ["name", "smiles", "spec_id", "dset_spec_id"]
    assert spec_df["spec_id"].tolist() == [0, 1]
    assert spec_df["dset_spec_id"].tolist() == [0, 1]


def test_nist_ei_constants():
    """Test NIST EI constants are set correctly."""
    module = load_prepare_module()

    assert module.NIST_EI_COLLISION_ENERGY == "70.0 eV"
    assert module.NIST_EI_PRECURSOR_TYPE == "[M]+"
    assert module.NIST_EI_SPEC_TYPE == "MS1"
    assert module.NIST_EI_INST_TYPE == "GC-MS"
    assert module.NIST_EI_ION_MODE == "P"
    assert module.NIST_EI_DATASET_NAME == "nist_ei"


def test_process_nist_ei_csv_basic():
    """Test NIST EI CSV processing."""
    module = load_prepare_module()

    csv_content = """name,peaks,exact_mass,formula,smiles,inchikey
Compound1,"50 100
75 200",100.0,C2H4O,CC(C)O,LFQSCWFLJHTTHZ-UHFFFAOYSA-N
Compound2,"60 300
80 400",150.0,C3H6O2,c1ccccc1,UHOVQNZJYSORNB-UHFFFAOYSA-N
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        csv_fp = f.name

    try:
        spec_df = module.process_nist_ei_csv(csv_fp)

        assert len(spec_df) == 2
        assert spec_df.iloc[0]["frag_mode"] == "EI"
        assert spec_df.iloc[0]["ion_type"] == "EI"
        assert spec_df.iloc[0]["col_energy"] == "70.0 eV"
        assert spec_df.iloc[0]["prec_type"] == "[M]+"
        assert spec_df.iloc[0]["spec_type"] == "MS1"
        assert spec_df.iloc[0]["inst_type"] == "GC-MS"
        assert spec_df.iloc[0]["ion_mode"] == "P"
        assert spec_df.iloc[0]["dset"] == "nist_ei"
        assert spec_df.iloc[0]["spec_id"] == 0
    finally:
        Path(csv_fp).unlink()


def test_msp_key_dict_has_required_keys():
    """Verify MSP key dictionary contains expected keys."""
    module = load_prepare_module()

    required_keys = ["Precursor_type", "PrecursorMZ", "Num peaks", "MS", "Name", "SMILES"]
    for key in required_keys:
        assert key in module.MSP_KEY_DICT, f"Missing key: {key}"


def test_pcdl_keys_dict_has_required_keys():
    """Verify PCDL key dictionary contains expected keys."""
    module = load_prepare_module()

    required_keys = [
        "ACCESSION:",
        "CH$SMILES:",
        "CH$FORMULA:",
        "MS$FOCUSED_ION: PRECURSOR_M/Z",
        "PK$NUM_PEAK:",
    ]
    for key in required_keys:
        assert key in module.PCDL_KEYS, f"Missing key: {key}"
