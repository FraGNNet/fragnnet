"""Unit tests for preproc_scripts/07_prepare_classyfire.py.

Tests validate:
- TSV loading from a zip archive
- SMILES canonicalization
- Semicolon list parsing
- mol_df / ClassyFire merge (happy path, missing molecules, duplicates)
- Output schema and column types
"""

import importlib.util
import zipfile
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------


def load_module():
    """Load preproc_scripts/07_prepare_classyfire.py as a Python module."""
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "preproc_scripts" / "07_prepare_classyfire.py"
    spec = importlib.util.spec_from_file_location("prepare_classyfire", str(script_path))
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Unable to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    """Return the loaded prepare_classyfire module."""
    return load_module()


# ---------------------------------------------------------------------------
# Fixtures: minimal synthetic data
# ---------------------------------------------------------------------------

CF_TSV_CONTENT = (
    "sid\tsmiles\tkingdom\tsuperklass\tklass\tsubklass\tdirect_parent\t"
    "geometric_descriptor\talternative_parents\tsubstituents\n"
    # caffeine (canonical: Cn1cnc2c1c(=O)n(c(=O)n2C)C)
    "1\tCn1cnc2c1c(=O)n(c(=O)n2C)C\torganic compounds\torganoheterocyclic compounds\t"
    "purines and purine derivatives\txanthines\tcaffeine\taromatic heteropolycyclic compounds\t"
    "azacyclic compounds;purines\theteroaromatic compound;xanthine\n"
    # aspirin
    "2\tCC(=O)Oc1ccccc1C(=O)O\torganic compounds\tbenzenoids\t"
    "benzene and substituted derivatives\tbenzoic acids and derivatives\taspirin\t"
    "aromatic monocyclic compounds\tcarboxylic acids;esters\tbenzenoid;ester\n"
    # a duplicate SMILES for caffeine (should be deduplicated)
    "3\tCn1cnc2c1c(=O)n(c(=O)n2C)C\torganic compounds\torganoheterocyclic compounds\t"
    "purines and purine derivatives\txanthines\tcaffeine\taromatic heteropolycyclic compounds\t"
    "azacyclic compounds\theteroaromatic compound\n"
)


def make_cf_zip(content: str, tmp_path: Path) -> str:
    """Write content as a TSV inside a zip file and return the path."""
    zip_path = tmp_path / "test_cf.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("classyfire.tsv", content)
    return str(zip_path)


def make_mol_df() -> pd.DataFrame:
    """Create a minimal synthetic mol_df."""
    return pd.DataFrame(
        {
            "mol_id": [101, 102, 103],
            # caffeine, aspirin, unknown molecule not in CF
            "smiles": [
                "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
                "CC(=O)Oc1ccccc1C(=O)O",
                "CCO",  # ethanol - not in ClassyFire fixture
            ],
            "inchikey_s": [
                "RYYVLZVUVIJVGH-UHFFFAOYSA-N",
                "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
                "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            ],
        }
    )


# ---------------------------------------------------------------------------
# Tests: load_classyfire_tsv
# ---------------------------------------------------------------------------


def test_load_classyfire_tsv_basic(mod, tmp_path):
    """TSV is loaded correctly from a valid zip archive."""
    zip_path = make_cf_zip(CF_TSV_CONTENT, tmp_path)
    df = mod.load_classyfire_tsv(zip_path)
    assert len(df) == 3
    assert list(df.columns) == [
        "sid",
        "smiles",
        "kingdom",
        "superklass",
        "klass",
        "subklass",
        "direct_parent",
        "geometric_descriptor",
        "alternative_parents",
        "substituents",
    ]
    assert df["klass"].iloc[0] == "purines and purine derivatives"


def test_load_classyfire_tsv_no_tsv(mod, tmp_path):
    """Raises FileNotFoundError when zip contains no TSV."""
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("readme.txt", "nothing here")
    with pytest.raises(FileNotFoundError):
        mod.load_classyfire_tsv(str(zip_path))


def test_load_classyfire_tsv_multiple_tsv(mod, tmp_path):
    """Raises ValueError when zip contains more than one TSV."""
    zip_path = tmp_path / "multi.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("a.tsv", "col\nval")
        zf.writestr("b.tsv", "col\nval")
    with pytest.raises(ValueError):
        mod.load_classyfire_tsv(str(zip_path))


# ---------------------------------------------------------------------------
# Tests: parse_semicolon_list
# ---------------------------------------------------------------------------


def test_parse_semicolon_list_normal(mod):
    """Semicolon-separated string is split and stripped correctly."""
    result = mod.parse_semicolon_list("azacyclic compounds;purines;xanthines")
    assert result == ["azacyclic compounds", "purines", "xanthines"]


def test_parse_semicolon_list_single(mod):
    """Single item (no semicolons) returns a one-element list."""
    result = mod.parse_semicolon_list("benzenoids")
    assert result == ["benzenoids"]


def test_parse_semicolon_list_empty_string(mod):
    """Empty string returns empty list."""
    assert mod.parse_semicolon_list("") == []


def test_parse_semicolon_list_nan(mod):
    """NaN / None returns empty list without raising."""
    assert mod.parse_semicolon_list(float("nan")) == []
    assert mod.parse_semicolon_list(None) == []


def test_parse_semicolon_list_whitespace(mod):
    """Leading/trailing whitespace around items is stripped."""
    result = mod.parse_semicolon_list(" a ; b ; c ")
    assert result == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Tests: canonicalize_smiles_series
# ---------------------------------------------------------------------------


def test_canonicalize_smiles_series_known(mod):
    """Caffeine SMILES in non-canonical form is canonicalized consistently."""
    series = pd.Series(["Cn1cnc2c1c(=O)n(c(=O)n2C)C", "CC(=O)Oc1ccccc1C(=O)O"])
    result = mod.canonicalize_smiles_series(series)
    # Both should be non-null
    assert result.notna().all()
    # Re-canonicalizing should be idempotent
    result2 = mod.canonicalize_smiles_series(result)
    assert (result == result2).all()


def test_canonicalize_smiles_series_invalid(mod):
    """Invalid SMILES returns None without raising."""
    series = pd.Series(["not_a_smiles", "CC(=O)O"])
    result = mod.canonicalize_smiles_series(series)
    assert result.iloc[0] is None
    assert result.iloc[1] is not None


def test_canonicalize_smiles_series_empty(mod):
    """Empty string returns None."""
    result = mod.canonicalize_smiles_series(pd.Series([""]))
    assert result.iloc[0] is None


# ---------------------------------------------------------------------------
# Tests: build_mol_classyfire_df
# ---------------------------------------------------------------------------


@pytest.fixture
def cf_df_raw(mod, tmp_path):
    """Load the synthetic ClassyFire DataFrame."""
    zip_path = make_cf_zip(CF_TSV_CONTENT, tmp_path)
    return mod.load_classyfire_tsv(zip_path)


def test_build_basic_merge(mod, cf_df_raw):
    """Caffeine and aspirin are merged; ethanol (not in CF) is excluded."""
    mol_df = make_mol_df()
    result = mod.build_mol_classyfire_df(mol_df, cf_df_raw)

    assert len(result) == 2  # caffeine + aspirin; ethanol excluded
    mol_ids = set(result["mol_id"].tolist())
    assert mol_ids == {101, 102}


def test_build_output_columns(mod, cf_df_raw):
    """Output has exactly the expected columns."""
    mol_df = make_mol_df()
    result = mod.build_mol_classyfire_df(mol_df, cf_df_raw)
    assert list(result.columns) == [
        "mol_id",
        "smiles",
        "inchikey",
        "kingdom",
        "superklass",
        "klass",
        "subklass",
        "direct_parent",
        "alternative_parents",
        "substituents",
    ]


def test_build_list_columns(mod, cf_df_raw):
    """alternative_parents and substituents are Python lists."""
    mol_df = make_mol_df()
    result = mod.build_mol_classyfire_df(mol_df, cf_df_raw)
    for col in ("alternative_parents", "substituents"):
        for val in result[col]:
            assert isinstance(val, list), f"{col} should be a list, got {type(val)}"


def test_build_caffeine_values(mod, cf_df_raw):
    """Caffeine row has correct taxonomy values."""
    mol_df = make_mol_df()
    result = mod.build_mol_classyfire_df(mol_df, cf_df_raw)
    row = result[result["mol_id"] == 101].iloc[0]
    assert row["kingdom"] == "organic compounds"
    assert row["superklass"] == "organoheterocyclic compounds"
    assert row["klass"] == "purines and purine derivatives"
    assert row["direct_parent"] == "caffeine"
    assert "purines" in row["alternative_parents"]


def test_build_deduplication(mod, cf_df_raw):
    """Duplicate SMILES in ClassyFire data does not produce duplicate mol rows."""
    mol_df = make_mol_df()
    result = mod.build_mol_classyfire_df(mol_df, cf_df_raw)
    # mol_id=101 (caffeine) should appear exactly once despite duplicate in CF fixture
    assert (result["mol_id"] == 101).sum() == 1


def test_build_all_missing(mod):
    """Returns empty DataFrame when no molecules match ClassyFire."""
    mol_df = pd.DataFrame(
        {
            "mol_id": [999],
            "smiles": ["CCO"],
            "inchikey_s": ["LFQSCWFLJHTTHZ-UHFFFAOYSA-N"],
        }
    )
    # ClassyFire df with a completely different molecule
    cf_df = pd.DataFrame(
        {
            "smiles": ["c1ccccc1"],  # benzene
            "kingdom": ["organic compounds"],
            "superklass": ["benzenoids"],
            "klass": ["benzene"],
            "subklass": [""],
            "direct_parent": ["benzene"],
            "alternative_parents": [""],
            "substituents": [""],
        }
    )
    result = mod.build_mol_classyfire_df(mol_df, cf_df)
    assert len(result) == 0
    assert list(result.columns) == [
        "mol_id",
        "smiles",
        "inchikey",
        "kingdom",
        "superklass",
        "klass",
        "subklass",
        "direct_parent",
        "alternative_parents",
        "substituents",
    ]


def test_build_reset_index(mod, cf_df_raw):
    """Output index is contiguous 0..N-1."""
    mol_df = make_mol_df()
    result = mod.build_mol_classyfire_df(mol_df, cf_df_raw)
    assert list(result.index) == list(range(len(result)))
