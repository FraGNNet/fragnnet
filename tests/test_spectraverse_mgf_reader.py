"""Unit tests for the Spectraverse MGF reader in preproc_scripts/01_prepare_df.py.

Covers the v1.0.2 format (single FOLD field, no INDEX), the legacy INDEX/FOLD_* format and
fold-less MGFs (v1.0.0/1.0.1), plus the exact-key matching of header lines.
"""

import importlib.util
import logging
from pathlib import Path

import numpy as np
import pytest


def load_prepare_module():
    """Load the preprocessing script as a module via importlib."""
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "preproc_scripts" / "01_prepare_df.py"
    spec = importlib.util.spec_from_file_location("prepare_df_spv", str(script_path))
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Unable to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(title_id: int, extra: list[str], smiles: str = "CCO") -> str:
    """Build one MGF record with the common fields plus `extra` header lines."""
    lines = [
        "BEGIN IONS",
        f"TITLE=SPECTRAVERSE{title_id:010d}",
        "SOURCE=MSnLib",
        f"SMILES={smiles} |t:1|",
        f"INCHI=InChI=1S/test{title_id}",
        f"INCHIKEY=KEY{title_id}-UHFFFAOYSA-N",
        "IONMODE=positive",
        "ADDUCT=[M+H]+",
        "NUM_PEAKS=2",
        "PRECURSOR_MZ=47.0491",
        "PARENT_MASS=46.0419",
        "MS_LEVEL=MS2",
        "INSTRUMENT_TYPE=orbitrap",
        "COLLISION_ENERGY_1=20.5",
        "COLLISION_ENERGY_2=nan",
        *extra,
        "41.003009 0.5",
        "44.282551 1.0",
        "END IONS",
        "",
    ]
    return "\n".join(lines)


@pytest.fixture
def mgf_v102(tmp_path: Path) -> Path:
    """v1.0.2-style MGF: one FOLD field, no INDEX, non-contiguous titles."""
    fp = tmp_path / "v102.mgf"
    fp.write_text(
        _record(1, ["FOLD=6"]) + _record(7, ["FOLD=10"]) + _record(3, ["FOLD=1"], smiles="CCC")
    )
    return fp


@pytest.fixture
def mgf_legacy(tmp_path: Path) -> Path:
    """Legacy-style MGF: INDEX + the four FOLD_* fields."""
    fp = tmp_path / "legacy.mgf"
    recs = []
    for i, idx in enumerate([10, 11]):
        recs.append(
            _record(
                i + 1,
                [
                    f"INDEX={idx}",
                    f"FOLD_INCHI={i + 1}",
                    f"FOLD_FORM={i + 2}",
                    f"FOLD_MCES={i + 3}",
                    f"FOLD_MCESFORM={i + 4}",
                ],
            )
        )
    fp.write_text("".join(recs))
    return fp


@pytest.fixture
def mgf_foldless(tmp_path: Path) -> Path:
    """v1.0.0/1.0.1-style MGF: neither INDEX nor any fold field."""
    fp = tmp_path / "foldless.mgf"
    fp.write_text(_record(1, []) + _record(2, []))
    return fp


def test_v102_parses_fold_and_ids(mgf_v102: Path):
    """FOLD becomes an int `fold` column; spec_id is record order; dset_spec_id from TITLE."""
    module = load_prepare_module()
    spec_df, split_df = module.process_specverse(str(mgf_v102))

    assert spec_df["spec_id"].tolist() == [0, 1, 2]
    assert spec_df["dset_spec_id"].tolist() == [1, 7, 3]
    assert "fold" not in spec_df.columns
    assert split_df is not None
    assert list(split_df.columns) == ["spec_id", "dset_spec_id", "fold"]
    assert split_df["dset_spec_id"].tolist() == [1, 7, 3]
    assert split_df["fold"].tolist() == [6, 10, 1]
    assert split_df["fold"].dtype.kind == "i"
    assert split_df["spec_id"].tolist() == [0, 1, 2]


def test_v102_inchi_is_not_inchikey(mgf_v102: Path):
    """`INCHI` must not match `INCHIKEY=` lines, whatever their order in the record."""
    module = load_prepare_module()
    spec_df, _ = module.process_specverse(str(mgf_v102))

    assert spec_df["inchi"].tolist() == [f"InChI=1S/test{i}" for i in (1, 7, 3)]
    assert spec_df["inchikey"].tolist() == [f"KEY{i}-UHFFFAOYSA-N" for i in (1, 7, 3)]


def test_v102_inchikey_before_inchi(tmp_path: Path):
    """The fix does not depend on the line order of INCHI/INCHIKEY."""
    module = load_prepare_module()
    fp = tmp_path / "order.mgf"
    fp.write_text(
        "BEGIN IONS\nTITLE=SPECTRAVERSE0000000001\nINCHIKEY=AAA-BBB-N\nINCHI=InChI=1S/x\n"
        "ADDUCT=[M+H]+\nFOLD=2\n41.0 1.0\nEND IONS\n"
    )
    spec_df, _ = module.process_specverse(str(fp))
    assert spec_df.loc[0, "inchi"] == "InChI=1S/x"
    assert spec_df.loc[0, "inchikey"] == "AAA-BBB-N"


def test_ion_mode_from_adduct_charge_sign(tmp_path: Path):
    """Negative adducts that contain a "+" ([M+Cl]-, [M+HCOOH-H]-) are N, not P."""
    module = load_prepare_module()
    fp = tmp_path / "ion_mode.mgf"
    adducts = ["[M+H]+", "[M-H]-", "[M+Cl]-", "[M+HCOOH-H]-", "[M]+"]
    fp.write_text(
        "".join(
            f"BEGIN IONS\nTITLE=SPECTRAVERSE{i + 1:010d}\nADDUCT={a}\nFOLD=1\n41.0 1.0\nEND IONS\n"
            for i, a in enumerate(adducts)
        )
    )
    spec_df, _ = module.process_specverse(str(fp))
    assert spec_df["ion_mode"].tolist() == ["P", "N", "N", "N", "P"]


def test_v102_field_values(mgf_v102: Path):
    """Typed fields: floats, eV energies, NaN energies, smiles trimmed, peaks text."""
    module = load_prepare_module()
    spec_df, _ = module.process_specverse(str(mgf_v102))

    row = spec_df.iloc[0]
    assert row["smiles"] == "CCO"
    assert row["prec_mz"] == pytest.approx(47.0491)
    assert row["exact_mass"] == pytest.approx(46.0419)
    assert row["prec_type"] == "[M+H]+"
    assert row["ion_mode"] == "P"
    assert row["col_energy"] == "20.5 eV"
    assert np.isnan(row["col_energy_extra_1"])
    assert row["num_peaks"] == 2
    assert row["peaks"] == "41.003009 0.5\n44.282551 1.0\n"
    assert row["dset"] == "spectraverse"
    assert spec_df["smiles"].iloc[2] == "CCC"


def test_legacy_format_still_parses(mgf_legacy: Path):
    """INDEX is kept as spec_id and the four legacy fold columns go to the split_df."""
    module = load_prepare_module()
    spec_df, split_df = module.process_specverse(str(mgf_legacy))

    assert [str(x) for x in spec_df["spec_id"]] == ["10", "11"]
    assert split_df is not None
    assert list(split_df.columns) == (
        ["spec_id", "dset_spec_id"] + module.SPECTRAVERSE_LEGACY_FOLD_COLS
    )
    assert split_df["fold_inchi"].tolist() == [1, 2]
    assert split_df["fold_form"].tolist() == [2, 3]
    assert split_df["fold_mces"].tolist() == [3, 4]
    assert split_df["fold_mcesform"].tolist() == [4, 5]
    assert not any(c in spec_df.columns for c in module.SPECTRAVERSE_LEGACY_FOLD_COLS)
    assert spec_df["dset_spec_id"].tolist() == [1, 2]


def test_foldless_mgf_has_no_fold(mgf_foldless: Path, caplog: pytest.LogCaptureFixture):
    """No fold field: no split_df, a warning, and spec_id from record order."""
    module = load_prepare_module()
    with caplog.at_level(logging.WARNING):
        spec_df, split_df = module.process_specverse(str(mgf_foldless))

    assert split_df is None
    assert "fold" not in spec_df.columns
    assert spec_df["spec_id"].tolist() == [0, 1]
    assert any("no fold field" in rec.message for rec in caplog.records)


def test_empty_fold_raises(tmp_path: Path):
    """An empty FOLD value is an error, not a silent NaN."""
    module = load_prepare_module()
    fp = tmp_path / "bad.mgf"
    fp.write_text(_record(1, ["FOLD="]))
    with pytest.raises(ValueError, match="fold"):
        module.process_specverse(str(fp))


def test_partial_fold_raises(tmp_path: Path):
    """FOLD present for only some records is an error."""
    module = load_prepare_module()
    fp = tmp_path / "partial.mgf"
    fp.write_text(_record(1, ["FOLD=3"]) + _record(2, []))
    with pytest.raises(ValueError, match="fold"):
        module.process_specverse(str(fp))


def test_empty_mgf_raises(tmp_path: Path):
    """A file without records is an error."""
    module = load_prepare_module()
    fp = tmp_path / "empty.mgf"
    fp.write_text("")
    with pytest.raises(ValueError, match="No spectra"):
        module.process_specverse(str(fp))


def _one(extra: list[str], **replace: str) -> str:
    """One v1.0.2-style record, with `replace` mapping an exact line to its substitute."""
    text = _record(1, ["FOLD=3", *extra])
    for old, new in replace.items():
        assert old in text, old
        text = text.replace(old, new)
    return text


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (
            _record(1, ["FOLD=3"]) + "BEGIN IONS\nTITLE=SPECTRAVERSE0000000002\n41.0 1.0\n",
            "no END IONS",
        ),
        (_one([], **{"44.282551 1.0": "-44.28 1.0"}), "unparsable line"),
        (_one([], **{"44.282551 1.0\n": ""}), "NUM_PEAKS=2 but 1 peak"),
        (_one([], **{"41.003009 0.5\n44.282551 1.0\n": ""}), "NUM_PEAKS=2 but 0 peak"),
        (_one(["FOLD=4"]), "repeats FOLD"),
        (_one([], **{"FOLD=3": "FOLD=0"}), "outside 1..10"),
        (_one([], **{"FOLD=3": "FOLD=11"}), "outside 1..10"),
        (_one([], **{"ADDUCT=[M+H]+\n": ""}), "ADDUCT"),
        (
            _one([], **{"TITLE=SPECTRAVERSE0000000001": "TITLE=FOOBARBAZQUX0000000007"}),
            "is not SPECTRAVERSE<int>",
        ),
        (
            _one([], **{"TITLE=SPECTRAVERSE0000000001": "TITLE=SPECTRAVERSEabc"}),
            "is not SPECTRAVERSE<int>",
        ),
        ("SMILES=CCO\n" + _record(1, ["FOLD=3"]), "outside BEGIN/END"),
        (_record(1, ["FOLD=3"]).replace("END IONS", "BEGIN IONS"), "inside record"),
    ],
)
def test_malformed_records_raise(tmp_path: Path, text: str, match: str):
    """Truncation, bad peaks, peak-count mismatch, bad FOLD/TITLE never pass silently."""
    module = load_prepare_module()
    fp = tmp_path / "bad.mgf"
    fp.write_text(text)
    with pytest.raises(ValueError, match=match):
        module.process_specverse(str(fp))


def test_zero_peak_record_with_num_peaks_zero(tmp_path: Path):
    """A record that declares NUM_PEAKS=0 and has no peak lines is accepted."""
    module = load_prepare_module()
    fp = tmp_path / "zero.mgf"
    fp.write_text(_one([], **{"NUM_PEAKS=2": "NUM_PEAKS=0", "41.003009 0.5\n44.282551 1.0\n": ""}))
    spec_df, _ = module.process_specverse(str(fp))
    assert spec_df["num_peaks"].tolist() == [0]
    assert spec_df["peaks"].isna().all()
