"""Tests for fragment pickle IO helpers in ``fragnnet.utils.frag_utils``."""

import bz2
import io
import pickle
import tarfile
import zipfile

import pytest

from fragnnet.utils import frag_utils

# ---------------------------------------------------------------------------
# get_frag_name – canonical naming (plain int, .pkl)
# ---------------------------------------------------------------------------


def test_get_frag_name_int_uncompressed():
    """Integer mol_id produces plain-int .pkl filename."""
    assert frag_utils.get_frag_name(123, is_compressed=False) == "123.pkl"


def test_get_frag_name_int_compressed():
    """Integer mol_id produces plain-int .pkl.bz2 filename."""
    assert frag_utils.get_frag_name(123, is_compressed=True) == "123.pkl.bz2"


def test_get_frag_name_str_numeric():
    """Numeric string mol_id is treated identically to int."""
    assert frag_utils.get_frag_name("123", is_compressed=False) == "123.pkl"


def test_get_frag_name_no_zero_padding():
    """mol_id 1 is not zero-padded."""
    assert frag_utils.get_frag_name(1, is_compressed=False) == "1.pkl"


def test_get_frag_name_large_id():
    """Large mol_id produces the expected filename."""
    assert frag_utils.get_frag_name(123456789, is_compressed=False) == "123456789.pkl"


# ---------------------------------------------------------------------------
# get_dag_output_path consistency with get_frag_name
# ---------------------------------------------------------------------------


def test_dag_output_path_no_compress_matches_frag_fp(tmp_path):
    """`get_dag_output_path` with no compression matches `get_frag_fp`."""
    mol_id = 42
    dag_dp = str(tmp_path)
    assert frag_utils.get_dag_output_path(dag_dp, mol_id, False, "bz2") == frag_utils.get_frag_fp(
        mol_id, dag_dp, is_compressed=False
    )


def test_dag_output_path_bz2_matches_frag_fp(tmp_path):
    """`get_dag_output_path` with bz2 matches `get_frag_fp` with is_compressed=True."""
    mol_id = 42
    dag_dp = str(tmp_path)
    assert frag_utils.get_dag_output_path(dag_dp, mol_id, True, "bz2") == frag_utils.get_frag_fp(
        mol_id, dag_dp, is_compressed=True
    )


def test_dag_output_path_gz(tmp_path):
    """`get_dag_output_path` with gz produces a plain-int .pkl.gz path."""
    path = frag_utils.get_dag_output_path(str(tmp_path), 7, True, "gz")
    assert path.endswith("7.pkl.gz")


def test_dag_output_path_invalid_format(tmp_path):
    with pytest.raises(ValueError, match="Invalid compression format"):
        frag_utils.get_dag_output_path(str(tmp_path), 1, True, "xz")


# ---------------------------------------------------------------------------
# save_frag_d / load_frag_d – folder backend, canonical format
# ---------------------------------------------------------------------------


def test_save_and_load_frag_local_uncompressed(tmp_path):
    """Round-trip with integer mol_id, uncompressed."""
    payload = {"dag": {"num_nodes": 3}, "meta": [1, 2, 3]}
    mol_id = 123

    frag_utils.save_frag_d(payload, mol_id, str(tmp_path), is_compressed=False)
    assert (tmp_path / "123.pkl").exists()
    loaded = frag_utils.load_frag_d(mol_id, str(tmp_path), is_compressed=False)
    assert loaded == payload


def test_save_and_load_frag_local_bz2(tmp_path):
    """Round-trip with integer mol_id, bz2-compressed."""
    payload = {"dag": {"num_nodes": 5}, "meta": [4, 5, 6]}
    mol_id = 456

    frag_utils.save_frag_d(payload, mol_id, str(tmp_path), is_compressed=True)
    assert (tmp_path / "456.pkl.bz2").exists()
    loaded = frag_utils.load_frag_d(mol_id, str(tmp_path), is_compressed=True)
    assert loaded == payload


def test_load_frag_via_get_dag_output_path_roundtrip(tmp_path):
    """File written by get_dag_output_path can be read back by load_frag_d."""
    payload = {"dag": {"num_nodes": 8}, "meta": "roundtrip"}
    mol_id = 99
    dag_dp = str(tmp_path)

    fp = frag_utils.get_dag_output_path(dag_dp, mol_id, compress_dags=False, compress_format="bz2")
    frag_utils.dump_dag_pickle(fp, payload)

    loaded = frag_utils.load_frag_d(mol_id, dag_dp, is_compressed=False)
    assert loaded == payload


# ---------------------------------------------------------------------------
# load_frag_d – folder backend, backward-compat (legacy .pickle naming)
# ---------------------------------------------------------------------------


def test_load_frag_legacy_pickle_uncompressed(tmp_path):
    """load_frag_d falls back to legacy '{mol_id}.pickle' when canonical is absent."""
    payload = {"dag": {"num_nodes": 2}, "legacy": "pickle"}
    mol_id = 7

    legacy_fp = tmp_path / f"{mol_id}.pickle"
    with open(legacy_fp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    loaded = frag_utils.load_frag_d(mol_id, str(tmp_path), is_compressed=False)
    assert loaded == payload


def test_load_frag_legacy_pickle_bz2(tmp_path):
    """load_frag_d falls back to legacy '{mol_id}.pickle.bz2' when canonical is absent."""
    payload = {"dag": {"num_nodes": 4}, "legacy": "pickle_bz2"}
    mol_id = 8

    legacy_fp = tmp_path / f"{mol_id}.pickle.bz2"
    with bz2.open(legacy_fp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    loaded = frag_utils.load_frag_d(mol_id, str(tmp_path), is_compressed=True)
    assert loaded == payload


def test_load_frag_legacy_zero_padded_uncompressed(tmp_path):
    """load_frag_d falls back to zero-padded '{mol_id:08d}.pkl' when canonical is absent."""
    payload = {"dag": {"num_nodes": 2}, "legacy": "zero_padded"}
    mol_id = 7

    legacy_fp = tmp_path / f"{mol_id:08d}.pkl"
    with open(legacy_fp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    loaded = frag_utils.load_frag_d(mol_id, str(tmp_path), is_compressed=False)
    assert loaded == payload


def test_load_frag_legacy_zero_padded_bz2(tmp_path):
    """load_frag_d falls back to zero-padded '{mol_id:08d}.pkl.bz2' when canonical is absent."""
    payload = {"dag": {"num_nodes": 4}, "legacy": "zero_padded_bz2"}
    mol_id = 8

    legacy_fp = tmp_path / f"{mol_id:08d}.pkl.bz2"
    with bz2.open(legacy_fp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    loaded = frag_utils.load_frag_d(mol_id, str(tmp_path), is_compressed=True)
    assert loaded == payload


def test_canonical_takes_priority_over_legacy(tmp_path):
    """When both canonical and legacy files exist, canonical is returned."""
    mol_id = 10
    canonical_payload = {"source": "canonical"}
    legacy_payload = {"source": "legacy"}

    frag_utils.save_frag_d(canonical_payload, mol_id, str(tmp_path), is_compressed=False)
    legacy_fp = tmp_path / f"{mol_id}.pickle"
    with open(legacy_fp, "wb") as f:
        pickle.dump(legacy_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    loaded = frag_utils.load_frag_d(mol_id, str(tmp_path), is_compressed=False)
    assert loaded == canonical_payload


# ---------------------------------------------------------------------------
# load_frag_d – tar archive backend
# ---------------------------------------------------------------------------


def _make_tar(path, member_name: str, content: bytes):
    info = tarfile.TarInfo(name=member_name)
    info.size = len(content)
    with tarfile.open(path, "w") as tf:
        tf.addfile(info, io.BytesIO(content))


def test_load_frag_from_tar_canonical_uncompressed(tmp_path):
    """Load fragment from tar where member uses canonical name."""
    payload = {"dag": {"num_nodes": 7}, "meta": [7, 8, 9]}
    mol_id = 789
    member_name = frag_utils.get_frag_name(mol_id, is_compressed=False)
    tar_fp = tmp_path / "frags.tar"

    _make_tar(tar_fp, member_name, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    assert frag_utils.load_frag_d(mol_id, str(tar_fp), is_compressed=False) == payload


def test_load_frag_from_tar_canonical_bz2(tmp_path):
    """Load fragment from tar where member uses canonical bz2 name."""
    payload = {"dag": {"num_nodes": 9}, "meta": [10, 11, 12]}
    mol_id = 321
    member_name = frag_utils.get_frag_name(mol_id, is_compressed=True)
    tar_fp = tmp_path / "frags_bz2.tar"

    _make_tar(
        tar_fp, member_name, bz2.compress(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    )
    assert frag_utils.load_frag_d(mol_id, str(tar_fp), is_compressed=True) == payload


def test_load_frag_from_tar_legacy_pickle_uncompressed(tmp_path):
    """load_frag_d falls back to legacy .pickle member name inside tar."""
    payload = {"dag": {"num_nodes": 7}, "meta": "tar_legacy"}
    mol_id = 789
    member_name = f"{mol_id}.pickle"
    tar_fp = tmp_path / "frags_legacy.tar"

    _make_tar(tar_fp, member_name, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    assert frag_utils.load_frag_d(mol_id, str(tar_fp), is_compressed=False) == payload


def test_load_frag_from_tar_legacy_pickle_bz2(tmp_path):
    """load_frag_d falls back to legacy .pickle.bz2 member name inside tar."""
    payload = {"dag": {"num_nodes": 9}, "meta": "tar_legacy_bz2"}
    mol_id = 321
    member_name = f"{mol_id}.pickle.bz2"
    tar_fp = tmp_path / "frags_legacy_bz2.tar"

    _make_tar(
        tar_fp, member_name, bz2.compress(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    )
    assert frag_utils.load_frag_d(mol_id, str(tar_fp), is_compressed=True) == payload


def test_load_frag_from_tar_legacy_zero_padded(tmp_path):
    """load_frag_d falls back to zero-padded member name inside tar."""
    payload = {"dag": {"num_nodes": 7}, "meta": "tar_zero_padded"}
    mol_id = 789
    member_name = f"{mol_id:08d}.pkl"
    tar_fp = tmp_path / "frags_zero_padded.tar"

    _make_tar(tar_fp, member_name, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    assert frag_utils.load_frag_d(mol_id, str(tar_fp), is_compressed=False) == payload


# ---------------------------------------------------------------------------
# load_frag_d – zip archive backend
# ---------------------------------------------------------------------------


def test_load_frag_from_zip_canonical_uncompressed(tmp_path):
    """Load fragment from zip where member uses canonical name."""
    payload = {"dag": {"num_nodes": 11}, "meta": [13, 14, 15]}
    mol_id = 654
    member_name = frag_utils.get_frag_name(mol_id, is_compressed=False)
    zip_fp = tmp_path / "frags.zip"

    content = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    with zipfile.ZipFile(zip_fp, "w") as zf:
        zf.writestr(member_name, content)

    assert frag_utils.load_frag_d(mol_id, str(zip_fp), is_compressed=False) == payload


def test_load_frag_from_zip_canonical_bz2(tmp_path):
    """Load fragment from zip where member uses canonical bz2 name."""
    payload = {"dag": {"num_nodes": 13}, "meta": [16, 17, 18]}
    mol_id = 987
    member_name = frag_utils.get_frag_name(mol_id, is_compressed=True)
    zip_fp = tmp_path / "frags_bz2.zip"

    content = bz2.compress(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    with zipfile.ZipFile(zip_fp, "w") as zf:
        zf.writestr(member_name, content)

    assert frag_utils.load_frag_d(mol_id, str(zip_fp), is_compressed=True) == payload


def test_load_frag_from_zip_legacy_pickle_uncompressed(tmp_path):
    """load_frag_d falls back to legacy .pickle member name inside zip."""
    payload = {"dag": {"num_nodes": 11}, "meta": "zip_legacy"}
    mol_id = 654
    member_name = f"{mol_id}.pickle"
    zip_fp = tmp_path / "frags_legacy.zip"

    content = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    with zipfile.ZipFile(zip_fp, "w") as zf:
        zf.writestr(member_name, content)

    assert frag_utils.load_frag_d(mol_id, str(zip_fp), is_compressed=False) == payload


def test_load_frag_from_zip_legacy_pickle_bz2(tmp_path):
    """load_frag_d falls back to legacy .pickle.bz2 member name inside zip."""
    payload = {"dag": {"num_nodes": 13}, "meta": "zip_legacy_bz2"}
    mol_id = 987
    member_name = f"{mol_id}.pickle.bz2"
    zip_fp = tmp_path / "frags_legacy_bz2.zip"

    content = bz2.compress(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    with zipfile.ZipFile(zip_fp, "w") as zf:
        zf.writestr(member_name, content)

    assert frag_utils.load_frag_d(mol_id, str(zip_fp), is_compressed=True) == payload


def test_load_frag_from_zip_legacy_zero_padded(tmp_path):
    """load_frag_d falls back to zero-padded member name inside zip."""
    payload = {"dag": {"num_nodes": 11}, "meta": "zip_zero_padded"}
    mol_id = 654
    member_name = f"{mol_id:08d}.pkl"
    zip_fp = tmp_path / "frags_zero_padded.zip"

    content = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    with zipfile.ZipFile(zip_fp, "w") as zf:
        zf.writestr(member_name, content)

    assert frag_utils.load_frag_d(mol_id, str(zip_fp), is_compressed=False) == payload
