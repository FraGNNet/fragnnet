"""Tests for the DAG filename migration script."""

import importlib.util
import io
import pickle
import sys
import tarfile
from pathlib import Path

import pytest

# preproc_scripts is not an installed package; load the module directly.
_SCRIPT = (
    Path(__file__).parent.parent
    / "preproc_scripts"
    / "dataset_migration"
    / "migrate_dag_zero_padded_to_plain_int.py"
)
_spec = importlib.util.spec_from_file_location("migrate_dag_naming", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_resolve_name = _mod._resolve_name
migrate_dir = _mod.migrate_dir
migrate_tar = _mod.migrate_tar


# ---------------------------------------------------------------------------
# _resolve_name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        # Already canonical — no change
        ("123.pkl", ("123.pkl", False)),
        ("123.pkl.bz2", ("123.pkl.bz2", False)),
        ("123.pkl.gz", ("123.pkl.gz", False)),
        # Zero-padded .pkl
        ("00000123.pkl", ("123.pkl", True)),
        ("00000123.pkl.bz2", ("123.pkl.bz2", True)),
        ("00000123.pkl.gz", ("123.pkl.gz", True)),
        ("00000001.pkl", ("1.pkl", True)),
        # Old .pickle extension (plain int)
        ("123.pickle", ("123.pkl", True)),
        ("123.pickle.bz2", ("123.pkl.bz2", True)),
        # Old .pickle extension (zero-padded)
        ("00000123.pickle", ("123.pkl", True)),
        ("00000123.pickle.bz2", ("123.pkl.bz2", True)),
        # Unrelated files — untouched
        ("meta_info.json", ("meta_info.json", False)),
        ("dags.h5", ("dags.h5", False)),
        # Tar member paths with directory prefix — prefix preserved
        ("dags/00000123.pkl", ("dags/123.pkl", True)),
        ("dags/00000001.pkl", ("dags/1.pkl", True)),
        ("dags/123.pickle", ("dags/123.pkl", True)),
        ("dags/00000123.pickle.bz2", ("dags/123.pkl.bz2", True)),
        ("dags/123.pkl", ("dags/123.pkl", False)),
        ("dags/meta_info.json", ("dags/meta_info.json", False)),
        # Nested prefix
        ("a/b/00000007.pkl", ("a/b/7.pkl", True)),
    ],
)
def test_resolve_name(name, expected):
    assert _resolve_name(name) == expected


# ---------------------------------------------------------------------------
# migrate_dir – directory backend
# ---------------------------------------------------------------------------


def _write_pkl(path: Path, payload: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def test_migrate_dir_zero_padded(tmp_path):
    """Zero-padded .pkl files are renamed to plain-int."""
    _write_pkl(tmp_path / "00000007.pkl", {"id": 7})
    _write_pkl(tmp_path / "00000042.pkl", {"id": 42})

    renamed, skipped, untouched = migrate_dir(tmp_path, apply=True)

    assert renamed == 2
    assert skipped == 0
    assert (tmp_path / "7.pkl").exists()
    assert (tmp_path / "42.pkl").exists()
    assert not (tmp_path / "00000007.pkl").exists()


def test_migrate_dir_old_pickle_ext(tmp_path):
    """Files with .pickle extension are renamed to .pkl."""
    _write_pkl(tmp_path / "7.pickle", {"id": 7})
    _write_pkl(tmp_path / "7.pickle.bz2", {"id": 7})  # wrong ext but valid legacy

    renamed, skipped, untouched = migrate_dir(tmp_path, apply=True)

    assert renamed == 2
    assert (tmp_path / "7.pkl").exists()
    assert (tmp_path / "7.pkl.bz2").exists()


def test_migrate_dir_dry_run_no_changes(tmp_path):
    """Dry run does not rename any files."""
    _write_pkl(tmp_path / "00000007.pkl", {"id": 7})

    renamed, skipped, untouched = migrate_dir(tmp_path, apply=False)

    assert renamed == 1
    assert (tmp_path / "00000007.pkl").exists()  # untouched
    assert not (tmp_path / "7.pkl").exists()


def test_migrate_dir_already_canonical(tmp_path):
    """Files already in canonical format are counted but not renamed."""
    _write_pkl(tmp_path / "7.pkl", {"id": 7})

    renamed, skipped, untouched = migrate_dir(tmp_path, apply=True)

    assert renamed == 0
    assert untouched == 1
    assert (tmp_path / "7.pkl").exists()


def test_migrate_dir_collision_skipped(tmp_path):
    """When canonical destination already exists, rename is skipped."""
    _write_pkl(tmp_path / "00000007.pkl", {"zero_padded": True})
    _write_pkl(tmp_path / "7.pkl", {"canonical": True})

    renamed, skipped, untouched = migrate_dir(tmp_path, apply=True)

    assert skipped == 1
    # Both files still present; canonical was not overwritten
    with open(tmp_path / "7.pkl", "rb") as f:
        assert pickle.load(f) == {"canonical": True}


def test_migrate_dir_mixed(tmp_path):
    """Mix of canonical, zero-padded, and old-ext files."""
    _write_pkl(tmp_path / "1.pkl", {"id": 1})           # canonical
    _write_pkl(tmp_path / "00000002.pkl", {"id": 2})    # zero-padded
    _write_pkl(tmp_path / "3.pickle", {"id": 3})        # old ext

    renamed, skipped, untouched = migrate_dir(tmp_path, apply=True)

    assert renamed == 2
    assert untouched == 1
    assert skipped == 0
    assert (tmp_path / "1.pkl").exists()
    assert (tmp_path / "2.pkl").exists()
    assert (tmp_path / "3.pkl").exists()


# ---------------------------------------------------------------------------
# migrate_tar – tar backend
# ---------------------------------------------------------------------------


def _make_tar(tar_path: Path, members: dict[str, dict]) -> None:
    """Write a tar archive with {member_name: payload} entries."""
    with tarfile.open(tar_path, "w") as tf:
        for name, payload in members.items():
            content = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            import io as _io
            tf.addfile(info, _io.BytesIO(content))


def _read_tar_members(tar_path: Path) -> dict[str, dict]:
    """Return {member_name: payload} from a tar archive."""
    result = {}
    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            f = tf.extractfile(member)
            result[member.name] = pickle.loads(f.read())
    return result


def test_migrate_tar_zero_padded(tmp_path):
    """Zero-padded members are renamed in the output tar."""
    src = tmp_path / "src.tar"
    dst = tmp_path / "dst.tar"
    _make_tar(src, {"00000007.pkl": {"id": 7}, "00000042.pkl": {"id": 42}})

    renamed, skipped, copied_unchanged = migrate_tar(src, dst, apply=True)

    assert renamed == 2
    members = _read_tar_members(dst)
    assert set(members) == {"7.pkl", "42.pkl"}
    assert members["7.pkl"] == {"id": 7}
    assert members["42.pkl"] == {"id": 42}


def test_migrate_tar_old_pickle_ext(tmp_path):
    """Members with .pickle extension are renamed to .pkl in output tar."""
    src = tmp_path / "src.tar"
    dst = tmp_path / "dst.tar"
    _make_tar(src, {"7.pickle": {"id": 7}, "42.pickle.bz2": {"id": 42}})

    renamed, skipped, copied_unchanged = migrate_tar(src, dst, apply=True)

    assert renamed == 2
    members = _read_tar_members(dst)
    assert set(members) == {"7.pkl", "42.pkl.bz2"}


def test_migrate_tar_already_canonical(tmp_path):
    """Canonical members are copied unchanged."""
    src = tmp_path / "src.tar"
    dst = tmp_path / "dst.tar"
    _make_tar(src, {"7.pkl": {"id": 7}})

    renamed, skipped, copied_unchanged = migrate_tar(src, dst, apply=True)

    assert renamed == 0
    assert copied_unchanged == 1
    members = _read_tar_members(dst)
    assert members == {"7.pkl": {"id": 7}}


def test_migrate_tar_dry_run_no_output(tmp_path):
    """Dry run does not write the output tar."""
    src = tmp_path / "src.tar"
    dst = tmp_path / "dst.tar"
    _make_tar(src, {"00000007.pkl": {"id": 7}})

    renamed, skipped, copied_unchanged = migrate_tar(src, dst, apply=False)

    assert renamed == 1
    assert not dst.exists()


def test_migrate_tar_mixed(tmp_path):
    """Mix of canonical, zero-padded, and old-ext members in one tar."""
    src = tmp_path / "src.tar"
    dst = tmp_path / "dst.tar"
    _make_tar(
        src,
        {
            "1.pkl": {"id": 1},
            "00000002.pkl": {"id": 2},
            "3.pickle": {"id": 3},
        },
    )

    renamed, skipped, copied_unchanged = migrate_tar(src, dst, apply=True)

    assert renamed == 2
    assert copied_unchanged == 1
    members = _read_tar_members(dst)
    assert set(members) == {"1.pkl", "2.pkl", "3.pkl"}


def test_migrate_tar_collision_with_existing_canonical(tmp_path):
    """Zero-padded member is skipped when canonical name already exists in archive."""
    src = tmp_path / "src.tar"
    dst = tmp_path / "dst.tar"
    # 7.pkl (canonical) and 00000007.pkl (legacy) both present — legacy must be skipped.
    _make_tar(src, {"7.pkl": {"source": "canonical"}, "00000007.pkl": {"source": "zero_padded"}})

    renamed, skipped, copied_unchanged = migrate_tar(src, dst, apply=True)

    assert skipped == 1
    assert copied_unchanged == 1
    assert renamed == 0
    members = _read_tar_members(dst)
    assert members["7.pkl"] == {"source": "canonical"}


def test_migrate_tar_cross_legacy_collision(tmp_path):
    """Two legacy names resolving to the same canonical are detected as a collision."""
    src = tmp_path / "src.tar"
    dst = tmp_path / "dst.tar"
    # 00000007.pkl and 7.pickle both resolve to 7.pkl.
    _make_tar(src, {"00000007.pkl": {"source": "zero_padded"}, "7.pickle": {"source": "old_ext"}})

    renamed, skipped, copied_unchanged = migrate_tar(src, dst, apply=True)

    assert skipped == 1
    assert renamed == 1  # first one wins; second keeps its legacy name
    members = _read_tar_members(dst)
    # The renamed entry gets the canonical name; the skipped entry keeps its original legacy name.
    assert "7.pkl" in members
    assert "7.pickle" in members  # not dropped, preserved under original name
    assert members["7.pkl"] == {"source": "zero_padded"}


def test_migrate_tar_prefixed_member_names(tmp_path):
    """Tar members with a directory prefix (e.g. dags/00000001.pkl) are renamed correctly."""
    src = tmp_path / "src.tar"
    dst = tmp_path / "dst.tar"
    _make_tar(
        src,
        {
            "dags/00000001.pkl": {"id": 1},
            "dags/00000123.pkl": {"id": 123},
            "dags/7.pkl": {"id": 7},          # already canonical
            "dags/meta_info.json": {"n": 3},  # unrelated, untouched
        },
    )

    renamed, skipped, copied_unchanged = migrate_tar(src, dst, apply=True)

    assert renamed == 2
    assert skipped == 0
    assert copied_unchanged == 2

    members = _read_tar_members(dst)
    assert members["dags/1.pkl"] == {"id": 1}
    assert members["dags/123.pkl"] == {"id": 123}
    assert members["dags/7.pkl"] == {"id": 7}
    assert members["dags/meta_info.json"] == {"n": 3}
    # Legacy names must not appear in output
    assert "dags/00000001.pkl" not in members
    assert "dags/00000123.pkl" not in members


def test_migrate_tar_in_place(tmp_path):
    """Passing src_tar == dst_tar overwrites the archive atomically."""
    src = tmp_path / "dags.tar"
    _make_tar(src, {"00000001.pkl": {"id": 1}, "42.pkl": {"id": 42}})
    original_size = src.stat().st_size

    renamed, skipped, copied_unchanged = migrate_tar(src, src, apply=True)

    assert renamed == 1
    assert skipped == 0
    assert copied_unchanged == 1
    assert src.exists()
    members = _read_tar_members(src)
    assert members["1.pkl"] == {"id": 1}
    assert members["42.pkl"] == {"id": 42}
    assert "00000001.pkl" not in members
    # File was rewritten (size may differ from original)
    _ = original_size  # just ensure no exception above


def test_migrate_tar_atomic_on_error(tmp_path, monkeypatch):
    """Temp file is cleaned up if writing fails midway."""
    src = tmp_path / "src.tar"
    dst = tmp_path / "dst.tar"
    _make_tar(src, {"00000007.pkl": {"id": 7}})

    original_extractfile = tarfile.TarFile.extractfile

    def _broken_extract(self, member):
        raise RuntimeError("simulated IO error")

    monkeypatch.setattr(tarfile.TarFile, "extractfile", _broken_extract)

    with pytest.raises(RuntimeError, match="simulated IO error"):
        migrate_tar(src, dst, apply=True)

    assert not dst.exists()
    # No leftover .tar.tmp files
    assert not list(tmp_path.glob("*.tar.tmp"))
