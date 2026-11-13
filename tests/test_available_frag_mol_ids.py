"""Tests for script_utils.available_frag_mol_ids.

run_inference.py used to enumerate available DAGs with os.listdir(frag_dp), which
raises NotADirectoryError on the packed-HDF5 backend (frag_dp is a dags.h5 file, not
a directory). available_frag_mol_ids handles both layouts, and for HDF5 it must still
exclude molecules whose fragmentation failed: the h5 backend does no missing-entry
check of its own, so a stale mol_df row surfaces later as a KeyError in the dataset.
"""

import json
import os

import h5py

from fragnnet.utils.frag_utils import PACKED_DAG_INDEX_DATASET, initialize_packed_dag_hdf5
from fragnnet.utils.script_utils import available_frag_mol_ids


def test_folder_backend_returns_mol_id_stems(tmp_path):
    for name in ("7.pkl.bz2", "12.pkl.bz2", "103.pkl"):
        (tmp_path / name).touch()

    assert available_frag_mol_ids(str(tmp_path)) == {"7", "12", "103"}


def _write_packed_h5(fp, stored_mol_ids, n_slots):
    """Write a packed dags.h5 whose index has an entry only for stored_mol_ids."""
    with h5py.File(fp, "w") as h5_file:
        initialize_packed_dag_hdf5(h5_file)
        index = h5_file[PACKED_DAG_INDEX_DATASET]
        index.resize((n_slots,))
        arr = index[:]
        for mol_id in stored_mol_ids:
            arr["length"][mol_id] = 8
        index[:] = arr


def test_h5_backend_reads_meta_info_when_present(tmp_path):
    h5_fp = str(tmp_path / "dags.h5")
    _write_packed_h5(h5_fp, [2, 5], n_slots=6)
    with open(tmp_path / "meta_info.json", "w") as f:
        json.dump({"2": [3, 2], "5": [4, 3]}, f)

    assert available_frag_mol_ids(h5_fp) == {"2", "5"}


def test_h5_backend_falls_back_to_packed_index(tmp_path):
    h5_fp = str(tmp_path / "dags.h5")
    _write_packed_h5(h5_fp, [2, 5], n_slots=6)

    # No meta_info.json next to the .h5 -> read the dense index directly. Slots 0,1,3,4
    # have length 0, meaning no DAG was stored for those mol_ids.
    assert not os.path.exists(tmp_path / "meta_info.json")
    assert available_frag_mol_ids(h5_fp) == {"2", "5"}


def test_h5_backend_excludes_failed_molecules(tmp_path):
    """A dense index slot with length 0 is a frag-gen failure, not an empty DAG."""
    h5_fp = str(tmp_path / "dags.h5")
    _write_packed_h5(h5_fp, [0, 3], n_slots=4)

    mol_ids = available_frag_mol_ids(h5_fp)
    assert mol_ids == {"0", "3"}
    assert "1" not in mol_ids and "2" not in mol_ids
