"""Tests for the per-molecule neutral-DAG LRU in `SpecMolFragDataset._get_neutral_frag`.

A DAG is per MOLECULE but `__getitem__` runs per SPECTRUM, and inference forces both
`frag_params.preprocess` and `frag_params.preload` off (see `apply_inference_config`), so
every collision energy / adduct of a molecule used to re-read the DAG and rebuild its PyG
graph. These tests pin the cache, the LRU eviction order, and the aliasing contract the
cache imposes on callers.
"""

import pytest

from fragnnet.dataset.spec_mol_frag_dataset import SpecMolFragDataset


class _StubDataset(SpecMolFragDataset):
    """Bare `SpecMolFragDataset` with the two expensive steps counted, not run.

    Built with `object.__new__` so none of the heavy dataframe/split setup happens; only
    the attributes `_get_neutral_frag` actually reads are set.
    """

    def __init__(self, cache_size: int, preload: bool = False):
        self.frag_params = {"preload": preload, "cache_size": cache_size}
        self._frag_cache = None
        self._frag_cache_size = cache_size
        self.load_calls: list[str] = []
        self.process_calls: list[str] = []
        self.frag_entries = {f"m{i}": {"entry": f"m{i}"} for i in range(10)}

    def _load_frag_entry(self, mol_id):
        self.load_calls.append(mol_id)
        return {"entry": mol_id}

    def _process_neutral_frag(self, frag_entry, mol=None):  # noqa: ARG002
        self.process_calls.append(frag_entry["entry"])
        return {"frag_formula_peak_mzs": f"mzs:{frag_entry['entry']}", "frag_pyg": object()}


def _mol_entry():
    return {"mol": None}


def test_repeat_molecule_hits_cache():
    """Many spectra, one molecule: load and process exactly once."""
    ds = _StubDataset(cache_size=16)
    results = [ds._get_neutral_frag("m0", _mol_entry()) for _ in range(5)]

    assert ds.load_calls == ["m0"]
    assert ds.process_calls == ["m0"]
    # every caller gets the SAME dict back, flagged as shared
    assert all(shared for _, shared in results)
    assert all(data is results[0][0] for data, _ in results)


def test_cache_disabled_reloads_every_time():
    ds = _StubDataset(cache_size=0)
    results = [ds._get_neutral_frag("m0", _mol_entry()) for _ in range(3)]

    assert ds.load_calls == ["m0", "m0", "m0"]
    assert ds.process_calls == ["m0", "m0", "m0"]
    # freshly built each time -> not shared, so the caller may skip the defensive clone
    assert all(not shared for _, shared in results)
    assert results[0][0] is not results[1][0]


def test_preload_path_also_cached_and_skips_disk():
    """With `preload`, the DAG bytes are already in RAM but the graph build still repeats."""
    ds = _StubDataset(cache_size=16, preload=True)
    for _ in range(4):
        ds._get_neutral_frag("m3", _mol_entry())

    assert ds.load_calls == []  # never touches disk/HDF5
    assert ds.process_calls == ["m3"]  # and builds the graph once


def test_lru_evicts_oldest_and_respects_size():
    ds = _StubDataset(cache_size=2)
    ds._get_neutral_frag("m0", _mol_entry())
    ds._get_neutral_frag("m1", _mol_entry())
    ds._get_neutral_frag("m2", _mol_entry())  # evicts m0

    assert list(ds._frag_cache.keys()) == ["m1", "m2"]
    assert len(ds._frag_cache) == 2

    ds._get_neutral_frag("m0", _mol_entry())  # miss: reloaded
    assert ds.process_calls == ["m0", "m1", "m2", "m0"]


def test_hit_refreshes_recency():
    ds = _StubDataset(cache_size=2)
    ds._get_neutral_frag("m0", _mol_entry())
    ds._get_neutral_frag("m1", _mol_entry())
    ds._get_neutral_frag("m0", _mol_entry())  # hit -> m0 becomes most recent
    ds._get_neutral_frag("m2", _mol_entry())  # so m1 is what gets evicted

    assert list(ds._frag_cache.keys()) == ["m0", "m2"]
    assert ds.process_calls == ["m0", "m1", "m2"]


def test_interleaved_molecules_still_hit_within_capacity():
    """Unsorted input (A B A B ...) is fine as long as the working set fits."""
    ds = _StubDataset(cache_size=4)
    for mol_id in ["m0", "m1", "m0", "m1", "m0", "m1"]:
        ds._get_neutral_frag(mol_id, _mol_entry())

    assert ds.process_calls == ["m0", "m1"]


def test_cache_is_lazy_so_it_is_not_pickled_to_workers():
    ds = _StubDataset(cache_size=16)
    assert ds._frag_cache is None
    ds._get_neutral_frag("m0", _mol_entry())
    assert ds._frag_cache is not None


def test_cache_size_required_during_dataset_setup(monkeypatch):
    """Missing cache configuration fails in the real constructor."""
    monkeypatch.setattr(SpecMolFragDataset, "_base_init", lambda self, **kwargs: None)
    with pytest.raises(KeyError, match="cache_size"):
        SpecMolFragDataset(
            spec_fp="unused",
            mol_fp="unused",
            split_dp="unused",
            split="predict_only",
            subsample_params={},
            spec_params={},
            mol_params={},
            frag_dp="unused",
            frag_params={"preload": False},
        )


@pytest.mark.parametrize("cache_size", [0, 1, 16])
def test_returned_mzs_match_requested_molecule(cache_size):
    """A cache keyed wrongly would hand back another molecule's DAG."""
    ds = _StubDataset(cache_size=cache_size)
    for mol_id in ["m0", "m1", "m2", "m1", "m0"]:
        data, _ = ds._get_neutral_frag(mol_id, _mol_entry())
        assert data["frag_formula_peak_mzs"] == f"mzs:{mol_id}"
