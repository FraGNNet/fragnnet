import pytest
import torch as th
from torch.utils.data import DataLoader

from fragnnet.dataset.iterable_spec_mol_frag_dataset import IterableSpecMolFragDataset


def passthrough_collate(batch):
    """Identity collate, mirroring how run_inference.py wires the DataLoader."""
    return batch


class FakePyG:
    def __init__(self, num_nodes: int, num_edges: int):
        self.num_nodes = num_nodes
        self.num_edges = num_edges

    def clone(self):
        # return a shallow-cloned object (different identity)
        return FakePyG(self.num_nodes, self.num_edges)


class FakeDataset:
    """Minimal dataset-like object used for testing the iterable wrapper."""

    def __init__(self, node_counts, edge_counts=None, shared=False):
        self.node_counts = list(node_counts)
        if edge_counts is None:
            self.edge_counts = [n * 2 for n in self.node_counts]
        else:
            self.edge_counts = list(edge_counts)
        self.shared = shared
        self.collate_fn_calls = 0
        if shared:
            # single shared pyg object returned for every index
            self._shared_pyg = FakePyG(self.node_counts[0], self.edge_counts[0])

    def __len__(self):
        return len(self.node_counts)

    def __getitem__(self, idx):
        if self.shared:
            pyg = self._shared_pyg
        else:
            pyg = FakePyG(self.node_counts[idx], self.edge_counts[idx])
        return {"frag_pyg": pyg, "idx": idx}

    def get_collate_fn(self):
        self.collate_fn_calls += 1

        # simple collate that returns batch_size and the underlying samples list
        def collate_fn(data_list):
            return {"batch_size": th.tensor(len(data_list)), "samples": data_list}

        return collate_fn


def collect_batches(iterable):
    return list(iterable)


def test_batches_by_nodes():
    # node counts: [3,4,5,2] max_num=7 should group as [3,4],[5,2]
    node_counts = [3, 4, 5, 2]
    ds = FakeDataset(node_counts)
    idx_stream = iter(range(len(node_counts)))
    it = IterableSpecMolFragDataset(
        dataset=ds,
        index_stream=idx_stream,
        max_num=7,
        limited_by="frag_node",
        skip_too_big=False,
        return_collated=True,
    )

    batches = collect_batches(it)
    assert len(batches) == 2
    # check sums
    sums = [sum(sample["frag_pyg"].num_nodes for sample in batch["samples"]) for batch in batches]
    assert sums == [7, 7]
    assert all(isinstance(b["batch_size"], th.Tensor) for b in batches)


def test_skip_too_big():
    # node counts: [5,12,4] with max_num=10 and skip_too_big True should skip 12
    node_counts = [5, 12, 4]
    ds = FakeDataset(node_counts)
    idx_stream = iter(range(len(node_counts)))
    it = IterableSpecMolFragDataset(
        dataset=ds,
        index_stream=idx_stream,
        max_num=10,
        limited_by="frag_node",
        skip_too_big=True,
        return_collated=False,
    )

    batches = collect_batches(it)
    # expected behavior: oversized sample (12) skipped; remaining samples may be
    # coalesced into a single batch: [5,4]
    assert len(batches) == 1
    assert sum(len(batch) for batch in batches) == 2
    # ensure skipped index 1 not present
    present_idxs = [s["idx"] for batch in batches for s in batch]
    assert 1 not in present_idxs


def test_return_list_mode_and_clone():
    # Test return_collated=False and clone_graphs_on_get behavior
    node_counts = [2, 2]
    # use shared pyg to ensure clone creates distinct objects
    ds = FakeDataset(node_counts, shared=True)
    idx_stream = iter(range(len(node_counts)))
    # when clone_graphs_on_get=True, yielded samples should not reference dataset._shared_pyg
    it = IterableSpecMolFragDataset(
        dataset=ds,
        index_stream=idx_stream,
        max_num=10,
        limited_by="frag_node",
        skip_too_big=False,
        return_collated=False,
        clone_graphs_on_get=True,
    )

    batches = collect_batches(it)
    assert len(batches) == 1
    batch = batches[0]
    assert isinstance(batch, list)
    for sample in batch:
        assert sample["frag_pyg"] is not ds._shared_pyg


def _make_iterable(node_counts, max_num, **kwargs):
    ds = FakeDataset(node_counts)
    it = IterableSpecMolFragDataset(
        dataset=ds,
        index_stream=range(len(node_counts)),
        max_num=max_num,
        limited_by="frag_node",
        **kwargs,
    )
    return ds, it


def test_no_len_because_batch_count_is_unknown():
    """Batches form lazily, so the stream has no length (and neither does its DataLoader)."""
    _, it = _make_iterable([3, 4, 5], max_num=7)
    with pytest.raises(TypeError):
        len(it)
    with pytest.raises(TypeError):
        len(DataLoader(it, batch_size=None, collate_fn=passthrough_collate))


def test_missing_frag_pyg_raises_instead_of_unbounded_batch():
    """A sample with no frag_pyg must fail loudly, not count as size 0 and grow the batch."""

    class NoFragDataset(FakeDataset):
        def __getitem__(self, idx):
            return {"idx": idx}

    ds = NoFragDataset([1, 1, 1])
    it = IterableSpecMolFragDataset(
        dataset=ds, index_stream=range(3), max_num=2, limited_by="frag_node"
    )
    with pytest.raises(KeyError, match="frag_pyg"):
        list(it)


def test_zero_size_samples_raise_instead_of_one_unbounded_batch():
    """frag_edge is 0 for every sample in configs without fragment edges.

    Sizing by it would never reach max_num, so the whole stream would land in one
    batch and OOM. Must fail loudly and name the fix instead.
    """
    ds = FakeDataset([3, 4, 5], edge_counts=[0, 0, 0])
    it = IterableSpecMolFragDataset(
        dataset=ds, index_stream=range(3), max_num=8, limited_by="frag_edge"
    )
    with pytest.raises(ValueError, match="frag_node"):
        list(it)

    # the same samples batch fine when sized by nodes
    it_ok = IterableSpecMolFragDataset(
        dataset=ds, index_stream=range(3), max_num=8, limited_by="frag_node"
    )
    assert len(collect_batches(it_ok)) == 2


def test_default_limited_by_is_frag_node():
    """The default must be the metric that is always populated."""
    ds = FakeDataset([3, 4], edge_counts=[0, 0])
    it = IterableSpecMolFragDataset(dataset=ds, index_stream=range(2), max_num=5)
    batches = collect_batches(it)
    assert len(batches) == 2  # sized by nodes: 3 then 4, so they split


def test_collate_fn_fetched_once_per_iteration():
    """get_collate_fn() is hoisted out of the per-batch path."""
    ds, it = _make_iterable([1] * 10, max_num=2)
    batches = collect_batches(it)
    assert len(batches) == 5
    assert ds.collate_fn_calls == 1


def test_collate_fn_not_fetched_when_uncollated():
    ds, it = _make_iterable([1, 1], max_num=2, return_collated=False)
    collect_batches(it)
    assert ds.collate_fn_calls == 0


def test_dataloader_passthrough_covers_every_index_once():
    """DataLoader(batch_size=None) must hand back the dataset's own batches untouched."""
    node_counts = [3, 4, 5, 2, 1, 6]
    _, it = _make_iterable(node_counts, max_num=7)
    dl = DataLoader(it, batch_size=None, collate_fn=passthrough_collate)

    batches = list(dl)
    seen = [s["idx"] for b in batches for s in b["samples"]]
    assert sorted(seen) == list(range(len(node_counts)))
    for b in batches:
        # batch_size must survive as the collate output, not be re-batched into a list
        assert isinstance(b, dict)
        assert int(b["batch_size"]) == len(b["samples"])
        assert sum(s["frag_pyg"].num_nodes for s in b["samples"]) <= 7
