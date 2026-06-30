import torch as th

from fragnnet.dataset.iterable_spec_mol_frag_dataset import IterableSpecMolFragDataset


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
        if shared:
            # single shared pyg object returned for every index
            self._shared_pyg = FakePyG(self.node_counts[0], self.edge_counts[0])

    def __getitem__(self, idx):
        if self.shared:
            pyg = self._shared_pyg
        else:
            pyg = FakePyG(self.node_counts[idx], self.edge_counts[idx])
        return {"frag_pyg": pyg, "idx": idx}

    def get_collate_fn(self):
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
