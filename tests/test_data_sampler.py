"""Unit tests for fragnnet.dataset.data_sampler module.

Tests for batch samplers, group samplers, and dynamic batch sampling strategies.
"""

from types import SimpleNamespace

import pytest
import torch as th
from torch.utils.data import DataLoader, RandomSampler

# Load data_sampler directly to avoid package-level import side-effects
from fragnnet.dataset import (
    DualGroupDynamicBatchSampler,
    FormulaPrecTypeGroupSampler,
    GreedyInferenceBatchSampler,
    GroupSampler,
    SpecMolFragDynamicBatchSampler,
)

# ============================================================================
# Mock and Helper Classes
# ============================================================================


class MockFrag:
    """Mock fragment for testing."""

    def __init__(self, nodes: int, edges: int = 0):
        self.num_nodes = nodes
        self.num_edges = edges


class FakeFrag:
    """Alternative mock fragment."""

    def __init__(self, nodes, edges):
        self.num_nodes = nodes
        self.num_edges = edges


class SyntheticDataset:
    """Synthetic dataset for testing samplers."""

    def __init__(self, items):
        self._items = items

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


class FakeDataset:
    """Lightweight dataset for dynamic batch sampler tests."""

    def __init__(self, sizes):
        self.sizes = sizes

    def __len__(self):
        return len(self.sizes)

    def __getitem__(self, idx):
        nodes, edges = self.sizes[idx]
        return {"frag_pyg": SimpleNamespace(num_nodes=nodes, num_edges=edges)}


class DummyDataset:
    """Dummy dataset for dual group sampler tests."""

    def __init__(self, samples):
        self._s = samples

    def __len__(self):
        return len(self._s)

    def __getitem__(self, idx):
        return self._s[idx]


class DS2:
    """Another dataset variant."""

    def __init__(self, sizes):
        self.sizes = sizes

    def __len__(self):
        return len(self.sizes)

    def __getitem__(self, idx):
        nodes, edges = self.sizes[idx]
        return {"frag_pyg": SimpleNamespace(num_nodes=nodes, num_edges=edges)}


# ============================================================================
# Tests for SpecMolFragDynamicBatchSampler
# ============================================================================


class TestSpecMolFragDynamicBatchSampler:
    """Tests for dynamic batch sampling with node/edge constraints."""

    def test_dynamic_batch_sampler(self):
        """Test basic dynamic batch sampler functionality."""
        sizes = [(1, 2), (3, 4), (2, 1), (8, 2), (5, 6), (2, 2), (4, 1), (6, 6)]
        ds = FakeDataset(sizes)
        sampler = SpecMolFragDynamicBatchSampler(data_source=ds, max_num=10, limited_by="frag_node")
        dl = DataLoader(ds, batch_sampler=sampler, collate_fn=lambda x: x)
        batches = list(dl)
        assert len(batches) > 0

    def test_dynamic_batch_sampler_basic(self):
        """Test batches respect max_num constraint."""
        data_list = [
            {"idx": i, "frag_pyg": MockFrag(100 - i * 5, 12000 - i * 1000)} for i in range(12)
        ]
        ds = SyntheticDataset(data_list)

        sampler = SpecMolFragDynamicBatchSampler(
            data_source=ds, max_num=50000, limited_by="frag_edge", skip_too_big=False, sampler=None
        )

        batches = list(sampler)

        # Verify batches respect max_num constraint
        batch_edges = [sum(ds[idx]["frag_pyg"].num_edges for idx in batch) for batch in batches]
        assert all(e <= 50000 for e in batch_edges), "Some batches exceed max_num limit"

        # Verify no samples are duplicated or missing
        all_indices = [idx for batch in batches for idx in batch]
        expected_indices = set(range(len(ds)))
        actual_indices = set(all_indices)
        assert actual_indices == expected_indices

    def test_dynamic_batch_sampler_epoch_aware(self):
        """Test epoch setting affects batching order."""
        data_list = [
            {"idx": i, "frag_pyg": MockFrag(100 - i * 5, 12000 - i * 1000)} for i in range(12)
        ]
        ds = SyntheticDataset(data_list)

        generator = th.Generator()
        base_sampler = RandomSampler(ds, generator=generator)

        sampler = SpecMolFragDynamicBatchSampler(
            data_source=ds,
            max_num=50000,
            limited_by="frag_edge",
            skip_too_big=False,
            sampler=base_sampler,
        )

        epoch0_batches = list(sampler)
        sampler.set_epoch(1)
        epoch1_batches = list(sampler)

        assert len(epoch0_batches) > 0
        assert len(epoch1_batches) > 0

        # Setting same epoch should be deterministic
        sampler.set_epoch(0)
        epoch0_again = list(sampler)
        assert epoch0_batches == epoch0_again

    def test_dynamic_batch_sampler_no_base_sampler(self):
        """Test sampler works without base sampler."""
        data_list = [
            {"idx": i, "frag_pyg": MockFrag(100 - i * 5, 12000 - i * 1000)} for i in range(12)
        ]
        ds = SyntheticDataset(data_list)

        sampler = SpecMolFragDynamicBatchSampler(
            data_source=ds, max_num=50000, limited_by="frag_edge", skip_too_big=False, sampler=None
        )

        batches1 = list(sampler)
        batches2 = list(sampler)
        assert batches1 == batches2

    def test_dynamic_batch_sampler_skip_too_big(self):
        """Test skip_too_big flag."""
        data_list = [
            {"idx": 0, "frag_pyg": MockFrag(100, 12000)},
            {"idx": 1, "frag_pyg": MockFrag(100, 60000)},  # Too big
            {"idx": 2, "frag_pyg": MockFrag(110, 13000)},
        ]
        ds = SyntheticDataset(data_list)

        # With skip_too_big=True
        sampler = SpecMolFragDynamicBatchSampler(
            data_source=ds, max_num=50000, limited_by="frag_edge", skip_too_big=True
        )
        batches = list(sampler)
        all_indices = [idx for batch in batches for idx in batch]
        assert 1 not in all_indices

        # With skip_too_big=False
        sampler = SpecMolFragDynamicBatchSampler(
            data_source=ds, max_num=50000, limited_by="frag_edge", skip_too_big=False
        )
        batches = list(sampler)
        all_indices = [idx for batch in batches for idx in batch]
        assert 1 in all_indices

    def test_specmolfrag_dynamic_batch_nodes_limit_skip_true(self):
        """Test nodes limit with skip_too_big=True."""
        sizes = [(3, 1), (4, 2), (12, 4), (2, 1)]  # index 2 is too large
        ds = DS2(sizes)
        sampler = SpecMolFragDynamicBatchSampler(
            ds, max_num=10, limited_by="frag_node", skip_too_big=True
        )
        batches = list(sampler)
        all_indices = [i for b in batches for i in b]
        assert 2 not in all_indices
        assert all(sum(ds[i]["frag_pyg"].num_nodes for i in b) <= 10 for b in batches)

    def test_specmolfrag_dynamic_batch_nodes_limit_skip_false(self):
        """Test nodes limit with skip_too_big=False allows overflow."""
        sizes = [(3, 1), (4, 2), (12, 4), (2, 1)]
        ds = DS2(sizes)
        sampler = SpecMolFragDynamicBatchSampler(
            ds, max_num=10, limited_by="frag_node", skip_too_big=False
        )
        batches = list(sampler)
        all_indices = [i for b in batches for i in b]
        # Index 2 should be included despite exceeding budget
        assert 2 in all_indices


# ============================================================================
# Tests for DualGroupDynamicBatchSampler
# ============================================================================


class TestDualGroupDynamicBatchSampler:
    """Tests for dual-group dynamic batch sampling."""

    def test_dual_group_dynamic_batch_sampler_nodes_limit(self):
        """Test node budget constraint."""
        samples = [
            {"frag_pyg": FakeFrag(3, 2), "prec_type": "A", "mol_id": 1, "formula": "X"},
            {"frag_pyg": FakeFrag(2, 1), "prec_type": "A", "mol_id": 1, "formula": "Y"},
            {"frag_pyg": FakeFrag(4, 3), "prec_type": "A", "mol_id": 2, "formula": "X"},
            {"frag_pyg": FakeFrag(1, 1), "prec_type": "B", "mol_id": 3, "formula": "Z"},
            {"frag_pyg": FakeFrag(5, 4), "prec_type": "B", "mol_id": 3, "formula": "Z"},
            {"frag_pyg": FakeFrag(2, 1), "prec_type": "C", "mol_id": 4, "formula": "W"},
        ]

        ds = DummyDataset(samples)

        gen = th.Generator()
        gen.manual_seed(42)

        sampler = DualGroupDynamicBatchSampler(
            ds,
            max_num=6,
            limited_by="frag_node",
            skip_too_big=True,
            generator=gen,
        )

        batches = list(iter(sampler))

        # Expect at least one batch and all indices included
        assert len(batches) > 0
        included = set(i for b in batches for i in b)
        assert included == set(range(len(ds)))

        # Each batch must respect node budget
        for b in batches:
            total_nodes = sum(samples[i]["frag_pyg"].num_nodes for i in b)
            assert total_nodes <= 6

    def test_dual_group_dynamic_batch_sampler_edges_limit(self):
        """Test edge budget constraint."""
        samples = [
            {"frag_pyg": FakeFrag(3, 5), "prec_type": "A", "mol_id": 1, "formula": "X"},
            {"frag_pyg": FakeFrag(2, 3), "prec_type": "A", "mol_id": 1, "formula": "Y"},
            {"frag_pyg": FakeFrag(4, 7), "prec_type": "A", "mol_id": 2, "formula": "X"},
            {"frag_pyg": FakeFrag(1, 2), "prec_type": "B", "mol_id": 3, "formula": "Z"},
        ]

        ds = DummyDataset(samples)
        gen = th.Generator()
        gen.manual_seed(42)

        sampler = DualGroupDynamicBatchSampler(
            ds,
            max_num=10,
            limited_by="frag_edge",
            skip_too_big=False,
            generator=gen,
        )

        batches = list(iter(sampler))

        # Each batch must respect edge budget
        for b in batches:
            total_edges = sum(samples[i]["frag_pyg"].num_edges for i in b)
            assert total_edges <= 10

    def test_group2_retains_all_indices(self):
        """group2 must retain all indices including those also in group1.

        Every sample is in group1 (keyed by adduct::mol_id). The old dedup
        logic wiped group2 entirely because group1_indices == all indices.
        Within-batch dedup is handled by used_mask in _sample_from_group_dict,
        so group2 should not be permanently pruned.
        """
        # Two isomers of formula X under adduct A (different mol_ids)
        samples = [
            {"frag_pyg": FakeFrag(3, 2), "prec_type": "A", "mol_id": 1, "formula": "X"},
            {"frag_pyg": FakeFrag(2, 1), "prec_type": "A", "mol_id": 2, "formula": "X"},
            {"frag_pyg": FakeFrag(4, 3), "prec_type": "B", "mol_id": 5, "formula": "Y"},
            {"frag_pyg": FakeFrag(1, 1), "prec_type": "C", "mol_id": 3, "formula": "Z"},
        ]

        ds = DummyDataset(samples)
        gen = th.Generator()
        gen.manual_seed(42)

        sampler = DualGroupDynamicBatchSampler(
            ds,
            max_num=10,
            limited_by="frag_node",
            generator=gen,
        )

        # group2 must not be empty: A::X has two isomers (idx 0 and 1)
        assert len(sampler._group2) > 0
        # A::X group in group2 must contain both isomer indices
        assert set(sampler._group2["A::X"].tolist()) == {0, 1}
        # group2 coverage: every index must appear in at least one group2 entry
        group2_all = set(i for t in sampler._group2.values() for i in t.tolist())
        assert group2_all == set(range(len(samples)))

    def test_no_duplicate_indices_within_batch(self):
        """used_mask prevents the same index from appearing twice in one batch."""
        # Two isomers share formula X; without within-batch dedup they could both
        # be picked by group1 and group2 in the same batch.
        samples = [
            {"frag_pyg": FakeFrag(1, 1), "prec_type": "A", "mol_id": 1, "formula": "X"},
            {"frag_pyg": FakeFrag(1, 1), "prec_type": "A", "mol_id": 2, "formula": "X"},
            {"frag_pyg": FakeFrag(1, 1), "prec_type": "A", "mol_id": 3, "formula": "X"},
        ]

        ds = DummyDataset(samples)
        gen = th.Generator()
        gen.manual_seed(0)

        sampler = DualGroupDynamicBatchSampler(
            ds,
            max_num=10,
            limited_by="frag_node",
            generator=gen,
        )

        for batch in sampler:
            assert len(batch) == len(set(batch)), f"Duplicate indices in batch: {batch}"

    def test_every_index_covered_exactly_once(self):
        """Every dataset index must appear in exactly one batch per epoch."""
        samples = [
            {"frag_pyg": FakeFrag(1, 1), "prec_type": "A", "mol_id": i, "formula": f"F{i % 3}"}
            for i in range(10)
        ]
        ds = DummyDataset(samples)
        gen = th.Generator()
        gen.manual_seed(7)

        sampler = DualGroupDynamicBatchSampler(ds, max_num=20, limited_by="frag_node", generator=gen)
        all_indices = [i for b in sampler for i in b]

        assert sorted(all_indices) == list(range(len(samples)))

    def test_isomer_contrast_in_same_batch(self):
        """With a large budget, two isomers (same formula, different mol_id) should
        land in the same batch: one via group1 (adduct::mol_id) and one via group2
        (adduct::formula)."""
        # idx 0 and 1 are isomers of A::X; budget fits both easily
        samples = [
            {"frag_pyg": FakeFrag(1, 1), "prec_type": "A", "mol_id": 1, "formula": "X"},
            {"frag_pyg": FakeFrag(1, 1), "prec_type": "A", "mol_id": 2, "formula": "X"},
        ]
        ds = DummyDataset(samples)
        gen = th.Generator()
        gen.manual_seed(0)

        sampler = DualGroupDynamicBatchSampler(
            ds, max_num=10, limited_by="frag_node", sample_k1=1, sample_k2=1, generator=gen
        )
        batches = list(sampler)

        # Both indices should end up in the same single batch
        assert len(batches) == 1
        assert set(batches[0]) == {0, 1}

    def test_sample_k1_k2_limits(self):
        """sample_k1=1 sample_k2=1 means at most 2 group-guided picks per batch,
        with the remainder filled freely."""
        # 3 isomers of A::X; with k1=1, k2=1 and a tight budget we expect
        # at most 2 group-guided picks per batch
        samples = [
            {"frag_pyg": FakeFrag(2, 1), "prec_type": "A", "mol_id": i, "formula": "X"}
            for i in range(6)
        ]
        ds = DummyDataset(samples)
        gen = th.Generator()
        gen.manual_seed(1)

        sampler = DualGroupDynamicBatchSampler(
            ds, max_num=4, limited_by="frag_node", sample_k1=1, sample_k2=1, generator=gen
        )
        # With budget=4 and each sample costing 2 nodes, each batch holds at most 2 samples
        for batch in sampler:
            assert len(batch) <= 2

    def test_adduct_scopes_formula_groups(self):
        """Same formula under different adducts must be separate group2 entries."""
        samples = [
            {"frag_pyg": FakeFrag(1, 1), "prec_type": "A", "mol_id": 1, "formula": "X"},
            {"frag_pyg": FakeFrag(1, 1), "prec_type": "B", "mol_id": 2, "formula": "X"},
        ]
        ds = DummyDataset(samples)
        gen = th.Generator()
        gen.manual_seed(0)

        sampler = DualGroupDynamicBatchSampler(ds, max_num=10, limited_by="frag_node", generator=gen)

        assert "A::X" in sampler._group2
        assert "B::X" in sampler._group2
        # Each adduct group should contain only its own index
        assert set(sampler._group2["A::X"].tolist()) == {0}
        assert set(sampler._group2["B::X"].tolist()) == {1}

    def test_single_molecule_formula_still_in_group2(self):
        """A formula with only one molecule must still appear in group2.
        The old dedup bug would delete it because its single index was in group1.
        """
        samples = [
            {"frag_pyg": FakeFrag(1, 1), "prec_type": "A", "mol_id": 1, "formula": "UNIQUE"},
            {"frag_pyg": FakeFrag(1, 1), "prec_type": "A", "mol_id": 2, "formula": "COMMON"},
            {"frag_pyg": FakeFrag(1, 1), "prec_type": "A", "mol_id": 3, "formula": "COMMON"},
        ]
        ds = DummyDataset(samples)
        gen = th.Generator()
        gen.manual_seed(0)

        sampler = DualGroupDynamicBatchSampler(ds, max_num=10, limited_by="frag_node", generator=gen)

        assert "A::UNIQUE" in sampler._group2
        assert set(sampler._group2["A::UNIQUE"].tolist()) == {0}

    def test_missing_required_key_raises(self):
        """Missing adduct/mol_id/formula keys must raise KeyError at init."""
        samples = [
            {"frag_pyg": FakeFrag(1, 1), "prec_type": "A", "mol_id": 1},  # missing formula
        ]
        ds = DummyDataset(samples)
        with pytest.raises(KeyError):
            DualGroupDynamicBatchSampler(ds, max_num=10, limited_by="frag_node")

    def test_invalid_limited_by_raises(self):
        """Invalid limited_by value must raise ValueError."""
        samples = [
            {"frag_pyg": FakeFrag(1, 1), "prec_type": "A", "mol_id": 1, "formula": "X"},
        ]
        ds = DummyDataset(samples)
        with pytest.raises(ValueError, match="limited_by"):
            DualGroupDynamicBatchSampler(ds, max_num=10, limited_by="invalid")

    def test_dual_group_sampler_skip_too_big(self):
        """Test skip_too_big flag."""
        samples = [
            {"frag_pyg": FakeFrag(3, 2), "prec_type": "A", "mol_id": 1, "formula": "X"},
            {"frag_pyg": FakeFrag(100, 1), "prec_type": "B", "mol_id": 2, "formula": "Y"},
        ]

        ds = DummyDataset(samples)
        gen = th.Generator()
        gen.manual_seed(42)

        sampler = DualGroupDynamicBatchSampler(
            ds,
            max_num=10,
            limited_by="frag_node",
            skip_too_big=True,
            generator=gen,
        )

        batches = list(iter(sampler))
        all_indices = [i for b in batches for i in b]
        assert 1 not in all_indices

    def test_dual_group_sampler_skip_false_keeps_oversized(self):
        """When skip_too_big=False, oversized samples should still be yielded."""
        samples = [
            {"frag_pyg": FakeFrag(3, 2), "prec_type": "A", "mol_id": 1, "formula": "X"},
            {"frag_pyg": FakeFrag(100, 1), "prec_type": "B", "mol_id": 2, "formula": "Y"},
        ]

        ds = DummyDataset(samples)
        gen = th.Generator()
        gen.manual_seed(42)

        sampler = DualGroupDynamicBatchSampler(
            ds,
            max_num=10,
            limited_by="frag_node",
            skip_too_big=False,
            generator=gen,
        )

        batches = list(iter(sampler))
        all_indices = [i for b in batches for i in b]
        assert 1 in all_indices

    def test_dual_group_sampler_deterministic_with_generator(self):
        """Test determinism with manual seed."""
        samples = [
            {"frag_pyg": FakeFrag(3, 2), "prec_type": "A", "mol_id": 1, "formula": "X"},
            {"frag_pyg": FakeFrag(2, 1), "prec_type": "A", "mol_id": 1, "formula": "Y"},
            {"frag_pyg": FakeFrag(4, 3), "prec_type": "B", "mol_id": 2, "formula": "X"},
        ]

        ds = DummyDataset(samples)

        g1 = th.Generator()
        g1.manual_seed(123)
        s1 = DualGroupDynamicBatchSampler(ds, max_num=10, generator=g1)
        out1 = list(iter(s1))

        g2 = th.Generator()
        g2.manual_seed(123)
        s2 = DualGroupDynamicBatchSampler(ds, max_num=10, generator=g2)
        out2 = list(iter(s2))

        assert out1 == out2

    def test_dual_group_sampler_set_epoch(self):
        """Test set_epoch changes order."""
        samples = [
            {"frag_pyg": FakeFrag(3, 2), "prec_type": "A", "mol_id": 1, "formula": "X"},
            {"frag_pyg": FakeFrag(2, 1), "prec_type": "A", "mol_id": 1, "formula": "Y"},
            {"frag_pyg": FakeFrag(4, 3), "prec_type": "B", "mol_id": 2, "formula": "X"},
        ]

        ds = DummyDataset(samples)
        gen = th.Generator()
        gen.manual_seed(99)
        sampler = DualGroupDynamicBatchSampler(ds, max_num=10, generator=gen)

        sampler.set_epoch(0)
        out0 = list(iter(sampler))

        sampler.set_epoch(1)
        out1 = list(iter(sampler))

        # Different epochs should produce different orders
        assert out0 != out1

    def test_k1_per_key_places_multiple_same_mol_in_batch(self):
        """With k1_per_key > 1, multiple spectra of the same molecule should
        appear in the same batch, providing same-molecule pairs for pairwise loss."""
        # Molecule 1 has 5 spectra (different CEs), molecule 2 has 1.
        # With k1_per_key=3 and a large budget, 3 spectra of mol 1 must land together.
        samples = [
            {"frag_pyg": FakeFrag(1, 1), "prec_type": "A", "mol_id": 1, "formula": "X"},
            {"frag_pyg": FakeFrag(1, 1), "prec_type": "A", "mol_id": 1, "formula": "X"},
            {"frag_pyg": FakeFrag(1, 1), "prec_type": "A", "mol_id": 1, "formula": "X"},
            {"frag_pyg": FakeFrag(1, 1), "prec_type": "A", "mol_id": 1, "formula": "X"},
            {"frag_pyg": FakeFrag(1, 1), "prec_type": "A", "mol_id": 1, "formula": "X"},
            {"frag_pyg": FakeFrag(1, 1), "prec_type": "B", "mol_id": 2, "formula": "Y"},
        ]
        mol1_indices = {0, 1, 2, 3, 4}

        ds = DummyDataset(samples)
        gen = th.Generator()
        gen.manual_seed(0)

        sampler = DualGroupDynamicBatchSampler(
            ds,
            max_num=100,
            limited_by="frag_node",
            sample_k1=1,
            sample_k1_per_key=3,
            generator=gen,
        )
        batches = list(sampler)

        # At least one batch must contain ≥ 2 mol-1 spectra (i.e., a same-mol pair)
        max_mol1_in_batch = max(len(set(b) & mol1_indices) for b in batches)
        assert max_mol1_in_batch >= 2, (
            f"Expected ≥ 2 same-molecule spectra in a batch, got max {max_mol1_in_batch}"
        )

    def test_k1_per_key_default_one_unchanged(self):
        """Default k1_per_key=1 behaviour must be identical to the old code:
        at most 1 sample per group1 key per group pass."""
        samples = [
            {"frag_pyg": FakeFrag(1, 1), "prec_type": "A", "mol_id": 1, "formula": "X"},
            {"frag_pyg": FakeFrag(1, 1), "prec_type": "A", "mol_id": 1, "formula": "X"},
            {"frag_pyg": FakeFrag(1, 1), "prec_type": "A", "mol_id": 1, "formula": "X"},
            {"frag_pyg": FakeFrag(1, 1), "prec_type": "B", "mol_id": 2, "formula": "Y"},
        ]
        ds = DummyDataset(samples)
        gen = th.Generator()
        gen.manual_seed(5)

        sampler = DualGroupDynamicBatchSampler(
            ds,
            max_num=1,  # budget = 1 node → only 1 sample per batch
            limited_by="frag_node",
            sample_k1=1,
            sample_k1_per_key=1,  # explicit default
            generator=gen,
        )
        # With budget 1 each batch can hold exactly 1 sample
        for batch in sampler:
            assert len(batch) == 1

    def test_every_index_covered_with_per_key(self):
        """All samples must be covered exactly once even with per_key > 1."""
        samples = [
            {"frag_pyg": FakeFrag(1, 1), "prec_type": "A", "mol_id": i // 3, "formula": f"F{i // 3}"}
            for i in range(9)
        ]
        ds = DummyDataset(samples)
        gen = th.Generator()
        gen.manual_seed(42)

        sampler = DualGroupDynamicBatchSampler(
            ds,
            max_num=100,
            limited_by="frag_node",
            sample_k1=5,
            sample_k1_per_key=3,
            generator=gen,
        )
        all_indices = [i for b in sampler for i in b]
        assert sorted(all_indices) == list(range(len(samples)))


# ============================================================================
# Tests for FormulaPrecTypeGroupSampler
# ============================================================================


class TestFormulaPrecTypeGroupSampler:
    """Tests for formula and precursor type grouping."""

    def test_grouping_basic(self):
        """Test basic grouping by formula and prec_type."""
        items = []
        # cluster A: prec_type 'A', formula 'F1'
        for i in range(3):
            items.append(
                {
                    "frag_pyg": MockFrag(nodes=1 + i, edges=0),
                    "formula": "F1",
                    "prec_type": "A",
                    "mol_id": f"m{i}",
                    "group_id": th.tensor([0]),
                }
            )
        # cluster B: prec_type 'B', formula 'F2'
        for i in range(2):
            items.append(
                {
                    "frag_pyg": MockFrag(nodes=2 + i, edges=0),
                    "formula": "F2",
                    "prec_type": "B",
                    "mol_id": f"m{3 + i}",
                    "group_id": th.tensor([1]),
                }
            )
        # single cluster C
        items.append(
            {
                "frag_pyg": MockFrag(nodes=4, edges=0),
                "formula": "F3",
                "prec_type": "C",
                "mol_id": "m5",
                "group_id": th.tensor([2]),
            }
        )

        ds = SyntheticDataset(items)
        gen = th.Generator()
        gen.manual_seed(42)
        sampler = FormulaPrecTypeGroupSampler(ds, sample_k=1, generator=gen)

        counts = {}
        for idx in sampler:
            item = ds[idx]
            key = f"{item['prec_type']}::{item['formula']}"
            counts[key] = counts.get(key, 0) + 1

        # Each key yields at most sample_k
        assert all(c <= 1 for c in counts.values())

    def test_integration_with_dynamic_batch_sampler(self):
        """Test integration with SpecMolFragDynamicBatchSampler."""
        items = []
        for i in range(3):
            items.append(
                {
                    "frag_pyg": MockFrag(nodes=1 + i, edges=0),
                    "formula": "F1",
                    "prec_type": "A",
                    "mol_id": f"m{i}",
                    "group_id": th.tensor([0]),
                }
            )
        for i in range(2):
            items.append(
                {
                    "frag_pyg": MockFrag(nodes=2 + i, edges=0),
                    "formula": "F2",
                    "prec_type": "B",
                    "mol_id": f"m{3 + i}",
                    "group_id": th.tensor([1]),
                }
            )

        ds = SyntheticDataset(items)
        gen = th.Generator()
        gen.manual_seed(123)
        sampler = FormulaPrecTypeGroupSampler(ds, sample_k=2, generator=gen)

        batch_sampler = SpecMolFragDynamicBatchSampler(
            data_source=ds, max_num=6, limited_by="frag_node", sampler=sampler
        )
        batch_sampler.set_epoch(0)

        underlying = set(list(sampler))

        for batch in batch_sampler:
            total_nodes = sum(ds[idx]["frag_pyg"].num_nodes for idx in batch)
            assert total_nodes <= 6
            for idx in batch:
                assert idx in underlying

    def test_set_epoch_and_determinism(self):
        """Test determinism with set_epoch."""
        items = []
        for i in range(3):
            items.append(
                {
                    "frag_pyg": MockFrag(nodes=1 + i, edges=0),
                    "formula": "F1",
                    "prec_type": "A",
                    "mol_id": f"m{i}",
                    "group_id": th.tensor([0]),
                }
            )

        ds = SyntheticDataset(items)

        g1 = th.Generator()
        g1.manual_seed(999)
        s1 = FormulaPrecTypeGroupSampler(ds, sample_k=2, generator=g1)
        b1 = SpecMolFragDynamicBatchSampler(ds, max_num=9999, sampler=s1)
        b1.set_epoch(2)
        out1 = list(b1)

        g2 = th.Generator()
        g2.manual_seed(999)
        s2 = FormulaPrecTypeGroupSampler(ds, sample_k=2, generator=g2)
        b2 = SpecMolFragDynamicBatchSampler(ds, max_num=9999, sampler=s2)
        b2.set_epoch(2)
        out2 = list(b2)

        assert out1 == out2

        # Changing epoch should change ordering
        b2.set_epoch(3)
        out3 = list(b2)
        assert out1 != out3

    def test_missing_keys_raises(self):
        """Test that missing keys raise KeyError."""
        items = [
            {
                "frag_pyg": MockFrag(1),
                "formula": "F1",
                "mol_id": "m0",
                "group_id": th.tensor([0]),
            }  # missing 'prec_type'
        ]
        ds = SyntheticDataset(items)
        with pytest.raises(KeyError):
            FormulaPrecTypeGroupSampler(ds)


# ============================================================================
# Tests for GroupSampler
# ============================================================================


class TestGroupSampler:
    """Tests for basic group sampling."""

    def test_group_sampler_basic_and_epoch(self):
        """Test basic GroupSampler functionality."""
        items = []
        sizes = [4, 2, 1]
        for g, size in enumerate(sizes):
            for i in range(size):
                items.append(
                    {
                        "frag_pyg": MockFrag(1 + i),
                        "group_id": th.tensor([g]),
                    }
                )

        ds = SyntheticDataset(items)
        gen = th.Generator()
        gen.manual_seed(2024)
        sampler = GroupSampler(ds, sample_k=2, generator=gen)

        # Count samples per group
        counts = {}
        for idx in sampler:
            gid = ds[idx]["group_id"].item()
            counts[gid] = counts.get(gid, 0) + 1

        # Each group yields at most sample_k
        assert all(counts.get(g, 0) <= 2 for g in range(len(sizes)))

        expected_total = sum(min(s, 2) for s in sizes)
        assert len(sampler) == expected_total

    def test_group_sampler_determinism(self):
        """Test determinism across same seed + epoch."""
        items = []
        sizes = [3, 2]
        for g, size in enumerate(sizes):
            for i in range(size):
                items.append(
                    {
                        "frag_pyg": MockFrag(1 + i),
                        "group_id": th.tensor([g]),
                    }
                )

        ds = SyntheticDataset(items)

        g1 = th.Generator()
        g1.manual_seed(7)
        s1 = GroupSampler(ds, sample_k=2, generator=g1)
        s1.set_epoch(1)
        out1 = list(s1)

        g2 = th.Generator()
        g2.manual_seed(7)
        s2 = GroupSampler(ds, sample_k=2, generator=g2)
        s2.set_epoch(1)
        out2 = list(s2)

        assert out1 == out2


# ============================================================================
# Smoke Tests
# ============================================================================


class TestDataloaderSmoke:
    """Smoke tests for dataloader functionality."""

    def test_dynamic_batch_sampler_smoke(self):
        """Smoke test for dynamic batch sampler with dataloader."""
        sizes = [(1, 2), (3, 4), (2, 1), (8, 2), (5, 6), (2, 2), (4, 1), (6, 6)]
        ds = FakeDataset(sizes)
        sampler = SpecMolFragDynamicBatchSampler(data_source=ds, max_num=10, limited_by="frag_node")
        dl = DataLoader(ds, batch_sampler=sampler, collate_fn=lambda x: x)
        batches = list(dl)
        assert len(batches) > 0

    def test_dual_group_sampler_smoke(self):
        """Smoke test for dual group sampler."""
        samples = [
            {
                "frag_pyg": SimpleNamespace(num_nodes=3, num_edges=2),
                "prec_type": "A",
                "mol_id": 1,
                "formula": "F1",
            },
            {
                "frag_pyg": SimpleNamespace(num_nodes=2, num_edges=1),
                "prec_type": "A",
                "mol_id": 1,
                "formula": "F2",
            },
            {
                "frag_pyg": SimpleNamespace(num_nodes=4, num_edges=3),
                "prec_type": "A",
                "mol_id": 2,
                "formula": "F1",
            },
            {
                "frag_pyg": SimpleNamespace(num_nodes=1, num_edges=1),
                "prec_type": "B",
                "mol_id": 3,
                "formula": "F3",
            },
            {
                "frag_pyg": SimpleNamespace(num_nodes=5, num_edges=4),
                "prec_type": "B",
                "mol_id": 3,
                "formula": "F3",
            },
            {
                "frag_pyg": SimpleNamespace(num_nodes=2, num_edges=1),
                "prec_type": "C",
                "mol_id": 4,
                "formula": "F4",
            },
        ]

        ds = DummyDataset(samples)
        gen = th.Generator()
        gen.manual_seed(123)

        sampler = DualGroupDynamicBatchSampler(
            ds, max_num=6, limited_by="frag_node", skip_too_big=True, generator=gen
        )
        batches = list(iter(sampler))
        assert len(batches) > 0
        included = set(i for b in batches for i in b)
        assert included == set(range(len(ds)))
        for b in batches:
            total_nodes = sum(samples[i]["frag_pyg"].num_nodes for i in b)
            assert total_nodes <= 6


# ============================================================================
# Tests for GreedyInferenceBatchSampler
# ============================================================================


class TestGreedyInferenceBatchSampler:
    """Tests for greedy sequential inference batch sampling."""

    def _make_ds(self, sizes):
        """Build a FakeDataset from a list of (nodes, edges) tuples."""
        return FakeDataset(sizes)

    def test_all_indices_covered_exactly_once(self):
        """Every dataset index must appear in exactly one batch."""
        sizes = [(3, 2), (5, 4), (1, 1), (8, 6), (2, 2), (4, 3)]
        ds = self._make_ds(sizes)
        sampler = GreedyInferenceBatchSampler(ds, max_num=10, limited_by="frag_edge")
        all_indices = [i for batch in sampler for i in batch]
        assert sorted(all_indices) == list(range(len(sizes)))

    def test_sequential_order_preserved(self):
        """Indices must appear in ascending sequential order across batches."""
        sizes = [(2, 1)] * 8
        ds = self._make_ds(sizes)
        sampler = GreedyInferenceBatchSampler(ds, max_num=6, limited_by="frag_edge")
        all_indices = [i for batch in sampler for i in batch]
        assert all_indices == sorted(all_indices)

    def test_batch_edge_budget_respected(self):
        """No batch should exceed max_num edges (except oversized singletons)."""
        sizes = [(1, 3), (1, 4), (1, 5), (1, 2), (1, 3)]
        ds = self._make_ds(sizes)
        max_num = 7
        sampler = GreedyInferenceBatchSampler(ds, max_num=max_num, limited_by="frag_edge")
        for batch in sampler:
            if len(batch) > 1:
                total = sum(sizes[i][1] for i in batch)
                assert total <= max_num

    def test_batch_node_budget_respected(self):
        """No batch should exceed max_num nodes (except oversized singletons)."""
        sizes = [(3, 1), (4, 2), (2, 1), (5, 3), (3, 2)]
        ds = self._make_ds(sizes)
        max_num = 8
        sampler = GreedyInferenceBatchSampler(ds, max_num=max_num, limited_by="frag_node")
        for batch in sampler:
            if len(batch) > 1:
                total = sum(sizes[i][0] for i in batch)
                assert total <= max_num

    def test_oversized_singleton_emitted(self):
        """A sample larger than max_num must be emitted as a singleton, not dropped."""
        sizes = [(2, 2), (1, 100), (3, 3)]  # index 1 is oversized
        ds = self._make_ds(sizes)
        sampler = GreedyInferenceBatchSampler(ds, max_num=10, limited_by="frag_edge")
        all_indices = [i for batch in sampler for i in batch]
        assert 1 in all_indices

    def test_len_matches_iteration(self):
        """__len__ must equal the number of batches yielded by iteration."""
        sizes = [(3, 2), (5, 4), (1, 1), (8, 6), (2, 2), (4, 3)]
        ds = self._make_ds(sizes)
        sampler = GreedyInferenceBatchSampler(ds, max_num=8, limited_by="frag_edge")
        assert len(sampler) == len(list(sampler))

    def test_deterministic_across_iterations(self):
        """Multiple iterations must yield identical batches."""
        sizes = [(3, 2), (5, 4), (1, 1), (8, 6), (2, 2)]
        ds = self._make_ds(sizes)
        sampler = GreedyInferenceBatchSampler(ds, max_num=8, limited_by="frag_edge")
        assert list(sampler) == list(sampler)

    def test_precom_meta_info_cache_used(self):
        """When dataset exposes precom_meta_info, no __getitem__ calls should be needed."""

        class CachedDataset:
            def __init__(self, meta):
                self.precom_meta_info = meta

            def __len__(self):
                return len(self.precom_meta_info)

            def __getitem__(self, idx):
                raise AssertionError("__getitem__ must not be called when cache is present")

        meta = [(1, 2), (3, 4), (2, 1), (5, 6)]
        ds = CachedDataset(meta)
        sampler = GreedyInferenceBatchSampler(ds, max_num=5, limited_by="frag_edge")
        all_indices = [i for batch in sampler for i in batch]
        assert sorted(all_indices) == list(range(len(meta)))

    def test_invalid_max_num_raises(self):
        """Non-positive max_num must raise ValueError."""
        ds = self._make_ds([(1, 1)])
        with pytest.raises(ValueError, match="max_num"):
            GreedyInferenceBatchSampler(ds, max_num=0)

    def test_invalid_limited_by_raises(self):
        """Unknown limited_by value must raise ValueError."""
        ds = self._make_ds([(1, 1)])
        with pytest.raises(ValueError, match="limited_by"):
            GreedyInferenceBatchSampler(ds, max_num=10, limited_by="frag_weight")

    def test_single_sample_dataset(self):
        """A single-sample dataset must produce exactly one batch with that index."""
        ds = self._make_ds([(5, 3)])
        sampler = GreedyInferenceBatchSampler(ds, max_num=10, limited_by="frag_edge")
        batches = list(sampler)
        assert batches == [[0]]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
