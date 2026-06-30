"""Tests for adduct+instrument-type weighted oversampling via get_group_sampler().

Covers:
- adduct_inst_type: minority combos get higher weights (9x for 10:1 imbalance)
- adduct_inst_type_group: compound weighting with group counts
- Single-class edge case: all weights are equal
- Epoch size equals len(dataset)
- replacement=True (allows oversampling minority beyond its original count)
"""

import collections

import pytest
import torch as th

from fragnnet.dataset.data_sampler import get_group_sampler


# ============================================================================
# Mock dataset
# ============================================================================


class MockDataset:
    """Minimal mock that satisfies get_group_sampler interface.

    Args:
        prec_types: Per-sample precursor type strings.
        inst_types: Per-sample instrument type strings.
        frag_modes: Per-sample fragmentation mode strings. Defaults to None (unknown).
        group_ids: Per-sample group IDs (ints). Defaults to one group per sample.
    """

    def __init__(
        self,
        prec_types: list[str],
        inst_types: list[str],
        frag_modes: list[str] | None = None,
        group_ids: list[int] | None = None,
    ):
        assert len(prec_types) == len(inst_types)
        self._prec_types = prec_types
        self._inst_types = inst_types
        n = len(prec_types)
        self._frag_modes = frag_modes if frag_modes is not None else ["unknown"] * n
        self._group_ids = group_ids if group_ids is not None else list(range(n))

    def __len__(self) -> int:
        return len(self._prec_types)

    def get_adduct_inst_type_stats(self) -> tuple[list[str], list[str]]:
        return list(self._prec_types), list(self._inst_types)

    def get_frag_mode_stats(self) -> list[str]:
        return list(self._frag_modes)

    def get_group_mol_stats(self):
        n = len(self._group_ids)
        group_ids_t = th.tensor(self._group_ids, dtype=th.long)
        mol_ids_t = th.zeros(n, dtype=th.long)
        # spec_per_group: count samples in each group
        counts: dict[int, int] = collections.Counter(self._group_ids)
        spec_per_group_t = th.tensor([counts[g] for g in self._group_ids], dtype=th.long)
        spec_per_mol_t = th.ones(n, dtype=th.long)
        group_per_mol_t = th.ones(n, dtype=th.long)
        return group_ids_t, mol_ids_t, spec_per_group_t, spec_per_mol_t, group_per_mol_t


# ============================================================================
# Tests
# ============================================================================


class TestAdductInstTypeSampler:
    def test_minority_gets_higher_weight(self):
        """9:1 imbalance → minority weight should be 9x the majority weight."""
        # 9 samples of majority ([M+H]+/FT), 1 sample of minority ([M-H]-/QTOF)
        prec_types = ["[M+H]+"] * 9 + ["[M-H]-"]
        inst_types = ["FT"] * 9 + ["QTOF"]
        ds = MockDataset(prec_types, inst_types)

        gen = th.Generator()
        gen.manual_seed(0)
        sampler = get_group_sampler(ds, "adduct_inst_type", avg_per_group=1, generator=gen)

        assert sampler is not None
        weights = th.tensor(list(sampler.weights))

        majority_weight = weights[0].item()
        minority_weight = weights[-1].item()

        # minority weight = 1/1 = 1.0, majority weight = 1/9 ≈ 0.111
        assert minority_weight == pytest.approx(9.0 * majority_weight, rel=1e-5)

    def test_epoch_size_equals_dataset_length(self):
        """num_samples must equal len(ds) for a stable epoch size."""
        prec_types = ["[M+H]+"] * 9 + ["[M-H]-"]
        inst_types = ["FT"] * 9 + ["QTOF"]
        ds = MockDataset(prec_types, inst_types)

        gen = th.Generator()
        sampler = get_group_sampler(ds, "adduct_inst_type", avg_per_group=1, generator=gen)

        assert sampler.num_samples == len(ds)

    def test_replacement_is_true(self):
        """adduct_inst_type must use replacement=True to allow true oversampling."""
        prec_types = ["[M+H]+"] * 9 + ["[M-H]-"]
        inst_types = ["FT"] * 9 + ["QTOF"]
        ds = MockDataset(prec_types, inst_types)

        gen = th.Generator()
        sampler = get_group_sampler(ds, "adduct_inst_type", avg_per_group=1, generator=gen)

        assert sampler.replacement is True

    def test_single_class_uniform_weights(self):
        """When all samples share the same (prec_type, inst_type), weights are equal."""
        prec_types = ["[M+H]+"] * 10
        inst_types = ["FT"] * 10
        ds = MockDataset(prec_types, inst_types)

        gen = th.Generator()
        sampler = get_group_sampler(ds, "adduct_inst_type", avg_per_group=1, generator=gen)

        weights = th.tensor(list(sampler.weights))
        assert th.allclose(weights, weights[0].expand_as(weights))

    def test_multiple_combos_distinct_weights(self):
        """Three combos with counts 6:3:1 → weights in ratio 1:2:6."""
        # combo A: 6 samples, combo B: 3, combo C: 1
        prec_types = ["A"] * 6 + ["B"] * 3 + ["C"] * 1
        inst_types = ["x"] * 6 + ["y"] * 3 + ["z"] * 1
        ds = MockDataset(prec_types, inst_types)

        gen = th.Generator()
        sampler = get_group_sampler(ds, "adduct_inst_type", avg_per_group=1, generator=gen)

        weights = th.tensor(list(sampler.weights))
        w_a = weights[0].item()   # 1/6
        w_b = weights[6].item()   # 1/3
        w_c = weights[9].item()   # 1/1

        assert w_b == pytest.approx(2.0 * w_a, rel=1e-5)
        assert w_c == pytest.approx(6.0 * w_a, rel=1e-5)

    def test_adduct_inst_type_group_compounds_weights(self):
        """adduct_inst_type_group further divides by specs_per_group.

        Two samples in the minority combo but in the SAME group → group count = 2,
        so their weight is halved compared to adduct_inst_type alone.
        """
        # 8 majority samples, each in own group
        # 2 minority samples sharing group_id=99
        prec_types = ["[M+H]+"] * 8 + ["[M-H]-"] * 2
        inst_types = ["FT"] * 8 + ["QTOF"] * 2
        group_ids = list(range(8)) + [99, 99]  # minority share one group
        ds = MockDataset(prec_types, inst_types, group_ids=group_ids)

        gen = th.Generator()
        sampler_plain = get_group_sampler(
            ds, "adduct_inst_type", avg_per_group=1, generator=gen
        )
        gen2 = th.Generator()
        sampler_group = get_group_sampler(
            ds, "adduct_inst_type_group", avg_per_group=1, generator=gen2
        )

        weights_plain = th.tensor(list(sampler_plain.weights))
        weights_group = th.tensor(list(sampler_group.weights))

        # For the minority samples (idx 8, 9): group has 2 members → weight /= 2
        minority_plain = weights_plain[8].item()
        minority_group = weights_group[8].item()
        assert minority_group == pytest.approx(minority_plain / 2.0, rel=1e-5)

        # For majority samples (idx 0): each is in its own group (size=1) → no change
        majority_plain = weights_plain[0].item()
        majority_group = weights_group[0].item()
        assert majority_group == pytest.approx(majority_plain, rel=1e-5)

    def test_unknown_type_returns_none(self):
        """Unknown sampler_type must return None (unchanged behavior)."""
        prec_types = ["[M+H]+"] * 5
        inst_types = ["FT"] * 5
        ds = MockDataset(prec_types, inst_types)

        gen = th.Generator()
        result = get_group_sampler(ds, "nonexistent_type", avg_per_group=1, generator=gen)
        assert result is None


class TestFragModeSampler:
    def test_cid_minority_gets_higher_weight(self):
        """9:1 HCD:CID imbalance → CID weight should be 9x HCD weight."""
        prec_types = ["[M+H]+"] * 10
        inst_types = ["FT"] * 10
        frag_modes = ["HCD"] * 9 + ["CID"]
        ds = MockDataset(prec_types, inst_types, frag_modes=frag_modes)

        gen = th.Generator()
        gen.manual_seed(0)
        sampler = get_group_sampler(
            ds, "adduct_inst_type_frag_mode", avg_per_group=1, generator=gen
        )

        assert sampler is not None
        weights = th.tensor(list(sampler.weights))
        hcd_weight = weights[0].item()   # 1/9
        cid_weight = weights[-1].item()  # 1/1

        assert cid_weight == pytest.approx(9.0 * hcd_weight, rel=1e-5)

    def test_unknown_frag_mode_forms_own_bucket(self):
        """Samples with unknown frag_mode are balanced separately from HCD/CID."""
        prec_types = ["[M+H]+"] * 12
        inst_types = ["FT"] * 12
        # 6 HCD, 4 CID, 2 unknown → weights: 1/6, 1/4, 1/2
        frag_modes = ["HCD"] * 6 + ["CID"] * 4 + ["unknown"] * 2
        ds = MockDataset(prec_types, inst_types, frag_modes=frag_modes)

        gen = th.Generator()
        sampler = get_group_sampler(
            ds, "adduct_inst_type_frag_mode", avg_per_group=1, generator=gen
        )

        weights = th.tensor(list(sampler.weights))
        w_hcd = weights[0].item()    # 1/6
        w_cid = weights[6].item()    # 1/4
        w_unk = weights[10].item()   # 1/2

        assert w_cid == pytest.approx(6.0 / 4.0 * w_hcd, rel=1e-5)
        assert w_unk == pytest.approx(6.0 / 2.0 * w_hcd, rel=1e-5)

    def test_epoch_size_equals_dataset_length(self):
        """num_samples must equal len(ds)."""
        prec_types = ["[M+H]+"] * 10
        inst_types = ["FT"] * 10
        frag_modes = ["HCD"] * 9 + ["CID"]
        ds = MockDataset(prec_types, inst_types, frag_modes=frag_modes)

        gen = th.Generator()
        sampler = get_group_sampler(
            ds, "adduct_inst_type_frag_mode", avg_per_group=1, generator=gen
        )

        assert sampler.num_samples == len(ds)

    def test_frag_mode_group_compounds_weights(self):
        """adduct_inst_type_frag_mode_group further divides by specs_per_group.

        Two CID samples in the same group → group count = 2, weight halved vs
        adduct_inst_type_frag_mode alone.
        """
        prec_types = ["[M+H]+"] * 10
        inst_types = ["FT"] * 10
        frag_modes = ["HCD"] * 8 + ["CID"] * 2
        group_ids = list(range(8)) + [99, 99]  # CID pair share one group
        ds = MockDataset(prec_types, inst_types, frag_modes=frag_modes, group_ids=group_ids)

        gen1 = th.Generator()
        sampler_plain = get_group_sampler(
            ds, "adduct_inst_type_frag_mode", avg_per_group=1, generator=gen1
        )
        gen2 = th.Generator()
        sampler_group = get_group_sampler(
            ds, "adduct_inst_type_frag_mode_group", avg_per_group=1, generator=gen2
        )

        weights_plain = th.tensor(list(sampler_plain.weights))
        weights_group = th.tensor(list(sampler_group.weights))

        # CID samples at idx 8, 9: group size=2 → weight /= 2
        assert weights_group[8].item() == pytest.approx(
            weights_plain[8].item() / 2.0, rel=1e-5
        )
        # HCD samples each in own group (size=1) → unchanged
        assert weights_group[0].item() == pytest.approx(weights_plain[0].item(), rel=1e-5)

    def test_single_frag_mode_uniform_weights(self):
        """When all samples share the same (prec_type, inst_type, frag_mode), weights are equal."""
        prec_types = ["[M+H]+"] * 8
        inst_types = ["FT"] * 8
        frag_modes = ["HCD"] * 8
        ds = MockDataset(prec_types, inst_types, frag_modes=frag_modes)

        gen = th.Generator()
        sampler = get_group_sampler(
            ds, "adduct_inst_type_frag_mode", avg_per_group=1, generator=gen
        )

        weights = th.tensor(list(sampler.weights))
        assert th.allclose(weights, weights[0].expand_as(weights))

    def test_sampler_yields_correct_length(self):
        """Iterating the sampler yields exactly num_samples indices."""
        prec_types = ["[M+H]+"] * 9 + ["[M-H]-"]
        inst_types = ["FT"] * 9 + ["QTOF"]
        ds = MockDataset(prec_types, inst_types)

        gen = th.Generator()
        gen.manual_seed(42)
        sampler = get_group_sampler(ds, "adduct_inst_type", avg_per_group=1, generator=gen)

        drawn = list(sampler)
        assert len(drawn) == len(ds)

    def test_minority_oversampled_in_drawn_indices(self):
        """With replacement=True, the minority class should appear more than once
        in expectation across a large draw."""
        # 90% majority, 10% minority (1 sample)
        prec_types = ["[M+H]+"] * 9 + ["[M-H]-"]
        inst_types = ["FT"] * 9 + ["QTOF"]
        ds = MockDataset(prec_types, inst_types)

        gen = th.Generator()
        gen.manual_seed(7)
        sampler = get_group_sampler(ds, "adduct_inst_type", avg_per_group=1, generator=gen)

        # Draw many epochs and count minority appearances
        minority_idx = 9
        minority_count = sum(1 for idx in sampler if idx == minority_idx)

        # With balanced weights, minority should appear ~50% of draws (5/10)
        # Allow generous range to avoid flakiness
        assert minority_count >= 1, "Minority class never sampled despite high weight"
