"""
Data samplers for spectrum-molecule-fragment datasets.

Includes:
- SpecMolFragDynamicBatchSampler: Dynamic batching to avoid OOM
- GreedyInferenceBatchSampler: Simple greedy sequential batching for inference
- GroupSampler: Sample K specs per group_id
- get_group_sampler: WeightedRandomSampler factory
"""

import logging
from collections.abc import Iterator

import numpy as np
import torch as th
from torch.utils.data import BatchSampler, Sampler, WeightedRandomSampler
from tqdm import tqdm

logger = logging.getLogger(__name__)

#####################################################
# Batch Samplers
#####################################################


class SpecMolFragDynamicBatchSampler(BatchSampler):
    """Dynamically adds samples to a mini-batch up to a maximum size either based on number of nodes on frag DAG or number of edges on frag DAG.
    This is used to avoid CUDA OOM errors, implmentation is inspired by PyG DynamicBatchSampler, and this should be used to replace default BatchSampler
    This should have the same random sampling behavior as RandomSampler
    """

    def __init__(
        self,
        data_source,
        max_num: int,
        limited_by: str = "frag_edge",
        skip_too_big: bool = False,
        num_samples: int | None = None,
        return_batch_at: int = 0,
        sampler: Sampler | None = None,
    ) -> None:
        """Initialize dynamic batch sampler.

        Args:
            data_source: Dataset with frag_pyg data
            max_num (int): Maximum number of nodes or edges per batch
            limited_by (str): 'frag_node' or 'frag_edge' (default: 'frag_edge')
            skip_too_big (bool): Skip samples larger than max_num (default: False)
            num_samples (Optional[int]): Number of samples to draw (default: all)
            return_batch_at (int): Return batch every N samples for gradient accumulation
            sampler (Optional[Sampler]): Base sampler (e.g., RandomSampler, GroupSampler)

        Raises:
            ValueError: If max_num <= 0 or limited_by not in ['frag_node', 'frag_edge']
        """
        if not isinstance(max_num, int) or max_num <= 0:
            raise ValueError(f"`max_num` should be a positive integer value (got {max_num}).")
        if limited_by not in ["frag_node", "frag_edge"]:
            raise ValueError(
                f"`limited_by` choice should be either 'frag_node' or 'frag_edge' (got '{limited_by}')."
            )

        if num_samples is None:
            num_samples = len(data_source)

        self.data_source = data_source
        self._max_num = max_num
        self._limited_by = limited_by
        self._skip_too_big = skip_too_big
        self._max_sampling_step = num_samples
        self._return_batch_at = return_batch_at
        self._batches = []
        self._data_meta = []  # becomes np.ndarray (N, 2) after _pre_load_batches
        self.sampler = sampler
        # If underlying sampler exposes a torch.Generator, capture its initial seed
        # so we can reseed it in set_epoch() for deterministic epoch-aware behavior.
        self._base_sampler_generator = None
        self._base_sampler_seed = None
        if self.sampler is not None and hasattr(self.sampler, "generator"):
            gen = self.sampler.generator
            if isinstance(gen, th.Generator):
                self._base_sampler_generator = gen
                try:
                    self._base_sampler_seed = int(self._base_sampler_generator.initial_seed())
                except Exception:
                    self._base_sampler_seed = 0
        self._epoch = 0
        self._pre_load_batches()
        self._pre_compute_batches()

    def _pre_load_batches(self):
        """Pre-load and cache frag graph metadata (nodes, edges) for each sample.

        Optimizations:
        - Only load metadata once (no reloading across epochs)
        - Use compact tuple storage instead of dict access
        - Disable tqdm for faster loading
        """
        # get data meta once and cache them
        assert len(self._data_meta) == 0, len(self._data_meta)
        expected_total = 0
        warning_msg = ""

        logger.info(
            f"[SpecMolFragDynamicBatchSampler] Pre-loading metadata for {len(self.data_source)} samples..."
        )

        for dataset_idx in tqdm(range(len(self.data_source))):
            if hasattr(self.data_source, "precom_meta_info"):
                self._data_meta.append(self.data_source.precom_meta_info[dataset_idx])
            else:
                data = self.data_source[dataset_idx]
                # Cache as compact tuple (num_nodes, num_edges)
                self._data_meta.append((data["frag_pyg"].num_nodes, data["frag_pyg"].num_edges))

            # Quick lookup using limited_by index (0=nodes, 1=edges)
            limited_by_idx = 0 if self._limited_by == "frag_node" else 1
            n = self._data_meta[dataset_idx][limited_by_idx]

            if not (n > self._max_num and self._skip_too_big):
                expected_total += 1
            else:
                warning_msg += f"Size of data sample at index {dataset_idx} is larger than {self._max_num} {self._limited_by}s (got {n}). "
                if self._skip_too_big:
                    warning_msg += "Sampler will skip this to prevent CUDA OOM ERROR.\n"
                else:
                    warning_msg += "Attempting to fit into batch, may cause CUDA OOM ERROR.\n"

        if warning_msg:
            logger.warning(f"[SpecMolFragDynamicBatchSampler] {warning_msg}")
        logger.info(
            f"[SpecMolFragDynamicBatchSampler] Metadata loaded. Expecting {expected_total}/{len(self.data_source)} samples with skip_too_big={self._skip_too_big}"
        )
        # Convert to numpy array for O(1) vectorised column reads in _pre_compute_batches
        self._data_meta = np.array(self._data_meta, dtype=np.int32)

    def set_epoch(self, epoch: int) -> None:
        """Set epoch number to trigger batch recomputation.

        Args:
            epoch (int): Current epoch number
        """
        self._epoch = epoch
        # If underlying sampler supports set_epoch, delegate to it
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)
        # If underlying sampler exposes a torch.Generator but lacks set_epoch,
        # reseed it deterministically based on the captured base seed.
        elif self._base_sampler_generator is not None:
            seed = self._base_sampler_seed if self._base_sampler_seed is not None else 0
            self._base_sampler_generator.manual_seed(seed + int(epoch))
        # Recompute batches after epoch change
        self._pre_compute_batches()

    def _pre_compute_batches(self):
        """Pre-compute batches respecting max_num constraint.

        Optimizations:
        - Cache limited_by_idx (0 or 1) to avoid string comparison per sample
        - Vectorised size pre-fetch via numpy to avoid per-element .item() and tuple indexing
        - Use zip(indices_list, sizes_list) to eliminate all per-element Python boxing
        - Skip empty batches
        """
        self._batches = []

        if self.sampler is not None:
            indices = th.as_tensor(list(self.sampler), dtype=th.long)
        else:
            indices = th.arange(len(self.data_source), dtype=th.long)

        # limited index to _max_sampling_step
        indices = indices[: self._max_sampling_step]

        # Cache this to avoid string comparison per iteration
        limited_by_idx = 0 if self._limited_by == "frag_node" else 1

        # Pre-fetch all sizes in one vectorised numpy read, then convert to plain Python
        # lists so the inner loop pays no per-element boxing or array-indexing overhead.
        indices_np = indices.numpy()
        sizes_list: list[int] = self._data_meta[indices_np, limited_by_idx].tolist()
        indices_list: list[int] = indices_np.tolist()

        num_processed = 0
        batch = []
        batch_n = 0
        batch_filled = False

        # Fill batch — plain zip over pre-computed Python lists, no .item() calls
        for idx_item, n in zip(indices_list, sizes_list):
            if n > self._max_num and self._skip_too_big:
                continue
            # check batch_filled condition
            if batch_n + n > self._max_num:
                # no more budget left, mini-batch filled
                batch_filled = True
            # check we need return at this point for ga
            if (
                self._return_batch_at > 0
                and num_processed > 0
                and num_processed % self._return_batch_at == 0
            ):
                # Mini-batch filled
                batch_filled = True
            if batch_filled:
                if batch:  # Only append non-empty batches
                    self._batches.append(batch)
                batch_n = 0
                batch = []
                batch_filled = False
            # Add sample to current batch
            batch.append(idx_item)
            num_processed += 1
            batch_n += n

        if batch:  # Don't forget final batch
            self._batches.append(batch)
        logger.info(
            f"[SpecMolFragDynamicBatchSampler] Batch indices computed, Expecting {len(self._batches)} mini-batches for next epoch"
        )

    def __iter__(self) -> Iterator[list[int]]:
        """Iterate over pre-computed batches.

        Yields:
            List[int]: Batch of sample indices
        """
        for batch in self._batches:
            yield batch

    def __len__(self) -> int:
        """Return number of batches.

        Note: __len__() is expected in calculations involving the length of a DataLoader.
        ref: https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler

        Returns:
            int: Number of batches
        """
        return len(self._batches)


class GreedyInferenceBatchSampler(BatchSampler):
    """Greedy sequential batch sampler for inference.

    Iterates samples in index order and accumulates them into a batch until adding
    the next sample would exceed `max_num` nodes or edges. No randomization or
    tight packing — once a sample doesn't fit the current batch is flushed and a
    new one starts. Oversized singletons (a single sample larger than `max_num`)
    are emitted as-is to avoid silently dropping data.

    Compared to `SpecMolFragDynamicBatchSampler` this class has no epoch logic,
    no gradient-accumulation return points, and no generator — it is intentionally
    stateless and deterministic.
    """

    def __init__(
        self,
        data_source,
        max_num: int,
        limited_by: str = "frag_edge",
    ) -> None:
        """Initialize greedy inference batch sampler.

        Args:
            data_source: Dataset with frag_pyg data or precom_meta_info cache.
            max_num: Maximum number of nodes or edges per batch.
            limited_by: Either 'frag_node' or 'frag_edge'. Defaults to 'frag_edge'.

        Raises:
            ValueError: If max_num <= 0 or limited_by is not a valid option.
        """
        if not isinstance(max_num, int) or max_num <= 0:
            raise ValueError(f"`max_num` should be a positive integer value (got {max_num}).")
        if limited_by not in ("frag_node", "frag_edge"):
            raise ValueError(
                f"`limited_by` must be 'frag_node' or 'frag_edge' (got '{limited_by}')."
            )

        self.data_source = data_source
        self._max_num = max_num
        self._limited_by = limited_by
        idx_col = 0 if limited_by == "frag_node" else 1

        logger.info(
            f"[GreedyInferenceBatchSampler] Building batches for {len(data_source)} samples..."
        )
        has_cache = hasattr(data_source, "precom_meta_info")
        batches: list[list[int]] = []
        batch: list[int] = []
        batch_n = 0
        for idx in tqdm(range(len(data_source))):
            if has_cache:
                n = data_source.precom_meta_info[idx][idx_col]
            else:
                data = data_source[idx]
                n = data["frag_pyg"].num_nodes if idx_col == 0 else data["frag_pyg"].num_edges
            if batch and batch_n + n > max_num:
                batches.append(batch)
                batch = []
                batch_n = 0
            batch.append(idx)
            batch_n += n
        if batch:
            batches.append(batch)

        self._batches = batches
        logger.info(
            f"[GreedyInferenceBatchSampler] {len(batches)} batches (max_{limited_by}={max_num})"
        )

    def __iter__(self) -> Iterator[list[int]]:
        """Yield pre-computed batches in sequential order.

        Yields:
            List of sample indices forming one batch.
        """
        yield from self._batches

    def __len__(self) -> int:
        """Return number of batches.

        Returns:
            Number of batches.
        """
        return len(self._batches)


class GroupSampler(Sampler):
    """Sample K spectra per group_id to keep experimental groups together.

    Use case: When spectra are grouped by experiment/acquisition (group_id),
    this sampler ensures that K random spectra from each group are sampled
    in each epoch.
    """

    def __init__(
        self, data_source, sample_k: int | None = None, generator: th.Generator | None = None
    ) -> None:
        """Initialize group sampler.

        Args:
            data_source: Dataset with group_id in each sample
            sample_k (Optional[int]): Specs to sample per group (default: 3)
            generator (Optional[th.Generator]): PyTorch generator for reproducibility
        """
        self.data_source = data_source
        self.num_samples = None
        self._data_meta_d = {}
        self.sample_k = 3 if sample_k is None else sample_k
        self.generator = generator
        self._epoch = 0
        self._pre_compute_meta()
        self._pre_compute_batches()

    def _pre_compute_meta(self):
        """Build group_id -> [dataset_indices] mapping."""
        for dataset_idx in range(len(self.data_source)):
            data = self.data_source[dataset_idx]
            group_id = data["group_id"].item()
            if group_id not in self._data_meta_d:
                self._data_meta_d[group_id] = []
            self._data_meta_d[group_id].append(dataset_idx)

        for group_id in self._data_meta_d:
            self._data_meta_d[group_id] = th.tensor(self._data_meta_d[group_id])

    def _pre_compute_batches(self):
        """Pre-compute sampled indices (one sample per group) with epoch-aware randomization."""
        if self.generator is None:
            generator = th.Generator()
        else:
            generator = self.generator

        sampled_indices = []
        for group_id in self._data_meta_d:
            group_indices = self._data_meta_d[group_id]
            # shuffle group indices and take up to sample_k
            if len(group_indices) == 0:
                continue
            take = min(len(group_indices), self.sample_k)
            if len(group_indices) == 1:
                sampled_group_indices = group_indices[:take]
            else:
                perm = th.randperm(len(group_indices), generator=generator)
                sampled_group_indices = group_indices[perm[:take]]
            sampled_indices.append(sampled_group_indices)

        if sampled_indices:
            sampled_indices = th.cat(sampled_indices)
            # shuffle across groups for variety: permute actual dataset indices
            if len(sampled_indices) > 1:
                perm_all = th.randperm(len(sampled_indices), generator=generator)
                self.sampled_indices = sampled_indices[perm_all]
            else:
                self.sampled_indices = sampled_indices
            self.num_samples = len(sampled_indices)
        else:
            self.sampled_indices = th.tensor([], dtype=th.long)
            self.num_samples = 0

    def set_epoch(self, epoch: int) -> None:
        """Set epoch number to trigger re-sampling.

        Args:
            epoch (int): Current epoch number
        """
        self._epoch = epoch
        self._pre_compute_batches()

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampled indices.

        Yields:
            int: Sample index
        """
        for i in range(self.num_samples):
            yield self.sampled_indices[i].item()

    def __len__(self) -> int:
        """Return number of sampled indices.

        Note: __len__() is not strictly required by DataLoader but is expected
        in calculations involving the length of a DataLoader.
        ref: https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler

        Returns:
            int: Number of sampled indices
        """
        assert self.num_samples is not None
        return self.num_samples


class FormulaPrecTypeGroupSampler(Sampler):
    """Sample indices grouped by composite key `prec_type::formula`.

    This sampler yields up to `sample_k` indices per formula+prec_type
    cluster each epoch. It is compatible with `SpecMolFragDynamicBatchSampler`
    via the `sampler` argument and exposes an optional `generator` for
    deterministic behavior.

    Args:
        data_source: dataset providing __len__ and __getitem__ returning dicts
            with keys for formula and adduct (defaults: 'formula', 'adduct').
        sample_k (int): max samples to take from each composite group.
        key_formula (str): key name to read formula from sample dict.
        key_adduct (str): key name to read adduct from sample dict.
        generator (Optional[th.Generator]): optional torch Generator for determinism.
        num_samples (Optional[int]): limit total number of sampled indices.
    """

    def __init__(
        self,
        data_source,
        sample_k: int = 1,
        key_formula: str = "formula",
        key_adduct: str = "prec_type",
        generator: th.Generator | None = None,
        num_samples: int | None = None,
    ) -> None:
        self.data_source = data_source
        self.sample_k = int(sample_k)
        self.key_formula = key_formula
        self.key_adduct = key_adduct
        self.generator = generator
        self._epoch = 0
        if num_samples is None:
            num_samples = len(data_source)
        self.num_samples = int(num_samples)

        # capture base seed if generator provided
        self._base_seed = None
        if self.generator is not None and hasattr(self.generator, "initial_seed"):
            try:
                self._base_seed = int(self.generator.initial_seed())
            except Exception:
                self._base_seed = 0

        # group -> tensor(indices)
        self._groups = {}
        self._pre_compute_meta()
        self._pre_compute_indices()

    def _pre_compute_meta(self):
        """Build composite key -> list(indices) mapping.
        Raises KeyError if required keys are missing.
        """
        for dataset_idx in range(len(self.data_source)):
            data = self.data_source[dataset_idx]
            if self.key_formula not in data or self.key_adduct not in data:
                raise KeyError(
                    f"Sample {dataset_idx} missing '{self.key_formula}' or '{self.key_adduct}' key."
                )
            # normalize to str for stable keys
            formula = str(data[self.key_formula])
            adduct = str(data[self.key_adduct])
            key = f"{adduct}::{formula}"
            if key not in self._groups:
                self._groups[key] = []
            self._groups[key].append(dataset_idx)

        # convert to tensors
        for k in list(self._groups.keys()):
            self._groups[k] = th.tensor(self._groups[k], dtype=th.long)

    def _pre_compute_indices(self):
        """Compute epoch-local sampled indices based on `sample_k` and `generator`.
        Stores results in `self.sampled_indices` and updates `self.num_samples`.
        """
        if self.generator is None:
            gen = th.Generator()
        else:
            gen = self.generator

        sampled_parts = []
        for k in self._groups:
            indices = self._groups[k]
            if len(indices) == 0:
                continue
            take = min(len(indices), self.sample_k)
            if len(indices) == 1:
                part = indices[:take]
            else:
                perm = th.randperm(len(indices), generator=gen)
                part = indices[perm][:take]
            sampled_parts.append(part)

        if sampled_parts:
            self.sampled_indices = th.cat(sampled_parts)
            # shuffle across groups for variety
            if len(self.sampled_indices) > 1:
                self.sampled_indices = self.sampled_indices[
                    th.randperm(len(self.sampled_indices), generator=gen)
                ]
            self.num_samples = len(self.sampled_indices)
        else:
            self.sampled_indices = th.tensor([], dtype=th.long)
            self.num_samples = 0

    def set_epoch(self, epoch: int) -> None:
        """Set epoch, reseed generator deterministically and recompute indices."""
        self._epoch = int(epoch)
        if self.generator is not None and hasattr(self.generator, "manual_seed"):
            try:
                base = self._base_seed if self._base_seed is not None else 0
                self.generator.manual_seed(base + int(epoch))
            except Exception:
                pass
        self._pre_compute_indices()

    def __iter__(self) -> Iterator[int]:
        for i in range(self.num_samples):
            yield int(self.sampled_indices[i].item())

    def __len__(self) -> int:
        return self.num_samples


class DualGroupDynamicBatchSampler(BatchSampler):
    """Build mini-batches that include samples grouped by two keys:
    1) adduct + mol_id
    2) adduct + formula

    Each produced batch will try to include at least one sample of each
    type (when available) and will generally respect maximum fragment
    node/edge constraints (like SpecMolFragDynamicBatchSampler).
    When `skip_too_big=False`, an oversized sample may be emitted as a
    singleton batch to avoid dropping data.
    """

    def __init__(
        self,
        data_source,
        max_num: int,
        limited_by: str = "frag_edge",
        skip_too_big: bool = False,
        num_samples: int | None = None,
        return_batch_at: int = 0,
        sample_k1: int = 1,
        sample_k2: int = 1,
        sample_k1_per_key: int = 1,
        sample_k2_per_key: int = 1,
        generator: th.Generator | None = None,
        key_adduct: str = "prec_type",
        key_mol_id: str = "mol_id",
        key_formula: str = "formula",
    ) -> None:
        if not isinstance(max_num, int) or max_num <= 0:
            raise ValueError(f"`max_num` should be a positive integer value (got {max_num}).")
        if limited_by not in ["frag_node", "frag_edge"]:
            raise ValueError(
                f"`limited_by` choice should be either 'frag_node' or 'frag_edge' (got '{limited_by}')."
            )

        if num_samples is None:
            num_samples = len(data_source)

        self.data_source = data_source
        self._max_num = max_num
        self._limited_by = limited_by
        self._skip_too_big = skip_too_big
        self._max_sampling_step = num_samples
        self._return_batch_at = return_batch_at
        self.sample_k1 = int(sample_k1)
        self.sample_k2 = int(sample_k2)
        self.sample_k1_per_key = int(sample_k1_per_key)
        self.sample_k2_per_key = int(sample_k2_per_key)
        self.generator = generator
        self._epoch = 0
        self._data_meta = []  # becomes np.ndarray (N, 2) after _pre_load_metadata_and_groups
        self._group1 = {}  # adduct+mol_id -> list(indices)
        self._group2 = {}  # adduct+formula -> list(indices)
        self._idx_to_group1_key: list[str] = []  # per-sample inverse map for O(1) count updates
        self._idx_to_group2_key: list[str] = []  # per-sample inverse map for O(1) count updates
        self._batches = []
        self.key_adduct = key_adduct
        self.key_mol_id = key_mol_id
        self.key_formula = key_formula

        self._pre_load_metadata_and_groups()
        self._pre_compute_batches()

    def _pre_load_metadata_and_groups(self):
        """Load frag metadata and build group mappings.

        Uses precom_meta_info and precom_group_info caches on the dataset when available,
        avoiding a full frag_pyg load for each sample (which would be expensive for
        HDF5/LMDB-backed datasets). Falls back to data_source[idx] otherwise.

        Also builds per-sample inverse maps (_idx_to_group1_key, _idx_to_group2_key) for
        O(1) available-count updates in _pre_compute_batches, replacing the original
        O(G*S) `any(idx in available ...)` scan that ran over all active keys per batch.
        """
        assert len(self._data_meta) == 0, len(self._data_meta)

        N = len(self.data_source)
        self._idx_to_group1_key = [""] * N
        self._idx_to_group2_key = [""] * N

        has_meta_cache = hasattr(self.data_source, "precom_meta_info")
        has_group_cache = hasattr(self.data_source, "precom_group_info")

        for dataset_idx in range(N):
            # --- node/edge counts ---
            if has_meta_cache:
                n_nodes, n_edges = self.data_source.precom_meta_info[dataset_idx]
            else:
                data = self.data_source[dataset_idx]
                n_nodes = data["frag_pyg"].num_nodes
                n_edges = data["frag_pyg"].num_edges
            self._data_meta.append((n_nodes, n_edges))

            # --- group keys (adduct, mol_id, formula) ---
            if has_group_cache:
                adduct, mol_id, formula = self.data_source.precom_group_info[dataset_idx]
            else:
                if not has_meta_cache:
                    # data already loaded above
                    pass
                else:
                    data = self.data_source[dataset_idx]
                for req_key in (self.key_adduct, self.key_mol_id, self.key_formula):
                    if req_key not in data:
                        raise KeyError(
                            f"Sample at index {dataset_idx} missing required key '{req_key}'."
                        )
                    if data[req_key] is None:
                        raise ValueError(
                            f"Sample at index {dataset_idx} has None value for required key '{req_key}'."
                        )
                adduct = data[self.key_adduct]
                mol_id = data[self.key_mol_id]
                formula = data[self.key_formula]

            key1 = f"{adduct}::{mol_id}"
            self._group1.setdefault(key1, []).append(dataset_idx)
            self._idx_to_group1_key[dataset_idx] = key1

            key2 = f"{adduct}::{formula}"
            self._group2.setdefault(key2, []).append(dataset_idx)
            self._idx_to_group2_key[dataset_idx] = key2

        # Within-batch deduplication is handled by taken_mask in _sample_from_group_dict,
        # so group2 intentionally retains all indices (including those also in group1).
        # This allows isomer contrast: group1 picks a sample first, taken_mask prevents
        # group2 from re-picking the same index in the same batch.

        # Keep groups as plain Python lists — no tensor conversion needed since we
        # access them by iteration, not by fancy indexing.

        # Convert group candidate lists to numpy arrays so callers/tests can
        # call `.tolist()` and iteration yields integer indices usable as
        # numpy array indices. Numpy arrays keep iteration semantics while
        # avoiding PyTorch-tensor indexing issues with numpy masks.
        for k in list(self._group1.keys()):
            self._group1[k] = np.array(self._group1[k], dtype=np.int64)
        for k in list(self._group2.keys()):
            self._group2[k] = np.array(self._group2[k], dtype=np.int64)

        # Convert metadata to numpy array for vectorised column reads
        self._data_meta = np.array(self._data_meta, dtype=np.int32)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        # reseed generator if provided
        if self.generator is not None and hasattr(self.generator, "manual_seed"):
            try:
                self.generator.manual_seed(int(epoch))
            except Exception:
                pass
        self._pre_compute_batches()

    def _sample_from_group_dict(
        self,
        group_dict,
        active_keys,
        taken_mask: np.ndarray,
        budget_left: int,
        take_k: int,
        limited_by_idx: int,
        per_key: int = 1,
    ):
        """Try to take up to `take_k` samples from group_dict (which maps key->list).

        Iterates over up to `take_k` group keys and takes up to `per_key` samples from
        each key. Using `per_key > 1` places multiple spectra of the same molecule (or
        same formula) in a single batch, which provides meaningful signal for pairwise
        similarity losses.

        Modifies `taken_mask` in-place by marking taken indices as True.

        Args:
            group_dict: Mapping of key -> list of dataset indices.
            active_keys: Set of keys in group_dict that still have available samples.
            taken_mask: Boolean numpy array of shape (N,); True = already assigned.
            budget_left: Remaining node/edge budget for the current batch.
            take_k: Maximum number of keys to visit (total samples taken ≤ take_k * per_key).
            limited_by_idx: 0 for frag_node, 1 for frag_edge.
            per_key: Maximum number of samples to take from each key. Defaults to 1.

        Returns:
            Tuple of (taken indices list, updated budget_left).
        """
        taken = []
        if not active_keys:
            return taken, budget_left
        keys = list(active_keys)
        # randomize key order using PyTorch Generator for consistency
        if self.generator is not None:
            perm = th.randperm(len(keys), generator=self.generator)
            keys = [keys[i] for i in perm.tolist()]
        # If no generator, keys remain in original order (deterministic)

        for key in keys:
            if take_k <= 0:
                break
            candidates = group_dict[key]  # plain Python list, no .tolist() needed
            count_from_key = 0
            for idx in candidates:
                if count_from_key >= per_key:
                    break
                if taken_mask[idx]:  # faster than `idx not in available` (set lookup)
                    continue
                # check size against remaining budget
                n = int(self._data_meta[idx, limited_by_idx])
                if n > budget_left and self._skip_too_big:
                    continue
                if n > budget_left:
                    # Match SpecMolFragDynamicBatchSampler behavior: when
                    # skip_too_big=False, allow an oversized singleton batch
                    # instead of silently dropping the sample.
                    if not (n > self._max_num and budget_left == self._max_num):
                        continue
                # take it — mark in taken_mask immediately so fill phase sees it
                taken_mask[idx] = True
                taken.append(idx)
                budget_left = max(0, budget_left - n)
                count_from_key += 1
            if count_from_key > 0:
                take_k -= 1
        return taken, budget_left

    def _pre_compute_batches(self):
        """Pre-compute batches for one epoch.

        Key optimisations vs. original:
        - Boolean numpy mask replaces Python set — O(1) membership test and mark.
        - Per-group available counts (group1_avail_count, group2_avail_count) replace
          the O(G*S) `any(idx in available for idx in group)` exhausted-key scan:
          counts are decremented O(1) per taken sample via the inverse maps built in
          _pre_load_metadata_and_groups.
        - Group candidate lists are plain Python lists (no .tolist() per call).
        - Fill phase uses np.where(~taken_mask) — vectorised C scan replaces
          list(python_set) materialisation every batch.
        """
        self._batches = []
        n = min(len(self.data_source), self._max_sampling_step)
        limited_by_idx = 0 if self._limited_by == "frag_node" else 1

        # Boolean mask over full dataset; pre-mark out-of-range indices as taken
        taken_mask = np.zeros(len(self.data_source), dtype=bool)
        if n < len(self.data_source):
            taken_mask[n:] = True
        remaining = n

        # Per-group available counts — O(1) exhausted-key detection
        if n == len(self.data_source):
            group1_avail_count = {k: len(v) for k, v in self._group1.items()}
            group2_avail_count = {k: len(v) for k, v in self._group2.items()}
        else:
            group1_avail_count = {k: sum(1 for i in v if i < n) for k, v in self._group1.items()}
            group2_avail_count = {k: sum(1 for i in v if i < n) for k, v in self._group2.items()}
        active_group1_keys = {k for k, c in group1_avail_count.items() if c > 0}
        active_group2_keys = {k for k, c in group2_avail_count.items() if c > 0}

        while remaining > 0:
            batch = []
            budget_left = self._max_num
            samples_to_take = self._return_batch_at if self._return_batch_at > 0 else remaining

            # Group1 phase
            if active_group1_keys and samples_to_take > 0:
                k1 = min(self.sample_k1, samples_to_take)
                taken1, budget_left = self._sample_from_group_dict(
                    self._group1,
                    active_group1_keys,
                    taken_mask,
                    budget_left,
                    k1,
                    limited_by_idx=limited_by_idx,
                    per_key=self.sample_k1_per_key,
                )
                batch.extend(taken1)
                samples_to_take -= len(taken1)
                remaining -= len(taken1)
                # O(taken1) count updates — replaces O(G*S) cleanup scan
                for idx in taken1:
                    k = self._idx_to_group1_key[idx]
                    group1_avail_count[k] -= 1
                    if group1_avail_count[k] == 0:
                        active_group1_keys.discard(k)
                    k2 = self._idx_to_group2_key[idx]
                    group2_avail_count[k2] -= 1
                    if group2_avail_count[k2] == 0:
                        active_group2_keys.discard(k2)

            # Group2 phase
            if active_group2_keys and samples_to_take > 0:
                k2 = min(self.sample_k2, samples_to_take)
                taken2, budget_left = self._sample_from_group_dict(
                    self._group2,
                    active_group2_keys,
                    taken_mask,
                    budget_left,
                    k2,
                    limited_by_idx=limited_by_idx,
                    per_key=self.sample_k2_per_key,
                )
                batch.extend(taken2)
                samples_to_take -= len(taken2)
                remaining -= len(taken2)
                for idx in taken2:
                    k1 = self._idx_to_group1_key[idx]
                    group1_avail_count[k1] -= 1
                    if group1_avail_count[k1] == 0:
                        active_group1_keys.discard(k1)
                    k2 = self._idx_to_group2_key[idx]
                    group2_avail_count[k2] -= 1
                    if group2_avail_count[k2] == 0:
                        active_group2_keys.discard(k2)

            # Fill phase: np.where vectorised scan replaces list(python_set) per batch
            if budget_left > 0 and samples_to_take > 0:
                avail_indices = np.where(~taken_mask[:n])[0]
                if len(avail_indices) > 0:
                    if self.generator is not None:
                        perm = th.randperm(len(avail_indices), generator=self.generator).numpy()
                        avail_indices = avail_indices[perm]
                    for idx in avail_indices.tolist():
                        if taken_mask[idx]:
                            continue
                        if samples_to_take <= 0:
                            break
                        n_val = int(self._data_meta[idx, limited_by_idx])
                        if n_val > self._max_num and self._skip_too_big:
                            continue
                        if n_val > budget_left:
                            if not (n_val > self._max_num and len(batch) == 0):
                                continue
                        batch.append(idx)
                        samples_to_take -= 1
                        remaining -= 1
                        taken_mask[idx] = True
                        k1 = self._idx_to_group1_key[idx]
                        group1_avail_count[k1] -= 1
                        if group1_avail_count[k1] == 0:
                            active_group1_keys.discard(k1)
                        k2 = self._idx_to_group2_key[idx]
                        group2_avail_count[k2] -= 1
                        if group2_avail_count[k2] == 0:
                            active_group2_keys.discard(k2)
                        budget_left = max(0, budget_left - n_val)

            if not batch:
                # Nothing could be placed (maybe all remaining too large)
                break

            self._batches.append(batch)

        logger.info(
            f"[DualGroupDynamicBatchSampler] Prepared {len(self._batches)} batches respecting max {self._limited_by}={self._max_num}"
        )

    def __iter__(self) -> Iterator[list[int]]:
        for b in self._batches:
            yield b

    def __len__(self) -> int:
        return len(self._batches)


def get_group_sampler(
    ds, sampler_type: str, avg_per_group: int, generator: th.Generator
) -> WeightedRandomSampler | None:
    """Create a WeightedRandomSampler based on group/molecule or adduct+instrument statistics.

    Balances sampling to ensure certain groups/molecules are sampled proportionally.

    Args:
        ds: Dataset with get_group_mol_stats() and get_adduct_inst_type_stats() methods.
        sampler_type (str): Weighting strategy:
            - 'group': Weight by 1/specs_per_group (replacement=False)
            - 'mol': Weight by 1/specs_per_mol (replacement=False)
            - 'group_mol': Weight by 1/(specs_per_group * group_per_mol) (replacement=False)
            - 'adduct_inst_type': Weight by 1/count(prec_type, inst_type) (replacement=True).
              Oversample minority (adduct, instrument) combinations. Epoch size = len(ds).
            - 'adduct_inst_type_group': Weight by 1/(count(prec_type, inst_type) *
              specs_per_group) (replacement=True). Combines adduct+instrument balancing
              with group-level balancing. Epoch size = len(ds).
            - 'adduct_inst_type_frag_mode': Weight by 1/count(prec_type, inst_type, frag_mode)
              (replacement=True). Extends adduct+instrument balancing to also upweight minority
              fragmentation modes (e.g. CID vs HCD). Epoch size = len(ds).
            - 'adduct_inst_type_frag_mode_group': Weight by 1/(count(prec_type, inst_type,
              frag_mode) * specs_per_group) (replacement=True). Combines frag-mode balancing
              with group-level balancing. Epoch size = len(ds).
        avg_per_group (int): Average samples per group to target (used by group/mol types).
        generator (th.Generator): PyTorch generator for reproducibility.

    Returns:
        WeightedRandomSampler | None: Configured sampler, or None if invalid sampler_type.

    Example:
        ```python
        sampler = get_group_sampler(ds, "adduct_inst_type", avg_per_group=1, generator=generator)
        loader = DataLoader(ds, batch_sampler=sampler)
        ```
    """
    if sampler_type in (
        "adduct_inst_type",
        "adduct_inst_type_group",
        "adduct_inst_type_frag_mode",
        "adduct_inst_type_frag_mode_group",
    ):
        prec_types, inst_types = ds.get_adduct_inst_type_stats()
        use_frag_mode = sampler_type in (
            "adduct_inst_type_frag_mode",
            "adduct_inst_type_frag_mode_group",
        )
        if use_frag_mode:
            frag_modes = ds.get_frag_mode_stats()
        combo_counts: dict[str, int] = {}
        for i, (pt, it) in enumerate(zip(prec_types, inst_types)):
            key = f"{pt}::{it}::{frag_modes[i]}" if use_frag_mode else f"{pt}::{it}"
            combo_counts[key] = combo_counts.get(key, 0) + 1
        if use_frag_mode:
            sample_weights = th.tensor(
                [
                    1.0 / combo_counts[f"{pt}::{it}::{fm}"]
                    for pt, it, fm in zip(prec_types, inst_types, frag_modes)
                ],
                dtype=th.float32,
            )
        else:
            sample_weights = th.tensor(
                [1.0 / combo_counts[f"{pt}::{it}"] for pt, it in zip(prec_types, inst_types)],
                dtype=th.float32,
            )
        if sampler_type in ("adduct_inst_type_group", "adduct_inst_type_frag_mode_group"):
            _, _, spec_per_group, _, _ = ds.get_group_mol_stats()
            sample_weights = sample_weights / spec_per_group.float()
        num_samples = len(ds)
        return WeightedRandomSampler(
            sample_weights, num_samples=num_samples, replacement=True, generator=generator
        )

    group_ids, _, spec_per_group, spec_per_mol, group_per_mol = ds.get_group_mol_stats()

    if sampler_type == "group":
        sample_weights = 1.0 / spec_per_group
    elif sampler_type == "mol":
        sample_weights = 1.0 / spec_per_mol
    elif sampler_type == "group_mol":
        sample_weights = 1.0 / (spec_per_group * group_per_mol)
    else:
        return None

    # Get unique groups and their counts
    _, counts = th.unique(group_ids, return_counts=True)
    # For each group, we only want to sample up to the number of available samples
    total_samples = sum(min(count.item(), avg_per_group) for count in counts)
    # Make sure we don't exceed the total dataset size
    num_samples = min(total_samples, len(ds))

    sampler = WeightedRandomSampler(
        sample_weights, num_samples=num_samples, replacement=False, generator=generator
    )
    return sampler
