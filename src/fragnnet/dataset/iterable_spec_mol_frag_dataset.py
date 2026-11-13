import itertools
import logging
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from torch.utils.data import IterableDataset, get_worker_info

logger = logging.getLogger(__name__)


class IterableSpecMolFragDataset(IterableDataset):
    """
    Iterable wrapper that consumes a stream/iterator of indices and groups
    them into dynamic mini-batches such that the total number of frag nodes
    or frag edges per batch does not exceed `max_num`.

    This is intended for inference: provide an `index_stream` that yields
    integer dataset indices (e.g. ``range(len(dataset))``) and the wrapper
    yields either collated batch dicts (default) or lists of sample dicts.

    Batches are formed lazily as the stream is consumed, so nothing here needs
    `len(dataset)`, a size cache (`precom_meta_info`), or an up-front pass that
    loads every sample once just to plan batches. Sizing happens in the worker
    as samples arrive, which also keeps the main process from touching the
    dataset (and so from opening an HDF5 handle before workers fork).

    Two wiring requirements follow from that:

    - Pair it with ``DataLoader(..., batch_size=None)``, since the batches are
      already collated; otherwise they get batched a second time.
    - The batch count is unknown until the stream is exhausted, so this class
      has no ``__len__`` and neither does a DataLoader wrapping it.

    With ``num_workers > 0`` the stream is sharded round-robin across workers,
    so prefer a re-sliceable `Sequence` (``range``/``list``) over a generator.

    ``limited_by`` defaults to ``frag_node`` to match `dynamic_batch_sampler_mode`
    in the config template. Only pick ``frag_edge`` for a config that actually
    materializes fragment edges; otherwise every sample measures 0 and batching
    silently stops bounding anything (guarded against in `_sample_size`).
    """

    def __init__(
        self,
        dataset: Any,
        index_stream: Iterable[int],
        max_num: int,
        limited_by: str = "frag_node",
        skip_too_big: bool = False,
        return_collated: bool = True,
        clone_graphs_on_get: bool = True,
        drop_last: bool = False,
    ) -> None:
        if limited_by not in ("frag_edge", "frag_node"):
            raise ValueError("`limited_by` must be either 'frag_edge' or 'frag_node'")
        if not isinstance(max_num, int) or max_num <= 0:
            raise ValueError("`max_num` should be a positive integer")

        self.dataset = dataset
        self.index_stream = index_stream
        self._max_num = max_num
        self._limited_by = limited_by
        self._skip_too_big = skip_too_big
        self._return_collated = return_collated
        self._clone_graphs_on_get = clone_graphs_on_get
        self._drop_last = drop_last

    def _shard_indices(self) -> Iterator[int]:
        """Return this worker's slice of the index stream (all of it when single-process)."""
        worker_info = get_worker_info()
        if worker_info is None:
            return iter(self.index_stream)

        worker_id = worker_info.id
        num_workers = worker_info.num_workers
        if isinstance(self.index_stream, Sequence):
            return iter(self.index_stream[worker_id::num_workers])
        # A bare iterator is forked per worker, so each one strides its own copy.
        return itertools.islice(self.index_stream, worker_id, None, num_workers)

    def _sample_size(self, sample: dict, idx: int) -> int:
        if "frag_pyg" not in sample:
            # Treating this as size 0 would let a batch grow without bound and OOM.
            raise KeyError(
                f"sample at index {idx} has no 'frag_pyg'; "
                f"{type(self).__name__} sizes batches by fragment graph size"
            )
        pyg = sample["frag_pyg"]
        n = int(pyg.num_nodes) if self._limited_by == "frag_node" else int(pyg.num_edges)
        if n == 0:
            # Every sample measuring 0 would never reach max_num, so the whole stream
            # would collect into one batch and OOM. `frag_edge` hits this whenever the
            # model config does not materialize fragment edges (num_edges is then 0 for
            # every sample) -- `frag_node` is always populated.
            raise ValueError(
                f"sample at index {idx} has {self._limited_by}=0, so it cannot bound a "
                f"batch. If this is every sample, the config does not populate "
                f"{self._limited_by}; use limited_by='frag_node' instead."
            )
        return n

    def __iter__(self) -> Iterator:
        indices_iter = self._shard_indices()
        collate_fn = self.dataset.get_collate_fn() if self._return_collated else None

        def emit(samples: list[dict]):
            return collate_fn(samples) if collate_fn is not None else samples

        batch_samples: list[dict] = []
        batch_count = 0

        for idx in indices_iter:
            sample = self.dataset[idx]
            # Clone when the dataset caches and hands back shared PyG objects, so a
            # later in-place edit cannot reach a sample already queued in this batch.
            if self._clone_graphs_on_get and "frag_pyg" in sample:
                sample["frag_pyg"] = sample["frag_pyg"].clone()

            n = self._sample_size(sample, idx)

            if n > self._max_num:
                if self._skip_too_big:
                    logger.warning(
                        f"Skipping sample at index {idx} with size {n} (max_num={self._max_num})"
                    )
                    continue
                logger.debug(
                    f"Sample at index {idx} with size {n} exceeds max_num={self._max_num}. "
                    "Yielding as a single-sample batch."
                )

            if batch_count + n > self._max_num and len(batch_samples) > 0:
                yield emit(batch_samples)
                batch_samples = []
                batch_count = 0

            # Add current sample to batch (even if it alone exceeds max_num,
            # unless skip_too_big was True above)
            batch_samples.append(sample)
            batch_count += n

        # flush leftover
        if len(batch_samples) > 0 and not self._drop_last:
            yield emit(batch_samples)
