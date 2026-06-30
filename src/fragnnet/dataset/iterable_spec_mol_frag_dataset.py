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
    integer dataset indices (e.g., iter(range(len(dataset)))) and the
    wrapper yields either collated batch dicts (default) or lists of
    sample dicts.
    """

    def __init__(
        self,
        dataset: Any,
        index_stream: Iterable[int],
        max_num: int,
        limited_by: str = "frag_edge",
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

    def __iter__(self) -> Iterator:
        # Support multi-worker by sharding the provided index_stream per-worker.
        worker_info = get_worker_info()

        # Build a per-worker index iterable
        if worker_info is None:
            indices_iter = iter(self.index_stream)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            # Use islice or sequence slicing to shard indices across workers.
            # This avoids materializing the whole stream in each worker if it's
            # already an iterator/generator.
            if isinstance(self.index_stream, Sequence):
                try:
                    indices_iter = iter(self.index_stream[worker_id::num_workers])
                except Exception:
                    indices_iter = itertools.islice(self.index_stream, worker_id, None, num_workers)
            else:
                indices_iter = itertools.islice(self.index_stream, worker_id, None, num_workers)

        batch_samples = []
        batch_count = 0

        def sample_size(sample: dict) -> int:
            if "frag_pyg" not in sample:
                return 0
            pyg = sample["frag_pyg"]
            if self._limited_by == "frag_node":
                return int(pyg.num_nodes)
            return int(pyg.num_edges)

        for idx in indices_iter:
            sample = self.dataset[idx]
            # Optionally clone PyG objects if dataset caches them to avoid
            # in-place modifications affecting future samples.
            if self._clone_graphs_on_get and "frag_pyg" in sample:
                try:
                    sample["frag_pyg"] = sample["frag_pyg"].clone()
                except Exception:
                    # If clone not supported, ignore and use as-is
                    pass

            n = sample_size(sample)

            if n > self._max_num:
                if self._skip_too_big:
                    logger.warning(
                        f"Skipping sample at index {idx} with size {n} (max_num={self._max_num})"
                    )
                    continue
                else:
                    logger.debug(
                        f"Sample at index {idx} with size {n} exceeds max_num={self._max_num}. "
                        "Yielding as a single-sample batch."
                    )

            if batch_count + n > self._max_num and len(batch_samples) > 0:
                # yield current batch
                if self._return_collated:
                    collate_fn = self.dataset.get_collate_fn()
                    yield collate_fn(batch_samples)
                else:
                    yield batch_samples
                batch_samples = []
                batch_count = 0

            # Add current sample to batch (even if it alone exceeds max_num,
            # unless skip_too_big was True above)
            batch_samples.append(sample)
            batch_count += n

        # flush leftover
        if len(batch_samples) > 0 and not self._drop_last:
            if self._return_collated:
                collate_fn = self.dataset.get_collate_fn()
                yield collate_fn(batch_samples)
            else:
                yield batch_samples

    def __len__(self) -> int:
        # Length is undefined for streaming iterators; return 0 to avoid
        # misleading behavior. Users should not rely on len() here.
        return 0
