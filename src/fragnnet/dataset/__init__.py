import importlib

from .base_dataset import BaseDataset
from .data_sampler import (
    DualGroupDynamicBatchSampler,
    FormulaPrecTypeGroupSampler,
    GreedyInferenceBatchSampler,
    GroupSampler,
    SpecMolFragDynamicBatchSampler,
    get_group_sampler,
)
from .mces_dataset import MCESDataset
from .spec_mol_dataset import SpecMolDataset
from .spec_mol_frag_dataset import SpecMolFragDataset


def __getattr__(name: str):
    if name == "IterableStreamBatchDataset":
        module = importlib.import_module("fragnnet.dataset.iterable_stream_batch_dataset")
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    "BaseDataset",
    "DualGroupDynamicBatchSampler",
    "FormulaPrecTypeGroupSampler",
    "GreedyInferenceBatchSampler",
    "GroupSampler",
    "SpecMolFragDynamicBatchSampler",
    "get_group_sampler",
    "MCESDataset",
    "SpecMolDataset",
    "SpecMolFragDataset",
    "IterableStreamBatchDataset",
]
