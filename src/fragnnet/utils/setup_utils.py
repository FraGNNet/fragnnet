from fragnnet.dataset import MCESDataset, SpecMolDataset, SpecMolFragDataset
from fragnnet.graff.dataset import SpecMolAnnDataset
from fragnnet.graff.pl_model import GrAFFPL
from fragnnet.iceberg.dataset import SpecMolMagmaGenDataset, SpecMolMagmaIntenDataset
from fragnnet.iceberg.pl_model import IcebergGenPL, IcebergIntenPL
from fragnnet.massformer.pl_model import MassFormerPL
from fragnnet.pl_model import (
    GNNPL,
    MCESPL,
    FraGNNetPL,
    NeimsPL,
    PrecursorPL,
    SpectrumMolClipPL,
)


def get_model_cls(model_type: str) -> type:
    """Get the model class for a given model type.

    Args:
        model_type (str): The type of the model.

    Raises:
        ValueError: If the model type is unknown.

    Returns:
        type: The model class corresponding to the model type.
    """
    if model_type in ["frag_gnn", "fraggnet"]:
        model_cls = FraGNNetPL
    elif model_type == "neims":
        model_cls = NeimsPL
    elif model_type == "iceberg_gen":
        model_cls = IcebergGenPL
    elif model_type == "iceberg_inten":
        model_cls = IcebergIntenPL
    elif model_type == "massformer":
        model_cls = MassFormerPL
    elif model_type == "graff":
        model_cls = GrAFFPL
    elif model_type == "precursor":
        model_cls = PrecursorPL
    elif model_type == "gnn":
        model_cls = GNNPL
    elif model_type == "mces_pretraining":
        model_cls = MCESPL
    elif model_type in ["clip", "spec_mol_clip_pretraining"]:
        model_cls = SpectrumMolClipPL
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model_cls


def get_dataset_cls(model_type: str) -> type:
    """Get the dataset class for a given model type.

    Args:
        model_type (str): The type of the model.

    Raises:
        ValueError: If the model type is unknown.

    Returns:
        type: The dataset class corresponding to the model type.
    """
    if model_type in ["frag_gnn", "fraggnet"]:
        dataset_cls = SpecMolFragDataset
    elif model_type == "iceberg_gen":
        dataset_cls = SpecMolMagmaGenDataset
    elif model_type == "iceberg_inten":
        dataset_cls = SpecMolMagmaIntenDataset
    elif model_type == "graff":
        dataset_cls = SpecMolAnnDataset
    elif model_type in [
        "neims",
        "massformer",
        "precursor",
        "gnn",
        "clip",
        "spec_mol_clip_pretraining",
    ]:
        dataset_cls = SpecMolDataset
    elif model_type == "mces_pretraining":
        dataset_cls = MCESDataset
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return dataset_cls
