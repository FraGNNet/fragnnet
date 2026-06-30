import os
from typing import Any

import pandas as pd
import torch as th
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from tqdm import tqdm

from fragnnet.utils.data_utils import min_max_normalize, zscore_normalize
from fragnnet.utils.feat_utils import get_mol_graph
from fragnnet.utils.misc_utils import get_mol_graph_size


class MCESDataset(Dataset):
    """Dataset for MCES (Maximum Common Edge Substructure) pairs.

    Each item is a pair of molecules (i, j) with their SMILES, MCES value, and PyG graphs.
    The dataset is constructed from a .npz file containing arrays for 'smiles' and 'mces'.
    """

    def __init__(
        self,
        mces_fp: str,
        mol_fp: str,
        split_dp: str,
        split: str,
        subsample_params: dict,
        mol_params: dict[str, Any],
        mol_pp_sd: dict[str | int, dict[str, Any]] | None = None,
        pl_enable_progress_bar: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize MCES dataset for molecular similarity learning.

        Args:
            mces_fp: File path to the MCES data pickle file.
            mol_fp: File path to the molecular data pickle file.
            split_dp: Directory path containing split CSV files.
            split: Name of the data split (e.g., 'train', 'val', 'test').
            subsample_params: Dictionary of subsampling parameters.
            mol_params: Dictionary of molecular processing parameters.
            mol_pp_sd: Optional shared dictionary for preprocessed molecular data.
            pl_enable_progress_bar: Whether to show tqdm progress bars during preprocessing.
            **kwargs: Additional keyword arguments including:
                - mces_normalization: Normalization type ('minmax', 'zscore', or 'none')
                - mces_min: Minimum MCES value for min-max normalization
                - mces_max: Maximum MCES value for min-max normalization
                - mces_mean: Mean MCES value for z-score normalization
                - mces_std: Standard deviation for z-score normalization
        """
        # Store split attribute (required by runner.py)
        self.split = split
        self.enable_progress_bar = pl_enable_progress_bar

        # Load data from .pkl file
        self.mces_df = pd.read_pickle(mces_fp)
        self.mol_params = mol_params
        self.mol_df = pd.read_pickle(mol_fp)
        self.mol_df = self.mol_df.reset_index(drop=True)
        # load split
        split_fp = os.path.join(split_dp, f"{split}_ids.csv")
        assert os.path.isfile(split_fp), split_fp
        split_df = pd.read_csv(split_fp)
        # this is a bit redundant but keep for clarity
        self.mces_df = self.mces_df[
            self.mces_df["mol_id_1"].isin(split_df["mol_id_1"])
            & self.mces_df["mol_id_2"].isin(split_df["mol_id_2"])
            & self.mces_df.index.isin(split_df["mces_id"])
        ].reset_index(drop=True)

        # Subsample if needed
        if subsample_params.get(split, False) and subsample_params["subsample_size"] > 0:
            if isinstance(subsample_params["subsample_size"], int):
                n = subsample_params["subsample_size"]
                frac = None
            else:
                assert isinstance(subsample_params["subsample_size"], float)
                n = None
                frac = subsample_params["subsample_size"]
            self.mces_df = self.mces_df.sample(
                n=n,
                frac=frac,
                random_state=subsample_params["subsample_seed"],
                replace=False,
            )

        self.mol_df = self.mol_df[
            self.mol_df["mol_id"].isin(self.mces_df["mol_id_1"])
            | self.mol_df["mol_id"].isin(self.mces_df["mol_id_2"])
        ]
        # use mol_id as index for speedy access
        self.mol_df = self.mol_df.set_index("mol_id", drop=False).sort_index().rename_axis(None)

        # Setup MCES normalization parameters from kwargs
        self.mces_normalization = kwargs.get("mces_normalization", "none")
        self.mces_min = kwargs.get("mces_min", 0.0)
        self.mces_max = kwargs.get("mces_max", 0.0)
        self.mces_mean = kwargs.get("mces_mean", 0.0)
        self.mces_std = kwargs.get("mces_std", 1.0)

        # Set normalization flag
        if self.mces_normalization == "minmax" and self.mces_max > self.mces_min:
            self.normalize_type = "minmax"
            print(
                f"> MCES normalization: min-max scaling [{self.mces_min}, {self.mces_max}] -> [0, 1]"
            )
        elif self.mces_normalization == "zscore" and self.mces_std > 0:
            self.normalize_type = "zscore"
            print(
                f"> MCES normalization: z-score (mean={self.mces_mean:.4f}, std={self.mces_std:.4f})"
            )
        else:
            self.normalize_type = None
            print("> MCES normalization: disabled")

        # Initialize shared dict for preprocessing
        if mol_pp_sd is None:
            mol_pp_sd = {}
        self.mol_pp_sd = mol_pp_sd

        # Only preprocess if data is empty (main process only)
        # This is not a problem til we run this on windows
        if self.mol_params.get("preprocess", False) and len(mol_pp_sd) == 0:
            # print("WARNING: Preprocessing in dataset init - should be done in main process!")
            self._preprocess_mol(mol_pp_sd)

    def _preprocess_mol(self, mol_pp_sd: dict) -> None:
        """
        Preload and pre-process all molecular data into shared dictionary.

        Args:
            mol_pp_sd: Shared dictionary to store preprocessed molecular data.

        Note:
            This method is copied from spec_mol_dataset.py.
            TODO: Move to a common molecular dataset base class.
        """
        # preload and pre-process molecules
        if self.mol_params["preprocess"]:
            self.mol_datas = mol_pp_sd
            total_mol_graph_size = 0
            for _, mol_entry in tqdm(
                self.mol_df.iterrows(),
                desc="> preprocess mol",
                total=len(self.mol_df),
                disable=not self.enable_progress_bar,
            ):
                mol_data = self._process_mol(mol_entry)
                total_mol_graph_size += get_mol_graph_size(mol_data, self.mol_params)
                self.mol_datas[mol_entry["mol_id"]] = mol_data
            print(f"> total_mol_graph_size: {total_mol_graph_size / 1e6:.2f} MB")

    def _process_mol(self, mol_entry: pd.Series) -> dict[str, Any]:
        """
        Process a single molecular entry into PyG graph format only.

        Args:
            mol_entry: DataFrame row containing molecular data

        Returns:
            dict: Dictionary containing PyG graph representation
        """
        mol_data = {}
        mol = mol_entry["mol"]

        # Generate PyTorch Geometric graph representation (PyG only for MCES)
        if self.mol_params.get("pyg", False):
            mol_pyg = get_mol_graph(
                mol,
                self.mol_params.get("pyg_node_feats", []),
                self.mol_params.get("pyg_edge_feats", []),
                self.mol_params.get("pyg_pe_embed_k", 0),
                self.mol_params.get("pyg_bigraph", False),
            )
            mol_data["mol_pyg"] = mol_pyg

        return mol_data

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a single MCES pair sample from the dataset.

        Args:
            idx: Index of the MCES pair to retrieve.

        Returns:
            Dictionary containing:
                - mces: MCES value (normalized if enabled)
                - mol_1_*: Keys from first molecule (e.g., mol_1_mol_pyg)
                - mol_2_*: Keys from second molecule (e.g., mol_2_mol_pyg)
        """
        mces_row = self.mces_df.iloc[idx]
        mces_value = mces_row["mces"]

        # Normalize MCES value if enabled
        if self.normalize_type == "minmax":
            # Min-max normalization: [min, max] -> [0, 1]
            mces_value = min_max_normalize(mces_value, self.mces_min, self.mces_max)
        elif self.normalize_type == "zscore":
            # Z-score normalization: (x - mean) / std
            mces_value = zscore_normalize(mces_value, self.mces_mean, self.mces_std)

        mol_id_1 = mces_row["mol_id_1"]
        mol_id_2 = mces_row["mol_id_2"]
        if self.mol_params["preprocess"]:
            # to do fix this, this is not really a deep copy
            mol_entry_1 = self.mol_datas[mol_id_1].copy()
            mol_entry_2 = self.mol_datas[mol_id_2].copy()
        else:
            mol_entry_1 = self._process_mol(self.mol_df.loc[mol_id_1])
            mol_entry_2 = self._process_mol(self.mol_df.loc[mol_id_2])

        result = {"mces": mces_value}

        # Add mol_entry_1 keys with mol_1_* prefix
        for key, value in mol_entry_1.items():
            result[f"mol_1_{key}"] = value

        # Add mol_entry_2 keys with mol_2_* prefix
        for key, value in mol_entry_2.items():
            result[f"mol_2_{key}"] = value

        return result

    def __len__(self) -> int:
        """
        Get the total number of MCES pairs in the dataset.

        Returns:
            Number of MCES pairs in the dataset.
        """
        return len(self.mces_df)

    @staticmethod
    def get_collate_fn():
        """
        Get the collate function for batching MCES dataset samples.

        Returns:
            Collate function to use with DataLoader.
        """
        return MCESDataset.collate_fn

    @staticmethod
    def collate_fn(data_list: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Collate a list of MCES data samples into a batched dictionary.

        Args:
            data_list: List of data dictionaries from dataset samples.

        Returns:
            Batched data dictionary ready for model input, containing:
                - mces: Batched MCES values as tensor
                - mol_1_mol_pyg: Batched PyG graphs for first molecules
                - mol_2_mol_pyg: Batched PyG graphs for second molecules
                - batch_size: Number of samples in the batch
        """

        batch_size = len(data_list)
        keys = list(data_list[0].keys())

        # Initialize collection structure
        collate_data: dict[str, Any] = {key: [] for key in keys}

        # Collect data for each key across all samples
        for data in data_list:
            for key in keys:
                collate_data[key].append(data[key])

        # Handle different data types
        for key in list(keys):
            values = collate_data[key]

            if key == "mces":
                # MCES values are floats, convert to tensor
                collate_data[key] = th.tensor(values, dtype=th.float)
            elif key.startswith("mol_1_") or key.startswith("mol_2_"):
                # Handle mol_1_* and mol_2_* keys
                # Extract the actual key name (e.g., "mol_pyg" from "mol_1_mol_pyg")
                actual_key = key.split("_", 2)[-1]  # Get part after "mol_1_" or "mol_2_"

                if actual_key == "mol_pyg" or "pyg" in actual_key:
                    # PyG molecular graphs need special batching
                    collate_data[key] = Batch.from_data_list(values)
                elif "smiles" in actual_key:
                    # SMILES strings - keep as list
                    collate_data[key] = values
                elif isinstance(values[0], th.Tensor):
                    # Standard tensors - concatenate
                    collate_data[key] = th.cat(values, dim=0)
                elif isinstance(values[0], list):
                    # Lists - flatten
                    flattened = []
                    for sublist in values:
                        flattened.extend(sublist)
                    collate_data[key] = flattened
                else:
                    # Keep as-is for other types
                    collate_data[key] = values
            elif isinstance(values[0], th.Tensor):
                # Standard tensors - concatenate
                collate_data[key] = th.cat(values, dim=0)
            elif isinstance(values[0], list):
                # Lists - flatten
                flattened = []
                for sublist in values:
                    flattened.extend(sublist)
                collate_data[key] = flattened
            else:
                # Keep as-is for other types
                collate_data[key] = values

        # Add batch size for reference
        collate_data["batch_size"] = th.tensor(batch_size, dtype=th.long)

        return collate_data

    @staticmethod
    def get_data_dict_types() -> list[str]:
        """
        Return list of data dictionary types used for caching.

        Returns:
            List containing data dictionary type identifiers.
                - 'mol_pp_sd': Molecular Preprocessed Shared Dictionary
        """
        return ["mol_pp_sd"]
