import collections
import copy
import os

import numpy as np
import pandas as pd
import torch as th
from torch.utils.data import Dataset
from tqdm import tqdm

from fragnnet.utils.data_utils import (
    CHARGE_FACTOR_MAP,
    fill_missing_ace,
    fill_missing_nce,
    get_charge,
)
from fragnnet.utils.formula_utils import PREC_TYPE_TO_MASS_DIFF
from fragnnet.utils.frag_utils import load_frag_d
from fragnnet.utils.isotope_utils import (
    detect_isotope_peaks_for_training_cleanup,
    estimate_coisolation_fraction_from_precursor,
)
from fragnnet.utils.misc_utils import flatten_lol, get_tensor_dict_memory_usage
from fragnnet.utils.proc_utils import merge_spec_df
from fragnnet.utils.spec_utils import batch_func


def _ce_list_to_stats(t: th.Tensor) -> th.Tensor:
    """Reduce one sample's CE list to (mean_raw, std_raw, valid) as shape (1, 3).

    Args:
        t: 1-D tensor of CE values for one sample; -1 means missing.

    Returns:
        Tensor of shape (1, 3): columns are (mean, std, valid_flag).
        All zeros when no valid CE is present.
    """
    valid = t[t >= 0].float()
    if len(valid) == 0:
        return th.zeros(1, 3, dtype=th.float32)
    mean = valid.mean()
    std = (valid - mean).pow(2).mean().sqrt()
    return th.stack([mean, std, th.ones(1, device=t.device)[0]], dim=0).unsqueeze(0)


class BaseDataset(Dataset):
    def _base_init(
        self,
        spec_fp: str,
        mol_fp: str,
        split_dp: str,
        split: str,
        subsample_params: dict,
        spec_params: dict,
        enable_progress_bar: bool = True,
    ) -> None:
        """
        Initialize base dataset with spectral and molecular data.

        Args:
            spec_fp: File path to the spectral data pickle file.
            mol_fp: File path to the molecular data pickle file.
            split_dp: Directory path containing split CSV files.
            split: Name of the data split (e.g., 'train', 'val', 'test').
            subsample_params: Dictionary of subsampling parameters.
            spec_params: Dictionary of spectral processing parameters.
            enable_progress_bar: Whether to show tqdm progress bars during preprocessing.
        """
        self.split = split
        self.subsample_params = subsample_params
        self.spec_params = spec_params
        self.enable_progress_bar = enable_progress_bar
        spec_df, mol_df, um_spec_df, split_df, id_key, ce_key = BaseDataset._setup_dfs(
            spec_fp_or_df=spec_fp,
            mol_fp_or_df=mol_fp,
            split_dp=split_dp,
            splits=[split],
            subsample_params=subsample_params,
            spec_params=spec_params,
        )
        self.spec_df = spec_df
        self.mol_df = mol_df
        self.um_spec_df = um_spec_df
        self.split_df = split_df
        self.id_key = id_key
        self.ce_key = ce_key
        self._setup_isotope_peak_cleanup()
        self._compute_counts()
        self._setup_prec_type_to_idx()
        self._setup_inst_type_to_idx()
        self._setup_frag_mode_to_idx()

    @staticmethod
    def _setup_dfs(
        spec_fp_or_df: str | pd.DataFrame,
        mol_fp_or_df: str | pd.DataFrame,
        split_dp: str,
        splits: list[str],
        subsample_params: dict,
        spec_params: dict,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str | None]:
        """
        Setup and process spectral and molecular dataframes with split information.

        Args:
            spec_fp_or_df: File path or DataFrame containing spectral data.
            mol_fp_or_df: File path or DataFrame containing molecular data.
            split_dp: Directory path containing split CSV files.
            splits: List of split names to process.
            subsample_params: Dictionary of subsampling parameters.
            spec_params: Dictionary of spectral processing parameters.

        Returns:
            Tuple containing:
                - spec_df: Processed spectral DataFrame
                - mol_df: Processed molecular DataFrame
                - um_spec_df: Unmerged spectral DataFrame
                - split_df: Split DataFrame
                - id_key: Identifier key ('spec_id' or 'group_id')
                - ce_key: Collision energy key ('ace', 'nce', or None)
        """
        spec_df = (
            spec_fp_or_df
            if isinstance(spec_fp_or_df, pd.DataFrame)
            else pd.read_pickle(spec_fp_or_df)
        )
        mol_df = (
            mol_fp_or_df if isinstance(mol_fp_or_df, pd.DataFrame) else pd.read_pickle(mol_fp_or_df)
        )

        split_dfs = []
        for split in splits:
            # assert split in ["train","val","test","secondary","predict_only"], split
            if split == "predict_only":
                assert len(splits) == 1, splits
                # predict_all split, just include everything, this is used for prediction
                split_df = pd.DataFrame()
                # fill these to keep compatible
                split_df["spec_id"] = spec_df["spec_id"]
                split_df["mol_id"] = spec_df["mol_id"]
                split_df["group_id"] = spec_df["spec_id"]
            else:
                split_fp = os.path.join(split_dp, f"{split}_ids.csv")
                assert os.path.isfile(split_fp), split_fp
                split_df = pd.read_csv(split_fp)
            # optionally subsample the split ids here to avoid loading/processing
            # large spec/mol DataFrames before sampling
            if subsample_params.get(split, False) and subsample_params["subsample_size"] > 0:
                if isinstance(subsample_params["subsample_size"], int):
                    n = subsample_params["subsample_size"]
                    frac = None
                else:
                    assert isinstance(subsample_params["subsample_size"], float)
                    n = None
                    frac = subsample_params["subsample_size"]
                split_df = split_df.sample(
                    n=n,
                    frac=frac,
                    random_state=subsample_params["subsample_seed"],
                    replace=False,
                ).reset_index(drop=True)
            split_dfs.append(split_df)
        split_df = pd.concat(split_dfs, ignore_index=True).reset_index(drop=True)

        # select spectra
        spec_df = spec_df[spec_df["spec_id"].isin(split_df["spec_id"])]
        # select molecules that actually have spectra in this split
        mol_df = mol_df[mol_df["mol_id"].isin(spec_df["mol_id"])]
        assert np.all(np.unique(mol_df["mol_id"]) == np.unique(spec_df["mol_id"]))

        # Resolve the per-split filter config.  In practice _setup_dfs is always called
        # with a single split (one dataset per split in init_dataset), so we look up
        # splits[0].  When called with multiple splits the per-split filter is skipped.
        _split_filter: dict = {}
        if len(splits) == 1:
            _split_filter = spec_params.get("split_filters", {}).get(splits[0], {})

        # Adduct filter: drop spectra whose prec_type is not in spec_params["prec_types"].
        # Applied before CE filling so Phase 1 only operates on retained spectra.
        if _split_filter.get("filter_prec_types", False):
            allowed_adducts = set(spec_params["prec_types"])
            orig_n = len(spec_df)
            spec_df = spec_df[spec_df["prec_type"].isin(allowed_adducts)]
            mol_df = mol_df[mol_df["mol_id"].isin(spec_df["mol_id"])]
            print(
                f">> split_filter [{splits[0]}] filter_prec_types "
                f"{sorted(allowed_adducts)}: {orig_n} → {len(spec_df)} spectra"
            )

        # Determine which CE columns to compute and which is primary.
        # nce/ace flags select the primary CE type (ce_key) fed to the model as spec_ce.
        # Both NCE and ACE are always computed when either flag is set so that spec_nce
        # and spec_ace are available in every batch for display / dual-CE input.
        if spec_params["nce"] and spec_params["ace"] or spec_params["nce"]:
            ce_keys_to_compute = [("nce", fill_missing_nce), ("ace", fill_missing_ace)]
            ce_key = "nce"
        elif spec_params["ace"]:
            ce_keys_to_compute = [("nce", fill_missing_nce), ("ace", fill_missing_ace)]
            ce_key = "ace"
        else:
            ce_keys_to_compute = []
            ce_key = None

        def _fill_ce_list(row, key):
            ce = row[key]
            ce_extra_1 = row[f"{key}_extra_1"]
            ce_extra_2 = row[f"{key}_extra_2"]

            # -1 sentinel means "no value"; normalise NaN to -1 for all three fields.
            if pd.isna(ce):
                ce = -1
            if pd.isna(ce_extra_1):
                ce_extra_1 = -1
            if pd.isna(ce_extra_2):
                ce_extra_2 = -1

            if ce != -1 and ce_extra_1 != -1 and ce_extra_2 != -1:
                # for stepped ce
                # ce is the lower bound, ce_extra_1 is the middle, ce_extra_2 is the upper bound
                assert ce < ce_extra_1 < ce_extra_2, row
                return [ce, ce_extra_1, ce_extra_2]
            elif ce != -1 and ce_extra_1 != -1:
                # for ramped ce
                ramped_step_size = 1.0
                # assert ce_extra_1 > ce, (ce, ce_extra_1)
                # assert (ce_extra_1 - ce) > ramped_step_size, (ce, ce_extra_1, ramped_step_size)
                ces = np.arange(ce, ce_extra_1 + 0.1, ramped_step_size).tolist()
                return ces
            else:
                # for single ce
                return [ce]

        if ce_keys_to_compute:
            # Phase 1: scalar CE fill (vectorized).
            # fill_missing_nce/ace each reduce to a single multiply/divide per row
            # (ace_to_nce_helper / nce_to_ace_helper). Parallelism overhead (pickling
            # each pd.Series row for IPC) dominates for such cheap per-row work, so
            # we vectorize with numpy instead — ~10-50x faster than parallel_apply.
            #
            # NOTE: must complete before Phase 2 so that fill_missing_ace can read
            # nce as a scalar (both columns must be scalars before either becomes a list).
            charge_factors = spec_df["prec_type"].map(
                lambda pt: CHARGE_FACTOR_MAP.get(np.abs(get_charge(pt)), CHARGE_FACTOR_MAP["large"])
            )
            prec_mz = spec_df["prec_mz"]

            def _fill_ce_scalar_vectorized(ace_col: str, nce_col: str, target_key: str) -> None:
                ace = spec_df[ace_col].where(spec_df[ace_col].notna() & (spec_df[ace_col] != -1))
                nce = spec_df[nce_col].where(spec_df[nce_col].notna() & (spec_df[nce_col] != -1))
                if target_key.startswith("nce"):
                    # fill missing nce from ace: nce = ace * 500 / (prec_mz * charge_factor)
                    computed = (ace * 500.0) / (prec_mz * charge_factors)
                    result = pd.to_numeric(nce, errors="coerce").fillna(
                        pd.to_numeric(computed, errors="coerce")
                    )
                else:
                    # fill missing ace from nce: ace = nce * prec_mz * charge_factor / 500
                    computed = (nce * prec_mz * charge_factors) / 500.0
                    result = pd.to_numeric(ace, errors="coerce").fillna(
                        pd.to_numeric(computed, errors="coerce")
                    )
                spec_df.loc[:, target_key] = result

            for key, _ in ce_keys_to_compute:
                for endfix in ["", "_extra_1", "_extra_2"]:
                    _fill_ce_scalar_vectorized(
                        ace_col=f"ace{endfix}", nce_col=f"nce{endfix}", target_key=f"{key}{endfix}"
                    )
                    if len(endfix) > 0:
                        spec_df.loc[:, f"{key}{endfix}"] = spec_df.loc[:, f"{key}{endfix}"].fillna(
                            -1
                        )

        # CE range filter by instrument type (after Phase 1 scalar fill, before Phase 2
        # list conversion so we can compare plain floats rather than lists).
        # Config shape (under spec_params.split_filters.<split>.filter_ce_by_inst):
        #   FT:   {nce_min: 0, nce_max: 60}
        #   QTOF: {ace_min: 0, ace_max: 40}
        # Instruments absent from the dict are not filtered.
        # Spectra of a listed instrument whose CE is missing/unfillable are dropped.
        # For ramped/stepped CEs ALL scalar CE columns (primary + extra_1 + extra_2)
        # must satisfy the range — e.g. a ramp [10–70] fails nce_max: 60.
        # NOTE: ace filters read the raw ace column and do NOT require ace: true.
        #   QTOF spectra always carry raw ace, so ace filters work with nce: true too.
        #   nce filters use Phase-1-filled values; require nce: true for correct results.
        _filter_ce_by_inst: dict = _split_filter.get("filter_ce_by_inst", {})
        if _filter_ce_by_inst:

            def _ce_range_valid(col_primary: str, col_e1: str, col_e2: str, lo, hi) -> pd.Series:
                """Return boolean Series: True when ALL present CE scalars are in [lo, hi].

                A CE column is considered present when it is neither NaN nor the -1
                sentinel.  The extra columns (col_e1, col_e2) are optional; they are
                only checked when present so that single-CE spectra are not penalised.
                """
                primary = spec_df[col_primary]
                e1 = spec_df[col_e1]
                e2 = spec_df[col_e2]

                e1_present = e1.notna() & (e1 != -1)
                e2_present = e2.notna() & (e2 != -1)

                # Primary must be valid (not missing)
                valid = primary.notna() & (primary != -1)
                if lo is not None:
                    valid &= primary >= lo
                    valid &= ~e1_present | (e1 >= lo)
                    valid &= ~e2_present | (e2 >= lo)
                if hi is not None:
                    valid &= primary <= hi
                    valid &= ~e1_present | (e1 <= hi)
                    valid &= ~e2_present | (e2 <= hi)
                return valid

            keep = pd.Series(True, index=spec_df.index)
            for _inst, _ce_cfg in _filter_ce_by_inst.items():
                _inst_mask = spec_df["inst_type"] == _inst
                _nce_min = _ce_cfg.get("nce_min", None)
                _nce_max = _ce_cfg.get("nce_max", None)
                _ace_min = _ce_cfg.get("ace_min", None)
                _ace_max = _ce_cfg.get("ace_max", None)
                if _nce_min is not None or _nce_max is not None:
                    _valid = _ce_range_valid(
                        "nce", "nce_extra_1", "nce_extra_2", _nce_min, _nce_max
                    )
                    # Only enforce for spectra of this instrument type
                    keep &= ~_inst_mask | _valid
                if _ace_min is not None or _ace_max is not None:
                    _valid = _ce_range_valid(
                        "ace", "ace_extra_1", "ace_extra_2", _ace_min, _ace_max
                    )
                    keep &= ~_inst_mask | _valid
            orig_n = len(spec_df)
            spec_df = spec_df[keep]
            mol_df = mol_df[mol_df["mol_id"].isin(spec_df["mol_id"])]
            print(
                f">> split_filter [{splits[0]}] filter_ce_by_inst "
                f"{list(_filter_ce_by_inst)}: {orig_n} → {len(spec_df)} spectra"
            )
            # Print CE distribution per filtered instrument type.
            for _inst, _ce_cfg in _filter_ce_by_inst.items():
                _inst_rows = spec_df[spec_df["inst_type"] == _inst]
                _ce_col = "ace" if ("ace_min" in _ce_cfg or "ace_max" in _ce_cfg) else "nce"
                _ce_vals = _inst_rows[_ce_col]
                _ce_vals = _ce_vals[_ce_vals.notna() & (_ce_vals != -1)]
                if len(_ce_vals) == 0:
                    print(f"   [{_inst}] {_ce_col}: no valid values after filter")
                    continue
                _q = _ce_vals.quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
                print(
                    f"   [{_inst}] {_ce_col} distribution (n={len(_ce_vals)}): "
                    f"min={_q[0.0]:.1f}  p10={_q[0.1]:.1f}  p25={_q[0.25]:.1f}  "
                    f"p50={_q[0.5]:.1f}  p75={_q[0.75]:.1f}  p90={_q[0.9]:.1f}  "
                    f"max={_q[1.0]:.1f}"
                )

        if ce_keys_to_compute:
            # Phase 2: convert scalar CE columns to lists (stepped / ramped / single).
            # _fill_ce_list does NaN checks + possibly np.arange over a small range —
            # microseconds per row. Process-based parallelism (joblib / pandarallel)
            # serializes each pd.Series row via pickle, costing ~50-200µs overhead per
            # row — more than the work itself. Plain .apply() is faster here.
            for key, _ in ce_keys_to_compute:
                result_series = spec_df.apply(lambda row, _k=key: _fill_ce_list(row, _k), axis=1)
                spec_df[key] = spec_df[key].astype(object)
                spec_df.loc[:, key] = result_series

        if spec_params["test_ces"] is not None and ce_key is not None and "test" in splits:
            test_split_fp = os.path.join(split_dp, "test_ids.csv")
            test_split_df = pd.read_csv(test_split_fp)
            orig_test_spec_df = spec_df[spec_df["spec_id"].isin(test_split_df["spec_id"])]
            test_ces_set = set(spec_params["test_ces"])
            test_drop_ids = orig_test_spec_df[
                ~orig_test_spec_df[ce_key].apply(
                    lambda ce_list: any(ce in test_ces_set for ce in ce_list)
                )
            ][["spec_id", "group_id"]]
            spec_df = spec_df[~(spec_df["spec_id"].isin(test_drop_ids["spec_id"]))]
            new_test_spec_df = spec_df[spec_df["spec_id"].isin(test_split_df["spec_id"])]
            orig_test_spec_count = orig_test_spec_df["spec_id"].nunique()
            orig_test_group_count = orig_test_spec_df["group_id"].nunique()
            new_test_spec_count = new_test_spec_df["spec_id"].nunique()
            new_test_group_count = new_test_spec_df["group_id"].nunique()
            print(f">> Dropping spectra unless {ce_key} is one of {spec_params['test_ces']}")
            print(f"> Before drop: {orig_test_spec_count} spectra, {orig_test_group_count} groups")
            print(f"> After drop: {new_test_spec_count} spectra, {new_test_group_count} groups")

        # merge spectra
        um_spec_cols = [
            "dset",
            "dset_spec_id",
            "spec_id",
            "group_id",
            "mol_id",
            "prec_type",
            "inst_type",
            "prec_mz",
            "peaks",
        ]
        um_spec_df = spec_df[[col for col in um_spec_cols if col in spec_df.columns]].copy()
        if spec_params["merge"]:
            spec_df = merge_spec_df(spec_df, keep_ces=spec_params["merge_keep_ces"])
            id_key = "group_id"
        else:
            id_key = "spec_id"

        # NOTE: subsampling is performed on the split ID lists earlier
        # (while reading the split files) to avoid processing large
        # `spec_df` and `mol_df` before sampling.

        # reset indices
        spec_df = spec_df.reset_index(drop=True)
        mol_df = mol_df.reset_index(drop=True)
        # use mol_id as index for speedy access
        mol_df = mol_df.set_index("mol_id", drop=False).sort_index().rename_axis(None)
        return spec_df, mol_df, um_spec_df, split_df, id_key, ce_key

    def _setup_isotope_peak_cleanup(self) -> None:
        """Precompute group-level co-isolation fractions for isotope cleanup."""
        self.group_coiso_fraction_by_k: dict[int, dict[int, float]] = {}
        self._isotope_dag_formula_cache: dict[int, tuple[np.ndarray, np.ndarray] | None] = {}
        if not self.spec_params.get("remove_isotope_peaks", False):
            return
        if not self.spec_params.get("remove_isotope_use_group_f", True):
            return
        if self.um_spec_df.empty:
            return
        required_cols = {"group_id", "mol_id", "prec_type", "inst_type", "prec_mz", "peaks"}
        if not required_cols.issubset(self.um_spec_df.columns):
            return

        max_isotope = self.spec_params.get("remove_isotope_max_isotope", 2)
        min_coiso_fraction = self.spec_params.get("remove_isotope_min_coiso_fraction", 0.05)
        max_leak_factor = self.spec_params.get("remove_isotope_max_leak_factor", 1.5)
        precursor_envelope_lo = self.spec_params.get("remove_isotope_precursor_envelope_lo", 0.2)
        precursor_envelope_hi = self.spec_params.get("remove_isotope_precursor_envelope_hi", 5.0)
        min_group_count = self.spec_params.get("remove_isotope_group_min_count", 1)
        max_group_cv = self.spec_params.get("remove_isotope_group_max_cv", 1.0)

        vals_by_group_k: dict[tuple[int, int], list[float]] = collections.defaultdict(list)
        for _, spec_entry in self.um_spec_df.iterrows():
            mol_id = spec_entry["mol_id"]
            if mol_id not in self.mol_df.index:
                continue
            formula = self.mol_df.loc[mol_id]["formula"]
            peaks = spec_entry["peaks"]
            if not peaks:
                continue
            mzs, ints = self._get_mzs_ints(peaks)
            f_by_k = estimate_coisolation_fraction_from_precursor(
                mzs.numpy(),
                ints.numpy(),
                prec_mz=float(spec_entry["prec_mz"]),
                inst_type=str(spec_entry["inst_type"]),
                formula=str(formula),
                prec_type=str(spec_entry["prec_type"]),
                max_isotope=max_isotope,
                min_coiso_fraction=min_coiso_fraction,
                max_leak_factor=max_leak_factor,
                precursor_envelope_lo=precursor_envelope_lo,
                precursor_envelope_hi=precursor_envelope_hi,
            )
            for k, f in f_by_k.items():
                vals_by_group_k[(int(spec_entry["group_id"]), int(k))].append(float(f))

        for (group_id, k), vals in vals_by_group_k.items():
            if len(vals) < min_group_count:
                continue
            vals_arr = np.asarray(vals, dtype=np.float64)
            med = float(np.median(vals_arr))
            if med <= 0:
                continue
            cv = float(np.std(vals_arr) / med)
            if cv > max_group_cv:
                continue
            self.group_coiso_fraction_by_k.setdefault(group_id, {})[k] = med

    def _get_isotope_dag_formula_arrays(
        self,
        mol_id: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return DAG M+0 m/z and formula arrays for isotope cleanup if configured."""
        if not self.spec_params.get("remove_isotope_use_dag_formula", True):
            return None

        dag_dp = self.spec_params.get("remove_isotope_dag_dp")
        if not dag_dp:
            return None

        mol_id = int(mol_id)
        if mol_id in self._isotope_dag_formula_cache:
            return self._isotope_dag_formula_cache[mol_id]

        try:
            frag_entry = load_frag_d(
                mol_id,
                str(dag_dp),
                is_compressed=self.spec_params.get("remove_isotope_dag_compressed", True),
            )
        except Exception:
            self._isotope_dag_formula_cache[mol_id] = None
            return None

        if not frag_entry or "formula_peak_mzs" not in frag_entry:
            self._isotope_dag_formula_cache[mol_id] = None
            return None

        formula_peak_mzs = frag_entry["formula_peak_mzs"]
        if hasattr(formula_peak_mzs, "detach"):
            formula_peak_mzs = formula_peak_mzs.detach().cpu().numpy()
        else:
            formula_peak_mzs = np.asarray(formula_peak_mzs)
        if formula_peak_mzs.ndim != 2 or formula_peak_mzs.shape[0] == 0:
            self._isotope_dag_formula_cache[mol_id] = None
            return None

        idx_to_formula = frag_entry.get("idx_to_formula", {})
        dag_formula_mzs = formula_peak_mzs[:, 0].astype(np.float64, copy=False)
        dag_formula_strs = np.array(
            [idx_to_formula.get(i, "") for i in range(len(dag_formula_mzs))],
            dtype=object,
        )
        out = (dag_formula_mzs, dag_formula_strs)
        self._isotope_dag_formula_cache[mol_id] = out
        return out

    def _compute_counts(self) -> None:
        """
        Compute and store counts of spectra per molecule, per group, and groups per molecule.

        Sets the following instance attributes:
            - group_per_mol: Dictionary mapping molecule IDs to group counts.
            - spec_per_mol: Dictionary mapping molecule IDs to spectrum counts.
            - spec_per_group: Dictionary mapping group IDs to spectrum counts.
        """
        self.group_per_mol = (
            self.spec_df[["mol_id", "group_id"]]
            .drop_duplicates()
            .groupby("mol_id")
            .size()
            .to_dict()
        )
        if self.spec_params["merge"]:
            self.spec_per_mol = copy.deepcopy(self.group_per_mol)
            self.spec_per_group = dict.fromkeys(self.spec_df["group_id"].unique(), 1)
        else:
            self.spec_per_mol = (
                self.spec_df[["mol_id", "spec_id"]]
                .drop_duplicates()
                .groupby("mol_id")
                .size()
                .to_dict()
            )
            self.spec_per_group = (
                self.spec_df[["group_id", "spec_id"]]
                .drop_duplicates()
                .groupby("group_id")
                .size()
                .to_dict()
            )

    def get_group_mol_stats(self) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Retrieve statistics about groups, molecules, and spectra counts.

        Returns:
            Tuple of tensors containing:
                - group_ids: Group IDs for each spectrum
                - mol_ids: Molecule IDs for each spectrum
                - spec_per_group_stats: Number of spectra per group
                - spec_per_mol_stats: Number of spectra per molecule
                - group_per_mol_stats: Number of groups per molecule
        """
        group_ids = []
        mol_ids = []
        spec_per_group_stats = []
        spec_per_mol_stats = []
        group_per_mol_stats = []
        for _, row in self.spec_df.iterrows():
            spec_entry = row
            mol_id = spec_entry["mol_id"]
            group_id = spec_entry["group_id"]
            spec_per_group = self.spec_per_group[group_id]
            spec_per_mol = self.spec_per_mol[mol_id]
            group_per_mol = self.group_per_mol[mol_id]
            group_ids.append(group_id)
            spec_per_group_stats.append(spec_per_group)
            mol_ids.append(mol_id)
            spec_per_mol_stats.append(spec_per_mol)
            group_per_mol_stats.append(group_per_mol)
        group_ids = th.tensor(group_ids)
        mol_ids = th.tensor(mol_ids)
        spec_per_group_stats = th.tensor(spec_per_group_stats)
        spec_per_mol_stats = th.tensor(spec_per_mol_stats)
        group_per_mol_stats = th.tensor(group_per_mol_stats)
        return (
            group_ids,
            mol_ids,
            spec_per_group_stats,
            spec_per_mol_stats,
            group_per_mol_stats,
        )

    def get_adduct_inst_type_stats(self) -> tuple[list[str], list[str]]:
        """Return per-sample precursor type and instrument type strings.

        Used by get_group_sampler() for adduct+instrument-type balanced sampling.

        Returns:
            Tuple of (prec_types, inst_types) where each is a list of strings
            with length equal to len(self).
        """
        prec_types = self.spec_df["prec_type"].astype(str).tolist()
        inst_types = self.spec_df["inst_type"].astype(str).tolist()
        return prec_types, inst_types

    def get_frag_mode_stats(self) -> list[str]:
        """Return per-sample fragmentation mode strings.

        Missing or NaN values are returned as "unknown".
        Used by get_group_sampler() for frag-mode balanced sampling.

        Returns:
            List of fragmentation mode strings (e.g. "HCD", "CID", "unknown")
            with length equal to len(self).
        """
        if "frag_mode" not in self.spec_df.columns:
            return ["unknown"] * len(self.spec_df)
        return self.spec_df["frag_mode"].fillna("unknown").astype(str).tolist()

    @staticmethod
    def get_data_dict_types() -> list[str]:
        """
        Return list of data dictionary types used for caching.

        Returns:
            List containing data dictionary type identifiers.
            - 'spec_pp_sd': Spectral Preprocessed Shared Dictionary
        """
        return ["spec_pp_sd"]

    def _preprocess_spec(self, spec_pp_sd: dict) -> None:
        """
        Preload and pre-process all spectral data into shared dictionary.

        Args:
            spec_pp_sd: Shared dictionary to store preprocessed spectral data.
        """
        # preload and pre-process spectra
        if self.spec_params["preprocess"]:
            self.spec_datas = spec_pp_sd
            total_spec_data_size = 0
            for idx, spec_entry in tqdm(
                self.spec_df.iterrows(),
                desc="> preprocess spec",
                total=len(self.spec_df),
                disable=not self.enable_progress_bar,
            ):
                spec_data = self._process_spec(spec_entry)
                total_spec_data_size += get_tensor_dict_memory_usage(**spec_data)
                self.spec_datas[idx] = spec_data
            print(f"> total_spec_data_size: {total_spec_data_size / 1e6:.2f} MB")

    @staticmethod
    def _get_mzs_ints(peaks: list) -> tuple[th.Tensor, th.Tensor]:
        """
        Convert peak list to m/z and intensity tensors.

        Args:
            peaks: List of (m/z, intensity) tuples.

        Returns:
            Tuple containing:
                - mzs: Tensor of m/z values
                - ints: Tensor of intensity values

        Note:
            It is the caller's responsibility to ensure data is valid.
        """
        mzs, ints = [], []
        for peak in peaks:
            p_mz, p_int = peak
            mzs.append(p_mz)
            ints.append(p_int)

        mzs = th.tensor(mzs, dtype=th.float)
        ints = th.tensor(ints, dtype=th.float)
        # mzs, ints = filter_func(mzs, ints, self.spec_params["ints_thresh"], self.spec_params["mz_max"])
        return mzs, ints

    def _setup_prec_type_to_idx(self) -> None:
        """
        Setup bidirectional mappings between precursor types and indices.

        Sets the following instance attributes:
            - prec_type_to_idx: Dictionary mapping precursor type strings to indices.
            - idx_to_prec_type: Dictionary mapping indices to precursor type strings.
            - num_prec_types: Total number of precursor types.
        """
        prec_types = sorted(self.spec_params["prec_types"])
        assert all(prec_type in PREC_TYPE_TO_MASS_DIFF for prec_type in prec_types), prec_types
        self.prec_type_to_idx = {prec_type: idx for idx, prec_type in enumerate(prec_types)}
        self.idx_to_prec_type = dict(enumerate(prec_types))
        self.num_prec_types = len(prec_types)

    def _setup_inst_type_to_idx(self) -> None:
        """
        Setup bidirectional mappings between instrument types and indices.

        Sets the following instance attributes:
            - inst_type_to_idx: Dictionary mapping instrument type strings to indices.
            - idx_to_inst_type: Dictionary mapping indices to instrument type strings.
            - num_inst_types: Total number of instrument types.
        """
        inst_types = sorted(self.spec_params["inst_types"])
        self.inst_type_to_idx = {inst_type: idx for idx, inst_type in enumerate(inst_types)}
        self.idx_to_inst_type = dict(enumerate(inst_types))
        self.num_inst_types = len(inst_types)

    def _setup_frag_mode_to_idx(self) -> None:
        """
        Setup bidirectional mappings between fragmentation modes and indices.

        Known modes (e.g. HCD, CID) are mapped to consecutive indices.  Any value
        not in the known set (including missing / NaN) is mapped to the fallback
        index ``num_frag_modes`` (the unknown token used by the embedding layer).

        Sets the following instance attributes:
            - frag_mode_to_idx: Dictionary mapping frag-mode strings to indices.
            - idx_to_frag_mode: Dictionary mapping indices to frag-mode strings.
            - num_frag_modes: Number of known fragmentation modes (excluding unknown).
        """
        frag_modes = sorted(self.spec_params.get("frag_modes", []))
        self.frag_mode_to_idx = {mode: idx for idx, mode in enumerate(frag_modes)}
        self.idx_to_frag_mode = dict(enumerate(frag_modes))
        self.num_frag_modes = len(frag_modes)

    def _process_spec(self, spec_entry: pd.Series) -> dict:
        """
        Process a single spectral entry into model-ready format.

        Args:
            spec_entry: DataFrame row containing spectral data.

        Returns:
            Dictionary containing processed spectral data with keys depending on spec_params:
                - spec_mzs: m/z values (if sparse)
                - spec_ints: Intensity values (if sparse)
                - spec_prec_type: Precursor type index (if prec_type enabled)
                - spec_prec_type_str: Precursor type string (if prec_type_str enabled)
                - spec_inst_type: Instrument type index (if inst_type enabled)
                - spec_frag_mode: Fragmentation mode index (if frag_mode enabled; unknown=num_frag_modes)
                - spec_prec_mass_diff: Precursor mass difference (if prec_mass_diff enabled)
                - spec_ce: Collision energy values (if nce or ace enabled)
                - spec_prec_mz: Precursor m/z value (if prec_mz enabled)
                - spec_unique_id: Unique identifier (if unique_id enabled)
                - group_id: Group identifier (if unique_id enabled)
                - mol_id: Molecule identifier (if unique_id enabled)
                - spec_per_mol: Spectra count per molecule (if counts enabled)
                - group_per_mol: Groups count per molecule (if counts enabled)
                - spec_per_group: Spectra count per group (if counts enabled)
        """
        spec_data = {}
        # peak data
        mzs, ints = BaseDataset._get_mzs_ints(spec_entry["peaks"])
        if self.spec_params.get("remove_isotope_peaks", False):
            mzs, ints = self._remove_isotope_peaks_from_spec(spec_entry, mzs, ints)
        if self.spec_params["sparse"]:
            # get sparse spectrum
            spec_data["spec_mzs"] = mzs
            spec_data["spec_ints"] = ints
        # metadata
        if self.spec_params["prec_type"]:
            prec_type = spec_entry["prec_type"]
            prec_type = th.tensor([self.prec_type_to_idx[prec_type]], dtype=th.long)
            spec_data["spec_prec_type"] = prec_type
        if self.spec_params["prec_type_str"]:
            prec_type_str = spec_entry["prec_type"]
            spec_data["spec_prec_type_str"] = np.array([prec_type_str])
        if self.spec_params["inst_type"]:
            inst_type = spec_entry["inst_type"]
            inst_type = th.tensor([self.inst_type_to_idx[inst_type]], dtype=th.long)
            spec_data["spec_inst_type"] = inst_type
        if self.spec_params.get("frag_mode", False):
            raw_mode = spec_entry.get("frag_mode", None)
            # Fall back to unknown token index when the column is missing or the value
            # is not in the configured known set (handles NaN, None, and unseen labels).
            if raw_mode is None or (
                isinstance(raw_mode, float) and __import__("math").isnan(raw_mode)
            ):
                frag_mode_idx = self.num_frag_modes  # unknown token
            else:
                frag_mode_idx = self.frag_mode_to_idx.get(str(raw_mode), self.num_frag_modes)
            spec_data["spec_frag_mode"] = th.tensor([frag_mode_idx], dtype=th.long)
        if self.spec_params["prec_mass_diff"]:
            prec_type = spec_entry["prec_type"]
            mass_diff = th.tensor([PREC_TYPE_TO_MASS_DIFF[prec_type]], dtype=th.float)
            spec_data["spec_prec_mass_diff"] = mass_diff
        if self.spec_params["nce"] or self.spec_params["ace"]:
            assert self.ce_key is not None
            assert (not self.spec_params["merge"]) or self.spec_params["merge_keep_ces"]
            ce = spec_entry[self.ce_key]
            # both merged and unmerged spectra are stored as list of ces
            # for merged spectra, it is list of ces used to merge
            # for unmerged spectra, it is list of ces available include both ramped/stepped ces
            assert isinstance(ce, list), type(ce)
            spec_data["spec_ce"] = th.tensor(ce, dtype=th.float)
            # Always include both spec_nce and spec_ace so batches carry both CE types
            # regardless of which is primary (ce_key). Used for per-instrument display and
            # dual-CE input. Both are computed in _setup_dfs whenever either flag is set.
            nce_val = spec_entry["nce"]
            ace_val = spec_entry["ace"]
            assert isinstance(nce_val, list), type(nce_val)
            assert isinstance(ace_val, list), type(ace_val)
            spec_data["spec_nce"] = th.tensor(nce_val, dtype=th.float)
            spec_data["spec_ace"] = th.tensor(ace_val, dtype=th.float)
        if self.spec_params["prec_mz"]:
            prec_mz = spec_entry["prec_mz"]
            prec_mz = th.tensor([float(prec_mz)], dtype=th.float)
            spec_data["spec_prec_mz"] = prec_mz
        if self.spec_params["unique_id"]:
            unique_id = spec_entry[self.id_key]
            unique_id = th.tensor([unique_id], dtype=th.long)
            spec_data["spec_unique_id"] = unique_id
            spec_data["group_id"] = th.tensor([spec_entry["group_id"]], dtype=th.long)
            spec_data["mol_id"] = spec_entry[
                "mol_id"
            ]  # mol id does not need to be an int #th.tensor([spec_entry['mol_id']],dtype=th.long)
        if self.spec_params["counts"]:
            spec_per_mol = self.spec_per_mol[spec_entry["mol_id"]]
            spec_per_mol = th.tensor([spec_per_mol], dtype=th.long)
            spec_data["spec_per_mol"] = spec_per_mol
            group_per_mol = self.group_per_mol[spec_entry["mol_id"]]
            group_per_mol = th.tensor([group_per_mol], dtype=th.long)
            spec_data["group_per_mol"] = group_per_mol
            spec_per_group = self.spec_per_group[spec_entry["group_id"]]
            spec_per_group = th.tensor([spec_per_group], dtype=th.long)
            spec_data["spec_per_group"] = spec_per_group
        return spec_data

    def _remove_isotope_peaks_from_spec(
        self, spec_entry: pd.Series, mzs: th.Tensor, ints: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor]:
        """Remove isotope peaks from a spectrum for monoisotopic training targets."""
        if mzs.numel() == 0:
            return mzs, ints
        required_cols = {"group_id", "mol_id", "prec_type", "inst_type", "prec_mz"}
        if not required_cols.issubset(spec_entry.index):
            return mzs, ints

        mol_id = spec_entry["mol_id"]
        formula = str(self.mol_df.loc[mol_id]["formula"]) if mol_id in self.mol_df.index else None
        group_f = self.group_coiso_fraction_by_k.get(int(spec_entry["group_id"]))
        dag_formula_arrays = self._get_isotope_dag_formula_arrays(int(mol_id))
        dag_formula_mzs = dag_formula_arrays[0] if dag_formula_arrays is not None else None
        dag_formula_strs = dag_formula_arrays[1] if dag_formula_arrays is not None else None
        remove_mask_np = detect_isotope_peaks_for_training_cleanup(
            mzs.numpy(),
            ints.numpy(),
            prec_mz=float(spec_entry["prec_mz"]),
            inst_type=str(spec_entry["inst_type"]),
            formula=formula,
            prec_type=str(spec_entry["prec_type"]),
            max_isotope=self.spec_params.get("remove_isotope_max_isotope", 2),
            remove_regular_isotopes=self.spec_params.get("remove_regular_isotope_peaks", True),
            remove_coisolation=self.spec_params.get("remove_coisolation_peaks", True),
            regular_mz_tol=self.spec_params.get("remove_isotope_regular_mz_tol"),
            coiso_fraction_by_k=group_f,
            min_coiso_fraction=self.spec_params.get("remove_isotope_min_coiso_fraction", 0.05),
            max_leak_factor=self.spec_params.get("remove_isotope_max_leak_factor", 1.5),
            precursor_envelope_lo=self.spec_params.get(
                "remove_isotope_precursor_envelope_lo", 0.2
            ),
            precursor_envelope_hi=self.spec_params.get(
                "remove_isotope_precursor_envelope_hi", 5.0
            ),
            dag_formula_mzs=dag_formula_mzs,
            dag_formula_strs=dag_formula_strs,
            dag_mz_tol=self.spec_params.get("remove_isotope_dag_mz_tol", 0.015),
            protect_dag_mono=self.spec_params.get("remove_isotope_protect_dag_mono", False),
        )
        if not remove_mask_np.any():
            return mzs, ints

        keep_mask = ~th.as_tensor(remove_mask_np, dtype=th.bool)
        # Avoid returning an empty target spectrum; keep the original if every
        # peak was classified as isotope.
        if not keep_mask.any():
            return mzs, ints
        return mzs[keep_mask], ints[keep_mask]

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single data sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Dictionary containing the processed data sample.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError()

    def __len__(self) -> int:
        """
        Get the total number of samples in the dataset.

        Returns:
            Number of spectral entries in the dataset.
        """
        return len(self.spec_df)

    @staticmethod
    def get_collate_fn():
        """
        Get the collate function for batching dataset samples.

        Returns:
            Collate function to use with DataLoader.
        """
        return BaseDataset.collate_fn

    @staticmethod
    def _setup_collate(data_list: list[dict]) -> tuple[int, list[str], dict]:
        """
        Setup initial collation structure for batching.

        Args:
            data_list: List of data dictionaries from dataset samples.

        Returns:
            Tuple containing:
                - batch_size: Number of samples in the batch
                - keys: List of data keys to collate
                - collate_data: Dictionary with lists of values for each key
        """
        batch_size = len(data_list)
        keys = list(data_list[0].keys())
        collate_data = {key: [] for key in keys}
        for data in data_list:
            for key in keys:
                collate_data[key].append(data[key])
        return batch_size, keys, collate_data

    @staticmethod
    def _special_collate(keys: list[str], collate_data: dict) -> None:
        """
        Handle special collation for sparse spectral data and collision energies.

        Mutates keys and collate_data in-place to batch sparse tensors efficiently.

        Args:
            keys: List of data keys being collated (modified in-place).
            collate_data: Dictionary containing lists of values to collate (modified in-place).
        """
        # handle sparse spectra
        if "spec_ints" in keys and "spec_mzs" in keys:
            # create batch_idxs
            mzs, ints, batch_idxs = batch_func(collate_data["spec_mzs"], collate_data["spec_ints"])
            collate_data["spec_mzs"] = mzs
            collate_data["spec_ints"] = ints
            collate_data["spec_batch_idxs"] = batch_idxs
            # remove from list
            keys.remove("spec_ints")
            keys.remove("spec_mzs")

        # handle sparse ces
        if "spec_ce" in keys:
            # create batch_idxs
            ces, _, batch_idxs = batch_func(
                collate_data["spec_ce"],
                collate_data["spec_ce"],  # duplicate for compatibility
            )
            collate_data["spec_ce"] = ces
            collate_data["spec_ce_batch_idxs"] = batch_idxs
            # remove from list
            keys.remove("spec_ce")
        for _ce_key in ("spec_nce", "spec_ace"):
            if _ce_key in keys:
                # Pre-reduce to (mean, std, valid) stats for CEScaler — no scatter in model.
                # Each sample contributes a (1, 3) tensor; _standard_collate cats to (B, 3).
                stats_key = f"{_ce_key}_stats"
                collate_data[stats_key] = [_ce_list_to_stats(t) for t in collate_data[_ce_key]]
                keys.append(stats_key)
                # Keep raw concat + batch_idxs for _embed_single_ce (averages embeddings).
                _ces, _, _batch_idxs = batch_func(
                    collate_data[_ce_key],
                    collate_data[_ce_key],
                )
                collate_data[_ce_key] = _ces
                collate_data[f"{_ce_key}_batch_idxs"] = _batch_idxs
                keys.remove(_ce_key)

    @staticmethod
    def _standard_collate(batch_size: int, keys: list[str], collate_data: dict) -> None:
        """
        Perform standard collation for generic data types.

        Mutates keys and collate_data in-place. Handles lists, tensors, and numpy arrays.

        Args:
            batch_size: Number of samples in the batch.
            keys: List of data keys being collated (modified in-place).
            collate_data: Dictionary containing lists of values to collate (modified in-place).

        Raises:
            TypeError: If an unsupported data type is encountered.
        """

        # handle generic data
        for key in keys:
            values = collate_data[key]
            if isinstance(values[0], list):
                # flatten
                values = flatten_lol(values)
                collate_data[key] = values
            elif isinstance(values[0], th.Tensor):
                # cat
                values = th.cat(values, dim=0)
                collate_data[key] = values
            elif isinstance(values[0], np.ndarray):
                # cat
                values = np.concatenate(values, axis=0)
                collate_data[key] = values
            elif key in ["mol_id", "prec_type", "formula"]:
                collate_data[key] = values
            else:
                raise TypeError(f"Unsupported type: {key} {type(values[0])}")
        # remove everything
        keys.clear()
        # add batch size
        collate_data["batch_size"] = th.tensor(batch_size, dtype=th.long)

    @staticmethod
    def collate_fn(data_list: list[dict]) -> dict:
        """
        Collate a list of data samples into a batched dictionary.

        Args:
            data_list: List of data dictionaries from dataset samples.

        Returns:
            Batched data dictionary ready for model input.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError()
