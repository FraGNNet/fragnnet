"""Unit tests for per-split spectrum filters in BaseDataset._setup_dfs.

Tests cover:
- filter_prec_types: drops spectra whose prec_type is not in spec_params["prec_types"]
- filter_ce_by_inst / nce_min / nce_max: NCE range filter for FT spectra
- filter_ce_by_inst / ace_min / ace_max: ACE range filter for QTOF spectra
- No filter when split_filters is absent or split key is missing
- Spectra with missing CE (NaN / -1 sentinel) are dropped when a CE range is active
- mol_df is updated to only retain molecules that still have spectra after filtering
"""

import os
import tempfile

import pandas as pd
import pytest

from fragnnet.dataset.base_dataset import BaseDataset

# ============================================================================
# Helpers
# ============================================================================


def _make_spec_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal spec_df compatible with _setup_dfs.

    Each row dict should contain at least: spec_id, mol_id, group_id,
    prec_type, inst_type, nce, ace, prec_mz, peaks.
    Optional extra columns (nce_extra_1, etc.) are filled with NaN.
    """
    df = pd.DataFrame(rows)
    for col in ["nce_extra_1", "nce_extra_2", "ace_extra_1", "ace_extra_2"]:
        if col not in df.columns:
            df[col] = float("nan")
    if "dset" not in df.columns:
        df["dset"] = "test_dset"
    if "dset_spec_id" not in df.columns:
        df["dset_spec_id"] = df["spec_id"]
    if "peaks" not in df.columns:
        df["peaks"] = [[(100.0, 1.0)]] * len(df)
    return df


def _make_mol_df(mol_ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"mol_id": mol_ids})


def _make_split_dir(split_name: str, spec_ids: list[int], mol_ids: list[int]) -> str:
    """Write a <split>_ids.csv to a temp directory and return the directory path."""
    tmp = tempfile.mkdtemp()
    pd.DataFrame({"spec_id": spec_ids, "mol_id": mol_ids}).to_csv(
        os.path.join(tmp, f"{split_name}_ids.csv"), index=False
    )
    return tmp


def _base_spec_params(**overrides) -> dict:
    """Minimal spec_params that disables most optional features."""
    defaults = {
        "merge": False,
        "merge_keep_ces": False,
        "sparse": False,
        "prec_type": False,
        "prec_type_str": False,
        "prec_types": ["[M+H]+", "[M-H]-"],
        "inst_type": False,
        "inst_types": ["FT", "QTOF"],
        "frag_mode": False,
        "frag_modes": [],
        "prec_mz": False,
        "prec_mass_diff": False,
        "nce": True,
        "ace": True,
        "preprocess": False,
        "unique_id": False,
        "counts": False,
        "test_ces": None,
        "split_filters": {},
    }
    defaults.update(overrides)
    return defaults


def _run_setup_dfs(
    spec_df: pd.DataFrame,
    mol_df: pd.DataFrame,
    split_name: str,
    spec_params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run BaseDataset._setup_dfs and return (spec_df, mol_df)."""
    split_dp = _make_split_dir(
        split_name,
        spec_ids=spec_df["spec_id"].tolist(),
        mol_ids=spec_df["mol_id"].tolist(),
    )
    result = BaseDataset._setup_dfs(
        spec_fp_or_df=spec_df,
        mol_fp_or_df=mol_df,
        split_dp=split_dp,
        splits=[split_name],
        subsample_params={},
        spec_params=spec_params,
    )
    out_spec_df, out_mol_df = result[0], result[1]
    return out_spec_df, out_mol_df


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mixed_spec_df():
    """10 spectra: 4 FT [M+H]+, 3 FT [M-H]-, 2 QTOF [M+H]+, 1 QTOF [M+Na]+."""
    rows = [
        # FT [M+H]+  — NCE in 30-60 range
        {
            "spec_id": 1,
            "mol_id": 1,
            "group_id": 1,
            "prec_type": "[M+H]+",
            "inst_type": "FT",
            "nce": 30.0,
            "ace": 15.0,
            "prec_mz": 300.0,
        },
        {
            "spec_id": 2,
            "mol_id": 1,
            "group_id": 1,
            "prec_type": "[M+H]+",
            "inst_type": "FT",
            "nce": 50.0,
            "ace": 25.0,
            "prec_mz": 300.0,
        },
        {
            "spec_id": 3,
            "mol_id": 2,
            "group_id": 2,
            "prec_type": "[M+H]+",
            "inst_type": "FT",
            "nce": 70.0,
            "ace": 35.0,
            "prec_mz": 300.0,
        },  # NCE > 60 → should be dropped
        {
            "spec_id": 4,
            "mol_id": 2,
            "group_id": 2,
            "prec_type": "[M+H]+",
            "inst_type": "FT",
            "nce": float("nan"),
            "ace": float("nan"),
            "prec_mz": 300.0,
        },  # missing → dropped
        # FT [M-H]-
        {
            "spec_id": 5,
            "mol_id": 3,
            "group_id": 3,
            "prec_type": "[M-H]-",
            "inst_type": "FT",
            "nce": 40.0,
            "ace": 20.0,
            "prec_mz": 298.0,
        },
        {
            "spec_id": 6,
            "mol_id": 3,
            "group_id": 3,
            "prec_type": "[M-H]-",
            "inst_type": "FT",
            "nce": 60.0,
            "ace": 30.0,
            "prec_mz": 298.0,
        },
        {
            "spec_id": 7,
            "mol_id": 4,
            "group_id": 4,
            "prec_type": "[M-H]-",
            "inst_type": "FT",
            "nce": 80.0,
            "ace": 40.0,
            "prec_mz": 298.0,
        },  # NCE > 60 → dropped
        # QTOF [M+H]+  — ACE in eV
        {
            "spec_id": 8,
            "mol_id": 5,
            "group_id": 5,
            "prec_type": "[M+H]+",
            "inst_type": "QTOF",
            "nce": float("nan"),
            "ace": 20.0,
            "prec_mz": 300.0,
        },  # ACE ≤ 40 → kept
        {
            "spec_id": 9,
            "mol_id": 5,
            "group_id": 5,
            "prec_type": "[M+H]+",
            "inst_type": "QTOF",
            "nce": float("nan"),
            "ace": 50.0,
            "prec_mz": 300.0,
        },  # ACE > 40 → dropped
        # QTOF [M+Na]+  — adduct not in prec_types
        {
            "spec_id": 10,
            "mol_id": 6,
            "group_id": 6,
            "prec_type": "[M+Na]+",
            "inst_type": "QTOF",
            "nce": float("nan"),
            "ace": 15.0,
            "prec_mz": 323.0,
        },
    ]
    return _make_spec_df(rows)


@pytest.fixture
def mixed_mol_df():
    return _make_mol_df(list(range(1, 7)))


# ============================================================================
# Tests: no filter
# ============================================================================


class TestNoFilter:
    def test_all_spectra_kept_when_no_split_filters(self, mixed_spec_df, mixed_mol_df):
        """Without split_filters, all spectra in the split survive."""
        params = _base_spec_params(split_filters={})
        out_spec, _ = _run_setup_dfs(mixed_spec_df, mixed_mol_df, "train", params)
        assert len(out_spec) == len(mixed_spec_df)

    def test_all_spectra_kept_when_split_not_in_filters(self, mixed_spec_df, mixed_mol_df):
        """A split not listed in split_filters receives no filter."""
        params = _base_spec_params(split_filters={"train": {"filter_prec_types": True}})
        # running for "val" which has no entry → no filter applied
        out_spec, _ = _run_setup_dfs(mixed_spec_df, mixed_mol_df, "val", params)
        assert len(out_spec) == len(mixed_spec_df)


# ============================================================================
# Tests: filter_prec_types
# ============================================================================


class TestFilterPrecTypes:
    def test_drops_unlisted_adducts(self, mixed_spec_df, mixed_mol_df):
        """[M+Na]+ is not in prec_types → its spectrum is dropped."""
        params = _base_spec_params(
            prec_types=["[M+H]+", "[M-H]-"],
            split_filters={"train": {"filter_prec_types": True}},
        )
        out_spec, _ = _run_setup_dfs(mixed_spec_df, mixed_mol_df, "train", params)
        assert "[M+Na]+" not in out_spec["prec_type"].values

    def test_keeps_listed_adducts(self, mixed_spec_df, mixed_mol_df):
        """All [M+H]+ and [M-H]- spectra survive after adduct filter."""
        params = _base_spec_params(
            prec_types=["[M+H]+", "[M-H]-"],
            split_filters={"train": {"filter_prec_types": True}},
        )
        out_spec, _ = _run_setup_dfs(mixed_spec_df, mixed_mol_df, "train", params)
        # spec 10 ([M+Na]+) should be gone; remaining 9 spectra should all pass
        assert set(out_spec["spec_id"]) == {1, 2, 3, 4, 5, 6, 7, 8, 9}

    def test_mol_df_updated_after_adduct_filter(self, mixed_spec_df, mixed_mol_df):
        """mol_id 6 (only linked to [M+Na]+) should be removed from mol_df."""
        params = _base_spec_params(
            prec_types=["[M+H]+", "[M-H]-"],
            split_filters={"train": {"filter_prec_types": True}},
        )
        _, out_mol = _run_setup_dfs(mixed_spec_df, mixed_mol_df, "train", params)
        assert 6 not in out_mol["mol_id"].values

    def test_false_flag_keeps_all_adducts(self, mixed_spec_df, mixed_mol_df):
        """filter_prec_types: false → [M+Na]+ is kept."""
        params = _base_spec_params(
            prec_types=["[M+H]+", "[M-H]-"],
            split_filters={"train": {"filter_prec_types": False}},
        )
        out_spec, _ = _run_setup_dfs(mixed_spec_df, mixed_mol_df, "train", params)
        assert "[M+Na]+" in out_spec["prec_type"].values


# ============================================================================
# Tests: filter_ce_by_inst — NCE for FT
# ============================================================================


class TestFilterCEByInstNCE:
    def _params_ft_nce(self, nce_min=None, nce_max=60.0):
        ft_cfg = {}
        if nce_min is not None:
            ft_cfg["nce_min"] = nce_min
        if nce_max is not None:
            ft_cfg["nce_max"] = nce_max
        return _base_spec_params(split_filters={"train": {"filter_ce_by_inst": {"FT": ft_cfg}}})

    def test_ft_spectra_above_nce_max_dropped(self, mixed_spec_df, mixed_mol_df):
        """FT spectra with NCE > 60 (spec_ids 3, 7) are dropped."""
        params = self._params_ft_nce(nce_max=60.0)
        out_spec, _ = _run_setup_dfs(mixed_spec_df, mixed_mol_df, "train", params)
        ft_ids = out_spec[out_spec["inst_type"] == "FT"]["spec_id"].tolist()
        assert 3 not in ft_ids
        assert 7 not in ft_ids

    def test_ft_spectra_within_nce_range_kept(self, mixed_spec_df, mixed_mol_df):
        """FT spectra with 0 ≤ NCE ≤ 60 (spec_ids 1, 2, 5, 6) are kept."""
        params = self._params_ft_nce(nce_min=0.0, nce_max=60.0)
        out_spec, _ = _run_setup_dfs(mixed_spec_df, mixed_mol_df, "train", params)
        ft_ids = set(out_spec[out_spec["inst_type"] == "FT"]["spec_id"].tolist())
        assert {1, 2, 5, 6}.issubset(ft_ids)

    def test_ft_missing_nce_dropped_when_filter_active(self, mixed_spec_df, mixed_mol_df):
        """Spec 4 (FT, NCE=NaN) is dropped when an NCE range is set."""
        params = self._params_ft_nce(nce_max=60.0)
        out_spec, _ = _run_setup_dfs(mixed_spec_df, mixed_mol_df, "train", params)
        assert 4 not in out_spec["spec_id"].values

    def test_qtof_spectra_unaffected_by_ft_nce_filter(self, mixed_spec_df, mixed_mol_df):
        """QTOF spectra pass through even when FT NCE filter is active."""
        params = self._params_ft_nce(nce_max=60.0)
        out_spec, _ = _run_setup_dfs(mixed_spec_df, mixed_mol_df, "train", params)
        qtof_ids = set(out_spec[out_spec["inst_type"] == "QTOF"]["spec_id"].tolist())
        # Both QTOF spec_ids (8 and 9) should survive the FT-only filter
        assert {8, 9}.issubset(qtof_ids)


# ============================================================================
# Tests: filter_ce_by_inst — ACE for QTOF
# ============================================================================


class TestFilterCEByInstACE:
    def _params_qtof_ace(self, ace_min=None, ace_max=40.0):
        qtof_cfg = {}
        if ace_min is not None:
            qtof_cfg["ace_min"] = ace_min
        if ace_max is not None:
            qtof_cfg["ace_max"] = ace_max
        return _base_spec_params(split_filters={"train": {"filter_ce_by_inst": {"QTOF": qtof_cfg}}})

    def test_qtof_spectra_above_ace_max_dropped(self, mixed_spec_df, mixed_mol_df):
        """Spec 9 (QTOF, ACE=50 eV > 40) is dropped."""
        params = self._params_qtof_ace(ace_max=40.0)
        out_spec, _ = _run_setup_dfs(mixed_spec_df, mixed_mol_df, "train", params)
        assert 9 not in out_spec["spec_id"].values

    def test_qtof_spectra_within_ace_range_kept(self, mixed_spec_df, mixed_mol_df):
        """Spec 8 (QTOF, ACE=20 eV ≤ 40) is kept."""
        params = self._params_qtof_ace(ace_min=0.0, ace_max=40.0)
        out_spec, _ = _run_setup_dfs(mixed_spec_df, mixed_mol_df, "train", params)
        assert 8 in out_spec["spec_id"].values

    def test_ft_spectra_unaffected_by_qtof_ace_filter(self, mixed_spec_df, mixed_mol_df):
        """FT spectra are not touched by a QTOF-only ACE filter."""
        params = self._params_qtof_ace(ace_max=40.0)
        out_spec, _ = _run_setup_dfs(mixed_spec_df, mixed_mol_df, "train", params)
        ft_ids = set(out_spec[out_spec["inst_type"] == "FT"]["spec_id"].tolist())
        assert {1, 2, 3, 4, 5, 6, 7}.issubset(ft_ids)

    def test_qtof_ace_filter_with_nce_only_mode(self, mixed_spec_df, mixed_mol_df):
        """ACE filter works when ace=False (nce-only mode) — real config scenario.

        The config fraggnn_d3_ma_mi_nist20_amolc_all_sf.yml uses nce=True, ace=False
        with QTOF: {ace_min: 0, ace_max: 40}.  The filter must read the raw ace column
        even though ace CE computation is disabled.
        """
        params = _base_spec_params(
            nce=True,
            ace=False,  # real config: nce-only mode
            split_filters={"train": {"filter_ce_by_inst": {"QTOF": {"ace_min": 0.0, "ace_max": 40.0}}}},
        )
        out_spec, _ = _run_setup_dfs(mixed_spec_df, mixed_mol_df, "train", params)
        assert 9 not in out_spec["spec_id"].values, "Spec 9 (QTOF ACE=50 > 40) should be dropped"
        assert 8 in out_spec["spec_id"].values, "Spec 8 (QTOF ACE=20 <= 40) should be kept"


# ============================================================================
# Tests: ramped and stepped CE — all scalars must satisfy the range
# ============================================================================


class TestRampedSteppedCE:
    """Ramped CE [lo–hi] and stepped CE [lo, mid, hi] store their bounds in
    nce_extra_1 / nce_extra_2.  The filter must check ALL present scalars so
    that a ramp whose upper bound exceeds nce_max is correctly dropped.
    """

    def _make_ramped_spec_df(self) -> pd.DataFrame:
        """Four FT spectra with different ramped/stepped NCE profiles."""
        rows = [
            # ramp 10–50: entirely within [0, 60] → kept
            {
                "spec_id": 1,
                "mol_id": 1,
                "group_id": 1,
                "prec_type": "[M+H]+",
                "inst_type": "FT",
                "nce": 10.0,
                "nce_extra_1": 50.0,
                "nce_extra_2": float("nan"),
                "ace": float("nan"),
                "ace_extra_1": float("nan"),
                "ace_extra_2": float("nan"),
                "prec_mz": 300.0,
            },
            # ramp 10–70: upper bound 70 > 60 → dropped
            {
                "spec_id": 2,
                "mol_id": 2,
                "group_id": 2,
                "prec_type": "[M+H]+",
                "inst_type": "FT",
                "nce": 10.0,
                "nce_extra_1": 70.0,
                "nce_extra_2": float("nan"),
                "ace": float("nan"),
                "ace_extra_1": float("nan"),
                "ace_extra_2": float("nan"),
                "prec_mz": 300.0,
            },
            # stepped 20/40/60: all within [0, 60] → kept
            {
                "spec_id": 3,
                "mol_id": 3,
                "group_id": 3,
                "prec_type": "[M+H]+",
                "inst_type": "FT",
                "nce": 20.0,
                "nce_extra_1": 40.0,
                "nce_extra_2": 60.0,
                "ace": float("nan"),
                "ace_extra_1": float("nan"),
                "ace_extra_2": float("nan"),
                "prec_mz": 300.0,
            },
            # stepped 20/40/80: extra_2=80 > 60 → dropped
            {
                "spec_id": 4,
                "mol_id": 4,
                "group_id": 4,
                "prec_type": "[M+H]+",
                "inst_type": "FT",
                "nce": 20.0,
                "nce_extra_1": 40.0,
                "nce_extra_2": 80.0,
                "ace": float("nan"),
                "ace_extra_1": float("nan"),
                "ace_extra_2": float("nan"),
                "prec_mz": 300.0,
            },
        ]
        return _make_spec_df(rows)

    def test_ramped_ce_upper_bound_enforced(self):
        """Ramp [10–70] is dropped when nce_max=60; ramp [10–50] is kept."""
        spec_df = self._make_ramped_spec_df()
        mol_df = _make_mol_df([1, 2, 3, 4])
        params = _base_spec_params(
            split_filters={"train": {"filter_ce_by_inst": {"FT": {"nce_max": 60.0}}}}
        )
        out_spec, _ = _run_setup_dfs(spec_df, mol_df, "train", params)
        assert 1 in out_spec["spec_id"].values  # ramp [10–50]: within range
        assert 2 not in out_spec["spec_id"].values  # ramp [10–70]: upper bound > 60

    def test_stepped_ce_all_steps_enforced(self):
        """Stepped [20/40/60] is kept; stepped [20/40/80] is dropped (extra_2 > 60)."""
        spec_df = self._make_ramped_spec_df()
        mol_df = _make_mol_df([1, 2, 3, 4])
        params = _base_spec_params(
            split_filters={"train": {"filter_ce_by_inst": {"FT": {"nce_max": 60.0}}}}
        )
        out_spec, _ = _run_setup_dfs(spec_df, mol_df, "train", params)
        assert 3 in out_spec["spec_id"].values  # stepped [20/40/60]: all ≤ 60
        assert 4 not in out_spec["spec_id"].values  # stepped [20/40/80]: 80 > 60

    def test_nce_min_enforced_on_primary_scalar(self):
        """nce_min=15 drops ramp [10–50] whose primary (start) is 10 < 15."""
        spec_df = self._make_ramped_spec_df()
        mol_df = _make_mol_df([1, 2, 3, 4])
        params = _base_spec_params(
            split_filters={"train": {"filter_ce_by_inst": {"FT": {"nce_min": 15.0}}}}
        )
        out_spec, _ = _run_setup_dfs(spec_df, mol_df, "train", params)
        assert 1 not in out_spec["spec_id"].values  # primary 10 < 15 → dropped


# ============================================================================
# Tests: combined FT NCE + QTOF ACE filter (realistic use case)
# ============================================================================


class TestCombinedFilter:
    def _params_combined(self):
        return _base_spec_params(
            prec_types=["[M+H]+", "[M-H]-"],
            split_filters={
                "train": {
                    "filter_prec_types": True,
                    "filter_ce_by_inst": {
                        "FT": {"nce_min": 0.0, "nce_max": 60.0},
                        "QTOF": {"ace_min": 0.0, "ace_max": 40.0},
                    },
                },
                "val": {},  # no filter for val
            },
        )

    def test_combined_surviving_spec_ids(self, mixed_spec_df, mixed_mol_df):
        """After all filters only spec_ids {1, 2, 5, 6, 8} should survive.

        Dropped:
          spec 3  — FT [M+H]+, NCE=70 > 60
          spec 4  — FT [M+H]+, NCE=NaN (missing)
          spec 7  — FT [M-H]-, NCE=80 > 60
          spec 9  — QTOF [M+H]+, ACE=50 > 40
          spec 10 — QTOF [M+Na]+ (unlisted adduct)
        """
        params = self._params_combined()
        out_spec, _ = _run_setup_dfs(mixed_spec_df, mixed_mol_df, "train", params)
        assert set(out_spec["spec_id"].tolist()) == {1, 2, 5, 6, 8}

    def test_val_split_unfiltered(self, mixed_spec_df, mixed_mol_df):
        """The val split has an empty filter entry → all spectra pass."""
        params = self._params_combined()
        out_spec, _ = _run_setup_dfs(mixed_spec_df, mixed_mol_df, "val", params)
        assert len(out_spec) == len(mixed_spec_df)

    def test_mol_df_pruned_correctly(self, mixed_spec_df, mixed_mol_df):
        """mol_ids only linked to dropped spectra are removed from mol_df.

        mol_id 4 (spec 7 only) and mol_id 6 (spec 10 only) should be gone.
        mol_id 2 (specs 3+4, both dropped) should also be gone.
        """
        params = self._params_combined()
        _, out_mol = _run_setup_dfs(mixed_spec_df, mixed_mol_df, "train", params)
        mol_ids = set(out_mol["mol_id"].tolist())
        assert 2 not in mol_ids  # both spectra for mol 2 dropped (NCE > 60 and NaN)
        assert 4 not in mol_ids  # spec 7 (only spec for mol 4) dropped
        assert 6 not in mol_ids  # spec 10 ([M+Na]+) dropped
        assert {1, 3, 5}.issubset(mol_ids)  # these mols still have surviving spectra
