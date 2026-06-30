import pandas as pd

from fragnnet.dataset.base_dataset import BaseDataset


def make_spec_df(n=5):
    spec_ids = [f"s{i}" for i in range(n)]
    mol_ids = [f"m{int(i / 2)}" for i in range(n)]
    group_ids = [f"g{int(i / 3)}" for i in range(n)]
    peaks = [[(100.0 + i, 1.0)] for i in range(n)]
    df = pd.DataFrame(
        {
            "spec_id": spec_ids,
            "mol_id": mol_ids,
            "group_id": group_ids,
            "peaks": peaks,
            "dset": "test",
            "dset_spec_id": list(range(n)),
        }
    )
    return df


def make_mol_df(spec_df):
    mol_ids = spec_df["mol_id"].unique().tolist()
    df = pd.DataFrame({"mol_id": mol_ids, "smiles": ["C"] * len(mol_ids)})
    return df


def test_split_level_subsample(tmp_path):
    spec_df = make_spec_df(n=5)
    mol_df = make_mol_df(spec_df)

    split_dp = tmp_path / "splits"
    split_dp.mkdir()
    # write full split file
    split_df = spec_df[["spec_id", "mol_id", "group_id"]].copy()
    split_file = split_dp / "train_ids.csv"
    split_df.to_csv(split_file, index=False)

    subsample_params = {"train": True, "subsample_size": 2, "subsample_seed": 0}
    spec_params = {
        "ace": False,
        "nce": False,
        "merge": False,
        "merge_keep_ces": False,
        "test_ces": None,
        "preprocess": False,
        "sparse": False,
        "prec_type": False,
        "prec_type_str": False,
        "inst_type": False,
        "prec_mass_diff": False,
        "prec_mz": False,
        "unique_id": False,
        "counts": False,
        "prec_types": [],
        "inst_types": [],
    }

    out_spec_df, out_mol_df, um_spec_df, out_split_df, id_key, ce_key = BaseDataset._setup_dfs(
        spec_fp_or_df=spec_df,
        mol_fp_or_df=mol_df,
        split_dp=str(split_dp),
        splits=["train"],
        subsample_params=subsample_params,
        spec_params=spec_params,
    )

    # Expect subsampled spec_df length == 2
    assert len(out_spec_df) == 2
    # split_df should also be size 2
    assert len(out_split_df) == 2
    # mol_df should only contain mols present in sampled specs
    sampled_mols = set(out_spec_df["mol_id"].unique().tolist())
    assert set(out_mol_df["mol_id"].unique().tolist()) == sampled_mols


def test_fractional_subsample(tmp_path):
    # use n=4 so frac=0.5 deterministically yields 2
    spec_df = make_spec_df(n=4)
    mol_df = make_mol_df(spec_df)

    split_dp = tmp_path / "splits2"
    split_dp.mkdir()
    split_df = spec_df[["spec_id", "mol_id", "group_id"]].copy()
    (split_dp / "train_ids.csv").write_text(split_df.to_csv(index=False))

    subsample_params = {"train": True, "subsample_size": 0.5, "subsample_seed": 1}
    spec_params = {
        "ace": False,
        "nce": False,
        "merge": False,
        "merge_keep_ces": False,
        "test_ces": None,
        "preprocess": False,
        "sparse": False,
        "prec_type": False,
        "prec_type_str": False,
        "inst_type": False,
        "prec_mass_diff": False,
        "prec_mz": False,
        "unique_id": False,
        "counts": False,
        "prec_types": [],
        "inst_types": [],
    }

    out_spec_df, out_mol_df, um_spec_df, out_split_df, id_key, ce_key = BaseDataset._setup_dfs(
        spec_fp_or_df=spec_df,
        mol_fp_or_df=mol_df,
        split_dp=str(split_dp),
        splits=["train"],
        subsample_params=subsample_params,
        spec_params=spec_params,
    )

    assert len(out_spec_df) == 2
    assert len(out_split_df) == 2


def test_no_subsample(tmp_path):
    spec_df = make_spec_df(n=6)
    mol_df = make_mol_df(spec_df)

    split_dp = tmp_path / "splits3"
    split_dp.mkdir()
    split_df = spec_df[["spec_id", "mol_id", "group_id"]].copy()
    (split_dp / "train_ids.csv").write_text(split_df.to_csv(index=False))

    subsample_params = {"train": False, "subsample_size": 0}
    spec_params = {
        "ace": False,
        "nce": False,
        "merge": False,
        "merge_keep_ces": False,
        "test_ces": None,
        "preprocess": False,
        "sparse": False,
        "prec_type": False,
        "prec_type_str": False,
        "inst_type": False,
        "prec_mass_diff": False,
        "prec_mz": False,
        "unique_id": False,
        "counts": False,
        "prec_types": [],
        "inst_types": [],
    }

    out_spec_df, out_mol_df, um_spec_df, out_split_df, id_key, ce_key = BaseDataset._setup_dfs(
        spec_fp_or_df=spec_df,
        mol_fp_or_df=mol_df,
        split_dp=str(split_dp),
        splits=["train"],
        subsample_params=subsample_params,
        spec_params=spec_params,
    )

    assert len(out_spec_df) == len(spec_df)
    assert len(out_split_df) == len(split_df)


def test_predict_only_subsample():
    # predict_only uses spec_df directly to create split_df
    spec_df = make_spec_df(n=6)
    mol_df = make_mol_df(spec_df)

    subsample_params = {"predict_only": True, "subsample_size": 3, "subsample_seed": 2}
    spec_params = {
        "ace": False,
        "nce": False,
        "merge": False,
        "merge_keep_ces": False,
        "test_ces": None,
        "preprocess": False,
        "sparse": False,
        "prec_type": False,
        "prec_type_str": False,
        "inst_type": False,
        "prec_mass_diff": False,
        "prec_mz": False,
        "unique_id": False,
        "counts": False,
        "prec_types": [],
        "inst_types": [],
    }

    out_spec_df, out_mol_df, um_spec_df, out_split_df, id_key, ce_key = BaseDataset._setup_dfs(
        spec_fp_or_df=spec_df,
        mol_fp_or_df=mol_df,
        split_dp=".",
        splits=["predict_only"],
        subsample_params=subsample_params,
        spec_params=spec_params,
    )

    assert len(out_spec_df) == 3
    assert len(out_split_df) == 3
