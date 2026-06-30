import argparse
import json
import os

import numpy as np
import pandas as pd

from fragnnet.utils import frag_utils
from fragnnet.utils.misc_utils import booltype, np_temp_seed
from fragnnet.utils.proc_utils import filter_spec_mol


def make_splits(args):
    mol_df = pd.read_pickle(os.path.join(args.proc_dp, "mol_df.pkl"))
    spec_df = pd.read_pickle(os.path.join(args.proc_dp, "spec_df.pkl"))

    print("> Spectrum filters")

    # filter spectra
    prec_types = args.prec_types
    frag_modes = args.frag_modes
    ion_modes = args.ion_modes
    inst_types = args.inst_types

    dsets = args.primary_dsets + args.secondary_dsets
    f_spec_df, f_mol_df = filter_spec_mol(
        spec_df,
        mol_df,
        dsets=dsets,
        prec_types=prec_types,
        max_peak_mz=args.max_peak_mz,
        max_prec_mz=args.max_prec_mz,
        min_prec_mz=args.min_prec_mz,
        elements=args.elements,
        frag_modes=frag_modes,
        ion_modes=ion_modes,
        inst_types=inst_types,
        ces=args.ces,
        spec_type=args.spec_type,
        max_heavy_atom=args.max_heavy_atom,
        max_bond=args.max_bond,
        ce_types=args.ce_types,
    )
    print(
        f"> Selected {f_spec_df.shape[0]}/{spec_df.shape[0]} spectra ({(f_spec_df.shape[0] / spec_df.shape[0]) * 100} % total spectrum)"
    )
    spec_df = f_spec_df
    mol_df = f_mol_df
    primary_spec_df = spec_df[spec_df["dset"].isin(args.primary_dsets)]
    secondary_spec_df = spec_df[spec_df["dset"].isin(args.secondary_dsets)]

    if args.dag_filtering:
        assert args.frag_dp is not None, "DAG filtering requires a fragment directory"
        print("> Read DAG stats")
        dag_stats_df = pd.read_pickle(
            os.path.join(args.frag_dp, f"{args.dag_filter_grouping}_stats_df.pkl")
        )
        print("> DAG filters")
        # filter dags
        node_key = "dag_num_nodes"
        edge_key = "dag_num_edges"
        wrecall_key = args.dag_wrecall_key
        dag_masks = []
        if args.max_num_dag_nodes != -1:
            print(f"> filtering on max dag nodes {args.max_num_dag_nodes}")
            dag_masks.append(dag_stats_df[node_key] <= args.max_num_dag_nodes)
        if args.max_num_dag_edges != -1:
            print(f"> filtering: max dag edges {args.max_num_dag_edges}")
            dag_masks.append(dag_stats_df[edge_key] <= args.max_num_dag_edges)
        print(f"> filtering: min dag wrecall {args.min_dag_wrecall}")
        dag_masks.append(dag_stats_df[wrecall_key] >= args.min_dag_wrecall)
        dag_masks = np.stack(dag_masks, axis=1)
        dag_mask = np.all(dag_masks, axis=1)
        print(
            f"> Selected {np.sum(dag_mask)}/{dag_mask.shape[0]} Dags ({np.mean(dag_mask) * 100} % total dags)"
        )
        dag_stats_df = dag_stats_df[dag_mask]
        # dag_mol_id = dag_stats_df["mol_id"]
        if args.dag_filter_grouping in ["m_spec", "m_mol"]:
            dag_spec_id = None
            dag_group_id = dag_stats_df["group_id"]
        else:
            dag_spec_id = dag_stats_df["spec_id"]
            dag_group_id = None

        print("> Intersection")
        # get intersection
        if args.dag_filter_grouping in ["m_spec", "m_mol"]:
            # Filters the spectrum dataframe to keep only spectra whose group_id has a corresponding DAG computation
            primary_both_group_id = np.intersect1d(primary_spec_df["group_id"], dag_group_id)
            primary_split_df = primary_spec_df[
                primary_spec_df["group_id"].isin(primary_both_group_id)
            ]
            secondary_both_group_id = np.intersect1d(secondary_spec_df["group_id"], dag_group_id)
            secondary_split_df = secondary_spec_df[
                secondary_spec_df["group_id"].isin(secondary_both_group_id)
            ]
        else:
            # Filters the spectrum dataframe to keep only spectra whose spec_id has a corresponding DAG computation
            primary_both_spec_id = np.intersect1d(primary_spec_df["spec_id"], dag_spec_id)
            primary_split_df = primary_spec_df[
                primary_spec_df["spec_id"].isin(primary_both_spec_id)
            ]
            secondary_both_spec_id = np.intersect1d(secondary_spec_df["spec_id"], dag_spec_id)
            secondary_split_df = secondary_spec_df[
                secondary_spec_df["spec_id"].isin(secondary_both_spec_id)
            ]
        primary_split_df = primary_split_df.merge(
            mol_df[["mol_id", args.split_key]], on="mol_id", how="inner"
        )
        secondary_split_df = secondary_split_df.merge(
            mol_df[["mol_id", args.split_key]], on="mol_id", how="inner"
        )
    else:
        primary_split_df = primary_spec_df.merge(
            mol_df[["mol_id", args.split_key]], on="mol_id", how="inner"
        )
        secondary_split_df = secondary_spec_df.merge(
            mol_df[["mol_id", args.split_key]], on="mol_id", how="inner"
        )

    print("> create split(s)")
    split_data_list = []
    if args.split_type in ["random", "random_folds"]:
        # split based on molecule
        # primary split
        primary_split_mol_id = np.unique(primary_split_df["mol_id"])
        split_keys = primary_split_df[primary_split_df["mol_id"].isin(primary_split_mol_id)][
            args.split_key
        ]
        split_keys = np.unique(split_keys)
        if args.total_frac == 1.0:
            total_num = split_keys.shape[0]
        else:
            total_num = int(np.ceil(split_keys.shape[0] * args.total_frac))

        if args.split_type == "random":
            test_num = int(np.ceil(total_num * args.test_frac))
            val_num = int(np.ceil(total_num * args.val_frac))
            with np_temp_seed(args.meta_rseed):
                if args.total_frac < 1.0:
                    split_keys = np.random.choice(split_keys, size=total_num, replace=False)
                test_keys = np.random.choice(split_keys, size=test_num, replace=False)
                train_val_keys = np.setdiff1d(split_keys, test_keys)
                val_keys = np.random.choice(train_val_keys, size=val_num, replace=False)
                train_keys = np.setdiff1d(train_val_keys, val_keys)

            train_df = primary_split_df[primary_split_df[args.split_key].isin(train_keys)][
                ["spec_id", "mol_id", "group_id"]
            ]
            val_df = primary_split_df[primary_split_df[args.split_key].isin(val_keys)][
                ["spec_id", "mol_id", "group_id"]
            ]
            test_df = primary_split_df[primary_split_df[args.split_key].isin(test_keys)][
                ["spec_id", "mol_id", "group_id"]
            ]

            # secondary split: optionally exclude any keys used in train/val/test to avoid leakage
            if args.block_secondary_overlap:
                blocked_keys = np.concatenate([train_val_keys, test_keys])
                secondary_df = secondary_split_df[
                    ~secondary_split_df[args.split_key].isin(blocked_keys)
                ][["spec_id", "mol_id", "group_id"]]
            else:
                secondary_df = secondary_split_df[
                    ~secondary_split_df[args.split_key].isin(train_val_keys)
                ][["spec_id", "mol_id", "group_id"]]
            split_data = {
                "split_dp": args.split_dp,
                "train_df": train_df,
                "val_df": val_df,
                "test_df": test_df,
                "secondary_df": secondary_df,
            }
            split_data_list.append(split_data)
        elif args.split_type == "random_folds":
            with np_temp_seed(args.meta_rseed):
                if args.total_frac < 1.0:
                    split_keys = np.random.choice(split_keys, size=total_num, replace=False)

                num_cv = int(1 / args.test_frac)
                test_keys_cvs = np.array_split(split_keys, num_cv)
                print(f"> creating split for {num_cv} cv folds for {len(split_keys)} grouped cases")
                for i in range(num_cv):
                    test_keys = test_keys_cvs[i]
                    print(f"> creating split for {i} th cv, {len(test_keys)} grouped test cases")
                    # test and val
                    train_val_keys = np.setdiff1d(split_keys, test_keys)
                    val_num = int(np.ceil(total_num * args.val_frac))
                    val_keys = np.random.choice(train_val_keys, size=val_num, replace=False)
                    train_keys = np.setdiff1d(train_val_keys, val_keys)

                    train_df = primary_split_df[primary_split_df[args.split_key].isin(train_keys)][
                        ["spec_id", "mol_id", "group_id"]
                    ]
                    val_df = primary_split_df[primary_split_df[args.split_key].isin(val_keys)][
                        ["spec_id", "mol_id", "group_id"]
                    ]
                    test_df = primary_split_df[primary_split_df[args.split_key].isin(test_keys)][
                        ["spec_id", "mol_id", "group_id"]
                    ]
                    # secondary split: optionally exclude any keys used in train/val/test to avoid leakage
                    if args.block_secondary_overlap:
                        blocked_keys = np.concatenate([train_val_keys, test_keys])
                        secondary_df = secondary_split_df[
                            ~secondary_split_df[args.split_key].isin(blocked_keys)
                        ][["spec_id", "mol_id", "group_id"]]
                    else:
                        secondary_df = secondary_split_df[
                            ~secondary_split_df[args.split_key].isin(train_val_keys)
                        ][["spec_id", "mol_id", "group_id"]]
                    split_data = {
                        "split_dp": os.path.join(args.split_dp, f"cv_{i}"),
                        "train_df": train_df,
                        "val_df": val_df,
                        "test_df": test_df,
                        "secondary_df": secondary_df,
                    }
                    split_data_list.append(split_data)

    elif args.split_type == "predefined":
        id_dtype = str if args.predefined_id_type == "dset_spec_id" else int
        if args.predefined_train_id_fp is not None:
            train_keys = (
                pd.read_csv(args.predefined_train_id_fp)[args.predefined_id_type]
                .astype(id_dtype)
                .tolist()
            )
        else:
            train_keys = args.predefined_train_ids if args.predefined_train_ids is not None else []

        if args.predefined_val_id_fp is not None:
            val_keys = (
                pd.read_csv(args.predefined_val_id_fp)[args.predefined_id_type]
                .astype(id_dtype)
                .tolist()
            )
        else:
            val_keys = args.predefined_val_ids if args.predefined_val_ids is not None else []

        if args.predefined_test_id_fp is not None:
            test_keys = (
                pd.read_csv(args.predefined_test_id_fp)[args.predefined_id_type]
                .astype(id_dtype)
                .tolist()
            )
        else:
            test_keys = args.predefined_test_ids if args.predefined_test_ids is not None else []

        if args.predefined_secondary_id_fp is not None:
            secondary_keys = (
                pd.read_csv(args.predefined_secondary_id_fp)[args.predefined_id_type]
                .astype(id_dtype)
                .tolist()
            )
        else:
            secondary_keys = (
                args.predefined_secondary_ids if args.predefined_secondary_ids is not None else []
            )

        if len(train_keys) == 0 and len(val_keys) == 0:
            print(
                f"> No predefined train/val ids, will randomly split train/val ( val = {args.val_frac} * <total>) from remaining primary data"
            )
            primary_split_mol_id = np.unique(primary_split_df["mol_id"])
            split_keys = primary_split_df[primary_split_df["mol_id"].isin(primary_split_mol_id)][
                args.predefined_id_type
            ]
            split_keys = np.unique(split_keys)
            total_num = split_keys.shape[0]
            val_num = int(np.ceil(total_num * args.val_frac))
            with np_temp_seed(args.meta_rseed):
                train_val_keys = np.setdiff1d(split_keys, test_keys)
                val_keys = np.random.choice(train_val_keys, size=val_num, replace=False)
                train_keys = np.setdiff1d(train_val_keys, val_keys)
            print(
                f"> created {len(train_keys)} train and {len(val_keys)} val keys from {total_num} total keys"
            )

        train_df = primary_split_df[primary_split_df[args.predefined_id_type].isin(train_keys)][
            ["spec_id", "mol_id", "group_id"]
        ]
        val_df = primary_split_df[primary_split_df[args.predefined_id_type].isin(val_keys)][
            ["spec_id", "mol_id", "group_id"]
        ]
        test_df = primary_split_df[primary_split_df[args.predefined_id_type].isin(test_keys)][
            ["spec_id", "mol_id", "group_id"]
        ]
        secondary_df = secondary_split_df[
            secondary_split_df[args.predefined_id_type].isin(secondary_keys)
        ][["spec_id", "mol_id", "group_id"]]

        split_data = {
            "split_dp": args.split_dp,
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "secondary_df": secondary_df,
        }
        split_data_list.append(split_data)

    else:
        raise ValueError(f"{args.split_type} is not supported")

    # create split directory
    for split_data in split_data_list:
        os.makedirs(split_data["split_dp"], exist_ok=True)
        print(f"> Save split to {split_data['split_dp']}")

        print("-" * 16)
        for split_type in ["train_df", "val_df", "test_df", "secondary_df"]:
            print(">", split_type)
            print("> number of unique mols", split_data[split_type]["mol_id"].nunique())
            print("> number of unique spec", split_data[split_type]["spec_id"].nunique())
            print("> number of unique groups", split_data[split_type]["group_id"].nunique())
            print("-" * 16)

        # save ids
        train_fp = os.path.join(split_data["split_dp"], "train_ids.csv")
        val_fp = os.path.join(split_data["split_dp"], "val_ids.csv")
        test_fp = os.path.join(split_data["split_dp"], "test_ids.csv")
        secondary_fp = os.path.join(split_data["split_dp"], "secondary_ids.csv")
        split_data["train_df"].to_csv(train_fp, index=False)
        split_data["val_df"].to_csv(val_fp, index=False)
        split_data["test_df"].to_csv(test_fp, index=False)
        split_data["secondary_df"].to_csv(secondary_fp, index=False)

        # save split metadata
        meta_d = vars(args)
        meta_fp = os.path.join(split_data["split_dp"], "meta.json")
        with open(meta_fp, "w") as f:
            json.dump(meta_d, f, indent=4)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_type",
        type=str,
        choices=["random", "predefined", "random_folds"],
        default="random",
        help="How to construct splits: random, predefined lists, or cross-validation folds",
    )
    parser.add_argument(
        "--split_key",
        type=str,
        choices=["inchikey_s", "scaffold"],
        default="inchikey_s",
        help="Grouping key used for splitting (prevents leakage across this key)",
    )
    parser.add_argument(
        "--predefined_id_type",
        type=str,
        choices=["spec_id", "mol_id", "group_id", "dset_spec_id"],
        default="group_id",
        help="Column name used when reading predefined id CSVs",
    )
    parser.add_argument(
        "--predefined_train_ids",
        type=str,
        nargs="+",
        required=False,
        help="Inline list of train ids when using predefined split",
    )
    parser.add_argument(
        "--predefined_val_ids",
        type=str,
        nargs="+",
        required=False,
        help="Inline list of validation ids when using predefined split",
    )
    parser.add_argument(
        "--predefined_test_ids",
        type=str,
        nargs="+",
        required=False,
        help="Inline list of test ids when using predefined split",
    )
    parser.add_argument(
        "--predefined_secondary_ids",
        type=str,
        nargs="+",
        required=False,
        help="Inline list of secondary ids when using predefined split",
    )
    parser.add_argument(
        "--predefined_train_id_fp",
        type=str,
        required=False,
        help="CSV path containing train ids (column name = predefined_id_type)",
    )
    parser.add_argument(
        "--predefined_val_id_fp",
        type=str,
        required=False,
        help="CSV path containing validation ids",
    )
    parser.add_argument(
        "--predefined_test_id_fp", type=str, required=False, help="CSV path containing test ids"
    )
    parser.add_argument(
        "--predefined_secondary_id_fp",
        type=str,
        required=False,
        help="CSV path containing secondary ids",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=0,
        help="Unused placeholder; CV folds inferred from test_frac when split_type=random_folds",
    )

    # spec filtering criteria
    parser.add_argument(
        "--primary_dsets",
        type=str,
        nargs="+",
        required=True,
        help="Dataset names considered primary (used for train/val/test)",
    )
    parser.add_argument(
        "--secondary_dsets",
        type=str,
        nargs="+",
        default=[],
        help="Dataset names considered secondary (optional extra data)",
    )
    parser.add_argument(
        "--max_peak_mz", type=float, default=1500.0, help="Maximum fragment peak m/z allowed"
    )
    parser.add_argument(
        "--max_prec_mz", type=float, default=1500.0, help="Maximum precursor m/z allowed"
    )
    parser.add_argument(
        "--min_prec_mz", type=float, default=0.0, help="Minimum precursor m/z allowed"
    )
    parser.add_argument(
        "--spec_type", type=str, default="MS2", help="Spectrum type to keep (e.g., MS2)"
    )
    parser.add_argument(
        "--ces",
        type=str,
        choices=["nce", "ace", "nce_or_ace", None],
        default="nce_or_ace",
        required=False,
        help="Collision energy scale used (nce or ace)",
    )
    parser.add_argument(
        "--ce_types",
        type=str,
        nargs="+",
        default=["ramped", "stepped", "single", "none"],
        choices=["ramped", "stepped", "single", "none"],
        required=False,
        help="allow ramped, stepped, or single collision energy spectra",
    )

    # mol filtering criteria
    parser.add_argument(
        "--max_heavy_atom",
        type=int,
        default=None,
        help="Maximum heavy atom count; use default filter",
    )
    parser.add_argument(
        "--max_bond", type=int, default=None, help="Maximum bond count; None use default filter"
    )

    # dag filtering criteria
    parser.add_argument(
        "--dag_filtering",
        type=booltype,
        default=True,
        help="Whether to filter to spectra with available DAG stats",
    )
    parser.add_argument(
        "--dag_filter_grouping",
        type=str,
        choices=["mol", "spec", "m_mol", "m_spec"],
        default="m_spec",
        help="Grouping used when joining DAG stats back to spectra",
    )
    parser.add_argument(
        "--max_num_dag_nodes",
        type=int,
        default=100000,
        help="Maximum DAG node count; -1 disables the filter",
    )
    parser.add_argument(
        "--max_num_dag_edges",
        type=int,
        default=250000,
        help="Maximum DAG edge count; -1 disables the filter",
    )
    parser.add_argument(
        "--dag_wrecall_key",
        type=str,
        default="wrecall_10ppm_h4",
        help="Column name in DAG stats dataframe used for weighted recall filtering (e.g. wrecall_10ppm_h4, wrecall_0.01_h4)",
    )
    parser.add_argument(
        "--min_dag_wrecall",
        type=float,
        default=0.00,
        help="Minimum weighted recall threshold for DAGs",
    )
    parser.add_argument(
        "--elements",
        type=str,
        nargs="+",
        default=frag_utils.ELEMENTS,
        help="Allowed elements for spectra/molecules",
    )
    parser.add_argument(
        "--prec_types",
        type=str,
        nargs="+",
        required=False,
        help="Allowed precursor types filter (optional)",
    )
    parser.add_argument(
        "--inst_types",
        type=str,
        nargs="+",
        required=False,
        help="Allowed instrument types filter (optional)",
    )
    parser.add_argument(
        "--frag_modes",
        type=str,
        nargs="+",
        required=False,
        help="Allowed fragmentation modes filter (optional)",
    )
    parser.add_argument(
        "--ion_modes",
        type=str,
        nargs="+",
        required=False,
        help="Allowed ionization modes filter (optional)",
    )

    # non-filtering args
    parser.add_argument(
        "--meta_rseed", type=int, default=42, help="Random seed for split reproducibility"
    )
    parser.add_argument(
        "--total_frac",
        type=float,
        default=1.0,
        help="Fraction of available primary keys to use before splitting",
    )
    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.2,
        help="Fraction of total keys allocated to test when random splitting",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.2,
        help="Fraction of total keys allocated to validation when random splitting",
    )
    parser.add_argument(
        "--block_secondary_overlap",
        type=booltype,
        default=True,
        help="If True, drop secondary spectra whose split_key overlaps train/val/test to prevent leakage",
    )
    parser.add_argument(
        "--proc_dp",
        type=str,
        required=True,
        help="Directory containing processed mol_df.pkl and spec_df.pkl",
    )
    parser.add_argument(
        "--frag_dp",
        type=str,
        required=False,
        help="Directory containing DAG stats (required if dag_filtering)",
    )
    parser.add_argument(
        "--split_dp",
        type=str,
        required=True,
        help="Output directory where splits and metadata are written",
    )
    args = parser.parse_args()

    make_splits(args)
