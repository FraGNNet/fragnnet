import argparse
import json
import logging
import os
import pickle
from typing import cast

import lmdb
import numpy as np
import pandas as pd
from tqdm import tqdm

import fragnnet.utils.data_utils as data_utils
import fragnnet.utils.frag_utils as frag_utils
from fragnnet.utils.frag_utils import run_frag_gen_hdf5
from fragnnet.utils.misc_utils import booltype, progress_wrapper
from fragnnet.utils.proc_utils import filter_spec_mol, merge_spec_df


def print_and_log(name: str, series: pd.Series, wandb_flag: bool, stats_d: dict) -> None:
    """Log series statistics to dictionary and optionally to W&B.

    Args:
        name: Statistic name prefix
        series: Pandas Series to compute statistics for
        wandb_flag: Whether to log to Weights & Biases
        stats_d: Dictionary to update with statistics
    """
    stats = series.describe()
    for stat in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
        stats_d[f"{name}/{stat}"] = stats[stat]

    if wandb_flag:
        import wandb

        log_d = {"step": 0}
        for stat in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
            log_d[f"{name}/{stat}"] = stats[stat]
        wandb.log(log_d)


def main(args):
    if args.max_time >= data_utils.JOBLIB_TIMEOUT:
        raise ValueError(
            f"max_time ({args.max_time}s) must be less than JOBLIB_TIMEOUT "
            f"({data_utils.JOBLIB_TIMEOUT}s)"
        )

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.compress_dags and args.compress_format not in {"gz", "bz2"}:
        raise ValueError(
            f"Invalid --compress_format '{args.compress_format}'. Expected one of: 'gz', 'bz2'."
        )

    if args.use_lmdb and args.compress_dags:
        logging.info(
            "--compress_dags is ignored when --use_lmdb=True because DAGs are stored in LMDB."
        )

    if args.use_hdf5 and args.use_lmdb:
        raise ValueError("--use_hdf5 and --use_lmdb are mutually exclusive.")

    elements = args.elements
    dsets = args.dsets
    # init wandb
    wandb_flag = args.wandb_mode != "off"
    if wandb_flag:
        import wandb

        wandb_config = vars(args)
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config=wandb_config,
            dir=args.project_dp,
            group="prepare_dag_feats",
        )

    # read in the molecule data
    mol_df_fp = os.path.join(args.project_dp, args.proc_dp, "mol_df.pkl")
    print(f"read in mol from {mol_df_fp}")
    mol_df = pd.read_pickle(mol_df_fp)

    sped_df_fp = os.path.join(args.project_dp, args.proc_dp, "spec_df.pkl")
    print(f"read in spec from {sped_df_fp}")
    spec_df = pd.read_pickle(sped_df_fp)
    print()

    # perform filter selection
    prec_types = args.prec_types
    frag_modes = args.frag_modes
    ion_modes = args.ion_modes
    inst_types = args.inst_types
    spec_type = args.spec_type

    spec_df, mol_df = filter_spec_mol(
        spec_df,
        mol_df,
        elements=elements,
        dsets=dsets,
        prec_types=prec_types,
        num_entries=args.num_entries,
        frag_modes=frag_modes,
        ion_modes=ion_modes,
        inst_types=inst_types,
        spec_type=spec_type,
    )

    n_dup = mol_df["mol_id"].duplicated().sum()
    if n_dup > 0:
        print(f">> WARNING: mol_df has {n_dup} duplicate mol_ids — deduplicating")
        mol_df = mol_df.drop_duplicates(subset=["mol_id"]).reset_index(drop=True)

    m_spec_df = merge_spec_df(spec_df)

    print(
        f">> {len(spec_df)} spectra, {len(mol_df)} molecules"
    )  # , {len(m_spec_df)} merged spectra")
    print(spec_df["prec_type"].value_counts())
    print(spec_df["inst_type"].value_counts())

    print()

    num_atoms = mol_df["num_atoms"]
    print_and_log("num_atoms", num_atoms, wandb_flag, {})

    num_bonds = mol_df["num_bonds"]
    print_and_log("num_bonds", num_bonds, wandb_flag, {})

    os.makedirs(args.frag_dp, exist_ok=True)
    dag_dp = os.path.join(args.frag_dp, "dags")
    os.makedirs(dag_dp, exist_ok=True)
    global_stats_fp = os.path.join(args.frag_dp, "global_stats.json")

    def compute_spectra_stats(
        peaks,
        formula_peak_mzs,
        formula_peak_probs,
        idx_by_h_delta,
        prec_mz,
        prec_type,
        tolerances: list,
        max_h_transfer: int,
    ):
        """_summary_

        Args:
            tolerances (list): _description_
            h_transfer (int): _description_

        Returns:
            _type_: _description_
        """
        results = []
        cols = []
        for tolerance in tolerances:
            for h_transfer in range(1, max_h_transfer + 1):
                keys = [
                    f"recall_{tolerance}_h{h_transfer}",
                    f"wrecall_{tolerance}_h{h_transfer}",
                    f"prec_{tolerance}_h{h_transfer}",
                    f"ppt_peak_{tolerance}_h{h_transfer}",
                    f"ppt_formula_{tolerance}_h{h_transfer}",
                    f"prec_recall_{tolerance}_h{h_transfer}",
                    f"prec_spec_recall_{tolerance}_h{h_transfer}",
                ]
                if "ppm" in tolerance:
                    result = frag_utils.compute_frag_peak_stats(
                        peaks,
                        formula_peak_mzs,
                        formula_peak_probs,
                        idx_by_h_delta,
                        prec_mz,
                        h_transfer,
                        tolerance=float(tolerance[:-3]),
                        prec_type=prec_type,
                        is_ppm=True,
                    )
                else:
                    result = frag_utils.compute_frag_peak_stats(
                        peaks,
                        formula_peak_mzs,
                        formula_peak_probs,
                        idx_by_h_delta,
                        prec_mz,
                        h_transfer,
                        tolerance=float(tolerance),
                        prec_type=prec_type,
                    )
                cols += keys
                results.append(result)
        result_series = pd.concat(results, axis=0)
        result_df = result_series.to_frame().T
        result_df.columns = cols
        return result_df

    print("> Compute Fragments")
    mol_input_rows = (
        [
            row["mol"],
            row["mol_id"],
            args.max_depth,
            True,  # h_prior
            args.max_h_transfer,
            args.max_time,
            args.isotopes,
            args.nb_isomorphic,
            args.wl_max_iterations,
            args.multi_cut_bfs,
            args.max_cut_size,
            args.smarts_prepass,
            args.min_frag_atoms,
        ]
        for _, row in mol_df.iterrows()
    )

    print("> Running frag gen")
    h5_fp = os.path.join(args.frag_dp, "dags.h5")
    meta_info_fp = os.path.join(args.frag_dp, "meta_info.json")

    if args.use_hdf5 and args.skip_stats:
        # Fast path: delegate entirely to the shared utility (no stats needed).
        import h5py

        with h5py.File(h5_fp, "w") as h5_file:
            meta_info = run_frag_gen_hdf5(
                mol_df=mol_df,
                h5_file=h5_file,
                max_depth=args.max_depth,
                max_h_transfer=args.max_h_transfer,
                max_time=args.max_time,
                isotopes=args.isotopes,
                nb_isomorphic=args.nb_isomorphic,
                wl_max_iterations=args.wl_max_iterations,
                multi_cut_bfs=args.multi_cut_bfs,
                max_cut_size=args.max_cut_size,
                smarts_prepass=args.smarts_prepass,
                min_frag_atoms=args.min_frag_atoms,
                disable_tqdm=args.disable_tqdm,
            )
        with open(meta_info_fp, "w") as f:
            f.write(json.dumps(meta_info))
        return

    # Full path: keep loop so stats can be accumulated alongside DAG writes.
    if args.use_lmdb:
        env = lmdb.open(
            dag_dp,
            map_size=args.map_size,
            meminit=False,
            map_async=True,
        )
        txn = env.begin(write=True)
        if txn is None:
            raise RuntimeError("Failed to open LMDB write transaction")
        h5_file = None
    elif args.use_hdf5:
        import h5py

        h5_file = h5py.File(h5_fp, "w")
        env = None
        txn = None
    else:
        env = None
        txn = None
        h5_file = None
    frag_results_gen = data_utils.par_apply(
        iter(mol_input_rows), frag_utils.timed_get_dags, True, return_as_generator=True
    )
    if frag_results_gen is None:
        raise RuntimeError("Parallel fragment generation did not return any results")
    frag_results_iter = cast("list[tuple[str, dict]]", frag_results_gen)
    frag_results = []
    meta_info = {}
    for fr in progress_wrapper(
        frag_results_iter,
        total=mol_df.shape[0],
        desc="Compute Frags",
        disable_tqdm=args.disable_tqdm,
    ):
        mol_id, dag_d = fr

        if len(dag_d) == 0:
            print("error in ", mol_id)
        else:
            meta_info[mol_id] = [dag_d["dag"].num_nodes, dag_d["dag"].num_edges]
            if args.use_lmdb:
                if txn is None or env is None:
                    raise RuntimeError("LMDB environment was not initialized")
                txn.put(
                    f"{int(mol_id)}".encode(), pickle.dumps(dag_d, protocol=pickle.HIGHEST_PROTOCOL)
                )
                if (int(mol_id) + 1) % 200 == 0:
                    txn.commit()
                    txn = env.begin(write=True)
            elif args.use_hdf5:
                if h5_file is None:
                    raise RuntimeError("HDF5 file was not initialized")
                grp = h5_file.create_group(f"{int(mol_id)}")
                frag_utils.dump_dag_hdf5(grp, dag_d)
            else:
                # Write one artifact per molecule when LMDB storage is disabled.
                fp = frag_utils.get_dag_output_path(
                    dag_dp=dag_dp,
                    mol_id=mol_id,
                    compress_dags=args.compress_dags,
                    compress_format=args.compress_format,
                )
                frag_utils.dump_dag_pickle(fp, dag_d)
        if not args.skip_stats:
            # Only retain the lightweight stats keys — the DGL graph (dag_d["dag"])
            # is already serialised to disk above and must not be held in memory.
            _STATS_KEYS = {
                "max_depth",
                "formula_peak_mzs",
                "formula_peak_probs",
                "idx_to_formula",
                "dag_num_edges",
                "dag_num_nodes",
                "dag_sparsity",
                "dag_num_nodes_nb",
                "formula_redundancy",
                "idx_by_h_delta",
            }
            frag_results.append({k: dag_d[k] for k in _STATS_KEYS if k in dag_d})

    if args.use_lmdb:
        if txn is None or env is None:
            raise RuntimeError("LMDB environment was not initialized")
        txn.commit()
        env.sync()
        env.close()
    elif args.use_hdf5:
        if h5_file is None:
            raise RuntimeError("HDF5 file was not initialized")
        h5_file.close()
    with open(meta_info_fp, "w") as f:
        f.write(json.dumps(meta_info))
    if args.skip_stats:
        return

    print("> Fragments Computed")
    # all the following code are not debuged
    frag_stats_df = pd.DataFrame({"mol_id": mol_df["mol_id"]})

    # Extract fields from frag_results
    keys_to_extract = [
        ("max_depth", "depth"),
        ("formula_peak_mzs", "formula_peak_mzs"),
        ("formula_peak_probs", "formula_peak_probs"),
        ("idx_to_formula", "idx_to_formula"),
        ("dag_num_edges", "dag_num_edges"),
        ("dag_num_nodes", "dag_num_nodes"),
        ("dag_sparsity", "dag_sparsity"),
        ("dag_num_nodes_nb", "dag_num_nodes_nb"),
        ("formula_redundancy", "formula_redundancy"),
        ("idx_by_h_delta", "idx_by_h_delta"),
    ]
    for result_key, col_name in keys_to_extract:
        frag_stats_df[col_name] = [result.pop(result_key, np.nan) for result in frag_results]

    del frag_results

    frag_stats_d = {}
    # count failures, then remove them
    num_failures = frag_stats_df.isna().any(axis=1).sum()
    frag_stats_d["total_num_failures"] = num_failures
    print(f"> total num failures: {num_failures}")
    print()
    if wandb_flag:
        wandb.log({"total_num_failures": num_failures, "step": 0})
    frag_stats_df = frag_stats_df.dropna(axis=0)

    ### global properties

    # compute total number of formulae
    unique_formulae = set()
    for idx_to_formula in frag_stats_df["idx_to_formula"].values:
        unique_formulae.update(list(idx_to_formula.values()))
    frag_stats_d["total_num_formulae"] = len(unique_formulae)
    print(f"> total num formulae: {len(unique_formulae)}")
    print()

    # compute total number of depths
    print("> depth:")
    print(frag_stats_df["depth"].value_counts())
    print()
    depth_d = {f"depth/{k}": v for k, v in frag_stats_df["depth"].value_counts().to_dict().items()}
    frag_stats_d.update(depth_d)
    if wandb_flag:
        depth_d["step"] = 0
        wandb.log(depth_d)

    ### molecule properties
    print_and_log("dag_num_edges", frag_stats_df["dag_num_edges"], wandb_flag, frag_stats_d)
    print_and_log("dag_num_nodes", frag_stats_df["dag_num_nodes"], wandb_flag, frag_stats_d)
    print_and_log("dag_sparsity", frag_stats_df["dag_sparsity"], wandb_flag, frag_stats_d)
    print_and_log("dag_num_nodes_nb", frag_stats_df["dag_num_nodes_nb"], wandb_flag, frag_stats_d)
    print_and_log(
        "formula_redundancy", frag_stats_df["formula_redundancy"], wandb_flag, frag_stats_d
    )

    # count number of formula per molecule
    frag_stats_df["num_formulae"] = frag_stats_df["idx_to_formula"].apply(lambda x: len(x) - 1)
    print_and_log("num_formulae", frag_stats_df["num_formulae"], wandb_flag, frag_stats_d)

    if not (frag_stats_df["num_formulae"] > 0).all():
        raise ValueError(
            f"All molecules must have at least 1 formula. "
            f"Found {(frag_stats_df['num_formulae'] <= 0).sum()} molecules with 0 or fewer formulae."
        )

    # drop idx_to_formula
    frag_stats_df = frag_stats_df.drop(columns=["idx_to_formula"])

    print("> Compute Spectra Stats")

    stats_cols = [
        "num_formulae",
        "depth",
        "formula_redundancy",
        "dag_num_edges",
        "dag_num_nodes",
        "dag_sparsity",
        "dag_num_nodes_nb",
    ]
    id_cols = ["mol_id"]
    data_cols = list(set(frag_stats_df.columns) - set(stats_cols) - set(id_cols))

    for merged in [False, True]:
        if not merged:
            spec_key = "spec_id"
            _spec_df = spec_df
            spec_prefix = "spec/"
            mol_prefix = "mol/"
            spec_stats_fp = os.path.join(args.frag_dp, "spec_stats_df.pkl")
            mol_stats_fp = os.path.join(args.frag_dp, "mol_stats_df.pkl")
        else:
            spec_key = "group_id"
            _spec_df = m_spec_df
            spec_prefix = "m_spec/"
            mol_prefix = "m_mol/"
            spec_stats_fp = os.path.join(args.frag_dp, "m_spec_stats_df.pkl")
            mol_stats_fp = os.path.join(args.frag_dp, "m_mol_stats_df.pkl")

        # compute spectra stats
        peak_spec_df = _spec_df[[spec_key, "mol_id", "peaks", "prec_mz", "prec_type"]].merge(
            frag_stats_df[id_cols + data_cols], on="mol_id", how="inner"
        )
        assert (
            peak_spec_df.shape[0]
            == peak_spec_df.drop_duplicates(subset=[spec_key, "mol_id"]).shape[0]
        )

        spectra_input_rows = []
        tolerances = args.tolerances
        for _, row in progress_wrapper(
            peak_spec_df.iterrows(),
            total=peak_spec_df.shape[0],
            desc="prepare spectra stats inputs",
            disable_tqdm=args.disable_tqdm,
        ):
            spectra_input_rows.append(
                [
                    row["peaks"],
                    row["formula_peak_mzs"],
                    row["formula_peak_probs"],
                    row["idx_by_h_delta"],
                    row["prec_mz"],
                    row["prec_type"],
                    tolerances,
                    args.max_h_transfer,
                ]
            )

        if args.disable_tqdm:
            stats_iter = spectra_input_rows
            logging.info(f"Computing spectra stats for {len(spectra_input_rows)} inputs")
        else:
            stats_iter = tqdm(
                spectra_input_rows,
                desc="Computing spectra stats",
                total=len(spectra_input_rows),
            )
        # run stats
        stats_results = data_utils.par_apply(stats_iter, compute_spectra_stats, True)
        # Loky and joblib should keep ordering
        stats_results_df = pd.concat(stats_results, axis=0, ignore_index=True)
        metric_keys = list(stats_results_df.columns)

        # add peak stats
        peak_spec_df = pd.concat((peak_spec_df[[spec_key, "mol_id"]], stats_results_df), axis=1)
        # add dag stats
        peak_spec_df = peak_spec_df.merge(
            frag_stats_df[id_cols + stats_cols], on="mol_id", how="inner"
        )

        # save spectrum-level stats
        peak_spec_df.to_pickle(spec_stats_fp)

        # update the frag stats d
        for key in metric_keys + stats_cols:
            print_and_log(spec_prefix + key, peak_spec_df[key], wandb_flag, frag_stats_d)

        print("> Compute Molecule Stats")

        # collect across molecules and report summary statistics
        peak_mol_df = (
            peak_spec_df.drop(columns=[spec_key]).groupby("mol_id").agg(np.nanmean).reset_index()
        )
        for key in metric_keys + stats_cols:
            print_and_log(mol_prefix + key, peak_mol_df[key], wandb_flag, frag_stats_d)

        # save molecule-level stats
        peak_mol_df.to_pickle(mol_stats_fp)

    # save the frag stats dict
    with open(global_stats_fp, "w", encoding="utf-8") as f:
        frag_stats_d = {k: str(v) for k, v in frag_stats_d.items()}
        json.dump(frag_stats_d, f, ensure_ascii=False, indent=4)

    if wandb_flag:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_entries", type=int, default=-1, help="Number of molecules to process (-1 for all)"
    )
    parser.add_argument(
        "--max_depth", type=int, default=4, help="Maximum fragmentation depth for DAG generation"
    )
    parser.add_argument(
        "--max_time",
        type=int,
        default=150,
        help="Maximum time (seconds) allowed per molecule fragmentation",
    )
    parser.add_argument(
        "--project_dp", type=str, default=os.getcwd(), help="Project root directory"
    )
    parser.add_argument(
        "--frag_dp",
        type=str,
        required=True,
        help="Output directory for fragment DAGs and statistics",
    )
    parser.add_argument(
        "--proc_dp", type=str, required=True, help="Directory containing spec_df.pkl and mol_df.pkl"
    )
    parser.add_argument(
        "--max_h_transfer",
        type=int,
        default=4,
        help="Maximum hydrogen transfer allowed during fragmentation",
    )
    parser.add_argument(
        "--nb_isomorphic",
        type=booltype,
        default=False,
        help="Whether to compute neighborhood isomorphism features",
    )
    parser.add_argument(
        "--wl_max_iterations",
        type=int,
        default=3,
        help="Max Weisfeiler-Lehman iterations for isomorphism detection",
    )
    parser.add_argument(
        "--dsets",
        type=str,
        nargs="+",
        required=True,
        help="Dataset names to include (e.g., nist20_hr)",
    )
    parser.add_argument(
        "--tolerances",
        type=str,
        nargs="+",
        default=["0.01", "0.005", "0.001", "0.0001", "10ppm", "5ppm"],
        help="Mass tolerance values for peak matching statistics",
    )
    parser.add_argument(
        "--isotopes",
        type=booltype,
        default=True,
        help="Whether to compute isotopic peak distributions",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="disabled",
        choices=["online", "offline", "disabled"],
        help="Weights & Biases logging mode",
    )
    parser.add_argument("--wandb_project", type=str, default="fragnnet", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default="fragnnet", help="W&B entity/team name")
    parser.add_argument("--wandb_dp", type=str, default="wandb", help="W&B logging directory")
    parser.add_argument(
        "--compress_dags",
        type=booltype,
        default=True,
        help=(
            "Whether to compress per-molecule DAG files when --use_lmdb=False. "
            "Ignored when --use_lmdb=True."
        ),
    )
    parser.add_argument(
        "--compress_format",
        type=str,
        default="bz2",
        choices=["gz", "bz2"],
        help="Compression format for DAG files when --compress_dags=True and --use_lmdb=False.",
    )
    parser.add_argument(
        "--wandb_run_name", type=str, help="Custom W&B run name (required when wandb_mode enabled)"
    )
    parser.add_argument(
        "--save_dag", type=booltype, default=True, help="Whether to save DAG structures to disk"
    )
    parser.add_argument(
        "--elements",
        type=str,
        nargs="+",
        default=frag_utils.ELEMENTS,
        help="Allowed chemical elements for molecules",
    )
    parser.add_argument(
        "--prec_types", nargs="+", required=False, help="Precursor types to filter (optional)"
    )
    parser.add_argument(
        "--inst_types", nargs="+", required=False, help="Instrument types to filter (optional)"
    )
    parser.add_argument(
        "--frag_modes", nargs="+", required=False, help="Fragmentation modes to filter (optional)"
    )
    parser.add_argument(
        "--ion_modes", nargs="+", required=False, help="Ion modes to filter (optional, e.g., P, N)"
    )
    parser.add_argument(
        "--spec_type", type=str, default="MS2", help="Spectrum type to keep (e.g., MS2)"
    )
    parser.add_argument(
        "--use_cached_dag",
        type=booltype,
        default=False,
        help="Whether to reuse cached DAG files if available",
    )
    parser.add_argument(
        "--use_hdf5",
        type=booltype,
        default=False,
        help="Save DAGs to <frag_dp>/dags.h5 (HDF5). Mutually exclusive with --use_lmdb.",
    )
    parser.add_argument(
        "--disable_tqdm",
        type=booltype,
        default=False,
        help="Disable tqdm progress bars and use logging instead",
    )
    parser.add_argument(
        "--skip_stats",
        type=booltype,
        default=False,
        help="whether skip stats compute and only compute and save the DAGs ",
    )
    parser.add_argument(
        "--map_size",
        type=int,
        default=10 * 1024**3,
        help="map size for LMDB (bytes). Default 10*1024**3 = 10 GiB",
    )
    parser.add_argument(
        "--use_lmdb",
        type=booltype,
        default=False,
        help=(
            "Whether to use LMDB to store DAGs. If False, each DAG will be saved as an "
            "individual pickle file under <frag_dp>/dags/. Default True."
        ),
    )
    parser.add_argument(
        "--multi_cut_bfs",
        type=booltype,
        default=False,
        help=(
            "Use ring-aware multi-bond-cut BFS (multi_cut_bfs.compute_ccs_multi_cut) "
            "instead of the standard single-bond BFS. Enables ring-opening fragmentation "
            "patterns (e.g. retro-Diels-Alder). Note: substantially slower on large "
            "fused-ring molecules; consider reducing --max_depth to 3 and --max_time to 30."
        ),
    )
    parser.add_argument(
        "--max_cut_size",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help=(
            "Maximum number of bonds cut simultaneously per BFS step "
            "(only used when --multi_cut_bfs True). "
            "1 = single-bond only (same as standard BFS on rings), "
            "2 = up to 2 ring bonds (recommended), "
            "3 = up to 3 ring bonds (depth-0 only, high cost)."
        ),
    )
    parser.add_argument(
        "--smarts_prepass",
        type=booltype,
        default=False,
        help=(
            "Run SMARTS rearrangement prepass and inject results as depth-1 DAG children. "
            "Works with both the default BFS and --multi_cut_bfs. "
            "With the default BFS, SMARTS fragments are post-merged into the existing DAG "
            "so ring-opening fragmentation is preserved. "
            "Covers acyclic 1,3-elimination and ester CO2-loss fragments "
            "(crf12_0/1, crf13_2 PPGB_MS2 rules)."
        ),
    )
    parser.add_argument(
        "--min_frag_atoms",
        type=int,
        default=0,
        help=(
            "Minimum number of heavy atoms a child fragment must have to be registered "
            "in the DAG. Fragments smaller than this threshold are silently dropped. "
            "0 disables the filter."
        ),
    )
    args = parser.parse_args()

    # Check if wandb_mode is enabled but run_name is missing
    if args.wandb_mode != "disabled" and not args.wandb_run_name:
        parser.error("--wandb_run_name is required when --wandb_mode is not 'disabled'")

    if not args.disable_tqdm:
        tqdm.pandas()

    main(args)
