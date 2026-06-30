"""Prepare fragment DAGs for Spectraverse candidate molecules.

Reads data/raw/spectraverse/candidates/smiles_SMILES_processed_min1_freq-avg.csv.gz,
takes the first N rows, builds a molecule dataframe, runs fragment generation in
parallel, and writes results to an HDF5 file.

Usage:
    python preproc_scripts/prepare_spectraverse_candidates_frags.py \
        --input_fp data/raw/spectraverse/candidates/smiles_SMILES_processed_min1_freq-avg.csv.gz \
        --output_dp data/processed/spectraverse_candidates \
        --n_rows 1000000 \
        --max_depth 3 \
        --max_h_transfer 4 \
        --max_time 150

Outputs written to --output_dp:
    mol_df.pkl   — processed molecule dataframe
    dags.h5      — fragment DAGs keyed by mol_id
    meta_info.json — {mol_id: [num_nodes, num_edges]} for successfully fragmented molecules
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from functools import partial

import h5py
import pandas as pd
from rdkit import RDLogger

import fragnnet.utils.data_utils as data_utils
from fragnnet.utils.data_utils import compute_mol_df_props, par_apply_series
from fragnnet.utils.frag_utils import run_frag_gen_hdf5
from fragnnet.utils.misc_utils import booltype

logger = logging.getLogger(__name__)


def build_mol_df(smiles_series: pd.Series) -> pd.DataFrame:
    """Build a molecule dataframe from a deduplicated series of SMILES strings.

    Canonicalizes SMILES via RDKit, assigns integer mol_ids sorted alphabetically,
    and computes molecular properties required for fragment generation.

    Args:
        smiles_series: Raw SMILES strings (may contain duplicates).

    Returns:
        DataFrame with one row per unique canonicalized SMILES and columns:
        smiles, mol_id, mol, inchikey_s, scaffold, formula, inchi,
        mw, exact_mw, num_atoms, num_bonds, charge, single_mol, num_radicals.
        Rows with unparseable SMILES are dropped.
    """
    # Canonicalize and deduplicate
    mol_series = par_apply_series(
        smiles_series, lambda s: data_utils.mol_from_smiles(s, ml_standardize=False)
    )
    canon_smiles = par_apply_series(mol_series, data_utils.mol_to_smiles)

    tmp_df = pd.DataFrame({"raw_smiles": smiles_series, "mol": mol_series, "smiles": canon_smiles})
    # Drop rows where parsing failed
    tmp_df = tmp_df.dropna(subset=["mol", "smiles"])
    tmp_df = tmp_df[tmp_df["smiles"] != ""]

    # Deduplicate on canonicalized SMILES
    unique_smiles = sorted(set(tmp_df["smiles"].tolist()))
    logger.info(f"> {len(unique_smiles)} unique canonicalized SMILES")

    mol_df = pd.DataFrame({"smiles": unique_smiles, "mol_id": list(range(len(unique_smiles)))})

    # Compute RDKit mol objects (with ml_standardize for mol_df, no tautomer canonicalization)
    mol_df["mol"] = par_apply_series(
        mol_df["smiles"],
        partial(data_utils.mol_from_smiles, canonicalize_tautomers=False),
    )
    compute_mol_df_props(mol_df)

    # Drop molecules where RDKit failed
    n_before = len(mol_df)
    mol_df = mol_df.dropna(subset=["mol"])
    n_dropped = n_before - len(mol_df)
    if n_dropped:
        logger.warning(f"> Dropped {n_dropped} molecules with invalid mol objects")

    return mol_df.reset_index(drop=True)


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Suppress RDKit warnings
    RDLogger.DisableLog("rdApp.*")

    os.makedirs(args.output_dp, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Read input CSV
    # ------------------------------------------------------------------ #
    logger.info(f"> Reading {args.input_fp} (first {args.n_rows} rows)")
    raw_df = pd.read_csv(args.input_fp, nrows=args.n_rows)
    logger.info(f"> Loaded {len(raw_df)} rows with columns: {raw_df.columns.tolist()}")

    smiles_col = "smiles"
    if smiles_col not in raw_df.columns:
        raise ValueError(
            f"Expected column '{smiles_col}' in input file. Found: {raw_df.columns.tolist()}"
        )

    # ------------------------------------------------------------------ #
    # 2. Build molecule dataframe
    # ------------------------------------------------------------------ #
    logger.info("> Building mol_df")
    mol_df = build_mol_df(raw_df[smiles_col])
    logger.info(f"> mol_df: {len(mol_df)} unique molecules")

    mol_df_fp = os.path.join(args.output_dp, "mol_df.pkl")
    mol_df.to_pickle(mol_df_fp)
    logger.info(f"> Saved mol_df to {mol_df_fp}")

    # ------------------------------------------------------------------ #
    # 3. Run fragment generation and write HDF5
    # ------------------------------------------------------------------ #
    h5_fp = os.path.join(args.output_dp, "dags.h5")
    logger.info(f"> Writing fragment DAGs to {h5_fp}")

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

    logger.info(f"> Done. {len(meta_info)} molecules stored.")

    meta_info_fp = os.path.join(args.output_dp, "meta_info.json")
    with open(meta_info_fp, "w") as f:
        json.dump(meta_info, f)
    logger.info(f"> Saved meta_info to {meta_info_fp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build mol_df and run frag-gen for Spectraverse candidate molecules."
    )
    parser.add_argument(
        "--input_fp",
        type=str,
        default="data/raw/spectraverse/candidates/smiles_SMILES_processed_min1_freq-avg.csv.gz",
        help="Path to input CSV/CSV.GZ file with a 'smiles' column",
    )
    parser.add_argument(
        "--output_dp",
        type=str,
        default="data/processed/spectraverse_candidates",
        help="Output directory for mol_df.pkl, dags.h5, and meta_info.json",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=1_000_000,
        help="Number of rows to read from the input file (default: 1M)",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=3,
        help="Maximum fragmentation depth for DAG generation",
    )
    parser.add_argument(
        "--max_h_transfer",
        type=int,
        default=4,
        help="Maximum hydrogen transfer allowed during fragmentation",
    )
    parser.add_argument(
        "--max_time",
        type=int,
        default=150,
        help="Maximum time (seconds) allowed per molecule fragmentation",
    )
    parser.add_argument(
        "--isotopes",
        type=booltype,
        default=True,
        help="Whether to compute isotopic peak distributions",
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
        "--multi_cut_bfs",
        type=booltype,
        default=True,
        help="Whether to enable ring-aware multi-bond cutting",
    )
    parser.add_argument(
        "--max_cut_size",
        type=int,
        default=2,
        help="Maximum number of bonds cut per BFS step",
    )
    parser.add_argument(
        "--smarts_prepass",
        type=booltype,
        default=True,
        help="Whether to run SMARTS rearrangement prepass",
    )
    parser.add_argument(
        "--min_frag_atoms",
        type=int,
        default=3,
        help="Minimum heavy atoms per fragment",
    )
    parser.add_argument(
        "--disable_tqdm",
        type=booltype,
        default=False,
        help="Disable tqdm progress bar",
    )
    args = parser.parse_args()
    main(args)
