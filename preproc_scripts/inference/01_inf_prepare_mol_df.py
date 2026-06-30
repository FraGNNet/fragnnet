"""
Prepare mol_df for FragGNN inference from various SMILES input sources.

This script takes SMILES strings from multiple input modes and creates a mol_df file
that contains molecular properties and metadata.

Input modes:
1. Command-line list: --smiles_list "CCO" "CC"
2. Text/SMILES file: --smiles_file smiles.txt
3. CSV/TSV file: --smiles_file data.csv --smiles_colname smiles
4. JSON with candidates: --json_file candidates.json (for MS2C datasets)

Usage:
    python 01_prepare_mol_df.py --smiles_list "CCO" "CC" --output_dir output/
    python 01_prepare_mol_df.py --smiles_file smiles.txt --output_dir output/
    python 01_prepare_mol_df.py --json_file candidates.json --output_dir output/
"""

import argparse
import json
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
from rdkit import RDLogger

import fragnnet.utils.formula_utils as formula_utils
import fragnnet.utils.frag_utils as frag_utils
from fragnnet.frag.compute_frags import MAX_NUM_NODES
from fragnnet.utils import data_utils
from fragnnet.utils.data_utils import compute_mol_df_props, par_apply_series

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_mol_df(
    smiles_list: list[str], ids_list: list[str] | None, max_heavy_atoms: int
) -> pd.DataFrame:
    """
    Create mol_df from a list of SMILES strings.

    Args:
        smiles_list (list): List of SMILES strings
        ids_list (list, optional): List of molecule IDs
        max_heavy_atoms (int): Maximum number of heavy atoms allowed

    Returns:
        pd.DataFrame: mol_df with required columns
    """
    logger.info(f"Creating mol_df for {len(smiles_list)} SMILES...")

    # Create initial DataFrame
    mol_df = pd.DataFrame(
        {
            "smiles": smiles_list,
            "mol_id": np.arange(len(smiles_list)),
            "input_mol_id": np.array([str(i) for i in range(len(smiles_list))])
            if ids_list is None
            else np.where(pd.isna(ids_list), np.nan, [str(x) for x in ids_list]),
        }
    )

    # Convert SMILES to RDKit molecules
    logger.info("Converting SMILES to RDKit molecules...")
    mol_df.loc[:, "mol"] = par_apply_series(mol_df["smiles"], data_utils.mol_from_smiles)

    # Remove invalid molecules
    initial_count = len(mol_df)
    mol_df = mol_df[mol_df["mol"].notna()]
    valid_count = len(mol_df)
    if initial_count != valid_count:
        logger.warning(f"Removed {initial_count - valid_count} invalid SMILES")

    # Generate molecular properties
    logger.info("Computing molecular properties...")
    compute_mol_df_props(mol_df, scaffold=False, inchi=False)
    # Heavy atoms = non-hydrogen atoms (alias for num_atoms with explicit heavy-atom API)
    mol_df.loc[:, "num_heavy_atoms"] = par_apply_series(
        mol_df["mol"], lambda m: int(m.GetNumHeavyAtoms()) if m is not None else np.nan
    )

    # filter by out_set_elements
    allowed_elements = set(frag_utils.ELEMENTS)
    mol_df.loc[:, "num_out_set_elements"] = mol_df["formula"].apply(
        lambda x: len(set(list(formula_utils.parse_formula(x).keys())) - allowed_elements)
    )

    # Filter out molecules based on quality criteria
    logger.info("Filtering molecules based on quality criteria...")
    initial_count_filter = len(mol_df)

    # Drop molecules that are not single molecules, have radicals, or have charges
    mol_df = mol_df[
        (mol_df["single_mol"] == True)
        & (mol_df["num_radicals"] == 0)
        & (mol_df["charge"] == 0)
        & (mol_df["num_heavy_atoms"] <= max_heavy_atoms)
        & (mol_df["num_out_set_elements"] == 0)
    ]

    filtered_count = len(mol_df)
    if initial_count_filter != filtered_count:
        logger.warning(
            f"Filtered out {initial_count_filter - filtered_count} molecules due to quality criteria"
        )
        logger.info("  - Kept only single molecules with no radicals and no charge")

    # Reset mol_id after filtering
    mol_df = mol_df.reset_index(drop=True)
    mol_df.loc[:, "mol_id"] = np.arange(len(mol_df))

    logger.info(f"Created mol_df with {len(mol_df)} molecules")
    return mol_df


def validate_inputs(smiles_list: list[str], max_heavy_atoms: int) -> None:
    """
    Validate input parameters.

    Args:
        smiles_list (list): List of SMILES strings
        max_heavy_atoms (int): Maximum number of heavy atoms
    """
    if not smiles_list:
        raise ValueError("SMILES list cannot be empty")

    if max_heavy_atoms < 1 or max_heavy_atoms > MAX_NUM_NODES:
        raise ValueError(f"max_heavy_atoms must be between 1 and {MAX_NUM_NODES}")


def save_per_query_mol_dfs(candidates_dict: dict, mol_df: pd.DataFrame, output_dp: str) -> None:
    """
    Save separate mol_df for each query SMILES (for MS2C datasets).

    Args:
        candidates_dict: Dictionary mapping query SMILES to candidate SMILES lists
        mol_df: Full mol_df DataFrame with all molecules
        output_dp: Output directory path
    """
    per_query_dp = os.path.join(output_dp, "per_query_mol_dfs")
    os.makedirs(per_query_dp, exist_ok=True)

    for idx, (query_smiles, candidate_list) in enumerate(candidates_dict.items()):
        # Get all SMILES for this query (query + candidates)
        query_smiles_set = {query_smiles}
        query_smiles_set.update(candidate_list)

        # Filter mol_df to only include these SMILES
        query_mol_df = mol_df[mol_df["smiles"].isin(query_smiles_set)].copy()

        # Reset mol_id to be sequential for this subset
        query_mol_df = query_mol_df.reset_index(drop=True)
        query_mol_df.loc[:, "mol_id"] = np.arange(len(query_mol_df))

        # Create filename based on index
        filename = f"mol_df_query_{idx:05d}.pkl"
        output_fp = os.path.join(per_query_dp, filename)
        query_mol_df.to_pickle(output_fp)

        if (idx + 1) % 100 == 0 or idx == 0 or idx == len(candidates_dict) - 1:
            logger.info(
                f"Saved {idx + 1}/{len(candidates_dict)}: {filename} "
                f"({len(query_mol_df)} molecules)"
            )

    logger.info(f"Saved {len(candidates_dict)} per-query mol_df files to {per_query_dp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare mol_df for FragGNN inference from various input sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            # Single molecule
            python 01_inf_prepare_mol_df.py --smiles_list "CCO" --output_dir output/

            # Multiple molecules
            python 01_inf_prepare_mol_df.py --smiles_list "CCO" "CC(=O)O" "CCN" --output_dir output/

            # From text/SMILES file
            python 01_inf_prepare_mol_df.py --smiles_file smiles.txt --output_dir output/

            # CSV/TSV file with IDs
            python 01_inf_prepare_mol_df.py --smiles_file data.csv --smiles_colname smiles --id_colname mol_id --output_dir output/

            # From JSON (MS2C candidates)
            python 01_inf_prepare_mol_df.py --json_file candidates.json --output_dir output/

            # From JSON with per-query mol_dfs
            python 01_inf_prepare_mol_df.py --json_file candidates.json --output_dir output/ --save_per_query
                """,
    )

    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smiles_list", nargs="+", help="List of SMILES strings")
    group.add_argument(
        "--smiles_file", type=str, help="File containing SMILES strings (one per line, or CSV/TSV)"
    )
    group.add_argument(
        "--json_file",
        type=str,
        help="JSON file with candidate SMILES {query_smiles: [candidate_smiles_list]} (MS2C mode)",
    )
    parser.add_argument("--smiles_colname", type=str, help="Column name containing SMILES strings")
    parser.add_argument("--id_colname", type=str, help="Column name containing mol IDs")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for mol_df.pkl.gz and config.json",
    )
    parser.add_argument(
        "--max_heavy_atoms",
        type=int,
        default=MAX_NUM_NODES,
        help="Maximum number of heavy atoms in a molecule",
    )
    parser.add_argument(
        "--save_per_query",
        action="store_true",
        help="(JSON mode only) Save separate mol_df for each query SMILES in per_query_mol_dfs/ subdirectory",
    )
    args = parser.parse_args()

    # Read SMILES based on input mode
    ids_list = None
    candidates_dict = None
    input_mode = None

    if args.smiles_list:
        smiles_list = args.smiles_list
        input_mode = "smiles_list"
    elif args.json_file:
        logger.info(f"Loading JSON from {args.json_file}")
        with open(args.json_file) as f:
            candidates_dict = json.load(f)

        # Collect all unique SMILES
        all_smiles = set()
        for query_smiles, candidate_list in candidates_dict.items():
            all_smiles.add(query_smiles)
            all_smiles.update(candidate_list)

        smiles_list = sorted(list(all_smiles))
        input_mode = "json_file"
        logger.info(f"Total unique SMILES from JSON: {len(smiles_list)}")
    else:
        logger.info(f"Reading SMILES from {args.smiles_file}")
        if (
            args.smiles_file.endswith(".smi")
            or args.smiles_file.endswith(".smiles")
            or args.smiles_file.endswith(".txt")
        ):
            with open(args.smiles_file) as f:
                smiles_list = [line.strip() for line in f if line.strip()]
            input_mode = "text_file"
        elif (
            args.smiles_file.endswith(".csv")
            or args.smiles_file.endswith(".tsv")
            or args.smiles_file.endswith(".csv.gz")
            or args.smiles_file.endswith(".tsv.gz")
        ):
            sep = (
                ","
                if args.smiles_file.endswith(".csv") or args.smiles_file.endswith(".csv.gz")
                else "\t"
            )
            smiles_df = pd.read_csv(args.smiles_file, sep=sep)
            if not args.smiles_colname or args.smiles_colname not in smiles_df.columns:
                raise ValueError(
                    f"Column name {args.smiles_colname} for SMILES not specified or not found in CSV columns: {smiles_df.columns.tolist()}"
                )
            smiles_list = smiles_df[args.smiles_colname].astype(str).tolist()
            if args.id_colname and args.id_colname in smiles_df.columns:
                ids_list = smiles_df[args.id_colname].astype(str).tolist()
                if len(ids_list) != len(smiles_list):
                    raise ValueError(
                        f"Length of IDs list ({len(ids_list)}) does not match length of SMILES list ({len(smiles_list)})"
                    )
            elif args.id_colname not in smiles_df.columns:
                logger.warning(
                    f"ID column name '{args.id_colname}' not found in CSV columns: {smiles_df.columns.tolist()}"
                )
            input_mode = "csv_file"

    # Validate inputs
    validate_inputs(smiles_list, args.max_heavy_atoms)

    logger.info("Input summary:")
    logger.info(f"  - Input mode: {input_mode}")
    logger.info(f"  - {len(smiles_list)} SMILES")
    logger.info(f"  - Max heavy atoms: {args.max_heavy_atoms}")
    if input_mode == "json_file":
        logger.info(f"  - JSON queries: {len(candidates_dict)}")
        logger.info(f"  - Save per-query: {args.save_per_query}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create mol_df
    mol_df = create_mol_df(smiles_list, ids_list, args.max_heavy_atoms)

    # Save files
    mol_df_path = os.path.join(args.output_dir, "mol_df.pkl.gz")

    logger.info(f"Saving mol_df to {mol_df_path}")
    mol_df.to_pickle(mol_df_path)

    # Save metadata config
    config_metadata = {
        "timestamp": datetime.now().isoformat(),
        "input_parameters": {
            "num_smiles": len(smiles_list),
            "input_mode": input_mode,
            "smiles_file": args.smiles_file
            if hasattr(args, "smiles_file") and args.smiles_file
            else None,
            "json_file": args.json_file if hasattr(args, "json_file") and args.json_file else None,
            "max_heavy_atoms": args.max_heavy_atoms,
        },
        "output_files": {
            "mol_df": mol_df_path,
        },
    }
    if input_mode == "json_file":
        config_metadata["input_parameters"]["num_queries"] = len(candidates_dict)
        config_metadata["input_parameters"]["save_per_query"] = args.save_per_query

    config_path = os.path.join(args.output_dir, "mol_df_config.json")
    with open(config_path, "w") as f:
        json.dump(config_metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Configuration saved to {config_path}")

    # Save per-query mol_dfs if in JSON mode and flag is set
    if input_mode == "json_file" and args.save_per_query:
        logger.info("\n=== Saving per-query mol_df files ===")
        save_per_query_mol_dfs(candidates_dict, mol_df, args.output_dir)

    # Print summary
    logger.info("Summary:")
    logger.info(f"  - mol_df: {len(mol_df)} molecules")
    logger.info(f"  - Output directory: {args.output_dir}")
    logger.info(f"  - Configuration saved: {config_path}")

    # Print first few entries for verification
    logger.info("\nFirst 5 molecules in mol_df:")
    print(mol_df.head(5))

    # Heavy atom statistics
    if "num_heavy_atoms" in mol_df.columns and len(mol_df) > 0:
        stats = mol_df["num_heavy_atoms"].describe()
        logger.info("\nHeavy atom count stats (num_heavy_atoms):")
        logger.info(f"  count: {int(stats['count'])}")
        logger.info(f"  mean:  {stats['mean']:.2f}")
        logger.info(f"  std:   {stats['std']:.2f}")
        logger.info(f"  min:   {int(stats['min'])}")
        logger.info(f"  25%:   {stats['25%']:.2f}")
        logger.info(f"  50%:   {stats['50%']:.2f}")
        logger.info(f"  75%:   {stats['75%']:.2f}")
        logger.info(f"  max:   {int(stats['max'])}")

    # num_bonds statistics
    if "num_bonds" in mol_df.columns and len(mol_df) > 0:
        stats_bonds = mol_df["num_bonds"].describe()
        logger.info("\nBond count stats (num_bonds):")
        logger.info(f"  count: {int(stats_bonds['count'])}")
        logger.info(f"  mean:  {stats_bonds['mean']:.2f}")
        logger.info(f"  std:   {stats_bonds['std']:.2f}")
        logger.info(f"  min:   {int(stats_bonds['min'])}")
        logger.info(f"  25%:   {stats_bonds['25%']:.2f}")
        logger.info(f"  50%:   {stats_bonds['50%']:.2f}")
        logger.info(f"  75%:   {stats_bonds['75%']:.2f}")
        logger.info(f"  max:   {int(stats_bonds['max'])}")
