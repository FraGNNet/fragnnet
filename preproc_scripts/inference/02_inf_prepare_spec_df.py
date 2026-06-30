"""
Prepare spec_df for FragGNN inference from mol_df and experimental conditions.

This script takes a mol_df and configuration of experimental conditions (precursor types,
collision energies, instrument types) and creates a spec_df file with spectrum metadata.

Usage:
    python 02_prepare_spec_df.py --mol_df_path output/mol_df.pkl.gz --prec_types "[M+H]+" --ft_ces 25.0NCE --inst_types "FT" --output_dir output/
    python 02_prepare_spec_df.py --mol_df_path output/mol_df.pkl.gz --prec_types "[M+H]+" --qtof_ces 20.0eV --inst_types "QTOF" --merged_spectra true --output_dir output/
"""

import argparse
import json
import logging
import os
import re
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from fragnnet.utils import data_utils
from fragnnet.utils.misc_utils import booltype

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_spec_df(mol_df: pd.DataFrame, config_d: dict[str, Any]) -> pd.DataFrame:
    """
    Create spec_df from mol_df and configuration dictionary.

    Args:
        mol_df (pd.DataFrame): Molecule dataframe with columns: mol_id, formula, inchikey_s, exact_mw
        config_d (dict): Configuration dictionary containing combinations of experimental conditions

    Returns:
        pd.DataFrame: spec_df with required columns
    """
    combinations = config_d["combinations"]
    num_molecules = len(mol_df)
    num_combinations = len(combinations)
    merged_spectra = config_d["input_parameters"]["merged_spectra"]

    logger.info(
        f"Creating spec_df for {num_molecules} molecules with {num_combinations} experimental combinations..."
    )

    # Create all combinations of molecules and experimental conditions
    spec_entries = []
    spec_id = 0
    group_id = 0

    # Ensure mol_df has mol_id as index or use it if it's a column
    if "mol_id" in mol_df.columns:
        mol_iterator = mol_df.iterrows()
    else:
        mol_iterator = mol_df.reset_index().rename(columns={"index": "mol_id"}).iterrows()

    for _, mol_row in mol_iterator:
        mol_id = mol_row["mol_id"]
        for config_combo in config_d["combinations"]:
            prec_type = config_combo["prec_type"]
            inst_type = config_combo["inst_type"]
            # Choose appropriate collision energy based on instrument type
            if config_combo["ce_unit"] == "NCE":
                nce_value = config_combo["ce"]
                ace_value = np.nan
            else:
                nce_value = np.nan
                ace_value = config_combo["ce"]

            spec_entries.append(
                {
                    "spec_id": spec_id,
                    "mol_id": mol_id,
                    "prec_type": prec_type,
                    "nce": nce_value,
                    "ace": ace_value,
                    "nce_extra_1": np.nan,
                    "ace_extra_1": np.nan,
                    "nce_extra_2": np.nan,
                    "ace_extra_2": np.nan,
                    "inst_type": inst_type,
                    "group_id": group_id,
                    "dset": "inference",
                    "dset_spec_id": spec_id,
                    "prec_mz": np.nan,  # Will be computed from molecular mass and precursor type
                    "peaks": [],  # Empty peaks list for prediction
                    "frag_mode": "CID",  # Default fragmentation mode
                    "spec_type": "MS2",  # Default spectrum type
                    "ion_mode": "P",  # Positive mode default
                    "formula": mol_row["formula"],
                    "inchikey": mol_row["inchikey_s"],
                    "exact_mw": mol_row["exact_mw"],
                }
            )
            spec_id += 1
            group_id += 1

    spec_df = pd.DataFrame(spec_entries)

    if merged_spectra:
        # Reset group_id based on prec_type, inst_type, and mol groups
        logger.info(
            "Adjusting group_ids for merged spectra based on prec_type, inst_type, and molecule groups..."
        )

        # Create a mapping for group assignment
        group_mapping = {}
        current_group_id = 0

        # Group by prec_type and inst_type combinations
        for (prec_type, inst_type), _ in spec_df.groupby(["prec_type", "inst_type"]):
            group_key = (prec_type, inst_type)
            if group_key not in group_mapping:
                group_mapping[group_key] = current_group_id
                current_group_id += 1

        # Assign new group_ids
        spec_df.loc[:, "group_id"] = spec_df.apply(
            lambda row: group_mapping[(row["prec_type"], row["inst_type"])], axis=1
        )

        logger.info(
            f"Created {len(group_mapping)} unique groups based on prec_type and inst_type combinations"
        )

    # Compute precursor m/z from molecular mass and precursor type
    logger.info("Computing precursor m/z values...")
    spec_df.loc[:, "prec_mz"] = spec_df.apply(lambda row: data_utils.infer_prec_mz(row), axis=1)

    # rename exact_mw to exact_mass if needed by downstream, or keep both?
    # original code had exact_mass in the dict but merged exact_mw.
    # Let's keep exact_mw as it's used by infer_prec_mz
    spec_df.rename(columns={"exact_mw": "exact_mass"}, inplace=True)

    logger.info(f"Created spec_df with {len(spec_df)} spectra")
    return spec_df


def validate_inputs(prec_types: list, qtof_ces: list, ft_ces: list, inst_types: list) -> None:
    """
    Validate input parameters.

    Args:
        prec_types (list): List of precursor types
        qtof_ces (list): List of collision energies (eV) for QToF
        ft_ces (list): List of normalized collision energies (%) for Orbitrap
        inst_types (list): List of instrument types
    """
    if not prec_types:
        raise ValueError("Precursor types list cannot be empty")

    if not inst_types:
        raise ValueError("Instrument types list cannot be empty")

    # Log CE unit expectations for each instrument type
    for inst_type in inst_types:
        if inst_type == "FT":
            logger.info(f"Instrument type '{inst_type}' will use NCE (%) values: {ft_ces}")
            if len(ft_ces) == 0:
                raise ValueError(
                    f"Instrument type '{inst_type}' requires NCE values, but none were provided."
                )
        elif inst_type in ["QTOF"]:
            logger.info(f"Instrument type '{inst_type}' will use ACE (eV) values: {qtof_ces}")
            if len(qtof_ces) == 0:
                raise ValueError(
                    f"Instrument type '{inst_type}' requires ACE values, but none were provided."
                )
        else:
            # TODO: update this later
            logger.info(
                f"Instrument type '{inst_type}', please ensure appropriate CE values are provided."
            )


def get_config(
    prec_types: list,
    qtof_ces: list,
    ft_ces: list,
    inst_types: list,
    output_dir: str,
    merged_spectra: bool = False,
) -> dict[str, Any]:
    """
    Build the configuration parameters for spec_df creation.

    Args:
        prec_types (list): List of precursor types
        qtof_ces (list): List of collision energies (eV) for QToF
        ft_ces (list): List of normalized collision energies (%) for Orbitrap
        inst_types (list): List of instrument types
        output_dir (str): Output directory
        merged_spectra (bool): Whether to generate merged spectra

    Returns:
        dict: Configuration dictionary with combinations
    """
    # Create list of all combinations
    combinations_list = []
    combo_id = 0

    # Parse and validate collision energy strings
    ft_ces_values = []
    qt_ces_values = []

    for ce_str in qtof_ces:
        match = re.match(r"(\d+(\.\d+)?)(\s*)?(eV)", ce_str, re.IGNORECASE)
        if match:
            number = match.group(1)
            unit = match.group(4)
            qt_ces_values.append((float(number), unit))
        else:
            raise ValueError(
                f"Invalid QToF collision energy format: {ce_str}. Expected format like '10.0eV'"
            )

    for ce_str in ft_ces:
        match = re.match(r"(\d+(\.\d+)?)(\s*)?(eV|%|%NCE|NCE)", ce_str, re.IGNORECASE)
        if match:
            number = match.group(1)
            unit = match.group(4)
            if unit in ["%", "%NCE", "NCE"]:
                unit = "NCE"
            else:
                unit = "eV"
            ft_ces_values.append((float(number), unit))
        else:
            raise ValueError(
                f"Invalid FT collision energy format: {ce_str}. Expected format like '10.0NCE' or '20.0%' or '30.0eV'"
            )

    for prec_type in prec_types:
        for inst_type in inst_types:
            # Choose appropriate collision energy values based on instrument type
            if inst_type == "FT":
                energy_values = ft_ces_values
            elif inst_type == "QTOF":
                energy_values = qt_ces_values

            for energy_config in energy_values:
                combinations_list.append(
                    {
                        "id": combo_id,
                        "prec_type": prec_type,
                        "ce": energy_config[0],
                        "ce_unit": energy_config[1],
                        "inst_type": inst_type,
                    }
                )
                combo_id += 1

    config = {
        "timestamp": datetime.now().isoformat(),
        "input_parameters": {
            "precursor_types": prec_types,
            "collision_energies_eV": qtof_ces,
            "normalized_collision_energies_pct": ft_ces,
            "instrument_types": inst_types,
            "output_directory": output_dir,
            "merged_spectra": merged_spectra,
        },
        "combinations": combinations_list,
        "output_files": {
            "spec_df": f"{output_dir}/spec_df.pkl.gz",
            "config": f"{output_dir}/spec_df_config.json",
        },
    }

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare spec_df for FraGNNet inference from mol_df",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            # Single condition (FT Orbitrap with NCE)
            python 02_inf_prepare_spec_df.py --mol_df_path output/mol_df.pkl.gz --prec_types "[M+H]+" --ft_ces 25.0NCE --inst_types "FT" --output_dir output/

            # Multiple conditions
            python 02_inf_prepare_spec_df.py --mol_df_path output/mol_df.pkl.gz --prec_types "[M+H]+" "[M-H]-" --ft_ces 15.0NCE 25.0NCE 35.0NCE --inst_types "FT" QTOF --output_dir output/

            # Merged spectra (one entry per molecule + condition type)
            python 02_inf_prepare_spec_df.py --mol_df_path output/mol_df.pkl.gz --prec_types "[M+H]+" --qtof_ces 20.0eV --inst_types "QTOF" --merged_spectra true --output_dir output/

            # QToF with eV collision energies
            python 02_inf_prepare_spec_df.py --mol_df_path output/mol_df.pkl.gz --prec_types "[M+H]+" --qtof_ces 10.0eV 20.0eV 30.0eV --inst_types "QTOF" --output_dir output/
                """,
    )

    # Input options
    parser.add_argument(
        "--mol_df_path",
        type=str,
        required=True,
        help="Path to mol_df.pkl.gz file created by 02_prepare_mol_df.py",
    )
    parser.add_argument(
        "--prec_types",
        nargs="+",
        required=True,
        choices=["[M+H]+", "[M-H]-"],  # to do update this list if needed
        help="List of precursor types (e.g., [M+H]+ [M-H]-)",
    )
    parser.add_argument(
        "--qtof_ces",
        nargs="+",
        type=str,
        required=False,
        default=[],
        help="List of collision energies in eV for QToF (e.g., 10.0eV 20.0eV)",
    )
    parser.add_argument(
        "--ft_ces",
        nargs="+",
        type=str,
        required=False,
        default=[],
        help="List of Orbitrap collision energies in NCE or eV (e.g., 10.0NCE 20.0% 30.0eV)",
    )
    parser.add_argument(
        "--inst_types",
        nargs="+",
        required=True,
        choices=["FT", "QTOF", "IT", "Q", "TOF", "QQQ"],  # to do update this list if needed
        help="List of instrument types (e.g., FT QTOF IT)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for spec_df.pkl.gz and config.json",
    )
    parser.add_argument(
        "--merged_spectra",
        type=booltype,
        default=False,
        help="Generate merged spectra entries (one per molecule + condition type) instead of individual entries. Use true/false, yes/no, 1/0.",
    )
    args = parser.parse_args()

    # Load mol_df
    logger.info(f"Loading mol_df from {args.mol_df_path}")
    mol_df = pd.read_pickle(args.mol_df_path)
    logger.info(f"Loaded mol_df with {len(mol_df)} molecules")

    # Validate inputs
    validate_inputs(args.prec_types, args.qtof_ces, args.ft_ces, args.inst_types)

    logger.info("Input summary:")
    logger.info(f"  - {len(mol_df)} molecules from mol_df")
    logger.info(f"  - {len(args.prec_types)} precursor types: {args.prec_types}")
    logger.info(f"  - {len(args.qtof_ces)} collision energies (eVs) for QToF: {args.qtof_ces}")
    logger.info(f"  - {len(args.ft_ces)} collision energies for Orbitrap (FT): {args.ft_ces}")
    logger.info(f"  - {len(args.inst_types)} instrument types: {args.inst_types}")
    logger.info(f"  - Merged spectra: {args.merged_spectra}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Build configuration
    config_d = get_config(
        prec_types=args.prec_types,
        qtof_ces=args.qtof_ces,
        ft_ces=args.ft_ces,
        inst_types=args.inst_types,
        output_dir=args.output_dir,
        merged_spectra=args.merged_spectra,
    )
    logger.info(f"  - Combinations per molecule: {len(config_d['combinations'])}")

    # Save config file
    config_path = os.path.join(args.output_dir, "spec_df_config.json")
    with open(config_path, "w") as f:
        json.dump(config_d, f, indent=2, ensure_ascii=False)

    logger.info(f"Configuration saved to {config_path}")

    # Create spec_df
    spec_df = create_spec_df(mol_df, config_d)

    # Save files
    spec_df_path = os.path.join(args.output_dir, "spec_df.pkl.gz")

    logger.info(f"Saving spec_df to {spec_df_path}")
    spec_df.to_pickle(spec_df_path)

    # Print summary
    logger.info("Summary:")
    logger.info(f"  - spec_df: {len(spec_df)} spectra")
    logger.info(f"  - Output directory: {args.output_dir}")
    logger.info(f"  - Configuration saved: {config_path}")

    logger.info("\nFirst 5 spectra in spec_df:")
    print(spec_df.head(5))
