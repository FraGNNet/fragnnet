import argparse
import csv
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

import fragnnet.utils.data_utils as data_utils
from fragnnet.utils.data_utils import (
    compute_mol_df_props,
    par_apply_series,
)

REQUIRED_CANDIDATE_COLS = [
    "spec_id",
    "mol_id",
    "peaks",
    "prec_type",
    "inst_type",
    "ace",
    "nce",
    "ace_extra_1",
    "nce_extra_1",
    "ace_extra_2",
    "nce_extra_2",
    "candidate_smiles_list",
]


def build_query_spec_df(row: pd.Series, size: int) -> pd.DataFrame:
    """Build a per-query spec_df from a single candidate_df row."""
    records: list[dict[str, Any]] = []

    for i in range(size):
        records.append(
            {
                "spec_id": i,
                "mol_id": i,
                "peaks": row["peaks"],
                "prec_type": row["prec_type"],
                "inst_type": row["inst_type"],
                "ace": row["ace"],
                "nce": row["nce"],
                "ace_extra_1": row["ace_extra_1"],
                "nce_extra_1": row["nce_extra_1"],
                "ace_extra_2": row["ace_extra_2"],
                "nce_extra_2": row["nce_extra_2"],
            }
        )
    spec_df = pd.DataFrame(records)
    return spec_df


def build_query_mol_df(candidate_smiles_list: list[str]) -> pd.DataFrame:
    """Build a per-query mol_df from candidate_smiles_list."""
    # de-duplicate smiles
    candidate_smiles_list = list(set(candidate_smiles_list))
    mol_df = pd.DataFrame(
        zip(sorted(candidate_smiles_list), list(range(len(candidate_smiles_list)))),
        columns=["smiles", "mol_id"],
    )
    mol_df.loc[:, "mol"] = par_apply_series(
        mol_df["smiles"], data_utils.mol_from_smiles, use_tqdm=False
    )
    compute_mol_df_props(mol_df, use_tqdm=False)
    if (mol_df["smiles"] == "").any():
        raise ValueError("Empty SMILES found in mol_df")
    if (mol_df["formula"] == "").any():
        raise ValueError("Empty formula found in mol_df")
    if (mol_df["exact_mw"] == 0).any():
        raise ValueError("Zero exact_mw found in mol_df")
    return mol_df


def create_query_mol_spec_df(
    candidate_df: pd.DataFrame, output_dp: Path, limit: int | None = None
) -> None:
    """Create per-query spec_df and mol_df files from candidate_df.

    For each row in candidate_df, creates:
    - A spec_df with candidate molecule IDs
    - A mol_df with candidate SMILES and properties (cached if same mol_id)
    - Records in a query_index.json for reference

    Args:
        candidate_df: DataFrame with columns including candidate_smiles_list
        output_dir: Directory to write query files and index
        limit: Optional limit on the number of candidates to process
    """
    output_dp.mkdir(parents=True, exist_ok=True)

    for col in REQUIRED_CANDIDATE_COLS:
        if col not in candidate_df.columns:
            raise ValueError(f"Missing required column in candidate_df: {col}")

    processed_mol_d = {}
    index_records = []

    # Prepare CSV index file: remove existing if present and write header
    index_path = output_dp / "query_index.csv"
    if index_path.exists():
        index_path.unlink()

    index_fieldnames = ["query_spec_id", "query_candidate_id"]
    index_f = index_path.open("w", newline="")
    index_writer = csv.DictWriter(index_f, fieldnames=index_fieldnames)
    index_writer.writeheader()

    if limit is not None:
        candidate_df = candidate_df.head(limit)
    for i, row in candidate_df.iterrows():
        spec_query_id = row["spec_id"]
        mol_query_id = row["mol_id"]
        # if pd.isna(spec_query_id):
        #    spec_query_id = f"no_spec_{i}"

        if row["mol_id"] in processed_mol_d:
            candidates_size, mol_df_path = processed_mol_d[row["mol_id"]]
        else:
            mol_df = build_query_mol_df(row["candidate_smiles_list"])
            mol_df_path = output_dp / f"mol_df_{mol_query_id}.pkl.gz"
            mol_df.to_pickle(mol_df_path)
            candidates_size = len(mol_df)
            processed_mol_d[row["mol_id"]] = (candidates_size, mol_df_path)

        spec_df = build_query_spec_df(row, size=candidates_size)
        query_spec_path = output_dp / f"spec_df_{spec_query_id}.pkl.gz"
        spec_df.to_pickle(query_spec_path)

        record = {
            "query_spec_id": spec_query_id,
            "query_candidate_id": mol_query_id,
        }
        index_records.append(record)

        # Write row to CSV index incrementally
        index_writer.writerow(record)

        if (i + 1) % 10 == 0:
            logging.getLogger(__name__).info(f"Processed {i + 1}/{len(candidate_df)} candidates")

    # Close CSV file
    index_f.close()

    # Also keep a fully-populated index in memory (same content) available if needed
    logger = logging.getLogger(__name__)
    logger.info(f"Processed all {len(candidate_df)} candidates successfully")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Convert candidate_df to per-query spec_df and shared mol_df subsets.",
    )
    parser.add_argument(
        "--candidate_df_path",
        type=str,
        default="data/ms2c/candidates/nps_nist23_candidate_df.pkl.gz",
        required=False,
        help="Path to candidate_df.pkl.gz",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="data/ms2c/proc/nps_nist23",
        help="Output directory for query files",
    )

    parser.add_argument(
        "--limit",
        type=int,
        required=False,
        default=100,
        help="Limit the number of candidates to process",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    candidate_df = pd.read_pickle(args.candidate_df_path)

    logger.info(f"Loaded candidate_df with {len(candidate_df)} rows")
    logger.info("Creating per-query spec_df and mol_df files...")
    if args.limit is not None:
        logger.info(f"Limiting to first {args.limit} candidates")
    create_query_mol_spec_df(candidate_df, Path(args.output_dir), limit=args.limit)
    logger.info(f"Saved query outputs to {args.output_dir}")
