"""Split spectraverse benchmark CV folds by adduct (precursor) type.

For each CV fold in the source split directory, reads the train/val/test CSVs,
joins with spec_df to get prec_type, then writes per-adduct split directories.

Output structure:
    <output_root>/<adduct_tag>/cv{N}/train_ids.csv
    <output_root>/<adduct_tag>/cv{N}/val_ids.csv
    <output_root>/<adduct_tag>/cv{N}/test_ids.csv

where adduct_tag is a sanitized version of the adduct string (e.g., m+h, m-h).

Usage:
    python preproc_scripts/spectraverse/split_by_adduct.py \
        --split_root data/split/spectraverse_benchmark \
        --spec_fp data/proc/spectraverse/spec_df.pkl \
        --output_root data/split/spectraverse_neims_adduct \
        --adducts "[M+H]+" "[M-H]-" "[M+Na]+"
"""

import argparse
import logging
import os
import re

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def sanitize_adduct(adduct: str) -> str:
    """Convert adduct string to a filesystem-safe tag.

    Args:
        adduct: Adduct string, e.g. '[M+H]+'.

    Returns:
        Lowercase tag with special chars replaced, e.g. 'm+h'.
    """
    tag = adduct.lower()
    tag = re.sub(r"[\[\]]", "", tag)   # remove brackets
    tag = re.sub(r"[^a-z0-9+\-]", "_", tag)  # replace other specials
    tag = tag.strip("_")
    return tag


def load_split_df(split_dir: str, split: str, spec_df: pd.DataFrame) -> pd.DataFrame:
    """Load a split CSV and join with spec_df to attach prec_type.

    Args:
        split_dir: Directory containing train/val/test_ids.csv.
        split: One of 'train', 'val', 'test'.
        spec_df: Spectrum dataframe with spec_id and prec_type columns.

    Returns:
        DataFrame with mol_id, spec_id, group_id, prec_type columns.
    """
    fp = os.path.join(split_dir, f"{split}_ids.csv")
    df = pd.read_csv(fp)
    df = df.merge(spec_df[["spec_id", "prec_type"]], on="spec_id", how="left")
    missing = df["prec_type"].isna().sum()
    if missing > 0:
        logger.warning("  %d spec_ids in %s have no prec_type in spec_df", missing, fp)
    return df


def write_split(df: pd.DataFrame, out_dir: str, split: str) -> None:
    """Write filtered split dataframe (without prec_type column) to CSV.

    Args:
        df: Filtered split dataframe.
        out_dir: Output directory.
        split: Split name ('train', 'val', 'test').
    """
    os.makedirs(out_dir, exist_ok=True)
    out_fp = os.path.join(out_dir, f"{split}_ids.csv")
    df[["mol_id", "spec_id", "group_id"]].to_csv(out_fp, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split_root",
        default="data/split/spectraverse_benchmark",
        help="Root dir containing cv0, cv1, ... subdirectories.",
    )
    parser.add_argument(
        "--spec_fp",
        default="data/proc/spectraverse/spec_df.pkl",
        help="Path to spec_df.pkl.",
    )
    parser.add_argument(
        "--output_root",
        default="data/split/spectraverse_neims_adduct",
        help="Root dir for output per-adduct split directories.",
    )
    parser.add_argument(
        "--adducts",
        nargs="+",
        default=["[M+H]+"],
        help="Adduct types to extract. Defaults to [M+H]+ only.",
    )
    parser.add_argument(
        "--cv_pattern",
        default="cv",
        help="Prefix for CV fold subdirectory names (default: 'cv').",
    )
    args = parser.parse_args()

    logger.info("Loading spec_df from %s", args.spec_fp)
    spec_df = pd.read_pickle(args.spec_fp)
    logger.info("  %d spectra, adduct distribution:", len(spec_df))
    for adduct, count in spec_df["prec_type"].value_counts().items():
        logger.info("    %-25s %d", adduct, count)

    # find CV fold directories
    cv_dirs = sorted(
        d
        for d in os.listdir(args.split_root)
        if d.startswith(args.cv_pattern)
        and os.path.isdir(os.path.join(args.split_root, d))
    )
    if not cv_dirs:
        raise FileNotFoundError(
            f"No CV directories matching '{args.cv_pattern}*' found in {args.split_root}"
        )
    logger.info("Found %d CV folds: %s", len(cv_dirs), cv_dirs)

    for adduct in args.adducts:
        tag = sanitize_adduct(adduct)
        logger.info("\nProcessing adduct: %s  (tag: %s)", adduct, tag)

        for cv in cv_dirs:
            cv_dir = os.path.join(args.split_root, cv)
            out_cv_dir = os.path.join(args.output_root, tag, cv)

            for split in ("train", "val", "test"):
                fp = os.path.join(cv_dir, f"{split}_ids.csv")
                if not os.path.exists(fp):
                    logger.warning("  Missing %s — skipping", fp)
                    continue

                df = load_split_df(cv_dir, split, spec_df)
                filtered = df[df["prec_type"] == adduct]
                write_split(filtered, out_cv_dir, split)
                logger.info(
                    "  %s/%s: %d → %d spectra (%d molecules)",
                    cv,
                    split,
                    len(df),
                    len(filtered),
                    filtered["mol_id"].nunique(),
                )

    logger.info("\nDone. Splits written to %s", args.output_root)


if __name__ == "__main__":
    main()
