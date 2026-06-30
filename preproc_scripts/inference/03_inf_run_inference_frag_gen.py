import argparse
import logging
import os
import time
from typing import cast

import pandas as pd
from tqdm import tqdm

import fragnnet.utils.data_utils as data_utils
import fragnnet.utils.frag_utils as frag_utils
from fragnnet.utils.misc_utils import booltype


def get_args() -> argparse.Namespace:
    """
    Parse command line arguments for fragment generation configuration.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - start_row/end_row: Row indices for chunked processing
            - Fragmentation parameters (max_depth, max_time, etc.)
            - Input/output directory paths
            - Processing options (caching, compression, progress display)
    """
    parser = argparse.ArgumentParser()
    # Chunking parameters for memory management and parallel processing
    parser.add_argument(
        "--start_row",
        type=int,
        default=0,
        help="Start row index (inclusive) for chunked processing",
    )
    parser.add_argument(
        "--end_row", type=int, default=None, help="End row index (exclusive) for chunked processing"
    )

    # Fragment generation algorithm parameters
    parser.add_argument(
        "--max_depth", type=int, default=3, help="Maximum fragmentation depth for DAG generation"
    )
    parser.add_argument(
        "--max_time",
        type=int,
        default=150,
        help="Maximum time (seconds) allowed for fragment generation per molecule, -1 for unlimited",
    )
    parser.add_argument(
        "--max_h_transfer",
        type=int,
        default=4,
        help="Maximum hydrogen transfers allowed during fragmentation",
    )
    parser.add_argument(
        "--wl_max_iterations",
        type=int,
        default=-1,
        help="Maximum iterations for wl hash (-1 for unlimited)",
    )

    # Directory and file paths
    parser.add_argument(
        "--project_dp", type=str, default=os.getcwd(), help="Project root directory path"
    )
    parser.add_argument(
        "--frag_dp", type=str, required=True, help="Path to the fragmentation output directory"
    )
    parser.add_argument(
        "--proc_dp",
        type=str,
        required=True,
        help="Path to the preprocessed mol_df and spec_df directory",
    )

    # Chemical and algorithmic constraints
    parser.add_argument(
        "--allowed_elements",
        type=str,
        nargs="+",
        default=frag_utils.ELEMENTS,
        help="List of allowed chemical elements",
    )
    parser.add_argument(
        "--nb_isomorphic",
        type=booltype,
        default=False,
        help="Whether to consider non-bonded isomorphic structures",
    )
    parser.add_argument(
        "--isotopes", type=booltype, default=False, help="Whether to consider isotopic variations"
    )

    # Processing options
    parser.add_argument(
        "--allow_cached",
        type=booltype,
        default=True,
        help="Allow using cached fragmentation results",
    )
    parser.add_argument(
        "--compressed", type=booltype, default=True, help="Use compression for output files"
    )
    parser.add_argument(
        "--show_tqdm",
        type=booltype,
        default=False,
        help="Show tqdm progress bar instead of interval logging",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Configure logging with timestamps for monitoring progress
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    logger = logging.getLogger("frag_gen")

    logger.info("=" * 60)
    logger.info("Starting MS2C Fragment Generation Pipeline")
    logger.info("=" * 60)

    # Parse command line arguments
    args = get_args()
    logger.info("Parsed command line arguments successfully")
    logger.info("Configuration parameters:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("-" * 40)

    # Locate and load molecular data file (supports both .pkl and .pkl.gz)
    logger.info("Step 1: Loading molecular data")
    mol_df_fp = os.path.join(args.proc_dp, "mol_df.pkl")
    logger.info(f"Searching for mol_df at: {mol_df_fp}")

    if os.path.isfile(mol_df_fp):
        logger.info(f"Found uncompressed mol_df: {mol_df_fp}")
        file_size_mb = os.path.getsize(mol_df_fp) / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")
    elif os.path.isfile(mol_df_fp + ".gz"):
        mol_df_fp = mol_df_fp + ".gz"
        logger.info(f"Found compressed mol_df: {mol_df_fp}")
        file_size_mb = os.path.getsize(mol_df_fp) / (1024 * 1024)
        logger.info(f"Compressed file size: {file_size_mb:.2f} MB")
    else:
        logger.error(f"mol_df.pkl or mol_df.pkl.gz not found in {args.proc_dp}")
        logger.error("Available files in directory:")
        try:
            for f in os.listdir(args.proc_dp):
                logger.error(f"  {f}")
        except Exception as e:
            logger.error(f"Could not list directory contents: {e}")
        raise ValueError(f"mol_df.pkl or mol_df.pkl.gz not found in {args.proc_dp}")

    # Load molecular dataframe containing RDKit molecule objects and metadata
    logger.info(f"Loading molecular dataframe from: {mol_df_fp}")
    try:
        mol_df = pd.read_pickle(mol_df_fp)
        logger.info(f"Successfully loaded mol_df with {len(mol_df)} molecules")
        logger.info(f"Dataframe columns: {list(mol_df.columns)}")
        logger.info(f"Memory usage: {mol_df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
    except Exception as e:
        logger.error(f"Failed to load mol_df: {e}")
        raise

    # Create output directory for fragment data
    logger.info("Step 2: Setting up output directory")
    logger.info(f"Output directory: {args.frag_dp}")

    if os.path.exists(args.frag_dp):
        logger.info("Output directory already exists")
        existing_files = len(
            [
                f
                for f in os.listdir(args.frag_dp)
                if f.endswith(".pickle") or f.endswith(".pickle.bz2")
            ]
        )
        logger.info(f"Found {existing_files} existing fragment files in output directory")
        if args.allow_cached:
            logger.info("Caching enabled - will skip molecules with existing fragment files")
        else:
            logger.info("Caching disabled - will regenerate all fragments")
    else:
        logger.info("Creating new output directory")
        os.makedirs(args.frag_dp, exist_ok=True)
        logger.info(f"Successfully created directory: {args.frag_dp}")

    logger.info("-" * 40)

    # Validate and configure chunk boundaries for processing subset of molecules
    logger.info("Step 3: Configuring chunk boundaries")
    start_row = args.start_row
    end_row = args.end_row if args.end_row is not None else len(mol_df)

    logger.info(f"Original dataset size: {len(mol_df)} molecules")
    logger.info(f"Requested chunk: rows {start_row} to {end_row}")

    # Input validation for chunk boundaries
    if start_row < 0 or start_row >= len(mol_df):
        logger.error(f"start_row {start_row} is out of bounds for mol_df of length {len(mol_df)}")
        raise ValueError(
            f"start_row {start_row} is out of bounds for mol_df of length {len(mol_df)}"
        )
    if end_row > len(mol_df):
        logger.warning(
            f"end_row {end_row} is greater than mol_df length {len(mol_df)}, adjusting to {len(mol_df)}"
        )
        end_row = len(mol_df)
    if end_row <= start_row:
        logger.error(f"end_row {end_row} must be greater than start_row {start_row}")
        raise ValueError(f"end_row {end_row} must be greater than start_row {start_row}")

    # Extract the specified chunk of molecules to process
    mol_df_chunk = mol_df.iloc[start_row:end_row]
    chunk_size = len(mol_df_chunk)
    percentage = (chunk_size / len(mol_df)) * 100
    logger.info(
        f"Processing chunk: rows {start_row}-{end_row} ({chunk_size} molecules, {percentage:.1f}% of total)"
    )

    # Log some basic statistics about the chunk
    if "mol_id" in mol_df_chunk.columns:
        logger.info(
            f"Molecule ID range: {mol_df_chunk['mol_id'].min()} to {mol_df_chunk['mol_id'].max()}"
        )
    if "num_heavy_atoms" in mol_df_chunk.columns:
        logger.info(
            f"Heavy atom count range: {mol_df_chunk['num_heavy_atoms'].min()}-{mol_df_chunk['num_heavy_atoms'].max()}"
        )
        logger.info(f"Average heavy atoms: {mol_df_chunk['num_heavy_atoms'].mean():.1f}")

    logger.info("-" * 40)

    # Prepare input parameters for fragment generation
    # Each entry contains all parameters needed for timed_get_dags function
    logger.info("Step 4: Preparing fragment generation inputs")
    dag_feat_inputs_l = []

    logger.info("Building parameter list for parallel processing...")
    logger.info("Fragment generation parameters:")
    logger.info(f"  Max depth: {args.max_depth}")
    logger.info(f"  Max time per molecule: {args.max_time} seconds")
    logger.info(f"  Max H transfers: {args.max_h_transfer}")
    logger.info(f"  Max WL iterations: {args.wl_max_iterations}")
    logger.info(f"  Include isotopes: {args.isotopes}")
    logger.info(f"  Non-bonded isomorphic: {args.nb_isomorphic}")
    logger.info(f"  Use caching: {args.allow_cached}")
    logger.info(f"  Use compression: {args.compressed}")

    skipped_cached = 0
    for idx, (_, row) in enumerate(mol_df_chunk.iterrows()):
        if args.allow_cached:
            cached_fp = frag_utils.get_frag_fp(row["mol_id"], args.frag_dp, args.compressed)
            if os.path.isfile(cached_fp):
                skipped_cached += 1
                continue

        dag_feat_inputs_l.append(
            [
                row["mol"],  # RDKit molecule object
                row["mol_id"],  # Unique molecule identifier
                args.max_depth,  # Maximum fragmentation depth
                True,  # h_prior: use hydrogen prior information
                args.max_h_transfer,  # Maximum hydrogen transfers
                args.max_time,  # Time limit per molecule (seconds)
                args.isotopes,  # Consider isotopic variations
                args.nb_isomorphic,  # Consider non-bonded isomorphic structures
                args.wl_max_iterations,  # Maximum fragmentation iterations
            ]
        )

        # Log progress for large datasets
        if (idx + 1) % 1000 == 0:
            logger.info(f"Prepared inputs for {idx + 1}/{len(mol_df_chunk)} molecules")

    logger.info(f"Completed input preparation for {len(dag_feat_inputs_l)} molecules")
    if skipped_cached > 0:
        logger.info(f"Skipped {skipped_cached} molecules with cached fragments")
    logger.info("-" * 40)

    # Execute parallel fragment generation using FraGNNet algorithm
    logger.info("Step 5: Executing fragment generation")
    logger.info(f"Starting parallel fragment generation for {len(dag_feat_inputs_l)} molecules")
    logger.info("Using FraGNNet timed_get_dags function with parallel processing")

    start_time = time.time()

    # Use parallel processing to generate fragment DAGs for all molecules
    # par_apply handles the multiprocessing and returns a generator for memory efficiency
    try:
        frag_results = data_utils.par_apply(
            iter(dag_feat_inputs_l), frag_utils.timed_get_dags, True, return_as_generator=True
        )
        if frag_results is None:
            raise RuntimeError("Parallel fragment generation did not return any results")
        frag_results_iter = cast("list[tuple[str, dict]]", frag_results)
        logger.info("Successfully initialized parallel fragment generation")
    except Exception as e:
        logger.error(f"Failed to initialize parallel processing: {e}")
        raise

    # Process results and track progress with enhanced logging
    n_done = 0
    n_errors = 0
    last_log_time = start_time

    if not args.show_tqdm:
        # Use interval-based logging instead of progress bar for better log files
        logger.info("Processing molecules with interval-based progress logging...")
        for mol_id, dag_d in frag_results_iter:
            n_done += 1
            if not dag_d:
                n_errors += 1
            else:
                frag_utils.save_frag_d(dag_d, mol_id, args.frag_dp, args.compressed)
            current_time = time.time()

            # Log every 100 molecules OR every 30 seconds
            if n_done % 100 == 0 or (current_time - last_log_time) >= 30:
                elapsed = current_time - start_time
                rate = n_done / elapsed if elapsed > 0 else 0
                remaining = len(dag_feat_inputs_l) - n_done
                eta = remaining / rate if rate > 0 else 0

                logger.info(
                    f"Progress: {n_done}/{len(dag_feat_inputs_l)} molecules ({n_done / len(dag_feat_inputs_l) * 100:.1f}%)"
                )
                logger.info(f"  Rate: {rate:.2f} molecules/sec")
                logger.info(f"  Elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s")
                if n_errors > 0:
                    logger.warning(f"  Errors encountered: {n_errors}")
                last_log_time = current_time
    else:
        # Use tqdm progress bar for interactive monitoring
        logger.info("Processing molecules with tqdm progress bar...")
        for mol_id, dag_d in tqdm(
            frag_results_iter, total=len(dag_feat_inputs_l), desc="Compute Frags"
        ):
            n_done += 1
            if not dag_d:
                n_errors += 1
            else:
                frag_utils.save_frag_d(dag_d, mol_id, args.frag_dp, args.compressed)
            current_time = time.time()

            # Still log every 100 molecules for permanent record
            if n_done % 100 == 0:
                elapsed = current_time - start_time
                rate = n_done / elapsed if elapsed > 0 else 0
                logger.info(f"Processed {n_done} molecules (rate: {rate:.2f}/sec)")
                if n_errors > 0:
                    logger.warning(f"Errors encountered: {n_errors}")

    # Log completion summary with timing and statistics
    end_time = time.time()
    total_elapsed = end_time - start_time
    avg_rate = n_done / total_elapsed if total_elapsed > 0 else 0

    logger.info("=" * 60)
    logger.info("Fragment generation completed successfully!")
    logger.info(f"Total molecules processed: {n_done}")
    logger.info(f"Total time elapsed: {total_elapsed:.2f} seconds")
    logger.info(f"Average processing rate: {avg_rate:.2f} molecules/second")
    if n_errors > 0:
        logger.warning(f"Total errors encountered: {n_errors}")
        logger.warning(f"Success rate: {(n_done - n_errors) / n_done * 100:.1f}%")
    else:
        logger.info("No errors encountered - 100% success rate")

    # Log output directory information
    try:
        output_files = [
            f
            for f in os.listdir(args.frag_dp)
            if f.endswith(".pickle") or f.endswith(".pickle.bz2")
        ]
        logger.info(f"Generated {len(output_files)} fragment files in {args.frag_dp}")
        total_size = sum(os.path.getsize(os.path.join(args.frag_dp, f)) for f in output_files) / (
            1024 * 1024
        )
        logger.info(f"Total output size: {total_size:.2f} MB")
    except Exception as e:
        logger.warning(f"Could not analyze output directory: {e}")

    logger.info("=" * 60)
