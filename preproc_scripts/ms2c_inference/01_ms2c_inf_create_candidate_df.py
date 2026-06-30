import argparse
import json
import logging
import tarfile
from pathlib import Path
from typing import Any

import pandas as pd
import fragnnet.utils.data_utils as data_utils
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
REQUIRED_SPEC_COLS = ["spec_id", "mol_id", "peaks"]
REQUIRED_MOL_COLS = ["mol_id", "smiles"]


def load_candidates_dict(json_file: str, logger: logging.Logger) -> dict[str, list[str]]:
    """Load a candidates dict mapping query SMILES to candidate SMILES lists."""
    logger.info("Loading JSON from %s", json_file)
    if json_file.endswith(".tar.gz"):
        with tarfile.open(json_file, "r:gz") as tar:
            json_members = [member for member in tar.getmembers(
            ) if member.name.endswith(".json")]
            if not json_members:
                raise ValueError(
                    f"No .json file found in archive: {json_file}")
            if len(json_members) > 1:
                logger.warning(
                    "Multiple .json files found in archive %s. Using first: %s",
                    json_file,
                    json_members[0].name,
                )
            json_file_obj = tar.extractfile(json_members[0])
            if json_file_obj is None:
                raise ValueError(
                    f"Failed to extract {json_members[0].name} from {json_file}")
            with json_file_obj:
                candidates_dict = json.load(json_file_obj)
    else:
        with open(json_file) as f:
            candidates_dict = json.load(f)

    if not isinstance(candidates_dict, dict):
        raise ValueError(
            "JSON root must be a dictionary of query_smiles -> candidate_smiles_list")

    return candidates_dict


def load_split(path):
    splits = os.listdir(path)
    splits = sorted(splits, key=lambda x: int(x.replace("cv", "")))
    print("splits founded:", splits)
    ret_ids = []

    for sp in splits:
        split = pd.read_csv(os.path.join(path, sp, "test_ids.csv"))
        ret_ids.append([set(split["mol_id"]), set(split["spec_id"])])
    return ret_ids


def validate_columns(df: pd.DataFrame, required_cols: list[str], df_name: str) -> None:
    """Validate required columns exist in a DataFrame."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {df_name}: {missing}")


def get_mol_feat(smi: str):
    mol = data_utils.mol_from_smiles(smi)
    assert not mol is None, smi
    save_e = {
        "smiles": smi,
        "mol": mol,
        "inchikey_s": data_utils.mol_to_inchikey_s(mol),
        "scaffold": data_utils.get_murcko_scaffold(mol),
        "formula": data_utils.mol_to_formula(mol),
        "inchi": data_utils.mol_to_inchi(mol),
        "mw": data_utils.mol_to_mol_weight(mol),
        "exact_mw": data_utils.mol_to_mol_weight(mol),
        "num_atoms": data_utils.mol_to_num_atoms(mol),
        "num_bonds": data_utils.mol_to_num_bonds(mol),
        "charge": data_utils.mol_to_charge(mol),
        "single_mol": data_utils.check_single_mol(mol),
        "num_radicals": data_utils.mol_to_num_radicals(mol),
    }
    # check validity
    skip = False
    for k in save_e:
        if save_e[k] is None or (isinstance(save_e[k], float) and np.isnan(save_e[k])):
            print("Encounter none or nan", save_e[k])
            skip = True
    if skip:
        return None
    return save_e

def get_mol_smi_to_id(mol_df, candidate_ids):
    validate_columns(mol_df, REQUIRED_MOL_COLS, "mol_df")
    candidate_mol_ids, candidate_spec_ids = candidate_ids
    mol_smiles_to_id: dict[str, int] = {}
    for _, row in mol_df.iterrows():
        smiles = row["smiles"]
        mol_id = row["mol_id"]
        if smiles in mol_smiles_to_id:
            continue
        if mol_id not in candidate_mol_ids:
            continue
        mol_smiles_to_id[smiles] = int(mol_id)
    return mol_smiles_to_id

def build_mol_df(
    candidates_dict: dict[str, list[str]],
    mol_smiles_to_id: dict,
    output_path: str,
    proc_num,
    logger,
    chunksize=1
):
    save_path = os.path.join(output_path, "mol_df.pkl")
    if os.path.exists(save_path):
        logger.info(f"{save_path} exists, will load from saved results")
        saved_mol_df = pd.read_pickle(save_path)
        map_smi_to_mol_id = {}
        for _, row in saved_mol_df.iterrows():
            map_smi_to_mol_id[row["smiles"]] = row["mol_id"]
        return map_smi_to_mol_id
    # get all candidate mols
    all_candidate_smis = []
    for query_smiles in tqdm(candidates_dict):
        if query_smiles not in mol_smiles_to_id:
            continue
        all_candidate_smis.extend(candidates_dict[query_smiles])
    all_candidate_smis=list(set(all_candidate_smis))
    logger.info(f"Candidate num is {len(all_candidate_smis)}")
    map_smi_to_mol_id = {}
    with Pool(proc_num) as pool:
        all_candidate_feats = list(
            tqdm(
                pool.imap(get_mol_feat, all_candidate_smis),
                total=len(all_candidate_smis)
            )
        )
    num_failed = [_ for _ in all_candidate_feats if _ is None]
    logger.info(f"{num_failed} failed in {len(all_candidate_feats)} in mol")
    mol_records = []
    for save_e in all_candidate_feats:
        if save_e is None:
            continue
        smi = save_e["smiles"]
        save_e["mol_id"] = len(mol_records)
        mol_records.append(
            save_e
        )
        assert smi not in map_smi_to_mol_id
        map_smi_to_mol_id[smi] = mol_records[-1]["mol_id"]
    logger.info("Finish processing mol, saving")
    pd.DataFrame(mol_records).to_pickle(save_path)
    logger.info("Finish saving mol")
    return map_smi_to_mol_id


def build_candidate_df(
    candidates_dict: dict[str, list[str]],
    spec_df: pd.DataFrame,
    mol_df: pd.DataFrame,
    logger: logging.Logger,
    output_path: str,
    candidate_ids: set,
    map_smi_to_mol_id: dict,
    mol_smiles_to_id:dict
) -> pd.DataFrame:
    """Build candidate_df compatible with ms2c_inference candidate_to_query_sets."""
    validate_columns(spec_df, REQUIRED_SPEC_COLS, "spec_df")
    validate_columns(mol_df, REQUIRED_MOL_COLS, "mol_df")
    candidate_mol_ids, candidate_spec_ids = candidate_ids

    spec_records: list[dict[str, Any]] = []

    for query_smiles in tqdm(candidates_dict):
        if query_smiles not in mol_smiles_to_id:
            continue
        candidate_smiles_list =list(set(candidates_dict[query_smiles])) 
        #assert len(candidate_smiles_list) == len(set(candidate_smiles_list))
        if not isinstance(candidate_smiles_list, list):
            raise ValueError(f"Expected list of candidates for {query_smiles}")
        if query_smiles not in mol_smiles_to_id:
            raise ValueError(
                f"Query SMILES not found in mol_df: {query_smiles}")

        query_mol_id = mol_smiles_to_id[query_smiles]
        query_spec_rows = spec_df[spec_df["mol_id"] == query_mol_id]
        if query_spec_rows.empty:
            raise ValueError(
                f"No spec_df rows found for query mol_id={query_mol_id}")
        for smi in candidate_smiles_list:
            if smi not in map_smi_to_mol_id:
                continue
            for _, spec_row in query_spec_rows.iterrows():
                if spec_row["spec_id"] not in candidate_spec_ids:
                    continue
                save_e = {}
                for k in spec_row.keys():
                    save_e[k] = spec_row[k]
                save_e["mol_id"] = map_smi_to_mol_id[smi]
                save_e["spec_id"] = len(spec_records)
                save_e["gt_spec_id"] = spec_row["spec_id"]
                spec_records.append(save_e)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    save_spec_records = pd.DataFrame(spec_records)
    save_spec_records.to_pickle(os.path.join(output_path, "spec_df.pkl"))
    logger.info(f"saving {len(spec_records)} spec_records to {output_path}")

    # generate split
    pd.DataFrame.from_dict({
        "mol_id": save_spec_records["mol_id"],
        "spec_id": save_spec_records["spec_id"],
        "group_id": save_spec_records["spec_id"],
    }).to_csv(os.path.join(output_path, "test_ids.csv"), index=False)


if __name__ == "__main__":
    # The following code reformat the json file to similar format as training
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Create candidate SMILES set from JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Example:
            python create_candidate_set.py --json_file candidates.json
        """,
    )

    parser.add_argument(
        "--json_file",
        type=str,
        # required=True,
        default="data/ms2c/nps/nps_candidates/nps_nist23/nps_nist23_clm_formula.json",
        help="Path to JSON file containing query SMILES and candidate SMILES lists",
    )

    parser.add_argument(
        "--spec_df_path",
        type=str,
        # required=True,
        default="data/proc/nps_nist23/spec_df.pkl",
        help="path to spec_df.pkl.gz file created by 02_prepare_spec_df.py which has specid and metadata",
    )
    parser.add_argument(
        "--mol_df_path",
        type=str,
        # required=True,
        default="data/proc/nps_nist23/mol_df.pkl",
        help="path to mol_df.pkl.gz file created by 02_prepare_spec_df.py which has molid and metadata",
    )
    parser.add_argument(
        "--split_path",
        type=str,
        default=None,
        help="path to training splits",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        # required=True,
        default="data/ms2c/candidates/nps_nist23_candidate_df.pkl.gz",
        help="Path to output candidate_df.pkl.gz",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        # required=True,
        default=20,
        help="the number of process to use",
    )
    parser.add_argument(
        "--split_idx",
        type=int,
        # required=True,
        default=None,
        help="split to handle",
    )
    args = parser.parse_args()

    # Load  files
    candidates_dict = load_candidates_dict(args.json_file, logger)
    query_smiles_list = list(candidates_dict.keys())
    logger.info(f"Total query SMILES collected: {len(query_smiles_list)}")
    mol_df = pd.read_pickle(args.mol_df_path)
    logger.info("Loaded mol_df with %d molecules", len(mol_df))
    spec_df = pd.read_pickle(args.spec_df_path)
    logger.info("Loaded spec_df with %d spectra", len(spec_df))
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Load split file, to consistant with cross-validation
    if args.split_path is not None:
        splits = load_split(args.split_path)
    else:
        splits=[[set(spec_df["mol_id"]), set(spec_df["spec_id"])]]
        
    if args.split_idx is None:
        candidates_split_idx=list(range(len(splits)))
    else:
        candidates_split_idx=[args.split_idx]
    
    # build spec_df
    for i in candidates_split_idx:
        s=splits[i]
        save_path=os.path.join(args.output_path, f"{i}")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        mol_smiles_to_id=get_mol_smi_to_id(mol_df, s)
        logger.info(f"{len(mol_smiles_to_id)} query mols loaded for split {i}")
        map_smi_to_mol_id = build_mol_df(candidates_dict=candidates_dict, output_path=save_path, logger=logger, proc_num=args.num_proc, mol_smiles_to_id=mol_smiles_to_id)
        build_candidate_df(candidates_dict, spec_df, mol_df, logger, save_path, s, map_smi_to_mol_id, mol_smiles_to_id=mol_smiles_to_id)

    # Create output directory if it doesn't exist
    logger.info("Saved candidate_df to %s", args.output_path)
