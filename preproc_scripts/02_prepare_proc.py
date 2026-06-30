import argparse
import logging
import os
from functools import partial

import numpy as np
import pandas as pd
from rdkit import RDLogger

import fragnnet.utils.data_utils as data_utils
from fragnnet.utils.data_utils import (
    par_apply_df_rows,
    par_apply_df_rows_t,
    par_apply_series,
    seq_apply_df_rows_t,
    compute_mol_df_props,
)

from fragnnet.utils.misc_utils import booltype

# Module-level logger
logger = logging.getLogger(__name__)

"""Prepare processed spectral, molecule, and annotation tables.

Outputs written to `proc_dp`:
- spec_df.pkl: one row per spectrum with columns
    - spec_id: integer id unique within processed set
    - mol_id: integer id linking to mol_df
    - prec_type: standardized precursor/adduct string
    - inst_type: standardized instrument type
    - frag_mode: standardized fragmentation mode
    - spec_type: spectrum type (e.g., MS2)
    - ion_mode: P/N
    - dset: dataset name (eg. nist20hr)
    - dset_spec_id: source-specific spectrum id
    - col_gas: collision gas (may be NaN)
    - res: inferred m/z resolution # this is not used
    - ace/nce:
        - ace/nce: parsed collision energies (absolute and normalized)
        - ace_extra_1: parsed collision energies (absolute and normalized) for additional collision energies
        - ace_extra_2: parsed collision energies (absolute and normalized) for additional collision energies
        for ramped: ace/nce and ace_extra_1/ace_extra_1 are set to the first and second collision energies
        for stepped: ace/nce,ace_extra_1/ace_extra_1, ace_extra_2/ace_extra_2 are set to the 3 collision energies
    - ce_type: collision energy pattern {none,single,ramped,stepped}
    - prec_mz: precursor m/z (float; inferred when missing)
    - peaks: list of (mz, intensity) pairs as floats
    - ri: retention index if present
    - formula: molecular formula from source
    - inchikey: full InChIKey from source when available
    - exact_mass: exact mass from source when available
    - group_id: grouping id for spectra sharing compound/precursor.
        Current grouping key: `mol_id` + `prec_type` + `inst_type`.
        Note: `ce_type` and `dset` are excluded from the group key so
        that identical molecules/adducts across datasets or CE patterns
        are grouped together (prevents molecule-level leakage across
        datasets when training on mixed sources).

- mol_df.pkl: one row per unique SMILES with columns
    - smiles: canonicalized SMILES
    - mol_id: integer id
    - mol: RDKit Mol object
    - inchikey_s: first 14 chars of InChIKey
    - scaffold: Murcko scaffold SMILES
    - formula: molecular formula
    - inchi: InChI string
    - mw/exact_mw: molecular weights (average/exact)
    - num_atoms/num_bonds/num_radicals: structural counts
    - charge: formal charge
    - single_mol: flag for single connected component

- ann_df.pkl: one row per annotated spectrum with columns
    - dset_spec_id/spec_id/mol_id
    - prec_type: standardized precursor/adduct
    - formula: molecular formula from mol_df
    - ann_peak_mzs: list of annotated peak m/z values
    - ann_products: list of product formula strings
    - ann_losses: list of neutral loss formula strings
    - ann_isotopes: list of isotope annotations
    - ann_exact_mzs: list of exact m/z values for annotations

Additional debug pickles (no_mol_df, diff_formula_df, diff_inchikey_df, diff_mass_df)
capture entries dropped or with inconsistencies.
"""


def load_df(df_dp: str, dsets: list[str], num_entries: int) -> pd.DataFrame:
    dfs = []
    for dset in dsets:
        dset_df = pd.read_csv(os.path.join(df_dp, f"{dset}_df.csv"), dtype=str)
        dset_df.loc[:, "dset"] = dset
        dfs.append(dset_df)
    if num_entries > 0:
        dfs = [df.sample(n=num_entries, replace=False, random_state=420) for df in dfs]
    if len(dfs) > 1:
        all_df = pd.concat(dfs, ignore_index=True)
    else:
        all_df = dfs[0]
    all_df = all_df.reset_index(drop=True)
    return all_df


def preprocess_spec(
    spec_df: pd.DataFrame,
    canonicalize_tautomers: bool = True,
    skip_standardize: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    # drop entries with the same dset_spec_id (this happens sometimes in MoNA)
    spec_df = spec_df.drop_duplicates(subset=["dset", "dset_spec_id"], keep="first")
    # convert smiles to mol and back (for standardization/stereochemistry)
    if skip_standardize:
        spec_df.loc[:, "mol"] = par_apply_series(spec_df["smiles"], partial(data_utils.mol_from_smiles, ml_standardize=False))
    else:
        spec_df.loc[:, "mol"] = par_apply_series(spec_df["smiles"], data_utils.mol_from_smiles)
    spec_df.loc[:, "smiles"] = par_apply_series(spec_df["mol"], data_utils.mol_to_smiles)
    no_mol_df = spec_df[spec_df["mol"].isna() | spec_df["smiles"].isna()][["dset_spec_id"]]
    spec_df = spec_df.dropna(subset=["mol", "smiles"])
    if (spec_df["smiles"] == "").any():
        raise ValueError("Empty SMILES encountered after molecule standardization")
    # enumerate smiles to create molecule ids
    smiles_set = set(spec_df["smiles"])
    if "" in smiles_set:
        raise ValueError("Empty SMILES present in smiles_set")
    logger.info(f"> num_smiles {len(smiles_set)}")
    logger.info("> sorting by smiles")
    smiles_to_mid = {smiles: i for i, smiles in enumerate(sorted(smiles_set))}
    logger.info("> updating mol_id")
    spec_df.loc[:, "mol_id"] = spec_df["smiles"].map(smiles_to_mid)  # .replace(smiles_to_mid)

    # copy for ann_df
    ann_df = spec_df[["dset_spec_id", "mol_id", "notes", "peaks", "dset"]].copy()

    # extract peak info (still represented as str)
    spec_df.loc[:, "peaks"] = par_apply_series(spec_df["peaks"], data_utils.parse_peaks_str)
    # get mz resolution
    spec_df.loc[:, "res"] = par_apply_series(spec_df["peaks"], data_utils.get_res)
    # standardize the instrument type and frag_mode
    inst_type, frag_mode = seq_apply_df_rows_t(spec_df, data_utils.parse_inst_info)
    spec_df.loc[:, "inst_type"] = inst_type
    spec_df.loc[:, "frag_mode"] = frag_mode
    # standardize ce
    # NOTE: this assume ace will ends with eV and NCE will ends with %
    spec_df.loc[:, "ace"] = par_apply_series(spec_df["col_energy"], data_utils.parse_ace_str)
    spec_df.loc[:, "nce"] = par_apply_series(spec_df["col_energy"], data_utils.parse_nce_str)
    spec_df.loc[:, "ace_extra_1"] = par_apply_series(
        spec_df["col_energy_extra_1"], data_utils.parse_ace_str
    )
    spec_df.loc[:, "nce_extra_1"] = par_apply_series(
        spec_df["col_energy_extra_1"], data_utils.parse_nce_str
    )
    spec_df.loc[:, "ace_extra_2"] = par_apply_series(
        spec_df["col_energy_extra_2"], data_utils.parse_ace_str
    )
    spec_df.loc[:, "nce_extra_2"] = par_apply_series(
        spec_df["col_energy_extra_2"], data_utils.parse_nce_str
    )
    spec_df = spec_df.drop(columns=["col_energy", "col_energy_extra_1", "col_energy_extra_2"])

    # summarize collision energy pattern
    has_any_ce = (
        spec_df[["ace", "nce", "ace_extra_1", "nce_extra_1", "ace_extra_2", "nce_extra_2"]]
        .notna()
        .any(axis=1)
    )
    has_extra_1 = spec_df[["ace_extra_1", "nce_extra_1"]].notna().any(axis=1)
    has_extra_2 = spec_df[["ace_extra_2", "nce_extra_2"]].notna().any(axis=1)
    spec_df.loc[:, "ce_type"] = np.select(
        [~has_any_ce, has_extra_2, has_extra_1], ["none", "stepped", "ramped"], default="single"
    )
    # spec_df.loc[:,'has_ace_extra_1'] = ~spec_df["ace_extra_1"].isna()

    # standardise prec_type
    spec_df.loc[:, "prec_type"] = par_apply_series(
        spec_df["prec_type"], data_utils.parse_prec_type_str
    )
    # convert prec_mz
    spec_df.loc[:, "prec_mz"] = pd.to_numeric(spec_df["prec_mz"], errors="coerce")
    # convert ion_mode
    spec_df.loc[:, "ion_mode"] = par_apply_series(
        spec_df["ion_mode"], data_utils.parse_ion_mode_str
    )
    # convert peaks to float
    spec_df.loc[:, "peaks"] = par_apply_series(spec_df["peaks"], data_utils.convert_peaks_to_float)
    # get retention index
    spec_df.loc[:, "ri"] = par_apply_series(spec_df["ri"], data_utils.parse_ri_str)
    # convert exact_mass
    spec_df.loc[:, "exact_mass"] = pd.to_numeric(spec_df["exact_mass"], errors="coerce")
    if "formula" not in spec_df:
        spec_df.loc[:, "formula"] = par_apply_series(spec_df["mol"], data_utils.mol_to_formula)
    # remove columns from spec_df
    spec_df = spec_df[
        [
            "spec_id",
            "mol_id",
            "prec_type",
            "inst_type",
            "frag_mode",
            "spec_type",
            "ion_mode",
            "dset",
            "dset_spec_id",
            "col_gas",
            "res",
            "ace",
            "ace_extra_1",
            "ace_extra_2",
            "nce",
            "nce_extra_1",
            "nce_extra_2",
            "ce_type",
            "prec_mz",
            "peaks",
            "ri",
            "formula",
            "inchikey",
            "exact_mass",
        ]
    ]
    # relabel spec_id (this is to make it unique across datasets)
    spec_df.loc[:, "spec_id"] = np.arange(spec_df.shape[0])
    # set group_id (group by mol_id + prec_type + inst_type)
    # Note: `ce_type` and `dset` are intentionally dropped so groups
    # combine identical molecules/adducts across datasets and CE patterns.

    group_key_cols = ["mol_id", "prec_type", "inst_type"]
    group_df = (
        spec_df[group_key_cols].drop_duplicates().sort_values(group_key_cols).reset_index(drop=True)
    )
    group_df.loc[:, "group_id"] = np.arange(group_df.shape[0])
    spec_df = spec_df.merge(group_df, how="inner", on=group_key_cols)

    # get mol df
    mol_df = pd.DataFrame(
        zip(sorted(smiles_set), list(range(len(smiles_set)))), columns=["smiles", "mol_id"]
    )
    
    # convert smiles to mol
    # use skip_standardize flag to control whether to apply the same standardization as in the spec_df processing step, this is to save time when we want to keep the original mols without standardization
    mol_df.loc[:, "mol"] = par_apply_series(mol_df["smiles"], partial(data_utils.mol_from_smiles, ml_standardize=not skip_standardize, canonicalize_tautomers=canonicalize_tautomers))
    
    compute_mol_df_props(mol_df)
    if (mol_df["smiles"] == "").any():
        raise ValueError("Empty SMILES found in mol_df")
    if (mol_df["formula"] == "").any():
        raise ValueError("Empty formula found in mol_df")
    if (mol_df["exact_mw"] == 0).any():
        raise ValueError("Zero exact_mw found in mol_df")

    # remove invalid mols and corresponding spectra
    all_mol_id = set(mol_df["mol_id"])
    mol_df = mol_df.dropna(subset=["mol"], axis=0)
    bad_mol_id = all_mol_id - set(mol_df["mol_id"])
    logger.info(f"> bad_mol_id {len(bad_mol_id)}")
    spec_df = spec_df[~spec_df["mol_id"].isin(bad_mol_id)]

    # check how many formulae/inchikeys are different
    formula_df = spec_df[["dset_spec_id", "spec_id", "mol_id", "formula"]].copy()
    formula_df = formula_df[formula_df["formula"] != ""]
    formula_df = formula_df.merge(mol_df[["mol_id", "formula"]], on="mol_id", how="inner")
    diff_formula_df = formula_df[formula_df["formula_x"] != formula_df["formula_y"]]

    logger.info("> formula inconsistencies")
    logger.info(
        f"> diff mol: {diff_formula_df['mol_id'].nunique()}, diff spec: {diff_formula_df['spec_id'].nunique()}"
    )
    inchikey_df = spec_df[["dset_spec_id", "spec_id", "mol_id", "inchikey"]].copy().dropna()
    inchikey_df["inchikey_s"] = inchikey_df["inchikey"].str[:14]
    inchikey_df = inchikey_df.drop(columns=["inchikey"])
    inchikey_df = inchikey_df.merge(mol_df[["mol_id", "inchikey_s"]], on="mol_id", how="inner")
    
    diff_inchikey_df = inchikey_df[inchikey_df["inchikey_s_x"] != inchikey_df["inchikey_s_y"]]
    logger.info("> inchikey_s inconsistencies")
    logger.info(
        f"> diff mol: {diff_inchikey_df['mol_id'].nunique()}, diff spec: {diff_inchikey_df['spec_id'].nunique()}"
    )

    mass_df = (
        spec_df[["dset_spec_id", "spec_id", "mol_id", "exact_mass"]]
        .copy()
        .rename(columns={"exact_mass": "exact_mw"})
    )
    # keep only non-trivial exact_mw annotations from the spec_df
    mass_df = mass_df[(~mass_df["exact_mw"].isna()) & (mass_df["exact_mw"] > 0.0)]
    # compare with reported exact_mw from the mol_df
    mass_df = mass_df.merge(mol_df[["mol_id", "exact_mw"]], on="mol_id", how="inner")
    diff_mass_df = mass_df[(mass_df["exact_mw_x"] - mass_df["exact_mw_y"]).abs() > 0.1]
    logger.info("> mass inconsistencies")
    logger.info(
        f"> diff mol: {diff_mass_df['mol_id'].nunique()}, diff spec: {diff_mass_df['spec_id'].nunique()}"
    )
    spec_df = spec_df.drop(columns=["formula", "inchikey", "exact_mass"])

    # fill in missing prec_mz by inferring them
    spec_df = spec_df[~spec_df["spec_id"].isin(diff_formula_df["spec_id"])]
    spec_df = spec_df[~spec_df["spec_id"].isin(diff_mass_df["spec_id"])]
    spec_df = spec_df.merge(mol_df[["mol_id", "exact_mw"]], on="mol_id", how="inner")
    spec_df.loc[:, "prec_mz"] = par_apply_df_rows(spec_df, data_utils.infer_prec_mz)
    spec_df = spec_df.drop(columns=["exact_mw"])

    # extract annotation info
    ann_df = ann_df.merge(
        spec_df[["dset_spec_id", "spec_id", "prec_type"]], on="dset_spec_id", how="inner"
    )
    ann_df = ann_df.merge(mol_df[["mol_id", "formula"]], on="mol_id", how="inner")
    ann_results = par_apply_df_rows_t(ann_df, data_utils.parse_annotations)
    ann_df.loc[:, "ann_peak_mzs"] = ann_results[0]
    ann_df.loc[:, "ann_products"] = ann_results[1]
    ann_df.loc[:, "ann_losses"] = ann_results[2]
    ann_df.loc[:, "ann_isotopes"] = ann_results[3]
    ann_df.loc[:, "ann_exact_mzs"] = ann_results[4]
    ann_df = ann_df.dropna(axis=0, how="any")
    ann_df = ann_df[ann_df["ann_peak_mzs"].apply(len) > 0]
    ann_df = ann_df.drop(columns=["notes", "peaks"])
    ann_df = ann_df.reset_index(drop=True)

    # reset indices
    spec_df = spec_df.reset_index(drop=True)
    mol_df = mol_df.reset_index(drop=True)

    debug_dfs = {
        "no_mol_df": no_mol_df,
        "diff_formula_df": diff_formula_df,
        "diff_inchikey_df": diff_inchikey_df,
        "diff_mass_df": diff_mass_df,
    }

    return spec_df, mol_df, ann_df, debug_dfs


if __name__ == "__main__":
    # Configure logging once when running as a script
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    RDLogger.DisableLog("rdApp.*")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--df_dp",
        type=str,
        default="data/df",
        help="this script assumes the df_dp contains the dataset csv files, as <dset_name>_df.csv) e.g. nist20_hr_df.csv, ms_gym_df.csv, etc.",
    )
    parser.add_argument("--proc_dp", type=str, required=True)
    parser.add_argument("--num_entries", type=int, default=-1)
    parser.add_argument(
        "--ow_spec_mol", type=booltype, default=True, help="overwrite existing spec mol"
    )
    parser.add_argument(
        "--dsets",
        type=str,
        nargs="+",
        required=True,
        help="dataset names to process, e.g. nist20_hr, ms_gym, ms_gym_extra.",
    )
    parser.add_argument(
        "--na_str_for_missing",
        type=booltype,
        default=True,
        help="whether use NA for missing str labels (such as inst_type, frag_mode, etc.) or use np.nan",
    )
    parser.add_argument(
        "--skip_standardize",
        action="store_true",
        help="whether skip standardize for mols"
    )
    parser.add_argument(
        "--canonicalize_tautomers",
        type=booltype,
        default=True,
        help="whether to canonicalize tautomers during mol standardization",
    )
    args = parser.parse_args()

    logger.info(f"> df_dp: {args.df_dp}")
    logger.info(f"> proc_dp: {args.proc_dp}")
    logger.info(f"> num_entries: {args.num_entries}")
    logger.info(f"> force update spec and mol df: {args.ow_spec_mol}")
    logger.info(f"> dsets {args.dsets}")
    logger.info(f"> canonicalize_tautomers: {args.canonicalize_tautomers}")

    spec_df_fp = os.path.join(args.proc_dp, "spec_df.pkl")
    mol_df_fp = os.path.join(args.proc_dp, "mol_df.pkl")
    ann_df_fp = os.path.join(args.proc_dp, "ann_df.pkl")

    if not args.ow_spec_mol and os.path.isfile(spec_df_fp) and os.path.isfile(mol_df_fp):
        logger.info("> loading previous spec_df, mol_df, ann_df")
        if not os.path.isdir(args.proc_dp):
            raise FileNotFoundError(f"Processed directory not found: {args.proc_dp}")
        if not os.path.isfile(spec_df_fp):
            raise FileNotFoundError(f"spec_df file not found: {spec_df_fp}")
        if not os.path.isfile(mol_df_fp):
            raise FileNotFoundError(f"mol_df file not found: {mol_df_fp}")
        spec_df = pd.read_pickle(spec_df_fp)
        mol_df = pd.read_pickle(mol_df_fp)
        ann_df = pd.read_pickle(ann_df_fp)

    else:
        logger.info("> creating new spec_df, mol_df, ann_df")
        if not os.path.isdir(args.df_dp):
            raise FileNotFoundError(f"Input dataframe directory not found: {args.df_dp}")
        os.makedirs(args.proc_dp, exist_ok=True)
        all_df = load_df(args.df_dp, args.dsets, args.num_entries)

        spec_df, mol_df, ann_df, debug_dfs = preprocess_spec(
            all_df,
            canonicalize_tautomers=args.canonicalize_tautomers,
            skip_standardize=args.skip_standardize,
        )

        if args.na_str_for_missing:
            str_cols = ["inst_type", "frag_mode", "spec_type", "ion_mode", "col_gas", "prec_type"]
            for col in str_cols:
                if col in spec_df:
                    spec_df.loc[spec_df[col].isna(), col] = "NA"
            logger.info("> converted missing str labels to NA")

        # save everything to file
        spec_df.to_pickle(spec_df_fp)
        mol_df.to_pickle(mol_df_fp)
        ann_df.to_pickle(ann_df_fp)
        logger.info(f"> saved spec_df to {spec_df_fp}")
        logger.info(f"> saved mol_df to {mol_df_fp}")

        for k, v in debug_dfs.items():
            v.to_pickle(os.path.join(args.proc_dp, f"{k}.pkl"))

    logger.info(f"> spec_df {spec_df.shape}")
    logger.info("> spec_df num nan:")
    logger.info(spec_df.isna().sum())
    logger.info(f"> mol_df {mol_df.shape}")
    logger.info("> mol_df num nan:")
    logger.info(mol_df.isna().sum())
    logger.info(f"> ann_df {ann_df.shape}")
    logger.info("> ann_df num nan:")
    logger.info(ann_df.isna().sum())
