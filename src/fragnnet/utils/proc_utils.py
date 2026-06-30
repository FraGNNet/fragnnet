import numpy as np

from fragnnet.frag.compute_frags import MAX_NUM_EDGES, MAX_NUM_NODES
from fragnnet.utils.formula_utils import parse_formula
from fragnnet.utils.spec_utils import merge_sparse_specs


def element_filter(formula, elements):
    try:
        element_counts = parse_formula(formula)
    except ValueError:
        return False
    for k, v in element_counts.items():
        if v > 0 and k not in elements:
            return False
    return True


def filter_spec_mol(
    spec_df,
    mol_df,
    elements=None,
    dsets=None,
    max_peak_mz=None,
    max_prec_mz=None,
    min_prec_mz=None,
    prec_types: list | None = None,
    num_entries=-1,
    inst_types: list | None = None,
    frag_modes: list | None = None,
    ion_modes: list | None = None,
    ces: str | None = None,
    spec_type: str | None = None,
    max_heavy_atom: int | None = None,
    max_bond: int | None = None,
    ce_types: list | None = None,
):
    """
    The purpose of this filtering is to prevent errors
    """
    masks = []

    ## spectrum criteria (TODO remove this)
    # dataset mask
    if dsets is not None:
        print(f">> dsets filter: {dsets}")
        print(">> dsets in data:", spec_df["dset"].unique())
        dset_mask = spec_df["dset"].isin(dsets)
        print(f">> dsets filter mean: {dset_mask.mean()}")
        masks.append(dset_mask)
    else:
        print(">> no dset filter")
        print(">> dsets in data:", spec_df["dset"].unique())

    # instrument type
    if inst_types is not None:
        print(f">> inst type: {inst_types}")
        print(">> inst type in data:", spec_df["inst_type"].unique())
        inst_type_mask = spec_df["inst_type"].isin(inst_types)
        print(f">> inst type filter mean: {inst_type_mask.mean()}")
        masks.append(inst_type_mask)
    else:
        print(">> no inst type filter")
        print(">> inst type in data:", spec_df["inst_type"].unique())

    # frag mode
    if frag_modes is not None:
        print(f">> frag mode filter: {frag_modes}")
        print(">> frag mode in data:", spec_df["frag_mode"].unique())
        frag_mode_mask = spec_df["frag_mode"].isin(frag_modes)
        print(f">> frag mode filter mean: {frag_mode_mask.mean()}")
        masks.append(frag_mode_mask)
    else:
        print(">> no frag mode filter")
        if "frag_mode" in spec_df:
            print(">> frag mode in data:", spec_df["frag_mode"].unique())

    # ce filter
    assert ces in ["ace", "nce", "nce_or_ace", None], ces
    if ces == "ace":
        print(">> require ace")
        ace_mask = ~spec_df["ace"].isna()
        print(f">> ace filter: {ace_mask.mean()}")
        masks.append(ace_mask)
    elif ces == "nce":
        print(">> require nce")
        nce_mask = ~spec_df["nce"].isna()
        print(f">> nce filter: {nce_mask.mean()}")
        masks.append(nce_mask)
    elif ces == "nce_or_ace":
        print(">> require ace or nce")
        mask = ~spec_df["nce"].isna() | ~spec_df["ace"].isna()
        print(f">> ace_or_nce filter: {mask.mean()}")
        masks.append(mask)
    else:
        print(f">> ces filter set to {ces},  no ce filter applied")

    # ce_types
    # print(ce_types)
    ce_types = ce_types if ce_types is not None else []
    if ce_types:
        print(f">> ce_types filter: {ce_types}")
        allowed = {"stepped", "ramped", "single", "none"}
        unknown = set(ce_types) - allowed
        if unknown:
            raise ValueError(f"unknown ce_types requested: {sorted(unknown)}")
        ce_mask = spec_df["ce_type"].isin(ce_types)
        print(f">> ce_type filter mean: {ce_mask.mean()}")
        masks.append(ce_mask)
    else:
        print(">> no ce_type filter")

    # ion mode
    if ion_modes is not None:
        print(f">> ion mode filter: {ion_modes}")
        print(">> ion mode in data:", spec_df["ion_mode"].unique())
        ion_mode_mask = spec_df["ion_mode"].isin(ion_modes)
        print(f">> ion mode filter mean: {ion_mode_mask.mean()}")
        masks.append(ion_mode_mask)
    else:
        print(">> no ion mode filter")
        print(">> ion mode in data:", spec_df["ion_mode"].unique())

    # precursor type
    if prec_types is not None:
        print(f">> prec type filter: {prec_types}")
        print(">> prec type in data: ", spec_df["prec_type"].unique())
        prec_type_mask = spec_df["prec_type"].isin(prec_types)
        print(f">> prec type filter mean: {prec_type_mask.mean()}")
        masks.append(prec_type_mask)
    else:
        print(">> no prec type filter")
        print(">> prec type in data: ", spec_df["prec_type"].unique())

    # resolution
    # res_mask = spec_df["res"].isin([1,2,3,4,5,6,7])
    # print(f">> res: {res_mask.mean()}")
    # masks.append(res_mask)

    # spectrum type
    if spec_type is not None:
        print(f">> spec type filter: {spec_type}")
        print(">> spec type in data: ", spec_df["spec_type"].unique())
        spec_type_mask = spec_df["spec_type"] == spec_type
        print(f">> spec type: {spec_type_mask.mean()}")
        masks.append(spec_type_mask)
    else:
        print(">> no spec type filter")
        print(">> spec type in data: ", spec_df["spec_type"].unique())

    # precursor mz
    if "prec_mz" in spec_df:
        prec_mz_mask = ~spec_df["prec_mz"].isna()
        print(f">> prec mz: {prec_mz_mask.mean()}")
        masks.append(prec_mz_mask)

    # max prec mz
    if max_prec_mz is not None:
        print(f">> max_prec_mz {max_prec_mz}")
        max_prec_mz_mask = spec_df["prec_mz"] <= max_prec_mz
        print(f">> max prec mz: {max_prec_mz_mask.mean()}")
        masks.append(max_prec_mz_mask)

    # min prec mz
    if min_prec_mz is not None:
        print(f">> min_prec_mz {min_prec_mz}")
        min_prec_mz_mask = spec_df["prec_mz"] >= min_prec_mz
        print(f">> min prec mz: {min_prec_mz_mask.mean()}")
        masks.append(min_prec_mz_mask)

    # max peak mz
    if max_peak_mz is not None:

        def get_max_mz(peaks):
            return max([peak[0] for peak in peaks])

        max_peak_mz_mask = spec_df["peaks"].apply(get_max_mz) <= max_peak_mz
        print(f">> max peak mz: {max_peak_mz_mask.mean()}")
        masks.append(max_peak_mz_mask)

    ## molecule criteria
    # single molecule
    single_mol_ids = mol_df[mol_df["single_mol"]]["mol_id"]
    single_mol_mask = spec_df["mol_id"].isin(single_mol_ids)
    print(f">> single mol: {single_mol_mask.mean()}")
    masks.append(single_mol_mask)
    # neutral or pre-formed +1 cation ([M]+ support: quaternary ammonium, sulfonium, etc.)
    neutral_ids = mol_df[mol_df["charge"].isin([0, 1])]["mol_id"]
    neutral_mask = spec_df["mol_id"].isin(neutral_ids)
    print(f">> neutral or [M]+: {neutral_mask.mean()}")
    masks.append(neutral_mask)
    # Cross-check: charge=+1 molecules are only valid with [M]+ spectra.
    # A charge=+1 molecule paired with [M+H]+ would apply +H to already-charged
    # fragment formulas, producing wrong m/z values.
    charged_mol_ids = set(mol_df[mol_df["charge"] == 1]["mol_id"].tolist())
    if charged_mol_ids:
        bad_adduct_mask = spec_df["mol_id"].isin(charged_mol_ids) & (
            spec_df["prec_type"] != "[M]+"
        )
        n_bad = int(bad_adduct_mask.sum())
        if n_bad > 0:
            print(
                f">> charge cross-check: dropping {n_bad} spectra of charge=+1 molecules "
                f"with prec_type != '[M]+'"
            )
        masks.append(~bad_adduct_mask)
    # element composition
    if elements is not None:
        element_ids = mol_df[
            mol_df["formula"].apply(lambda formula: element_filter(formula, elements))
        ]["mol_id"]
        element_mask = spec_df["mol_id"].isin(element_ids)
        print(f">> element: {element_mask.mean()}")
        masks.append(element_mask)

    # atom count
    max_heavy_atom = max_heavy_atom if max_heavy_atom is not None else MAX_NUM_NODES
    print(f">> max atoms allowed: {max_heavy_atom}")
    atom_count_ids = mol_df[mol_df["num_atoms"] <= max_heavy_atom]["mol_id"]
    atom_count_mask = spec_df["mol_id"].isin(atom_count_ids)
    print(f">> atom count: {atom_count_mask.mean()}")
    masks.append(atom_count_mask)

    max_bond = max_bond if max_bond is not None else MAX_NUM_EDGES
    print(f">> max bonds allowed: {max_bond}")
    # bond count
    bond_count_ids = mol_df[mol_df["num_bonds"] <= max_bond]["mol_id"]
    bond_count_mask = spec_df["mol_id"].isin(bond_count_ids)
    print(f">> bond count: {bond_count_mask.mean()}")
    masks.append(bond_count_mask)

    # radical stats
    non_radical_ids = mol_df[mol_df["num_radicals"] == 0]["mol_id"]
    non_radical_mask = spec_df["mol_id"].isin(non_radical_ids)
    print(f">> non radical: {non_radical_mask.mean()}")
    masks.append(non_radical_mask)
    # put them together
    all_mask = masks[0]
    for mask in masks:
        all_mask = all_mask & mask
    if np.sum(all_mask) == 0:
        raise ValueError("select removed all items")
    print(f">> everything: {all_mask.mean()}")
    spec_df = spec_df[all_mask].reset_index(drop=True)
    # sample
    if num_entries > 0:
        spec_df = spec_df.sample(n=num_entries, replace=False, random_state=420).reset_index(
            drop=True
        )
    # only keep molecules that are in the spectra
    mol_df = mol_df[mol_df["mol_id"].isin(spec_df["mol_id"])].reset_index(drop=True)
    return spec_df, mol_df


def merge_spec_df(spec_df, renormalize=False, sum_ints=True, keep_ces=False):
    # merge the peaks (they are not normalized here)
    m_peaks_func = lambda peakses: merge_sparse_specs(
        *peakses, renormalize=renormalize, sum_ints=sum_ints
    )
    m_spec_peaks_df = (
        spec_df[["group_id", "peaks"]]
        .groupby("group_id")
        .agg({"peaks": m_peaks_func})
        .reset_index()
    )
    m_spec_meta_df = spec_df.drop(columns=["peaks", "spec_id"]).drop_duplicates(subset=["group_id"])
    m_spec_df = m_spec_peaks_df.merge(m_spec_meta_df, on=["group_id"], how="inner")
    if keep_ces:
        m_spec_ce_df = (
            spec_df[["group_id", "nce", "ace"]]
            .groupby("group_id")
            .agg({"nce": list, "ace": list})
            .reset_index()
        )
        m_spec_df = m_spec_df.drop(columns=["nce", "ace"])
        m_spec_df = m_spec_df.merge(m_spec_ce_df, on=["group_id"], how="inner")
    else:
        m_spec_df.loc[:, "nce"] = np.nan
        m_spec_df.loc[:, "ace"] = np.nan
    return m_spec_df
