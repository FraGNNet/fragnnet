import numpy as np
import pandas as pd
import pytest

from fragnnet.utils.proc_utils import element_filter, filter_spec_mol, merge_spec_df


def make_spec_and_mol_frames():
    spec_df = pd.DataFrame(
        [
            {
                "dset": "nist",
                "inst_type": "orbitrap",
                "frag_mode": "hcd",
                "ce_type": "stepped",
                "nce": 20.0,
                "ace": np.nan,
                "ion_mode": "positive",
                "prec_type": "[M+H]+",
                "spec_type": "centroid",
                "prec_mz": 150.0,
                "peaks": [(10.0, 0.5), (20.0, 0.5)],
                "mol_id": 1,
                "spec_id": 101,
                "group_id": 1,
            },
            {
                "dset": "nist",
                "inst_type": "orbitrap",
                "frag_mode": "hcd",
                "ce_type": "single",
                "nce": np.nan,
                "ace": np.nan,
                "ion_mode": "positive",
                "prec_type": "[M+H]+",
                "spec_type": "centroid",
                "prec_mz": 220.0,
                "peaks": [(10.0, 1.0)],
                "mol_id": 2,
                "spec_id": 102,
                "group_id": 1,
            },
            {
                "dset": "nist",
                "inst_type": "orbitrap",
                "frag_mode": "cid",
                "ce_type": "single",
                "nce": 25.0,
                "ace": np.nan,
                "ion_mode": "positive",
                "prec_type": "[M+Na]+",
                "spec_type": "centroid",
                "prec_mz": 120.0,
                "peaks": [(15.0, 0.7)],
                "mol_id": 3,
                "spec_id": 103,
                "group_id": 2,
            },
        ]
    )
    mol_df = pd.DataFrame(
        [
            {
                "mol_id": 1,
                "single_mol": True,
                "charge": 0,
                "formula": "C2H6O",
                "num_atoms": 3,
                "num_bonds": 2,
                "num_radicals": 0,
            },
            {
                "mol_id": 2,
                "single_mol": True,
                "charge": 0,
                "formula": "C6H6",
                "num_atoms": 6,
                "num_bonds": 6,
                "num_radicals": 0,
            },
            {
                "mol_id": 3,
                "single_mol": True,
                "charge": 0,
                "formula": "C6H5Cl",
                "num_atoms": 7,
                "num_bonds": 6,
                "num_radicals": 0,
            },
        ]
    )
    return spec_df, mol_df


def test_element_filter_behaviors():
    assert element_filter("C6H6", {"C", "H"}) is True
    assert element_filter("CCl4", {"C", "H"}) is False
    assert element_filter("not_a_formula", {"C", "H"}) is False


def test_filter_spec_mol_filters_by_nce_and_ce_type():
    spec_df, mol_df = make_spec_and_mol_frames()

    filtered_spec, filtered_mol = filter_spec_mol(
        spec_df,
        mol_df,
        frag_modes=["hcd"],
        inst_types=["orbitrap"],
        ces="nce",
        ce_types=["stepped"],
        max_prec_mz=200.0,
    )

    assert len(filtered_spec) == 1
    assert filtered_spec.loc[0, "mol_id"] == 1
    assert len(filtered_mol) == 1
    assert filtered_mol.loc[0, "mol_id"] == 1


def test_filter_spec_mol_raises_on_unknown_ce_type():
    spec_df, mol_df = make_spec_and_mol_frames()

    with pytest.raises(ValueError):
        filter_spec_mol(spec_df, mol_df, ce_types=["invalid"])


def test_filter_spec_mol_raises_when_no_entries_remain():
    spec_df, mol_df = make_spec_and_mol_frames()

    with pytest.raises(ValueError):
        filter_spec_mol(spec_df, mol_df, dsets=["missing_dataset"])


def test_filter_spec_mol_keeps_charge_1_molecules():
    """Molecules with charge=1 (pre-formed cations) must pass the neutral filter for [M]+."""
    spec_df, mol_df = make_spec_and_mol_frames()
    # Append a charge=+1 molecule and its [M]+ spectrum
    cation_mol = pd.DataFrame(
        [
            {
                "mol_id": 4,
                "single_mol": True,
                "charge": 1,
                "formula": "C4H12N",
                "num_atoms": 5,
                "num_bonds": 4,
                "num_radicals": 0,
            }
        ]
    )
    cation_spec = pd.DataFrame(
        [
            {
                "dset": "nist",
                "inst_type": "orbitrap",
                "frag_mode": "ei",
                "ce_type": "single",
                "nce": np.nan,
                "ace": np.nan,
                "ion_mode": "positive",
                "prec_type": "[M]+",
                "spec_type": "centroid",
                "prec_mz": 74.0,
                "peaks": [(58.0, 1.0)],
                "mol_id": 4,
                "spec_id": 104,
                "group_id": 3,
            }
        ]
    )
    spec_df = pd.concat([spec_df, cation_spec], ignore_index=True)
    mol_df = pd.concat([mol_df, cation_mol], ignore_index=True)

    filtered_spec, filtered_mol = filter_spec_mol(spec_df, mol_df, prec_types=["[M]+"])
    assert 4 in filtered_mol["mol_id"].values, "charge=1 molecule should survive filter"
    assert 104 in filtered_spec["spec_id"].values, "[M]+ spectrum should survive filter"


def test_filter_spec_mol_drops_charge_2_molecules():
    """Molecules with charge=2 must still be filtered out."""
    spec_df, mol_df = make_spec_and_mol_frames()
    dication_mol = pd.DataFrame(
        [
            {
                "mol_id": 5,
                "single_mol": True,
                "charge": 2,
                "formula": "C4H12N2",
                "num_atoms": 6,
                "num_bonds": 5,
                "num_radicals": 0,
            }
        ]
    )
    dication_spec = pd.DataFrame(
        [
            {
                "dset": "nist",
                "inst_type": "orbitrap",
                "frag_mode": "hcd",
                "ce_type": "single",
                "nce": 30.0,
                "ace": np.nan,
                "ion_mode": "positive",
                "prec_type": "[M+H]+",
                "spec_type": "centroid",
                "prec_mz": 74.0,
                "peaks": [(58.0, 1.0)],
                "mol_id": 5,
                "spec_id": 105,
                "group_id": 4,
            }
        ]
    )
    spec_df = pd.concat([spec_df, dication_spec], ignore_index=True)
    mol_df = pd.concat([mol_df, dication_mol], ignore_index=True)

    filtered_spec, filtered_mol = filter_spec_mol(spec_df, mol_df)
    assert 5 not in filtered_mol["mol_id"].values, "charge=2 molecule should be filtered out"
    assert 105 not in filtered_spec["spec_id"].values, "dication spectrum should be filtered out"


def test_filter_spec_mol_cross_check_drops_wrong_adduct_for_charged_mol():
    """charge=+1 molecule paired with [M+H]+ spectrum must be dropped by the cross-check."""
    spec_df, mol_df = make_spec_and_mol_frames()
    cation_mol = pd.DataFrame(
        [
            {
                "mol_id": 6,
                "single_mol": True,
                "charge": 1,
                "formula": "C4H12N",
                "num_atoms": 5,
                "num_bonds": 4,
                "num_radicals": 0,
            }
        ]
    )
    # Two spectra for the same charge=+1 molecule: one [M]+ (valid) and one [M+H]+ (invalid)
    mixed_specs = pd.DataFrame(
        [
            {
                "dset": "nist",
                "inst_type": "orbitrap",
                "frag_mode": "ei",
                "ce_type": "single",
                "nce": np.nan,
                "ace": np.nan,
                "ion_mode": "positive",
                "prec_type": "[M]+",
                "spec_type": "centroid",
                "prec_mz": 74.0,
                "peaks": [(58.0, 1.0)],
                "mol_id": 6,
                "spec_id": 106,
                "group_id": 5,
            },
            {
                "dset": "nist",
                "inst_type": "orbitrap",
                "frag_mode": "hcd",
                "ce_type": "single",
                "nce": 30.0,
                "ace": np.nan,
                "ion_mode": "positive",
                "prec_type": "[M+H]+",
                "spec_type": "centroid",
                "prec_mz": 75.0,
                "peaks": [(58.0, 1.0)],
                "mol_id": 6,
                "spec_id": 107,
                "group_id": 5,
            },
        ]
    )
    spec_df = pd.concat([spec_df, mixed_specs], ignore_index=True)
    mol_df = pd.concat([mol_df, cation_mol], ignore_index=True)

    filtered_spec, filtered_mol = filter_spec_mol(spec_df, mol_df)
    # [M]+ spectrum of the cation should survive
    assert 106 in filtered_spec["spec_id"].values, "[M]+ spectrum of cation should be kept"
    # [M+H]+ spectrum of the cation must be dropped
    assert 107 not in filtered_spec["spec_id"].values, (
        "[M+H]+ spectrum of charge=+1 molecule should be dropped by cross-check"
    )
    # the molecule itself survives (it still has a valid [M]+ spectrum)
    assert 6 in filtered_mol["mol_id"].values, "cation molecule should survive when [M]+ exists"


def test_filter_spec_mol_sampling_is_deterministic():
    spec_df, mol_df = make_spec_and_mol_frames()
    sampled_first, _ = filter_spec_mol(spec_df, mol_df, num_entries=1)

    spec_df, mol_df = make_spec_and_mol_frames()
    sampled_second, _ = filter_spec_mol(spec_df, mol_df, num_entries=1)

    assert sampled_first.iloc[0]["spec_id"] == sampled_second.iloc[0]["spec_id"]


def test_merge_spec_df_merges_peaks_and_handles_ces():
    spec_df = pd.DataFrame(
        [
            {
                "group_id": 1,
                "peaks": [(10.0, 1.0), (20.0, 2.0)],
                "spec_id": 501,
                "dset": "nist",
                "inst_type": "orbitrap",
                "frag_mode": "hcd",
                "ce_type": "stepped",
                "nce": 10.0,
                "ace": np.nan,
                "ion_mode": "positive",
                "prec_type": "[M+H]+",
                "spec_type": "centroid",
                "prec_mz": 150.0,
                "mol_id": 1,
            },
            {
                "group_id": 1,
                "peaks": [(10.0, 1.0), (30.0, 1.0)],
                "spec_id": 502,
                "dset": "nist",
                "inst_type": "orbitrap",
                "frag_mode": "hcd",
                "ce_type": "stepped",
                "nce": 20.0,
                "ace": 5.0,
                "ion_mode": "positive",
                "prec_type": "[M+H]+",
                "spec_type": "centroid",
                "prec_mz": 150.0,
                "mol_id": 1,
            },
        ]
    )

    merged_with_ces = merge_spec_df(spec_df, renormalize=True, sum_ints=True, keep_ces=True)
    peaks = merged_with_ces.loc[0, "peaks"]

    assert len(merged_with_ces) == 1
    assert peaks == [(10.0, 0.4), (20.0, 0.4), (30.0, 0.2)]

    nce_list = merged_with_ces.loc[0, "nce"]
    ace_list = merged_with_ces.loc[0, "ace"]
    assert nce_list == [10.0, 20.0]
    assert len(ace_list) == 2
    assert np.isnan(ace_list[0])
    assert ace_list[1] == 5.0

    merged_no_ces = merge_spec_df(spec_df, renormalize=False, sum_ints=False, keep_ces=False)
    assert len(merged_no_ces) == 1
    assert merged_no_ces.loc[0, "peaks"] == [(10.0, 1.0), (20.0, 2.0), (30.0, 1.0)]
    assert np.isnan(merged_no_ces.loc[0, "nce"])
    assert np.isnan(merged_no_ces.loc[0, "ace"])
