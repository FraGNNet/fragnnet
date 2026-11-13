"""Prediction-only runs must not be cleaned against ground truth they do not have.

`spec_params.drop_empty_spectra` defaults to True in `config/template.yml`, and
`BaseDataset._setup_dfs` runs the whole spectrum cleanup group whenever the config
carries it. An inference `spec_df` from `02_inf_prepare_spec_df.py` has `peaks == []`
by construction, so the cleanup dropped every spectrum: the dataset came back empty and
`run_inference.py` reported "Total predictions: 0" while exiting 0.

`apply_inference_config(drop_empty_spectra=False)` is the opt-out. It stays opt-in
because the same helper configures evaluation runs, which score against real spectra
and must keep their cleanup behavior.
"""

import pandas as pd
import pytest

from fragnnet.runner import load_config
from fragnnet.utils.clean_spectra_utils import clean_spectra
from fragnnet.utils.script_utils import apply_inference_config

CLEANUP_PARAMS = {
    "centroid_peaks",
    "centroid_ppm",
    "centroid_use_instrument_tolerance",
    "centroid_method",
    "filter_subformula_peaks",
    "subformula_ppm",
    "subformula_use_instrument_tolerance",
    "drop_empty_spectra",
}


@pytest.fixture
def config():
    return load_config("config/template.yml", None, None)


@pytest.fixture
def peakless_dfs():
    """Two spectra of one molecule, shaped like `02_inf_prepare_spec_df.py` output."""
    spec_df = pd.DataFrame(
        {
            "spec_id": [0, 1],
            "mol_id": [0, 0],
            "prec_type": ["[M+H]+", "[M-H]-"],
            "inst_type": ["FT", "FT"],
            "prec_mz": [181.0495, 179.0350],
            "peaks": [[], []],
        }
    )
    mol_df = pd.DataFrame({"mol_id": [0], "smiles": ["CC(=O)Oc1ccccc1C(=O)O"]})
    return spec_df, mol_df


def test_template_default_would_drop_every_inference_spectrum(config, peakless_dfs):
    """The behavior being opted out of: peakless input survives nothing."""
    spec_df, mol_df = peakless_dfs
    assert config["spec_params"]["drop_empty_spectra"] is True
    cleaned = clean_spectra(
        spec_df=spec_df,
        mol_df=mol_df,
        merge_peak_signals=False,
        merge_ppm=20.0,
        filter_by_subformula=False,
        subformula_ppm=5.0,
        drop_empty_spectra=True,
        use_instrument_tolerance=True,
        use_merge_instrument_tolerance=False,
        merge_method="common_overlap",
    )
    assert len(cleaned) == 0
    kept = clean_spectra(
        spec_df=spec_df,
        mol_df=mol_df,
        merge_peak_signals=False,
        merge_ppm=20.0,
        filter_by_subformula=False,
        subformula_ppm=5.0,
        drop_empty_spectra=False,
        use_instrument_tolerance=True,
        use_merge_instrument_tolerance=False,
        merge_method="common_overlap",
    )
    assert len(kept) == 2


def test_false_disables_the_drop(config):
    apply_inference_config(config, drop_empty_spectra=False)
    assert config["spec_params"]["drop_empty_spectra"] is False


def test_omitting_it_leaves_evaluation_behavior_alone(config):
    before = config["spec_params"]["drop_empty_spectra"]
    apply_inference_config(config)
    assert config["spec_params"]["drop_empty_spectra"] == before


def test_legacy_config_without_the_cleanup_group_is_not_given_one(config):
    """Writing the key alone would trip the missing-cleanup-params check in the dataset."""
    for key in CLEANUP_PARAMS:
        config["spec_params"].pop(key)
    apply_inference_config(config, drop_empty_spectra=False)
    assert CLEANUP_PARAMS.isdisjoint(config["spec_params"])
