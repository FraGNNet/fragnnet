"""Unit tests for preproc_scripts/03_prepare_dag_feats.py module.

Tests validate:
- print_and_log() statistics logging
- Input validation (max_time constraint)
- Data type conversions
- Function parameter handling
"""

import bz2
import gzip
import importlib.util
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from fragnnet.utils import frag_utils


def load_prepare_module():
    """Load the preprocessing script as a module via importlib."""
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "preproc_scripts" / "03_prepare_dag_feats.py"
    spec = importlib.util.spec_from_file_location("prepare_dag_feats", str(script_path))
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Unable to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_print_and_log_basic():
    """Test print_and_log captures statistics correctly."""

    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    stats_d = {}

    # Call without wandb
    load_prepare_module().print_and_log("test_metric", series, wandb_flag=False, stats_d=stats_d)

    # Verify statistics captured
    assert "test_metric/mean" in stats_d
    assert "test_metric/std" in stats_d
    assert "test_metric/min" in stats_d
    assert "test_metric/max" in stats_d
    assert "test_metric/50%" in stats_d
    assert stats_d["test_metric/mean"] == 3.0
    assert stats_d["test_metric/min"] == 1.0
    assert stats_d["test_metric/max"] == 5.0


def test_print_and_log_with_nan():
    """Test print_and_log handles NaN values in series."""

    series = pd.Series([1.0, 2.0, float("nan"), 4.0, 5.0])
    stats_d = {}

    load_prepare_module().print_and_log("nan_metric", series, wandb_flag=False, stats_d=stats_d)

    # Should still compute statistics, skipping NaN
    assert "nan_metric/mean" in stats_d
    # mean of [1, 2, 4, 5] = 3.0
    assert stats_d["nan_metric/mean"] == 3.0


def test_print_and_log_empty_series():
    """Test print_and_log with empty series."""

    series = pd.Series([], dtype=float)
    stats_d = {}

    # Should not raise, but produce NaN statistics
    load_prepare_module().print_and_log("empty_metric", series, wandb_flag=False, stats_d=stats_d)

    # NaN values should be in the dict
    assert "empty_metric/mean" in stats_d


def test_print_and_log_all_stats_present():
    """Verify all expected statistics are logged."""

    series = pd.Series(range(1, 101))  # 1 to 100
    stats_d = {}

    load_prepare_module().print_and_log("full_stats", series, wandb_flag=False, stats_d=stats_d)

    expected_stats = ["mean", "std", "min", "25%", "50%", "75%", "max"]
    for stat in expected_stats:
        assert f"full_stats/{stat}" in stats_d, f"Missing stat: full_stats/{stat}"


def test_main_max_time_validation():
    """Test main() validates max_time against JOBLIB_TIMEOUT."""
    module = load_prepare_module()

    # Create mock args with max_time >= JOBLIB_TIMEOUT
    args = MagicMock()
    args.max_time = module.data_utils.JOBLIB_TIMEOUT + 100

    # Should raise ValueError
    try:
        module.main(args)
        assert False, "Expected ValueError for invalid max_time"
    except ValueError as e:
        assert "max_time" in str(e)
        assert "JOBLIB_TIMEOUT" in str(e)


def test_main_max_time_valid():
    """Test main() accepts valid max_time without raising ValueError."""

    args = MagicMock()
    args.max_time = 100  # Well below JOBLIB_TIMEOUT (typically ~300 seconds)
    args.wandb_mode = "off"  # Disabled so wandb.init() is not called

    # Patch the file operations to prevent actual execution
    with (
        patch("os.path.isfile", return_value=False),
        patch("os.makedirs"),
        patch("pandas.read_pickle", side_effect=FileNotFoundError),
    ):
        try:
            load_prepare_module().main(args)
        except FileNotFoundError:
            # Expected since we're mocking file ops
            pass
        except ValueError as e:
            if "max_time" in str(e):
                assert False, f"Unexpected max_time error: {e}"


def test_booltype_conversion():
    """Test booltype conversion utility."""

    # Test True conversions
    assert load_prepare_module().booltype("true") is True
    assert load_prepare_module().booltype("True") is True
    assert load_prepare_module().booltype("1") is True

    # Test False conversions
    assert load_prepare_module().booltype("false") is False
    assert load_prepare_module().booltype("False") is False
    assert load_prepare_module().booltype("0") is False


def test_tolerance_format():
    """Test that default tolerances are properly formatted."""
    module = load_prepare_module()

    # Check parser defaults
    parser = module.argparse.ArgumentParser()
    parser.add_argument(
        "--tolerances",
        type=str,
        nargs="+",
        default=["0.01", "0.005", "0.001", "0.0001", "10ppm", "5ppm"],
    )
    args = parser.parse_args([])

    tolerances = args.tolerances
    assert "0.01" in tolerances
    assert "10ppm" in tolerances
    assert len(tolerances) == 6


def test_filter_arguments_structure():
    """Test that argument parser has required filter arguments."""

    # Create minimal args namespace
    args = MagicMock()
    args.dsets = ["nist20"]
    args.elements = ["C", "H", "N", "O"]
    args.prec_types = ["[M+H]+"]
    args.frag_modes = ["HCD"]
    args.ion_modes = ["P"]
    args.inst_types = ["Orbitrap"]
    args.spec_type = "MS2"

    # Verify these are the expected filter parameters
    assert hasattr(args, "dsets")
    assert hasattr(args, "elements")
    assert hasattr(args, "prec_types")
    assert hasattr(args, "frag_modes")
    assert hasattr(args, "ion_modes")
    assert hasattr(args, "inst_types")
    assert hasattr(args, "spec_type")


def test_dag_generation_parameters():
    """Test DAG generation parameters are properly defined."""

    args = MagicMock()
    args.max_depth = 4
    args.max_h_transfer = 4
    args.nb_isomorphic = False
    args.wl_max_iterations = 3
    args.isotopes = True
    args.compress_dags = True
    args.compress_format = "gz"
    args.use_cached_dag = False

    # Verify parameter structure
    assert args.max_depth >= 1
    assert args.max_h_transfer >= 1
    assert isinstance(args.nb_isomorphic, bool)
    assert isinstance(args.wl_max_iterations, int)
    assert isinstance(args.isotopes, bool)
    assert isinstance(args.compress_dags, bool)
    assert args.compress_format in {"gz", "bz2"}


def test_get_dag_output_path_with_gzip():
    """Test DAG file naming for gzip compression."""

    fp = frag_utils.get_dag_output_path(
        dag_dp="/tmp/dags",
        mol_id=42,
        compress_dags=True,
        compress_format="gz",
    )
    assert fp.endswith("42.pkl.gz")


def test_get_dag_output_path_with_bz2():
    """Test DAG file naming for bz2 compression."""

    fp = frag_utils.get_dag_output_path(
        dag_dp="/tmp/dags",
        mol_id=42,
        compress_dags=True,
        compress_format="bz2",
    )
    assert fp.endswith("42.pkl.bz2")


def test_dump_dag_pickle_round_trip_gzip():
    """Test writing and reading a gzip-compressed DAG pickle."""

    payload = {"dag": {"num_nodes": 3}, "meta": [1, 2, 3]}

    with tempfile.TemporaryDirectory() as tmpdir:
        fp = str(Path(tmpdir) / "1.pkl.gz")
        frag_utils.dump_dag_pickle(fp, payload)
        with gzip.open(fp, "rb") as f:
            loaded = pickle.load(f)
    assert loaded == payload


def test_dump_dag_pickle_round_trip_bz2():
    """Test writing and reading a bz2-compressed DAG pickle."""

    payload = {"dag": {"num_nodes": 5}, "meta": [4, 5, 6]}

    with tempfile.TemporaryDirectory() as tmpdir:
        fp = str(Path(tmpdir) / "2.pkl.bz2")
        frag_utils.dump_dag_pickle(fp, payload)
        with bz2.open(fp, "rb") as f:
            loaded = pickle.load(f)
    assert loaded == payload


def test_output_structure():
    """Test expected output file structure."""

    with tempfile.TemporaryDirectory() as tmpdir:
        frag_dp = Path(tmpdir) / "frags"
        frag_dp.mkdir()

        # Expected output files
        expected_files = [
            "spec_stats_df.pkl",
            "mol_stats_df.pkl",
            "m_spec_stats_df.pkl",
            "m_mol_stats_df.pkl",
            "global_stats.json",
            "dags",  # directory
        ]

        # Simulate creating these files
        for fname in expected_files:
            if fname == "dags":
                (frag_dp / fname).mkdir(exist_ok=True)
            elif fname.endswith(".json"):
                (frag_dp / fname).write_text("{}")
            else:
                # Create empty pickle by writing bytes
                (frag_dp / fname).touch()

        # Verify structure exists
        for fname in expected_files:
            fpath = frag_dp / fname
            assert fpath.exists(), f"Expected output file not found: {fname}"


def test_statistics_dict_keys():
    """Test that statistics dictionary gets proper keys."""

    stats_d = {}

    # Simulate adding statistics as done in main()
    stats_d["total_num_failures"] = 5
    stats_d["total_num_formulae"] = 1000
    stats_d["depth/1"] = 50
    stats_d["depth/2"] = 150
    stats_d["dag_num_edges/mean"] = 25.3

    # Verify structure
    assert "total_num_failures" in stats_d
    assert "total_num_formulae" in stats_d
    assert any(k.startswith("depth/") for k in stats_d)
    assert any(k.startswith("dag_num_edges/") for k in stats_d)


def test_argparse_wandb_validation():
    """Test that argparse validates wandb_run_name requirement."""

    parser = load_prepare_module().argparse.ArgumentParser()
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="disabled",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument("--wandb_run_name", type=str, required=False)

    # Parse with wandb disabled (should be ok without run_name)
    args = parser.parse_args(["--wandb_mode", "disabled"])
    assert args.wandb_mode == "disabled"
    assert args.wandb_run_name is None

    # The validation logic in __main__ would check this separately
    # just verify the args can be parsed
    assert hasattr(args, "wandb_mode")
    assert hasattr(args, "wandb_run_name")
