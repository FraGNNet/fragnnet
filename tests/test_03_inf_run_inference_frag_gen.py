"""
Unit tests for 03_inf_run_inference_frag_gen.py.
"""

import importlib.util
from pathlib import Path


# Helper to import modules with numeric names
def _load_script_module(script_name):
    """Load a preprocessing script as a module."""
    script_path = Path(__file__).parent.parent / "preproc_scripts" / "inference" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_module():
    """Test that the module can be loaded."""
    module = _load_script_module("03_inf_run_inference_frag_gen.py")
    assert module is not None


def test_get_args_exists():
    """Test that get_args function exists."""
    module = _load_script_module("03_inf_run_inference_frag_gen.py")
    assert hasattr(module, "get_args")
