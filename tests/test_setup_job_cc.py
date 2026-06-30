import argparse
import importlib.util
from pathlib import Path

import pytest


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    fp = repo_root / "scripts" / "setup_job_cc.py"
    spec = importlib.util.spec_from_file_location("setup_job_cc", str(fp))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return load_module()


def make_args(**kwargs) -> argparse.Namespace:
    """Return a Namespace with sensible defaults, overridden by kwargs."""
    defaults = {
        "gpu_profile": None,
        "gpu": 1,
        "cpu": 8,
        "mem": 128,
        "gres_gpu": None,
        "gpus_per_node": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# _normalize_profile_name
# ---------------------------------------------------------------------------


class TestNormalizeProfileName:
    def test_lowercases(self, mod):
        assert mod._normalize_profile_name("H100-80GB") == "h100-80gb"

    def test_strips_leading_trailing_whitespace(self, mod):
        assert mod._normalize_profile_name("  a100-40gb  ") == "a100-40gb"

    def test_removes_internal_spaces(self, mod):
        # e.g. user types "fir: h100-80gb"
        assert mod._normalize_profile_name("fir: h100-80gb") == "fir:h100-80gb"

    def test_already_normalized_unchanged(self, mod):
        assert mod._normalize_profile_name("narval:a100-40gb") == "narval:a100-40gb"

    def test_empty_string(self, mod):
        assert mod._normalize_profile_name("") == ""


# ---------------------------------------------------------------------------
# GPU_PROFILES structure
# ---------------------------------------------------------------------------

REQUIRED_PROFILE_KEYS = {"cluster", "gpu_label", "gres_gpu", "cpu", "mem", "price", "notes"}


class TestGpuProfilesStructure:
    def test_all_profiles_have_required_keys(self, mod):
        for name, profile in mod.GPU_PROFILES.items():
            missing = REQUIRED_PROFILE_KEYS - profile.keys()
            assert not missing, f"Profile '{name}' missing keys: {missing}"

    def test_cpu_and_mem_are_positive(self, mod):
        for name, profile in mod.GPU_PROFILES.items():
            assert profile["cpu"] > 0, f"Profile '{name}' has non-positive cpu"
            assert profile["mem"] > 0, f"Profile '{name}' has non-positive mem"

    def test_gres_starts_with_gpu(self, mod):
        for name, profile in mod.GPU_PROFILES.items():
            assert profile["gres_gpu"].startswith("gpu"), (
                f"Profile '{name}' gres_gpu does not start with 'gpu': {profile['gres_gpu']}"
            )

    def test_known_fir_h100_80gb_values(self, mod):
        p = mod.GPU_PROFILES["fir:h100-80gb"]
        assert p["cluster"] == "fir"
        assert p["cpu"] == 12
        assert p["mem"] == 280
        assert p["gres_gpu"] == "gpu:h100:1"

    def test_known_narval_a100_40gb_values(self, mod):
        p = mod.GPU_PROFILES["narval:a100-40gb"]
        assert p["cluster"] == "narval"
        assert p["cpu"] == 12
        assert p["mem"] == 124
        assert p["gres_gpu"] == "gpu:a100:1"


# ---------------------------------------------------------------------------
# GPU_PROFILE_ALIASES
# ---------------------------------------------------------------------------


class TestGpuProfileAliases:
    def test_all_aliases_point_to_valid_profiles(self, mod):
        for alias, target in mod.GPU_PROFILE_ALIASES.items():
            assert target in mod.GPU_PROFILES, (
                f"Alias '{alias}' points to unknown profile '{target}'"
            )

    def test_h100_80gb_alias(self, mod):
        assert mod.GPU_PROFILE_ALIASES["h100-80gb"] == "fir:h100-80gb"

    def test_a100_40gb_alias(self, mod):
        assert mod.GPU_PROFILE_ALIASES["a100-40gb"] == "narval:a100-40gb"

    def test_a6000_alias(self, mod):
        assert mod.GPU_PROFILE_ALIASES["a6000"] == "vulcan:l40s"


# ---------------------------------------------------------------------------
# apply_gpu_profile
# ---------------------------------------------------------------------------


class TestApplyGpuProfile:
    def test_no_profile_is_noop(self, mod):
        args = make_args(gpu_profile=None, cpu=4, mem=64)
        mod.apply_gpu_profile(args)
        assert args.cpu == 4
        assert args.mem == 64

    def test_full_profile_key_sets_resources(self, mod):
        args = make_args(gpu_profile="fir:h100-80gb", gpu=0)
        mod.apply_gpu_profile(args)
        expected = mod.GPU_PROFILES["fir:h100-80gb"]
        assert args.cpu == expected["cpu"]
        assert args.mem == expected["mem"]
        assert args.gres_gpu == expected["gres_gpu"]

    def test_alias_resolves_to_correct_profile(self, mod):
        args = make_args(gpu_profile="a100-40gb", gpu=0)
        mod.apply_gpu_profile(args)
        expected = mod.GPU_PROFILES["narval:a100-40gb"]
        assert args.cpu == expected["cpu"]
        assert args.mem == expected["mem"]
        assert args.gres_gpu == expected["gres_gpu"]

    def test_case_insensitive_input(self, mod):
        args = make_args(gpu_profile="FIR:H100-80GB", gpu=0)
        mod.apply_gpu_profile(args)
        expected = mod.GPU_PROFILES["fir:h100-80gb"]
        assert args.cpu == expected["cpu"]
        assert args.gres_gpu == expected["gres_gpu"]

    def test_whitespace_in_input(self, mod):
        args = make_args(gpu_profile="  narval:a100-40gb  ", gpu=0)
        mod.apply_gpu_profile(args)
        expected = mod.GPU_PROFILES["narval:a100-40gb"]
        assert args.cpu == expected["cpu"]

    def test_unknown_profile_raises_value_error(self, mod):
        args = make_args(gpu_profile="nonexistent:gpu")
        with pytest.raises(ValueError, match="Unknown gpu profile"):
            mod.apply_gpu_profile(args)

    def test_gpu_set_to_1_when_zero(self, mod):
        """gpu=0 should be bumped to 1 after applying a profile."""
        args = make_args(gpu_profile="fir:h100-1g.10gb", gpu=0)
        mod.apply_gpu_profile(args)
        assert args.gpu == 1

    def test_gpu_not_overwritten_when_already_positive(self, mod):
        """Pre-existing positive gpu value must be preserved."""
        args = make_args(gpu_profile="fir:h100-1g.10gb", gpu=2)
        mod.apply_gpu_profile(args)
        assert args.gpu == 2

    def test_mig_slice_sets_correct_gres(self, mod):
        args = make_args(gpu_profile="narval:a100-1g.5gb", gpu=0)
        mod.apply_gpu_profile(args)
        assert args.gres_gpu == "gpu:a100_1g.5gb:1"

    def test_all_profiles_apply_without_error(self, mod):
        """Smoke test: every declared profile applies cleanly."""
        for name in mod.GPU_PROFILES:
            args = make_args(gpu_profile=name, gpu=0)
            mod.apply_gpu_profile(args)

    def test_alias_case_insensitive(self, mod):
        """Aliases should also be resolved case-insensitively."""
        args = make_args(gpu_profile="A100-40GB", gpu=0)
        mod.apply_gpu_profile(args)
        expected = mod.GPU_PROFILES["narval:a100-40gb"]
        assert args.cpu == expected["cpu"]
