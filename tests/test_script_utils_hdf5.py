"""Tests for the logic merged in script_utils.run_inference_hdf5.

The conflict at line ~1151 combined two changes:
  HEAD:    if k not in preds_l: continue       (skip absent sequential keys)
  ccx_run: .detach().cpu().numpy()             (safe conversion for GPU/grad tensors)

We test these behaviours directly against inline versions of the loop,
and via the HDF5-writing helper, without importing the full script_utils
module (which has optional heavy dependencies: lmdb, wandb, etc.).
"""
import tempfile

import h5py
import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Inline replica of the merged save loop from run_inference_hdf5
# (lines 1149-1158 in script_utils.py after resolution)
# ---------------------------------------------------------------------------

def _run_sequential_save_loop(preds_l: dict, sequential_name: set, save_file: h5py.File, spec_start_point: int):
    """Replica of the merged sequential-key save loop."""
    def resize_and_save(k, save_v, start):
        if k not in save_file:
            save_file.create_dataset(k, shape=(0,), dtype=save_v.dtype, maxshape=(None,))
        save_file[k].resize(start + len(save_v), axis=0)
        save_file[k][start:start + len(save_v)] = save_v

    save_length = None
    for k in sequential_name:
        if k not in preds_l:           # ← HEAD guard
            continue
        save_v = torch.cat(preds_l[k], 0).detach().cpu().numpy()  # ← ccx_run
        if save_length is None:
            save_length = len(save_v)
        else:
            assert len(save_v) == save_length, k
        resize_and_save(k, save_v, spec_start_point - save_length)
    return save_length


# ---------------------------------------------------------------------------
# Tests: HEAD guard — missing sequential keys skipped silently
# ---------------------------------------------------------------------------

class TestMissingSequentialKeyGuard:
    """if k not in preds_l: continue — absent keys must not raise."""

    def test_all_keys_missing_no_error(self, tmp_path):
        """When preds_l is empty, iterating sequential_name must not raise."""
        preds_l = {}
        sequential = {"pred_mzs", "pred_ints", "frag_formula_peak_idxs", "frag_formula_peak_mzs"}
        h5_fp = str(tmp_path / "out.h5")
        with h5py.File(h5_fp, "w") as f:
            save_length = _run_sequential_save_loop(preds_l, sequential, f, spec_start_point=0)
        assert save_length is None
        with h5py.File(h5_fp, "r") as f:
            assert len(f.keys()) == 0

    def test_partial_keys_missing_only_present_saved(self, tmp_path):
        """frag_formula_* absent: only pred_mzs and pred_ints are saved."""
        t1 = [torch.tensor([1.0, 2.0]), torch.tensor([3.0])]
        t2 = [torch.tensor([0.5, 0.6]), torch.tensor([0.7])]
        preds_l = {"pred_mzs": t1, "pred_ints": t2}
        sequential = {"pred_mzs", "pred_ints", "frag_formula_peak_idxs", "frag_formula_peak_mzs"}
        n_total = sum(len(t) for t in t1)  # 3
        spec_start = n_total  # so offset = spec_start - save_length = 0

        h5_fp = str(tmp_path / "partial.h5")
        with h5py.File(h5_fp, "w") as f:
            _run_sequential_save_loop(preds_l, sequential, f, spec_start_point=spec_start)

        with h5py.File(h5_fp, "r") as f:
            assert "pred_mzs" in f
            assert "pred_ints" in f
            assert "frag_formula_peak_idxs" not in f
            assert "frag_formula_peak_mzs" not in f

    def test_no_key_error_without_guard_would_raise(self):
        """Demonstrate that accessing preds_l[k] for a missing key raises KeyError."""
        preds_l = {}
        sequential = {"pred_mzs"}
        with pytest.raises(KeyError):
            _ = preds_l["pred_mzs"]  # what happens without the guard

    def test_single_absent_key_among_present_skipped(self, tmp_path):
        """Only frag_formula_peak_idxs is absent; other three keys are saved."""
        n = 4
        t = [torch.rand(n)]
        preds_l = {
            "pred_mzs": t,
            "pred_ints": t,
            "frag_formula_peak_mzs": t,
            # frag_formula_peak_idxs intentionally omitted
        }
        sequential = {"pred_mzs", "pred_ints", "frag_formula_peak_idxs", "frag_formula_peak_mzs"}
        h5_fp = str(tmp_path / "skip_one.h5")
        with h5py.File(h5_fp, "w") as f:
            _run_sequential_save_loop(preds_l, sequential, f, spec_start_point=n)
        with h5py.File(h5_fp, "r") as f:
            assert "frag_formula_peak_idxs" not in f
            assert "pred_mzs" in f


# ---------------------------------------------------------------------------
# Tests: ccx_run change — .detach().cpu().numpy() handles grad + GPU tensors
# ---------------------------------------------------------------------------

class TestDetachCpuConversion:
    """torch.cat(...).detach().cpu().numpy() must handle requires_grad and GPU tensors."""

    def test_requires_grad_tensor_saved_without_error(self, tmp_path):
        """Tensor with requires_grad=True must be saved; without .detach() this raises."""
        t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        # Verify that without .detach() it raises:
        with pytest.raises(RuntimeError, match="requires grad"):
            _ = t.numpy()

        # With .detach() it works:
        result = t.detach().cpu().numpy()
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_list_of_grad_tensors_cat_and_convert(self, tmp_path):
        """List of tensors with requires_grad → cat → detach → cpu → numpy."""
        tensors = [
            torch.tensor([1.0, 2.0], requires_grad=True),
            torch.tensor([3.0, 4.0], requires_grad=True),
        ]
        preds_l = {"pred_mzs": tensors}
        sequential = {"pred_mzs"}
        h5_fp = str(tmp_path / "grad.h5")
        with h5py.File(h5_fp, "w") as f:
            _run_sequential_save_loop(preds_l, sequential, f, spec_start_point=4)
        with h5py.File(h5_fp, "r") as f:
            np.testing.assert_allclose(f["pred_mzs"][:], [1.0, 2.0, 3.0, 4.0])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_tensor_moved_to_cpu(self, tmp_path):
        """Tensor on CUDA must be moved to CPU before numpy conversion."""
        tensors = [torch.tensor([5.0, 6.0]).cuda()]
        preds_l = {"pred_mzs": tensors}
        sequential = {"pred_mzs"}
        h5_fp = str(tmp_path / "cuda.h5")
        with h5py.File(h5_fp, "w") as f:
            _run_sequential_save_loop(preds_l, sequential, f, spec_start_point=2)
        with h5py.File(h5_fp, "r") as f:
            np.testing.assert_allclose(f["pred_mzs"][:], [5.0, 6.0])

    def test_cpu_tensor_without_grad_unaffected(self, tmp_path):
        """Normal CPU tensors still work correctly after the change."""
        tensors = [torch.tensor([10.0, 20.0]), torch.tensor([30.0])]
        preds_l = {"pred_mzs": tensors}
        sequential = {"pred_mzs"}
        h5_fp = str(tmp_path / "normal.h5")
        with h5py.File(h5_fp, "w") as f:
            _run_sequential_save_loop(preds_l, sequential, f, spec_start_point=3)
        with h5py.File(h5_fp, "r") as f:
            np.testing.assert_allclose(f["pred_mzs"][:], [10.0, 20.0, 30.0])


# ---------------------------------------------------------------------------
# Tests: data integrity — values round-trip correctly
# ---------------------------------------------------------------------------

class TestDataIntegrity:
    """Values written to HDF5 match the input tensors."""

    def test_mz_values_round_trip(self, tmp_path):
        mzs = [torch.tensor([100.0, 200.0, 300.0]), torch.tensor([50.0, 150.0])]
        preds_l = {"pred_mzs": mzs}
        sequential = {"pred_mzs"}
        total = sum(len(t) for t in mzs)  # 5
        h5_fp = str(tmp_path / "vals.h5")
        with h5py.File(h5_fp, "w") as f:
            _run_sequential_save_loop(preds_l, sequential, f, spec_start_point=total)
        with h5py.File(h5_fp, "r") as f:
            np.testing.assert_allclose(f["pred_mzs"][:], [100.0, 200.0, 300.0, 50.0, 150.0])

    def test_length_consistency_assertion_fires(self, tmp_path):
        """If two sequential keys have different lengths, the assertion inside the loop raises."""
        preds_l = {
            "pred_mzs":  [torch.rand(3)],
            "pred_ints": [torch.rand(5)],   # different length → should assert
        }
        sequential = {"pred_mzs", "pred_ints"}
        h5_fp = str(tmp_path / "mismatch.h5")
        with h5py.File(h5_fp, "w") as f:
            with pytest.raises(AssertionError):
                _run_sequential_save_loop(preds_l, sequential, f, spec_start_point=8)

    def test_multiple_chunks_append_correctly(self, tmp_path):
        """Calling the loop twice simulates chunk accumulation; values concatenate."""
        preds_l_1 = {"pred_mzs": [torch.tensor([1.0, 2.0])]}
        preds_l_2 = {"pred_mzs": [torch.tensor([3.0, 4.0, 5.0])]}
        sequential = {"pred_mzs"}
        h5_fp = str(tmp_path / "chunks.h5")
        with h5py.File(h5_fp, "w") as f:
            _run_sequential_save_loop(preds_l_1, sequential, f, spec_start_point=2)
            _run_sequential_save_loop(preds_l_2, sequential, f, spec_start_point=5)
        with h5py.File(h5_fp, "r") as f:
            np.testing.assert_allclose(f["pred_mzs"][:], [1.0, 2.0, 3.0, 4.0, 5.0])
