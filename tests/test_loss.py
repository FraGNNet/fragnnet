"""Unit tests for fragnnet.model.loss module.

Tests for pairwise similarity and distance metrics:
- Cosine similarity
- Cross-entropy
- Jensen-Shannon divergence
"""

import math

import pytest
import torch as th

from fragnnet.model.loss import (
    _compute_pair_ce_hun,
    _compute_pair_jss_hun,
    get_pairwise_cossim,
    get_pairwise_cross_entropy,
    get_pairwise_jss_sim,
    get_pairwise_jss_sim_hun,
    sparse_jensen_shannon_divergence,
    sparse_jensen_shannon_divergence_hungarian,
    sparse_jensen_shannon_divergence_hungarian_vec,
)


class TestPairwiseCossim:
    """Tests for cosine similarity computation."""

    def test_pairwise_cossim_orthogonal_spectra(self):
        """Test cosine similarity for orthogonal (non-overlapping) spectra."""
        mzs = th.tensor([50.0, 150.0])
        logprobs = th.tensor([0.0, 0.0])
        batch_idxs = th.tensor([0, 1], dtype=th.long)
        batch_size = 2
        mz_max = 200.0
        mz_bin_res = 1.0

        S = get_pairwise_cossim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)

        # Diagonal should be 1 (self-similarity)
        assert th.allclose(S[0, 0], th.tensor(1.0), atol=1e-6), "Self-similarity should be 1"
        assert th.allclose(S[1, 1], th.tensor(1.0), atol=1e-6), "Self-similarity should be 1"
        # Off-diagonal: orthogonal vectors have zero cosine similarity
        assert S[0, 1] == 0.0 or th.isnan(S[0, 1]), "Orthogonal spectra should have similarity ~0"
        assert S[1, 0] == 0.0 or th.isnan(S[1, 0]), "Orthogonal spectra should have similarity ~0"

    def test_pairwise_cossim_empty_input(self):
        """Test cosine similarity with empty spectra."""
        mzs = th.tensor([], dtype=th.float32)
        logprobs = th.tensor([], dtype=th.float32)
        batch_idxs = th.tensor([], dtype=th.long)
        batch_size = 3
        mz_max = 100.0
        mz_bin_res = 1.0

        S = get_pairwise_cossim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
        assert S.shape == (batch_size, batch_size)
        assert th.allclose(S, th.zeros_like(S))

    def test_pairwise_cossim_symmetry(self):
        """Test that cosine similarity matrix is symmetric."""
        mzs = th.tensor([50.0, 75.0, 100.0, 50.0, 150.0])
        logprobs = th.tensor([0.0, 1.0, 0.5, 1.5, 0.2])
        batch_idxs = th.tensor([0, 1, 0, 1, 2], dtype=th.long)
        batch_size = 3
        mz_max = 200.0
        mz_bin_res = 1.0

        S = get_pairwise_cossim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)

        # Check symmetry
        assert th.allclose(S, S.t(), atol=1e-6), "Matrix should be symmetric"
        # Check diagonal is 1 (self-similarity)
        assert th.allclose(th.diag(S), th.ones(batch_size), atol=1e-6), "Diagonal should be 1"
        # Check bounds [0, 1]
        assert th.all(S >= -1e-6) and th.all(S <= 1 + 1e-6), "Similarity should be in [0, 1]"

    def test_pairwise_cossim_chunking(self):
        """Test that chunked computation matches non-chunked."""
        mzs = th.tensor([50.0, 75.0, 100.0, 125.0, 150.0, 175.0])
        logprobs = th.tensor([0.0, 1.0, 0.5, 1.5, 0.2, 0.8])
        batch_idxs = th.tensor([0, 1, 0, 1, 2, 2], dtype=th.long)
        batch_size = 3
        mz_max = 200.0
        mz_bin_res = 1.0

        # Compute without chunking
        S_no_chunk = get_pairwise_cossim(
            mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res, chunk_size=None
        )

        # Compute with chunking
        S_chunk = get_pairwise_cossim(
            mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res, chunk_size=2
        )

        assert th.allclose(S_no_chunk, S_chunk, atol=1e-5), (
            "Chunked and non-chunked results should match"
        )

    def test_pairwise_cossim_output_shape(self):
        """Test output shape is correct."""
        mzs = th.tensor([50.0, 100.0, 150.0])
        logprobs = th.tensor([0.0, 1.0, 0.5])
        batch_idxs = th.tensor([0, 1, 2], dtype=th.long)
        batch_size = 3
        mz_max = 200.0
        mz_bin_res = 1.0

        S = get_pairwise_cossim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
        assert S.shape == (batch_size, batch_size), (
            f"Expected shape {(batch_size, batch_size)}, got {S.shape}"
        )

    def test_pairwise_cossim_identical_spectra(self):
        """Regression: identical spectra in different batch slots must have cosine sim = 1.

        Before the batch-offset fix, batched_bin_func returned batch-offset bin indices.
        Each batch element occupied exclusively different columns in the V matrix, making
        all off-diagonal dot products zero. This test catches that regression: two
        identical single-peak spectra must produce off-diagonal similarity = 1, not 0.
        """
        # Two batch elements, each with one peak at m/z=100, equal intensity.
        mzs = th.tensor([100.0, 100.0])
        logprobs = th.tensor([0.0, 0.0])  # log(1) — single peak per spectrum
        batch_idxs = th.tensor([0, 1], dtype=th.long)
        batch_size = 2
        mz_max = 200.0
        mz_bin_res = 1.0

        S = get_pairwise_cossim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)

        assert th.allclose(S[0, 0], th.tensor(1.0), atol=1e-6), "Self-similarity should be 1"
        assert th.allclose(S[1, 1], th.tensor(1.0), atol=1e-6), "Self-similarity should be 1"
        assert th.allclose(S[0, 1], th.tensor(1.0), atol=1e-6), (
            "Identical spectra must have cosine similarity 1, not 0 (batch-offset bug)"
        )
        assert th.allclose(S[1, 0], th.tensor(1.0), atol=1e-6), (
            "Identical spectra must have cosine similarity 1, not 0 (batch-offset bug)"
        )

    def test_pairwise_cossim_known_partial_overlap(self):
        """Spectra sharing half their peaks should have cosine similarity = 0.5.

        Spectrum 0: equal-intensity peaks at m/z=50 and m/z=100.
        Spectrum 1: equal-intensity peaks at m/z=50 and m/z=150.
        After L2 normalization both vectors are [1/√2, 1/√2, 0] and [1/√2, 0, 1/√2]
        in the three-bin space {50, 100, 150}. Their dot product = 1/2.
        """
        import math

        log_half = math.log(0.5)
        mzs = th.tensor([50.0, 100.0, 50.0, 150.0])
        logprobs = th.tensor([log_half, log_half, log_half, log_half])
        batch_idxs = th.tensor([0, 0, 1, 1], dtype=th.long)
        batch_size = 2
        mz_max = 200.0
        mz_bin_res = 1.0

        S = get_pairwise_cossim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)

        assert th.allclose(S[0, 0], th.tensor(1.0), atol=1e-6), "Self-similarity should be 1"
        assert th.allclose(S[1, 1], th.tensor(1.0), atol=1e-6), "Self-similarity should be 1"
        assert th.allclose(S[0, 1], th.tensor(0.5), atol=1e-5), (
            f"Expected cosine similarity 0.5 for 50% peak overlap, got {S[0, 1].item():.6f}"
        )
        assert th.allclose(S[1, 0], th.tensor(0.5), atol=1e-5), "Matrix must be symmetric"

    def test_pairwise_cossim_mse_nonzero_when_spectra_differ(self):
        """MSE between two different pairwise sim matrices must be nonzero.

        This is the end-to-end regression for the pairwise cosine loss in fragnnet_pl.py:
        if pred_sim ≠ true_sim, F.mse_loss(pred_sim, true_sim) must be > 0.
        Before the fix, both matrices were always the identity, giving MSE ≈ 0 regardless
        of predictions.
        """
        import torch.nn.functional as F

        # true: two identical spectra at m/z=100 → sim matrix all-ones
        mzs_true = th.tensor([100.0, 100.0])
        logprobs_true = th.tensor([0.0, 0.0])
        batch_idxs = th.tensor([0, 1], dtype=th.long)
        batch_size = 2
        mz_max = 200.0
        mz_bin_res = 1.0

        # pred: two orthogonal spectra → off-diagonal sim = 0
        mzs_pred = th.tensor([50.0, 150.0])
        logprobs_pred = th.tensor([0.0, 0.0])

        true_sim = get_pairwise_cossim(
            mzs_true, logprobs_true, batch_idxs, batch_size, mz_max, mz_bin_res
        )
        pred_sim = get_pairwise_cossim(
            mzs_pred, logprobs_pred, batch_idxs, batch_size, mz_max, mz_bin_res
        )

        mse = F.mse_loss(pred_sim, true_sim)
        assert mse.item() > 0.1, (
            f"MSE between different pairwise sim matrices must be > 0.1, got {mse.item():.2e}. "
            "If this is ~1e-9, the batch-offset bug is present."
        )


class TestPairwiseCrossEntropy:
    """Tests for cross-entropy computation."""

    def test_pairwise_cross_entropy_disjoint_spectra(self):
        """Test cross-entropy for disjoint (non-overlapping) spectra."""
        mzs = th.tensor([50.0, 150.0])
        logprobs = th.tensor([0.0, 0.0])
        batch_idxs = th.tensor([0, 1], dtype=th.long)
        batch_size = 2
        mz_max = 200.0
        mz_bin_res = 1.0

        CE = get_pairwise_cross_entropy(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
        # Diagonal should be 0 (CE between identical distributions)
        assert th.allclose(th.diag(CE), th.zeros(batch_size), atol=1e-6), "Diagonal should be 0"
        # Off-diagonal for completely disjoint spectra should be high
        assert CE[0, 1] > 5.0, "CE for disjoint spectra should be high"
        assert CE[1, 0] > 5.0, "CE for disjoint spectra should be high"

    def test_pairwise_cross_entropy_empty_input(self):
        """Test cross-entropy with empty spectra."""
        mzs = th.tensor([], dtype=th.float32)
        logprobs = th.tensor([], dtype=th.float32)
        batch_idxs = th.tensor([], dtype=th.long)
        batch_size = 3
        mz_max = 100.0
        mz_bin_res = 1.0

        CE = get_pairwise_cross_entropy(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
        assert CE.shape == (batch_size, batch_size)
        assert th.allclose(CE, th.zeros_like(CE))

    def test_pairwise_cross_entropy_non_symmetric(self):
        """Test that cross-entropy is not necessarily symmetric."""
        mzs = th.tensor([50.0, 100.0, 150.0])
        logprobs = th.tensor([0.0, 1.0, 0.5])
        batch_idxs = th.tensor([0, 1, 2], dtype=th.long)
        batch_size = 3
        mz_max = 200.0
        mz_bin_res = 1.0

        CE = get_pairwise_cross_entropy(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)

        # Diagonal should be 0
        assert th.allclose(th.diag(CE), th.zeros(batch_size), atol=1e-6), "Diagonal should be 0"

    def test_pairwise_cross_entropy_chunking(self):
        """Test that chunked computation matches non-chunked."""
        mzs = th.tensor([50.0, 75.0, 100.0, 125.0, 150.0, 175.0])
        logprobs = th.tensor([0.0, 1.0, 0.5, 1.5, 0.2, 0.8])
        batch_idxs = th.tensor([0, 1, 0, 1, 2, 2], dtype=th.long)
        batch_size = 3
        mz_max = 200.0
        mz_bin_res = 1.0

        # Compute without chunking
        CE_no_chunk = get_pairwise_cross_entropy(
            mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res, chunk_size=None
        )

        # Compute with chunking
        CE_chunk = get_pairwise_cross_entropy(
            mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res, chunk_size=2
        )

        assert th.allclose(CE_no_chunk, CE_chunk, atol=1e-5), (
            "Chunked and non-chunked results should match"
        )

    def test_pairwise_cross_entropy_output_shape(self):
        """Test output shape is correct."""
        mzs = th.tensor([50.0, 100.0, 150.0])
        logprobs = th.tensor([0.0, 1.0, 0.5])
        batch_idxs = th.tensor([0, 1, 2], dtype=th.long)
        batch_size = 3
        mz_max = 200.0
        mz_bin_res = 1.0

        CE = get_pairwise_cross_entropy(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
        assert CE.shape == (batch_size, batch_size), (
            f"Expected shape {(batch_size, batch_size)}, got {CE.shape}"
        )

    def test_pairwise_cross_entropy_non_negative(self):
        """Test that cross-entropy is non-negative."""
        mzs = th.tensor([50.0, 75.0, 100.0, 125.0, 150.0])
        logprobs = th.tensor([0.0, 1.0, 0.5, 1.5, 0.2])
        batch_idxs = th.tensor([0, 1, 0, 1, 2], dtype=th.long)
        batch_size = 3
        mz_max = 200.0
        mz_bin_res = 1.0

        CE = get_pairwise_cross_entropy(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)

        # Cross-entropy should be non-negative
        assert th.all(CE >= -1e-6), "Cross-entropy should be non-negative"

    def test_pairwise_cross_entropy_multiple_chunks(self):
        """Test with different chunk sizes."""
        mzs = th.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
        logprobs = th.tensor([0.0, 1.0, 0.5, 1.5, 0.2, 0.8, 0.3, 1.2])
        batch_idxs = th.tensor([0, 1, 0, 1, 2, 2, 3, 3], dtype=th.long)
        batch_size = 4
        mz_max = 100.0
        mz_bin_res = 1.0

        # Test with various chunk sizes
        for chunk_size in [1, 2, 4, None]:
            CE = get_pairwise_cross_entropy(
                mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res, chunk_size=chunk_size
            )
            assert CE.shape == (batch_size, batch_size)


class TestPairwiseJssSim:
    """Tests for Jensen-Shannon divergence based similarity."""

    def test_pairwise_jss_sim_single_peak(self):
        """Test JSS similarity for single-peak spectra."""
        mzs = th.tensor([50.0, 100.0])
        logprobs = th.tensor([0.0, 0.0])
        batch_idxs = th.tensor([0, 1], dtype=th.long)
        batch_size = 2
        mz_max = 200.0
        mz_bin_res = 1.0

        JSS = get_pairwise_jss_sim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
        # Diagonal should be 1 (self-similarity)
        assert th.allclose(th.diag(JSS), th.ones(batch_size), atol=1e-6), "Diagonal should be 1"
        # Off-diagonal for different peaks should be small
        assert JSS[0, 1] < 0.5, "Different spectra should have lower similarity"
        assert JSS[1, 0] < 0.5, "Different spectra should have lower similarity"
        # Symmetry
        assert th.allclose(JSS[0, 1], JSS[1, 0], atol=1e-6), "JSS should be symmetric"

    def test_pairwise_jss_sim_empty_input(self):
        """Test JSS similarity with empty spectra."""
        mzs = th.tensor([], dtype=th.float32)
        logprobs = th.tensor([], dtype=th.float32)
        batch_idxs = th.tensor([], dtype=th.long)
        batch_size = 4
        mz_max = 100.0
        mz_bin_res = 1.0

        JSS = get_pairwise_jss_sim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
        assert JSS.shape == (batch_size, batch_size)
        assert th.allclose(JSS, th.zeros_like(JSS))

    def test_pairwise_jss_sim_symmetry(self):
        """Test that JSS similarity matrix is symmetric."""
        mzs = th.tensor([50.0, 75.0, 100.0, 50.0, 150.0])
        logprobs = th.tensor([0.0, 1.0, 0.5, 1.5, 0.2])
        batch_idxs = th.tensor([0, 1, 0, 1, 2], dtype=th.long)
        batch_size = 3
        mz_max = 200.0
        mz_bin_res = 1.0

        JSS = get_pairwise_jss_sim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)

        # Check symmetry
        assert th.allclose(JSS, JSS.t(), atol=1e-6), "Matrix should be symmetric"
        # Check diagonal is 1 (self-similarity)
        assert th.allclose(th.diag(JSS), th.ones(batch_size), atol=1e-6), "Diagonal should be 1"
        # Check bounds [0, 1]
        assert th.all(JSS >= -1e-6) and th.all(JSS <= 1 + 1e-6), "JSS should be in [0, 1]"

    def test_pairwise_jss_sim_chunking(self):
        """Test that chunked computation matches non-chunked."""
        mzs = th.tensor([50.0, 75.0, 100.0, 125.0, 150.0, 175.0])
        logprobs = th.tensor([0.0, 1.0, 0.5, 1.5, 0.2, 0.8])
        batch_idxs = th.tensor([0, 1, 0, 1, 2, 2], dtype=th.long)
        batch_size = 3
        mz_max = 200.0
        mz_bin_res = 1.0

        # Compute without chunking
        JSS_no_chunk = get_pairwise_jss_sim(
            mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res, chunk_size=None
        )

        # Compute with chunking
        JSS_chunk = get_pairwise_jss_sim(
            mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res, chunk_size=2
        )

        assert th.allclose(JSS_no_chunk, JSS_chunk, atol=1e-5), (
            "Chunked and non-chunked results should match"
        )

    def test_pairwise_jss_sim_output_shape(self):
        """Test output shape is correct."""
        mzs = th.tensor([50.0, 100.0, 150.0])
        logprobs = th.tensor([0.0, 1.0, 0.5])
        batch_idxs = th.tensor([0, 1, 2], dtype=th.long)
        batch_size = 3
        mz_max = 200.0
        mz_bin_res = 1.0

        JSS = get_pairwise_jss_sim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
        assert JSS.shape == (batch_size, batch_size), (
            f"Expected shape {(batch_size, batch_size)}, got {JSS.shape}"
        )

    def test_pairwise_jss_sim_orthogonal_spectra(self):
        """Test orthogonal spectra have lower similarity."""
        mzs = th.tensor([50.0, 150.0])
        logprobs = th.tensor([0.0, 0.0])
        batch_idxs = th.tensor([0, 1], dtype=th.long)
        batch_size = 2
        mz_max = 200.0
        mz_bin_res = 1.0

        JSS = get_pairwise_jss_sim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)

        # Off-diagonal should be 0 (no overlap)
        assert th.allclose(JSS[0, 1], th.tensor(0.0), atol=1e-6), (
            "Non-overlapping spectra should have 0 similarity"
        )
        assert th.allclose(JSS[1, 0], th.tensor(0.0), atol=1e-6), (
            "Non-overlapping spectra should have 0 similarity"
        )

    def test_pairwise_jss_sim_multiple_chunks(self):
        """Test with different chunk sizes."""
        mzs = th.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
        logprobs = th.tensor([0.0, 1.0, 0.5, 1.5, 0.2, 0.8, 0.3, 1.2])
        batch_idxs = th.tensor([0, 1, 0, 1, 2, 2, 3, 3], dtype=th.long)
        batch_size = 4
        mz_max = 100.0
        mz_bin_res = 1.0

        # Test with various chunk sizes
        for chunk_size in [1, 2, 4, None]:
            JSS = get_pairwise_jss_sim(
                mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res, chunk_size=chunk_size
            )
            assert JSS.shape == (batch_size, batch_size)


class TestHungarianJSS:
    """Tests for JSS computation using Hungarian matching."""

    def test_compute_pair_jss_hun_identical(self):
        """Test JSS for identical spectra is 1.0."""
        mzs = th.tensor([100.0, 200.0])
        logprobs = th.tensor([0.0, 0.0])  # Equal intensities

        sim = _compute_pair_jss_hun(
            mzs, logprobs, mzs, logprobs, tolerance=1e-5, relative=True, tolerance_min_mz=1.0
        )
        assert th.allclose(sim, th.tensor(1.0), atol=1e-6)

    def test_compute_pair_jss_hun_orthogonal(self):
        """Test JSS for orthogonal spectra is 0.0."""
        mzs1 = th.tensor([100.0])
        logprobs1 = th.tensor([0.0])
        mzs2 = th.tensor([200.0])
        logprobs2 = th.tensor([0.0])

        sim = _compute_pair_jss_hun(
            mzs1, logprobs1, mzs2, logprobs2, tolerance=1e-5, relative=True, tolerance_min_mz=1.0
        )
        assert th.allclose(sim, th.tensor(0.0), atol=1e-6)

    def test_compute_pair_jss_hun_tolerance(self):
        """Test spectra matching within tolerance."""
        mzs1 = th.tensor([100.0])
        logprobs1 = th.tensor([0.0])
        mzs2 = th.tensor([100.0001])
        logprobs2 = th.tensor([0.0])

        # Within tolerance
        sim = _compute_pair_jss_hun(
            mzs1, logprobs1, mzs2, logprobs2, tolerance=0.01, relative=False, tolerance_min_mz=1.0
        )
        assert th.allclose(sim, th.tensor(1.0), atol=1e-6)

        # Outside tolerance
        sim_out = _compute_pair_jss_hun(
            mzs1, logprobs1, mzs2, logprobs2, tolerance=1e-6, relative=False, tolerance_min_mz=1.0
        )
        assert th.allclose(sim_out, th.tensor(0.0), atol=1e-6)

    def test_get_pairwise_jss_sim_hun(self):
        """Test the pairwise wrapper for Hungarian JSS."""
        mzs = th.tensor([100.0, 200.0])
        logprobs = th.tensor([0.0, 0.0])
        batch_idxs = th.tensor([0, 1], dtype=th.long)
        batch_size = 2

        S = get_pairwise_jss_sim_hun(
            mzs, logprobs, batch_idxs, batch_size, tolerance=1e-5, relative=True
        )

        assert S.shape == (2, 2)
        assert th.allclose(S[0, 0], th.tensor(1.0))
        assert th.allclose(S[1, 1], th.tensor(1.0))
        assert th.allclose(S[0, 1], th.tensor(0.0), atol=1e-6)
        assert th.allclose(S[1, 0], th.tensor(0.0), atol=1e-6)

    def test_sparse_jss_hun_vec_matches_sequential(self):
        """Vectorized training JSS-Hungarian should match the sequential loss."""
        true_mzs = th.tensor([100.0, 200.0, 100.0, 300.0])
        pred_mzs = th.tensor([100.0, 250.0, 100.0, 305.0])
        true_logprobs = th.log(th.tensor([0.6, 0.4, 0.7, 0.3]))
        pred_logprobs = th.log(th.tensor([0.5, 0.5, 0.8, 0.2]))
        batch_idxs = th.tensor([0, 0, 1, 1], dtype=th.long)

        seq = sparse_jensen_shannon_divergence_hungarian(
            true_mzs,
            true_logprobs,
            batch_idxs,
            pred_mzs,
            pred_logprobs,
            batch_idxs,
            tolerance=0.01,
            relative=False,
        )
        vec = sparse_jensen_shannon_divergence_hungarian_vec(
            true_mzs,
            true_logprobs,
            batch_idxs,
            pred_mzs,
            pred_logprobs,
            batch_idxs,
            tolerance=0.01,
            relative=False,
            loss_batch_size=2,
        )

        assert th.allclose(vec, seq, atol=1e-6), f"vec={vec.tolist()} seq={seq.tolist()}"

    def test_sparse_jss_hun_vec_per_sample_tolerance(self):
        """Vectorized JSS-Hungarian should honor per-sample tolerance overrides."""
        true_mzs = th.tensor([500.000, 500.000])
        pred_mzs = th.tensor([500.007, 500.007])
        logprobs = th.tensor([0.0, 0.0])
        batch_idxs = th.tensor([0, 1], dtype=th.long)

        vec = sparse_jensen_shannon_divergence_hungarian_vec(
            true_mzs,
            logprobs,
            batch_idxs,
            pred_mzs,
            logprobs,
            batch_idxs,
            tolerance=1e-5,
            relative=True,
            tolerance_min_mz=200.0,
            tol_per_sample=th.tensor([1e-5, 2e-5]),
            min_mz_per_sample=th.tensor([200.0, 750.0]),
            loss_batch_size=2,
        )

        assert vec[0].item() > vec[1].item()
        assert vec[1].item() == pytest.approx(0.0, abs=1e-5)


class TestHungarianCE:
    """Tests for Cross-Entropy computation using Hungarian matching."""

    def test_compute_pair_ce_hun_identical(self):
        """Test CE for identical spectra is low (minimizing CE)."""
        mzs = th.tensor([100.0])
        logprobs = th.tensor([0.0])

        ce = _compute_pair_ce_hun(
            mzs, logprobs, mzs, logprobs, tolerance=1e-5, relative=True, tolerance_min_mz=1.0
        )
        # For P=Q=[1], CE = -sum(1 * log(1)) = 0.
        # However, we use log(1+eps) for stability, so it should be close to 0.
        assert ce < 1e-3

    def test_compute_pair_ce_hun_orthogonal(self):
        """Test CE for orthogonal spectra is high."""
        mzs1 = th.tensor([100.0])
        logprobs1 = th.tensor([0.0])
        mzs2 = th.tensor([200.0])
        logprobs2 = th.tensor([0.0])

        ce = _compute_pair_ce_hun(
            mzs1, logprobs1, mzs2, logprobs2, tolerance=1e-5, relative=True, tolerance_min_mz=1.0
        )
        # For orthogonal, it should mismatch and use log_eps.
        # CE = -log(eps)

        from fragnnet.model.loss import EPS

        expected = -math.log(EPS)
        assert th.allclose(ce, th.tensor(expected), atol=1e-3)


class TestSparseJensenShannonDivergence:
    """Tests for sparse_jensen_shannon_divergence (wraps jss_helper from spec_utils)."""

    def test_identical_single_peak(self):
        """Identical single-peak spectra should have zero divergence."""
        mzs = th.tensor([100.0])
        logprobs = th.tensor([0.0])
        batch_idxs = th.tensor([0], dtype=th.long)

        jsd = sparse_jensen_shannon_divergence(
            mzs,
            logprobs,
            batch_idxs,
            mzs,
            logprobs,
            batch_idxs,
            mz_max=200.0,
            mz_bin_res=1.0,
        )
        assert th.allclose(jsd, th.zeros(1), atol=1e-6), (
            "Identical spectra should have zero divergence"
        )

    def test_orthogonal_spectra(self):
        """Completely non-overlapping spectra should have normalized divergence = 1."""
        mzs1 = th.tensor([50.0])
        logprobs1 = th.tensor([0.0])
        batch_idxs1 = th.tensor([0], dtype=th.long)

        mzs2 = th.tensor([150.0])
        logprobs2 = th.tensor([0.0])
        batch_idxs2 = th.tensor([0], dtype=th.long)

        jsd = sparse_jensen_shannon_divergence(
            mzs1,
            logprobs1,
            batch_idxs1,
            mzs2,
            logprobs2,
            batch_idxs2,
            mz_max=200.0,
            mz_bin_res=1.0,
        )
        assert th.allclose(jsd, th.ones(1), atol=1e-6), (
            "Orthogonal spectra should have divergence = 1"
        )

    def test_known_partial_overlap(self):
        """Spectra sharing one of two equal-intensity peaks should have divergence = 0.5.

        P = [0.5, 0.5] at mz=50,100; Q = [0.5, 0.5] at mz=50,150.
        M = [0.5, 0.25, 0.25]; KL(P||M) = KL(Q||M) = 0.5*log2.
        JSD = 0.5*log2; normalized by log2 → 0.5.
        """
        mzs_true = th.tensor([50.0, 100.0])
        logprobs_true = th.tensor([0.0, 0.0])
        batch_idxs_true = th.tensor([0, 0], dtype=th.long)

        mzs_pred = th.tensor([50.0, 150.0])
        logprobs_pred = th.tensor([0.0, 0.0])
        batch_idxs_pred = th.tensor([0, 0], dtype=th.long)

        jsd = sparse_jensen_shannon_divergence(
            mzs_true,
            logprobs_true,
            batch_idxs_true,
            mzs_pred,
            logprobs_pred,
            batch_idxs_pred,
            mz_max=200.0,
            mz_bin_res=1.0,
        )
        assert th.allclose(jsd, th.tensor([0.5]), atol=1e-5), (
            f"Expected divergence 0.5, got {jsd.item():.6f}"
        )

    def test_non_negative_many_overlapping_bins(self):
        """Regression: JSS must stay non-negative with many overlapping bins.

        Before the renormalization fix in jss_helper, floating-point accumulation
        in scatter_reduce could cause union_bin_ints to sum slightly below 1,
        inflating the KL terms beyond log(2) and yielding a negative JSS.
        """
        N = 1000
        mzs = th.arange(1.0, N + 1.0)
        logprobs = th.full((N,), float(-math.log(N)))
        batch_idxs = th.zeros(N, dtype=th.long)

        jsd = sparse_jensen_shannon_divergence(
            mzs,
            logprobs,
            batch_idxs,
            mzs,
            logprobs,
            batch_idxs,
            mz_max=float(N + 1),
            mz_bin_res=1.0,
        )
        assert jsd.item() >= -1e-6, f"JSD must be non-negative, got {jsd.item()}"
        assert jsd.item() <= 1e-5, f"Identical spectra should have JSD ≈ 0, got {jsd.item()}"

    def test_symmetry(self):
        """Divergence should be symmetric: JSD(P, Q) == JSD(Q, P)."""
        mzs1 = th.tensor([50.0, 75.0, 100.0])
        logprobs1 = th.tensor([0.0, 1.0, 0.5])
        batch_idxs1 = th.tensor([0, 0, 0], dtype=th.long)

        mzs2 = th.tensor([100.0, 150.0])
        logprobs2 = th.tensor([0.5, 1.0])
        batch_idxs2 = th.tensor([0, 0], dtype=th.long)

        jsd_pq = sparse_jensen_shannon_divergence(
            mzs1,
            logprobs1,
            batch_idxs1,
            mzs2,
            logprobs2,
            batch_idxs2,
            mz_max=200.0,
            mz_bin_res=1.0,
        )
        jsd_qp = sparse_jensen_shannon_divergence(
            mzs2,
            logprobs2,
            batch_idxs2,
            mzs1,
            logprobs1,
            batch_idxs1,
            mz_max=200.0,
            mz_bin_res=1.0,
        )
        assert th.allclose(jsd_pq, jsd_qp, atol=1e-6), (
            f"JSD should be symmetric, got {jsd_pq.item():.6f} vs {jsd_qp.item():.6f}"
        )

    def test_batch_multiple_pairs(self):
        """Test JSD computation for a batch with multiple spectrum pairs.

        Pair 0: identical single-peak spectra → JSD = 0.
        Pair 1: orthogonal single-peak spectra → JSD = 1.
        """
        mzs_true = th.tensor([50.0, 100.0])
        logprobs_true = th.tensor([0.0, 0.0])
        batch_idxs_true = th.tensor([0, 1], dtype=th.long)

        mzs_pred = th.tensor([50.0, 200.0])
        logprobs_pred = th.tensor([0.0, 0.0])
        batch_idxs_pred = th.tensor([0, 1], dtype=th.long)

        jsd = sparse_jensen_shannon_divergence(
            mzs_true,
            logprobs_true,
            batch_idxs_true,
            mzs_pred,
            logprobs_pred,
            batch_idxs_pred,
            mz_max=300.0,
            mz_bin_res=1.0,
        )
        assert jsd.shape == (2,), f"Expected shape (2,), got {jsd.shape}"
        assert th.allclose(jsd[0], th.tensor(0.0), atol=1e-6), (
            f"Pair 0 (identical): expected JSD=0, got {jsd[0].item()}"
        )
        assert th.allclose(jsd[1], th.tensor(1.0), atol=1e-6), (
            f"Pair 1 (orthogonal): expected JSD=1, got {jsd[1].item()}"
        )


class TestPairwiseBinningGeometry:
    """Tests for configurable m/z grid geometry in pairwise losses."""

    @staticmethod
    def _simple_two_spectra() -> tuple[th.Tensor, th.Tensor, th.Tensor, int]:
        """Create two identical single-peak spectra in different batch slots."""
        mzs = th.tensor([100.0, 100.0])
        logprobs = th.tensor([0.0, 0.0])
        batch_idxs = th.tensor([0, 1], dtype=th.long)
        batch_size = 2
        return mzs, logprobs, batch_idxs, batch_size

    def test_pairwise_losses_support_non_default_resolution(self):
        """Pairwise helpers should work with a non-default valid bin geometry."""
        mzs, logprobs, batch_idxs, batch_size = self._simple_two_spectra()
        mz_max = 200.0
        mz_bin_res = 0.5  # 400 bins, valid integer grid

        cossim = get_pairwise_cossim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
        jss = get_pairwise_jss_sim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
        ce = get_pairwise_cross_entropy(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)

        assert th.allclose(cossim[0, 1], th.tensor(1.0), atol=1e-6)
        assert th.allclose(jss[0, 1], th.tensor(1.0), atol=1e-6)
        assert th.allclose(ce[0, 1], th.tensor(0.0), atol=1e-6)

    @pytest.mark.parametrize(
        "fn",
        [get_pairwise_cossim, get_pairwise_jss_sim, get_pairwise_cross_entropy],
    )
    def test_pairwise_losses_raise_on_non_integer_bin_geometry(self, fn):
        """Fail fast when mz_max / mz_bin_res does not define an integer bin count."""
        mzs, logprobs, batch_idxs, batch_size = self._simple_two_spectra()
        mz_max = 200.0
        mz_bin_res = 0.3  # 666.666..., invalid grid

        with pytest.raises(ValueError, match="mz_max / mz_bin_res"):
            fn(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)

    def test_batched_bin_func_validates_geometry(self):
        """Verify batched_bin_func validates bin geometry on invalid config."""
        from fragnnet.utils.spec_utils import batched_bin_func

        mzs = th.tensor([100.0, 150.0])
        ints = th.tensor([1.0, 2.0])
        batch_idxs = th.tensor([0, 1], dtype=th.long)

        # Valid geometry (400 bins) should work
        result = batched_bin_func(
            mzs, ints, batch_idxs, mz_max=250.0, mz_bin_res=0.5, agg="sum", sparse=True
        )
        assert result[0].numel() > 0  # Should return non-empty result

        # Invalid geometry (666.666... bins) should raise ValueError
        with pytest.raises(ValueError, match="mz_max / mz_bin_res"):
            batched_bin_func(
                mzs, ints, batch_idxs, mz_max=250.0, mz_bin_res=0.3, agg="sum", sparse=True
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
