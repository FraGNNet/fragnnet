"""Unit tests for the CEScaler class in fragnnet.model.base_model.

Tests:
- Identity at initialization (all-zero last layer → log_scale=0, bias=0 → passthrough).
- FiLM scaling with known manual weights: verify expected scale and bias applied.
- Missing CE values produce zero feature vector → identity scaling at init.
- Multiple CEs per sample aggregated to (mean, std, valid) before MLP.
- Integration: CEScaler output shape matches input shape.
- _ce_list_to_stats: single, ramped, stepped, and missing CE lists.
"""

import pytest
import torch as th

from fragnnet.dataset.base_dataset import _ce_list_to_stats
from fragnnet.model.base_model import CEScaler

NCE_MEAN = 60.0
NCE_STD = 40.0
OUTPUT_DIM = 5
HIDDEN_DIM = 16
BATCH_SIZE = 2
NUM_NODES = 6  # 3 nodes per sample


@pytest.fixture()
def scaler() -> CEScaler:
    """Minimal CEScaler with identity-initialized weights (NCE only)."""
    return CEScaler(
        nce_mean=NCE_MEAN,
        nce_std=NCE_STD,
        output_dim=OUTPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        use_nce=True,
        use_ace=False,
        ace_mean=0.0,
        ace_std=1.0,
    )


def _make_inputs(
    batch_size: int = BATCH_SIZE,
    num_nodes: int = NUM_NODES,
    output_dim: int = OUTPUT_DIM,
    ce_lists: list[th.Tensor] | None = None,
    device: th.device = th.device("cpu"),
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Return (logits, nce_stats, frag_node_batch_idxs) for tests.

    Args:
        batch_size: Number of samples.
        num_nodes: Total fragment nodes across all samples.
        output_dim: Logit output dimension.
        ce_lists: Per-sample CE tensors; defaults to [30.0] and [90.0].
        device: Target device.

    Returns:
        Tuple of (logits, nce_stats, frag_node_batch_idxs).
    """
    th.manual_seed(0)
    logits = th.randn(num_nodes, output_dim, device=device)
    if ce_lists is None:
        ce_lists = [th.tensor([30.0]), th.tensor([90.0])]
    nce_stats = th.cat([_ce_list_to_stats(t) for t in ce_lists], dim=0).to(device)
    nodes_per_sample = num_nodes // batch_size
    frag_node_batch_idxs = th.repeat_interleave(
        th.arange(batch_size, device=device),
        th.tensor([nodes_per_sample] * batch_size, device=device),
    )
    return logits, nce_stats, frag_node_batch_idxs


class TestCEListToStats:
    """Unit tests for _ce_list_to_stats."""

    def test_single_ce_std_zero(self) -> None:
        """Single CE produces std=0."""
        stats = _ce_list_to_stats(th.tensor([30.0]))
        assert stats.shape == (1, 3)
        assert stats[0, 1].item() == pytest.approx(0.0, abs=1e-6)
        assert stats[0, 2].item() == pytest.approx(1.0, abs=1e-6)

    def test_ramped_ce_std_positive(self) -> None:
        """Ramped CE produces std > 0."""
        stats = _ce_list_to_stats(th.arange(20.0, 41.0, 1.0))
        assert stats[0, 1].item() > 0.0
        assert stats[0, 2].item() == pytest.approx(1.0, abs=1e-6)

    def test_stepped_ce(self) -> None:
        """Stepped CE [lo, mid, hi] has correct mean and valid=1."""
        stats = _ce_list_to_stats(th.tensor([20.0, 30.0, 40.0]))
        assert stats[0, 0].item() == pytest.approx(30.0, abs=1e-5)
        assert stats[0, 2].item() == pytest.approx(1.0, abs=1e-6)

    def test_missing_ce(self) -> None:
        """Missing CE (-1 sentinel) gives all-zero stats."""
        stats = _ce_list_to_stats(th.tensor([-1.0]))
        assert th.allclose(stats, th.zeros(1, 3))

    def test_output_shape(self) -> None:
        """Output is always (1, 3)."""
        for t in [th.tensor([30.0]), th.arange(10.0, 50.0), th.tensor([-1.0])]:
            assert _ce_list_to_stats(t).shape == (1, 3)


class TestCEScalerIdentity:
    """Verify identity behavior at initialization (all-zero last layer)."""

    def test_identity_init(self, scaler: CEScaler) -> None:
        """Output equals input when last-layer weights and biases are zero."""
        logits, nce_stats, frag_node_batch_idxs = _make_inputs()
        with th.no_grad():
            out = scaler(logits, frag_node_batch_idxs, nce_stats=nce_stats)
        assert out.shape == logits.shape
        assert th.allclose(out, logits, atol=1e-6), f"Max diff: {(out - logits).abs().max()}"

    def test_output_shape(self, scaler: CEScaler) -> None:
        """Output shape matches (num_nodes, output_dim)."""
        logits, nce_stats, frag_node_batch_idxs = _make_inputs()
        with th.no_grad():
            out = scaler(logits, frag_node_batch_idxs, nce_stats=nce_stats)
        assert out.shape == (NUM_NODES, OUTPUT_DIM)


class TestCEScalerFilm:
    """Verify FiLM scaling with manually set network weights."""

    def test_known_scale(self, scaler: CEScaler) -> None:
        """Set last-layer bias so log_scale=0.5 for all dims; verify exp(0.5) scaling."""
        log_scale_val = 0.5
        with th.no_grad():
            scaler.net[-1].bias[:OUTPUT_DIM] = log_scale_val
            scaler.net[-1].bias[OUTPUT_DIM:] = 0.0

        logits, nce_stats, frag_node_batch_idxs = _make_inputs()
        with th.no_grad():
            out = scaler(logits, frag_node_batch_idxs, nce_stats=nce_stats)

        expected = logits * th.exp(th.tensor(log_scale_val))
        assert th.allclose(out, expected, atol=1e-5), f"Max diff: {(out - expected).abs().max()}"

    def test_known_bias(self, scaler: CEScaler) -> None:
        """Set last-layer bias so log_scale=0, bias=2.0; verify additive shift."""
        bias_val = 2.0
        with th.no_grad():
            scaler.net[-1].bias[:OUTPUT_DIM] = 0.0
            scaler.net[-1].bias[OUTPUT_DIM:] = bias_val

        logits, nce_stats, frag_node_batch_idxs = _make_inputs()
        with th.no_grad():
            out = scaler(logits, frag_node_batch_idxs, nce_stats=nce_stats)

        expected = logits + bias_val
        assert th.allclose(out, expected, atol=1e-5), f"Max diff: {(out - expected).abs().max()}"


class TestCEScalerMissingCE:
    """Verify that missing CE values produce identity scaling at init."""

    def test_missing_ce_identity(self, scaler: CEScaler) -> None:
        """When all CEs are -1, feature vector is zero → identity output at init."""
        logits, _, frag_node_batch_idxs = _make_inputs()
        missing_stats = th.zeros(BATCH_SIZE, 3)

        with th.no_grad():
            out = scaler(logits, frag_node_batch_idxs, nce_stats=missing_stats)

        assert th.allclose(out, logits, atol=1e-6), f"Max diff: {(out - logits).abs().max()}"

    def test_mixed_missing(self, scaler: CEScaler) -> None:
        """One sample valid, one missing; output shape is correct."""
        logits, _, frag_node_batch_idxs = _make_inputs()
        nce_stats = th.cat([
            _ce_list_to_stats(th.tensor([60.0])),
            _ce_list_to_stats(th.tensor([-1.0])),
        ], dim=0)

        with th.no_grad():
            out = scaler(logits, frag_node_batch_idxs, nce_stats=nce_stats)

        assert out.shape == logits.shape


class TestCEScalerMultiCE:
    """Verify (mean, std, valid) aggregation for multiple CEs per sample."""

    def test_ramped_ce_produces_nonzero_std(self, scaler: CEScaler) -> None:
        """Ramped CE list results in std > 0 in the stats; output shape correct."""
        logits, nce_stats, frag_node_batch_idxs = _make_inputs(
            ce_lists=[th.arange(20.0, 41.0, 1.0), th.arange(30.0, 61.0, 1.0)]
        )
        assert nce_stats[0, 1].item() > 0.0, "sample 0 std should be positive"
        assert nce_stats[1, 1].item() > 0.0, "sample 1 std should be positive"

        with th.no_grad():
            out = scaler(logits, frag_node_batch_idxs, nce_stats=nce_stats)
        assert out.shape == (NUM_NODES, OUTPUT_DIM)

    def test_two_identical_ces_same_as_one(self, scaler: CEScaler) -> None:
        """Two identical CEs give same mean and std=0 as a single CE at that value."""
        logits = th.ones(4, OUTPUT_DIM)
        frag_node_batch_idxs = th.tensor([0, 0, 1, 1])

        # Single CE at mean → normalized to 0 → identity at init
        nce_single = th.cat([
            _ce_list_to_stats(th.tensor([NCE_MEAN])),
            _ce_list_to_stats(th.tensor([NCE_MEAN])),
        ], dim=0)
        # Two CEs both at mean → same mean, std=0
        nce_double = th.cat([
            _ce_list_to_stats(th.tensor([NCE_MEAN, NCE_MEAN])),
            _ce_list_to_stats(th.tensor([NCE_MEAN, NCE_MEAN])),
        ], dim=0)

        with th.no_grad():
            out_single = scaler(logits, frag_node_batch_idxs, nce_stats=nce_single)
            out_double = scaler(logits, frag_node_batch_idxs, nce_stats=nce_double)

        assert th.allclose(out_single, out_double, atol=1e-6)
