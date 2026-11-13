"""Tests for AdductScaler and StratumScaler FiLM modules (§3a and §3b)."""

import pytest
import torch

from fragnnet.model.base_model import AdductScaler, StratumScaler

# ---------------------------------------------------------------------------
# AdductScaler
# ---------------------------------------------------------------------------


@pytest.fixture
def adduct_scaler():
    return AdductScaler(num_adduct_types=4, output_dim=5)


def test_adduct_scaler_identity_at_init(adduct_scaler):
    """Zero-init embedding means the layer is the identity at initialisation."""
    logits = torch.randn(10, 5)
    prec_type = torch.zeros(3, dtype=torch.long)
    node_batch_idxs = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    out = adduct_scaler(logits, prec_type, node_batch_idxs)
    assert torch.allclose(out, logits), "AdductScaler should be identity at init"


def test_adduct_scaler_output_shape(adduct_scaler):
    logits = torch.randn(12, 5)
    prec_type = torch.zeros(4, dtype=torch.long)
    node_batch_idxs = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    out = adduct_scaler(logits, prec_type, node_batch_idxs)
    assert out.shape == logits.shape


def test_adduct_scaler_differentiates_adducts():
    """After one gradient step, different adduct indices produce different outputs."""
    scaler = AdductScaler(num_adduct_types=4, output_dim=3)
    logits = torch.ones(4, 3)
    prec_type = torch.tensor([0, 1, 2, 3])
    node_batch_idxs = torch.arange(4)

    optimizer = torch.optim.SGD(scaler.parameters(), lr=1.0)
    target = torch.randn(4, 3)
    loss = (scaler(logits, prec_type, node_batch_idxs) - target).pow(2).sum()
    loss.backward()
    optimizer.step()

    out = scaler(logits, prec_type, node_batch_idxs)
    # All four adduct types should produce distinct outputs after training
    assert not torch.allclose(out[0], out[1]), "Adduct 0 and 1 should diverge after training"
    assert not torch.allclose(out[0], out[2]), "Adduct 0 and 2 should diverge after training"


def test_adduct_scaler_unknown_token():
    """Index == num_adduct_types is the unknown fallback and should work without error."""
    scaler = AdductScaler(num_adduct_types=4, output_dim=3)
    logits = torch.randn(2, 3)
    prec_type = torch.tensor([4, 4])  # unknown token
    node_batch_idxs = torch.tensor([0, 1])
    out = scaler(logits, prec_type, node_batch_idxs)
    assert out.shape == logits.shape


# ---------------------------------------------------------------------------
# StratumScaler
# ---------------------------------------------------------------------------


@pytest.fixture
def stratum_scaler():
    return StratumScaler(
        num_inst_types=3,
        num_frag_modes=2,
        num_adduct_types=4,
        embed_dim=8,
        output_dim=5,
    )


def test_stratum_scaler_identity_at_init(stratum_scaler):
    """Zero-init projection means the layer is the identity at initialisation."""
    logits = torch.randn(6, 5)
    inst_type = torch.zeros(3, dtype=torch.long)
    frag_mode = torch.zeros(3, dtype=torch.long)
    prec_type = torch.zeros(3, dtype=torch.long)
    node_batch_idxs = torch.tensor([0, 0, 1, 1, 2, 2])
    out = stratum_scaler(logits, inst_type, frag_mode, prec_type, node_batch_idxs)
    assert torch.allclose(out, logits), "StratumScaler should be identity at init"


def test_stratum_scaler_output_shape(stratum_scaler):
    logits = torch.randn(9, 5)
    inst_type = torch.zeros(3, dtype=torch.long)
    frag_mode = torch.zeros(3, dtype=torch.long)
    prec_type = torch.zeros(3, dtype=torch.long)
    node_batch_idxs = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    out = stratum_scaler(logits, inst_type, frag_mode, prec_type, node_batch_idxs)
    assert out.shape == logits.shape


def test_stratum_scaler_differentiates_strata():
    """After one gradient step, different strata produce different outputs."""
    scaler = StratumScaler(
        num_inst_types=2, num_frag_modes=2, num_adduct_types=2, embed_dim=4, output_dim=3
    )
    logits = torch.ones(4, 3)
    inst_type = torch.tensor([0, 0, 1, 1])
    frag_mode = torch.tensor([0, 1, 0, 1])
    prec_type = torch.zeros(4, dtype=torch.long)
    node_batch_idxs = torch.arange(4)

    optimizer = torch.optim.SGD(scaler.parameters(), lr=1.0)
    target = torch.randn(4, 3)
    loss = (scaler(logits, inst_type, frag_mode, prec_type, node_batch_idxs) - target).pow(2).sum()
    loss.backward()
    optimizer.step()

    out = scaler(logits, inst_type, frag_mode, prec_type, node_batch_idxs)
    assert not torch.allclose(out[0], out[2]), "Different inst_type should produce different output"
