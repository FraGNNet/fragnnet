"""Unit tests for FragModeModel embedding and FragModeScaler output scaling.

Tests verify:
- FragModeModel embedding produces correct shapes for known, unknown, and missing modes.
- FragModeModel location dimensions are set correctly.
- FragModeScaler starts as the identity transform (zero-initialised weights).
- FragModeScaler applies learned scale and bias after training on a trivial signal.
- Edge cases: batch of size 1, unknown-only batch, all-same mode.
"""

import pytest
import torch as th
import torch.nn as nn

from fragnnet.model.base_model import FragModeModel, FragModeScaler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ConcreteFragModeModel(nn.Module, FragModeModel):
    """Minimal concrete subclass that satisfies the abstract location check."""

    def _frag_mode_location_check(self) -> None:
        assert self.frag_mode_insert_location in {"none", "mol", "mlp"}


def make_model(location: str = "mlp", embed_size: int = 8, num_modes: int = 2):
    model = _ConcreteFragModeModel()
    model._frag_mode_init(
        frag_mode_insert_location=location,
        frag_mode_insert_size=embed_size,
        frag_mode_num_types=num_modes,
    )
    return model


# ---------------------------------------------------------------------------
# FragModeModel tests
# ---------------------------------------------------------------------------


class TestFragModeModelInit:
    def test_mlp_location_dims(self):
        model = make_model(location="mlp", embed_size=8)
        assert model.frag_mode_mlp_input_dim == 8
        assert model.frag_mode_mol_input_dim == 0

    def test_mol_location_dims(self):
        model = make_model(location="mol", embed_size=16)
        assert model.frag_mode_mol_input_dim == 16
        assert model.frag_mode_mlp_input_dim == 0

    def test_none_location_dims(self):
        model = make_model(location="none", embed_size=8)
        assert model.frag_mode_mol_input_dim == 0
        assert model.frag_mode_mlp_input_dim == 0

    def test_embedding_table_size(self):
        # num_modes + 1 rows (extra row for the unknown token)
        model = make_model(num_modes=2, embed_size=8)
        weight = model.frag_mode_embedder.weight
        assert weight.shape == (3, 8)  # HCD=0, CID=1, unknown=2


class TestFragModeModelEmbed:
    def test_embed_returns_none_for_none_location(self):
        model = make_model(location="none")
        result = model.embed_frag_mode(None)
        assert result is None

    def test_embed_known_mode(self):
        model = make_model(location="mlp", embed_size=8, num_modes=2)
        frag_mode = th.tensor([0, 1, 0])  # HCD, CID, HCD
        embed = model.embed_frag_mode(frag_mode)
        assert embed is not None
        assert embed.shape == (3, 8)

    def test_embed_unknown_mode(self):
        """Index num_modes is the unknown token; must not raise."""
        model = make_model(location="mlp", embed_size=8, num_modes=2)
        frag_mode = th.tensor([2])  # unknown token index
        embed = model.embed_frag_mode(frag_mode)
        assert embed is not None
        assert embed.shape == (1, 8)

    def test_embed_raises_when_none_and_location_set(self):
        model = make_model(location="mlp")
        with pytest.raises(ValueError, match="frag_mode is None"):
            model.embed_frag_mode(None)

    def test_embed_single_sample(self):
        model = make_model(location="mlp", embed_size=4, num_modes=2)
        frag_mode = th.tensor([1])
        embed = model.embed_frag_mode(frag_mode)
        assert embed.shape == (1, 4)


# ---------------------------------------------------------------------------
# FragModeScaler tests
# ---------------------------------------------------------------------------


class TestFragModeScalerInit:
    def test_identity_at_init(self):
        """With zero-initialised weights the scaler is the identity transform."""
        scaler = FragModeScaler(num_frag_modes=2, output_dim=5)
        # All weights should be zero → scale=exp(0)=1, bias=0 → output == input
        assert th.all(scaler.params.weight == 0.0)

    def test_weight_shape(self):
        scaler = FragModeScaler(num_frag_modes=2, output_dim=5)
        # (num_modes + 1) rows, output_dim * 2 cols
        assert scaler.params.weight.shape == (3, 10)


class TestFragModeScalerForward:
    def _make_batch(self, num_nodes=6, output_dim=5, batch_size=2, num_modes=2):
        """Return (logits, frag_mode, node_batch_idxs) for a synthetic batch."""
        logits = th.randn(num_nodes, output_dim)
        # Alternate modes per sample: sample0=HCD, sample1=CID
        frag_mode = th.tensor([0, 1])  # shape (batch_size,)
        # 3 nodes per sample
        node_batch_idxs = th.tensor([0, 0, 0, 1, 1, 1])
        return logits, frag_mode, node_batch_idxs

    def test_identity_at_init(self):
        """Zero-init weights → scaler output equals input."""
        scaler = FragModeScaler(num_frag_modes=2, output_dim=5)
        logits, frag_mode, node_batch_idxs = self._make_batch(output_dim=5)
        out = scaler(logits, frag_mode, node_batch_idxs)
        assert th.allclose(out, logits, atol=1e-6), "Scaler with zero weights must be identity"

    def test_output_shape(self):
        scaler = FragModeScaler(num_frag_modes=2, output_dim=5)
        logits, frag_mode, node_batch_idxs = self._make_batch(output_dim=5)
        out = scaler(logits, frag_mode, node_batch_idxs)
        assert out.shape == logits.shape

    def test_different_modes_produce_different_outputs_after_training(self):
        """After one gradient step the two modes should produce different scalings."""
        scaler = FragModeScaler(num_frag_modes=2, output_dim=3)
        # Only update via gradient on a trivial loss
        opt = th.optim.SGD(scaler.parameters(), lr=1.0)
        logits = th.ones(4, 3)
        frag_mode = th.tensor([0, 1])  # 2 samples
        node_batch_idxs = th.tensor([0, 0, 1, 1])  # 2 nodes per sample

        # target: first sample logits should be 2× the input, second sample unchanged
        target = th.cat([th.full((2, 3), 2.0), th.ones(2, 3)], dim=0)
        loss = ((scaler(logits, frag_mode, node_batch_idxs) - target) ** 2).mean()
        loss.backward()
        opt.step()

        # After the step, mode 0 and mode 1 embeddings must differ
        w = scaler.params.weight
        assert not th.allclose(w[0], w[1]), "Mode embeddings should diverge after a gradient step"

    def test_unknown_mode_token(self):
        """Unknown mode (index == num_modes) must be usable without error."""
        scaler = FragModeScaler(num_frag_modes=2, output_dim=4)
        logits = th.randn(3, 4)
        frag_mode = th.tensor([2])  # unknown token
        node_batch_idxs = th.tensor([0, 0, 0])
        out = scaler(logits, frag_mode, node_batch_idxs)
        assert out.shape == (3, 4)

    def test_single_node_single_sample(self):
        scaler = FragModeScaler(num_frag_modes=2, output_dim=9)
        logits = th.randn(1, 9)
        frag_mode = th.tensor([0])
        node_batch_idxs = th.tensor([0])
        out = scaler(logits, frag_mode, node_batch_idxs)
        assert out.shape == (1, 9)
