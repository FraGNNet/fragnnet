"""Tests for CLIP loss variants: clip_loss, siglip_loss, soft_clip_loss."""

import pytest
import torch as th
import torch.nn.functional as F

from fragnnet.model.loss import clip_loss, siglip_loss, soft_clip_loss


def _unit(t: th.Tensor) -> th.Tensor:
    return F.normalize(t, dim=-1)


# ---------------------------------------------------------------------------
# clip_loss
# ---------------------------------------------------------------------------


class TestClipLoss:
    def test_perfect_alignment_beats_random(self):
        th.manual_seed(0)
        embed = _unit(th.randn(8, 32))
        loss_aligned = clip_loss(embed, embed, temperature=0.07)
        loss_random = clip_loss(embed, _unit(th.randn(8, 32)), temperature=0.07)
        assert loss_aligned < loss_random

    def test_symmetric(self):
        th.manual_seed(1)
        a = _unit(th.randn(6, 16))
        b = _unit(th.randn(6, 16))
        assert th.isclose(clip_loss(a, b), clip_loss(b, a))

    def test_scalar_output(self):
        a = _unit(th.randn(4, 8))
        assert clip_loss(a, a).shape == ()

    def test_known_value_identity_temp1(self):
        # With orthonormal embeddings and temp=1:
        # logits = I → softmax row i has exp(1)/(exp(1) + N-1) for correct class.
        # Expected loss = -log(e / (e + N-1)) for N=4.
        N = 4
        embed = th.eye(N)
        expected = -th.log(th.tensor(th.e / (th.e + N - 1)))
        assert th.isclose(clip_loss(embed, embed, temperature=1.0), expected, atol=1e-5)


# ---------------------------------------------------------------------------
# siglip_loss
# ---------------------------------------------------------------------------


class TestSigLIPLoss:
    def test_scalar_output(self):
        a = _unit(th.randn(4, 8))
        assert siglip_loss(a, a).shape == ()

    def test_perfect_alignment_beats_random(self):
        th.manual_seed(2)
        embed = _unit(th.randn(8, 32))
        loss_aligned = siglip_loss(embed, embed, temperature=0.07)
        loss_random = siglip_loss(embed, _unit(th.randn(8, 32)), temperature=0.07)
        assert loss_aligned < loss_random

    def test_symmetric(self):
        # Swapping spec/mol transposes logits; since label matrix is also symmetric,
        # the mean loss is unchanged.
        th.manual_seed(3)
        a = _unit(th.randn(6, 16))
        b = _unit(th.randn(6, 16))
        assert th.isclose(siglip_loss(a, b), siglip_loss(b, a))

    def test_known_value_identity_temp1(self):
        # With orthonormal embeddings and temp=1:
        # positives (diagonal): logit=1  → -log_sigmoid(1)
        # negatives (off-diag): logit=0  → -log_sigmoid(-0) = -log(0.5) = log2
        # loss = (N*(-log_sigmoid(1)) + N*(N-1)*log2) / N^2
        N = 4
        embed = th.eye(N)
        pos_term = -F.logsigmoid(th.tensor(1.0))
        neg_term = -F.logsigmoid(th.tensor(-0.0))
        expected = (N * pos_term + N * (N - 1) * neg_term) / (N * N)
        assert th.isclose(siglip_loss(embed, embed, temperature=1.0), expected, atol=1e-5)

    def test_gradient_flows(self):
        a = _unit(th.randn(4, 8, requires_grad=False))
        b = th.randn(4, 8, requires_grad=True)
        b_norm = F.normalize(b, dim=-1)
        loss = siglip_loss(a, b_norm, temperature=0.07)
        loss.backward()
        assert b.grad is not None
        assert not th.all(b.grad == 0)


# ---------------------------------------------------------------------------
# soft_clip_loss
# ---------------------------------------------------------------------------


class TestSoftClipLoss:
    def test_scalar_output(self):
        a = _unit(th.randn(4, 8))
        fps = (th.rand(4, 64) > 0.5).float()
        assert soft_clip_loss(a, a, fps).shape == ()

    def test_identity_fps_matches_clip_loss(self):
        # Non-overlapping single-bit fps → Tanimoto matrix = identity →
        # soft targets = one-hot → soft_clip_loss reduces exactly to clip_loss.
        th.manual_seed(4)
        N = 6
        spec = _unit(th.randn(N, 16))
        mol = _unit(th.randn(N, 16))
        # each molecule has exactly one unique bit set
        fps = th.eye(N)

        loss_soft = soft_clip_loss(spec, mol, fps, temperature=0.07)
        loss_clip = clip_loss(spec, mol, temperature=0.07)
        assert th.isclose(loss_soft, loss_clip, atol=1e-5), (
            f"Expected soft_clip≈clip_loss with identity fps, got {loss_soft:.6f} vs {loss_clip:.6f}"
        )

    def test_duplicate_fps_reduces_penalty_for_misaligned_embeddings(self):
        # Two molecules with identical fps (Tanimoto=1 for all pairs) →
        # targets are uniform → cross-entropy is lower than standard CLIP which
        # treats the off-diagonal as a hard negative even when mols are identical.
        # Embeddings are swapped (worst case for CLIP), so standard CLIP loss is high.
        th.manual_seed(5)
        N = 4
        mol = _unit(th.randn(N, 16))
        spec = mol.flip(0)  # perfectly misaligned with mol
        fps_identical = th.ones(N, 8)  # all identical → Tanimoto=1 everywhere

        loss_clip = clip_loss(spec, mol, temperature=1.0)
        loss_soft = soft_clip_loss(spec, mol, fps_identical, temperature=1.0)
        assert loss_soft < loss_clip

    def test_tanimoto_diagonal_is_one(self):
        # Each molecule's Tanimoto with itself must be 1.
        th.manual_seed(6)
        N = 5
        fps = (th.rand(N, 128) > 0.6).float()
        spec = _unit(th.randn(N, 16))
        mol = _unit(th.randn(N, 16))

        # Patch to expose the internal tanimoto matrix via known fps.
        dots = fps @ fps.t()
        norms = fps.sum(dim=1)
        union = norms[:, None] + norms[None, :] - dots
        tanimoto = dots / union.clamp_min(1.0)

        assert th.allclose(tanimoto.diagonal(), th.ones(N), atol=1e-6)
        assert (tanimoto >= 0).all()
        assert (tanimoto <= 1 + 1e-6).all()

    def test_perfect_alignment_beats_random(self):
        # Sparse non-overlapping fps → Tanimoto ≈ identity → soft target ≈ one-hot.
        # Under near-identity targets aligned embeddings must win, same as clip_loss.
        th.manual_seed(7)
        N, D = 8, 32
        embed = _unit(th.randn(N, D))
        # Two unique bits per molecule, no overlap → Tanimoto off-diagonal = 0
        fps = th.zeros(N, N * 4)
        for i in range(N):
            fps[i, i * 4 : i * 4 + 2] = 1.0

        loss_aligned = soft_clip_loss(embed, embed, fps, temperature=0.07)
        loss_random = soft_clip_loss(embed, _unit(th.randn(N, D)), fps, temperature=0.07)
        assert loss_aligned < loss_random

    def test_gradient_flows(self):
        fps = (th.rand(4, 64) > 0.5).float()
        a = _unit(th.randn(4, 8, requires_grad=False))
        b = th.randn(4, 8, requires_grad=True)
        b_norm = F.normalize(b, dim=-1)
        loss = soft_clip_loss(a, b_norm, fps, temperature=0.07)
        loss.backward()
        assert b.grad is not None
        assert not th.all(b.grad == 0)
