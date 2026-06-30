"""Unit tests for MCES pretrained weight loading integration.

Tests for verifying MCES pretrained weights are correctly loaded into the model.
"""

import pytest
import torch as th


@pytest.mark.skip(reason="integration helper; requires real model")
def test_mces_weight_loading(model, mces_ckpt_path=None):
    """
    Test if MCES pretrained weights are loaded correctly.

    Args:
        model: FraGNNetPL model instance
        mces_ckpt_path: Path to MCES checkpoint (optional)
    """

    print("\n" + "=" * 70)
    print("Testing MCES Pretrained Weight Loading")
    print("=" * 70)

    # 1. Check if pretrained weights are enabled
    if not hasattr(model.hparams, "mces_pretrain"):
        print("❌ Model does not have mces_pretrain hyperparameter")
        return

    if not model.hparams.mces_pretrain:
        print("⚠️  MCES pretraining is disabled")
        return

    print("✅ MCES pretraining enabled")
    print(f"   Checkpoint: {model.hparams.mces_pretrain_ckpt_fp}")

    # 2. Check shared components exist
    shared_components = ["mol_embedder", "mol_pool"]
    for comp in shared_components:
        if not hasattr(model.model, comp):
            print(f"❌ {comp} NOT found in model")
        else:
            print(f"✅ {comp} found in model")

    # 3. Check if weights are frozen (if specified)
    if hasattr(model.hparams, "freeze_mces_weights"):
        if model.hparams.freeze_mces_weights:
            print("\n🔒 Freeze pretrained weights: ENABLED")
            frozen_count = sum(
                1 for p in model.model.mol_embedder.parameters() if not p.requires_grad
            )
            total_count = sum(1 for _ in model.model.mol_embedder.parameters())
            print(f"   Frozen params: {frozen_count}/{total_count}")
        else:
            print("\n🔓 Freeze pretrained weights: DISABLED")

    # 4. Check weight statistics (detect if still random)
    print("\n📊 Weight Statistics:")
    for comp in shared_components:
        if hasattr(model.model, comp):
            component = getattr(model.model, comp)
            params = list(component.parameters())
            if params:
                p = params[0]
                print(f"   {comp}:")
                print(f"     mean: {p.mean().item():.4f}, std: {p.std().item():.4f}")


def verify_weight_loading_in_training_step(pl_module, batch_idx):
    """
    Quick check during training to verify pretrained weights.

    Call this in your training_step:

    ```python
    def training_step(self, batch, batch_idx):
        if batch_idx == 0 and self.current_epoch == 0:
            verify_weight_loading_in_training_step(self, batch_idx)
        # ... rest of training step
    ```
    """

    if not hasattr(pl_module.hparams, "mces_pretrain"):
        print("⚠️  No mces_pretrain hyperparameter")
        return

    if not pl_module.hparams.mces_pretrain:
        print("ℹ️  MCES pretraining is disabled")
        return

    print(f"\n[Epoch {pl_module.current_epoch}, Batch {batch_idx}]")
    print("Checking pretrained weights...")

    # Check gradient flow
    mol_embedder = pl_module.model.mol_embedder
    first_param = next(mol_embedder.parameters())

    print("  mol_embedder.input_project.weight:")
    print(f"    requires_grad: {first_param.requires_grad}")
    print(f"    has grad: {first_param.grad is not None}")

    if pl_module.hparams.freeze_mces_weights and first_param.requires_grad:
        print("    ⚠️  Warning: freeze_mces_weights=True but requires_grad=True")


# ---------------------------------------------------------------------------
# Unit tests with lightweight dummy models
# ---------------------------------------------------------------------------


class _DummyParams:
    def __init__(self, mces_pretrain=True, freeze=False, ckpt_fp="ckpt.ckpt"):
        self.mces_pretrain = mces_pretrain
        self.freeze_mces_weights = freeze
        self.mces_pretrain_ckpt_fp = ckpt_fp


class _DummyInnerModel(th.nn.Module):
    def __init__(self, freeze: bool):
        super().__init__()
        self.mol_embedder = th.nn.Linear(10, 10)
        self.mol_pool = th.nn.Linear(10, 5)
        if freeze:
            for p in self.mol_embedder.parameters():
                p.requires_grad = False


class _DummyPLModule:
    def __init__(self, hparams: _DummyParams):
        self.hparams = hparams
        self.model = _DummyInnerModel(hparams.freeze_mces_weights)
        self.current_epoch = 0


def _run_helper_and_capture(model, capsys):
    # Use the helper directly; it returns None but prints diagnostics
    test_mces_weight_loading(model)
    captured = capsys.readouterr()
    return captured.out


def test_helper_skips_when_disabled(capsys):
    hparams = _DummyParams(mces_pretrain=False)
    model = _DummyPLModule(hparams)
    out = _run_helper_and_capture(model, capsys)
    assert "pretraining is disabled" in out


def test_helper_reports_missing_component(capsys):
    hparams = _DummyParams(mces_pretrain=True)
    model = _DummyPLModule(hparams)
    # Remove mol_pool to trigger missing component branch
    delattr(model.model, "mol_pool")
    out = _run_helper_and_capture(model, capsys)
    assert "mol_pool" in out and "NOT found" in out


def test_helper_freeze_outputs_counts(capsys):
    hparams = _DummyParams(mces_pretrain=True, freeze=True)
    model = _DummyPLModule(hparams)
    out = _run_helper_and_capture(model, capsys)
    assert "Freeze pretrained weights: ENABLED" in out
    assert "Frozen params" in out


def test_verify_weight_loading_prints(capsys):
    hparams = _DummyParams(mces_pretrain=True, freeze=True)
    pl_module = _DummyPLModule(hparams)
    verify_weight_loading_in_training_step(pl_module, batch_idx=0)
    out = capsys.readouterr().out
    assert "Checking pretrained weights" in out
    assert "requires_grad: False" in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
