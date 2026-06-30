from types import SimpleNamespace

import torch as th

from fragnnet.pl_model.spectrum_mol_clip_pl import SpectrumMolClipPL


def test_retrieval_metrics_perfect_alignment() -> None:
    embed = th.eye(4)

    metrics = SpectrumMolClipPL._retrieval_metrics(embed, embed, "s2m", top_k=5)

    assert th.isclose(metrics["s2m_top1"], th.tensor(1.0))
    assert th.isclose(metrics["s2m_top4"], th.tensor(1.0))
    assert th.isclose(metrics["s2m_mrr"], th.tensor(1.0))


def test_retrieval_metrics_detects_bad_ranking() -> None:
    query = th.eye(3)
    target = th.roll(th.eye(3), shifts=1, dims=0)

    metrics = SpectrumMolClipPL._retrieval_metrics(query, target, "s2m", top_k=2)

    assert metrics["s2m_top1"] < 1.0
    assert metrics["s2m_top2"] <= 1.0
    assert metrics["s2m_mrr"] < 1.0


def test_build_peak_features_normalizes_mz_and_intensity() -> None:
    module = SimpleNamespace(
        clip_intensity_transform="sqrt",
        hparams=SimpleNamespace(mz_max=1000.0),
    )
    batch = {
        "spec_mzs": th.tensor([100.0, 200.0, 50.0]),
        "spec_ints": th.tensor([1.0, 4.0, 9.0]),
        "spec_batch_idxs": th.tensor([0, 0, 1]),
        "batch_size": th.tensor(2),
    }

    features = SpectrumMolClipPL._build_peak_features(module, batch)

    expected = th.tensor(
        [
            [0.10, 0.5],
            [0.20, 1.0],
            [0.05, 1.0],
        ]
    )
    assert th.allclose(features, expected, atol=1e-6)
