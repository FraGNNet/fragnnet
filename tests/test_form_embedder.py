import torch as th

from fragnnet.model.form_embedder import LearnedFeaturizer, OneHotFeaturizer


def _reference_int_featurizer_forward(featurizer, tensor: th.Tensor) -> th.Tensor:
    tensor = tensor.long()
    orig_shape = tensor.shape

    norm_mask = (tensor >= 0) & (tensor < featurizer.max_count_int)
    if featurizer.missing_token is None:
        missing_mask = th.zeros_like(tensor, dtype=th.bool)
    else:
        missing_mask = tensor == featurizer.missing_token
    extra_mask = (~norm_mask) & (~missing_mask)

    norm_indices = tensor.clone()
    norm_indices[~norm_mask] = 0
    norm_embeds = featurizer.int_to_feat_matrix[norm_indices]

    missing_embeds = featurizer._extra_embeddings[0].expand_as(norm_embeds)

    extra_token_indices = th.clamp(tensor - featurizer.max_count_int, min=0)
    extra_token_indices = th.clamp(extra_token_indices, max=featurizer.num_extra_embeddings - 2) + 1
    extra_embeds = featurizer._extra_embeddings[extra_token_indices]
    extra_embeds[~extra_mask] = 0

    out = (
        norm_mask.unsqueeze(-1) * norm_embeds
        + missing_mask.unsqueeze(-1) * missing_embeds
        + extra_mask.unsqueeze(-1) * extra_embeds
    )
    return out.reshape(*orig_shape[:-1], -1)


def test_int_featurizer_matches_reference_with_missing_and_extra_tokens():
    featurizer = LearnedFeaturizer(
        feature_dim=3,
        max_count_int=5,
        num_extra_embeddings=3,
        missing_token=-99,
    )
    with th.no_grad():
        featurizer.int_to_feat_matrix.copy_(th.arange(15, dtype=th.float32).reshape(5, 3))
        featurizer._extra_embeddings.copy_(
            th.tensor(
                [
                    [100.0, 101.0, 102.0],
                    [200.0, 201.0, 202.0],
                    [300.0, 301.0, 302.0],
                ]
            )
        )

    tensor = th.tensor([[0, 2, -99, 5, 7, -1]])
    expected = _reference_int_featurizer_forward(featurizer, tensor)
    actual = featurizer(tensor)

    assert th.equal(actual, expected)


def test_int_featurizer_matches_reference_when_only_one_extra_embedding_exists():
    featurizer = OneHotFeaturizer(max_count_int=4, num_extra_embeddings=1, missing_token=None)
    tensor = th.tensor([[0, 3, 4, 9, -3]])

    expected = _reference_int_featurizer_forward(featurizer, tensor)
    actual = featurizer(tensor)

    assert th.equal(actual, expected)


def test_int_featurizer_preserves_gradients_for_learned_and_extra_embeddings():
    featurizer = LearnedFeaturizer(
        feature_dim=2,
        max_count_int=4,
        num_extra_embeddings=3,
        missing_token=-7,
    )
    tensor = th.tensor([[0, 1, -7, 4, 6]], dtype=th.long)

    out = featurizer(tensor)
    loss = out.sum()
    loss.backward()

    assert featurizer.int_to_feat_matrix.grad is not None
    assert featurizer._extra_embeddings.grad is not None
    assert th.all(th.isfinite(featurizer.int_to_feat_matrix.grad))
    assert th.all(th.isfinite(featurizer._extra_embeddings.grad))
    assert featurizer.int_to_feat_matrix.grad[0].abs().sum() > 0
    assert featurizer.int_to_feat_matrix.grad[1].abs().sum() > 0
    assert featurizer._extra_embeddings.grad[0].abs().sum() > 0
    assert featurizer._extra_embeddings.grad[1].abs().sum() > 0
    assert featurizer._extra_embeddings.grad[2].abs().sum() > 0
