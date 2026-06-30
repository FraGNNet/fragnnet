"""
land of feature embedders
"""

import logging
from typing import Any

import numpy as np
import torch as th
import torch.nn as nn

# the maximum number of integers that we are going to see as a "count", i.e. 0 to MAX_COUNT_INT-1
# Number of extra embeddings to learn -- one for the "to be confirmed" embedding.

DEFAULT_MAX_COUNT_INT = 255
DEFAULT_NUM_EXTRA_EMBEDDINGS = 1

logger = logging.getLogger(__name__)


class IntFeaturizer(nn.Module):
    """
    Base class for mapping integers to a vector representation (primarily to be used as a "richer" embedding for NNs
    processing integers).

    Subclasses should define `self.int_to_feat_matrix`, a matrix where each row is the vector representation for that
    integer, i.e. to get a vector representation for `5`, one could call `self.int_to_feat_matrix[5]`.

    This base class also handles a fixed number of "extra" learned embeddings. The first one is reserved for missing data,
    and the others can represent other out-of-range or special tokens.
    NOTE: All Classes inheriting from this class with treat Negative values and values >= max_count_int as "extra" tokens.
    """

    def __init__(
        self,
        embedding_dim: int,
        max_count_int: int = DEFAULT_MAX_COUNT_INT,
        num_extra_embeddings: int = DEFAULT_NUM_EXTRA_EMBEDDINGS,
        missing_token: int | None = None,
    ) -> None:
        super().__init__()
        self.max_count_int = max_count_int
        self.num_extra_embeddings = num_extra_embeddings
        self.missing_token = missing_token
        self.embedding_dim = embedding_dim

        weights = th.zeros(self.num_extra_embeddings, embedding_dim)
        self._extra_embeddings = nn.Parameter(weights, requires_grad=True)
        nn.init.normal_(self._extra_embeddings, 0.0, 1.0)
        logger.warning(
            "IntFeaturizer: Negative values and values >= max_count_int will be treated as 'extra' tokens."
        )

    def forward(self, tensor: th.Tensor) -> th.Tensor:
        """
        Convert the integer `tensor` into its new representation.
        - Use `int_to_feat_matrix` if 0 <= x < max_count_int
        - Use _extra_embeddings[0] if x == missing_token
        - Use _extra_embeddings[i+1] if x >= max_count_int and x != missing_token
        """
        tensor = tensor.long()
        orig_shape = tensor.shape
        flat_tensor = tensor.reshape(-1)
        out_tensor = th.empty(
            (flat_tensor.shape[0], self.embedding_dim),
            dtype=self.int_to_feat_matrix.dtype,
            device=flat_tensor.device,
        )

        # all negative values and values >= max_count_int will be treated as "extra" tokens.
        # TODO: may be we should default to negtive values being missing tokens?
        norm_mask = (flat_tensor >= 0) & (flat_tensor < self.max_count_int)

        if self.missing_token is None:
            missing_mask = th.zeros_like(flat_tensor, dtype=th.bool)
        else:
            missing_mask = flat_tensor == self.missing_token

        extra_mask = (~norm_mask) & (~missing_mask)

        if th.any(norm_mask):
            out_tensor[norm_mask] = self.int_to_feat_matrix[flat_tensor[norm_mask]]
        if th.any(missing_mask):
            out_tensor[missing_mask] = self._extra_embeddings[0]
        if th.any(extra_mask):
            extra_token_indices = th.clamp(flat_tensor[extra_mask] - self.max_count_int, min=0)
            extra_token_indices = (
                th.clamp(extra_token_indices, max=self.num_extra_embeddings - 2) + 1
            )
            out_tensor[extra_mask] = self._extra_embeddings[extra_token_indices]

        return out_tensor.reshape(*orig_shape[:-1], -1)

    @property
    def num_dim(self) -> int:
        return self.int_to_feat_matrix.shape[1]


class FourierFeaturizer(IntFeaturizer):
    """
    Inspired by:
    Tancik, M., Srinivasan, P.P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R.,
    Barron, J.T. and Ng, R. (2020) 'Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional
     Domains', arXiv [cs.CV]. Available at: http://arxiv.org/abs/2006.10739.

    Some notes:
    * we'll put the frequencies at powers of 1/2 rather than random Gaussian samples; this means it will match the
        Binarizer quite closely but be a bit smoother.
    """

    def __init__(
        self,
        max_count_int: int = DEFAULT_MAX_COUNT_INT,
        num_extra_embeddings: int = DEFAULT_NUM_EXTRA_EMBEDDINGS,
        missing_token: int | None = None,
    ) -> None:
        num_freqs = int(np.ceil(np.log2(max_count_int))) + 2
        # ^ need at least this many to ensure that the whole input range can be represented on the half circle.

        freqs = 0.5 ** th.arange(num_freqs, dtype=th.float32)
        freqs_time_2pi = 2 * np.pi * freqs

        super().__init__(
            embedding_dim=2 * freqs_time_2pi.shape[0],
            max_count_int=max_count_int,
            num_extra_embeddings=num_extra_embeddings,
            missing_token=missing_token,
        )  # 2 for cosine and sine

        # we will define the features at this frequency up front (as we only will ever see a fixed number of counts):
        combo_of_sinusoid_args = (
            th.arange(self.max_count_int, dtype=th.float32)[:, None] * freqs_time_2pi[None, :]
        )
        all_features = th.cat(
            [th.cos(combo_of_sinusoid_args), th.sin(combo_of_sinusoid_args)],
            dim=1,
        )

        # ^ shape:  MAX_COUNT_INT x 2 * num_freqs
        self.int_to_feat_matrix = nn.Parameter(all_features.float())
        self.int_to_feat_matrix.requires_grad = False


class FourierFeaturizerSines(IntFeaturizer):
    """
    Like other fourier feats but sines only

    Inspired by:
    Tancik, M., Srinivasan, P.P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R.,
    Barron, J.T. and Ng, R. (2020) Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional
     Domains, arXiv [cs.CV]. Available at: http://arxiv.org/abs/2006.10739.

    Some notes:
    * we'll put the frequencies at powers of 1/2 rather than random Gaussian samples; this means it will match the
        Binarizer quite closely but be a bit smoother.
    """

    def __init__(
        self,
        max_count_int: int = DEFAULT_MAX_COUNT_INT,
        num_extra_embeddings: int = DEFAULT_NUM_EXTRA_EMBEDDINGS,
        missing_token: int | None = None,
    ) -> None:
        num_freqs = int(np.ceil(np.log2(max_count_int))) + 2
        # ^ need at least this many to ensure that the whole input range can be represented on the half circle.

        freqs = (0.5 ** th.arange(num_freqs, dtype=th.float32))[2:]
        freqs_time_2pi = 2 * np.pi * freqs

        super().__init__(
            embedding_dim=freqs_time_2pi.shape[0],
            max_count_int=max_count_int,
            num_extra_embeddings=num_extra_embeddings,
            missing_token=missing_token,
        )

        # we will define the features at this frequency up front (as we only will ever see a fixed number of counts):
        combo_of_sinusoid_args = (
            th.arange(self.max_count_int, dtype=th.float32)[:, None] * freqs_time_2pi[None, :]
        )
        # ^ shape:  MAX_COUNT_INT x 2 * num_freqs
        self.int_to_feat_matrix = nn.Parameter(th.sin(combo_of_sinusoid_args).float())
        self.int_to_feat_matrix.requires_grad = False


class FourierFeaturizerAbsoluteSines(IntFeaturizer):
    """
    Like other fourier feats but sines only and absoluted.

    Inspired by:
    Tancik, M., Srinivasan, P.P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R.,
    Barron, J.T. and Ng, R. (2020) Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional
     Domains, arXiv [cs.CV]. Available at: http://arxiv.org/abs/2006.10739.

    Some notes:
    * we'll put the frequencies at powers of 1/2 rather than random Gaussian samples; this means it will match the
        Binarizer quite closely but be a bit smoother.
    """

    def __init__(
        self,
        max_count_int: int = DEFAULT_MAX_COUNT_INT,
        num_extra_embeddings: int = DEFAULT_NUM_EXTRA_EMBEDDINGS,
        missing_token: int | None = None,
    ) -> None:
        num_freqs = int(np.ceil(np.log2(max_count_int))) + 2

        freqs = (0.5 ** th.arange(num_freqs, dtype=th.float32))[2:]
        freqs_time_2pi = 2 * np.pi * freqs

        super().__init__(
            embedding_dim=freqs_time_2pi.shape[0],
            max_count_int=max_count_int,
            num_extra_embeddings=num_extra_embeddings,
            missing_token=missing_token,
        )

        # we will define the features at this frequency up front (as we only will ever see a fixed number of counts):
        combo_of_sinusoid_args = (
            th.arange(self.max_count_int, dtype=th.float32)[:, None] * freqs_time_2pi[None, :]
        )
        # ^ shape:  MAX_COUNT_INT x 2 * num_freqs
        self.int_to_feat_matrix = nn.Parameter(th.abs(th.sin(combo_of_sinusoid_args)).float())
        self.int_to_feat_matrix.requires_grad = False


class RBFFeaturizer(IntFeaturizer):
    """
    A featurizer that puts radial basis functions evenly between 0 and max_count-1. These will have a width of
    (max_count-1) / (num_funcs) to decay to about 0.6 of its original height at reaching the next func.

    """

    def __init__(
        self,
        num_funcs: int = 32,
        max_count_int: int = DEFAULT_MAX_COUNT_INT,
        num_extra_embeddings: int = DEFAULT_NUM_EXTRA_EMBEDDINGS,
        missing_token: int | None = None,
    ) -> None:
        """
        :param num_funcs: number of radial basis functions to use: their width will automatically be chosen -- see class
                            docstring.
        """
        super().__init__(
            embedding_dim=num_funcs,
            max_count_int=max_count_int,
            num_extra_embeddings=num_extra_embeddings,
            missing_token=missing_token,
        )
        width = (self.max_count_int - 1) / num_funcs
        centers = th.linspace(0, self.max_count_int - 1, num_funcs)

        pre_exponential_terms = (
            -0.5 * ((th.arange(self.max_count_int)[:, None] - centers[None, :]) / width) ** 2
        )
        # ^ shape: MAX_COUNT_INT x num_funcs
        feats = th.exp(pre_exponential_terms)

        self.int_to_feat_matrix = nn.Parameter(feats.float())
        self.int_to_feat_matrix.requires_grad = False


class OneHotFeaturizer(IntFeaturizer):
    """
    A featurizer that turns integers into their one hot encoding.

    Represents:
     - 0 as 1000000000...
     - 1 as 0100000000...
     - 2 as 0010000000...
     and so on.
    """

    def __init__(
        self,
        max_count_int: int = DEFAULT_MAX_COUNT_INT,
        num_extra_embeddings: int = DEFAULT_NUM_EXTRA_EMBEDDINGS,
        missing_token: int | None = None,
    ) -> None:
        super().__init__(
            embedding_dim=max_count_int,
            max_count_int=max_count_int,
            num_extra_embeddings=num_extra_embeddings,
            missing_token=missing_token,
        )
        feats = th.eye(self.max_count_int)
        self.int_to_feat_matrix = nn.Parameter(feats.float())
        self.int_to_feat_matrix.requires_grad = False


class LearnedFeaturizer(IntFeaturizer):
    """
    Learns the features for the different integers.

    Pretty much `nn.Embedding` but we get to use the forward of the superclass which behaves a bit differently.
    """

    def __init__(
        self,
        feature_dim: int = 32,
        max_count_int: int = DEFAULT_MAX_COUNT_INT,
        num_extra_embeddings: int = DEFAULT_NUM_EXTRA_EMBEDDINGS,
        missing_token: int | None = None,
    ) -> None:
        super().__init__(
            embedding_dim=feature_dim,
            max_count_int=max_count_int,
            num_extra_embeddings=num_extra_embeddings,
            missing_token=missing_token,
        )
        weights = th.zeros(self.max_count_int, feature_dim)
        self.int_to_feat_matrix = nn.Parameter(weights, requires_grad=True)
        nn.init.normal_(self.int_to_feat_matrix, 0.0, 1.0)


# class FloatFeaturizer(IntFeaturizer):
#    """
#    Norms the features
#    """

#    def __init__(self):
#        # Norm vec
#        # Placeholder..
#        super().__init__(embedding_dim=1)
#        self.norm_vec = th.from_numpy(common.NORM_VEC).float()
#        self.norm_vec = nn.Parameter(self.norm_vec)
#        self.norm_vec.requires_grad = False

#    def forward(self, tensor):
#        """
#        Convert the integer `tensor` into its new representation -- note that it gets stacked along final dimension.
#        """
#        tens_shape = tensor.shape
#        out_shape = [1] * (len(tens_shape) - 1) + [-1]
#        return tensor / self.norm_vec.reshape(*out_shape)

#    @property
#    def num_dim(self):
#        return 1


def get_embedder(embedder: str, **kwargs: Any) -> IntFeaturizer:
    if embedder == "fourier":
        return FourierFeaturizer(**kwargs)
    elif embedder == "rbf":
        return RBFFeaturizer(**kwargs)
    elif embedder == "one-hot":
        return OneHotFeaturizer(**kwargs)
    elif embedder == "learnt":
        return LearnedFeaturizer(**kwargs)
    elif embedder == "float":
        raise NotImplementedError
        # embedder = FloatFeaturizer()
    elif embedder == "fourier-sines":
        return FourierFeaturizerSines(**kwargs)
    elif embedder == "abs-sines":
        return FourierFeaturizerAbsoluteSines(**kwargs)
    else:
        raise NotImplementedError(f"Unknown embedder type: {embedder}")
