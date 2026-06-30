"""Loss functions for sparse mass-spectrum training and retrieval.

Important binning assumption:
    Several distance/similarity losses use binned spectra with a nominal
    support size of ``num_bins = round(mz_max / mz_bin_res)``.
    With the common defaults ``mz_max=1500.0`` and ``mz_bin_res=0.01``,
    this implies a 150,000-bin spectrum grid.

    Pairwise helpers in this module remap to occupied bins to avoid
    materializing a full 150,000-wide dense matrix when possible, but the
    numerical behavior is still defined by that underlying bin grid.
"""

import logging
import math
from collections.abc import Callable

import torch as th
import torch.nn.functional as F

from fragnnet.utils.misc_utils import (
    EPS,
    LOG_ZERO,
    TOLERANCE_MIN_MZ,
    safelog,
    scatter_logl2normalize,
    scatter_logsoftmax,
    scatter_logsumexp,
    scatter_reduce,
    validate_bin_geometry,
)
from fragnnet.utils.spec_utils import (
    batch_jss_hun_helper,
    batched_bin_func,
    calculate_match_mzs,
    jss_helper,
    linear_sum_assignment,
    round_aggregate_peaks,
)

logger = logging.getLogger(__name__)

# LOG_TRUNC_FACTOR = {
#     i: float(th.log(th.erf(th.tensor(i)/np.sqrt(2)))) for i in range(0,6)
# }


def mog_log_prob(
    samples: th.Tensor,
    means: th.Tensor,
    variances: th.Tensor,
    log_weights: th.Tensor,
    log_trunc_factors: th.Tensor,
) -> th.Tensor:
    """Mixture of Gaussian
        We are modeling each peaks are mixture of guassian for given set of fragments
    Args:
        samples (_type_): b_true_mz,
        means (_type_): b_pred_mzs
        variances (_type_): b_variances
        log_weights (_type_): b_pred_logprobs
        log_trunc_factors (_type_): b_log_trunc_factors

    Returns:
        _type_: _description_
    """

    samples = samples.reshape(-1, 1)
    means = means.reshape(1, -1)
    variances = variances.reshape(1, -1)
    log_weights = log_weights.reshape(1, -1)
    log_trunc_factors = log_trunc_factors.reshape(1, -1)
    # [B,K]
    normal_log_probs = -0.5 * (samples - means) ** 2 / variances - 0.5 * th.log(
        2 * th.pi * variances
    )
    normal_log_probs = normal_log_probs - log_trunc_factors
    # [B]
    log_probs = th.logsumexp(normal_log_probs + log_weights, dim=1)
    return log_probs


def mog_ce_fn(
    b_true_mzs: th.Tensor,
    b_pred_mzs: th.Tensor,
    b_pred_logprobs: th.Tensor,
    relative: bool,
    tolerance: float,
    tolerance_min_mz: float,
    tolerance_multiple: float,
    gaussian_renormalize: bool,
    **kwargs,
) -> th.Tensor:
    if relative:
        b_stds = th.clamp(b_pred_mzs, min=tolerance_min_mz) * tolerance
    else:
        b_stds = tolerance * th.ones_like(b_pred_mzs)
    b_vars = b_stds**2
    if gaussian_renormalize:
        assert tolerance_multiple > 0, tolerance_multiple
        # log_trunc_factor = LOG_TRUNC_FACTOR.get(tolerance_multiple,1.)
        # b_log_trunc_factors = log_trunc_factor*th.ones_like(b_pred_mzs)
        sigma = math.sqrt(2)
        trunc_factor = math.erf(tolerance_multiple / sigma)
        b_log_trunc_factors = th.log(
            th.tensor(trunc_factor, device=b_pred_mzs.device) * th.ones_like(b_pred_mzs)
        )
    else:
        b_log_trunc_factors = th.zeros_like(b_pred_mzs)
    b_logprobs = mog_log_prob(b_true_mzs, b_pred_mzs, b_vars, b_pred_logprobs, b_log_trunc_factors)

    return b_logprobs


def pm_ce_fn(
    b_pred_mzs: th.Tensor,
    b_true_mzs: th.Tensor,
    b_pred_logprobs: th.Tensor,
    relative: bool,
    tolerance: float,
    tolerance_multiple: float,
    tolerance_min_mz: float,
    **kwargs,
) -> th.Tensor:
    """
    Compute per-true-peak log-probability via peak-marginal aggregation.

    For each true peak in `b_true_mzs`, this function aggregates the
    log-probability mass assigned by the predicted peaks `b_pred_mzs`
    within an m/z tolerance window. Concretely, it performs a
    log-sum-exp over the predicted log-probabilities `b_pred_logprobs`
    that match each true peak, yielding a length-`T` tensor (one value
    per true peak).

    Args:
        b_pred_mzs: [P] Predicted m/z values for the current spectrum.
        b_true_mzs: [T] Ground-truth m/z values for the current spectrum.
        b_pred_logprobs: [P] Predicted log-probabilities (intensities) per predicted peak.
        relative: If True, the tolerance is relative to m/z; otherwise, absolute.
        tolerance: Base tolerance value (absolute or the factor used for relative).
        tolerance_multiple: Multiplicative factor applied to `tolerance` when matching.
        tolerance_min_mz: Minimum m/z used when applying relative tolerance.

    Returns:
        [T] Log-probabilities per true peak obtained by aggregating matching
        predicted peak log-probabilities via log-sum-exp.
    """

    # Compute boolean match mask [T, P] indicating which predicted peaks
    # fall within the m/z tolerance window of each true peak.
    b_match_mask = calculate_match_mzs(
        b_true_mzs,
        b_pred_mzs,
        tolerance=tolerance * tolerance_multiple,
        relative=relative,
        tolerance_min_mz=tolerance_min_mz,
    )
    b_match_mask = th.as_tensor(b_match_mask, device=b_pred_logprobs.device)
    # Aggregate matched predicted log-probs for each true peak via log-sum-exp.
    # Unmatched (false mask) entries are suppressed by adding a large negative
    # value (`LOG_ZERO`) so they do not contribute to the sum.
    b_logprobs = th.logsumexp(
        b_pred_logprobs.reshape(1, -1) * b_match_mask.to(b_pred_logprobs.dtype)
        + LOG_ZERO(b_pred_logprobs.dtype) * (~b_match_mask).to(b_pred_logprobs.dtype),
        dim=1,
    )
    return b_logprobs


def sparse_cross_entropy_seq(
    true_mzs: th.Tensor,
    true_logprobs: th.Tensor,
    true_batch_idxs: th.Tensor,
    pred_mzs: th.Tensor,
    pred_logprobs: th.Tensor,
    pred_batch_idxs: th.Tensor,
    pred_oos_logprobs: th.Tensor,
    ce_fn: Callable,
    tolerance: float,
    relative: bool,
    tolerance_min_mz: float,
    tolerance_multiple: float,
    gaussian_renormalize: bool,
    tol_per_sample: th.Tensor | None = None,
    min_mz_per_sample: th.Tensor | None = None,
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """compute sparse cross entropy in squence

    Args:
        true_mzs (_type_): groundtruth mzs
        true_logprobs (_type_): groundtruth log prob of each mz aka intensities
        true_batch_idxs (_type_): spec idx signed to each groundtruth
        pred_mzs (_type_): predicted mzs
        pred_logprobs (_type_): predicted log prob of each mz aka intensities
        pred_batch_idxs (_type_): pec idx signed to each prediction
        ce_fn (_type_, optional): _description_. Defaults to mog_log_prob.
        tolerance (_type_, optional): _description_. Defaults to 1e-3.
        relative (bool, optional): _description_. Defaults to False.
        oos_tolerance_multiple (int, optional): _description_.
        gaussian_renormalize (bool, optional): _description_.
        pm_tolerance_multiple (int, optional): _description_.

    Returns:
        tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]: ios_ces, oos_ces, true_oos_logprobs, true_oos_entropies
    """

    mask_value = LOG_ZERO(true_logprobs.dtype)
    batch_size = int(th.max(true_batch_idxs).item() + 1)
    # ces = th.zeros([batch_size],dtype=true_logprobs.dtype,device=true_logprobs.device)
    ios_ces = th.zeros(batch_size, dtype=true_logprobs.dtype, device=true_logprobs.device)
    oos_ces = th.zeros(batch_size, dtype=true_logprobs.dtype, device=true_logprobs.device)
    true_oos_logprobs = th.zeros(batch_size, dtype=true_logprobs.dtype, device=true_logprobs.device)
    true_oos_entropies = th.zeros_like(true_oos_logprobs)
    for batch_idx in range(batch_size):
        # get groundtruth belongs to this batch idx
        b_true_mask = true_batch_idxs == batch_idx
        # get predictions belongs to this batch idx
        b_pred_mask = pred_batch_idxs == batch_idx
        # get mzs and logprobs
        b_true_logprobs = true_logprobs[b_true_mask]
        b_true_mzs = true_mzs[b_true_mask]
        b_pred_logprobs = pred_logprobs[b_pred_mask]
        b_pred_mzs = pred_mzs[b_pred_mask]
        # per-sample tolerance (falls back to global scalars when not provided)
        b_tolerance = tol_per_sample[batch_idx].item() if tol_per_sample is not None else tolerance
        b_min_mz = (
            min_mz_per_sample[batch_idx].item()
            if min_mz_per_sample is not None
            else tolerance_min_mz
        )
        # get matches
        b_match_mask = calculate_match_mzs(
            b_true_mzs,
            b_pred_mzs,
            tolerance=tolerance_multiple * b_tolerance,
            relative=relative,
            tolerance_min_mz=b_min_mz,
        )
        b_match_mask = th.as_tensor(b_match_mask, device=b_true_mzs.device)
        if tolerance_multiple == -1:
            b_true_match_mask = th.ones_like(b_true_mzs, dtype=th.bool)
        else:
            assert tolerance_multiple > 0, tolerance_multiple
            b_true_match_mask = th.any(b_match_mask, dim=1)
        if not th.any(b_true_match_mask):
            logger.warning("Everything is OOS!")
        # calculate ios logprobs
        b_ios_logprobs = ce_fn(
            b_true_mzs=b_true_mzs,
            b_pred_mzs=b_pred_mzs,
            b_pred_logprobs=b_pred_logprobs,
            relative=relative,
            tolerance=b_tolerance,
            tolerance_min_mz=b_min_mz,
            tolerance_multiple=tolerance_multiple,
            gaussian_renormalize=gaussian_renormalize,
        )
        b_ios_ce = th.sum(
            b_true_match_mask.to(b_true_logprobs.dtype) * -th.exp(b_true_logprobs) * b_ios_logprobs
        )
        # calculate oos logprobs
        b_true_oos_logprobs = th.logsumexp(
            mask_value * b_true_match_mask.to(b_true_logprobs.dtype)
            + b_true_logprobs * (~b_true_match_mask).to(b_true_logprobs.dtype),
            dim=0,
        )
        b_true_oos_entropy = -th.sum(
            th.exp(b_true_logprobs[~b_true_match_mask] - b_true_oos_logprobs)
            * (b_true_logprobs[~b_true_match_mask] - b_true_oos_logprobs),
            dim=0,
        )
        b_pred_oos_logprobs = pred_oos_logprobs[batch_idx]
        b_oos_ce = -th.exp(b_true_oos_logprobs) * b_pred_oos_logprobs
        # combine
        ios_ces[batch_idx] = b_ios_ce
        oos_ces[batch_idx] = b_oos_ce
        true_oos_logprobs[batch_idx] = b_true_oos_logprobs
        true_oos_entropies[batch_idx] = b_true_oos_entropy
    return ios_ces, oos_ces, true_oos_logprobs, true_oos_entropies


def sparse_cross_entropy_vec(
    true_mzs: th.Tensor,
    true_logprobs: th.Tensor,
    true_batch_idxs: th.Tensor,
    pred_mzs: th.Tensor,
    pred_logprobs: th.Tensor,
    pred_batch_idxs: th.Tensor,
    pred_oos_logprobs: th.Tensor,
    tolerance: float,
    relative: bool,
    tolerance_min_mz: float,
    oos_tolerance_multiple: float,
    gaussian_renormalize: bool,
    loss_batch_size: int,
    tol_per_sample: th.Tensor | None = None,
    min_mz_per_sample: th.Tensor | None = None,
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """
    Vectorized computation of sparse cross-entropy loss for batched spectra.

    Args:
        true_mzs: Ground truth m/z values.
        true_logprobs: Ground truth log-probabilities (intensities).
        true_batch_idxs: Batch indices for ground truth peaks.
        pred_mzs: Predicted m/z values.
        pred_logprobs: Predicted log-probabilities (intensities).
        pred_batch_idxs: Batch indices for predicted peaks.
        pred_oos_logprobs: Predicted log-probabilities for out-of-sample (OOS) peaks.
        tolerance: m/z matching tolerance.
        relative: Whether tolerance is relative to m/z.
        tolerance_min_mz: Minimum m/z for relative tolerance.
        oos_tolerance_multiple: Multiplier for OOS tolerance.
        gaussian_renormalize: Whether to renormalize Gaussian.
        loss_batch_size: Batch size for loss computation.

    Returns:
        ios_ce: In-sample cross-entropy per batch.
        oos_ce: Out-of-sample cross-entropy per batch.
        0., 1.: Placeholders for compatibility.
    """

    # Set up types and constants
    float_dtype = true_logprobs.dtype
    device = true_logprobs.device
    mask_value = LOG_ZERO(float_dtype)
    batch_size = int(th.max(true_batch_idxs).item() + 1)

    # Calculate number of loss batches for memory efficiency
    loss_num_batches = int((batch_size // loss_batch_size) + int(batch_size % loss_batch_size > 0))

    # Compute cumulative sum of counts for batch slicing.
    # .tolist() transfers batch_size+1 ints once, turning loop-body indexing into Python-int
    # slicing (no GPU→CPU sync per iteration). th.bincount is O(N) vs th.unique's O(N log N).
    true_batch_cumsum: list[int] = th.cat(
        [true_batch_idxs.new_zeros(1), th.bincount(true_batch_idxs, minlength=batch_size).cumsum(0)]
    ).tolist()
    pred_batch_cumsum: list[int] = th.cat(
        [pred_batch_idxs.new_zeros(1), th.bincount(pred_batch_idxs, minlength=batch_size).cumsum(0)]
    ).tolist()

    # Precompute loop-invariant constants as Python scalars (avoids per-iteration tensor creation).
    _LOG_2PI = math.log(2 * math.pi)
    if gaussian_renormalize:
        if oos_tolerance_multiple <= 0:
            raise ValueError(f"oos_tolerance_multiple must be > 0, got {oos_tolerance_multiple}")
        _log_trunc_factor: float = math.log(math.erf(oos_tolerance_multiple / math.sqrt(2)))
    else:
        _log_trunc_factor = 0.0

    # Initialize output tensors
    ios_ce = th.zeros(batch_size, dtype=float_dtype, device=device)
    oos_ce = th.zeros(batch_size, dtype=float_dtype, device=device)
    true_oos_logprobs = th.zeros(batch_size, dtype=float_dtype, device=device)

    # Expand per-sample tolerances to per-peak if provided
    if tol_per_sample is not None:
        tol_per_true_peak = tol_per_sample[true_batch_idxs].to(float_dtype)
        min_mz_per_true_peak = min_mz_per_sample[true_batch_idxs].to(float_dtype)
        tol_per_pred_peak = tol_per_sample[pred_batch_idxs].to(float_dtype)
        min_mz_per_pred_peak = min_mz_per_sample[pred_batch_idxs].to(float_dtype)
    else:
        tol_per_true_peak = None
        min_mz_per_true_peak = None
        tol_per_pred_peak = None
        min_mz_per_pred_peak = None

    # Loop over loss batches for memory efficiency
    for bl in range(loss_num_batches):
        # Compute batch slice indices (Python ints from .tolist() — no GPU sync)
        bl_lower = bl * loss_batch_size
        bl_upper = min((bl + 1) * loss_batch_size, batch_size)
        bl_true_lower = true_batch_cumsum[bl_lower]
        bl_true_upper = true_batch_cumsum[bl_upper]
        bl_pred_lower = pred_batch_cumsum[bl_lower]
        bl_pred_upper = pred_batch_cumsum[bl_upper]
        bl_batch_size = bl_upper - bl_lower

        # Slice batch data and adjust indices to local batch
        bl_true_batch_idxs = true_batch_idxs[bl_true_lower:bl_true_upper] - bl_lower
        bl_pred_batch_idxs = pred_batch_idxs[bl_pred_lower:bl_pred_upper] - bl_lower
        bl_true_logprobs = true_logprobs[bl_true_lower:bl_true_upper]
        bl_pred_logprobs = pred_logprobs[bl_pred_lower:bl_pred_upper]
        bl_true_mzs = true_mzs[bl_true_lower:bl_true_upper]
        bl_pred_mzs = pred_mzs[bl_pred_lower:bl_pred_upper]
        bl_pred_oos_logprobs = pred_oos_logprobs[bl_lower:bl_upper]

        # Slice per-peak tolerance arrays for this loss batch
        bl_tol_per_true = (
            tol_per_true_peak[bl_true_lower:bl_true_upper]
            if tol_per_true_peak is not None
            else None
        )
        bl_min_mz_per_true = (
            min_mz_per_true_peak[bl_true_lower:bl_true_upper]
            if min_mz_per_true_peak is not None
            else None
        )
        bl_tol_per_pred = (
            tol_per_pred_peak[bl_pred_lower:bl_pred_upper]
            if tol_per_pred_peak is not None
            else None
        )
        bl_min_mz_per_pred = (
            min_mz_per_pred_peak[bl_pred_lower:bl_pred_upper]
            if min_mz_per_pred_peak is not None
            else None
        )

        # Create mask for matching batch indices
        bl_batch_mask = bl_true_batch_idxs.reshape(-1, 1) == bl_pred_batch_idxs.reshape(1, -1)
        # Create mask for matching m/z values within tolerance
        bl_match_mask = calculate_match_mzs(
            bl_true_mzs,
            bl_pred_mzs,
            tolerance=oos_tolerance_multiple * tolerance,
            relative=relative,
            tolerance_min_mz=tolerance_min_mz,
            tol_per_true=(
                oos_tolerance_multiple * bl_tol_per_true if bl_tol_per_true is not None else None
            ),
            min_mz_per_true=bl_min_mz_per_true,
        )
        bl_match_mask = th.as_tensor(bl_match_mask, device=device)
        # Combine masks: only consider matches within same batch and within m/z tolerance
        bl_both_mask = bl_batch_mask & bl_match_mask
        del bl_batch_mask, bl_match_mask

        # Mask for true peaks that have any matching prediction
        bl_true_both_mask = th.any(bl_both_mask, dim=1)

        # Compute Gaussian stds for predictions
        if relative:
            if bl_tol_per_pred is not None:
                bl_stds = th.maximum(bl_pred_mzs, bl_min_mz_per_pred) * bl_tol_per_pred
            else:
                bl_stds = th.clamp(bl_pred_mzs, min=tolerance_min_mz) * tolerance
        else:
            bl_stds = tolerance * th.ones_like(bl_pred_mzs)
        bl_vars = bl_stds**2

        # Compute log-probabilities for in-sample peaks (vectorized Gaussian mixture).
        # _LOG_2PI and _log_trunc_factor are Python scalars — no tensor allocation per iteration.
        bl_ios_log_probs = (
            -0.5
            * (bl_true_mzs.reshape(-1, 1) - bl_pred_mzs.reshape(1, -1)) ** 2
            / bl_vars.reshape(1, -1)
            - 0.5 * (_LOG_2PI + th.log(bl_vars.reshape(1, -1)))
            + bl_pred_logprobs.reshape(1, -1)
            - _log_trunc_factor
        )
        # Mask out non-matching pairs with mask_value (large negative)
        bl_ios_log_probs = bl_ios_log_probs + (~bl_both_mask).to(float_dtype) * mask_value
        # Log-sum-exp over predictions for each true peak
        bl_ios_log_probs = th.logsumexp(bl_ios_log_probs, dim=1)

        # In-sample CE: scatter_reduce handles empty mask → zeros (no GPU sync needed).
        bl_ios_ce = scatter_reduce(
            (-th.exp(bl_true_logprobs) * bl_ios_log_probs)[bl_true_both_mask],
            bl_true_batch_idxs[bl_true_both_mask],
            "sum",
            dim=0,
            dim_size=bl_batch_size,
            default=0.0,
        )
        ios_ce[bl_lower:bl_upper] = bl_ios_ce

        # OOS logprobs for unmatched true peaks. scatter_logsumexp is empty-safe → LOG_ZERO.
        # No GPU sync needed — the th.all() conditional has been removed.
        bl_unmatched_mask = ~bl_true_both_mask
        bl_true_oos_logprobs = scatter_logsumexp(
            bl_true_logprobs[bl_unmatched_mask],
            bl_true_batch_idxs[bl_unmatched_mask],
            dim_size=bl_batch_size,
        )
        # Compute out-of-sample cross-entropy
        bl_oos_ce = -th.exp(bl_true_oos_logprobs) * bl_pred_oos_logprobs
        oos_ce[bl_lower:bl_upper] = bl_oos_ce
        true_oos_logprobs[bl_lower:bl_upper] = bl_true_oos_logprobs

    return ios_ce, oos_ce, true_oos_logprobs, th.zeros_like(ios_ce)


def get_sparse_cross_entropy_fn(
    dist: str,
    vectorized: bool,
    tolerance: float,
    relative: bool,
    tolerance_min_mz: float,
    oos_tolerance_multiple: float,
    gaussian_renormalize: bool,
    pm_tolerance_multiple: float,
    loss_batch_size: int,
) -> Callable:
    if dist == "gaussian":
        ce_fn = mog_ce_fn
    elif dist == "peak_marginal":
        ce_fn = pm_ce_fn
    else:
        msg = (
            f"Distribution '{dist}' not implemented. "
            "Supported distributions: 'gaussian', 'peak_marginal'."
        )
        raise NotImplementedError(msg)

    if vectorized:
        if dist != "gaussian":
            msg = f"Vectorized loss only supports 'gaussian' distribution, got '{dist}'"
            raise NotImplementedError(msg)

        def sce_fn(*args, tol_per_sample=None, min_mz_per_sample=None):
            return sparse_cross_entropy_vec(
                *args,
                tolerance=tolerance,
                relative=relative,
                tolerance_min_mz=tolerance_min_mz,
                oos_tolerance_multiple=oos_tolerance_multiple,
                gaussian_renormalize=gaussian_renormalize,
                loss_batch_size=loss_batch_size,
                tol_per_sample=tol_per_sample,
                min_mz_per_sample=min_mz_per_sample,
            )
    else:
        if dist == "gaussian":
            tolerance_multiple = oos_tolerance_multiple
        else:
            tolerance_multiple = pm_tolerance_multiple

        def sce_fn(*args, tol_per_sample=None, min_mz_per_sample=None):
            return sparse_cross_entropy_seq(
                *args,
                ce_fn=ce_fn,
                tolerance=tolerance,
                relative=relative,
                tolerance_min_mz=tolerance_min_mz,
                tolerance_multiple=tolerance_multiple,
                gaussian_renormalize=gaussian_renormalize,
                tol_per_sample=tol_per_sample,
                min_mz_per_sample=min_mz_per_sample,
            )

    return sce_fn


def _pad_flat_by_batch(
    values: th.Tensor,
    batch_idxs: th.Tensor,
    batch_size: int,
    pad_value: float,
) -> tuple[th.Tensor, th.Tensor]:
    """Convert flat ragged values plus batch indices into a padded [B, N] tensor."""
    counts = th.bincount(batch_idxs, minlength=batch_size)
    max_count = int(counts.max().item()) if counts.numel() > 0 else 0
    padded = values.new_full((batch_size, max_count), pad_value)
    valid = th.arange(max_count, device=values.device).unsqueeze(0) < counts.unsqueeze(1)
    if values.numel() == 0 or max_count == 0:
        return padded, valid

    sort_idx = th.argsort(batch_idxs)
    sorted_batch_idxs = batch_idxs[sort_idx]
    sorted_values = values[sort_idx]
    row_starts = th.cumsum(counts, dim=0) - counts
    positions = th.arange(values.numel(), device=values.device) - th.repeat_interleave(
        row_starts, counts
    )
    padded[sorted_batch_idxs, positions] = sorted_values
    return padded, valid


def sparse_jensen_shannon_divergence_hungarian_vec(
    true_mzs: th.Tensor,
    true_logprobs: th.Tensor,
    true_batch_idxs: th.Tensor,
    pred_mzs: th.Tensor,
    pred_logprobs: th.Tensor,
    pred_batch_idxs: th.Tensor,
    tolerance: float = 1e-5,
    relative: bool = True,
    tolerance_min_mz: float = TOLERANCE_MIN_MZ,
    log_min: float = EPS,
    tol_per_sample: th.Tensor | None = None,
    min_mz_per_sample: th.Tensor | None = None,
    loss_batch_size: int = 32,
) -> th.Tensor:
    """Batched Jensen-Shannon distance with Hungarian peak matching."""
    true_mzs, true_logprobs, true_batch_idxs = round_aggregate_peaks(
        true_mzs, true_logprobs, true_batch_idxs, agg="lse"
    )
    pred_mzs, pred_logprobs, pred_batch_idxs = round_aggregate_peaks(
        pred_mzs, pred_logprobs, pred_batch_idxs, agg="lse"
    )
    true_sort_idx = th.argsort(true_batch_idxs)
    true_mzs = true_mzs[true_sort_idx]
    true_logprobs = true_logprobs[true_sort_idx]
    true_batch_idxs = true_batch_idxs[true_sort_idx]
    pred_sort_idx = th.argsort(pred_batch_idxs)
    pred_mzs = pred_mzs[pred_sort_idx]
    pred_logprobs = pred_logprobs[pred_sort_idx]
    pred_batch_idxs = pred_batch_idxs[pred_sort_idx]

    batch_size = int(th.max(true_batch_idxs).item() + 1)
    if loss_batch_size is None or loss_batch_size <= 0:
        loss_batch_size = batch_size
    loss_num_batches = int((batch_size // loss_batch_size) + int(batch_size % loss_batch_size > 0))

    true_counts = th.bincount(true_batch_idxs, minlength=batch_size)
    pred_counts = th.bincount(pred_batch_idxs, minlength=batch_size)
    true_cumsum = th.cat([true_batch_idxs.new_zeros(1), true_counts.cumsum(0)]).tolist()
    pred_cumsum = th.cat([pred_batch_idxs.new_zeros(1), pred_counts.cumsum(0)]).tolist()

    jsd_hun = th.zeros(batch_size, device=true_logprobs.device, dtype=true_logprobs.dtype)
    log2 = math.log(2.0)

    for bl in range(loss_num_batches):
        bl_lower = bl * loss_batch_size
        bl_upper = min((bl + 1) * loss_batch_size, batch_size)
        bl_true_lower = true_cumsum[bl_lower]
        bl_true_upper = true_cumsum[bl_upper]
        bl_pred_lower = pred_cumsum[bl_lower]
        bl_pred_upper = pred_cumsum[bl_upper]
        bl_batch_size = bl_upper - bl_lower

        bl_true_batch_idxs = true_batch_idxs[bl_true_lower:bl_true_upper] - bl_lower
        bl_pred_batch_idxs = pred_batch_idxs[bl_pred_lower:bl_pred_upper] - bl_lower
        bl_true_mzs = true_mzs[bl_true_lower:bl_true_upper]
        bl_pred_mzs = pred_mzs[bl_pred_lower:bl_pred_upper]
        bl_true_ints = th.exp(true_logprobs[bl_true_lower:bl_true_upper])
        bl_pred_ints = th.exp(pred_logprobs[bl_pred_lower:bl_pred_upper])

        batch_true_mzs, true_valid = _pad_flat_by_batch(
            bl_true_mzs, bl_true_batch_idxs, bl_batch_size, 0.0
        )
        batch_pred_mzs, pred_valid = _pad_flat_by_batch(
            bl_pred_mzs, bl_pred_batch_idxs, bl_batch_size, 0.0
        )
        batch_true_ints, _ = _pad_flat_by_batch(
            bl_true_ints, bl_true_batch_idxs, bl_batch_size, 0.0
        )
        batch_pred_ints, _ = _pad_flat_by_batch(
            bl_pred_ints, bl_pred_batch_idxs, bl_batch_size, 0.0
        )

        if batch_true_mzs.shape[1] == 0 or batch_pred_mzs.shape[1] == 0:
            graph_zero = bl_pred_ints.sum() * 0.0
            jsd_hun[bl_lower:bl_upper] = 1.0 + graph_zero
            continue

        if tol_per_sample is not None:
            bl_tol_per_true = tol_per_sample[bl_lower:bl_upper].to(batch_true_mzs.dtype)
            bl_min_mz_per_true = min_mz_per_sample[bl_lower:bl_upper].to(batch_true_mzs.dtype)
            tol_per_true = bl_tol_per_true[:, None].expand_as(batch_true_mzs)
            min_mz_per_true = bl_min_mz_per_true[:, None].expand_as(batch_true_mzs)
        else:
            tol_per_true = None
            min_mz_per_true = None

        batch_match_mask = calculate_match_mzs(
            batch_true_mzs,
            batch_pred_mzs,
            tolerance=tolerance,
            relative=relative,
            tolerance_min_mz=tolerance_min_mz,
            tol_per_true=tol_per_true,
            min_mz_per_true=min_mz_per_true,
        )
        batch_match_mask = th.as_tensor(batch_match_mask, device=true_logprobs.device)
        batch_match_mask = batch_match_mask & true_valid[:, :, None] & pred_valid[:, None, :]

        if not bool(batch_match_mask.any().item()):
            graph_zero = bl_pred_ints.sum() * 0.0
            jsd_hun[bl_lower:bl_upper] = 1.0 + graph_zero
            continue

        batch_true_match_mask = batch_match_mask.any(dim=2)
        batch_pred_match_mask = batch_match_mask.any(dim=1)
        batch_true_prec_mask = th.zeros_like(batch_true_ints, dtype=th.bool)
        batch_pred_prec_mask = th.zeros_like(batch_pred_ints, dtype=th.bool)

        jss_hun = batch_jss_hun_helper(
            batch_true_ints,
            batch_pred_ints,
            batch_match_mask,
            batch_true_match_mask,
            batch_pred_match_mask,
            False,
            batch_true_prec_mask,
            batch_pred_prec_mask,
            log_min,
        )
        jsd_hun[bl_lower:bl_upper] = th.clamp(1.0 - jss_hun / log2, min=0.0, max=1.0)

    return jsd_hun


def sparse_entropy_fn(
    logprobs: th.Tensor,
    batch_idxs: th.Tensor,
    oos_logprobs: th.Tensor | None = None,
    renormalize: bool = False,
    support_size_delta: float = 0.0,
) -> tuple[th.Tensor, th.Tensor]:
    if renormalize:
        logpartition = scatter_logsumexp(logprobs, batch_idxs)
        if oos_logprobs is not None:
            logpartition = th.logsumexp(th.stack([logpartition, oos_logprobs], dim=1), dim=1)
            oos_logprobs = oos_logprobs - logpartition
        logprobs = logprobs - th.gather(input=logpartition, index=batch_idxs, dim=0)
    logprobs = th.clamp(logprobs, max=0.0)
    probs = th.exp(logprobs)
    plogp = probs * logprobs
    batch_size = int(th.max(batch_idxs).item() + 1)
    entropy = -scatter_reduce(
        src=plogp,
        index=batch_idxs,
        reduce="sum",
        dim_size=batch_size,
    )
    if th.min(entropy) < -0.001:
        raise ValueError(f"Negative entropy: {th.min(entropy)}")

    support_size = scatter_reduce(
        src=th.ones_like(batch_idxs, dtype=logprobs.dtype),
        index=batch_idxs,
        reduce="sum",
        dim_size=batch_size,
    )
    support_size += support_size_delta
    if oos_logprobs is not None:
        assert oos_logprobs.shape[0] == batch_size, (oos_logprobs.shape, batch_size)
        entropy = entropy - th.exp(oos_logprobs) * oos_logprobs
        support_size = support_size + 1.0
    max_entropy = safelog(support_size)
    # rescale entropy
    max_entropy = th.clamp(max_entropy, min=EPS)
    entropy = th.clamp(entropy, min=th.zeros_like(max_entropy), max=max_entropy)
    norm_entropy = entropy / max_entropy
    if th.any(norm_entropy < 0.0) or th.any(norm_entropy > 1.0):
        raise ValueError("Normalized entropy out of bounds [0, 1]")
    return entropy, norm_entropy


def sparse_conditional_entropy_fn(
    marginal_logprobs: th.Tensor,
    marginal_batch_idxs: th.Tensor,
    conditional_logprobs: th.Tensor,
    conditional_idxs: th.Tensor,
    conditional_batch_idxs: th.Tensor,
    conditional_support_size_delta: float = 0.0,
) -> tuple[th.Tensor, th.Tensor]:
    batch_size = int(th.max(conditional_batch_idxs).item() + 1)
    marginal_support_size = marginal_logprobs.shape[0]
    marginal_logprobs = th.clamp(marginal_logprobs, max=0.0)
    conditional_logprobs = th.clamp(conditional_logprobs, max=0.0)
    marginal_probs = th.exp(marginal_logprobs)
    conditional_plogp = th.exp(conditional_logprobs) * conditional_logprobs
    conditional_entropy = -scatter_reduce(
        conditional_plogp, index=conditional_idxs, reduce="sum", dim_size=marginal_support_size
    )
    entropy = scatter_reduce(
        marginal_probs * conditional_entropy,
        index=marginal_batch_idxs,
        reduce="sum",
        dim_size=batch_size,
    )
    if th.min(entropy) < -0.001:
        raise ValueError(f"Negative conditional entropy: {th.min(entropy)}")

    conditional_support_size = scatter_reduce(
        th.ones_like(conditional_logprobs, dtype=marginal_logprobs.dtype),
        index=conditional_idxs,
        reduce="sum",
        dim_size=marginal_support_size,
    )
    conditional_support_size += conditional_support_size_delta
    max_entropy = scatter_reduce(
        marginal_probs * th.clamp(safelog(conditional_support_size), min=0.0),
        index=marginal_batch_idxs,
        reduce="sum",
        dim_size=batch_size,
    )
    # rescale entropy
    max_entropy = th.clamp(max_entropy, min=EPS)
    entropy = th.clamp(entropy, min=th.zeros_like(max_entropy), max=max_entropy)
    norm_entropy = entropy / max_entropy
    if th.any(norm_entropy < 0.0) or th.any(norm_entropy > 1.0):
        raise ValueError("Normalized conditional entropy out of bounds [0, 1]")
    return entropy, norm_entropy


def get_edge_loss_fn(edge_loss_fn_type: str, constant: float) -> Callable:
    if edge_loss_fn_type == "quadratic":

        def quad_loss(x):
            return constant * x**2

        return quad_loss
    elif edge_loss_fn_type == "linear":

        def linear_loss(x):
            return constant * th.abs(x)

        return linear_loss
    elif edge_loss_fn_type == "exponential":

        def exp_loss(x):
            return constant * th.exp(th.abs(x))

        return exp_loss
    else:
        raise NotImplementedError(f"Edge loss function type {edge_loss_fn_type} not implemented")


def sparse_cosine_distance(
    true_mzs: th.Tensor,
    true_logprobs: th.Tensor,
    true_batch_idxs: th.Tensor,
    pred_mzs: th.Tensor,
    pred_logprobs: th.Tensor,
    pred_batch_idxs: th.Tensor,
    mz_max: float = 1500.0,
    mz_bin_res: float = 0.01,
    log_distance: bool = False,
) -> th.Tensor:
    """Compute sparse cosine distance after m/z binning.

    Note:
        The implicit spectrum support size is
        ``num_bins = round(mz_max / mz_bin_res)``. With defaults,
        this is 150,000 bins.

    Args:
        true_mzs: Ground truth m/z values.
        true_logprobs: Ground truth log-probabilities (intensities).
        true_batch_idxs: Batch indices for ground truth peaks.
        pred_mzs: Predicted m/z values.
        pred_logprobs: Predicted log-probabilities (intensities).
        pred_batch_idxs: Batch indices for predicted peaks.
        mz_max: Maximum m/z value for binning. Defaults to 1500.0.
        mz_bin_res: Resolution of m/z bins. Defaults to 0.01.
        log_distance: If True, return log(1-cosine); else return 1-cosine.

    Returns:
        Tensor of cosine distances per batch.
    """

    # sparse bin
    true_bin_idxs, true_bin_logprobs, true_bin_batch_idxs = batched_bin_func(
        true_mzs,
        true_logprobs,
        true_batch_idxs,
        mz_max=mz_max,
        mz_bin_res=mz_bin_res,
        agg="lse",
        sparse=True,
    )
    pred_bin_idxs, pred_bin_logprobs, pred_bin_batch_idxs = batched_bin_func(
        pred_mzs,
        pred_logprobs,
        pred_batch_idxs,
        mz_max=mz_max,
        mz_bin_res=mz_bin_res,
        agg="lse",
        sparse=True,
    )
    return sparse_cosine_distance_binned(
        true_bin_idxs,
        true_bin_logprobs,
        true_bin_batch_idxs,
        pred_bin_idxs,
        pred_bin_logprobs,
        pred_bin_batch_idxs,
        log_distance=log_distance,
    )


def sparse_cosine_distance_binned(
    true_bin_idxs: th.Tensor,
    true_bin_logprobs: th.Tensor,
    true_bin_batch_idxs: th.Tensor,
    pred_bin_idxs: th.Tensor,
    pred_bin_logprobs: th.Tensor,
    pred_bin_batch_idxs: th.Tensor,
    log_distance: bool = False,
) -> th.Tensor:
    # l2 normalize
    true_bin_logprobs = scatter_logl2normalize(true_bin_logprobs, true_bin_batch_idxs)
    pred_bin_logprobs = scatter_logl2normalize(pred_bin_logprobs, pred_bin_batch_idxs)
    # dot product
    pred_mask = th.isin(pred_bin_idxs, true_bin_idxs)
    true_mask = th.isin(true_bin_idxs, pred_bin_idxs)
    batch_size = int(th.max(true_bin_batch_idxs).item() + 1)
    if th.any(pred_mask):
        both_bin_logprobs = pred_bin_logprobs[pred_mask] + true_bin_logprobs[true_mask]
        assert th.all(pred_bin_batch_idxs[pred_mask] == true_bin_batch_idxs[true_mask])
        log_cos_sim = scatter_logsumexp(
            both_bin_logprobs, pred_bin_batch_idxs[pred_mask], dim_size=batch_size
        )
    else:
        # cosine similarities are all zero
        log_cos_sim = th.full(
            size=(batch_size,),
            fill_value=LOG_ZERO(pred_bin_logprobs.dtype),
            dtype=pred_bin_logprobs.dtype,
            device=pred_bin_logprobs.device,
        )
        # involve pred_logprobs to keep gradient
        log_cos_sim = log_cos_sim + 0.0 * th.mean(pred_bin_logprobs, dim=0)
    if log_distance:
        cos_dist = th.log1p(-th.exp(log_cos_sim))
    else:
        cos_dist = 1.0 - th.exp(log_cos_sim)
    return cos_dist


def sparse_cosine_distance_hungarian(
    true_mzs: th.Tensor,
    true_logprobs: th.Tensor,
    true_batch_idxs: th.Tensor,
    pred_mzs: th.Tensor,
    pred_logprobs: th.Tensor,
    pred_batch_idxs: th.Tensor,
    tolerance: float = 1e-5,
    relative: bool = True,
    tolerance_min_mz: float = TOLERANCE_MIN_MZ,
    log_distance: bool = False,
    tol_per_sample: th.Tensor | None = None,
    min_mz_per_sample: th.Tensor | None = None,
) -> th.Tensor:
    """
    Compute the sparse cosine distance between spectra using the Hungarian algorithm for optimal matching.

    Args:
        true_mzs: Ground truth m/z values.
        true_logprobs: Ground truth log-probabilities (intensities).
        true_batch_idxs: Batch indices for ground truth peaks.
        pred_mzs: Predicted m/z values.
        pred_logprobs: Predicted log-probabilities (intensities).
        pred_batch_idxs: Batch indices for predicted peaks.
        tolerance: m/z matching tolerance.
        relative: Whether tolerance is relative to m/z.
        tolerance_min_mz: Minimum m/z for relative tolerance.
        log_distance: If True, return log(1-cosine); else return 1-cosine.

    Returns:
        Tensor of cosine distances per batch.
    """

    # Aggregate and round peaks for both true and predicted spectra
    true_mzs, true_logprobs, true_batch_idxs = round_aggregate_peaks(
        true_mzs, true_logprobs, true_batch_idxs, agg="lse"
    )
    pred_mzs, pred_logprobs, pred_batch_idxs = round_aggregate_peaks(
        pred_mzs, pred_logprobs, pred_batch_idxs, agg="lse"
    )
    # Get number of batches
    batch_size = int(th.max(true_batch_idxs).item() + 1)
    cos_dist_hun = th.zeros(batch_size, device=true_logprobs.device, dtype=true_logprobs.dtype)
    for batch_idx in range(batch_size):
        # Select peaks for current batch
        b_true_mask = true_batch_idxs == batch_idx
        b_pred_mask = pred_batch_idxs == batch_idx
        b_true_mzs = true_mzs[b_true_mask]
        b_pred_mzs = pred_mzs[b_pred_mask]
        b_true_logprobs = true_logprobs[b_true_mask]
        b_pred_logprobs = pred_logprobs[b_pred_mask]
        # L2 normalize log-probabilities for cosine similarity
        b_true_logprobs = b_true_logprobs - 0.5 * th.logsumexp(2 * b_true_logprobs, dim=0)
        b_pred_logprobs = b_pred_logprobs - 0.5 * th.logsumexp(2 * b_pred_logprobs, dim=0)
        # per-sample tolerance (falls back to global scalars when not provided)
        b_tolerance = tol_per_sample[batch_idx].item() if tol_per_sample is not None else tolerance
        b_min_mz = (
            min_mz_per_sample[batch_idx].item()
            if min_mz_per_sample is not None
            else tolerance_min_mz
        )
        # Compute m/z matching mask within tolerance
        b_match_mzs = calculate_match_mzs(
            b_true_mzs,
            b_pred_mzs,
            tolerance=b_tolerance,
            relative=relative,
            tolerance_min_mz=b_min_mz,
        )
        b_match_mzs = th.as_tensor(b_match_mzs, device=b_true_mzs.device)
        # Identify which true and predicted peaks have matches
        b_true_match_mzs = th.any(b_match_mzs, dim=1)
        b_pred_match_mzs = th.any(b_match_mzs, dim=0)
        # Compute score matrix for matched peaks (outer sum of logprobs, exponentiated)
        b_score = th.exp(
            b_true_logprobs[b_true_match_mzs].detach().unsqueeze(1)
            + b_pred_logprobs[b_pred_match_mzs].detach().unsqueeze(0)
        )
        # Mask out unmatched pairs with LOG_ZERO
        b_score[~b_match_mzs[b_true_match_mzs][:, b_pred_match_mzs]] = LOG_ZERO(b_score.dtype)
        # Hungarian algorithm for optimal matching (maximizing score)
        b_true_idxs, b_pred_idxs = linear_sum_assignment(b_score, maximize=True)
        # Compute log-cosine similarity for matched pairs
        b_log_cos_hun = th.logsumexp(
            b_true_logprobs[b_true_match_mzs][b_true_idxs]
            + b_pred_logprobs[b_pred_match_mzs][b_pred_idxs],
            dim=0,
        )
        # Convert to cosine distance
        if log_distance:
            b_cos_dist_hun = th.log1p(-th.exp(b_log_cos_hun))
        else:
            b_cos_dist_hun = 1.0 - th.exp(b_log_cos_hun)
        cos_dist_hun[batch_idx] = b_cos_dist_hun
    return cos_dist_hun


def sparse_jensen_shannon_divergence_hungarian(
    true_mzs: th.Tensor,
    true_logprobs: th.Tensor,
    true_batch_idxs: th.Tensor,
    pred_mzs: th.Tensor,
    pred_logprobs: th.Tensor,
    pred_batch_idxs: th.Tensor,
    tolerance: float = 1e-5,
    relative: bool = True,
    tolerance_min_mz: float = TOLERANCE_MIN_MZ,
    log_min: float = EPS,
    tol_per_sample: th.Tensor | None = None,
    min_mz_per_sample: th.Tensor | None = None,
) -> th.Tensor:
    """
    Compute the sparse Jensen-Shannon divergence between spectra using the Hungarian algorithm for optimal matching.

    Args:
        true_mzs: Ground truth m/z values.
        true_logprobs: Ground truth log-probabilities (intensities).
        true_batch_idxs: Batch indices for ground truth peaks.
        pred_mzs: Predicted m/z values.
        pred_logprobs: Predicted log-probabilities (intensities).
        pred_batch_idxs: Batch indices for predicted peaks.
        tolerance: m/z matching tolerance.
        relative: Whether tolerance is relative to m/z.
        tolerance_min_mz: Minimum m/z for relative tolerance.
        log_min: Minimum log value to avoid log(0).

    Returns:
        Tensor of Jensen-Shannon divergences per batch (normalized to [0,1]).
    """

    # Aggregate and round peaks for both true and predicted spectra
    true_mzs, true_logprobs, true_batch_idxs = round_aggregate_peaks(
        true_mzs, true_logprobs, true_batch_idxs, agg="lse"
    )
    pred_mzs, pred_logprobs, pred_batch_idxs = round_aggregate_peaks(
        pred_mzs, pred_logprobs, pred_batch_idxs, agg="lse"
    )
    # Get number of batches
    batch_size = int(th.max(true_batch_idxs).item() + 1)
    jsd_hun = th.zeros(batch_size, device=true_logprobs.device, dtype=true_logprobs.dtype)
    LOG2 = math.log(2.0)  # precompute: avoids th.log(th.tensor(2.0)) inside the loop

    # Split peaks by batch element once (O(N)) instead of masking per element (O(B×N)).
    true_counts = th.bincount(true_batch_idxs, minlength=batch_size).tolist()
    pred_counts = th.bincount(pred_batch_idxs, minlength=batch_size).tolist()
    true_mzs_list = th.split(true_mzs, true_counts)
    true_logprobs_list = th.split(true_logprobs, true_counts)
    pred_mzs_list = th.split(pred_mzs, pred_counts)
    pred_logprobs_list = th.split(pred_logprobs, pred_counts)

    for batch_idx in range(batch_size):
        b_true_mzs = true_mzs_list[batch_idx]
        b_pred_mzs = pred_mzs_list[batch_idx]
        b_true_logprobs = true_logprobs_list[batch_idx]
        b_pred_logprobs = pred_logprobs_list[batch_idx]

        # Normalize probabilities (sum to 1) without GPU sync.
        # clamp(min=EPS) avoids division-by-zero; when all probs are ~0, sum≈0/EPS=0 anyway.
        b_true_probs = th.exp(b_true_logprobs)
        b_true_probs = b_true_probs / b_true_probs.sum().clamp(min=EPS)

        b_pred_probs = th.exp(b_pred_logprobs)
        b_pred_probs = b_pred_probs / b_pred_probs.sum().clamp(min=EPS)

        # per-sample tolerance (falls back to global scalars when not provided)
        b_tolerance = tol_per_sample[batch_idx].item() if tol_per_sample is not None else tolerance
        b_min_mz = (
            min_mz_per_sample[batch_idx].item()
            if min_mz_per_sample is not None
            else tolerance_min_mz
        )
        # Compute m/z matching mask within tolerance
        b_match_mzs = calculate_match_mzs(
            b_true_mzs,
            b_pred_mzs,
            tolerance=b_tolerance,
            relative=relative,
            tolerance_min_mz=b_min_mz,
        )
        b_match_mzs = th.as_tensor(b_match_mzs, device=b_true_mzs.device)

        # Identify which true and predicted peaks have matches
        b_true_match_mzs = th.any(b_match_mzs, dim=1)
        b_pred_match_mzs = th.any(b_match_mzs, dim=0)

        # Filter to only matched peaks to reduce matrix size
        b_true_probs_m = b_true_probs[b_true_match_mzs]
        b_pred_probs_m = b_pred_probs[b_pred_match_mzs]

        # Compute score matrix (Gain)
        # Gain = 0.5 * [ (p+q) log(p+q) - p log p - q log q ]
        p_expand = b_true_probs_m.unsqueeze(1)
        q_expand = b_pred_probs_m.unsqueeze(0)
        sum_pq = p_expand + q_expand

        term1 = sum_pq * th.log(sum_pq + EPS)
        term2 = p_expand * th.log(p_expand + EPS)
        term3 = q_expand * th.log(q_expand + EPS)

        b_score = 0.5 * (term1 - term2 - term3)

        # Mask invalid matches
        valid_matches = b_match_mzs[b_true_match_mzs][:, b_pred_match_mzs]
        # Set invalid matches to a large negative number
        b_score[~valid_matches] = -1e9

        # Hungarian algorithm
        if b_score.numel() == 0:
            # No matched peaks => assignment matrix is empty.
            # Keep `total_gain` connected to the computation graph so downstream
            # loss still has `requires_grad=True` (manual backward would otherwise crash).
            if b_pred_probs.numel() > 0:
                total_gain = b_pred_probs.sum() * 0.0
            elif b_true_probs.numel() > 0:
                total_gain = b_true_probs.sum() * 0.0
            else:
                total_gain = th.tensor(0.0, device=b_score.device, dtype=b_score.dtype)
        else:
            # We detach b_score for the assignment step because the Hungarian algorithm
            # is non-differentiable (calculates indices).
            # Gradients will flow through the values selected from the original b_score below.
            b_true_idxs, b_pred_idxs = linear_sum_assignment(b_score.detach(), maximize=True)

            # Filter invalid matches (where score is -1e9)
            matched_gains = b_score[b_true_idxs, b_pred_idxs]
            valid_mask = matched_gains > -1e8
            if th.any(valid_mask):
                total_gain = th.sum(matched_gains[valid_mask])
            else:
                # All matches invalid; return a graph-connected zero.
                total_gain = b_score.sum() * 0.0

        # Normalized JSD = 1 - Total Gain / log 2
        b_jsd = 1.0 - total_gain / LOG2
        b_jsd = th.clamp(b_jsd, min=0.0, max=1.0)
        jsd_hun[batch_idx] = b_jsd

    return jsd_hun


def sparse_jensen_shannon_divergence(
    true_mzs: th.Tensor,
    true_logprobs: th.Tensor,
    true_batch_idxs: th.Tensor,
    pred_mzs: th.Tensor,
    pred_logprobs: th.Tensor,
    pred_batch_idxs: th.Tensor,
    mz_max: float = 1500.0,
    mz_bin_res: float = 0.01,
    log_min: float = EPS,
) -> th.Tensor:
    """
    Compute the sparse Jensen-Shannon divergence between two sets of spectra.

    Args:
        true_mzs: Ground truth m/z values.
        true_logprobs: Ground truth log-probabilities (intensities).
        true_batch_idxs: Batch indices for ground truth peaks.
        pred_mzs: Predicted m/z values.
        pred_logprobs: Predicted log-probabilities (intensities).
        pred_batch_idxs: Batch indices for predicted peaks.
        mz_max: Maximum m/z value for binning.
        mz_bin_res: Resolution of m/z bins.
        log_min: Minimum log value to avoid log(0).

    Note:
        The implicit spectrum support size is
        ``num_bins = round(mz_max / mz_bin_res)``. With defaults,
        this is 150,000 bins.

    Returns:
        Jensen-Shannon divergence (distance) between the binned spectra, per batch.
    """

    # Bin the ground truth peaks into sparse bins using batched_bin_func
    true_bin_idxs, true_bin_logprobs, true_bin_batch_idxs = batched_bin_func(
        true_mzs,
        true_logprobs,
        true_batch_idxs,
        mz_max=mz_max,
        mz_bin_res=mz_bin_res,
        agg="lse",  # Use log-sum-exp aggregation for intensities
        sparse=True,  # Use sparse representation for efficiency
    )
    # Bin the predicted peaks into sparse bins
    pred_bin_idxs, pred_bin_logprobs, pred_bin_batch_idxs = batched_bin_func(
        pred_mzs,
        pred_logprobs,
        pred_batch_idxs,
        mz_max=mz_max,
        mz_bin_res=mz_bin_res,
        agg="lse",
        sparse=True,
    )
    # Calculate Jensen-Shannon similarity using the helper function
    jss = jss_helper(
        true_bin_idxs,
        true_bin_logprobs.exp(),  # Convert log-probabilities to probabilities
        true_bin_batch_idxs,
        pred_bin_idxs,
        pred_bin_logprobs.exp(),
        pred_bin_batch_idxs,
        log_min=log_min,
    )
    # Return 1 - similarity to get the divergence (distance)
    return 1.0 - jss


###########################################
# clip loss function
###########################################


def clip_loss(spectrum_embeddings, mol_embeddings, temperature=0.07):
    """
    Compute the CLIP contrastive loss between spectrum and molecule embeddings.

    Embeddings must be L2-normalized before calling this function (e.g. via
    F.normalize). The dot product then equals cosine similarity.

    Args:
        spectrum_embeddings: torch.Tensor of shape [batch_size, embedding_dim], L2-normalized.
        mol_embeddings: torch.Tensor of shape [batch_size, embedding_dim], L2-normalized.
        temperature: scalar temperature parameter for scaling logits.

    Returns:
        loss: scalar, the symmetric contrastive loss
    """
    logits_per_spectrum = spectrum_embeddings @ mol_embeddings.t() / temperature
    logits_per_mol = logits_per_spectrum.t()

    batch_size = spectrum_embeddings.size(0)
    labels = th.arange(batch_size, device=spectrum_embeddings.device)

    loss = (F.cross_entropy(logits_per_spectrum, labels) + F.cross_entropy(logits_per_mol, labels)) / 2
    return loss


###########################################
# Pairwise loss functions
###########################################


def _build_dense_chunk(
    sorted_batch_idxs: th.Tensor,
    sorted_bin_idxs: th.Tensor,
    sorted_vals: th.Tensor,
    row_ptr_cpu: th.Tensor,
    row_start: int,
    row_end: int,
    num_unique_bins: int,
    device: th.device,
) -> th.Tensor:
    """Build a dense chunk [row_end-row_start, num_unique_bins] from sorted COO data."""
    chunk = th.zeros((row_end - row_start, num_unique_bins), device=device, dtype=sorted_vals.dtype)
    start_idx = int(row_ptr_cpu[row_start])
    end_idx = int(row_ptr_cpu[row_end])
    if end_idx <= start_idx:
        return chunk
    rows = sorted_batch_idxs[start_idx:end_idx] - row_start
    cols = sorted_bin_idxs[start_idx:end_idx]
    chunk[rows, cols] = sorted_vals[start_idx:end_idx]
    return chunk


def get_pairwise_cossim(
    mzs: th.Tensor,
    logprobs: th.Tensor,
    batch_idxs: th.Tensor,
    batch_size: int,
    mz_max: float,
    mz_bin_res: float,
    chunk_size: int = 32,
) -> th.Tensor:
    """Compute pairwise cosine similarity matrix for a batch of spectra.

    Avoids materializing the full [batch_size, num_unique_bins] dense matrix by
    building chunk-sized row slices on the fly. Peak GPU memory for the spectrum
    vectors is O(chunk_size * num_unique_bins) instead of O(batch_size * num_unique_bins),
    which prevents OOM when batch_size is large.

    Args:
        mzs: m/z values.
        logprobs: log-probabilities (intensities).
        batch_idxs: Batch indices.
        batch_size: Number of spectra in the batch.
        mz_max: Maximum m/z value.
        mz_bin_res: m/z bin resolution.
        chunk_size: Number of rows to process at once. Controls the memory/speed
            trade-off: smaller values use less memory but more iterations.

    Note:
        The nominal full spectrum grid size is
        ``num_bins = mz_max / mz_bin_res`` and must be an integer.
        For ``mz_max=1500.0`` and ``mz_bin_res=0.01``, this is 150,000 bins.
        This function compacts to occupied bins to reduce memory.

    Returns:
        Pairwise similarity matrix of shape [batch_size, batch_size].
    """
    batch_size = int(batch_size)
    if mzs.numel() == 0:
        return th.zeros((batch_size, batch_size), device=mzs.device, dtype=logprobs.dtype)

    # Binning
    bin_idxs, bin_logprobs, bin_batch_idxs = batched_bin_func(
        mzs, logprobs, batch_idxs, mz_max=mz_max, mz_bin_res=mz_bin_res, agg="lse", sparse=True
    )

    # Normalize (L2 on probabilities)
    norm_logprobs = scatter_logl2normalize(bin_logprobs, bin_batch_idxs)
    norm_probs = norm_logprobs.exp()

    # batched_bin_func returns batch-offset bin indices (batch element k uses range [k*num_bins,
    # (k+1)*num_bins)). Strip the offset to get per-spectrum bin positions so that the same m/z
    # bin in different spectra maps to the same column in the V matrix, enabling valid dot-product
    # cosine similarity. Without this, every spectrum occupies exclusively different columns and
    # all off-diagonal similarities are zero by construction.
    num_bins = validate_bin_geometry(mz_max, mz_bin_res)
    intra_bin_idxs = bin_idxs % num_bins

    # Compact bin space: avoids 150k-wide matrices when only a few bins are occupied
    unique_bins = th.unique(intra_bin_idxs)
    num_unique_bins = unique_bins.numel()
    bin_idx_map = th.full(
        (int(intra_bin_idxs.max().item()) + 1,), -1, device=mzs.device, dtype=th.long
    )
    bin_idx_map[unique_bins] = th.arange(num_unique_bins, device=mzs.device, dtype=th.long)
    compact_bin_idxs = bin_idx_map[intra_bin_idxs]

    effective_chunk = batch_size if (chunk_size is None or chunk_size >= batch_size) else chunk_size

    sort_idx = th.argsort(bin_batch_idxs)
    sorted_batch_idxs = bin_batch_idxs[sort_idx]
    sorted_bin_idxs = compact_bin_idxs[sort_idx]
    sorted_probs = norm_probs[sort_idx]
    counts = th.bincount(sorted_batch_idxs, minlength=batch_size)
    row_ptr = th.zeros(batch_size + 1, device=mzs.device, dtype=th.long)
    row_ptr[1:] = th.cumsum(counts, dim=0)
    row_ptr_cpu = row_ptr.detach().cpu()

    S = th.zeros((batch_size, batch_size), device=mzs.device, dtype=norm_probs.dtype)

    # Exploit symmetry: only compute upper triangle (j >= i) and mirror.
    # V_i is built once per i-chunk; V_j is built once per (i, j) pair with j >= i.
    for i in range(0, batch_size, effective_chunk):
        i_end = min(i + effective_chunk, batch_size)
        V_i = _build_dense_chunk(
            sorted_batch_idxs,
            sorted_bin_idxs,
            sorted_probs,
            row_ptr_cpu,
            i,
            i_end,
            num_unique_bins,
            mzs.device,
        )
        for j in range(i, batch_size, effective_chunk):
            j_end = min(j + effective_chunk, batch_size)
            if i == j:
                block = th.mm(V_i, V_i.t())
                S[i:i_end, i:i_end] = block
            else:
                V_j = _build_dense_chunk(
                    sorted_batch_idxs,
                    sorted_bin_idxs,
                    sorted_probs,
                    row_ptr_cpu,
                    j,
                    j_end,
                    num_unique_bins,
                    mzs.device,
                )
                block = th.mm(V_i, V_j.t())
                S[i:i_end, j:j_end] = block
                S[j:j_end, i:i_end] = block.t()

    return S


def get_pairwise_jss_sim(
    mzs: th.Tensor,
    logprobs: th.Tensor,
    batch_idxs: th.Tensor,
    batch_size: int,
    mz_max: float,
    mz_bin_res: float,
    chunk_size: int = 32,
) -> th.Tensor:
    """Compute pairwise Jensen-Shannon similarity matrix for a batch of spectra.

    Avoids materializing the full [batch_size, num_unique_bins] dense matrix by
    building chunk-sized row slices on the fly.

    Args:
        mzs: m/z values.
        logprobs: log-probabilities (intensities).
        batch_idxs: Batch indices.
        batch_size: Number of spectra in the batch.
        mz_max: Maximum m/z value.
        mz_bin_res: m/z bin resolution.
        chunk_size: Number of rows to process at once. Controls the memory/speed
            trade-off: smaller values use less memory but more iterations.

    Note:
        The nominal full spectrum grid size is
        ``num_bins = mz_max / mz_bin_res`` and must be an integer.
        For ``mz_max=1500.0`` and ``mz_bin_res=0.01``, this is 150,000 bins.
        This function compacts to occupied bins to reduce memory.

    Returns:
        Pairwise similarity matrix of shape [batch_size, batch_size].
    """
    batch_size = int(batch_size)
    if mzs.numel() == 0:
        return th.zeros((batch_size, batch_size), device=mzs.device, dtype=logprobs.dtype)

    # Binning
    bin_idxs, bin_logprobs, bin_batch_idxs = batched_bin_func(
        mzs, logprobs, batch_idxs, mz_max=mz_max, mz_bin_res=mz_bin_res, agg="lse", sparse=True
    )

    # Normalize (L1 on probabilities)
    log_probs = scatter_logsoftmax(bin_logprobs, bin_batch_idxs)
    probs = log_probs.exp()

    # Strip batch offsets (same fix as get_pairwise_cossim): batched_bin_func returns
    # batch-offset bin indices; without stripping, each spectrum occupies exclusively
    # different columns, making all pairwise mixtures trivially equal to 0.5*(P+0)=0.5*P
    # and H(M) collapses to a function of individual entropies only.
    num_bins = validate_bin_geometry(mz_max, mz_bin_res)
    intra_bin_idxs = bin_idxs % num_bins

    # Compact bin space
    unique_bins = th.unique(intra_bin_idxs)
    num_unique_bins = unique_bins.numel()
    bin_idx_map = th.full(
        (int(intra_bin_idxs.max().item()) + 1,), -1, device=mzs.device, dtype=th.long
    )
    bin_idx_map[unique_bins] = th.arange(num_unique_bins, device=mzs.device, dtype=th.long)
    compact_bin_idxs = bin_idx_map[intra_bin_idxs]

    effective_chunk = batch_size if (chunk_size is None or chunk_size >= batch_size) else chunk_size

    # Build full [batch_size, num_unique_bins] probability matrix.
    # batched_bin_func + scatter_logsoftmax guarantee unique (batch, bin) pairs,
    # so plain index assignment is safe (no duplicate scatter).
    P = th.zeros((batch_size, num_unique_bins), device=mzs.device, dtype=probs.dtype)
    P[bin_batch_idxs, compact_bin_idxs] = probs

    H_P = -(P * th.log(P + EPS)).sum(dim=1)  # [batch_size]

    # H(M) where M = 0.5*(P_i + P_j): exploit upper-triangle symmetry.
    # Each block is materialized as [ci, cj, K] and reduced along K in one
    # CUDA kernel — no inner Python loop, no row_ptr scaffolding.
    H_M = th.zeros((batch_size, batch_size), device=mzs.device, dtype=probs.dtype)
    for i in range(0, batch_size, effective_chunk):
        i_end = min(i + effective_chunk, batch_size)
        P_i = P[i:i_end]  # [ci, K]
        for j in range(i, batch_size, effective_chunk):
            j_end = min(j + effective_chunk, batch_size)
            P_j = P[j:j_end]  # [cj, K]
            M = 0.5 * (P_i.unsqueeze(1) + P_j.unsqueeze(0))  # [ci, cj, K]
            block = -(M * th.log(M + EPS)).sum(dim=2)  # [ci, cj]
            H_M[i:i_end, j:j_end] = block
            if i != j:
                H_M[j:j_end, i:i_end] = block.t()

    JSD = H_M - 0.5 * H_P.unsqueeze(1) - 0.5 * H_P.unsqueeze(0)
    # Clamp guards against tiny negative JSD (EPS perturbation in log) giving JSS > 1.
    JSS = th.clamp(1.0 - JSD / math.log(2.0), min=0.0, max=1.0)

    return JSS


def get_pairwise_cross_entropy(
    mzs: th.Tensor,
    logprobs: th.Tensor,
    batch_idxs: th.Tensor,
    batch_size: int,
    mz_max: float,
    mz_bin_res: float,
    chunk_size: int = 32,
) -> th.Tensor:
    """
    Compute pairwise cross-entropy matrix for a batch of spectra.
    Cross-Entropy(P, Q) = - sum(P * log(Q))

    Args:
        mzs: m/z values.
        logprobs: log-probabilities (intensities).
        batch_idxs: Batch indices.
        batch_size: Number of spectra in the batch.
        mz_max: Maximum m/z value.
        mz_bin_res: m/z bin resolution.
        chunk_size: Batch size for chunked computation to save memory.

    Note:
        The nominal full spectrum grid size is
        ``num_bins = mz_max / mz_bin_res`` and must be an integer.
        For ``mz_max=1500.0`` and ``mz_bin_res=0.01``, this is 150,000 bins.
        Bins are compacted to occupied indices before sparse/dense assembly.

    Returns:
        Pairwise cross-entropy matrix of shape [batch_size, batch_size].
    """

    batch_size = int(batch_size)
    if mzs.numel() == 0:
        return th.zeros((batch_size, batch_size), device=mzs.device, dtype=logprobs.dtype)

    # Binning
    bin_idxs, bin_logprobs, bin_batch_idxs = batched_bin_func(
        mzs, logprobs, batch_idxs, mz_max=mz_max, mz_bin_res=mz_bin_res, agg="lse", sparse=True
    )

    # Normalize to get probabilities (L1 normalization)
    log_probs = scatter_logsoftmax(bin_logprobs, bin_batch_idxs)
    probs = log_probs.exp()

    # Strip batch offsets: batched_bin_func returns batch-offset bin indices; without
    # stripping, each spectrum occupies exclusively different columns so P and Q never
    # share any bin and cross-entropy reduces to -sum(P*log(eps)) for all off-diagonal
    # pairs — masking any real signal from shared peaks.
    num_bins = validate_bin_geometry(mz_max, mz_bin_res)
    intra_bin_idxs = bin_idxs % num_bins

    unique_bins = th.unique(intra_bin_idxs)
    num_unique_bins = unique_bins.numel()
    bin_idx_map = th.full(
        (int(intra_bin_idxs.max().item()) + 1,), -1, device=mzs.device, dtype=th.long
    )
    bin_idx_map[unique_bins] = th.arange(num_unique_bins, device=mzs.device, dtype=th.long)
    compact_bin_idxs = bin_idx_map[intra_bin_idxs]

    effective_chunk = batch_size if (chunk_size is None or chunk_size >= batch_size) else chunk_size

    sort_idx = th.argsort(bin_batch_idxs)
    sorted_batch_idxs = bin_batch_idxs[sort_idx]
    sorted_bin_idxs = compact_bin_idxs[sort_idx]
    sorted_probs = probs[sort_idx]
    counts = th.bincount(sorted_batch_idxs, minlength=batch_size)
    row_ptr = th.zeros(batch_size + 1, device=mzs.device, dtype=th.long)
    row_ptr[1:] = th.cumsum(counts, dim=0)
    row_ptr_cpu = row_ptr.detach().cpu()

    CE = th.zeros((batch_size, batch_size), device=mzs.device, dtype=probs.dtype)

    for i in range(0, batch_size, effective_chunk):
        i_end = min(i + effective_chunk, batch_size)
        P_i = _build_dense_chunk(
            sorted_batch_idxs,
            sorted_bin_idxs,
            sorted_probs,
            row_ptr_cpu,
            i,
            i_end,
            num_unique_bins,
            mzs.device,
        )
        log_P_i = th.log(P_i + EPS)
        for j in range(0, batch_size, effective_chunk):
            j_end = min(j + effective_chunk, batch_size)
            # Reuse log_P_i for diagonal blocks (j == i) to avoid rebuilding the same chunk.
            log_Q_j = (
                log_P_i
                if i == j
                else th.log(
                    _build_dense_chunk(
                        sorted_batch_idxs,
                        sorted_bin_idxs,
                        sorted_probs,
                        row_ptr_cpu,
                        j,
                        j_end,
                        num_unique_bins,
                        mzs.device,
                    )
                    + EPS
                )
            )
            CE[i:i_end, j:j_end] = -th.mm(P_i, log_Q_j.t())

    return CE


############################################
# Pairwise with with Hungarian matching
############################################


def _compute_pair_jss_hun(
    b_true_mzs: th.Tensor,
    b_true_logprobs: th.Tensor,
    b_pred_mzs: th.Tensor,
    b_pred_logprobs: th.Tensor,
    tolerance: float,
    relative: bool,
    tolerance_min_mz: float,
) -> th.Tensor:
    """
    Helper to compute Jensen-Shannon similarity for a single pair of spectra using Hungarian matching.
    Returns scalar tensor (Similarity in [0, 1]).
    """
    # Normalize probabilities (sum to 1)
    b_true_probs = th.exp(b_true_logprobs)
    b_true_sum = th.sum(b_true_probs)
    if b_true_sum > 0:
        b_true_probs = b_true_probs / b_true_sum

    b_pred_probs = th.exp(b_pred_logprobs)
    b_pred_sum = th.sum(b_pred_probs)
    if b_pred_sum > 0:
        b_pred_probs = b_pred_probs / b_pred_sum

    # Compute m/z matching mask within tolerance
    b_match_mzs = calculate_match_mzs(
        b_true_mzs,
        b_pred_mzs,
        tolerance=tolerance,
        relative=relative,
        tolerance_min_mz=tolerance_min_mz,
    )
    b_match_mzs = th.as_tensor(b_match_mzs, device=b_true_mzs.device)

    # Identify which true and predicted peaks have matches
    b_true_match_mzs = th.any(b_match_mzs, dim=1)
    b_pred_match_mzs = th.any(b_match_mzs, dim=0)

    # Filter to only matched peaks
    b_true_probs_m = b_true_probs[b_true_match_mzs]
    b_pred_probs_m = b_pred_probs[b_pred_match_mzs]

    # Compute score matrix (Gain)
    p_expand = b_true_probs_m.unsqueeze(1)
    q_expand = b_pred_probs_m.unsqueeze(0)
    sum_pq = p_expand + q_expand
    term1 = sum_pq * th.log(sum_pq + EPS)
    term2 = p_expand * th.log(p_expand + EPS)
    term3 = q_expand * th.log(q_expand + EPS)
    b_score = 0.5 * (term1 - term2 - term3)

    # Mask invalid matches
    valid_matches = b_match_mzs[b_true_match_mzs][:, b_pred_match_mzs]
    b_score[~valid_matches] = -1e9

    # Hungarian algorithm
    if b_score.numel() == 0:
        if b_pred_probs.numel() > 0:
            total_gain = b_pred_probs.sum() * 0.0
        elif b_true_probs.numel() > 0:
            total_gain = b_true_probs.sum() * 0.0
        else:
            total_gain = th.tensor(0.0, device=b_score.device, dtype=b_score.dtype)
    else:
        b_true_idxs, b_pred_idxs = linear_sum_assignment(b_score.detach(), maximize=True)
        matched_gains = b_score[b_true_idxs, b_pred_idxs]
        valid_mask = matched_gains > -1e8
        if th.any(valid_mask):
            total_gain = th.sum(matched_gains[valid_mask])
        else:
            total_gain = b_score.sum() * 0.0

    LOG2 = math.log(2.0)
    b_jss = total_gain / LOG2
    b_jss = th.clamp(b_jss, min=0.0, max=1.0)
    return b_jss


def get_pairwise_jss_sim_hun(
    mzs: th.Tensor,
    logprobs: th.Tensor,
    batch_idxs: th.Tensor,
    batch_size: int,
    tolerance: float = 1e-5,
    relative: bool = True,
    tolerance_min_mz: float = TOLERANCE_MIN_MZ,
) -> th.Tensor:
    """
    Compute pairwise Jensen-Shannon Similarity matrix for a batch of spectra using Hungarian matching.
    Sim = 1 - JSD.

    Args:
        mzs: m/z values.
        logprobs: log-probabilities (intensities).
        batch_idxs: Batch indices.
        batch_size: Number of spectra.
        tolerance: m/z matching tolerance.
        relative: Whether tolerance is relative.
        tolerance_min_mz: Minimum m/z for relative tolerance.

    Returns:
        Matrix of shape [batch_size, batch_size] containing pairwise similarities.
    """

    batch_size = int(batch_size)
    # Pre-aggregate peaks to save work in loop
    mzs, logprobs, batch_idxs = round_aggregate_peaks(mzs, logprobs, batch_idxs, agg="lse")

    sim_matrix = th.zeros((batch_size, batch_size), device=mzs.device, dtype=logprobs.dtype)

    # Separate spectra using split (O(num_peaks) vs O(batch_size * num_peaks) mask loop)
    counts = th.bincount(batch_idxs, minlength=batch_size).tolist()
    spectra_mzs = list(th.split(mzs, counts))
    spectra_logprobs = list(th.split(logprobs, counts))

    for i in range(batch_size):
        # Self similarity is always 1.
        sim_matrix[i, i] = 1.0

        for j in range(i + 1, batch_size):
            sim = _compute_pair_jss_hun(
                spectra_mzs[i],
                spectra_logprobs[i],
                spectra_mzs[j],
                spectra_logprobs[j],
                tolerance,
                relative,
                tolerance_min_mz,
            )
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

    return sim_matrix


def get_pairwise_cross_entropy_hun(
    mzs: th.Tensor,
    logprobs: th.Tensor,
    batch_idxs: th.Tensor,
    batch_size: int,
    tolerance: float = 1e-5,
    relative: bool = True,
    tolerance_min_mz: float = TOLERANCE_MIN_MZ,
    chunk_size: int = 32,
) -> th.Tensor:
    """
    Compute pairwise Cross Entropy matrix for a batch of spectra using Hungarian matching.
    CE(P, Q) = - sum P_i log Q_matching_i.

    Args:
        mzs: m/z values.
        logprobs: log-probabilities (intensities).
        batch_idxs: Batch indices.
        batch_size: Number of spectra.
        tolerance: m/z matching tolerance.
        relative: Whether tolerance is relative.
        tolerance_min_mz: Minimum m/z for relative tolerance.
        chunk_size: Ignored in this iterative implementation, but kept for API compatibility.

    Returns:
        Matrix of shape [batch_size, batch_size] containing pairwise cross entropies.
    """

    batch_size = int(batch_size)
    # Pre-aggregate peaks to save work in loop
    mzs, logprobs, batch_idxs = round_aggregate_peaks(mzs, logprobs, batch_idxs, agg="lse")

    ce_matrix = th.zeros((batch_size, batch_size), device=mzs.device, dtype=logprobs.dtype)

    # Separate spectra using split (O(num_peaks) vs O(batch_size * num_peaks) mask loop)
    counts = th.bincount(batch_idxs, minlength=batch_size).tolist()
    spectra_mzs = list(th.split(mzs, counts))
    spectra_logprobs = list(th.split(logprobs, counts))

    # Fill matrix
    # CE is NOT symmetric. Loop over all pairs.
    # Iterate over all i, j.
    for i in range(batch_size):
        for j in range(batch_size):
            ce_matrix[i, j] = _compute_pair_ce_hun(
                spectra_mzs[i],
                spectra_logprobs[i],
                spectra_mzs[j],
                spectra_logprobs[j],
                tolerance,
                relative,
                tolerance_min_mz,
            )
            # Optimization: could skip self-check if needed, but good for validation.

    return ce_matrix


def _compute_pair_ce_hun(
    b_true_mzs: th.Tensor,
    b_true_logprobs: th.Tensor,
    b_pred_mzs: th.Tensor,
    b_pred_logprobs: th.Tensor,
    tolerance: float,
    relative: bool,
    tolerance_min_mz: float,
) -> th.Tensor:
    """
    Helper to compute Cross Entropy for a single pair of spectra using Hungarian matching.
    CE(P, Q) = - sum P_i log Q_matching_i
    If no match, log Q is effectively log(epsilon).
    Returns scalar tensor.
    """
    # Normalize probabilities (sum to 1)
    b_true_probs = th.exp(b_true_logprobs)
    b_true_sum = th.sum(b_true_probs)
    if b_true_sum > 0:
        b_true_probs = b_true_probs / b_true_sum

    b_pred_probs = th.exp(b_pred_logprobs)
    b_pred_sum = th.sum(b_pred_probs)
    if b_pred_sum > 0:
        b_pred_probs = b_pred_probs / b_pred_sum

    # If P is empty (sum=0), CE = 0 (convention, 0 * log(Q))
    if b_true_sum == 0:
        return th.tensor(0.0, device=b_true_mzs.device, dtype=b_true_logprobs.dtype)

    # Q needs to be in log space for CE calculation.
    # But we need log(Q) where Q matches P.
    # Base contribution (if P_i has no match in Q): P_i * log(epsilon)
    # Gain of matching P_i to Q_j: P_i * log(Q_j) - P_i * log(epsilon) = P_i * (log(Q_j) - log(epsilon))
    # We want to MAXIMIZE sum(P_i * log(Q_matching_i)).

    log_eps = th.log(th.tensor(EPS, device=b_true_mzs.device, dtype=b_true_logprobs.dtype))

    # If Q is empty (sum=0), all matches are missing.
    # CE = - sum P_i * log(eps) = - 1.0 * log(eps) = -log(eps) (since P is normalized)
    if b_pred_sum == 0:
        return -log_eps

    # Compute m/z matching mask within tolerance
    b_match_mzs = calculate_match_mzs(
        b_true_mzs,
        b_pred_mzs,
        tolerance=tolerance,
        relative=relative,
        tolerance_min_mz=tolerance_min_mz,
    )
    b_match_mzs = th.as_tensor(b_match_mzs, device=b_true_mzs.device)

    # Identify which true and predicted peaks have matches
    b_true_match_mzs = th.any(b_match_mzs, dim=1)
    b_pred_match_mzs = th.any(b_match_mzs, dim=0)

    # Filter to only matched peaks
    b_true_probs_m = b_true_probs[b_true_match_mzs]
    b_pred_probs_m = b_pred_probs[b_pred_match_mzs]

    # Compute Score (Gain) Matrix
    # Gain(i, j) = P_i * (log(Q_j) - log(eps))
    # Note: Q_j are probabilities.
    # We use log(Q_j + eps) to avoid log(0) if Q_j is very small, though here Q is normalized props.
    q_log_m = th.log(b_pred_probs_m + EPS)

    # Outer product logic: P_i * (log Q_j - log_eps)
    # P_i is (N,). log_Q_j is (M,).
    # Gain is (N, M).
    gain_matrix = b_true_probs_m.unsqueeze(1) * (q_log_m.unsqueeze(0) - log_eps)

    # Mask invalid matches
    valid_matches = b_match_mzs[b_true_match_mzs][:, b_pred_match_mzs]

    # Set invalid matches to a large negative number
    # Gain can be negative if log(Q) < log(eps), but unlikely given eps is small.
    # But to prevent selection, use -infinity.
    gain_matrix[~valid_matches] = -1e9

    # Hungarian algorithm
    if gain_matrix.numel() == 0:
        total_gain = th.tensor(0.0, device=gain_matrix.device, dtype=gain_matrix.dtype)
    else:
        b_true_idxs, b_pred_idxs = linear_sum_assignment(gain_matrix.detach(), maximize=True)

        matched_gains = gain_matrix[b_true_idxs, b_pred_idxs]
        valid_mask = matched_gains > -1e8
        if th.any(valid_mask):
            total_gain = th.sum(matched_gains[valid_mask])
        else:
            total_gain = th.tensor(0.0, device=gain_matrix.device, dtype=gain_matrix.dtype)

    # Total Score (sum P log Q) = Total Gain + Base Score
    # Base Score = sum_i P_i * log(eps) = 1.0 * log(eps) = log(eps)
    total_score = total_gain + log_eps

    # CE = - Total Score
    ce = -total_score
    return ce
