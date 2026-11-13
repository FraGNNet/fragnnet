"""Candidate-retrieval scoring: spectrum similarities, hit rates and bootstrap intervals.

Scores a query (measured) spectrum against predicted spectra of retrieval candidates. Three
families of similarity are provided:

* **ms-pred replicas** (``mspred_*``): the binned cosine and entropy similarity of ms-pred's
  ``retrieval/retrieval_benchmark.py`` at commit ``318bb7b``, which GLACIER's published
  MassSpecGym retrieval numbers use. Targets are max-normalized, square-rooted and binned with
  max pooling; predictions are square-rooted intensities binned with sum pooling; bins are
  ``floor(mz * (num_bins - 1) / upper_limit) + 1``.
* **Entropy similarity at instrument tolerance** (:func:`entropy_similarity_inst_tol`): the same
  unweighted entropy similarity on sum-normalized sqrt intensities, but with peaks matched one to
  one within a per-peak m/z tolerance instead of binned.
* **Cosine at instrument tolerance** (:func:`cosine_hungarian_inst_tol`): the training
  ``cos_hun_inst_tol`` metric, linear or sqrt intensities, peaks matched one to one.
* **Jensen-Shannon similarity at instrument tolerance** (:func:`jss_hungarian_inst_tol`): the
  training ``jss_hun_inst_tol`` metric, linear or sqrt intensities, peaks matched one to one.
* **Query log-likelihood** (:func:`query_log_likelihood`): the query's log-likelihood under the
  full predicted distribution, with predicted intensity summed within the tolerance window.

Rankings use :func:`expected_hits` (expected hit@k under random tie-breaking; any truth counts),
and intervals use :func:`molecule_bootstrap_ci` (resampling molecules, so correlated spectra of
one molecule move together) or :func:`mspred_bootstrap_ci` (ms-pred's i.i.d. BCa replica).
"""

from __future__ import annotations

from math import comb

import numpy as np
import torch as th
from scipy.stats import bootstrap

from fragnnet.utils.spec_utils import (
    _solve_batch_assignment,
    batch_cos_hun_helper,
    batch_jss_hun_helper,
    calculate_match_mzs,
    round_aggregate_peaks,
)

MSPRED_NUM_BINS = 150000
MSPRED_UPPER_LIMIT = 1500.0
LN4 = float(np.log(4.0))


def x_log_x(values: th.Tensor) -> th.Tensor:
    """Return ``x log x`` with the ``x = 0`` limit, so padded entries contribute nothing."""
    return values * values.clamp_min(th.finfo(values.dtype).tiny).log()


# ---------------------------------------------------------------------------------------------
# ms-pred replicas
# ---------------------------------------------------------------------------------------------


def mspred_bin(
    mzs: np.ndarray,
    intens: np.ndarray,
    pool: str,
    num_bins: int = MSPRED_NUM_BINS,
    upper_limit: float = MSPRED_UPPER_LIMIT,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin a peak list the way ms-pred does (``common.bin_spectra`` / ``_bin_spec_sparse``).

    Bin index ``floor(mz * (num_bins - 1) / upper_limit) + 1``; indices outside
    ``[0, num_bins)`` are dropped; empty bins are omitted.

    Args:
        mzs: Peak m/z, shape ``(n,)``.
        intens: Peak intensities, shape ``(n,)``.
        pool: ``"max"`` (ms-pred targets) or ``"add"`` (ms-pred predictions).
        num_bins: Number of bins (150000 gives 0.01 Da bins up to 1500).
        upper_limit: Upper m/z limit.

    Returns:
        ``(bin_indices, values)``, sorted by bin index, values > 0.

    Raises:
        ValueError: If ``pool`` is not ``"max"`` or ``"add"``.
    """
    if pool not in ("max", "add"):
        raise ValueError(f"pool must be 'max' or 'add', got {pool!r}")
    mzs = np.asarray(mzs, dtype=np.float64)
    intens = np.asarray(intens, dtype=np.float64)
    idx = np.floor(mzs * ((num_bins - 1) / upper_limit)).astype(np.int64) + 1
    ok = (idx >= 0) & (idx < num_bins)
    idx, intens = idx[ok], intens[ok]
    uniq, inv = np.unique(idx, return_inverse=True)
    vals = np.zeros(len(uniq))
    (np.maximum if pool == "max" else np.add).at(vals, inv, intens)
    keep = vals > 0
    return uniq[keep], vals[keep]


def mspred_target(
    mzs: np.ndarray, intens: np.ndarray, parent_mass: float
) -> tuple[np.ndarray, np.ndarray]:
    """ms-pred's processed target: drop m/z > parent_mass + 1, max-normalize, sqrt, max-pool bins.

    Mirrors ``common.process_spec_file`` (``merge_specs=False``) followed by the retrieval
    benchmark's ``bin_spectra`` call (max pooling).

    Args:
        mzs: Measured peak m/z, shape ``(n,)``.
        intens: Measured intensities (any scale), shape ``(n,)``.
        parent_mass: The value ms-pred's labels carry as ``parentmass`` (precursor m/z here).

    Returns:
        ``(bin_indices, sqrt_values)`` as from :func:`mspred_bin`.
    """
    mzs = np.asarray(mzs, dtype=np.float64)
    intens = np.asarray(intens, dtype=np.float64)
    keep = (mzs <= parent_mass + 1) & (intens > 0)
    mzs, intens = mzs[keep], intens[keep]
    if len(intens) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    return mspred_bin(mzs, np.sqrt(intens / intens.max()), "max")


def mspred_prediction(
    mzs: np.ndarray, intens: np.ndarray, sqrt_input: bool, min_rel_intensity: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """ms-pred's binned prediction: sqrt intensities, sum-pooled bins.

    Args:
        mzs: Predicted peak m/z, shape ``(n,)``.
        intens: Predicted intensities, shape ``(n,)``. Linear unless ``sqrt_input``.
        sqrt_input: True when ``intens`` is already sqrt intensity (GLACIER/ms-pred output).
        min_rel_intensity: If set, drop peaks whose *linear* intensity is below this fraction of
            the spectrum's base peak before binning (e.g. 0.01 for a 1% filter).

    Returns:
        ``(bin_indices, sqrt_values)`` as from :func:`mspred_bin`.
    """
    mzs = np.asarray(mzs, dtype=np.float64)
    lin = np.asarray(intens, dtype=np.float64)
    lin = lin**2 if sqrt_input else lin
    keep = lin > 0
    mzs, lin = mzs[keep], lin[keep]
    if min_rel_intensity is not None and len(lin):
        keep = lin >= min_rel_intensity * lin.max()
        mzs, lin = mzs[keep], lin[keep]
    return mspred_bin(mzs, np.sqrt(lin), "add")


def mspred_cosine(
    target: tuple[np.ndarray, np.ndarray], pred: tuple[np.ndarray, np.ndarray]
) -> float:
    """ms-pred's sparse cosine similarity (``_cosine_sparse``) on binned sqrt intensities.

    ``sum_b t_b p_b / ((||t|| + 1e-22)(||p|| + 1e-22))``; an empty prediction scores 0.

    Args:
        target: ``(bin_indices, values)`` from :func:`mspred_target`.
        pred: ``(bin_indices, values)`` from :func:`mspred_prediction`.

    Returns:
        Cosine similarity in ``[0, 1]``.
    """
    (ti, tv), (pi, pv) = target, pred
    if len(pv) == 0:
        return 0.0
    _, a, b = np.intersect1d(ti, pi, assume_unique=True, return_indices=True)
    return float((tv[a] * pv[b]).sum() / ((np.sqrt(tv @ tv) + 1e-22) * (np.sqrt(pv @ pv) + 1e-22)))


def _entropy(probs: np.ndarray) -> float:
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs + 1e-22)).sum())


def mspred_entropy(
    target: tuple[np.ndarray, np.ndarray], pred: tuple[np.ndarray, np.ndarray]
) -> float:
    """ms-pred's unweighted spectral entropy similarity (``_entropy_sparse``) on binned values.

    With ``q``, ``p`` the sum-normalized target and prediction and ``M = (p + q) / 2``:
    ``S = 1 - (2 H(M) - H(p) - H(q)) / ln 4``, natural-log entropies. An empty prediction scores 0.

    Args:
        target: ``(bin_indices, values)`` from :func:`mspred_target`.
        pred: ``(bin_indices, values)`` from :func:`mspred_prediction`.

    Returns:
        Entropy similarity in ``[0, 1]``.
    """
    (ti, tv), (pi, pv) = target, pred
    if len(pv) == 0 or pv.sum() <= 0 or tv.sum() <= 0:
        return 0.0
    q, p = tv / tv.sum(), pv / pv.sum()
    union = np.union1d(ti, pi)
    mix = np.zeros(len(union))
    mix[np.searchsorted(union, ti)] += q / 2
    mix[np.searchsorted(union, pi)] += p / 2
    return 1 - (2 * _entropy(mix) - _entropy(p) - _entropy(q)) / np.log(4)


# ---------------------------------------------------------------------------------------------
# Instrument-tolerance similarities
# ---------------------------------------------------------------------------------------------


def peak_tolerance(mzs: np.ndarray | th.Tensor, rel: float, min_mz: float):
    """Absolute matching tolerance ``rel * max(mz, min_mz)`` (Da), numpy or torch."""
    if isinstance(mzs, th.Tensor):
        return rel * th.maximum(mzs, th.full_like(mzs, min_mz))
    return rel * np.maximum(np.asarray(mzs, dtype=np.float64), min_mz)


def prepare_sqrt_peaks(
    mzs: np.ndarray,
    lin_intens: np.ndarray,
    max_mz: float | None = None,
    min_rel_intensity: float | None = None,
) -> tuple[th.Tensor, th.Tensor]:
    """Peaks for :func:`entropy_similarity_inst_tol`: aggregate, filter, sqrt, sum-normalize.

    Peaks are rounded to 4 decimals and summed in linear intensity (``round_aggregate_peaks``),
    peaks above ``max_mz`` dropped, peaks below ``min_rel_intensity`` of the base peak dropped,
    then intensities are square-rooted and normalized to sum to 1.

    Args:
        mzs: Peak m/z, shape ``(n,)``.
        lin_intens: Linear intensities, shape ``(n,)``.
        max_mz: Drop peaks with m/z above this (e.g. precursor m/z + tolerance); None keeps all.
        min_rel_intensity: Optional base-peak fraction filter (linear intensity).

    Returns:
        ``(mzs, probs)`` float64 CPU tensors, shape ``(k,)``; ``probs`` sums to 1 unless empty.
    """
    m, i, _ = round_aggregate_peaks(
        th.as_tensor(np.asarray(mzs, dtype=np.float64)),
        th.as_tensor(np.asarray(lin_intens, dtype=np.float64)),
        th.zeros(len(mzs), dtype=th.long),
        agg="sum",
    )
    keep = i > 0
    if max_mz is not None:
        keep &= m <= max_mz
    m, i = m[keep], i[keep]
    if min_rel_intensity is not None and len(i):
        keep = i >= min_rel_intensity * i.max()
        m, i = m[keep], i[keep]
    s = th.sqrt(i)
    return m, (s / s.sum() if len(s) else s)


def _chunk_bounds(
    preds: list[tuple[th.Tensor, th.Tensor]], n_t: int, chunk_size: int, max_elems: int
) -> list[tuple[int, int]]:
    """``[c0, c1)`` ranges of at most ``chunk_size`` predictions whose padded ``b * n_t * n_p``
    pair matrix stays under ``max_elems`` (a lone prediction above it gets its own chunk)."""
    bounds, c0, n_p = [], 0, 1
    for c1, (m, _) in enumerate(preds):
        n_p_new = max(n_p, len(m))
        if c1 > c0 and (c1 - c0 >= chunk_size or (c1 - c0 + 1) * max(n_t, 1) * n_p_new > max_elems):
            bounds.append((c0, c1))
            c0, n_p_new = c1, max(1, len(m))
        n_p = n_p_new
    if c0 < len(preds):
        bounds.append((c0, len(preds)))
    return bounds


def entropy_similarity_inst_tol(
    target: tuple[th.Tensor, th.Tensor],
    preds: list[tuple[th.Tensor, th.Tensor]],
    rel: float,
    min_mz: float,
    mz_max: float = MSPRED_UPPER_LIMIT,
    chunk_size: int = 32,
    max_elems: int = 1 << 26,
    device: str | th.device = "cpu",
) -> list[float]:
    """Unweighted entropy similarity with one-to-one peak matching at instrument tolerance.

    For sum-normalized ``q`` (target) and ``p`` (prediction), ms-pred's entropy similarity
    ``1 - (2 H(M) - H(p) - H(q)) / ln 4`` with ``M = (p + q) / 2`` decomposes over matched pairs:
    each matched pair ``(q_i, p_j)`` adds ``w_ij = (q_i + p_j) ln(q_i + p_j) - q_i ln q_i -
    p_j ln p_j``, unmatched peaks add nothing, and ``S = sum(w) / ln 4``. The one-to-one match
    maximizing ``S`` is a linear assignment over pairs within tolerance (target peak tolerance
    ``rel * max(min(mz, mz_max), min_mz)``, the ``cos_hun_inst_tol`` matching).

    Args:
        target: ``(mzs, probs)`` from :func:`prepare_sqrt_peaks`.
        preds: Candidate predictions, each ``(mzs, probs)`` from :func:`prepare_sqrt_peaks`.
        rel: Relative tolerance (1e-5 is 10 ppm).
        min_mz: Floor of the m/z the relative tolerance is applied to.
        mz_max: Cap on the m/z the tolerance is computed from.
        chunk_size: Candidates per padded batch (bounds memory; results do not depend on it).
        max_elems: Cap on a batch's padded target x prediction peak pairs (bounds memory).
        device: Device for the matching and assignment.

    Returns:
        One similarity in ``[0, 1]`` per prediction, in input order.
    """
    tm, tq = target
    out: list[float] = []
    for c0, c1 in _chunk_bounds(preds, len(tm), chunk_size, max_elems):
        chunk = preds[c0:c1]
        b, n_p = len(chunk), max(1, max(len(m) for m, _ in chunk))
        pmz = th.full((b, n_p), float("inf"), dtype=th.float64)
        pp = th.zeros((b, n_p), dtype=th.float64)
        for j, (m, p) in enumerate(chunk):
            pmz[j, : len(m)], pp[j, : len(m)] = m, p
        tmz = tm[None].expand(b, -1).to(device)
        q = tq[None].expand(b, -1).to(device)
        pmz, pp = pmz.to(device), pp.to(device)
        tol = peak_tolerance(tmz.clamp(max=mz_max), rel, min_mz)
        mask = calculate_match_mzs(tmz, pmz, tolerance=0.0, relative=False, tol_per_true=tol)
        # The inputs are probabilities (nonnegative); x_log_x takes the x = 0 limit.
        w = x_log_x(q[:, :, None] + pp[:, None, :]) - x_log_x(q)[:, :, None] - x_log_x(pp)[:, None, :]
        w = w * mask
        if w.shape[1] == 0:
            out += [0.0] * b
            continue
        rows, cols = _solve_batch_assignment(w)
        bi = th.arange(b, device=w.device)[:, None]
        out += (w[bi, rows, cols].sum(-1) / LN4).clamp(0.0, 1.0).cpu().tolist()
    return out


def prepare_hun_peaks(
    mzs: np.ndarray, lin_intens: np.ndarray, sqrt: bool
) -> tuple[th.Tensor, th.Tensor]:
    """Peaks for the ``*_hungarian_inst_tol`` metrics: aggregate as training does, optionally sqrt.

    Peaks are rounded to 4 decimals and summed in linear intensity (``round_aggregate_peaks``,
    as ``spectrum_pl`` does before its ``cos_hun_inst_tol`` metrics); nothing is dropped.

    Returns:
        ``(mzs, intens)`` float64 CPU tensors; ``intens`` is square-rooted when ``sqrt``.
    """
    m, i, _ = round_aggregate_peaks(
        th.as_tensor(np.asarray(mzs, dtype=np.float64)),
        th.as_tensor(np.asarray(lin_intens, dtype=np.float64)),
        th.zeros(len(mzs), dtype=th.long),
        agg="sum",
    )
    keep = i > 0
    m, i = m[keep], i[keep]
    return m, (th.sqrt(i) if sqrt else i)


def cosine_hungarian_inst_tol(
    target: tuple[th.Tensor, th.Tensor],
    preds: list[tuple[th.Tensor, th.Tensor]],
    rel: float,
    min_mz: float,
    mz_max: float = MSPRED_UPPER_LIMIT,
    chunk_size: int = 32,
    max_elems: int = 1 << 26,
    device: str | th.device = "cpu",
) -> list[float]:
    """The training ``cos_hun_inst_tol`` metric: cosine with one-to-one matching at instrument tolerance.

    Both spectra are L2-normalized over all their peaks; peak pairs within the target peak's
    tolerance ``rel * max(min(mz, mz_max), min_mz)`` are matched one to one by linear assignment
    maximizing the summed products (``spec_utils.batch_cos_hun_helper``, precursor peak kept).

    Args:
        target: ``(mzs, intens)`` from :func:`prepare_hun_peaks`.
        preds: Candidate predictions, each ``(mzs, intens)`` from :func:`prepare_hun_peaks`.
        rel: Relative tolerance (1e-5 is 10 ppm).
        min_mz: Floor of the m/z the relative tolerance is applied to.
        mz_max: Cap on the m/z the tolerance is computed from.
        chunk_size: Candidates per padded batch (bounds memory; results do not depend on it).
        max_elems: Cap on a batch's padded target x prediction peak pairs (bounds memory).
        device: Device for the matching and assignment.

    Returns:
        One similarity in ``[0, 1]`` per prediction, in input order.
    """
    tm, ti = target
    out: list[float] = []
    for c0, c1 in _chunk_bounds(preds, len(tm), chunk_size, max_elems):
        chunk = preds[c0:c1]
        b, n_p = len(chunk), max(1, max(len(m) for m, _ in chunk))
        if len(tm) == 0:
            out += [0.0] * b
            continue
        pmz = th.full((b, n_p), float("inf"), dtype=th.float64)
        pi = th.zeros((b, n_p), dtype=th.float64)
        for j, (m, p) in enumerate(chunk):
            pmz[j, : len(m)], pi[j, : len(m)] = m, p
        tmz = tm[None].expand(b, -1).to(device)
        tint = ti[None].expand(b, -1).to(device)
        pmz, pi = pmz.to(device), pi.to(device)
        tol = peak_tolerance(tmz.clamp(max=mz_max), rel, min_mz)
        mask = calculate_match_mzs(tmz, pmz, tolerance=0.0, relative=False, tol_per_true=tol)
        s = batch_cos_hun_helper(tint, pi, mask, mask.any(-1), mask.any(-2), False, None, None)
        out += s.clamp(0.0, 1.0).cpu().tolist()
    return out



def jss_hungarian_inst_tol(
    target: tuple[th.Tensor, th.Tensor],
    preds: list[tuple[th.Tensor, th.Tensor]],
    rel: float,
    min_mz: float,
    mz_max: float = MSPRED_UPPER_LIMIT,
    chunk_size: int = 32,
    max_elems: int = 1 << 26,
    device: str | th.device = "cpu",
    log_min: float = 1e-9,
) -> list[float]:
    """The training ``jss_hun_inst_tol`` metric: Jensen-Shannon similarity at instrument tolerance.

    Both spectra are L1-normalized over all their peaks; peak pairs within the target peak's
    tolerance ``rel * max(min(mz, mz_max), min_mz)`` are matched one to one by linear assignment
    maximizing the summed matched mass ``P_u + Q_v`` (``spec_utils.batch_jss_hun_helper``,
    precursor peak kept), and the JSS ``1 - JSD / ln 2`` is taken over the union of peaks.
    Unlike :func:`entropy_similarity_inst_tol`, the matching maximizes mass, not entropy gain,
    and no peak above the precursor is removed.

    Args:
        target: ``(mzs, intens)`` from :func:`prepare_hun_peaks`.
        preds: Candidate predictions, each ``(mzs, intens)`` from :func:`prepare_hun_peaks`.
        rel: Relative tolerance (1e-5 is 10 ppm).
        min_mz: Floor of the m/z the relative tolerance is applied to.
        mz_max: Cap on the m/z the tolerance is computed from.
        chunk_size: Candidates per padded batch (bounds memory; results do not depend on it).
        max_elems: Cap on a batch's padded target x prediction peak pairs (bounds memory).
        device: Device for the matching and assignment.
        log_min: ``safelog`` floor, the template's ``log_min``.

    Returns:
        One similarity in ``[0, 1]`` per prediction, in input order.
    """
    tm, ti = target
    out: list[float] = []
    for c0, c1 in _chunk_bounds(preds, len(tm), chunk_size, max_elems):
        chunk = preds[c0:c1]
        b, n_p = len(chunk), max(1, max(len(m) for m, _ in chunk))
        if len(tm) == 0:
            out += [0.0] * b
            continue
        pmz = th.full((b, n_p), float("inf"), dtype=th.float64)
        pi = th.zeros((b, n_p), dtype=th.float64)
        for j, (m, p) in enumerate(chunk):
            pmz[j, : len(m)], pi[j, : len(m)] = m, p
        tmz = tm[None].expand(b, -1).to(device)
        tint = ti[None].expand(b, -1).to(device)
        pmz, pi = pmz.to(device), pi.to(device)
        tol = peak_tolerance(tmz.clamp(max=mz_max), rel, min_mz)
        mask = calculate_match_mzs(tmz, pmz, tolerance=0.0, relative=False, tol_per_true=tol)
        s = batch_jss_hun_helper(
            tint, pi, mask, mask.any(-1), mask.any(-2), False, None, None, log_min=log_min
        )
        out += (s / np.log(2.0)).clamp(0.0, 1.0).cpu().tolist()
    return out

def query_log_likelihood(
    query_mzs: np.ndarray,
    query_probs: np.ndarray,
    pred_mzs_sorted: np.ndarray,
    pred_probs_sorted: np.ndarray,
    rel: float,
    min_mz: float,
    eps: float = 1e-5,
) -> float:
    """Log-likelihood of the query under the predicted distribution, at instrument tolerance.

    ``L = sum_k y_k log max(p_hat(m_k), eps)``, where ``p_hat(m_k)`` sums the predicted
    probability within ``m_k +- rel * max(m_k, min_mz)``.

    Args:
        query_mzs: Query peak m/z, shape ``(n,)``.
        query_probs: Query intensities normalized to sum to 1, shape ``(n,)``.
        pred_mzs_sorted: Predicted m/z sorted ascending, shape ``(m,)``.
        pred_probs_sorted: Predicted intensities (sum 1) in the same order, shape ``(m,)``.
        rel: Relative tolerance.
        min_mz: Tolerance floor m/z.
        eps: Probability floor for query peaks with no predicted intensity in the window.

    Returns:
        The log-likelihood (<= 0).
    """
    tol = peak_tolerance(query_mzs, rel, min_mz)
    lo = np.searchsorted(pred_mzs_sorted, query_mzs - tol, "left")
    hi = np.searchsorted(pred_mzs_sorted, query_mzs + tol, "right")
    csum = np.concatenate([[0.0], np.cumsum(pred_probs_sorted)])
    return float((query_probs * np.log(np.maximum(csum[hi] - csum[lo], eps))).sum())


# ---------------------------------------------------------------------------------------------
# Ranking metrics and intervals
# ---------------------------------------------------------------------------------------------


def expected_hits(
    scores: np.ndarray, is_truth: np.ndarray, ks: tuple[int, ...] = (1, 5, 20)
) -> list[float]:
    """Expected hit@k for one query under random tie-breaking; any truth candidate counts.

    With ``s*`` the best truth score, ``g`` non-truth candidates scoring above it, and ``t_n``
    non-truth / ``t_t`` truth candidates tied at ``s*``, the probability that a truth lands in the
    top ``k`` is ``1 - C(t_n, r) / C(t_n + t_t, r)`` with ``r = k - g`` free slots. NaN scores rank
    last. Never break ties by candidate order: MassSpecGym lists the truth first.

    Args:
        scores: Candidate scores (higher is better), shape ``(n,)``.
        is_truth: Boolean truth mask, shape ``(n,)``, at least one True.
        ks: Cutoffs.

    Returns:
        Expected hit@k for each ``k``.

    Raises:
        ValueError: If no candidate is a truth.
    """
    is_truth = np.asarray(is_truth, dtype=bool)
    if not is_truth.any():
        raise ValueError("query has no truth candidate")
    s = np.nan_to_num(np.asarray(scores, dtype=float), nan=-np.inf)
    best = s[is_truth].max()
    above = int((s[~is_truth] > best).sum())
    tie_n, tie_t = int((s[~is_truth] == best).sum()), int((s[is_truth] == best).sum())
    out = []
    for k in ks:
        free = k - above
        if free <= 0:
            out.append(0.0)
        elif free >= tie_n + 1:
            out.append(1.0)
        else:
            out.append(1 - comb(tie_n, free) / comb(tie_n + tie_t, free))
    return out


def molecule_bootstrap_ci(
    values: np.ndarray,
    groups: np.ndarray,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean and percentile interval of per-query values, resampling molecules (clusters).

    All queries of a resampled molecule enter together, so correlated spectra of one molecule do
    not shrink the interval. Pass a paired difference (model A - model B per query) to get a
    paired interval.

    Args:
        values: Per-query values, shape ``(n,)``.
        groups: Per-query molecule key (e.g. 2D InChIKey), shape ``(n,)``.
        n_resamples: Bootstrap resamples.
        level: Interval coverage.
        seed: RNG seed.

    Returns:
        ``(mean, low, high)``.
    """
    values = np.asarray(values, dtype=float)
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    n_g = inv.max() + 1
    sums = np.bincount(inv, weights=values, minlength=n_g)
    counts = np.bincount(inv, minlength=n_g)
    draw = np.random.default_rng(seed).integers(0, n_g, (n_resamples, n_g))
    means = sums[draw].sum(1) / counts[draw].sum(1)
    a = (1 - level) / 2
    return float(values.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def mspred_bootstrap_ci(
    hits: np.ndarray, level: float = 0.999, n_resamples: int = 20000, seed: int = 1
) -> tuple[float, float, float]:
    """ms-pred's interval (``retrieval/bootstrap_metrics.py``): i.i.d. BCa over queries.

    ms-pred rebuilds a 0/1 list from the rounded hit rate and runs ``scipy.stats.bootstrap``
    (BCa, 99.9%, 20,000 resamples, ``np.random.seed(1)``). Queries are treated as independent,
    so with several spectra per molecule the interval is narrower than a molecule bootstrap.

    Args:
        hits: Per-query hit indicators (0/1), shape ``(n,)``.
        level: Interval coverage.
        n_resamples: Bootstrap resamples.
        seed: Seed of the legacy numpy stream (ms-pred calls ``np.random.seed(1)``).

    Returns:
        ``(mean, low, high)``.
    """
    hits = np.asarray(hits, dtype=float)
    rate, n = hits.mean(), len(hits)
    sample = np.array(int(rate * n) * [1.0] + int((1 - rate) * n) * [0.0])
    res = bootstrap(
        (sample,),
        np.mean,
        confidence_level=level,
        n_resamples=n_resamples,
        batch=2000,
        random_state=np.random.RandomState(seed),  # the stream ms-pred's np.random.seed(1) gives
    )
    return float(rate), float(res.confidence_interval.low), float(res.confidence_interval.high)
