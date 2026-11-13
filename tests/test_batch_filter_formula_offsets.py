"""Peak->formula mapping in `batch_filter_by_peak_intensity` must not depend on batch size.

The local formula index carried by `frag_formula_peak_idxs` is turned into a row of the
batch-global formula arrays by adding a per-spectrum offset. That offset has to count
formula ROWS per spectrum: the model trims one null formula per molecule, so peaks per
spectrum and formula rows per spectrum drift apart, and a peak-based offset makes every
spectrum after the first select another molecule's formulas. Spectrum 0 has offset 0
either way, so batch_size=1 cannot catch this.
"""

import torch as th

from fragnnet.utils.script_utils import PredArrays, batch_filter_by_peak_intensity

# Two spectra, each with 3 peaks and 4 formula rows, so the peak offset (3) and the
# formula-row offset (4) disagree for spectrum 1.
N_SPEC = 2
PEAKS_PER_SPEC = 3
FORMULAS_PER_SPEC = 4


def _build(min_peak_int, keep_all=False):
    pred_batch_idxs = th.repeat_interleave(th.arange(N_SPEC), PEAKS_PER_SPEC)
    pred_formula_batch_idxs = th.repeat_interleave(th.arange(N_SPEC), FORMULAS_PER_SPEC)
    # Global formula ids, contiguous per spectrum: spec0 -> 0..3, spec1 -> 4..7
    pred_formula_formula_idxs = th.arange(N_SPEC * FORMULAS_PER_SPEC)
    frag_formula_cumsizes = th.tensor([0, FORMULAS_PER_SPEC, 2 * FORMULAS_PER_SPEC])
    # Each peak points at a local formula index within its own molecule.
    frag_formula_peak_idxs = th.tensor([0, 1, 2, 0, 1, 2])
    # Only the last peak of each spectrum is below threshold when filtering.
    pred_ints = th.tensor([10.0, 10.0, 0.0, 10.0, 10.0, 0.0])
    if keep_all:
        pred_ints = th.full((N_SPEC * PEAKS_PER_SPEC,), 10.0)
    pred_mzs = th.arange(N_SPEC * PEAKS_PER_SPEC, dtype=th.float32)

    arrays = PredArrays(
        pred_batch_idxs=pred_batch_idxs,
        pred_ints=pred_ints,
        pred_mzs=pred_mzs,
        frag_formula_peak_idxs=frag_formula_peak_idxs,
        pred_formula_batch_idxs=pred_formula_batch_idxs,
        pred_formula_formula_idxs=pred_formula_formula_idxs,
    )
    return {
        "arrays": arrays,
        "min_peak_int": min_peak_int,
        "frag_formula_cumsizes": frag_formula_cumsizes,
        "frag_num_nodes": None,
    }


def _surviving_formulas_per_spec(min_peak_int):
    out = batch_filter_by_peak_intensity(**_build(min_peak_int))
    formula_batch_idxs = out.pred_formula_batch_idxs
    formula_formula_idxs = out.pred_formula_formula_idxs
    return {
        b: sorted(formula_formula_idxs[formula_batch_idxs == b].tolist()) for b in range(N_SPEC)
    }


def test_surviving_formulas_stay_within_their_own_spectrum():
    """Each spectrum must keep formulas from its own block, never a neighbour's."""
    per_spec = _surviving_formulas_per_spec(min_peak_int=1.0)
    # Indices are rebased to per-molecule at the end, so both spectra land in 0..3.
    for b, kept in per_spec.items():
        assert kept, f"spectrum {b} kept no formulas"
        assert all(0 <= f < FORMULAS_PER_SPEC for f in kept), (
            f"spectrum {b} kept out-of-block formulas {kept}"
        )


def test_both_spectra_keep_the_same_formulas_for_identical_input():
    """The two spectra are constructed identically, so they must survive identically.

    With a peak-based offset spectrum 1 reads one row short and keeps a different set.
    """
    per_spec = _surviving_formulas_per_spec(min_peak_int=1.0)
    assert per_spec[0] == per_spec[1], f"batch position changed the result: {per_spec}"


def test_peak_to_formula_offset_is_row_based_not_peak_based():
    """Pin the offset itself: spectrum 1's formula rows start at 4, not at 3."""
    peaks_per_spec = th.bincount(th.repeat_interleave(th.arange(N_SPEC), PEAKS_PER_SPEC))
    rows_per_spec = th.bincount(th.repeat_interleave(th.arange(N_SPEC), FORMULAS_PER_SPEC))
    assert int(peaks_per_spec[0]) != int(rows_per_spec[0]), (
        "fixture no longer distinguishes the two offset spaces"
    )
    per_spec = _surviving_formulas_per_spec(min_peak_int=1.0)
    # Peak 0 and 1 of each spectrum survive -> local formulas 0 and 1.
    assert per_spec[1] == [0, 1], f"spectrum 1 selected the wrong rows: {per_spec[1]}"
