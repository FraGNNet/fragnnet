"""Plotting utilities for FraGNNet visualization.

This module provides functions for creating visualizations including:
- Converting matplotlib figures to PIL images
- Plotting histograms
- Image trimming and processing
- Molecular structure visualization
- Spectrum comparison plots
- Dual-axis plots for metric visualization
- Spectrum binning and matching
"""

import io
from typing import Any

import matplotlib as mpl
import matplotlib.collections as mc
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image, ImageOps
from rdkit.Chem import Mol
from rdkit.Chem.Draw import rdMolDraw2D

from fragnnet.utils.data_utils import mol_from_smiles
from fragnnet.utils.misc_utils import TOLERANCE_MIN_MZ
from fragnnet.utils.spec_utils import calculate_match_mzs

EPS = np.finfo(np.float32).eps


def fig_to_data(fig: Figure, **kwargs: Any) -> Image.Image:
    """Convert a matplotlib figure to a PIL Image.

    Args:
        fig: Matplotlib figure to convert.
        **kwargs: Additional arguments passed to fig.savefig().

    Returns:
        PIL Image object containing the rendered figure.
    """
    buf = io.BytesIO()
    fig.savefig(buf, **kwargs)
    buf.seek(0)
    image = Image.open(buf)
    return image


def plot_histogram(
    values: np.ndarray | list[float], title: str, log: bool = False, bins: int = 10
) -> Image.Image:
    """Create a histogram plot and return as PIL Image.

    Args:
        values: Array of values to plot.
        title: Title for the histogram.
        log: Whether to use logarithmic scale for y-axis. Defaults to False.
        bins: Number of histogram bins. Defaults to 10.

    Returns:
        PIL Image object containing the histogram plot.
    """
    fig, ax = plt.subplots()
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_yscale("log" if log else "linear")
    data = fig_to_data(fig)
    plt.close("all")
    return data


def trim_img_by_white(img: Image.Image, padding: int = 0) -> Image.Image:
    """Crop a PIL image to the minimum rectangle based on whiteness/transparency.

    This function removes white and transparent pixels from the edges of an image,
    with automatic 5-pixel padding applied. Adapted from:
    https://github.com/connorcoley/retrosim/blob/master/retrosim/utils/draw.py

    Args:
        img: PIL Image object to crop.
        padding: Border padding in pixels to add after cropping. Defaults to 0.

    Returns:
        Cropped PIL Image object with optional padding.
    """

    # Convert to array
    as_array = np.array(img)  # N x N x (r,g,b,a)
    assert as_array.ndim == 3 and as_array.shape[2] == 3, as_array.shape
    # Content defined as non-white and non-transparent pixel
    has_content = np.sum(as_array, axis=2, dtype=np.uint32) != 255 * 3
    xs, ys = np.nonzero(has_content)
    # Crop down
    x_range = max([min(xs) - 5, 0]), min([max(xs) + 5, as_array.shape[0]])
    y_range = max([min(ys) - 5, 0]), min([max(ys) + 5, as_array.shape[1]])
    as_array_cropped = as_array[x_range[0] : x_range[1], y_range[0] : y_range[1], 0:3]
    img = Image.fromarray(as_array_cropped, mode="RGB")
    return ImageOps.expand(img, border=padding, fill=(255, 255, 255, 0))


def get_mol_im(smiles: str, width: int = 1000, height: int = 1000) -> Image.Image:
    """Generate a PIL image of a molecule from its SMILES string.

    Attempts to use cairosvg for high-quality SVG rendering, with fallback to
    RDKit's Cairo drawer if cairosvg is not available.

    Args:
        smiles: SMILES string representation of the molecule.
        width: Width of the output image in pixels. Defaults to 1000.
        height: Height of the output image in pixels. Defaults to 1000.

    Returns:
        PIL Image object containing the rendered molecule structure.
    """

    mol = mol_from_smiles(smiles)

    try:
        import cairosvg

        d = rdMolDraw2D.MolDraw2DSVG(width, height)
        d.DrawMolecule(mol)
        d.FinishDrawing()
        svg_buf = d.GetDrawingText()
        png_buf = cairosvg.svg2png(svg_buf)
        im = Image.open(io.BytesIO(png_buf))
    except ImportError:
        # Fallback to PNG drawer if cairosvg is not available
        d = rdMolDraw2D.MolDraw2DCairo(width, height)
        d.DrawMolecule(mol)
        d.FinishDrawing()
        png_buf = d.GetDrawingText()
        im = Image.open(io.BytesIO(png_buf))

    im = trim_img_by_white(im, padding=15)
    return im


def plot_dual_axis(
    steps: np.ndarray | list[float],
    ax1_metrics_d: dict[str, np.ndarray | list[float]],
    ax2_metrics_d: dict[str, np.ndarray | list[float]],
    fp: str,
    figsize: tuple[float, float] = (10, 5),
    dpi: int = 200,
    size: int = 20,
) -> None:
    """Create a plot with dual y-axes for different metric groups.

    Args:
        steps: X-axis values (steps/iterations).
        ax1_metrics_d: Dictionary of metrics to plot on left y-axis.
        ax2_metrics_d: Dictionary of metrics to plot on right y-axis.
        fp: File path to save the figure.
        figsize: Figure size as (width, height) in inches. Defaults to (10, 5).
        dpi: Dots per inch for the figure. Defaults to 200.
        size: Base font size for labels and ticks. Defaults to 20.

    Returns:
        None. Saves figure to disk at the specified file path.
    """
    tick_size = int(0.6 * size)
    font_size = int(0.8 * size)

    # Set up the figure and axes
    fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
    ax2 = ax1.twinx()

    # Determine the number of lines for each axis
    num_lines_ax1 = len(ax1_metrics_d)
    num_lines_ax2 = len(ax2_metrics_d)

    # Generate color schemes for the lines
    ax1_colors = mpl.colormaps["tab10"].colors[:num_lines_ax1]
    ax2_colors = mpl.colormaps["tab10"].colors[num_lines_ax1 : num_lines_ax1 + num_lines_ax2]

    # Plot lines for the first y-axis
    ax1_lines = []
    ax1_labels = []
    for i, (label, values) in enumerate(ax1_metrics_d.items()):
        color = ax1_colors[i]
        (line,) = ax1.plot(steps, values, label=label, color=color)
        ax1_lines.append(line)
        ax1_labels.append(label)

    # Plot lines for the second y-axis
    ax2_lines = []
    ax2_labels = []
    for i, (label, values) in enumerate(ax2_metrics_d.items()):
        color = ax2_colors[i]
        (line,) = ax2.plot(steps, values, label=label, color=color)
        ax2_lines.append(line)
        ax2_labels.append(label)

    # Set labels for the axes and legends
    ax1.set_xlabel("Steps", fontsize=font_size)
    ax1.set_ylabel(" // ".join(ax1_labels), fontsize=font_size)
    ax2.set_ylabel(" // ".join(ax2_labels), fontsize=font_size)

    ax1.tick_params(axis="both", which="major", labelsize=tick_size)
    ax2.tick_params(axis="both", which="major", labelsize=tick_size)

    ax1.legend(
        ax1_lines + ax2_lines,
        [line.get_label() for line in ax1_lines + ax2_lines],
        loc="center right",  # this is usually the least interesting part
        fontsize=font_size,
    )

    fig.savefig(fp, format="png")
    plt.close("all")


def bin_spectrum(
    mzs: np.ndarray | list[float],
    ints: np.ndarray | list[float],
    mz_res: float,
    mz_max: float,
) -> np.ndarray:
    """Bin mass spectrum values into uniform m/z bins.

    Args:
        mzs: Array of m/z (mass-to-charge) values.
        ints: Array of intensity values corresponding to mzs.
        mz_res: m/z resolution (bin width).
        mz_max: Maximum m/z value for binning.

    Returns:
        Binned spectrum as numpy array of intensities for each bin.
    """
    bins = np.arange(0, mz_max + mz_res, mz_res)
    bin_spec = np.zeros_like(bins, dtype=np.float32)
    bin_idx = np.digitize(mzs, bins, right=True)
    for i in range(len(bin_idx)):
        bin_spec[bin_idx[i]] += ints[i]
    return bin_spec


def get_mz_bin_idx(mz: float | np.ndarray, mz_res: float, mz_max: float) -> int | np.ndarray:
    """Get bin indices for m/z values based on binning parameters.

    Args:
        mz: m/z value(s) to bin (scalar or array).
        mz_res: m/z resolution (bin width).
        mz_max: Maximum m/z value for binning.

    Returns:
        Bin index or indices corresponding to the input m/z value(s).
    """
    bins = np.arange(0, mz_max + mz_res, mz_res)
    bin_idx = np.digitize(mz, bins, right=True)
    return bin_idx


def plot_spectra_sparse(
    true_mzs: np.ndarray | list[float],
    true_ints: np.ndarray | list[float],
    pred_mzs: np.ndarray | list[float],
    pred_ints: np.ndarray | list[float],
    smiles: str | None,
    fp: str | None = None,
    mz_res: float = 1e-3,
    mz_max: float = 1000.0,
    figsize: tuple[float, float] = (20, 20),
    size: int = 24,
    log: bool = True,
    true_ints_thresh: float = 0.0,
    pred_ints_thresh: float = 0.0,
    plot_title: bool = True,
    custom_title: str = "",
    match_peaks: bool = True,
    match_tolerance: float = 1e-5,
    match_relative: bool = True,
    match_tolerance_min_mz: float = TOLERANCE_MIN_MZ,
    return_data: bool = False,
    colors: list[str] | None = None,
    bar_width: float = 3.0,
    dpi: int = 200,
) -> Image.Image | None:
    """Plot true vs predicted mass spectra with optional molecular structure.

    Creates a dual-spectrum comparison plot with optional molecular structure image.
    Supports peak matching, intensity thresholding, and various visualization options.

    Args:
        true_mzs: Array of true m/z values.
        true_ints: Array of true intensities.
        pred_mzs: Array of predicted m/z values.
        pred_ints: Array of predicted intensities.
        smiles: SMILES string of the molecule (None to skip molecule image).
        fp: File path to save the figure. If None, figure is not saved.
        mz_res: m/z resolution for visualization. Defaults to 1e-3.
        mz_max: Maximum m/z value to display. Defaults to 1000.0.
        figsize: Figure size as (width, height). Defaults to (20, 20).
        size: Base font size for labels. Defaults to 24.
        log: Whether to use logarithmic scale. Defaults to True.
        true_ints_thresh: Threshold for true intensity filtering. Defaults to 0.0.
        pred_ints_thresh: Threshold for predicted intensity filtering. Defaults to 0.0.
        plot_title: Whether to add a title to the plot. Defaults to True.
        custom_title: Custom title text. Defaults to empty string.
        match_peaks: Whether to match peaks between true and predicted. Defaults to True.
        match_tolerance: Tolerance for peak matching. Defaults to 1e-5.
        match_relative: Whether tolerance is relative to m/z. Defaults to True.
        match_tolerance_min_mz: Minimum m/z for relative tolerance. Defaults to TOLERANCE_MIN_MZ.
        return_data: Whether to return the figure as PIL Image. Defaults to False.
        colors: List of [true_color, pred_color]. Uses defaults if None.
        bar_width: Width of spectrum bars. Defaults to 3.0.
        dpi: Dots per inch for the figure. Defaults to 200.

    Returns:
        PIL Image if return_data=True, None otherwise.
    """
    true_mask = true_ints > true_ints_thresh
    true_mzs = true_mzs[true_mask]
    true_ints = true_ints[true_mask]
    pred_mask = pred_ints > pred_ints_thresh
    pred_mzs = pred_mzs[pred_mask]
    pred_ints = pred_ints[pred_mask]
    true_mz_max = max(np.max(true_mzs), np.max(pred_mzs))

    mz_max = int(np.floor(1.05 * min(mz_max, true_mz_max)))
    true_mask = true_mzs < mz_max
    pred_mask = pred_mzs < mz_max
    true_mzs = true_mzs[true_mask]
    true_ints = true_ints[true_mask]
    pred_mzs = pred_mzs[pred_mask]
    pred_ints = pred_ints[pred_mask]

    x_max = mz_max
    ints_max = np.max(true_ints)
    y_max = 1.05 * ints_max

    fig = plt.figure(figsize=figsize, dpi=dpi)

    if plot_title:
        title_y = 0.975
        title_font_size = 20
        fig.suptitle(custom_title, fontsize=title_font_size, y=title_y)
        height_ratios = [1, 3, 3]
    else:
        height_ratios = [1, 2, 2]

    # Adding extra subplot so both plots have common x-axis and y-axis labels
    y_pad = int(2.0 * size)
    x_pad = int(size)
    tick_size = int(0.8 * size)
    font_size = int(0.8 * size)

    # defualt colors
    if colors == None or len(colors) < 2:
        colors = ["#52B4FA", "#E63434"]

    # set up gridspec
    left = 0.12
    right = 0.98
    bottom = 0.10
    if smiles is not None:
        top = 0.95
        gs = mpl.gridspec.GridSpec(3, 1, height_ratios=height_ratios)
        gs.update(left=left, right=right, top=top, bottom=bottom, hspace=0.0)
        mol_im = get_mol_im(smiles)
        mol_im_arr = np.array(mol_im)
        mol_im_ax = fig.add_subplot(gs[0])
        mol_cim = mol_im_ax.imshow(mol_im_arr)
        mol_im_ax.axis("off")
        ylabel_y = bottom + (height_ratios[-1] / sum(height_ratios)) * (top - bottom)
    else:
        top = 0.98
        gs = mpl.gridspec.GridSpec(2, 1, height_ratios=height_ratios[1:])
        gs.update(left=left, right=right, top=top, bottom=bottom, hspace=0.0)
        mol_im_ax = None
        ylabel_y = bottom + 0.5 * (top - bottom)

    ax_top = fig.add_subplot(gs[-2], facecolor="white")
    ax_top.set_ylim(0, y_max)
    ax_top.set_xlim(0, x_max)
    # get peaks
    lines_top, styles_top = [], []
    if match_peaks:
        mz_match_matrix = calculate_match_mzs(
            true_mzs,
            pred_mzs,
            tolerance=match_tolerance,
            relative=match_relative,
            tolerance_min_mz=match_tolerance_min_mz,
        )
        mz_mask = np.any(mz_match_matrix, axis=1)
        # print(mz_mask.sum())
    else:
        mz_mask = np.ones_like(true_mzs, dtype=np.bool)
    for i in range(len(true_mzs)):
        mz = true_mzs[i]
        ints = true_ints[i]
        line_start = (mz, 0)
        line_end = (mz, ints)
        lines_top.append((line_start, line_end))
        if mz_mask[i]:
            styles_top.append("solid")
        else:
            styles_top.append((0, (1, 1)))
    # print(lines_top[:10])
    lc_top = mc.LineCollection(
        lines_top, colors=colors[0], linewidths=bar_width, linestyles=styles_top
    )
    ax_top.add_collection(lc_top)

    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.grid(color="black", linewidth=0.1)

    ax_bottom = fig.add_subplot(gs[-1], facecolor="white")
    # Invert the direction of y-axis ticks for bottom graph.
    ax_bottom.set_ylim(y_max, 0)
    ax_bottom.set_xlim(0, x_max)
    # get peaks
    lines_bottom = []
    for i in range(len(pred_mzs)):
        mz = pred_mzs[i]
        ints = pred_ints[i]
        line_start = (mz, 0)
        line_end = (mz, ints)
        lines_bottom.append((line_start, line_end))
    # print(lines_bottom[:10])
    lc_bottom = mc.LineCollection(lines_bottom, colors=colors[1], linewidths=bar_width)
    ax_bottom.add_collection(lc_bottom)

    # Remove overlapping 0's from middle of y-axis
    yticks_bottom = ax_bottom.yaxis.get_major_ticks()
    yticks_bottom[0].label1.set_visible(False)

    ax_bottom.grid(color="black", linewidth=0.1)

    for ax in [ax_top, ax_bottom]:
        ax.minorticks_on()
        ax.tick_params(axis="y", which="minor", left=False)
        ax.tick_params(axis="y", which="minor", right=False)
        ax.tick_params(axis="both", which="major", labelsize=tick_size)

    ax_top.tick_params(axis="x", which="minor", top=False)

    # x/y axis labels
    fig.supxlabel(
        "Mass/Charge (m/z)",
        fontsize=font_size,
        x=left + 0.5 * (right - left),
        y=0.2 * bottom,
    )
    fig.supylabel("Intensity", fontsize=font_size, y=ylabel_y, x=0.2 * left)

    leg_font_size = 0.9 * font_size
    leg_kws = {"ncol": 1, "fontsize": leg_font_size, "loc": "upper left"}
    handles = [
        patches.Patch(facecolor=colors[0], label="Ground Truth", alpha=1.0),
        patches.Patch(facecolor=colors[1], label="Predicted", alpha=1.0),
    ]
    leg = ax_top.legend(handles=handles, **leg_kws)

    if fp is not None:
        fig.savefig(fp)

    data = None
    if return_data:
        data = fig_to_data(fig)

    plt.close("all")

    return data


def get_highlighted_bonds(mol: Mol, node_atom_indices: list[int] | set) -> list[int]:
    """Extract bonds connecting atoms within a specified set of atom indices.

    Args:
        mol: RDKit Mol object.
        node_atom_indices: Set or list of atom indices to include.

    Returns:
        List of bond indices connecting atoms in the specified set.
    """
    atoms = []
    for a in mol.GetAtoms():
        atoms.append(a.GetIdx())
    bonds = []
    for bond in mol.GetBonds():
        aid1 = atoms[bond.GetBeginAtomIdx()]
        aid2 = atoms[bond.GetEndAtomIdx()]
        if aid1 in node_atom_indices and aid2 in node_atom_indices:
            bonds.append(mol.GetBondBetweenAtoms(aid1, aid2).GetIdx())
    return bonds
