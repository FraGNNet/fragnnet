import numpy as np
import pandas as pd
import torch as th
from rdkit import Chem, RDLogger
from scipy.linalg import eigh
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix

from fragnnet.utils.data_utils import formula_to_mass

RDLogger.DisableLog("rdApp.*")


def fit_ann_library(
    spec_df,
    ann_df,
    library_size,
    max_isotope,
    debug=False,
    huge_mz_diff_threshold=0.1,
    big_mz_diff_ppm_threshold=1e-5,
):
    """Fit an annotation library to spectra, filtering bad peak-formula matches.

    Matches annotated peaks to observed spectrum peaks and filters out
    annotations where the observed m/z deviates too far from the theoretical
    exact m/z. Two independent filters are applied in sequence:

    1. **Huge absolute error** (``huge_mz_diff_threshold``): if any annotation
       for a spectrum has ``|peak_mz - exact_mz| > huge_mz_diff_threshold``,
       the entire spectrum is dropped. This catches grossly wrong formula
       assignments.

    2. **Big relative error** (``big_mz_diff_ppm_threshold``): annotations
       where ``|peak_mz - exact_mz| / max(peak_mz, 200.0) > big_mz_diff_ppm_threshold``
       are pruned individually; the rest of the spectrum is retained.
       The denominator is floored at 200 Da so low-m/z peaks are not
       unfairly penalised. A value of ``1e-5`` is approximately 10 ppm
       relative to the 200 Da floor.

    Args:
        spec_df: DataFrame with columns ``spec_id``, ``mzs``, ``ints``, etc.
        ann_df: DataFrame with columns ``spec_id``, ``ann_products``,
            ``ann_losses``, ``ann_isotopes``, ``ann_exact_mzs``, ``peak_mzs``.
        library_size: Maximum number of library entries to retain.
        max_isotope: Maximum allowed isotope offset; annotations with
            ``isotopes > max_isotope`` or ``isotopes < 0`` are dropped.
        debug: If True, print m/z diff and isotope distributions.
        huge_mz_diff_threshold: Absolute m/z error threshold in Da. Spectra
            with any annotation exceeding this are dropped entirely.
            Pass ``None`` to disable. Defaults to 0.1.
        big_mz_diff_ppm_threshold: Relative m/z error threshold (fraction).
            Individual annotations exceeding this are pruned. Pass ``None``
            to disable. Defaults to 1e-5.

    Returns:
        Tuple of ``(ann_df, spec_df)`` after filtering, with annotation
        counts normalised and library entries clipped to ``library_size``.
    """
    orig_spec_ids = spec_df[["dset", "dset_spec_id", "spec_id"]].copy()
    spec_df = spec_df[spec_df["spec_id"].isin(ann_df["spec_id"])]
    spec_df = spec_df[["spec_id", "peaks"]]
    # spec_df.loc[:,"peaks"] = spec_df["peaks"].apply(l1_normalize_peaks)
    spec_df = spec_df.explode("peaks")
    spec_df[["mzs", "ints"]] = pd.DataFrame(spec_df["peaks"].tolist(), index=spec_df.index)
    spec_df = spec_df.drop(columns=["peaks"])
    total_ints = (
        spec_df[["spec_id", "ints"]]
        .groupby("spec_id")["ints"]
        .sum()
        .reset_index()
        .rename(columns={"ints": "total_ints"})
    )
    spec_df = spec_df.merge(total_ints, on="spec_id", how="inner")
    spec_df["ints"] = spec_df["ints"] / spec_df["total_ints"]
    spec_df = spec_df.drop(columns=["total_ints"])

    # select which (unmerged) spectra to use
    # map mzs to (unmerged) intensities
    ann_df = ann_df[ann_df["spec_id"].isin(spec_df["spec_id"])]
    ann_df = ann_df[
        ["spec_id", "ann_peak_mzs", "ann_products", "ann_losses", "ann_isotopes", "ann_exact_mzs"]
    ]
    ann_df = ann_df.rename(
        columns={
            "ann_peak_mzs": "peak_mzs",
            "ann_products": "products",
            "ann_losses": "losses",
            "ann_isotopes": "isotopes",
            "ann_exact_mzs": "exact_mzs",
        }
    )
    ann_df["joint"] = ann_df.apply(
        lambda row: list(
            zip(row["peak_mzs"], row["products"], row["losses"], row["isotopes"], row["exact_mzs"])
        ),
        axis=1,
    )
    ann_df = ann_df.drop(columns=["peak_mzs", "products", "losses", "isotopes", "exact_mzs"])
    ann_df = ann_df.explode("joint")
    ann_df[["peak_mzs", "products", "losses", "isotopes", "exact_mzs"]] = pd.DataFrame(
        ann_df["joint"].tolist(), index=ann_df.index
    )
    ann_df = ann_df.drop(columns=["joint"])
    ann_df["peak_mzs"] = ann_df["peak_mzs"].astype("float64")
    ann_df["isotopes"] = ann_df["isotopes"].astype("int64")
    ann_df["exact_mzs"] = ann_df["exact_mzs"].astype("float64")
    ann_df = ann_df.merge(
        spec_df[["spec_id", "mzs", "ints"]].rename(columns={"mzs": "peak_mzs"}),
        on=["spec_id", "peak_mzs"],
        how="inner",
    )

    # verify mz diffs
    mz_diffs = ann_df[["spec_id", "peak_mzs", "exact_mzs"]].copy()
    mz_diffs["mz_diffs"] = (mz_diffs["peak_mzs"] - mz_diffs["exact_mzs"]).abs()
    mz_diffs["mz_diffs_ppm"] = mz_diffs["mz_diffs"].to_numpy() / np.maximum(
        mz_diffs["peak_mzs"].to_numpy(), 200.0
    )
    if debug:
        print("> distribution of mzs diff (abs):")
        print(mz_diffs["mz_diffs"].describe())
        print("> distribution of mzs diff (ppm):")
        print(mz_diffs["mz_diffs_ppm"].describe())
    if huge_mz_diff_threshold is None:
        huge_mz_diff_threshold = float("inf")
    if big_mz_diff_ppm_threshold is None:
        big_mz_diff_ppm_threshold = float("inf")
    huge_mz_diffs = mz_diffs[mz_diffs["mz_diffs"] > huge_mz_diff_threshold]
    big_mz_diffs = mz_diffs[mz_diffs["mz_diffs_ppm"] > big_mz_diff_ppm_threshold]
    huge_mz_diff_ids = huge_mz_diffs["spec_id"].unique()
    big_mz_diff_ids = big_mz_diffs["spec_id"].unique()
    if huge_mz_diff_ids.shape[0] > 0:
        print(
            f"> warning: dropping spectra with huge mz diffs: {huge_mz_diffs.shape[0]} annotations, {huge_mz_diff_ids.shape[0]} spectra"
        )
        ann_df = ann_df[~(ann_df["spec_id"].isin(huge_mz_diff_ids))]
    if big_mz_diff_ids.shape[0] > 0:
        print(
            f"> warning: dropping annotations with big mz diffs: {big_mz_diffs.shape[0]} annotations, {big_mz_diff_ids.shape[0]} spectra"
        )
        ann_df = ann_df[~(ann_df.index.isin(big_mz_diffs.index))]

    # verify isotopes
    if debug:
        print("> distribution of isotopes:")
        print(ann_df["isotopes"].value_counts())
    invalid_isotopes = (ann_df["isotopes"] > max_isotope) | (ann_df["isotopes"] < 0)
    if invalid_isotopes.any():
        print(f"> warning: dropping invalid isotopes: {invalid_isotopes.sum()} annotations")
        ann_df = ann_df[~invalid_isotopes]

    # evenly divide intensity for each annotation that maps to the same bin
    ann_counts = (
        ann_df[["spec_id", "peak_mzs", "products"]]
        .groupby(["spec_id", "peak_mzs"])["products"]
        .count()
        .reset_index()
        .rename(columns={"products": "counts"})
    )
    ann_df = ann_df.merge(ann_counts, on=["spec_id", "peak_mzs"], how="inner")
    ann_df["ints"] = ann_df["ints"] / ann_df["counts"]
    ann_df = ann_df.drop(columns=["counts"])

    # check fractions
    ann_frac = ann_df[["spec_id", "ints"]].groupby(by=["spec_id"])["ints"].sum().reset_index()
    ann_frac = ann_frac.merge(
        pd.DataFrame({"spec_id": orig_spec_ids["spec_id"]}), on="spec_id", how="right"
    ).fillna(0.0)
    iso_frac = (
        ann_df[ann_df["isotopes"] > 0][["spec_id", "ints"]]
        .groupby(by=["spec_id"])["ints"]
        .sum()
        .reset_index()
    )
    iso_frac = iso_frac.merge(
        pd.DataFrame({"spec_id": orig_spec_ids["spec_id"]}), on="spec_id", how="right"
    ).fillna(0.0)
    if debug:
        print("> distribution of annotated intensity fraction:")
        print(ann_frac["ints"].describe())
        print("> distribution of isotope intensity fraction:")
        print(iso_frac["ints"].describe())

    # aggregate intensity by annotation
    product_df = ann_df[["products", "ints"]].groupby("products")["ints"].sum().reset_index()
    product_df["kind"] = "product"
    product_df = product_df.rename(columns={"products": "formula"})
    loss_df = ann_df[["losses", "ints"]].groupby("losses")["ints"].sum().reset_index()
    loss_df["kind"] = "loss"
    loss_df = loss_df.rename(columns={"losses": "formula"})
    library_df = pd.concat([product_df, loss_df], axis=0, ignore_index=True)

    # take top-k
    library_df = library_df.sort_values(by="ints", ascending=False)
    assert library_df.shape[0] >= library_size, (
        f"library size {library_df.shape[0]} < {library_size}"
    )
    library_df = library_df.head(library_size).reset_index()

    # enumerate
    library_df["ann_id"] = np.arange(library_df.shape[0])

    # calculate mzs
    library_df["mzs"] = library_df["formula"].apply(formula_to_mass)

    if debug:
        return library_df, big_mz_diff_ids
    else:
        return library_df


### chemical featurization stuff

x_map = {
    "atomic_num": list(range(0, 119)),
    "chirality": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
    ],
    "degree": list(range(0, 11)),
    "formal_charge": list(range(-5, 7)),
    "num_hs": list(range(0, 9)),
    "num_radical_electrons": list(range(0, 5)),
    "hybridization": [
        "UNSPECIFIED",
        "S",
        "SP",
        "SP2",
        "SP3",
        "SP3D",
        "SP3D2",
        "OTHER",
    ],
    "is_aromatic": [False, True],
    "is_in_ring": [False, True],
}

x_map = {k: {v: i for i, v in enumerate(vs)} for k, vs in x_map.items()}

e_map = {
    "bond_type": [
        "misc",
        "SINGLE",
        "DOUBLE",
        "TRIPLE",
        "AROMATIC",
    ],
    "stereo": [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
    ],
    "is_conjugated": [False, True],
}

e_map = {k: {v: i for i, v in enumerate(vs)} for k, vs in e_map.items()}

# def from_smiles(smiles, with_hydrogen=False, kekulize=False):
#     mol = Chem.MolFromSmiles(smiles)
#     return from_mol(mol, with_hydrogen, kekulize)


def from_mol(mol, with_hydrogen=False, kekulize=False):
    smiles = Chem.MolToSmiles(mol)
    if mol is None:
        mol = Chem.MolFromSmiles("")
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        mol = Chem.Kekulize(mol)

    xs = []
    for atom in mol.GetAtoms():
        x = [
            x_map["atomic_num"][atom.GetAtomicNum()],
            x_map["chirality"][str(atom.GetChiralTag())],
            x_map["degree"][atom.GetTotalDegree()],
            x_map["formal_charge"][atom.GetFormalCharge()],
            x_map["num_hs"][atom.GetTotalNumHs()],
            x_map["num_radical_electrons"][atom.GetNumRadicalElectrons()],
            x_map["hybridization"][str(atom.GetHybridization())],
            x_map["is_aromatic"][atom.GetIsAromatic()],
            x_map["is_in_ring"][atom.IsInRing()],
        ]
        xs.append(x)

    x = th.tensor(np.array(xs), dtype=th.long).view(-1, len(x_map))

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        idx = bond.GetIdx()

        e = [
            e_map["bond_type"][str(bond.GetBondType())],
            e_map["stereo"][str(bond.GetStereo())],
            e_map["is_conjugated"][bond.GetIsConjugated()],
        ]

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = th.tensor(np.array(edge_indices))
    edge_index = edge_index.t().to(th.long).view(2, -1)
    edge_attr = th.tensor(np.array(edge_attrs), dtype=th.long).view(-1, len(e_map))

    if edge_index.numel() > 0:
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)


# def mol_to_graph(mol):
#     graph = from_mol(mol)
#     graph.x = graph.x.byte()
#     graph.edge_attr = graph.edge_attr.byte()
#     return graph

# def smiles_to_graph(smiles):
#     return mol_to_graph(Chem.MolFromSmiles(smiles))


def add_virtual_node(g, inplace=False, add_flags=True, mask_value=0):
    edge_index = []
    edge_index.extend([(i, g.num_nodes) for i in range(g.num_nodes)])
    edge_index.extend([(g.num_nodes, i) for i in range(g.num_nodes)])
    edge_index = th.LongTensor(edge_index).view(-1, 2).T
    edge_attr = (
        th.zeros(
            edge_index.shape[1],
            g.edge_attr.shape[1],
            dtype=g.edge_attr.dtype,
            device=g.edge_attr.device,
        )
        + mask_value
    )
    num_nodes = g.num_nodes
    num_edges = g.num_edges

    x = th.zeros(1, g.x.shape[1], dtype=g.x.dtype, device=g.x.device) + mask_value

    if not inplace:
        g = g.clone()
    g.x = th.cat([g.x, x], 0)
    g.edge_index = th.cat([g.edge_index, edge_index], 1)
    g.edge_attr = th.cat([g.edge_attr, edge_attr], 0)
    if add_flags:
        n_in = n_out = edge_index.shape[1] // 2
        is_virt_node = th.BoolTensor([0] * num_nodes + [1]).unsqueeze(-1)
        is_virt_in_edge = th.BoolTensor([0] * num_edges + [1] * n_in + [0] * n_out).unsqueeze(-1)
        is_virt_out_edge = th.BoolTensor([0] * num_edges + [0] * n_in + [1] * n_out).unsqueeze(-1)
        g.x = th.cat([g.x, is_virt_node], 1)
        g.edge_attr = th.cat([g.edge_attr, is_virt_in_edge, is_virt_out_edge], 1)

    return g


def graph_laplacian(graph, k, padding_value=0):
    num_nodes = graph.num_nodes

    V = th.empty(num_nodes, k)
    D = th.empty(k)

    if num_nodes > 0:
        edge_index, edge_weight = get_laplacian(
            graph.edge_index,
            normalization="sym",
            num_nodes=num_nodes,
        )

        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

        eig_vals, eig_vecs = eigh(L.toarray())

        idx = eig_vals.argsort()
        eig_vecs = np.real(eig_vecs[:, idx])
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, 1 : k + 1]
        eig_vals = eig_vals[1 : k + 1]

        eig_vecs = eig_vecs / np.linalg.norm(eig_vecs, axis=0)

        V[:] = padding_value
        V[:, : eig_vecs.shape[1]] = th.from_numpy(eig_vecs)

        D[:] = padding_value
        D[: eig_vals.shape[0]] = th.from_numpy(eig_vals)

    graph.eigvecs = V
    graph.eigvals = D.unsqueeze(0)

    return graph


def graff_preprocess(mol, num_eigs):
    g = from_mol(mol)
    g = graph_laplacian(g, num_eigs)
    g = add_virtual_node(g)
    # must pad the eigenfeatures for the virtual node
    eig_pad = th.zeros(
        g.num_nodes - g.eigvecs.shape[0],
        g.eigvecs.shape[1],
        dtype=g.eigvecs.dtype,
        device=g.eigvecs.device,
    )
    g.eigvecs = th.cat([g.eigvecs, eig_pad], 0)

    return g
