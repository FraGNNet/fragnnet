import os

import pandas as pd
import torch as th
from pyteomics.mass import Composition
from tqdm import tqdm

from fragnnet.dataset import BaseDataset, SpecMolDataset
from fragnnet.graff.data_utils import fit_ann_library, graff_preprocess
from fragnnet.utils.data_utils import composition_to_string
from fragnnet.utils.formula_utils import NEUTRON_MASS
from fragnnet.utils.misc_utils import get_tensor_dict_memory_usage, scatter_reduce
from fragnnet.utils.spec_utils import batch_func


def calculate_prec_formula(prec_formula, prec_type):
    prec_formula_c = Composition(formula=prec_formula)
    if prec_type == "[M+H]+":
        prec_formula_c = prec_formula_c + Composition(formula="H")
    elif prec_type == "[M-H]-":
        prec_formula_c = prec_formula_c - Composition(formula="H")
    else:
        raise ValueError(f"Unknown precursor type: {prec_type}")
    return prec_formula_c


class SpecMolAnnDataset(SpecMolDataset):
    def __init__(
        self,
        spec_fp: str,
        mol_fp: str,
        split_dp: str,
        split: str,
        subsample_params: dict,
        spec_params: dict,
        mol_params: dict,
        ann_fp: str,
        ann_params: dict,
        spec_pp_sd: dict = dict(),
        mol_pp_sd: dict = dict(),
        ann_pp_sd: dict = dict(),
        **kwargs,
    ):
        assert not ann_params["filter_peaks"]
        BaseDataset.__init__(self)
        self._base_init(
            spec_fp=spec_fp,
            mol_fp=mol_fp,
            split_dp=split_dp,
            split=split,
            subsample_params=subsample_params,
            spec_params=spec_params,
        )
        self.mol_params = mol_params
        self.ann_fp = ann_fp
        self.ann_params = ann_params
        self._setup_ann(spec_fp=spec_fp, mol_fp=mol_fp, split_dp=split_dp)
        self._preprocess_spec(spec_pp_sd)
        self._preprocess_mol(mol_pp_sd)
        self._preprocess_ann(ann_pp_sd)

    @staticmethod
    def get_data_dict_types():
        # return list of data dict used for caching
        # Spectral Preprocessed Shared Dictionary: spec_pp_sd
        # Molecular Preprocessed Shared Dictionary: mol_pp_sd
        # Annotation Preprocessed Shared Dictionary: ann_pp_sd
        return ["spec_pp_sd", "mol_pp_sd", "ann_pp_sd"]

    def _setup_ann(self, spec_fp, mol_fp, split_dp):
        um_spec_df = BaseDataset._setup_dfs(
            spec_fp_or_df=spec_fp,
            mol_fp_or_df=mol_fp,
            split_dp=split_dp,
            splits=self.ann_params["library_splits"],
            subsample_params=self.subsample_params,
            spec_params=self.spec_params,
        )[2]
        ann_df = pd.read_pickle(self.ann_fp)
        # only fit it if it's in training mode
        assert self.ann_params["max_isotope"] >= 0, self.ann_params["max_isotope"]
        self.num_isotopes = self.ann_params["max_isotope"] + 1
        if self.ann_params["library_fp"] is None:
            library_df = fit_ann_library(
                um_spec_df.copy(),
                ann_df.copy(),
                self.ann_params["library_size"],
                self.ann_params["max_isotope"],
                huge_mz_diff_threshold=self.ann_params["huge_mz_diff_threshold"],
                big_mz_diff_ppm_threshold=self.ann_params["big_mz_diff_ppm_threshold"],
            )
        else:
            assert os.path.isfile(self.ann_params["library_fp"])
            library_df = pd.read_pickle(self.ann_params["library_fp"])
        self.library_df = library_df
        # setup mzs stuff
        self.library_mzs = th.as_tensor(library_df["mzs"].values, dtype=th.float32)
        self.library_mzs = self.library_mzs.unsqueeze(1).repeat(1, self.num_isotopes)
        self.library_loss_mask = th.as_tensor(
            self.library_df.loc[:, "kind"].apply(lambda x: int(x == "loss")).values,
            dtype=th.float32,
        )
        self.library_loss_mask = self.library_loss_mask.unsqueeze(1).repeat(1, self.num_isotopes)
        for i in range(self.num_isotopes):
            self.library_mzs[:, i] = self.library_mzs[:, i] + (
                -2 * self.library_loss_mask[:, i] + 1
            ) * (i * NEUTRON_MASS)
        self.library_loss_mask = self.library_loss_mask.flatten()
        self.library_mzs = self.library_mzs.flatten()
        # setup isotope stuff
        self.isotope_df = ann_df[["spec_id", "ann_isotopes"]]
        self.isotope_df.loc[:, "has_isotopes"] = ann_df["ann_isotopes"].apply(any)
        self.isotope_df = self.isotope_df.drop(columns=["ann_isotopes"])
        self.isotope_df = self.isotope_df[
            self.isotope_df["spec_id"].isin(self.um_spec_df["spec_id"])
        ]
        self.isotope_df = self.isotope_df.merge(
            self.um_spec_df[["spec_id", "group_id"]], on="spec_id", how="right"
        ).fillna(False)
        if self.spec_params["merge"]:
            self.isotope_df = (
                self.isotope_df.groupby("group_id")["has_isotopes"].apply(any).reset_index()
            )
        self.isotope_df = self.isotope_df.set_index(self.id_key, drop=False)

    def _preprocess_ann(self, ann_pp_sd: dict):
        if self.ann_params["preprocess"]:
            self.ann_datas = ann_pp_sd
            total_ann_data_size = 0
            for idx, spec_entry in tqdm(
                self.spec_df.iterrows(), desc="> preprocess ann", total=len(self.spec_df)
            ):
                mol_entry = self.mol_df.loc[spec_entry["mol_id"]]
                ann_data = self._process_ann(spec_entry, mol_entry)
                total_ann_data_size += get_tensor_dict_memory_usage(**ann_data)
                self.ann_datas[idx] = ann_data
            print(f"> total_ann_data_size = {total_ann_data_size / 1e6:.2f} MB")

    def _process_ann(self, spec_entry, mol_entry):
        # set up mzs
        prec_mz = spec_entry["prec_mz"]
        mzs, ints = SpecMolAnnDataset._get_mzs_ints(spec_entry["peaks"])
        library_mzs = prec_mz * self.library_loss_mask + self.library_mzs * (
            -2 * self.library_loss_mask + 1
        )
        # check for isotopes
        has_isotopes = self.isotope_df.loc[spec_entry[self.id_key]]["has_isotopes"]
        isotope_mask = th.tensor([has_isotopes], dtype=th.float32)
        # look for duplicates
        formula = mol_entry["formula"]
        prec_type = spec_entry["prec_type"]
        prec_formula_c = calculate_prec_formula(formula, prec_type)
        product_formulae_df = self.library_df[self.library_df["kind"] == "product"][
            ["ann_id", "formula"]
        ]
        loss_formulae_df = self.library_df[self.library_df["kind"] == "loss"][["ann_id", "formula"]]
        loss_formulae_df.loc[:, "formula"] = loss_formulae_df["formula"].apply(
            lambda formula: composition_to_string(prec_formula_c - Composition(formula=formula))
        )
        # invalid_formulae = loss_formulae_df["formula"].str.contains("-")
        # loss_formulae_df = loss_formulae_df[~invalid_formulae]
        both_formulae_df = pd.concat([product_formulae_df, loss_formulae_df], axis=0).sort_values(
            "ann_id", ascending=True
        )
        dup_formulae_df = product_formulae_df.merge(loss_formulae_df, on="formula", how="inner")
        assert len(set(dup_formulae_df["ann_id_x"]) & set(dup_formulae_df["ann_id_y"])) == 0
        # dedup duplicate formulae
        # dup_mask = th.zeros([both_formulae_df.shape[0]],dtype=th.bool)
        # dup_mask[dup_formulae_df["ann_id_x"]] = True
        # dup_mask[dup_formulae_df["ann_id_y"]] = True
        # dup_idxs = th.arange(both_formulae_df.shape[0])[dup_mask]
        # dup_idxs = th.repeat_interleave(dup_idxs,self.num_isotopes) + th.arange(self.num_isotopes).repeat(dup_idxs.shape[0])
        # import pdb; pdb.set_trace()
        library_df_dedup = both_formulae_df.merge(
            dup_formulae_df[["ann_id_x", "ann_id_y"]].rename(columns={"ann_id_y": "ann_id"}),
            on="ann_id",
            how="left",
        )
        library_df_dedup["ann_id"] = (
            library_df_dedup["ann_id_x"].fillna(library_df_dedup["ann_id"]).astype(int)
        )
        library_df_dedup = library_df_dedup.drop(columns=["ann_id_x"])
        ann_ids = th.as_tensor(library_df_dedup["ann_id"].to_numpy(), dtype=th.long)
        ann_ids_un, ann_idxs_inv = th.unique(ann_ids, return_inverse=True)
        ann_idxs_un = th.arange(ann_ids_un.shape[0])
        isotope_idxs = th.arange(self.num_isotopes).repeat(ann_idxs_inv.shape[0])
        ann_iso_idxs = (
            th.repeat_interleave(ann_idxs_inv * self.num_isotopes, self.num_isotopes, dim=0)
            + isotope_idxs
        )
        # isotope_idxs_un_ = th.arange(self.num_isotopes).repeat(ann_idxs_un.shape[0])
        # ann_iso_idxs_un_ = th.repeat_interleave(ann_idxs_un*self.num_isotopes,self.num_isotopes,dim=0) + isotope_idxs_un_
        # library_mzs_un_ = library_mzs[ann_iso_idxs_un_]
        # TODO: put this in the model?
        isotope_idxs_un = scatter_reduce(
            isotope_idxs,
            ann_iso_idxs,
            reduce="amax",
            dim_size=ann_idxs_un.shape[0] * self.num_isotopes,
            include_self=False,
        )
        library_mzs_un = scatter_reduce(
            library_mzs,
            ann_iso_idxs,
            reduce="amax",
            dim_size=ann_idxs_un.shape[0] * self.num_isotopes,
            include_self=False,
        )
        # these are already deduped
        if self.ann_params["filter_peaks"]:
            raise NotImplementedError
        ann_data = {
            "ann_library_mzs": library_mzs_un,
            "ann_isotope_mask": isotope_mask,
            # "ann_dup_idxs": dup_idxs,
            "ann_formula_idxs": ann_iso_idxs,
            "ann_isotope_idxs": isotope_idxs_un,
        }
        if self.ann_params["formula_str"]:
            ann_iso_idxs_un = scatter_reduce(
                ann_iso_idxs,
                ann_iso_idxs,
                reduce="amax",
                dim_size=ann_idxs_un.shape[0] * self.num_isotopes,
                include_self=False,
            )
            library_formula_df_un = library_df_dedup.drop_duplicates(subset="ann_id").sort_values(
                "ann_id", ascending=True
            )
            library_formula_un = (
                library_formula_df_un["formula"].iloc[ann_iso_idxs_un.numpy() // self.num_isotopes]
            ).to_numpy()
            ann_data["ann_library_formulae"] = library_formula_un
        return ann_data

    def _process_mol(self, mol_entry):
        mol_data = super()._process_mol(mol_entry)
        if self.mol_params["graff"]:
            mol_graff = graff_preprocess(mol_entry["mol"], self.ann_params["num_eigs"])
            mol_data["mol_graff"] = mol_graff
        return mol_data

    def __getitem__(self, idx):
        spec_entry = self.spec_df.iloc[idx]
        mol_id = spec_entry["mol_id"]
        mol_entry = self.mol_df.loc[mol_id]
        if self.spec_params["preprocess"]:
            spec_data = self.spec_datas[idx].copy()
        else:
            spec_data = self._process_spec(spec_entry)
        spec_data = self._process_spec(spec_entry)
        if self.mol_params["preprocess"]:
            mol_data = self.mol_datas[mol_id].copy()
        else:
            mol_data = self._process_mol(mol_entry)
        if self.ann_params["preprocess"]:
            ann_data = self.ann_datas[idx].copy()
        else:
            ann_data = self._process_ann(spec_entry, mol_entry)
        data = {**spec_data, **mol_data, **ann_data}
        return data

    @staticmethod
    def get_collate_fn():
        return SpecMolAnnDataset.collate_fn

    @staticmethod
    def _special_collate(keys, collate_data):
        assert "ann_library_mzs" in keys
        assert "ann_formula_idxs" in keys
        assert "ann_isotope_idxs" in keys
        # assert "ann_dup_idxs" in keys
        # collate ann_library_mzs
        ann_batched = batch_func(
            *[
                collate_data["ann_library_mzs"],
                collate_data["ann_isotope_idxs"],
                collate_data["ann_formula_idxs"],
            ],
            offset_flags=[False, False, True],
        )
        # # collate ann_formula_idxs
        # ann_batched_2 = batch_func(
        #   *[
        #       collate_data["ann_formula_idxs"],
        #       collate_data["ann_dup_idxs"]
        #   ],
        #   offset_flags=[False,True],
        # )
        collate_data["ann_library_mzs"] = ann_batched[0]
        collate_data["ann_isotope_idxs"] = ann_batched[1]
        collate_data["ann_formula_idxs"] = ann_batched[2]
        collate_data["ann_batch_idxs"] = ann_batched[3]
        # collate_data["ann_formula_idxs"] = ann_batched[0]
        # collate_data["ann_dup_idxs"] = ann_batched_2[1]
        # remove from list
        keys.remove("ann_library_mzs")
        keys.remove("ann_formula_idxs")
        keys.remove("ann_isotope_idxs")
        # keys.remove("ann_dup_idxs")
        SpecMolDataset._special_collate(keys, collate_data)

    @staticmethod
    def collate_fn(data_list):
        batch_size, keys, collate_data = SpecMolAnnDataset._setup_collate(data_list)
        SpecMolAnnDataset._special_collate(keys, collate_data)
        SpecMolAnnDataset._standard_collate(batch_size, keys, collate_data)
        return collate_data
