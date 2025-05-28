from fragnnet.utils.misc_utils import LOG_ZERO, flatten_lol, js_div
from fragnnet.utils.frag_utils import load_frag_d
from fragnnet.utils.data_utils import composition_to_string
from fragnnet.utils.spec_utils import calculate_match_mzs
from fragnnet.iceberg.common.chem_utils import vec_to_formula, element_to_ind

import numpy as np
import torch as th
import pandas as pd
import argparse	
import os
from tqdm import tqdm
from pyteomics.mass import Composition


parser = argparse.ArgumentParser()
parser.add_argument(
    "--split_types", 
    type=str, 
    nargs="+", 
    default=["inchikey","scaffold"]
)
parser.add_argument(
    "--model_types", 
    type=str, 
    nargs="+", 
    default=[
        "fragnnet_d4",
        "iceberg_inten",
        "graff"
    ]
)
parser.add_argument(
    "--vals_dp", 
    type=str, 
    required=True
)
parser.add_argument(
    "--seeds",
    type=int,
    nargs="+",
    default=[0,1,2,3,4]
)
parser.add_argument(
    "--thresholds",
    type=float,
    nargs="+",
    default=[
        0. #, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2
    ]
)
parser.add_argument("--num_samples", type=int, default=-1)
parser.add_argument("--output_dp", type=str, required=True)
args = parser.parse_args()

if __name__ == "__main__":

    # load ann_df
    # get formulae from products
    proc_dp = "./data/proc/nist20"
    spec_df = pd.read_pickle(f"{proc_dp}/spec_df.pkl")
    mol_df = pd.read_pickle(f"{proc_dp}/mol_df.pkl")
    ann_df = pd.read_pickle(f"{proc_dp}/ann_df.pkl")

    print(ann_df.columns)

    # merge ann_df
    m_ann_df = ann_df[["spec_id","ann_products","ann_exact_mzs"]].copy()
    m_ann_df = m_ann_df.merge(spec_df[["spec_id","group_id"]], on="spec_id", how="inner")
    m_ann_df = m_ann_df.drop(columns="spec_id").groupby("group_id").agg({"ann_products":list, "ann_exact_mzs":list}).reset_index()

    # dedup formulae
    def formula_dedup_fn(row):
        formulae = flatten_lol(row["ann_products"])
        mzs = flatten_lol(row["ann_exact_mzs"])
        assert len(formulae) == len(mzs), (len(formulae), len(mzs))
        formulae_dedup = []
        mzs_dedup = []
        for i in range(len(formulae)):
            formula = formulae[i]
            mz = mzs[i]
            if formula not in formulae_dedup:
                assert mz not in mzs_dedup, (i, mz, mzs_dedup)
                formulae_dedup.append(formula)
                mzs_dedup.append(mz)
        # assert len(formulae_dedup) == len(set(formulae_dedup)), (len(formulae_dedup), len(set(formulae_dedup)))
        out_d = {
            "group_id": row["group_id"],
            "formulae_dedup": formulae_dedup,
            "mzs_dedup": mzs_dedup
        }
        return out_d

    m_ann_df = m_ann_df.apply(formula_dedup_fn, axis=1, result_type="expand")

    print(ann_df["spec_id"].nunique())
    print(m_ann_df["group_id"].nunique())
    print(m_ann_df.columns)

    split_types = args.split_types
    model_types = args.model_types
    vals_dp = args.vals_dp
    seeds = args.seeds
    thresholdses = args.thresholds
    num_samples = args.num_samples

    stats_df_rows = []
    stats_d = {}

    for split_type in split_types:

        print(f">>>>> Split Type = {split_type}")

        for model_type in model_types:
            
            print(f">>>> Model Type = {model_type}")

            frag_model = ("fragnnet_d3" in model_type) or ("fragnnet_d4" in model_type)
            graff_model = "graff" in model_type

            # load vals for s0
            formula_vals_fp = os.path.join(vals_dp,split_type,model_type,f"s{seeds[0]}","test.pkl")
            formula_vals = th.load(formula_vals_fp)

            # get molecule ids
            group_ids = formula_vals["unique_id"].flatten().numpy()
            id_df = spec_df[["spec_id","mol_id","group_id"]]
            id_df = id_df[id_df["group_id"].isin(group_ids)]
            group_mol_df = id_df[["group_id","mol_id"]].drop_duplicates()
            mol_ids = group_mol_df.set_index("group_id").loc[group_ids,"mol_id"].values

            for n in seeds:

                print(f">>> Seed = {n}")

                if n != seeds[0]:
                
                    formula_vals_fp = os.path.join(vals_dp,split_type,model_type,f"s{n}","test.pkl")
                    formula_vals = th.load(formula_vals_fp)

                print(len(mol_ids))

                if frag_model:
                    print(np.isin(group_ids,m_ann_df["group_id"]).mean())

                unique_ids = formula_vals["unique_id"]
                true_batch_idxs = formula_vals["true_batch_idxs"]
                pred_batch_idxs = formula_vals["pred_batch_idxs"]
                true_mzs = formula_vals["true_mzs"]
                true_logprobs = formula_vals["true_logprobs"]
                pred_mzs = formula_vals["pred_mzs"]
                pred_logprobs = formula_vals["pred_logprobs"]
                spec_batch_idxs = th.arange(unique_ids.shape[0])
                
                # formula_spec_logprobs are the logprobs assigned to peaks centered on those formulae in the spectrum
                # formula_logprobs are the logprobs assigned to those formulae in the formula distribution
                # these are not the same in FraGNNet, but they are in GrAFF
                # undefined in ICEBERG
                if frag_model:
                    pred_formula_batch_idxs = formula_vals["pred_formula_batch_idxs"]
                    pred_formula_logprobs = formula_vals["pred_formula_logprobs"]
                    pred_formula_str = formula_vals["pred_formula_str"]
                    # remove null formula
                    pred_nonnull_formula_mask = pred_formula_str != ""
                    pred_formula_str = pred_formula_str[pred_nonnull_formula_mask]
                    pred_formula_logprobs = pred_formula_logprobs[th.as_tensor(pred_nonnull_formula_mask)]
                    pred_formula_batch_idxs = pred_formula_batch_idxs[th.as_tensor(pred_nonnull_formula_mask)]
                    pred_formula_mzs = pred_mzs
                    pred_formula_spec_logprobs = pred_logprobs
                    assert pred_formula_mzs.shape[0] == pred_formula_logprobs.shape[0] == pred_formula_batch_idxs.shape[0], (pred_formula_mzs.shape[0], pred_formula_logprobs.shape[0], pred_formula_batch_idxs.shape[0])

                elif graff_model:
                    # remove isotopes that are non-zero
                    pred_isotope_idxs = formula_vals["pred_isotope_idxs"]
                    pred_no_isotope_mask = pred_isotope_idxs == 0
                    pred_formula_batch_idxs = pred_batch_idxs[pred_no_isotope_mask]
                    pred_formula_spec_logprobs = pred_formula_logprobs = pred_logprobs[pred_no_isotope_mask]
                    pred_formula_mzs = pred_mzs[pred_no_isotope_mask]
                    pred_formula_str = formula_vals["pred_formula_str"][pred_no_isotope_mask.numpy()]
                
                else:
                    # iceberg model - no formula logprobs!
                    pred_formula_batch_idxs = formula_vals["pred_formula_batch_idxs"]
                    pred_formula_mzs = formula_vals["pred_formula_mzs"]
                    pred_formula_str = formula_vals["pred_formula_str"]

                m_ann_df_cp = m_ann_df.copy().set_index("group_id")

                if frag_model:
                    thresholds = thresholdses
                else:
                    thresholds = [0.]

                jsds, ann_peak_covs, ann_w_peak_covs = [], [], []
                both_t_formula_covs, both_t_peak_covs, both_t_w_peak_covs, both_t_peak_covs_2, both_t_w_peak_covs_2 = [], [], [], [], []
                both_p_formula_covs, both_p_peak_covs, both_p_w_peak_covs, both_p_peak_covs_2, both_p_w_peak_covs_2 = [], [], [], [], []
                mode_peak_covs, mode_w_peak_covs = [], []

                if num_samples > 0:
                    num_samples = min(len(group_ids), num_samples)
                else:
                    num_samples = len(group_ids)

                # iterate over the group truth data
                for i in tqdm(range(num_samples)):

                    i_group_id = group_ids[i]
                    i_mol_id = mol_ids[i]
                    if i_group_id not in m_ann_df_cp.index:
                        continue

                    # pprint(i_formula_idx_to_str)
                    i_mask = (unique_ids == i_group_id).flatten()
                    i_batch_idx = spec_batch_idxs[i_mask].item()
                    i_pred_peak_mask = (pred_batch_idxs == i_batch_idx)
                    i_true_peak_mask = (true_batch_idxs == i_batch_idx)
                    i_true_mzs = true_mzs[i_true_peak_mask]
                    i_true_logprobs = true_logprobs[i_true_peak_mask]
                    assert th.isclose(th.logsumexp(i_true_logprobs,dim=0),th.tensor(0.),rtol=0.,atol=1e-4), th.logsumexp(i_true_logprobs,dim=0)

                    i_formula_mask = (pred_formula_batch_idxs == i_batch_idx).flatten()
                    i_pred_formulae = pred_formula_str[i_formula_mask]
                    i_pred_formula_mzs = pred_formula_mzs[i_formula_mask]
                    assert len(i_pred_formulae) == len(np.unique(i_pred_formulae)), (len(i_pred_formulae), len(np.unique(i_pred_formulae)))

                    i_ann_formulae = m_ann_df_cp.loc[i_group_id,"formulae_dedup"]
                    i_ann_formulae = [
                        composition_to_string(Composition(formula=formula)) for formula in i_ann_formulae
                    ]
                    i_ann_formulae = np.array(i_ann_formulae)
                    assert len(i_ann_formulae) == len(np.unique(i_ann_formulae)), (len(i_ann_formulae), len(np.unique(i_ann_formulae)))
                    i_ann_mzs = th.tensor(m_ann_df_cp.loc[i_group_id,"mzs_dedup"])

                    # get idxs
                    i_both_formulae = np.union1d(i_pred_formulae, i_ann_formulae)
                    i_both_formula_idxs_2 = np.arange(len(i_both_formulae))
                    i_pred_formula_idxs_2 = th.as_tensor(i_both_formula_idxs_2[np.isin(i_both_formulae, i_pred_formulae)])
                    i_ann_formula_idxs_2 = th.as_tensor(i_both_formula_idxs_2[np.isin(i_both_formulae, i_ann_formulae)])
                    assert i_pred_formula_idxs_2.shape[0] == len(i_pred_formulae), (i_pred_formula_idxs_2.shape[0], len(i_pred_formulae))
                    assert i_ann_formula_idxs_2.shape[0] == len(i_ann_formulae), (i_ann_formula_idxs_2.shape[0], len(i_ann_formulae))

                    # match formulae within 10 ppm
                    i_ann_mz_match_mask = calculate_match_mzs(
                        true_mzs=i_true_mzs,
                        pred_mzs=i_ann_mzs, 
                        tolerance=1e-5, # 10 ppm
                        relative=True
                    ).T # A x T
                    i_ann_mz_match_any = i_ann_mz_match_mask.any(dim=0) # T

                    i_ann_formula_logprobs = (i_true_logprobs.unsqueeze(0) - th.log(th.sum(i_ann_mz_match_mask, dim=1, keepdim=True)))
                    i_ann_formula_logprobs[~i_ann_mz_match_mask] = LOG_ZERO(i_ann_formula_logprobs.dtype)
                    i_ann_formula_logprobs = th.logsumexp(i_ann_formula_logprobs, dim=1)
                    
                    # calculate fraction with annotations
                    i_ann_peak_cov = i_ann_mz_match_any.float().mean()
                    i_ann_w_peak_cov = th.exp(th.logsumexp(i_true_logprobs[i_ann_mz_match_any], dim=0))
                    ann_peak_covs.append(i_ann_peak_cov.item())
                    ann_w_peak_covs.append(i_ann_w_peak_cov.item())

                    i_pred_mz_match_mask = calculate_match_mzs(
                        true_mzs=i_true_mzs,
                        pred_mzs=i_pred_formula_mzs, 
                        tolerance=1e-5, # 10 ppm
                        relative=True
                    ).T # P x T
                    i_pred_mz_match_any = i_pred_mz_match_mask.any(dim=0) # T

                    if frag_model or graff_model:
                        # get formula logprobs
                        i_pred_formula_logprobs = pred_formula_logprobs[i_formula_mask]
                        i_pred_formula_spec_logprobs = pred_formula_spec_logprobs[i_formula_mask]
                        # calculate JSD
                        assert i_pred_formula_idxs_2.shape[0] == i_pred_formula_logprobs.shape[0], (i_pred_formula_idxs_2.shape[0], i_pred_formula_logprobs.shape[0])
                        assert i_ann_formula_idxs_2.shape[0] == i_ann_formula_logprobs.shape[0], (i_ann_formula_idxs_2.shape[0], i_ann_formula_logprobs.shape[0])
                        i_formula_jsd = js_div(
                            i_pred_formula_idxs_2,
                            i_pred_formula_logprobs,
                            i_ann_formula_idxs_2,
                            i_ann_formula_logprobs
                        )
                        jsds.append(i_formula_jsd.item())

                        i_pred_mz_match_argmax = th.argmax(i_pred_mz_match_mask * i_pred_formula_logprobs.exp().unsqueeze(1), dim=0)
                        i_mode_mz_match_mask = th.zeros_like(i_pred_mz_match_mask)
                        i_mode_mz_match_mask[i_pred_mz_match_argmax[i_pred_mz_match_any],th.arange(i_pred_mz_match_mask.shape[1],device=i_mode_mz_match_mask.device)[i_pred_mz_match_any]] = 1

                    # calculate cov
                    i_both_t_formula_covs, i_both_t_peak_covs, i_both_t_w_peak_covs, i_both_t_peak_covs_2, i_both_t_w_peak_covs_2 = [], [], [], [], []
                    i_both_p_formula_covs, i_both_p_peak_covs, i_both_p_w_peak_covs, i_both_p_peak_covs_2, i_both_p_w_peak_covs_2 = [], [], [], [], []
                    for threshold in thresholds:
                        # note: this is not the same shape as the i_formula_mask
                        if threshold == 0.:
                            i_pt_formula_mask = th.ones_like(i_pred_formula_mzs,dtype=th.bool)
                        else:
                            assert frag_model or graff_model
                            i_pt_formula_mask = (i_pred_formula_logprobs >= np.log(threshold))
                        i_pt_pred_formulae = i_pred_formulae[i_pt_formula_mask.numpy()]
                        i_both_formula_match_mask = th.as_tensor(
                            i_ann_formulae.reshape(-1,1) == i_pt_pred_formulae.reshape(1,-1)
                        ) # A x P
                        
                        # true spectrum stuff
                        i_both_t_formula_cov = len(np.intersect1d(i_pt_pred_formulae,i_ann_formulae)) / len(np.unique(i_ann_formulae))
                        i_ann_mz_match_mask_2 = (i_both_formula_match_mask.long() @ i_pred_mz_match_mask[i_pt_formula_mask].long()) > 0 # A x T
                        i_both_t_peak_match = (i_ann_mz_match_mask & i_ann_mz_match_mask_2).any(dim=0) # T
                        # numerator: number of true peaks with an true annotation that is matched by a predicted annotation
                        # denominator: number of true peaks with an true annotation
                        i_both_t_peak_cov = i_both_t_peak_match.float().sum() / i_ann_mz_match_any.sum()
                        # same but weighted
                        i_both_t_w_peak_cov = (i_both_t_peak_match.float() * i_true_logprobs.exp()).sum() / (i_ann_mz_match_any * i_true_logprobs.exp()).sum()
                        # numerator: number of true peaks with an true annotation that is matched by a predicted annotation
                        # denominator: number of true peaks
                        i_both_t_peak_cov_2 = i_both_t_peak_match.float().mean()
                        # same but weighted (i_true_logprobs sums to 1)
                        i_both_t_w_peak_cov_2 = (i_both_t_peak_match.float() * i_true_logprobs.exp()).sum()

                        # pred spectrum stuff
                        i_both_p_formula_cov = len(np.intersect1d(i_pt_pred_formulae,i_ann_formulae)) / len(np.unique(i_pt_pred_formulae))
                        # which peak annotations match which true annotations
                        i_pred_mz_match_mask_2 = (i_both_formula_match_mask.T[i_pt_formula_mask].long() @ i_ann_mz_match_mask.long()) > 0 # P x T
                        i_pred_mz_match_mask_3 = (i_pred_mz_match_mask[i_pt_formula_mask].long() @ i_ann_mz_match_mask.T.long()) > 0 # P x A
                        i_pred_ann_mask = i_pred_mz_match_mask_3.any(dim=1) # P
                        i_both_p_peak_match = (i_pred_mz_match_mask[i_pt_formula_mask] & i_pred_mz_match_mask_2).any(dim=1) # P
                        # numerator: number of predicted peaks with a predicted annotation that is matched by a true annotation
                        # denominator: number of predicted peaks that are matched by a true peak with an annotation
                        i_both_p_peak_cov = i_both_p_peak_match.float().sum() / i_pred_ann_mask.sum()
                        # numerator: number of predicted peaks with a predicted annotation that is matched by a true annotation
                        # denominator: number of predicted peaks
                        i_both_p_peak_cov_2 = i_both_p_peak_match.float().mean()

                        if frag_model or graff_model:
                            # same but weighted
                            i_both_p_w_peak_cov = (i_both_p_peak_match.float() * i_pred_formula_spec_logprobs.exp()).sum() / (i_pred_ann_mask * i_pred_formula_spec_logprobs.exp()).sum()
                            # same but weighted (note that i_pred_formula_spec_logprobs does not necessarily sum to 1, because of OOS)
                            i_both_p_w_peak_cov_2 = (i_both_p_peak_match.float() * i_pred_formula_spec_logprobs.exp()).sum()

                        # update lists
                        i_both_t_formula_covs.append(i_both_t_formula_cov)
                        i_both_t_peak_covs.append(i_both_t_peak_cov.item())
                        i_both_t_w_peak_covs.append(i_both_t_w_peak_cov.item())
                        i_both_t_peak_covs_2.append(i_both_t_peak_cov_2.item())
                        i_both_t_w_peak_covs_2.append(i_both_t_w_peak_cov_2.item())
                        i_both_p_formula_covs.append(i_both_p_formula_cov)
                        i_both_p_peak_covs.append(i_both_p_peak_cov.item())
                        i_both_p_peak_covs_2.append(i_both_p_peak_cov_2.item())
                        if frag_model or graff_model:
                            i_both_p_w_peak_covs.append(i_both_p_w_peak_cov.item())
                            i_both_p_w_peak_covs_2.append(i_both_p_w_peak_cov_2.item())

                        if threshold == 0. and (frag_model or graff_model):
                            i_mode_mz_match_mask_2 = (i_both_formula_match_mask.long() @ i_mode_mz_match_mask.long()) > 0
                            i_mode_peak_match = (i_ann_mz_match_mask & i_mode_mz_match_mask_2).any(dim=0)
                            i_mode_peak_cov = i_mode_peak_match.float().sum() / i_ann_mz_match_any.sum()
                            i_mode_w_peak_cov = (i_mode_peak_match.float() * i_true_logprobs.exp()).sum() / (i_ann_mz_match_any * i_true_logprobs.exp()).sum()
                            mode_peak_covs.append(i_mode_peak_cov.item())
                            mode_w_peak_covs.append(i_mode_w_peak_cov.item())

                    both_t_formula_covs.append(i_both_t_formula_covs)
                    both_t_peak_covs.append(i_both_t_peak_covs)
                    both_t_w_peak_covs.append(i_both_t_w_peak_covs)
                    both_t_peak_covs_2.append(i_both_t_peak_covs_2)
                    both_t_w_peak_covs_2.append(i_both_t_w_peak_covs_2)
                    both_p_formula_covs.append(i_both_p_formula_covs)
                    both_p_peak_covs.append(i_both_p_peak_covs)
                    both_p_peak_covs_2.append(i_both_p_peak_covs_2)
                    if frag_model or graff_model:
                        both_p_w_peak_covs.append(i_both_p_w_peak_covs)
                        both_p_w_peak_covs_2.append(i_both_p_w_peak_covs_2)

                jsds = np.array(jsds)
                ann_peak_covs = np.array(ann_peak_covs)
                ann_w_peak_covs = np.array(ann_w_peak_covs)
                both_t_formula_covs = np.array(both_t_formula_covs)
                both_t_peak_covs = np.array(both_t_peak_covs)
                both_t_w_peak_covs = np.array(both_t_w_peak_covs)
                both_t_peak_covs_2 = np.array(both_t_peak_covs_2)
                both_t_w_peak_covs_2 = np.array(both_t_w_peak_covs_2)
                both_p_formula_covs = np.array(both_p_formula_covs)
                both_p_peak_covs = np.array(both_p_peak_covs)
                both_p_w_peak_covs = np.array(both_p_w_peak_covs)
                both_p_peak_covs_2 = np.array(both_p_peak_covs_2)
                both_p_w_peak_covs_2 = np.array(both_p_w_peak_covs_2)
                mode_peak_covs = np.array(mode_peak_covs)
                mode_w_peak_covs = np.array(mode_w_peak_covs)

                stats_df_row = {
                    "split_type": split_type,
                    "model_type": model_type,
                    "seed": n
                }

                if frag_model or graff_model:
                    jsd_mean = np.nanmean(jsds)
                    jsd_std = np.nanstd(jsds)
                    print(f">> JSD: {jsd_mean:.4f} +/- {jsd_std:.4f}")
                    stats_df_row["jsd_mean"] = jsd_mean
                    stats_df_row["jsd_std"] = jsd_std
                else:
                    stats_df_row["jsd_mean"] = np.nan
                    stats_df_row["jsd_std"] = np.nan

                print(f">> Ann Peak Coverage: {np.nanmean(ann_peak_covs):.4f} +/- {np.nanstd(ann_peak_covs):.4f}")
                print(f">> Ann W Peak Coverage: {np.nanmean(ann_w_peak_covs):.4f} +/- {np.nanstd(ann_w_peak_covs):.4f}")
                stats_df_row["ann_peak_cov_mean"] = np.nanmean(ann_peak_covs)
                stats_df_row["ann_peak_cov_std"] = np.nanstd(ann_peak_covs)
                stats_df_row["ann_w_peak_cov_mean"] = np.nanmean(ann_w_peak_covs)
                stats_df_row["ann_w_peak_cov_std"] = np.nanstd(ann_w_peak_covs)

                if frag_model or graff_model:
                    print(f">> Mode Peak Coverage: {np.nanmean(mode_peak_covs):.4f} +/- {np.nanstd(mode_peak_covs):.4f}")
                    print(f">> Mode W Peak Coverage: {np.nanmean(mode_w_peak_covs):.4f} +/- {np.nanstd(mode_w_peak_covs):.4f}")
                    stats_df_row["mode_peak_cov_mean"] = np.nanmean(mode_peak_covs)
                    stats_df_row["mode_peak_cov_std"] = np.nanstd(mode_peak_covs)
                    stats_df_row["mode_w_peak_cov_mean"] = np.nanmean(mode_w_peak_covs)
                    stats_df_row["mode_w_peak_cov_std"] = np.nanstd(mode_w_peak_covs)

                for i, threshold in enumerate(thresholds):
                    print(f">> Threshold = {threshold}")
                    print(f"> Both True Formula Coverage: {np.nanmean(both_t_formula_covs[:,i]):.4f} +/- {np.nanstd(both_t_formula_covs[:,i]):.4f}") 
                    print(f"> Both True Peak Coverage: {np.nanmean(both_t_peak_covs[:,i]):.4f} +/- {np.nanstd(both_t_peak_covs[:,i]):.4f}")
                    print(f"> Both True W Peak Coverage: {np.nanmean(both_t_w_peak_covs[:,i]):.4f} +/- {np.nanstd(both_t_w_peak_covs[:,i]):.4f}")
                    print(f"> Both True Peak Coverage 2: {np.nanmean(both_t_peak_covs_2[:,i]):.4f} +/- {np.nanstd(both_t_peak_covs_2[:,i]):.4f}")
                    print(f"> Both True W Peak Coverage 2: {np.nanmean(both_t_w_peak_covs_2[:,i]):.4f} +/- {np.nanstd(both_t_w_peak_covs_2[:,i]):.4f}")
                    stats_df_row[f"both_t_formula_cov_{threshold}_mean"] = np.nanmean(both_t_formula_covs[:,i])
                    stats_df_row[f"both_t_formula_cov_{threshold}_std"] = np.nanstd(both_t_formula_covs[:,i])
                    stats_df_row[f"both_t_peak_cov_{threshold}_mean"] = np.nanmean(both_t_peak_covs[:,i])
                    stats_df_row[f"both_t_peak_cov_{threshold}_std"] = np.nanstd(both_t_peak_covs[:,i])
                    stats_df_row[f"both_t_w_peak_cov_{threshold}_mean"] = np.nanmean(both_t_w_peak_covs[:,i])
                    stats_df_row[f"both_t_w_peak_cov_{threshold}_std"] = np.nanstd(both_t_w_peak_covs[:,i])
                    stats_df_row[f"both_t_peak_cov_2_{threshold}_mean"] = np.nanmean(both_t_peak_covs_2[:,i])
                    stats_df_row[f"both_t_peak_cov_2_{threshold}_std"] = np.nanstd(both_t_peak_covs_2[:,i])
                    stats_df_row[f"both_t_w_peak_cov_2_{threshold}_mean"] = np.nanmean(both_t_w_peak_covs_2[:,i])
                    stats_df_row[f"both_t_w_peak_cov_2_{threshold}_std"] = np.nanstd(both_t_w_peak_covs_2[:,i])
                    print(f"> Both Pred Formula Coverage: {np.nanmean(both_p_formula_covs[:,i]):.4f} +/- {np.nanstd(both_p_formula_covs[:,i]):.4f}") 
                    print(f"> Both Pred Peak Coverage: {np.nanmean(both_p_peak_covs[:,i]):.4f} +/- {np.nanstd(both_p_peak_covs[:,i]):.4f}")
                    print(f"> Both Pred Peak Coverage 2: {np.nanmean(both_p_peak_covs_2[:,i]):.4f} +/- {np.nanstd(both_p_peak_covs_2[:,i]):.4f}")
                    stats_df_row[f"both_p_formula_cov_{threshold}_mean"] = np.nanmean(both_p_formula_covs[:,i])
                    stats_df_row[f"both_p_formula_cov_{threshold}_std"] = np.nanstd(both_p_formula_covs[:,i])
                    stats_df_row[f"both_p_peak_cov_{threshold}_mean"] = np.nanmean(both_p_peak_covs[:,i])
                    stats_df_row[f"both_p_peak_cov_{threshold}_std"] = np.nanstd(both_p_peak_covs[:,i])
                    stats_df_row[f"both_p_peak_cov_2_{threshold}_mean"] = np.nanmean(both_p_peak_covs_2[:,i])
                    stats_df_row[f"both_p_peak_cov_2_{threshold}_std"] = np.nanstd(both_p_peak_covs_2[:,i])
                    if frag_model or graff_model:
                        print(f"> Both Pred W Peak Coverage: {np.nanmean(both_p_w_peak_covs[:,i]):.4f} +/- {np.nanstd(both_p_w_peak_covs[:,i]):.4f}")
                        print(f"> Both Pred W Peak Coverage 2: {np.nanmean(both_p_w_peak_covs_2[:,i]):.4f} +/- {np.nanstd(both_p_w_peak_covs_2[:,i]):.4f}")
                        stats_df_row[f"both_p_w_peak_cov_{threshold}_mean"] = np.nanmean(both_p_w_peak_covs[:,i])
                        stats_df_row[f"both_p_w_peak_cov_{threshold}_std"] = np.nanstd(both_p_w_peak_covs[:,i])
                        stats_df_row[f"both_p_w_peak_cov_2_{threshold}_mean"] = np.nanmean(both_p_w_peak_covs_2[:,i])
                        stats_df_row[f"both_p_w_peak_cov_2_{threshold}_std"] = np.nanstd(both_p_w_peak_covs_2[:,i])

                stats_df_rows.append(stats_df_row)

                stats_d[(split_type,model_type,n)] = {
                    "jsds": th.as_tensor(jsds),
                    "ann_peak_covs": th.as_tensor(ann_peak_covs),
                    "ann_w_peak_covs": th.as_tensor(ann_w_peak_covs),
                    "mode_peak_covs": th.as_tensor(mode_peak_covs),
                    "mode_w_peak_covs": th.as_tensor(mode_w_peak_covs),
                    "both_t_formula_covs": th.as_tensor(both_t_formula_covs),
                    "both_t_peak_covs": th.as_tensor(both_t_peak_covs),
                    "both_t_w_peak_covs": th.as_tensor(both_t_w_peak_covs),
                    "both_t_peak_covs_2": th.as_tensor(both_t_peak_covs_2),
                    "both_t_w_peak_covs_2": th.as_tensor(both_t_w_peak_covs_2),
                    "both_p_formula_covs": th.as_tensor(both_p_formula_covs),
                    "both_p_peak_covs": th.as_tensor(both_p_peak_covs),
                    "both_p_w_peak_covs": th.as_tensor(both_p_w_peak_covs),
                    "both_p_peak_covs_2": th.as_tensor(both_p_peak_covs_2),
                    "both_p_w_peak_covs_2": th.as_tensor(both_p_w_peak_covs_2),
                    "thresholds": th.tensor(thresholds)
                }

    stats_df = pd.DataFrame(stats_df_rows)
    print(stats_df)

    os.makedirs(args.output_dp, exist_ok=True)
    stats_df.to_csv(os.path.join(args.output_dp,"formula_stats_df.csv"), index=False)
    th.save(stats_d, os.path.join(args.output_dp,"formula_stats_d.pt"))

    import pdb; pdb.set_trace()

    # plot_histogram(
    #     w_peak_coverages[:,0], 
    #     bins=20, 
    #     x_range=(0,1),
    #     title=f"Weighted Peak Coverages (t={thresholds[0]})",
    #     x_label="Weighted Peak Coverage"
    # )

    # plot_histogram(
    #     w_peak_coverages[:,1], 
    #     bins=20, 
    #     x_range=(0,1),
    #     title=f"Weighted Peak Coverages (t={thresholds[1]})",
    #     x_label="Weighted Peak Coverage"
    # )

    # plot_histogram(
    #     w_peak_coverages[:,2], 
    #     bins=20, 
    #     x_range=(0,1),
    #     title=f"Weighted Peak Coverages (t={thresholds[2]})",
    #     x_label="Weighted Peak Coverage"
    # )
