
from fragnnet.utils.misc_utils import scatter_reduce, scatter_argtopk,  booltype, scatter_logsumexp, LOG_ZERO
from fragnnet.utils.script_utils import select_model_vals, log_mean, pearson_r
from fragnnet.loss import  sparse_cosine_distance_hungarian

import numpy as np
import torch as th
import argparse	
import os
import matplotlib.pyplot as plt
from scipy.stats import linregress


def ensemble_vals(vals, nb=True, spec=False, ens_conds=False):

    select_keys = [
        "pred_formula_logprobs",
        "pred_formula_batch_idxs",
        "pred_formula_formula_idxs",
    ]
    if spec:
        select_keys += [
            "pred_mzs",
            "pred_logprobs",
            "pred_batch_idxs",
            "true_mzs",
            "true_logprobs",
            "true_batch_idxs",
            "oos_prob",
            "true_oos_prob"
        ]

    if nb:
        select_keys += [
            "pred_nb_joint_logprobs",
            "pred_nb_formula_node_logprobs",
            "pred_nb_joint_formula_idxs",
            "pred_nb_joint_batch_idxs",
            "pred_nb_joint_node_idxs"
        ]
    else:
        select_keys += [
            "pred_joint_logprobs",
            "pred_formula_node_logprobs",
            "pred_joint_formula_idxs",
            "pred_joint_batch_idxs",
            "pred_joint_node_idxs",
        ]

    select_d = select_model_vals(vals,select_keys,stack_dim=0)
    # formula stuff
    ens_formula_logprobs = log_mean(select_d["pred_formula_logprobs"],dim=0)
    ens_formula_batch_idxs = select_d["pred_formula_batch_idxs"][0]
    ens_formula_formula_idxs = select_d["pred_formula_formula_idxs"][0]
    
    ens_d = {
        "ens_formula_logprobs": ens_formula_logprobs,
        "ens_formula_batch_idxs": ens_formula_batch_idxs,
        "ens_formula_formula_idxs": ens_formula_formula_idxs,
    }
    
    if spec:

        ens_pred_logprobs = log_mean(select_d["pred_logprobs"],dim=0)
        ens_pred_mzs = select_d["pred_mzs"][0]
        ens_pred_batch_idxs = select_d["pred_batch_idxs"][0]
        ens_true_logprobs = select_d["true_logprobs"][0]
        ens_true_mzs = select_d["true_mzs"][0]
        ens_true_batch_idxs = select_d["true_batch_idxs"][0]
        ens_pred_oos_prob = log_mean(select_d["oos_prob"],dim=0)
        ens_true_oos_prob = select_d["true_oos_prob"][0]
        ens_d.update({
            "ens_pred_logprobs": ens_pred_logprobs,
            "ens_pred_mzs": ens_pred_mzs,
            "ens_pred_batch_idxs": ens_pred_batch_idxs,
            "ens_true_logprobs": ens_true_logprobs,
            "ens_true_mzs": ens_true_mzs,
            "ens_true_batch_idxs": ens_true_batch_idxs,
            "ens_pred_oos_prob": ens_pred_oos_prob,
            "ens_true_oos_prob": ens_true_oos_prob
        })

    if nb:
        
        # non-iso joint stuff
        ens_nb_joint_formula_idxs = select_d["pred_nb_joint_formula_idxs"][0]
        ens_nb_joint_batch_idxs = select_d["pred_nb_joint_batch_idxs"][0]
        ens_nb_joint_node_idxs = select_d["pred_nb_joint_node_idxs"][0]
        if ens_conds:
            ens_nb_formula_node_logprobs = log_mean(select_d["pred_nb_formula_node_logprobs"],dim=0)
            ens_nb_joint_logprobs = ens_nb_formula_node_logprobs + ens_formula_logprobs[ens_nb_joint_formula_idxs]
        else:
            ens_nb_joint_logprobs = log_mean(select_d["pred_nb_joint_logprobs"],dim=0)
            ens_nb_formula_node_logprobs = ens_nb_joint_logprobs - ens_formula_logprobs[ens_nb_joint_formula_idxs]
            ens_nb_formula_node_logprobs = th.clamp(ens_nb_formula_node_logprobs, max=0.)
        ens_nb_formula_node_lse = scatter_logsumexp(ens_nb_formula_node_logprobs, ens_nb_joint_formula_idxs, dim_size=ens_formula_formula_idxs.shape[0])
    
        ens_d.update({
            "ens_nb_joint_logprobs": ens_nb_joint_logprobs,
            "ens_nb_joint_formula_idxs": ens_nb_joint_formula_idxs,
            "ens_nb_joint_batch_idxs": ens_nb_joint_batch_idxs,
            "ens_nb_joint_node_idxs": ens_nb_joint_node_idxs,
            "ens_nb_formula_node_logprobs": ens_nb_formula_node_logprobs,
            "ens_nb_formula_node_lse": ens_nb_formula_node_lse
        })

    else:

        # iso joint stuff
        ens_joint_formula_idxs = select_d["pred_joint_formula_idxs"][0]
        ens_joint_batch_idxs = select_d["pred_joint_batch_idxs"][0]
        ens_joint_node_idxs = select_d["pred_joint_node_idxs"][0]
        if ens_conds:
            ens_formula_node_logprobs = log_mean(select_d["pred_formula_node_logprobs"],dim=0)
            ens_joint_logprobs = ens_formula_node_logprobs + ens_formula_logprobs[ens_joint_formula_idxs]
        else:
            ens_joint_logprobs = log_mean(select_d["pred_joint_logprobs"],dim=0)
            ens_formula_node_logprobs = ens_joint_logprobs - ens_formula_logprobs[ens_joint_formula_idxs]
            ens_formula_node_logprobs = th.clamp(ens_formula_node_logprobs, max=0.)
        ens_formula_node_lse = scatter_logsumexp(ens_formula_node_logprobs, ens_joint_formula_idxs, dim_size=ens_formula_formula_idxs.shape[0])
    
        ens_d.update({
            "ens_joint_logprobs": ens_joint_logprobs,
            "ens_joint_formula_idxs": ens_joint_formula_idxs,
            "ens_joint_batch_idxs": ens_joint_batch_idxs,
            "ens_joint_node_idxs": ens_joint_node_idxs,
            "ens_formula_node_logprobs": ens_formula_node_logprobs,
            "ens_formula_node_lse": ens_formula_node_lse,
        })
    
    return ens_d

def get_topk_formula_node_idxs(ens_vals, k, nb=True, filter_thresh=0.01):

    ens_formula_logprobs = ens_vals["ens_formula_logprobs"]
    # boolean mask based on probability threshold
    ens_formula_mask = ens_formula_logprobs > np.log(filter_thresh)

    if nb:
        prefix = "nb_"
    else:
        prefix = ""

    # number of formulae
    ens_counts = scatter_reduce(
        src=th.ones_like(ens_vals[f"ens_{prefix}formula_node_logprobs"], dtype=th.long),
        index=ens_vals[f"ens_{prefix}joint_formula_idxs"],
        reduce="sum",
        dim_size=th.max(ens_vals[f"ens_{prefix}joint_formula_idxs"])+1
    )

    # argtop-k nodes per formula (represent with formula indices)
    ens_nb_formula_node_argtopk = scatter_argtopk(
        ens_vals[f"ens_{prefix}formula_node_logprobs"],
        ens_vals[f"ens_{prefix}joint_formula_idxs"],
        ens_vals[f"ens_{prefix}joint_node_idxs"],
        k,
        dim_size=ens_formula_mask.shape[0],
        return_max=False
    )

    return ens_formula_mask, ens_counts, ens_nb_formula_node_argtopk

def calculate_compare(masks, argtopks, batch_idxs):

    num_ens = len(masks)
    compare_mat_formula = th.zeros((num_ens,num_ens),dtype=th.float32)
    compare_mat_spec = th.zeros((num_ens,num_ens),dtype=th.float32)
    num_spec = batch_idxs.max() + 1

    for i in range(num_ens):
        for j in range(num_ens):
            # formula masks (based on probability and count thresholding)
            mask_i = masks[i]
            mask_j = masks[j]
            # top-k nodes per formula
            argtopk_i = argtopks[i]
            argtopk_j = argtopks[j]
            # compare top-k node assignments for valid formulae
            mask_ij = mask_i & mask_j
            equal_ij = th.all(argtopk_i[mask_ij] == argtopk_j[mask_ij], dim=1)
            compare_spec_ij_num = scatter_reduce(
                equal_ij.float(),
                batch_idxs[mask_ij],
                "sum",
                dim_size=num_spec
            )
            compare_spec_ij_den = scatter_reduce(
                mask_ij.float(),
                batch_idxs,
                "sum",
                dim_size=num_spec
            )
            compare_spec_ij = compare_spec_ij_num / compare_spec_ij_den
            compare_mat_spec[i,j] = compare_spec_ij.nanmean()
            compare_mat_formula[i,j] = equal_ij.float().mean()

    # collect all formula masks, top-k nodes
    mask_all = th.stack(masks, dim=0).all(dim=0)
    argtopks_all = th.stack(argtopks, dim=0)
    # get the formulae where all models agree
    equal_all = (argtopks_all == argtopks[0].unsqueeze(0))
    equal_all = equal_all.all(2).all(0)
    equal_all[~mask_all] = False
    # calculate the fraction of valid formulae where all models agree
    compare_all_formula = equal_all.float().sum() / mask_all.float().sum()
    compare_all_spec_num = scatter_reduce(
        equal_all.float(),
        batch_idxs,
        "sum",
        dim_size=num_spec
    )
    compare_all_spec_den = scatter_reduce(
        mask_all.float(),
        batch_idxs,
        "sum",
        dim_size=num_spec
    )
    compare_all_spec = (compare_all_spec_num / compare_all_spec_den).nanmean()

    return compare_mat_formula, compare_mat_spec, compare_all_formula, compare_all_spec, equal_all


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=str, default="d4", choices=["d3","d4"])
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--nb", type=booltype, default=True)
    parser.add_argument("--ens_conds", type=booltype, default=True)
    parser.add_argument("--output_dp", type=str, default="inference_outputs/entropy/")
    parser.add_argument("--ens_dp", type=str, default="inference_outputs/entropy_ens/")
    parser.add_argument("--ens_types", type=str, nargs="+", default=["baseline", "low", "high", "mix"])
    parser.add_argument("--stats_dp", type=str, default="figs/ensemble/")
    args = parser.parse_args()

    depth = args.depth
    k = args.k
    nb = args.nb
    ens_conds = args.ens_conds
    output_dp = args.output_dp
    ens_dp = args.ens_dp
    ens_types = args.ens_types
    nb_flag = "_nb" if nb else ""
    conds_flag = "_c" if ens_conds else ""

    ens_vals_d = {}
    ens_type_to_key = {
        "baseline": "baseline",
        "low": "posfn",
        "high": "negfn",
        "mix": "mix"
    }

    stats_d = {}

    for ens_type in ens_types:

        ens_type_key = ens_type_to_key[ens_type]
        entropy_fp = os.path.join(output_dp, f"scaffold_{depth}_{ens_type_key}_test_False/vals.pkl")
        ens_type_dp = os.path.join(ens_dp, f"scaffold_{depth}_{ens_type}_test_False")
        ens_type_fp = os.path.join(ens_type_dp, "ens_vals" + nb_flag + conds_flag + ".pkl")

        if os.path.isfile(ens_type_fp):

            print(f"> loading \"{ens_type}\" ensemble values from {ens_type_fp}")
            ens_vals = th.load(ens_type_fp)

        else:

            print(f"> loading \"{ens_type}\" entropy values from {entropy_fp} to create ensemble")
            assert os.path.isfile(entropy_fp), f"File not found: {entropy_fp}"
            vals = th.load(entropy_fp)
            ens_vals = ensemble_vals(vals, nb=nb, ens_conds=ens_conds, spec=(ens_type == "mix"))
            del vals
            os.makedirs(ens_type_dp, exist_ok=True)
            th.save(ens_vals, ens_type_fp)

        ens_vals_d[ens_type] = ens_vals

    baseline_ens_vals = ens_vals_d["baseline"]
    low_ens_vals = ens_vals_d["low"]
    high_ens_vals = ens_vals_d["high"]
    mix_ens_vals = ens_vals_d["mix"]

    print(">> baseline")
    baseline_mask, baseline_counts, baseline_argtopk = get_topk_formula_node_idxs(baseline_ens_vals, k, nb=nb)
    print(th.unique(baseline_counts[baseline_mask], return_counts=True)[1][:10])
    print((baseline_counts[baseline_mask] < k).float().mean().item())
    print((baseline_argtopk[baseline_mask] == -1).any(dim=1).float().mean().item())

    print(">> low")
    low_mask, low_counts, low_argtopk = get_topk_formula_node_idxs(low_ens_vals, k, nb=nb)
    print(th.unique(low_counts[low_mask], return_counts=True)[1][:10])
    print((low_counts[low_mask] < k).float().mean().item())
    print((low_argtopk[low_mask] == -1).any(dim=1).float().mean().item())

    print(">> high")
    high_mask, high_counts, high_argtopk = get_topk_formula_node_idxs(high_ens_vals, k, nb=nb)
    print(th.unique(high_counts[high_mask], return_counts=True)[1][:10])
    print((high_counts[high_mask] < k).float().mean().item())
    print((high_argtopk[high_mask] == -1).any(dim=1).float().mean().item())

    print(">> mix")
    mix_mask, mix_counts, mix_argtopk = get_topk_formula_node_idxs(mix_ens_vals, k, nb=nb)
    print(th.unique(mix_counts[mix_mask], return_counts=True)[1][:10])
    print((mix_counts[mix_mask] < k).float().mean().item())
    print((mix_argtopk[mix_mask] == -1).any(dim=1).float().mean().item())

    baseline_k_mask = baseline_mask & (baseline_counts > k)
    low_k_mask = low_mask & (low_counts > k)
    high_k_mask = high_mask & (high_counts > k)
    mix_k_mask = mix_mask & (mix_counts > k)
    print(baseline_mask.float().mean(), baseline_k_mask.float().mean().item())
    print(low_mask.float().mean(), low_k_mask.float().mean().item())
    print(high_mask.float().mean(), high_k_mask.float().mean().item())
    print(mix_mask.float().mean(), mix_k_mask.float().mean().item())    

    print(f">> compare ensembles")
    print(f"> compare_mat")
    compare_mat_formula, compare_mat_spec, compare_all_formula, compare_all_spec, equal_all = calculate_compare(
        [baseline_mask, low_mask, high_mask, mix_mask],
        [baseline_argtopk, low_argtopk, high_argtopk, mix_argtopk],
        baseline_ens_vals["ens_formula_batch_idxs"]
    )
    print(compare_mat_formula)
    print(compare_mat_spec)
    print(f"> compare_all")
    print(compare_all_formula.item())
    print(compare_all_spec.item())
    # update stats_d
    stats_d["compare_mat_formula"] = compare_mat_formula
    stats_d["compare_mat_spec"] = compare_mat_spec
    stats_d["compare_all_formula"] = compare_all_formula
    stats_d["compare_all_spec"] = compare_all_spec

    print(f">> compare ensembles (k > {k})")
    print(f"> compare_mat_{k}")
    compare_mat_formula_k, compare_mat_spec_k, compare_all_formula_k, compare_all_spec_k, equal_all_k = calculate_compare(
        [baseline_k_mask, low_k_mask, high_k_mask, mix_k_mask],
        [baseline_argtopk, low_argtopk, high_argtopk, mix_argtopk],
        baseline_ens_vals["ens_formula_batch_idxs"]
    )
    print(compare_mat_formula_k)
    print(compare_mat_spec_k)
    print(f"> compare_all_{k}")
    print(compare_all_formula_k.item())
    print(compare_all_spec_k.item())
    # update stats_d
    stats_d[f"compare_mat_formula_{k}"] = compare_mat_formula_k
    stats_d[f"compare_mat_spec_{k}"] = compare_mat_spec_k
    stats_d[f"compare_all_formula_{k}"] = compare_all_formula_k
    stats_d[f"compare_all_spec_{k}"] = compare_all_spec_k

    print(">> mix ensemble cos_hun")
    mix_ens_cos_hun = 1. - sparse_cosine_distance_hungarian(
        mix_ens_vals["ens_true_mzs"],
        mix_ens_vals["ens_true_logprobs"],
        mix_ens_vals["ens_true_batch_idxs"],
        mix_ens_vals["ens_pred_mzs"],
        mix_ens_vals["ens_pred_logprobs"],
        mix_ens_vals["ens_pred_batch_idxs"],
    )
    print(mix_ens_cos_hun.mean().item())

    print(">> mix ensemble oos")
    ens_pred_oos_prob = mix_ens_vals["ens_pred_oos_prob"]
    ens_true_oos_prob = mix_ens_vals["ens_true_oos_prob"]
    ens_cos_hun = mix_ens_cos_hun.clone()
    corr_pred_oos_prob_true_oos_prob = pearson_r(ens_pred_oos_prob, ens_true_oos_prob).item()
    corr_pred_oos_prob_cos_hun = pearson_r(ens_pred_oos_prob, ens_cos_hun).item()
    corr_true_oos_prob_cos_hun = pearson_r(ens_true_oos_prob, ens_cos_hun).item()
    print("> corr pred_oos_prob vs true_oos_prob", corr_pred_oos_prob_true_oos_prob)
    print("> corr pred_oos_prob vs cos_hun", corr_pred_oos_prob_cos_hun)
    print("> corr true_oos_prob vs cos_hun", corr_true_oos_prob_cos_hun)    
    print("> corr pred_oos_prob vs cos_hun (thresholds)")
    for i in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        print(
            i, 
            (ens_pred_oos_prob >= i).float().mean().item(), 
            ens_cos_hun[ens_pred_oos_prob >= i].mean().item(), 
            ens_cos_hun[ens_pred_oos_prob < i].mean().item()
        )
    # update stats_d
    stats_d["corr_pred_oos_prob_true_oos_prob"] = corr_pred_oos_prob_true_oos_prob
    stats_d["corr_pred_oos_prob_cos_hun"] = corr_pred_oos_prob_cos_hun
    stats_d["corr_true_oos_prob_cos_hun"] = corr_true_oos_prob_cos_hun

    print(">> unequal max")

    ens_formula_batch_idxs = mix_ens_vals["ens_formula_batch_idxs"].clone()
    ens_formula_logprobs = mix_ens_vals["ens_formula_logprobs"].clone()
    ens_cos_hun = mix_ens_cos_hun.clone()
    ens_formula_logprobs[equal_all] = LOG_ZERO(th.float32)

    unequal_max_logprobs = scatter_reduce(
        ens_formula_logprobs,
        ens_formula_batch_idxs,
        dim_size=th.max(ens_formula_batch_idxs)+1,
        reduce="amax",
        default=LOG_ZERO(th.float32)
    )

    print(unequal_max_logprobs.shape)
    print(
        unequal_max_logprobs.exp().mean().item(), 
        unequal_max_logprobs.exp().min().item(), 
        unequal_max_logprobs.exp().max().item()
    )
    print(pearson_r(unequal_max_logprobs.exp(), ens_cos_hun).item())
    for i in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        print(
            i, 
            (unequal_max_logprobs.exp() >= i).float().mean().item(), 
            ens_cos_hun[unequal_max_logprobs.exp() >= i].mean().item(), 
            ens_cos_hun[unequal_max_logprobs.exp() < i].mean().item()
        )

    print(">> unequal mean")

    ens_formula_batch_idxs = mix_ens_vals["ens_formula_batch_idxs"].clone()
    ens_formula_logprobs = mix_ens_vals["ens_formula_logprobs"].clone()
    ens_cos_hun = mix_ens_cos_hun.clone()
    ens_formula_logprobs[equal_all] = LOG_ZERO(th.float32)

    unequal_mean_logprobs = scatter_logsumexp(
        ens_formula_logprobs,
        ens_formula_batch_idxs,
        dim_size=th.max(ens_formula_batch_idxs)+1
    )
    # unequal_mean_counts = scatter_reduce(
    #     (ens_formula_logprobs > LOG_ZERO(th.float32)).float(),
    #     ens_formula_batch_idxs,
    #     dim_size=th.max(ens_formula_batch_idxs)+1,
    #     reduce="sum"
    # )
    # unequal_mean_logprobs = unequal_mean_logprobs - safelog(unequal_mean_counts.float())

    print(unequal_mean_logprobs.shape)
    print(
        unequal_mean_logprobs.exp().mean().item(), 
        unequal_mean_logprobs.exp().min().item(), 
        unequal_mean_logprobs.exp().max().item()
    )
    print(pearson_r(unequal_mean_logprobs.exp(), ens_cos_hun).item())
    for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        print(
            i, 
            (unequal_mean_logprobs.exp() >= i).float().mean().item(), 
            ens_cos_hun[unequal_mean_logprobs.exp() >= i].mean().item(), 
            ens_cos_hun[unequal_mean_logprobs.exp() < i].mean().item(),
        )

    os.makedirs(args.stats_dp, exist_ok=True)
    stats_fp = os.path.join(args.stats_dp, f"ensemble_agreement_stats" + nb_flag + conds_flag + ".pkl")
    th.save(stats_d, stats_fp)
    print(f">> saved stats to {stats_fp}")