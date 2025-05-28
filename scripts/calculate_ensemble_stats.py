import torch as th
import argparse	
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dp", type=str, default="inference_outputs/entropy/")
    parser.add_argument("--stats_dp", type=str, default="figs/ensemble/")
    parser.add_argument("--split_types", type=str, nargs="+", default=["scaffold"])
    parser.add_argument("--depths", type=str, nargs="+", default=["d4"], choices=["d3","d4"])
    parser.add_argument("--ens_types", type=str, nargs="+", default=["baseline", "low", "high", "mix"])
    args = parser.parse_args()

    stats_d = {}
    ens_type_to_key = {
        "baseline": "baseline",
        "low": "posfn",
        "high": "negfn",
        "mix": "mix"
    }

    assert os.path.isdir(args.output_dp), f"ensemble output_dp {args.output_dp} not found"
    
    for split_type in args.split_types:
        
        for depth in args.depths:
    
            for ens_type in args.ens_types:
                
                ens_type_key = ens_type_to_key[ens_type]
                ens_output_dp = os.path.join(args.output_dp,f"{split_type}_{depth}_{ens_type_key}_test_False")
                ens_stats_d = {}

                sims_fp = os.path.join(ens_output_dp,"sims.pkl")
                entropies_fp = os.path.join(ens_output_dp,"entropies.pkl")

                assert os.path.isfile(sims_fp), f"sims_fp {sims_fp} not found"
                sims_d = th.load(sims_fp)
                ind_cos_sim_bin = sims_d["ind_cos_sim_bin"]
                ind_cos_hun = sims_d["ind_cos_hun"]
                ens_cos_sim_bin = sims_d["ens_cos_sim_bin"]
                ens_cos_hun = sims_d["ens_cos_hun"]
                
                ind_cos_sim_bin_mean = ind_cos_sim_bin.mean().item()
                ind_cos_sim_bin_std = ind_cos_sim_bin.mean(dim=1).std().item()
                ind_cos_hun_mean = ind_cos_hun.mean().item()
                ind_cos_hun_std = ind_cos_hun.mean(dim=1).std().item()
                ens_cos_sim_bin_mean = ens_cos_sim_bin.mean().item()
                ens_cos_hun_mean = ens_cos_hun.mean().item()

                print(">> average individual")
                print(f"> cos_sim_0.01: {ind_cos_sim_bin_mean} +/- {ind_cos_sim_bin_std}")
                print(f"> cos_hun: {ind_cos_hun_mean} +/- {ind_cos_hun_std}")

                print(">> ensemble")
                print(f"> cos_sim_0.01: {ens_cos_sim_bin_mean}")
                print(f"> cos_hun: {ens_cos_hun_mean}")

                ens_stats_d["ind_cos_sim_bin_mean"] = ind_cos_sim_bin_mean
                ens_stats_d["ind_cos_sim_bin_std"] = ind_cos_sim_bin_std
                ens_stats_d["ind_cos_hun_mean"] = ind_cos_hun_mean
                ens_stats_d["ind_cos_hun_std"] = ind_cos_hun_std
                ens_stats_d["ens_cos_sim_bin_mean"] = ens_cos_sim_bin_mean
                ens_stats_d["ens_cos_hun_mean"] = ens_cos_hun_mean

                print(">>> Entropy")

                assert os.path.isfile(entropies_fp), f"entropies_fp {entropies_fp} not found"

                entropies_d = th.load(entropies_fp)
                ind_formula_node_e = entropies_d["ind_formula_node_e"]
                ind_formula_node_ne = entropies_d["ind_formula_node_ne"]
                ind_nb_formula_node_e = entropies_d["ind_nb_formula_node_e"]
                ind_nb_formula_node_ne = entropies_d["ind_nb_formula_node_ne"]
                ens_formula_node_e = entropies_d["ens_formula_node_e"]
                ens_formula_node_ne = entropies_d["ens_formula_node_ne"]
                ens_nb_formula_node_e = entropies_d["ens_nb_formula_node_e"]
                ens_nb_formula_node_ne = entropies_d["ens_nb_formula_node_ne"]
                
                ind_formula_node_ne_mean = ind_formula_node_ne.mean().item()
                ind_formula_node_ne_std = ind_formula_node_ne.mean(dim=1).std().item()
                ind_nb_formula_node_ne_mean = ind_nb_formula_node_ne.mean().item()
                ind_nb_formula_node_ne_std = ind_nb_formula_node_ne.mean(dim=1).std().item()
                ens_formula_node_ne_mean = ens_formula_node_ne.mean().item()
                ens_nb_formula_node_ne_mean = ens_nb_formula_node_ne.mean().item()

                print(">> average individual")
                # print(f"> formula_node_e: {ind_formula_node_e.mean()} +/- {ind_formula_node_e.mean(dim=1).std()}")
                print(f"> formula_node_ne: {ind_formula_node_ne.mean()} +/- {ind_formula_node_ne.mean(dim=1).std()}")
                # print(f"> nb_formula_node_e: {ind_nb_formula_node_e.mean()} +/- {ind_nb_formula_node_e.mean(dim=1).std()}")
                print(f"> nb_formula_node_ne: {ind_nb_formula_node_ne.mean()} +/- {ind_nb_formula_node_ne.mean(dim=1).std()}")

                print(">> ensemble")
                # print(f"> formula_node_e: {ens_formula_node_e.mean()}")
                print(f"> formula_node_ne: {ens_formula_node_ne.mean()}")
                # print(f"> nb_formula_node_e: {ens_nb_formula_node_e.mean()}")
                print(f"> nb_formula_node_ne: {ens_nb_formula_node_ne.mean()}")

                ens_stats_d["ind_formula_node_ne_mean"] = ind_formula_node_ne_mean
                ens_stats_d["ind_formula_node_ne_std"] = ind_formula_node_ne_std
                ens_stats_d["ind_nb_formula_node_ne_mean"] = ind_nb_formula_node_ne_mean
                ens_stats_d["ind_nb_formula_node_ne_std"] = ind_nb_formula_node_ne_std
                ens_stats_d["ens_formula_node_ne_mean"] = ens_formula_node_ne_mean
                ens_stats_d["ens_nb_formula_node_ne_mean"] = ens_nb_formula_node_ne_mean

                stats_d[f"{split_type}_{depth}_{ens_type}"] = ens_stats_d

    os.makedirs(args.stats_dp,exist_ok=True)
    stats_fp = os.path.join(args.stats_dp,"ensemble_stats.pkl")
    th.save(stats_d,stats_fp)
    print(f">> saved stats to {stats_fp}")
