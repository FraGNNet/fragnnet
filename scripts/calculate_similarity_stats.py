from fragnnet.utils.script_utils import pearson_r

import numpy as np
import torch as th
import pandas as pd
import argparse
import os


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--split_types", type=str, nargs="+", default=["inchikey","scaffold"])
	parser.add_argument("--model_types", type=str, nargs="+", default=[
		"fragnnet_d3",
		"fragnnet_d4",
		"iceberg_inten",
		"iceberg_inten_opt",
		"neims",
		"massformer",
		"graff",
		"precursor"
	])
	parser.add_argument("--eval_split", type=str, default="test", choices=["train","val","test","secondary"])
	parser.add_argument("--num_seeds", type=int, default=5)
	parser.add_argument("--vals_dp", type=str, required=True)
	parser.add_argument("--output_dp", type=str, required=True)
	args = parser.parse_args()

	eval_split = args.eval_split
	num_seeds = args.num_seeds
	vals_dp = args.vals_dp
	pd.options.display.float_format = '{:.4f}'.format

	split_types = args.split_types
	model_types = args.model_types

	stats_df = []

	for split_type in split_types:
		for model_type in model_types:
			mean_sims_binned = []
			mean_sims_unbinned = []
			mean_true_oos = []
			mean_pred_oos = []
			mean_corr_oos = []
			mean_tv_oos = []
			for seed in range(num_seeds):
				seed_vals_fp = os.path.join(args.vals_dp,split_type,model_type,f"s{seed}",f"{eval_split}.pkl")
				print(f">>> Loading {seed_vals_fp}")
				if not os.path.isfile(seed_vals_fp):
					print(f">> file not found, skipping")
					continue
				seed_vals = th.load(seed_vals_fp,map_location="cpu")
				sims_binned = seed_vals["cos_sim_0.01"]
				sims_unbinned = seed_vals["cos_hun"]
				true_oos_prob = seed_vals.get("true_oos_prob",th.tensor(np.nan))
				pred_oos_prob = seed_vals.get("oos_prob",th.tensor(np.nan))
				corr_oos = pearson_r(true_oos_prob,pred_oos_prob)
				tv_oos = th.abs(true_oos_prob - pred_oos_prob)
				mean_sims_binned.append(sims_binned.mean())
				mean_sims_unbinned.append(sims_unbinned.mean())
				mean_true_oos.append(true_oos_prob.mean())
				mean_pred_oos.append(pred_oos_prob.mean())
				mean_corr_oos.append(corr_oos)
				mean_tv_oos.append(tv_oos.mean())
			mean_sims_binned = th.stack(mean_sims_binned,dim=0)
			mean_sims_unbinned = th.stack(mean_sims_unbinned,dim=0)
			mean_true_oos = th.stack(mean_true_oos,dim=0)
			mean_pred_oos = th.stack(mean_pred_oos,dim=0)
			mean_corr_oos = th.stack(mean_corr_oos,dim=0)
			mean_tv_oos = th.stack(mean_tv_oos,dim=0)
			stats_df_entry = {
				"split_type": split_type,
				"model_type": model_type,
				"cos_sim_0.01_mean": mean_sims_binned.mean().item(),
				"cos_sim_0.01_std": mean_sims_binned.std().item(),
				"cos_hun_mean": mean_sims_unbinned.mean().item(),
				"cos_hun_std": mean_sims_unbinned.std().item(),
				"true_oos_prob_mean": mean_true_oos.mean().item(),
				"true_oos_prob_std": mean_true_oos.std().item(),
				"pred_oos_prob_mean": mean_pred_oos.mean().item(),
				"pred_oos_prob_std": mean_pred_oos.std().item(),
				"pearson_r_oos_mean": mean_corr_oos.mean().item(),
				"pearson_r_oos_std": mean_corr_oos.std().item(),
				"tv_oos_mean": mean_tv_oos.mean().item(),
				"tv_oos_std": mean_tv_oos.std().item(),
			}
			stats_df.append(stats_df_entry)
	
	stats_df = pd.DataFrame(stats_df)
	os.makedirs(args.output_dp,exist_ok=True)
	stats_df.to_csv(os.path.join(args.output_dp,f"{eval_split}_df.csv"),index=False)
	print(stats_df)
