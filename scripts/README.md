## Runnable Scripts

- [run_pl_model_fit.py](./run_pl_model_fit.py): script for training a model (see [config README](../config/README.md) for more details)
- [run_compute_frags.py](./run_compute_frags.py): standalone script for computing the fragmentation DAG for a molecule

## Non-Runnable Scripts

These scripts are not runnable, but are useful for understanding the analysis presented in the paper.

- [run_save_inference.py](./run_save_inference.py): script for downloading a model checkpoint from Weights and Biases and running inference using that checkpoint (used for c2ms_sim_exp and formula_ann_exp)
- [run_save_entropy.py](./run_save_entropy.py): similar to run_save_inference.py, but also performs model ensembling (used for frag_ann_exp)
- [run_save_inference_ablations.py](./run_save_inference_ablations.py): similar to run_save_inference.py (used for ablation_exp)
- [run_wandb_sweep.py](./run_wandb_sweep.py): script called by the sweep agent to run a single configuration in a Weights and Biases sweep (see [sweep/README.md](../sweep/README.md) for more details)
- [calculate_similarity_stats.py](./calculate_similarity_stats.py): script for calculating the similarity statistics from saved inference outputs (used for c2ms_sim_exp)
- [calculate_formula_stats.py](./calculate_formula_stats.py): script for calculating the formula annotation statistics from saved inference outputs (used for formula_ann_exp)
- [calculate_ensemble_stats.py](./calculate_ensemble_stats.py): script for calculating ensemble similarities and entropies from saved inference outputs (used for frag_ann_exp)
- [calculate_ensemble_agreement_stats.py](./calculate_ensemble_agreement_stats.py): script for calculating the ensemble fragment annotation statistics from saved inference outputs (used for frag_ann_exp)
- [calculate_ablation_stats.py](./calculate_ablation_stats.py): similar to calculate_similarity_stats.py (used for ablation_exp)
- [ms2c/*.py](./ms2c/): various scripts for preparing and running MS2C retrieval experiments (used in ms2c_retrieval_exp)