# Experiment Configs

## Setting up a Training Run

To train a model, you can use the following command:

```bash
python scripts/run_pl_model_fit.py -c config/${DATASET}_${SPLIT}/${MODEL}/s${SEED}.yml
```

For example, to train a model on the NIST 2020 dataset using the InChIKey split and the NEIMS model with seed 0, you can use the following command:

```bash
python scripts/run_pl_model_fit.py -c config/nist20_inchikey/neims/s0.yml
```

If you installed the package with Weights and Biases (i.e. using the `dev` option), you can enable logging with a flag:

```bash
python scripts/run_pl_model_fit.py -c config/nist20_inchikey/neims/s0.yml -w online
```

## Available Configs

All provided configs use the NIST 2020 dataset.

### Main Experiment Configs

The following seeds are used:
- SEED=0: 397623
- SEED=1: 634340
- SEED=2: 444303
- SEED=3: 103976
- SEED=4: 132548

FraGNNet-D3: [inchikey configs](./nist20_inchikey/fragnnet_d3/), [scaffold configs](./nist20_scaffold/fragnnet_d3/)

FraGNNet-D4: [inchikey configs](./nist20_inchikey/fragnnet_d4/), [scaffold configs](./nist20_scaffold/fragnnet_d4/)

NEIMS: [inchikey configs](./nist20_inchikey/neims/), [scaffold configs](./nist20_scaffold/neims/)

MassFormer: [inchikey configs](./nist20_inchikey/massformer/), [scaffold configs](./nist20_scaffold/massformer/)

ICEBERG (Generator model): [inchikey configs](./nist20_inchikey/iceberg_gen/), [scaffold configs](./nist20_scaffold/iceberg_gen/)

ICEBERG (Intensity model): [inchikey configs](./nist20_inchikey/iceberg/), [scaffold configs](./nist20_scaffold/iceberg/)

ICEBERG (Optimal Intensity model): [inchikey configs](./nist20_inchikey/iceberg_opt/), [scaffold configs](./nist20_scaffold/iceberg_opt/)

Note: the ICEBERG Optimal Intensity model is presented in the ablations section in the paper as ICEBERG (+Opt).

### Ablation Experiment Configs

All ablations are performed using the InChIKey split.

The following seeds are used:
- SEED=0: 397623
- SEED=1: 634340
- SEED=2: 444303
- SEED=3: 103976
- SEED=4: 132548

FraGNNet-D3 (+Edges) [configs](./ablations/fraggnn_d3_edges/)
FraGNNet-D3 (-CE) [configs](./ablations/fraggnn_d3_noce/)
FraGNNet-D4 (-CE) [configs](./ablations/fraggnn_d4_noce/)

### Entropy Experiment Configs

All entropy experiments are performed using the Scaffold split, using the FraGNNet-D4 model.

The following seeds are used:
- SEED=0: 397623
- SEED=1: 634340
- SEED=2: 444303
- SEED=3: 103976
- SEED=4: 132548
- SEED=5: 470932
- SEED=6: 499158
- SEED=7: 842766
- SEED=8: 771113
- SEED=9: 742567
- SEED=10: 315874
- SEED=11: 155679
- SEED=12: 885718
- SEED=13: 263526
- SEED=14: 623677


Baseline Ensemble: [configs](./entropy/d4_baseline_ens/)

Low Entropy Ensemble: [configs](./entropy/d4_low_ens/)

High Entropy Ensemble: [configs](./entropy/d4_high_ens/)

