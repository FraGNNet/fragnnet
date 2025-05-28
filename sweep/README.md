## Weights and Biases Sweeps

- [fragnnet_d3_sweep.yml](./fragnnet_d3_sweep.yml): sweep for FraGNNet-D3
- [fragnnet_d4_sweep.yml](./fragnnet_d4_sweep.yml): sweep for FraGNNet-D4
- [graff_sweep.yml](./graff_sweep.yml): sweep for GraFF
- [iceberg_gen_sweep.yml](./iceberg_gen_sweep.yml): sweep for Iceberg Gen
- [iceberg_inten_sweep.yml](./iceberg_inten_sweep.yml): sweep for Iceberg Inten
- [iceberg_inten_opt_sweep.yml](./iceberg_inten_opt_sweep.yml): sweep for Iceberg Inten (+OPT)
- [neims_sweep.yml](./neims_sweep.yml): sweep for NeIMS
- [massformer_sweep.yml](./massformer_sweep.yml): sweep for massformer

Each wandb sweep used a budget of 100 samples and a random search strategy. The sweeps were initialized using the following command:

```bash
wandb sweep --project fragnnet sweep/${MODEL}_sweep.yml
```

