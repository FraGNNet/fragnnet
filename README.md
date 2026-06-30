# FraGNNet: A Deep Probabilistic Model for Mass Spectrum Prediction

## Overview

This repository contains the code and models for the paper **FraGNNet: A Deep Probabilistic Model for Mass Spectrum Prediction**. FraGNNet is a deep probabilistic graph neural network (GNN) model for predicting tandem mass spectra (MS/MS) from molecular structures. FraGNNet predicts MS/MS spectra from molecular structures using a combination of:
- A **recursive bond-breaking algorithm** for fragment generation.
- A **graph neural network (GNN)** for predicting the spectrum from fragments.

This method provides:
- High mass accuracy.
- Scalability for large compound libraries.
- Interpretability through latent variable distributions (formula and fragment annotations).

---

## Project Structure

**Experiment configs**: see [config/README.md](config/README.md)

**Data preprocessing**: see [preproc_scripts/README.md](preproc_scripts/README.md)

**Training and Analysis scripts**: see [scripts/README.md](scripts/README.md)

**Analysis notebooks**: see [notebooks/README.md](notebooks/README.md)

If you are having trouble preprocessing the data or reproducing the results, please send an email to ayoung [AT] cs [DOT] toronto [DOT] edu.

## Installation

**For detailed installation instructions, see [INSTALL.md](INSTALL.md)**

### Quick Start (Recommended)

```bash
# Using conda (GPU, recommended)
bash env/env_setup_gpu_py311_cu121_250602.sh
conda activate FRAGNNET-GPU
```

For development with testing tools:
```bash
pip install -e ".[dev]"
```

### Alternative: Manual Pip Installation

```bash
# Create virtual environment and install PyTorch
python -m venv venv
source venv/bin/activate
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install fragnnet and dev dependencies
pip install -e ".[dev]"
```

See [INSTALL.md](INSTALL.md) for detailed options, troubleshooting, and HPC cluster setup.

Note that the -e flag is used to install the package in editable mode.

### Install PyTorch-related Dependencies using Script

Use the following script to install PyTorch-related dependencies:

```bash
python install_lib.py
```

The scripts attempts to detect your CUDA version and install the appropriate packages. There are two supported CUDA versions (11.8 and 12.1). You can also manually specify the CUDA version using `--force-cuda XX.x` or perform a CPU-only installation with `--force-cpu`. Other CUDA versions (i.e. CUDA 12.x) may also install correctly but were not tested. All experiments in the paper were run using CUDA 11.8.

## Contributors

- Adamo Young ([adamoyoung](https://github.com/adamoyoung))
- Fei Wang ([7FeiW](https://github.com/7FeiW))

## Citation

````bibtex
@article{fragnnet,
  title={FraGNNet: A Deep Probabilistic Model for Mass Spectrum Prediction},
  author={Adamo Young, Fei Wang, David Wishart, Bo Wang, Hannes Röst, Russ Greiner},
  year={2024},
}
````

  author={Anonymous},
  year={2025},
}

## Dynamic batch sampler and formula-based hard negatives

This project supports dynamic batching to avoid OOM and a formula-based hard-negative sampler for contrastive/pairwise training.

- `formula_hard_negative_dynamic_sampler` (bool): when True the runner will attempt to use the combined formula-based hard-negative dynamic batch sampler during training. See the runner integration at [src/fragnnet/runner.py](src/fragnnet/runner.py).
- `dynamic_batch_sampler_mode` (string): either `frag_node` or `frag_edge`. Passed as `limited_by` to the dynamic batch samplers and selects whether batching budgets are computed from `frag_pyg.num_nodes` (`frag_node`) or `frag_pyg.num_edges` (`frag_edge`). Implementations live in [src/fragnnet/dataset/data_sampler.py](src/fragnnet/dataset/data_sampler.py).
- `dynamic_batch_sampler_max` (int): maximum number of nodes/edges per batch used by dynamic samplers.
- `formula_hard_negative_min_specs_per_formula` (int): minimum spectra per formula required for formula-based grouping; formulas with fewer spectra are skipped.

Notes:
- `FormulaHardNegativeDynamicBatchSampler` groups spectra by molecular formula then builds dynamic batches that respect the chosen `limited_by` metric (nodes or edges). See `FormulaHardNegativeDynamicBatchSampler` and `SpecMolFragDynamicBatchSampler` in [src/fragnnet/dataset/data_sampler.py](src/fragnnet/dataset/data_sampler.py).
- The runner enforces valid modes (`frag_node` or `frag_edge`). See the config checks in [src/fragnnet/runner.py](src/fragnnet/runner.py).

````