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

### Set up Conda Environment

Install miniconda from [here](https://docs.conda.io/en/latest/miniconda.html).

Create a new conda environment called `FRAGNNET-GPU` with python 3.10:

```
conda create -n FRAGNNET-GPU python=3.10
```

### Install FraGNNet Python Package

```bash
pip install .
```

If you want dev features (wandb logging and jupyter notebook), you can install the package with the dev flag.

```bash
pip install -e .[dev]
```

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
