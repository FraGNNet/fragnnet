#!/bin/bash

set -e


source ~/.bashrc

conda create -n FRAG-GNN-GPU
conda activate FRAG-GNN-GPU

conda install python=3.10

pip install torch==2.1.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.26.4
pip install rdkit-pypi==2022.9.4
# don't actually need pyg_lib, torch_sparse, torch_cluster, torch_spline_conv
pip install torch_geometric==2.4 torch_scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install cython
pip install cysignals
pip install pandas
pip install networkx
pip install wandb
pip install matplotlib
pip install pytest
pip install CairoSVG
pip install lightning==2.1.2
pip install dgl==1.0.4 -f https://data.dgl.ai/wheels/cu118/repo.html
pip install omegaconf
pip install pyteomics
pip install notebook

pip install -I -e .