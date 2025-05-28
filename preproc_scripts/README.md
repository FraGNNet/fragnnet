# Data Preprocessing

## Step 0: Exporting the NIST 2020 MS/MS library

All of the data used in the paper is based on the NIST 2020 MS/MS library. Since this library is not publicly available, we are unfortunately unable to provide access to it.

The raw NIST data can be exported using the instructions [here](https://github.com/Roestlab/massformer). It can then be processed using the scripts in the [preroc_scripts directory](preproc_scripts/) (see [this README](preproc_scripts/README.md) for more details).

## Step 1: Processing raw data into CSV

This assumes that the NIST20 data is in a folder `data/raw/nist_20`, containing two files: `hr_nist_msms.MSP` (the MSP file with the spectrum information) and `hr_nist_msms.MOL` (the MOL file with the molecule structures).

This will create a new directory `data/df`

```bash
python preproc_scripts/01_prepare_df.py --msp_file nist_20/hr_nist_msms.MSP --mol_dir nist_20/hr_nist_msms.MOL --input_format msp+mol --output_format csv --output_dp data/df --output_name nist20_hr
```

## Step 2: Processing CSV into Pickle Files

This step converts the data into binary format (pickle) for training and evaluation.

`spec_df.pkl` contains spectrum information.
`mol_df.pkl` contains molecule information.
`ann_df.pkl` contains annotation information.

```bash
python preproc_scripts/02_prepare_proc.py --df_dp data/df --dsets nist20_hr --proc_dp data/proc/nist20
```

## Step 3: Generating Fragmentation DAGs

This step generates fragmentation DAGs that are used for the FraGNNet models. Since fragmentation can be slow, we recommend using a compute with many cores (by default, the script will use all available cores). 

Frag configuration: depth 3 (used for FraGNNet-D3), isomorphism check with no bond information (nb, 3 WL iterations)

```bash
python preproc_scripts/03_prepare_dag_feats.py --max_depth 3 --frag_dp data/frag/nist20_d3 --proc_dp data/proc/nist20 --prec_types "[M+H]+" --inst_types FT --frag_modes HCD --ion_modes P --wandb_mode disabled
```

Frag configuration: depth 4 (used for FraGNNet-D4), isomorphism check with no bond information (nb, 3 WL iterations)

```bash
python preproc_scripts/03_prepare_dag_feats.py --max_depth 4 --frag_dp data/frag/nist20_d4 --proc_dp data/proc/nist20 --prec_types "[M+H]+" --inst_types FT --frag_modes HCD --ion_modes P --wandb_mode disabled
```

## Step 4: Preparing Splits

This step splits the data into training, validation, and test sets.

Inchikey split:

```bash
python preproc_scripts/04_prepare_split.py --split_key inchikey_s --primary_dsets nist20_hr --prec_types "[M+H]+" --inst_types FT --frag_modes HCD --ion_modes P --proc_dp data/proc/nist20 --frag_dp data/frag/nist20_d4 --split_dp data/split/nist20_inchikey
```

Scaffold split:

```bash
python preproc_scripts/04_prepare_split.py --split_key scaffold --primary_dsets nist20_hr --prec_types "[M+H]+" --inst_types FT --frag_modes HCD --ion_modes P --proc_dp data/proc/nist20 --frag_dp data/frag/nist20_d4 --split_dp data/split/nist20_scaffold
```

## Step 5: Preparing Optimal MAGMa DAGs (ICEBERG)

This step generates optimal MAGMa DAGs for training/evaluating the ICEBERG generator model optimal ICEBERG intensity model.

```bash
python preproc_scripts/05_prepare_magma_feats.py --magma_dp data/magma/gen/nist20mona23v3 --proc_dp data/proc/nist20mona23v3 --dsets nist20_hr mona23
```

## Step 6: Preparing Approximate MAGMa DAGs (ICEBERG)

This step generates approximate MAGMa DAGs using a trained ICEBERG generator model, for training and evaluating the ICEBERG intensity model.

For each seed `${SEED}` (see [config README](../config/README.md)) repeat the following:

```bash
python preproc_scripts/06_predict_magma_dags.py --magma_dp data/magma/inten/nist20_inchikey_s$SEED --proc_dp data/proc/nist20 --gen_ckpt_fp data/magma/ckpt/nist20_inchikey_s$SEED.ckpt --dsets nist20_hr
```

Don't forget to symlink the formula directory:

```bash
ln -s $HOME/frag-gnn/data/magma/gen/nist20/magma_formula $HOME/frag-gnn/data/magma/inten/nist20_inchikey_s$SEED/magma_formula
```
