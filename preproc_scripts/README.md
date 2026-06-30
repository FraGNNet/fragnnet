# Data Preprocessing

## Step 0: Exporting the NIST 2020 MS/MS library

All of the data used in the paper is based on the NIST 2020 MS/MS library. Since this library is not publicly available, we are unfortunately unable to provide access to it.

The raw NIST data can be exported using the instructions [here](https://github.com/Roestlab/massformer).

## Step 1: Processing raw data into CSV

This step Writes `<output_name>_df.csv` (or `.json`) in `data/df/` by default. Each row is one spectrum. Columns are defined here:
- `spec_id`: integer id within this exported file.
- `dset`: dataset/source name.
- `dset_spec_id`: source-specific spectrum identifier.
- `peaks`: multiline string, one line per peak as `mz intensity`.
- `prec_mz`: precursor m/z (float).
- `prec_type`: precursor/adduct string (e.g., `[M+H]+`).
- `spec_type`: spectrum type/level (e.g., `MS2`).
- `ion_mode`: `P`/`N` when known.
- `ion_type`: ionization type (e.g., `ESI`, `EI`) when known.
- `inst_type` / `inst`: instrument metadata when present.
- `col_energy`, `col_energy_extra_1`, `col_energy_extra_2`: collision energies (strings with units).
- `frag_mode`: fragmentation mode (e.g., `CID`, `HCD`, `EI`) when present.
- `num_peaks`: peak count if provided by source.
- `exact_mass` / `mw`: exact or molecular weight when available.
- `formula` / `smiles` / `inchi` / `inchikey`: structure metadata if present.
- `name` / `title`: compound name or title when present.
- `ramped` / `stepped` / `normalized`: flags parsed from collision energy fields when applicable.
- Misc placeholders: `notes`, `rating`, `cas_num`, `pressure`, `ri`, `ionization` extras, etc., kept for upstream completeness.

Some input formats also emit helper split files (e.g., `*_fold.csv` for MS-Gym or Spectraverse) containing only ids needed for downstream splitting.

### NIST 20
This assumes that the NIST20 data is in a folder `data/raw/nist_20`, containing two files: `hr_nist_msms.MSP` (the MSP file with the spectrum information) and `hr_nist_msms.MOL` (the MOL file with the molecule structures).

This will create a new directory `data/df`

```bash
python preproc_scripts/01_prepare_df.py --msp_file nist_20/hr_nist_msms.MSP --mol_dir nist_20/hr_nist_msms.MOL --input_format msp+mol --output_format csv --output_dp data/df --output_name nist20_hr
```

### MassSpecGym
This assumes that the Spectraverse data is in `data/raw/msgym/MassSpecGym_with_test.tsv`, this will use `ms_gym`,`ms_gym_extra` as dataset name for each part of MassSpecGym

```bash
# for sim challenege
python preproc_scripts/01_prepare_df.py --input_format=ms_gym --output_format=csv --output_name=ms_gym  --ms_gym_tsv=msgym/MassSpecGym_with_test.tsv

# for extra data in msgym
python preproc_scripts/01_prepare_df.py --input_format=ms_gym_extra --output_format=csv --output_name=ms_gym_extra --ms_gym_tsv=msgym/MassSpecGym_with_test.tsv
```

### Spectraverse
This assumes that the Spectraverse data is in `data/raw/spectraverse/spectraverse-0.0.1.mgf`

```bash
python preproc_scripts/01_prepare_df.py \
	--input_format=spectraverse \
    --output_format=csv \
    --output_name=spectraverse \
    --mgf_file=spectraverse/spectraverse-0.0.1.mgf
```

## Step 2: Processing CSV into Pickle Files

This step  reads the stage-1 CSVs, converts the data into binary format (pickle), writes pickles to `proc_dp` for training and evaluation.

- `spec_df.pkl` — contains spectrum information, one row per spectrum:
	- `spec_id`: integer id unique within the processed set.
	- `mol_id`: integer id linking to `mol_df`.
	- `prec_type`: standardized precursor/adduct string.
	- `inst_type`: standardized instrument type.
	- `frag_mode`: standardized fragmentation mode.
	- `spec_type`: spectrum type (e.g., `MS2`).
	- `ion_mode`: `P`/`N`.
	- `dset`: dataset name.
	- `dset_spec_id`: source-specific spectrum id.
	- `col_gas`: collision gas (may be NaN).
	- `res`: inferred m/z resolution (currently informational).
	- `ace`, `ace_extra_1`, `ace_extra_2`: parsed absolute collision energies (first/second/third for ramped or stepped).
	- `nce`, `nce_extra_1`, `nce_extra_2`: parsed normalized collision energies (first/second/third for ramped or stepped).
	- `has_ace`, `has_nce`: booleans for CE presence.
	- `prec_mz`: precursor m/z (float; inferred when missing using molecular mass).
	- `peaks`: list of `(mz, intensity)` float pairs.
	- `ri`: retention index if present.
	- `formula`: molecular formula from source (dropped before saving when inconsistent with molecule formula).
	- `inchikey`: full InChIKey from source (dropped before saving when inconsistent).
	- `exact_mass`: exact mass from source (dropped before saving when inconsistent).
	- `group_id`: grouping id for spectra sharing compound/precursor context.

- `mol_df.pkl` — contains molecule information, one row per unique canonical SMILES:
	- `smiles`, `mol_id`, `mol` (RDKit Mol object).
	- `inchikey_s`: first 14 chars of InChIKey.
	- `scaffold`: Murcko scaffold SMILES.
	- `formula`, `inchi`.
	- `mw`, `exact_mw`: average and exact molecular weights.
	- `num_atoms`, `num_bonds`, `num_radicals`.
	- `charge`, `single_mol` (connected component flag).

- `ann_df.pkl` — contains annotation information, one row per annotated spectrum (only when annotations parsed):
	- `dset_spec_id`, `spec_id`, `mol_id`, `prec_type`, `formula`.
	- `ann_peak_mzs`: list of annotated peak m/z values.
	- `ann_products`: product formula strings.
	- `ann_losses`: neutral loss formula strings.
	- `ann_isotopes`: isotope annotations.
	- `ann_exact_mzs`: exact m/z values for annotations.

- Debug pickles: `no_mol_df.pkl`, `diff_formula_df.pkl`, `diff_inchikey_df.pkl`, `diff_mass_df.pkl` record rows dropped or with inconsistencies.


### NIST 20
```bash
python preproc_scripts/02_prepare_proc.py \
    --df_dp data/df \
    --dsets nist20_hr \
    --proc_dp data/proc/nist20
```

### MSGYM
```bash
# for all data in msgym
python preproc_scripts/02_prepare_proc.py \
    --proc_dp=data/proc/ms_gym_all \
    --dsets ms_gym ms_gym_extra
```
### Spectraverse
```bash
python preproc_scripts/02_prepare_proc.py \
	--proc_dp=data/proc/spectraverse \
    --dsets spectraverse
```
## Step 3: Generating Fragmentation DAGs

### NIST 20
This step generates fragmentation DAGs that are used for the FraGNNet models. Since fragmentation can be slow, we recommend using a compute with many cores (by default, the script will use all available cores). 

Frag configuration: depth 3 (used for FraGNNet-D3), isomorphism check with no bond information (nb, 3 WL iterations)

```bash
python preproc_scripts/03_prepare_dag_feats.py --max_depth 3 --frag_dp data/frag/nist20_d3 --proc_dp data/proc/nist20 --prec_types "[M+H]+" --inst_types FT --frag_modes HCD --ion_modes P --wandb_mode disabled
```

Frag configuration: depth 4 (used for FraGNNet-D4), isomorphism check with no bond information (nb, 3 WL iterations)

```bash
python preproc_scripts/03_prepare_dag_feats.py --max_depth 4 --frag_dp data/frag/nist20_d4 --proc_dp data/proc/nist20 --prec_types "[M+H]+" --inst_types FT --frag_modes HCD --ion_modes P --dsets nist20_hr--wandb_mode disabled
```

### MassSpecGym

```bash
# Benchmark split (sim chanllenge)
python preproc_scripts/04_prepare_split.py \
  --proc_dp=data/proc/ms_gym_all \
  --ces ace \
  --split_dp=./data/split/ms_gym_d3_h4_benchmarkv2 \
  --frag_dp=./data/frag/ms_gym_d3_h4v2 \
  --primary_dsets ms_gym \
  --max_prec_mz=1000.0 \
  --inst_types FT QTOF \
  --split_type=predefined_dsetid_csv \
  --predefined_dsetid_csv=data/df/ms_gym_fold.csv
```

### Spectraverse

```bash
# Benchmark split (sim chanllenge)
python ./preproc_scripts/04_prepare_split.py \
  	--split_type predefined \
    --predefined_id_type spec_id \
    --predefined_test_id_fp data/df/spectraverse_fold_inchi_1.csv \
    --primary_dsets spectraverse \
    --split_dp data/split/spectraverse_nemis_inchikey_cv1 \
    --proc_dp data/proc/spectraverse \
    --dag_filtering False 
```

## Step 4: Preparing Splits

This step splits the data into training, validation, and test sets. To exactly reproduce the splits used in the paper, download the split id files ([inchikey](), [scaffold]()) and place them in `data/split/nist20_inchikey/` and `data/split/nist20_scaffold/` respectively. Then, run the following scripts to remap the ids to work with the processed dataset from the previous step:

Inchikey split:

```bash
python preproc_scripts/04_prepare_split.py --split_type predefined --primary_dsets nist20_hr --prec_types "[M+H]+" --inst_types FT --frag_modes HCD --ion_modes P --dag_filtering False --proc_dp data/proc/nist20 --predefined_id_type dset_spec_id --predefined_train_id_fp data/split/nist20_inchikey/train_dset_ids.csv --predefined_val_id_fp data/split/nist20_inchikey/val_dset_ids.csv --predefined_test_id_fp data/split/nist20_inchikey/test_dset_ids.csv --split_dp data/split/nist20_inchikey
```

Scaffold split:

```bash
python preproc_scripts/04_prepare_split.py --split_type predefined --primary_dsets nist20_hr --prec_types "[M+H]+" --inst_types FT --frag_modes HCD --ion_modes P --dag_filtering False --proc_dp data/proc/nist20 --predefined_id_type dset_spec_id --predefined_train_id_fp data/split/nist20_scaffold/train_dset_ids.csv --predefined_val_id_fp data/split/nist20_scaffold/val_dset_ids.csv --predefined_test_id_fp data/split/nist20_scaffold/test_dset_ids.csv --split_dp data/split/nist20_scaffold
```

If you don't want to use the exact splits that we used (just ones with similar statistics), you can use the same script to generate random splits:

Inchikey split:

```bash
python preproc_scripts/04_prepare_split.py --split_type random --split_key inchikey_s --primary_dsets nist20_hr --prec_types "[M+H]+" --inst_types FT --frag_modes HCD --ion_modes P --proc_dp data/proc/nist20 --frag_dp data/frag/nist20_d4 --split_dp data/split/nist20_inchikey
```

Scaffold split:

```bash
python preproc_scripts/04_prepare_split.py --split_type random --split_key scaffold --primary_dsets nist20_hr --prec_types "[M+H]+" --inst_types FT --frag_modes HCD --ion_modes P --proc_dp data/proc/nist20 --frag_dp data/frag/nist20_d4 --split_dp data/split/nist20_scaffold
```

## Step 5: Preparing Optimal MAGMa DAGs (ICEBERG)

This step generates optimal MAGMa DAGs for training/evaluating the ICEBERG generator model optimal ICEBERG intensity model.

```bash
python preproc_scripts/05_prepare_magma_feats.py --magma_dp data/magma/gen/nist20 --proc_dp data/proc/nist20 --dsets nist20_hr
```

## Step 6: Preparing Approximate MAGMa DAGs (ICEBERG)

This step generates approximate MAGMa DAGs using a trained ICEBERG generator model, for training and evaluating the ICEBERG intensity model.

For each seed `${SEED}` (see [config README](../config/README.md)) repeat the following:

```bash
python preproc_scripts/06_predict_magma_dags.py --magma_dp data/magma/inten/nist20_inchikey_s$SEED --proc_dp data/proc/nist20 --gen_ckpt_fp data/magma/ckpt/nist20_inchikey_s$SEED.ckpt --dsets nist20_hr
```

Don't forget to symlink the formula directory:

```bash
ln -s $HOME/fragnnet/data/magma/gen/nist20/magma_formula $HOME/fragnnet/data/magma/inten/nist20_inchikey_s$SEED/magma_formula
```

## Step 8: Preparing ClassyFire Labels

This step merges ClassyFire chemical taxonomy annotations onto the molecule dataframe via SMILES canonicalization. The output is a pickle keyed by `mol_id` with columns: `kingdom`, `superklass`, `klass`, `subklass`, `direct_parent`, `alternative_parents`, `substituents`. Run once per dataset.

### NIST 20
```bash
python preproc_scripts/08_prepare_classyfire.py \
  --mol_df data/proc/nist20/mol_df.pkl \
  --cf_zip data/classyfire/20231201_nist20mona23_1.zip \
  --output data/classyfire/nist20_mol_classyfire.pkl
```

### NIST 23
```bash
python preproc_scripts/08_prepare_classyfire.py \
  --mol_df data/proc/nist23/mol_df.pkl \
  --cf_zip data/classyfire/20231201_nist23mona23_1.zip \
  --output data/classyfire/nist23_mol_classyfire.pkl
```
