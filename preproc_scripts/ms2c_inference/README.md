# MS2C Inference Preprocessing Scripts

This directory contains scripts for preparing MS2C (Mass Spectrum to Candidate) inference data.

## Pipeline Overview

```
JSON candidates file    spec_df.pkl    mol_df.pkl
        │                   │              │
        └───────────────────┼──────────────┘
                            │
                            ▼
            01_ms2c_inf_create_candidate_df.py
                            │
                            ▼
                    candidate_df.pkl.gz
                            │
                            ▼
            02_ms2c_inf_candidate_to_proc.py
                            │
                            ▼
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
  spec_df_{id}.pkl.gz  mol_df_{id}.pkl.gz  query_index.csv
```

## Scripts

### 01_ms2c_inf_create_candidate_df.py

Creates a `candidate_df` from a JSON file mapping query SMILES to candidate SMILES lists.

**Usage:**
```bash
python preproc_scripts/ms2c_inference/01_ms2c_inf_create_candidate_df.py \
    --json_file data/ms2c/nps/nps_candidates/nps_nist23/nps_nist23_clm_formula.json \
    --spec_df_path data/proc/nps_nist23/spec_df.pkl \
    --mol_df_path data/proc/nps_nist23/mol_df.pkl \
    --output_path data/ms2c/candidates/nps_nist23_candidate_df.pkl.gz
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `--json_file` | Path to JSON file with query SMILES → candidate SMILES lists |
| `--spec_df_path` | Path to spec_df.pkl from data preprocessing |
| `--mol_df_path` | Path to mol_df.pkl from data preprocessing |
| `--output_path` | Output path for candidate_df.pkl.gz |

**Input JSON Format:**
```json
{
  "CCO": ["CCO", "OCC", "C(C)O"],
  "c1ccccc1": ["c1ccccc1", "C1=CC=CC=C1"]
}
```

### 02_ms2c_inf_candidate_to_proc.py

Converts `candidate_df` into per-query spec_df and mol_df files for inference.

**Usage:**
```bash
python preproc_scripts/ms2c_inference/02_ms2c_inf_candidate_to_proc.py \
    --candidate_df_path data/ms2c/candidates/nps_nist23_candidate_df.pkl.gz \
    --output_dir data/ms2c/proc/nps_nist23 \
    --limit 100
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `--candidate_df_path` | Path to candidate_df.pkl.gz from step 01 |
| `--output_dir` | Output directory for per-query files |
| `--limit` | (Optional) Limit number of candidates to process |

**Output Files:**
- `spec_df_{spec_id}.pkl.gz` - Per-query spectrum DataFrame
- `mol_df_{mol_id}.pkl.gz` - Per-query molecule DataFrame (shared across same mol_id)
- `query_index.csv` - Index mapping query IDs to file paths

---

## Data Structures

### candidate_df

DataFrame created by `01_ms2c_inf_create_candidate_df.py`. Each row represents a query spectrum with its candidate molecules.

| Column | Type | Description |
|--------|------|-------------|
| `spec_id` | int | Spectrum identifier |
| `mol_id` | int | Query molecule identifier |
| `peaks` | list | Spectrum peaks as `[[mz1, int1], [mz2, int2], ...]` |
| `prec_type` | str | Precursor type (e.g., "[M+H]+") |
| `inst_type` | str | Instrument type |
| `ace` | float | Absolute collision energy |
| `nce` | float | Normalized collision energy |
| `ace_extra_1` | float | Extra ACE parameter 1 |
| `nce_extra_1` | float | Extra NCE parameter 1 |
| `ace_extra_2` | float | Extra ACE parameter 2 |
| `nce_extra_2` | float | Extra NCE parameter 2 |
| `candidate_smiles_list` | list[str] | List of candidate SMILES strings |

### Per-query spec_df

DataFrame with one row per candidate molecule, all sharing the same query spectrum.

| Column | Type | Description |
|--------|------|-------------|
| `spec_id` | int | Local spectrum ID (0 to n-1) |
| `mol_id` | int | Local molecule ID (0 to n-1) |
| `peaks` | list | Query spectrum peaks (same for all rows) |
| `prec_type` | str | Precursor type |
| `inst_type` | str | Instrument type |
| `ace` | float | Absolute collision energy |
| `nce` | float | Normalized collision energy |
| `ace_extra_1` | float | Extra ACE parameter 1 |
| `nce_extra_1` | float | Extra NCE parameter 1 |
| `ace_extra_2` | float | Extra ACE parameter 2 |
| `nce_extra_2` | float | Extra NCE parameter 2 |

### Per-query mol_df

DataFrame with candidate molecule properties.

| Column | Type | Description |
|--------|------|-------------|
| `mol_id` | int | Local molecule ID (0 to n-1) |
| `smiles` | str | Canonical SMILES |
| `mol` | RDKit Mol | RDKit molecule object |
| `inchikey_s` | str | Short InChIKey (first 14 chars) |
| `scaffold` | str | Murcko scaffold SMILES |
| `formula` | str | Molecular formula |
| `inchi` | str | InChI string |
| `mw` | float | Average molecular weight |
| `exact_mw` | float | Exact molecular weight |
| `num_atoms` | int | Number of atoms |
| `num_bonds` | int | Number of bonds |
| `charge` | int | Formal charge |
| `single_mol` | bool | True if single connected component |
| `num_radicals` | int | Number of radical electrons |

### query_index.csv

CSV file indexing all processed queries.

| Column | Type | Description |
|--------|------|-------------|
| `query_id` | int | Query identifier |
| `mol_query_id` | int | Molecule query identifier |
| `spec_df_path` | str | Path to per-query spec_df |
| `mol_df_path` | str | Path to per-query mol_df |

---

## Running Inference

### run_ms2c_predict_val.py

Runs inference on MS2C prediction datasets using a trained FraGNNet checkpoint.

**Usage:**
```bash
python scripts/ms2c/run_ms2c_predict_val.py \
    --proc_dp data/ms2c/proc/nps_nist23 \
    --frag_dp data/ms2c/frags/d3_h4_isoFalse/nps_nist23 \
    --save_fp data/ms2c/predicted/nps_nist23_predictions.pkl \
    --wandb_run_id wf1zkb7n \
    --auxiliary_scores cos_sim cos_hun jss true_oos_prob \
    --eval_mz_bin_res 0.01 0.1
```

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--device` | str | `cuda:0` | Device to run inference on |
| `--save_fp` | str | - | Output file path for predictions |
| `--proc_dp` | str | - | Directory containing processed mol_df.pkl.gz and spec_df.pkl.gz |
| `--frag_dp` | str | - | Directory containing fragmentation DAG data |
| `--wandb_run_id` | str | - | W&B run ID to download checkpoint from |
| `--model_ckpt` | str | None | Local checkpoint path (alternative to wandb_run_id) |
| `--model_save_dp` | str | `./saved_ckpts` | Directory to save downloaded checkpoints |
| `--use_cached_ckpt` | bool | False | Use cached checkpoint if available |
| `--batch_size` | int | 50 | Batch size for inference |
| `--num_workers` | int | CPU count | Number of dataloader workers |
| `--auxiliary_scores` | list | `[cos_sim, cos_hun, true_oos_prob, true_oos_e]` | Auxiliary scores to compute |
| `--eval_mz_bin_res` | list | `[0.01]` | m/z bin resolutions for binned metrics |
| `--custom_fp` | str | None | Custom config file path |
| `--template_fp` | str | `./config/template.yml` | Template config file path |

---

## Available Auxiliary Scores

When running inference with `run_ms2c_predict_val.py`, you can specify auxiliary scores to compute via the `--auxiliary_scores` argument.

### Score Types

| Score | Description | Binned | Hungarian |
|-------|-------------|--------|-----------|
| `cos_sim` | Cosine similarity | ✓ | |
| `cos_hun` | Cosine similarity with Hungarian matching | | ✓ |
| `jss` | Jensen-Shannon similarity | ✓ | |
| `jss_hun` | JSS with Hungarian matching | | ✓ |
| `opt_cos_sim` | Optimal cosine similarity | ✓ | |
| `ndcg` | Normalized DCG (union/intersection variants) | | ✓ |
| `wrecall` | Weighted recall | | ✓ |
| `wprecision` | Weighted precision | | ✓ |
| `true_oos_prob` | True out-of-sample probability | | ✓ |

### Score Modifiers

**Binned scores** (`cos_sim`, `jss`, `opt_cos_sim`) support:
- `_sqrt` suffix: Apply sqrt transform to intensities before scoring
- `_np` suffix: Remove precursor peak before scoring
- `_{bin_res}` suffix: m/z bin resolution (e.g., `_0.01`)

**Hungarian scores** (`cos_hun`, `jss_hun`) support:
- `_sqrt` suffix: Apply sqrt transform to intensities
- `_np` suffix: Remove precursor peak

### Generated Metric Names

For `auxiliary_scores=["cos_sim"]` with `eval_mz_bin_res=[0.01]`:
- `cos_sim_0.01` - Standard cosine similarity at 0.01 Da
- `cos_sim_sqrt_0.01` - With sqrt intensity transform
- `cos_sim_np_0.01` - With precursor peak removed
- `cos_sim_sqrt_np_0.01` - Both sqrt and precursor removed

For `auxiliary_scores=["cos_hun"]`:
- `cos_hun` - Standard cosine with Hungarian matching
- `cos_hun_sqrt` - With sqrt intensity transform
- `cos_hun_np` - With precursor peak removed
- `cos_hun_sqrt_np` - Both sqrt and precursor removed

### Example: Generated Metrics

To compute sqrt variants, explicitly include `_sqrt` metrics in `--auxiliary_scores`:

```bash
# Only regular metrics
python scripts/ms2c/run_ms2c_predict_val.py \
    --auxiliary_scores cos_sim cos_hun \
    --eval_mz_bin_res 0.01

# Generated: cos_sim_0.01, cos_hun

# Include sqrt variants
python scripts/ms2c/run_ms2c_predict_val.py \
    --auxiliary_scores cos_sim cos_sim_sqrt cos_hun cos_hun_sqrt \
    --eval_mz_bin_res 0.01

# Generated: cos_sim_0.01, cos_sim_sqrt_0.01, cos_hun, cos_hun_sqrt
```

The `_sqrt` suffix in auxiliary_scores automatically enables the corresponding config:
- `cos_sim_sqrt`, `jss_sqrt`, `opt_cos_sim_sqrt` → enables `eval_bin_sqrt`
- `cos_hun_sqrt`, `jss_hun_sqrt` → enables `eval_hun_sqrt`
