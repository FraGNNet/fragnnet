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
