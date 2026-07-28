# Controlled-access inputs

This directory holds **pre-GAM / pre-CovBat** inputs that require age, sex,
scanner, and/or identifiable subject keys. Contents are **gitignored** and
must not be committed.

## Layout

```
data/controlled/
  metadata/       # demo (age/sex), scanner IDs, clinical tables
  derivatives/    # pyAFQ profiles, mni_micro HDF5, covbat inputs, early pipeline
  inclusion/      # RID-identified inclusion lists (local only)
  subject_outcomes.csv
```

## Typical contents

| Path | Purpose |
|------|---------|
| `metadata/demo_*.csv` | `sub,age,sex` for CovBat/GAM covariates |
| `metadata/scanner_ids_*.csv` | Batch variable for CovBat |
| `metadata/penn_age_overrides.csv` | Optional `record_id,age` fills (not in open code) |
| `metadata/clinical_*.csv` | Clinical fields for controlled analyses |
| `derivatives/pyafq/` | Along-tract profiles before CovBat |
| `derivatives/mni_micro/` | Region/tract HDF5 before CovBat |
| `derivatives/covbat/` | CovBat inputs/outputs |
| `derivatives/qsiprep|qsirecon|...` | Imaging pipeline outputs |

## Local development

Point `data_controlled_dir` in `config.yaml` at an existing workspace (e.g.
`structural_tractometry`) instead of copying large trees into this folder.

## Sharing

Release under a data-use agreement / controlled-access repository (e.g. dbGaP,
institutional DUA). Do not place identifiable or covariate files under
`data/open/`.
