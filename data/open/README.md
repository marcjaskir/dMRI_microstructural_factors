# Tier 1 — open / publishable analysis products

This directory holds the **minimum open dataset** for reproducing manuscript
figures from post-GAM residual z-scores onward. No subject IDs, age, or sex.

## Layout

```
data/open/
  atlases/       # Label TSVs, tract metadata, centroids (no NIfTI volumes)
  metadata/      # Scalar label/color JSON only (no demographics)
  gam/           # Post-GAM residual z-scores
  analysis/      # Factor loadings/scores, asymmetry, gradients
  inclusion/     # Anonymized cohort table
```

## Column schemas

### `gam/**/*.csv` (Tier 1)

| Column | Required | Notes |
|--------|----------|-------|
| `anon_id` | yes | Stable anonymous subject key |
| `group` | yes | e.g. penn_controls, hcpya, penn_epilepsy |
| `*_z` / `node*_z` | yes | Residual z-scores used by factor/asymmetry analyses |

Do **not** include: age, sex, scanner/`bat`, real BIDS IDs.

### `inclusion/*.csv`

| Column | Required | Notes |
|--------|----------|-------|
| `anon_id` | yes | Matches `gam` keys |
| `group` | yes | Cohort label |
| `laterality` | for TLE | left / right (ipsi/contra) |
| `lobe` | for TLE | e.g. temporal filter |

Do **not** include: age, sex, real RID, clinical outcomes.

### `analysis/`

Factor loadings, factor scores/z, tract/region asymmetry summaries, gradient
CSVs — keyed by `anon_id` / `group` only.

## How to populate

From the repo root (with source derivatives available):

```bash
python -u code/lib/export_tier1_open.py
```

This maps real `sub` → `anon_id`, drops age/sex/scanner/clinical columns from
GAM CSVs, and writes products into the paths above. The reversible ID map is
written only to `data/controlled/anon_id_map.csv` (gitignored).

Large trees (`gam/`, `analysis/`) are gitignored; keep them local or deposit on
Zenodo. Small products (`atlases/`, `metadata/`, `inclusion/`) can be committed.

See root `README.md` and `code/docs/reproducibility.md`.
