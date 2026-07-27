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

Export from your local controlled workspace (e.g. `structural_tractometry`) by:

1. Mapping real `sub` → `anon_id` consistently across all tables
2. Dropping age/sex/scanner/clinical columns from GAM CSVs
3. Writing products into the paths above

See root `README.md` and `code/docs/reproducibility.md`.
