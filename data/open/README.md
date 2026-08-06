# Open data (`data/open/`)

De-identified products for manuscript reproduction. Analysis code reads
`gam_dir` / `analysis_dir` (and related roots) under this tree via `config.yaml`.

**Distribution:** OSF project **[https://osf.io/xsr7y](https://osf.io/xsr7y)** hosts
the `data/open/` tree (typically as `dmri_microstructural_factors_open_v1.zip`).
There is no HDF5 unpack step.

## After clone

1. Clone this repository (ships small committed trees: `atlases/`, `metadata/`,
   `inclusion/`, and placeholders).
2. Download open data from [https://osf.io/xsr7y](https://osf.io/xsr7y) into
   `data/open/` so `analysis/` and `gam/` sit beside those committed stubs.
3. Copy `config.example.yaml` → `config.yaml` and set `workspace_root`.

```bash
# config.yaml: set workspace_root; open_osf_url is https://osf.io/xsr7y
mkdir -p data/open
# Prefer the OSF zip (contents are rooted at data/open/):
unzip /path/to/dmri_microstructural_factors_open_v1.zip -d data/open
# Or sync individual OSF storage files into data/open/
```

## Layout

```
data/open/
  README.md
  atlases/ metadata/ inclusion/   # small; committed in git
  analysis/                       # from OSF (gitignored)
  gam/                            # from OSF (gitignored)
```

### Committed (git)

- Atlas label tables / metadata JSON filtered to manuscript scalars & tracts
- Anonymized inclusion lists (`anon_id`, laterality/lobe; no age/sex)

### From OSF (typically)

- `analysis/` — factor loadings/z, flattened LE gradient CSVs,
  `factor_representation/factor_matched_subject_similarity.csv`, tract subject
  asymmetry (`tract_asymmetry`), and group digests under
  `microstructural_asymmetries/` (not the full subject `region_asymmetry_tle`
  tree; regenerate GM reports from controlled data if needed)
- `gam/` — manuscript-allowlisted residual-z GAM tables under `pyafq/` and
  `mni_micro/` (`anon_id` + `*_z` only; no age/sex/scanner/batch)

You may upload only `analysis/` + `gam/` (users keep git’s `atlases/` /
`metadata/` / `inclusion/`), or the whole `data/open/` tree / zip for a
one-stop download.

## Lab export (maintainers)

```bash
# From controlled structural_tractometry source
python -u code/lib/export_tier1_open.py --core
# Or refresh GAM only:
python -u code/lib/export_tier1_open.py --gam-only --force-gam
# Optional: rebuild zip for OSF upload
# (from repo)  cd data/open && zip -r ../dmri_microstructural_factors_open_v1.zip .
```

**Intentionally omitted from open products:** raw dMRI, CovBat covariates
(age/sex/scanner), reversible ID maps, NIfTI volumes, HTML dumps, and
scalars / WM tracts outside the manuscript analysis set
(`lib/manuscript_features.py`).

## Privacy

Open CSVs use `anon_id` only. The reversible map stays in
`data/controlled/anon_id_map.csv` (gitignored).
