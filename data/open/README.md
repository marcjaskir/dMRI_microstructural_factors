# Open data (`data/open/`)

De-identified products for manuscript reproduction. Analysis code reads
`gam_dir` / `analysis_dir` (and related roots) under this tree via `config.yaml`.

**Distribution model:** OSF hosts the **`data/open/` directory tree** (CSV/JSON/TSV).
There is no HDF5 unpack step.

## After clone

1. Clone this repository (ships small committed trees: `atlases/`, `metadata/`,
   `inclusion/`, and placeholders).
2. Download the OSF **storage** files into `data/open/` so `analysis/` and `gam/`
   sit beside those committed stubs (merge/overwrite as needed).
3. Run golden tests (see repo root `README.md`).

```bash
# 1. config.yaml: set workspace_root and open_osf_url (project/storage link)
# 2. Sync OSF files into data/open/ (browser, osfclient, or manual zip extract)
# 3. Confirm analysis/ and gam/ exist, then:
export PYTHONPATH="$PWD/code:$PYTHONPATH"
python code/tests/golden/run_golden_tests.py
```

Placeholder until the OSF project exists: set `open_osf_url` in `config.yaml` to
`https://osf.io/XXXXX/` (or your storage/files URL).

## Layout

```
data/open/
  README.md
  atlases/ metadata/ inclusion/   # small; committed in git
  analysis/                       # from OSF (gitignored)
  gam/                            # from OSF (gitignored; core sample or fuller export)
```

### Committed (git)

- Atlas label tables / metadata JSON filtered to manuscript scalars & tracts
- Anonymized inclusion lists (`anon_id`, laterality/lobe; no age/sex)

### From OSF (typically)

- `analysis/` — factor loadings/scores/z, LE gradient CSVs, asymmetry digests
- `gam/` — at least the minimal core sample
  (`gam/pyafq/HCP1065/ILF_L/ILF_L_dti_md_stat-mean_gam.csv`), or a fuller GAM
  dump if you export with `export_tier1_open.py` without `--core`

You may upload only `analysis/` + `gam/` (users keep git’s `atlases/` /
`metadata/` / `inclusion/`), or the whole `data/open/` tree for a one-stop
download.

## Core manuscript share

Default lab export for OSF is the **core** product set (golden tests + published
digests), not a full multi‑GB GAM re-export:

```bash
# Lab only — from controlled structural_tractometry source
python -u code/lib/export_tier1_open.py --core
```

Omit `--core` for a fuller CSV tree (all manuscript-allowlisted GAM tables).

**Intentionally omitted from open products:** raw dMRI, CovBat covariates
(age/sex/scanner), reversible ID maps, NIfTI volumes, HTML dumps, and
scalars / WM tracts outside the manuscript analysis set
(`lib/manuscript_features.py`).

## Privacy

Open CSVs use `anon_id` only. The reversible map stays in
`data/controlled/anon_id_map.csv` (gitignored). Run:

```bash
python code/tests/test_no_phi.py
```
