# Open analysis products

Publishable products for reproducing manuscript figures/tables from post-GAM
residual **z-scores** onward. No subject IDs, age, or sex.

Distribution model: **one OSF-hosted HDF5**, unpacked once into this directory.

## Preferred workflow (OSF)

```bash
# In config.yaml set open_h5_osf_url (or export DMRI_MICRO_OSF_URL)
python -u code/lib/fetch_open_data.py
```

This downloads `dmri_microstructural_factors_open_v1.h5` into `data/open/` and
unpacks CSV/JSON/TSV tables beside this README. The HDF5 itself is gitignored.

**OSF placeholder (replace after upload):**

```
OSF_URL=https://osf.io/XXXXX/
```

Upload checklist: file name `dmri_microstructural_factors_open_v1.h5`, root attr
`schema_version=1`, publish SHA256 in this README once available.

Local core pack checksum (regenerate after re-pack):

```
bbb60b0006889ef1d28f9e45b78c6fa6f9b1abfc739f160f58814aab8f128258  dmri_microstructural_factors_open_v1.h5
```

## Layout after unpack

```
data/open/
  dmri_microstructural_factors_open_v1.h5   # cache (gitignored)
  atlases/       # Label TSVs, tract metadata, centroids (also small files may be committed)
  metadata/      # Scalar label/color JSON for the manuscript n=26 scalars only
  inclusion/     # Anonymized cohort tables
  gam/           # Selected post-GAM residual z tables
  analysis/      # Factor loadings/scores, LE gradients, asymmetry digests
```

Committed in git: this README, small `atlases/` / `metadata/` / `inclusion/`
stubs. Large trees and the HDF5 are local/OSF only.

Open products contain only the **manuscript feature set**: 26 microstructural
scalars and 48 HCP1065 WM tracts (24 L/R pairs). Packaging/export uses
allowlists in `code/lib/manuscript_features.py`; analysis code reads whatever
is present and does not apply exclusion lists.

## HDF5 schema

Root attributes:

| Attr | Meaning |
|------|---------|
| `schema_version` | `1` |
| `paper_citation` | Manuscript citation stub |
| `created_utc` | Pack timestamp |
| `profile` | `core` or `full_csv` |
| `osf_url` | Download URL / placeholder |
| `n_files` | Number of packed payloads |
| `sha256` | Archive checksum |

Group `/files/<id>`: gzip-compressed file bytes with attrs `relpath` (posix path
under `data/open/`), `content_type`, `sha256`, `nbytes`.

### Column rules

| Product | Keep | Never include |
|---------|------|----------------|
| GAM / factor z | `anon_id`, `group`, `*_z` / `node*_z` | age, sex, scanner/`bat`, real BIDS IDs |
| Inclusion | `anon_id`, `group`, `laterality`, `lobe` | age, sex, RID, outcomes |
| Gradients / asymmetry | ROI labels, G1/G2, Cohen’s d, Mahalanobis digests | demographics, `anon_id_map` |

### Excluded from the archive

NIfTI / voxelwise volumes, HTML reports, exploratory PNGs, diffusion-map
(`diffusion_embedding`) trees, reversible `anon_id_map`, age/sex/scanner columns.

## Lab-only: build the pack

From a populated open tree (e.g. after `export_tier1_open.py`):

```bash
python -u code/lib/pack_open_h5.py pack --profile core
python -u code/lib/pack_open_h5.py check
python -u code/lib/pack_open_h5.py unpack   # round-trip
```

`core` covers golden tests + manuscript digests. `full_csv` adds broader GAM /
normative covariance CSVs (much larger).

See root `README.md` and `code/docs/reproducibility.md`.
