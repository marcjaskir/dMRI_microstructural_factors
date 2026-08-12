# Open data (`data/open/`)

De-identified products for manuscript reproduction.

**Distribution:** OSF project **[https://osf.io/xsr7y](https://osf.io/xsr7y)** hosts
the `data/open/` tree (typically as `dmri_microstructural_factors_open_v1.zip`)

## Layout

```
data/open/
  README.md
  atlases/ metadata/ inclusion/   # small; committed in git
  analysis/                       # from OSF (gitignored)
  gam/                            # from OSF (gitignored)
```

## Privacy

Open CSVs use `anon_id` only. The reversible map stays in
`data/controlled/anon_id_map.csv` (gitignored).
