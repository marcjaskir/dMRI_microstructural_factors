# dMRI Microstructural Factors

Code accompanying the publication on diffusion MRI microstructural factor analysis and temporal lobe epilepsy asymmetry.

This repository contains **code only**. Imaging data, derivatives, and subject metadata are kept outside the git tree and configured via [`config.yaml`](config.yaml).

## Quick start

```bash
git clone git@github.com:marcjaskir/dMRI_microstructural_factors.git
cd dMRI_microstructural_factors
cp config.example.yaml config.yaml
# Edit config.yaml: set project_root to your data workspace

conda env create -f environment.yml
conda activate dmri_microstructural_factors

python tests/golden/run_golden_tests.py
```

## Configuration

All filesystem roots are defined in [`config.yaml`](config.yaml) (see [`config.example.yaml`](config.example.yaml)):

| Key | Purpose |
|-----|---------|
| `project_root` | Root of data workspace (`data/`, `derivatives/`, `results/`) |
| `data_dir` | BIDS inputs, atlases, metadata JSON |
| `derivatives_dir` | Pipeline and analysis outputs |
| `results_dir` | Inclusion tables and curated summaries |
| `inclusion_subdir` | Subfolder under `results_dir` for inclusion CSVs |
| `subject_outcome_csv` | External surgical outcome table (not in repo) |

Python scripts import [`lib/paths.py`](lib/paths.py). Shell scripts resolve `$BASE` from the same config.

## Pipeline overview

```
BIDS / HCP-YA  →  qsiprep  →  freesurfer  →  qsirecon
                              ↓
                         acpc_mni_xfm
                              ↓
              trekker  →  bundleseg  →  pyafq / mni_micro
                              ↓
                         covbat  →  gam
                              ↓
              factor_analysis  →  factor_z-scores  →  gradients
                              ↓
         tract/region asymmetry  →  microstructural asymmetry reports
```

See [docs/pipeline.md](docs/pipeline.md) and [docs/analysis.md](docs/analysis.md) for details.

## Privacy

No subject identifiers, demographics, or clinical outcomes are stored in this repository. Inclusion lists and outcome labels must be provided locally via `config.yaml`.

## Documentation

- [docs/index.md](docs/index.md) — GitHub Pages home
- [docs/reproducibility.md](docs/reproducibility.md) — golden tests and data layout

Each top-level folder and `analysis/` subfolder has its own `README.md`.

## Citation

> *[Publication citation to be added]*

## License

See repository license file (to be added upon public release).
