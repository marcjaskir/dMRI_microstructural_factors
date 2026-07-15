# Reproducibility

## Data layout

The code expects a workspace rooted at `project_root`:

```
project_root/
  data/           # BIDS, atlases, metadata JSON (not in git)
  derivatives/    # Pipeline and analysis outputs (not in git)
  results/        # Inclusion tables (not in git)
```

Only the code repository is version-controlled. Large imaging data and subject metadata remain local.

## Configuration

Copy `config.example.yaml` to `config.yaml` and set:

```yaml
project_root: /path/to/your/workspace
inclusion_subdir: inclusion   # or "1_inclusion" for legacy layouts
```

Override at runtime with environment variables:

- `DMRI_MICRO_CONFIG` — path to alternate config file
- `DMRI_MICRO_ROOT` — override `project_root`

## Golden-output tests

Located in [`tests/golden/`](../tests/golden/).

```bash
conda activate dmri_microstructural_factors  # or structural_tractometry
python tests/golden/run_golden_tests.py capture  # first run only
python tests/golden/run_golden_tests.py
```

Tests verify:
1. ILF along-tract mean profile (100 nodes) matches pre-cleanup values
2. Per-subject tract asymmetry summary statistics unchanged
3. `profile_thirds_example` PNG output unchanged (MD5)

Floating-point tolerance: `rtol=1e-6`, `atol=1e-8`.

## Singularity images

QSIPrep, QSIRecon, and Trekker `.sif` files are not shipped. Obtain images from their respective projects and configure paths locally.

## Subject metadata

Surgical outcomes and inclusion demographics are loaded from external CSV files specified in `config.yaml`. No subject IDs are embedded in repository source code.
