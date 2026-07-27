# Golden-output regression tests

These tests verify that cleaned code produces the same numeric results as before reorganization.

## Setup

1. Copy `config.example.yaml` to `config.yaml` and set `project_root` to your data workspace.
2. For legacy `structural_tractometry` layouts, set `inclusion_subdir: 1_inclusion`.

## Capture baseline (first run)

```bash
cd /path/to/dMRI_microstructural_factors
python tests/golden/run_golden_tests.py capture
```

This writes CSV summaries to `tests/golden/baseline/` (gitignored).

## Run tests

```bash
python tests/golden/run_golden_tests.py
```

Tests compare:
- ILF mean along-tract profile (100 nodes) from `profile_thirds_example`
- Per-subject tract asymmetry summary statistics aggregated from existing derivatives
- Optional PNG checksum for `profile_thirds_example` output

Tolerances: `rtol=1e-6`, `atol=1e-8` for floating-point columns.
