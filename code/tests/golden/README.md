# Golden-output regression tests

These tests verify that cleaned code and open products produce the same numeric
results across refactors (`rtol≈1e-6`).

## Setup

```bash
cd /path/to/dMRI_microstructural_factors
cp config.example.yaml config.yaml   # set workspace_root
export PYTHONPATH="$PWD/code:$PYTHONPATH"
```

Open products must be present under `data/open/` (download from
[https://osf.io/xsr7y](https://osf.io/xsr7y); see root README).

## Capture baseline (first run / after justified refresh)

```bash
python code/tests/golden/run_golden_tests.py capture
```

Writes under `code/tests/golden/baseline/` (gitignored):

- `profile_means.csv` — ILF mean along-tract residual-z profile
- `asymmetry_tract_summary.csv` — per-subject tract Cohen’s d aggregates
- `manuscript_digests.json` — SHA256 + numeric digests for loadings, factor z,
  LE G1/G2 / neuroaxis correlations, and group asymmetry summaries

## Run tests

```bash
python code/tests/golden/run_golden_tests.py
```

Also run contract / privacy guards:

```bash
python code/tests/test_factor_labels.py
python code/tests/test_no_phi.py
```

Tolerances: `rtol=1e-6`, `atol=1e-8` for floating-point columns. Refresh baselines
only with explicit justification (method change), not during pure reorg.
