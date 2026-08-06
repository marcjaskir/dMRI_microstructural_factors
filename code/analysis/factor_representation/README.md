# Factor representation

## Purpose

Measure how well model statistics (NODDI, MAP-MRI, DKI) represent whole-brain
factor-score gradients (overall / non-Gaussian / anisotropic diffusivity).

## Entry points

`factor_representation.py`

## Inputs

`factor_analysis`, `factor_z-scores`

## Outputs

`derivatives/analysis/factor_representation` — per-model similarity / triplet CSVs,
factor-bar figures, and a consolidated subject-level table
`factor_matched_subject_similarity.csv` (all NODDI/MAP-MRI/DKI statistics with
factor-matched |loading| > 0.5; columns include per-subject |cosine| and
|Pearson r|). The open OSF share ships only that consolidated CSV.

## Configuration

Paths are resolved via [`config.yaml`](../../config.example.yaml) and [`lib/paths.py`](../../lib/paths.py).
