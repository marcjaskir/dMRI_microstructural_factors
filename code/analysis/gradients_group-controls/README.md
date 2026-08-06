# Control gradients

## Purpose

Laplacian eigenmaps (BrainSpace) on control factor scores — manuscript G1/G2
gradients. Diffusion-map embedding remains available as an exploratory option
(`--method diffusion_embedding`) but is not a paper deliverable.

## Entry points

```bash
python compute_gradients.py --method laplacian_eigenmodes
```

## Inputs

`factor_z-scores`, atlas centroids / labels

## Outputs

`derivatives/analysis/gradients_group-controls/laplacian_eigenmodes/`

## Configuration

Paths are resolved via [`config.yaml`](../../config.example.yaml) and [`lib/paths.py`](../../lib/paths.py).
