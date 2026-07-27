# QSIRecon

## Purpose

Run QSIRecon reconstructors (DKI, NODDI, GQI, MAP-MRI, atlases).

## Entry points

run_qsirecon.sh, recon_spec.yaml, recon_spec_custom_penn.yaml, recon_spec_custom_hcpaging.yaml

## Inputs

qsiprep, freesurfer (Penn / HCP-Aging); existing HCP derivatives for HCP-YA

## Outputs

derivatives/qsirecon

## Configuration

Paths are resolved via [`config.yaml`](../config.example.yaml) and [`lib/paths.py`](../lib/paths.py).
