# CovBat harmonization

## Purpose

Prepare wide CSVs and run CovBat batch correction on pyAFQ and mni_micro features.

## Entry points

prep_pyafq_covbat.py, covbat_pyafq_nodewise.R

## Inputs

pyafq, mni_micro, metadata

## Outputs

derivatives/covbat

## Configuration

Paths are resolved via [`config.yaml`](../config.example.yaml) and [`lib/paths.py`](../lib/paths.py).
