# ACPC–MNI transforms

## Purpose

Disassemble QSIPrep composite ACPC↔MNI transforms into affine and displacement fields for BundleSeg.

## Entry points

get_acpc_mni_affine.sh, ingress2qsirecon_hcpya.sh

## Inputs

qsiprep, qsirecon

## Outputs

derivatives/acpc_mni_xfm

## Configuration

Paths are resolved via [`config.yaml`](../config.example.yaml) and [`lib/paths.py`](../lib/paths.py).
