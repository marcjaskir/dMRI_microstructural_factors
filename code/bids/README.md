# BIDS preparation

## Purpose

Link Penn NeuroBridge BIDS data; HCP Lifespan to BIDS conversion.

HCP Young Adult / Aging data were converted to BIDS using
[HCPLifespan2BIDS](https://github.com/ellisdg/HCPLifespan2BIDS).

## Entry points

link_bids_penn.sh, add_topup_intended_for_penn.py

## Inputs

external BIDS

## Outputs

data/

## Configuration

Paths are resolved via [`config.yaml`](../config.example.yaml) and [`lib/paths.py`](../lib/paths.py).
