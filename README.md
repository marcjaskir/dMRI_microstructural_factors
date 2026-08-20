# dMRI Microstructural Factors

Code and data for reproducing results of Jaskir et al. 2026, Mapping Whole-Brain
Factors of Microstructural Similarity with Diffusion MRI.

## Repository layout

```
dMRI_microstructural_factors/
  code/                 # All pipelines and analyses
  data/
    open/               # Small committed trees + OSF-synced analysis/gam
  config.example.yaml
  README.md
```

## Data availability and privacy

**Open data for manuscript reproduction:** download from OSF
([https://osf.io/xsr7y](https://osf.io/xsr7y)) and place under `data/open/`

See [`data/open/README.md`](data/open/README.md) and
[`code/docs/reproducibility.md`](code/docs/reproducibility.md) for layout and
what is intentionally omitted (raw dMRI, CovBat covariates).

## Pipeline overview

Cohort-specific ingress differs before the shared tractography path:

- **Penn + HCP-Aging:** BIDS → qsiprep → freesurfer → qsirecon
- **HCP-YA:** → qsirecon (skip qsiprep/freesurfer; use existing HCP derivatives)

```
Penn / HCP-Aging:  BIDS  →  qsiprep  →  freesurfer  →  qsirecon
HCP-YA:                                     →  qsirecon
                              ↓
                         acpc_mni_xfm
                              ↓
              trekker  →  bundleseg  →  pyafq / mni_micro
                              ↓
                      covbat  →  gam
                              ↓
   factor_analysis  →  factor_z-scores  →  Laplacian eigenmaps
                              ↓
              tract/region asymmetry  →  microstructural_asymmetries
```

HCP Young Adult / Aging BIDS conversion uses
[HCPLifespan2BIDS](https://github.com/ellisdg/HCPLifespan2BIDS).

All steps after GAM fitting are under code/analysis.

Additional Docs: [`code/docs/pipeline.md`](code/docs/pipeline.md),
[`code/docs/analysis.md`](code/docs/analysis.md),
[`code/docs/reproducibility.md`](code/docs/reproducibility.md).

## Quick start

```bash
git clone git@github.com:marcjaskir/dMRI_microstructural_factors.git
cd dMRI_microstructural_factors
cp config.example.yaml config.yaml

# At this point, edit config.yaml to specify desired local paths

conda env create -f environment.yml
conda activate dmri_microstructural_factors

export PYTHONPATH="$PWD/code:$PYTHONPATH"

# Download open data from OSF (https://osf.io/xsr7y), e.g. the open zip, then:
mkdir -p data/open
unzip /path/to/dmri_microstructural_factors_open_v1.zip -d data/open
```

All filesystem roots are defined in `config.yaml` (see
[`config.example.yaml`](config.example.yaml)).

## Study overview figure from Jaskir et. al 2016
<img width="7643" height="9555" alt="Fig1_Study_Overview" src="https://github.com/user-attachments/assets/d90bb01d-faf4-4dff-9abc-1a5a114deea1" />

## Citation

Mapping Whole-Brain Factors of Microstructural Similarity with Diffusion MRI
Marc Jaskir, Alfredo Lucas, Daniel J. Zhou, William K.S. Ojemann, Justin Chin, Mariam Josyula, Nina Petillo, Emily Zhang, Briana Macedo, Nishant Sinha, Tyler M. Moore, Sandhitsu R. Das, Joel M. Stein, Matthew Cieslak, Theodore D. Satterthwaite, Kathryn A. Davis
bioRxiv 2026.08.11.740985; doi: https://doi.org/10.64898/2026.08.11.740985
