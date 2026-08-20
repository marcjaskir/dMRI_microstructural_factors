# dMRI Microstructural Factors

Code and data for reproducing results of Jaskir et al. 2026, Mapping Whole-Brain
Factors of Microstructural Similarity with Diffusion MRI.

## Pipeline overview
- **Penn + HCP-Aging:** BIDS → qsiprep → freesurfer → qsirecon
- **HCP-YA:** → qsirecon (minimally preprocessed publicly released data were used)

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

Additional code documentation:
* [`code/docs/pipeline.md`](code/docs/pipeline.md),
* [`code/docs/analysis.md`](code/docs/analysis.md),
* [`code/docs/reproducibility.md`](code/docs/reproducibility.md).

## Data

**Open data for manuscript reproduction:** download from OSF
([https://osf.io/xsr7y](https://osf.io/xsr7y)) and place under `data/open/`

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

## Citation

Marc Jaskir, Alfredo Lucas, Daniel J. Zhou, William K.S. Ojemann, Justin Chin, Mariam Josyula, Nina Petillo, Emily Zhang, Briana Macedo, Nishant Sinha, Tyler M. Moore, Sandhitsu R. Das, Joel M. Stein, Matthew Cieslak, Theodore D. Satterthwaite, Kathryn A. Davis. Mapping Whole-Brain Factors of Microstructural Similarity with Diffusion MRI. bioRxiv. https://doi.org/10.64898/2026.08.11.740985


## Study overview figure from Jaskir et. al 2016
<img width="7643" height="9555" alt="Fig1_Study_Overview" src="https://github.com/user-attachments/assets/916088b7-46b4-4d2e-a870-343704ad6834" />

