# Imaging pipeline

End-to-end flow from raw data to GAM-adjusted microstructural features.

## Stage 0: Data ingress

| Module | Purpose | Entry points |
|--------|---------|--------------|
| [bids](../bids/) | Link Penn BIDS; HCP Lifespan conversion | `link_bids_penn.sh`, `HCPLifespan2BIDS/` |
| [hcpya](../hcpya/) | HCP-YA DataLad pull, QC, derivative organization | `pull_hcpya_datalad.sh`, `gen_hcpya_qc.py` |
| [metadata](../metadata/) | Generate demo/scanner/clinical tables from local sources | `gen_penn_metadata.py`, `gen_hcpaging_metadata.py` |

## Stage 1: Preprocessing and reconstruction

| Module | Purpose | Entry points |
|--------|---------|--------------|
| [qsiprep](../qsiprep/) | QSIPrep container preprocessing | `run_qsiprep.sh` → `qsiprep.sh` |
| [freesurfer](../freesurfer/) | FreeSurfer recon-all; HCP ingress | `run_recon_all.sh`, `recon_all.sh` |
| [qsirecon](../qsirecon/) | QSIRecon microstructural maps | `run_qsirecon.sh`, `recon_spec_*.yaml` |
| [acpc_mni_xfm](../acpc_mni_xfm/) | ACPC↔MNI transforms for BundleSeg | `get_acpc_mni_affine.sh` |
| [dmri_convert](../dmri_convert/) | DSI Studio ↔ MRtrix conversion | `dsistudio_to_mrtrix.py` |

## Stage 2: Tractography and segmentation

| Module | Purpose | Entry points |
|--------|---------|--------------|
| [trekker](../trekker/) | Whole-brain Trekker tracking | `run_trekker.sh` → `trekker.sh` |
| [bundleseg](../bundleseg/) | BundleSeg tract segmentation | `wrapper_run_bundleseg.sh`, `run_bundleseg.py` |
| [atlases](../atlases/) | Atlas construction and QC helpers | `gen_glasser_atlas_centroids.py`, `trk_to_nii_HCP1065.py` |
| [atlas_centroids](../atlas_centroids/) | Unified MNI centroid tables | `compute_atlas_centroids.py` |
| [centroid_coordinates](../centroid_coordinates/) | Node-wise streamline centroids | `centroid_coordinates.py` |

## Stage 3: Along-tract and region microstructure

| Module | Purpose | Entry points |
|--------|---------|--------------|
| [pyafq](../pyafq/) | Along-tract scalar profiling (HCP1065) | `run_pyafq.sh` → `pyafq.py` |
| [mni_micro](../mni_micro/) | Region/tract MNI-space scalar means (HDF5) | `mni_atlas_micro.py` |

## Stage 4: Harmonization and normative modeling

| Module | Purpose | Entry points |
|--------|---------|--------------|
| [covbat](../covbat/) | CovBat batch correction (pyAFQ + mni_micro) | `prep_pyafq_covbat.py`, `covbat_pyafq_nodewise.R` |
| [gam](../gam/) | Age/sex GAMs; residual z-scores | `gam_pyafq.R`, `gam_mni_micro.R` |

## Dependencies

Singularity images for QSIPrep, QSIRecon, and Trekker are **not** included. Document local image paths in your site config or environment.

## Config keys used

- `project_root`, `data_dir`, `derivatives_dir`
- `penn_bids_dir`, `freesurfer_image`, `singularity_tmpdir` (site-specific)
