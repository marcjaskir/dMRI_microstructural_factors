## Multi-atlas bundle segmentation

This data is made to be used with the following script:
https://github.com/scilus/scilpy/blob/master/scripts/scil_tractogram_segment_with_bundleseg.py


*Etienne St-Onge, Kurt Schilling, Francois Rheault, "BundleSeg: A versatile, reliable and reproducible approach to whitte matter bundle segmentation.", arXiv, 2308.10958 (2023)*
*Rheault, François. "Analyse et reconstruction de faisceaux de la matière blanche." Computer Science (Université de Sherbrooke) (2020), https://savoirs.usherbrooke.ca/handle/11143/17255*

## Usage
Here is an example (for more details use `scil_tractogram_segment_with_bundleseg.py -h`) :

```bash
antsRegistrationSyNQuick.sh -d 3 -f ${T1} -m mni_masked.nii.gz -t a -n 4
scil_tractogram_segment_with_bundleseg.py ${TRACTOGRAM} config_fss_1.json atlas/ output0GenericAffine.mat --out_dir ${OUTPUT_DIR}/ --log_level DEBUG --processes 8 --seeds 0 --inverse -f
```

To facilitate interpretation, all endpoints were uniformized head/tail. To see, which side of a bundle is head or tail, you can load the atlas bundle into the software MI-Brain https://github.com/imeka/mi-brain

(If you are processing multiple subjects, this pipeline could be useful for you https://github.com/scilus/rbx_flow)

## Notes on bundles
- The bundles follow the overall anatomical definition of TractSeg (initially from TractQuerier), but are in fact an heavily processed union to discard false positive, outliers, unrealistic path, etc.
- CG has 3 possible endpoint locations. However, the full extent of the tail is difficult to track 
- The cerebellum is often cut due to acquisition FOV. In such a case, all projection bundles will be more difficult to recognize and most cerebellum bundles will be missing (ICP, MCP, SCP).
- All the bundles starting with T_ (Thalamo) or ST_ (Striato) are based on region of interest, and are not usually part of classical major pathways.

## New Nomenclature Table

| #  | File Name      | Description                       |
|----|----------------|-----------------------------------|
| 1  | AF_left        | Arcuate fascicle (left)           |
| 2  | AF_right       | Arcuate fascicle (right)          |
| 3  | ATR_left       | Anterior Thalamic Radiation (left)|
| 4  | ATR_right      | Anterior Thalamic Radiation (right)|
| 5  | CA             | Commissure Anterior               |
| 6  | CC_1           | Rostrum                           |
| 7  | CC_2           | Genu                              |
| 8  | CC_3           | Rostral body (Premotor)           |
| 9  | CC_4           | Anterior midbody (Primary Motor)  |
| 10 | CC_5           | Posterior midbody (Primary Somatosensory) |
| 11 | CC_6           | Isthmus                           |
| 12 | CC_7           | Splenium                          |
| 13 | CG_left        | Cingulum (left)                   |
| 14 | CG_right       | Cingulum (right)                  |
| 15 | CST_left       | Corticospinal tract (left)        |
| 16 | CST_right      | Corticospinal tract (right)       |
| 17 | MLF_left       | Middle longitudinal fascicle (left) |
| 18 | MLF_right      | Middle longitudinal fascicle (right) |
| 19 | FPT_left       | Fronto-pontine tract (left)       |
| 20 | FPT_right      | Fronto-pontine tract (right)      |
| 21 | FX_left        | Fornix (left)                     |
| 22 | FX_right       | Fornix (right)                    |
| 23 | ICP_left       | Inferior cerebellar peduncle (left) |
| 24 | ICP_right      | Inferior cerebellar peduncle (right) |
| 25 | IFO_left       | Inferior occipito-frontal fascicle (left) |
| 26 | IFO_right      | Inferior occipito-frontal fascicle (right) |
| 27 | ILF_left       | Inferior longitudinal fascicle (left) |
| 28 | ILF_right      | Inferior longitudinal fascicle (right) |
| 29 | MCP            | Middle cerebellar peduncle        |
| 30 | OR_left        | Optic radiation (left)            |
| 31 | OR_right       | Optic radiation (right)           |
| 32 | POPT_left      | Parieto-occipital pontine (left)  |
| 33 | POPT_right     | Parieto-occipital pontine (right) |
| 34 | SCP_left       | Superior cerebellar peduncle (left) |
| 35 | SCP_right      | Superior cerebellar peduncle (right) |
| 36 | SLF_I_left     | Superior longitudinal fascicle I (left) |
| 37 | SLF_I_right    | Superior longitudinal fascicle I (right) |
| 38 | SLF_II_left    | Superior longitudinal fascicle II (left) |
| 39 | SLF_II_right   | Superior longitudinal fascicle II (right) |
| 40 | SLF_III_left   | Superior longitudinal fascicle III (left) |
| 41 | SLF_III_right  | Superior longitudinal fascicle III (right) |
| 42 | STR_left       | Superior Thalamic Radiation (left)|
| 43 | STR_right      | Superior Thalamic Radiation (right)|
| 44 | UF_left        | Uncinate fascicle (left)          |
| 45 | UF_right       | Uncinate fascicle (right)         |
| 46 | CC             | Corpus Callosum - all             |
| 47 | T_PREF_left    | Thalamo-prefrontal (left)         |
| 48 | T_PREF_right   | Thalamo-prefrontal (right)        |
| 49 | T_PREM_left    | Thalamo-premotor (left)           |
| 50 | T_PREM_right   | Thalamo-premotor (right)          |
| 51 | T_PREC_left    | Thalamo-precentral (left)         |
| 52 | T_PREC_right   | Thalamo-precentral (right)        |
| 53 | T_POSTC_left   | Thalamo-postcentral (left)        |
| 54 | T_POSTC_right  | Thalamo-postcentral (right)       |
| 55 | T_PAR_left     | Thalamo-parietal (left)           |
| 56 | T_PAR_right    | Thalamo-parietal (right)          |
| 57 | T_OCC_left     | Thalamo-occipital (left)          |
| 58 | T_OCC_right    | Thalamo-occipital (right)         |
| 59 | ST_FO_left     | Striato-fronto-orbital (left)     |
| 60 | ST_FO_right    | Striato-fronto-orbital (right)    |
| 61 | ST_PREF_left   | Striato-prefrontal (left)         |
| 62 | ST_PREF_right  | Striato-prefrontal (right)        |
| 63 | ST_PREM_left   | Striato-premotor (left)           |
| 64 | ST_PREM_right  | Striato-premotor (right)          |
| 65 | ST_PREC_left   | Striato-precentral
