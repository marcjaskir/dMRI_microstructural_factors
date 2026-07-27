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
