import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from lib.paths import project_root, data_dir, derivatives_dir, results_dir
PROJECT_ROOT = project_root()
import pandas as pd

def merge_cohort_redcap(cohort_file, redcap_file, output_file=None):
    """
    Merge cohort CSV file with redcap CSV file.
    
    Parameters:
    -----------
    cohort_file : str
        Path to the cohort CSV file (no header, single column of subject IDs)
    redcap_file : str
        Path to the redcap CSV file (has header, contains record_id column)
    output_file : str, optional
        Path to save the merged output. If None, prints to console.
    """
    # Read input files
    cohort_df = pd.read_csv(cohort_file, header=None, names=['record_id'])
    redcap_df = pd.read_csv(redcap_file)
    
    # Convert record_id to sub-RID format
    def convert_to_sub_rid(record_id):
        if pd.isna(record_id) or '-' in str(record_id):
            return None
        return f"sub-RID{int(record_id):04d}"
    
    # Ensure record_id column exists in redcap_df
    if 'record_id' not in redcap_df.columns:
        raise ValueError("'record_id' column not found in redcap file")
        
    redcap_df['record_id'] = redcap_df['record_id'].apply(convert_to_sub_rid)
    
    # Merge dataframes
    merged_df = pd.merge(cohort_df, redcap_df, on='record_id', how='inner')

    # Rename record_id to sub
    merged_df.rename(columns={'record_id': 'sub'}, inplace=True)

    # Recode sex (1.0 --> M, 2.0 --> F)
    merged_df['sex'] = merged_df['sex'].map({1.0: 'M', 2.0: 'F'})

    # Rename t3_ageatscan to age
    merged_df.rename(columns={'t3_ageatscan': 'age'}, inplace=True)

    # Rename age_seizure_onset_pecclinical to age_onset
    merged_df.rename(columns={'age_seizure_onset_pecclinical': 'age_onset'}, inplace=True)

    # Recode emu_seizure_lateralization_pecclinical
    seizure_lateralization_mapping = {
        1: 'left',
        2: 'right',
        3: 'bilateral',
        4: 'left > right',
        5: 'right > left',
        6: 'generalized',
        7: 'inconclusive'
    }
    merged_df['emu_seizure_lateralization_pecclinical'] = merged_df['emu_seizure_lateralization_pecclinical'].map(seizure_lateralization_mapping)
    merged_df.rename(columns={'emu_seizure_lateralization_pecclinical': 'seizure_lateralization'}, inplace=True)
    
    # Recode emu_seizure_localization_pecclinical
    seizure_localization_mapping = {
        1: 'frontal',
        2: 'temporal',
        3: 'parietal',
        4: 'occipital',
        5: 'generalized',
        6: 'central',
        7: 'multifocal',
        8: 'nonlocalizable',
        9: 'insular'
    }
    merged_df['emu_seizure_localization_pecclinical'] = merged_df['emu_seizure_localization_pecclinical'].map(seizure_localization_mapping)
    merged_df.rename(columns={'emu_seizure_localization_pecclinical': 'seizure_localization'}, inplace=True)

    # Recode intervention_pecclinical and rename to intervention_type:
    # 1 --> VNS, 2 --> RNS, 3 --> DBS, 4 --> laser_ablation, 5 --> resection, 6 --> other, 7 --> no_intervention
    intervention_mapping = {
        1: 'VNS',
        2: 'RNS',
        3: 'DBS',
        4: 'laser_ablation',
        5: 'resection',
        6: 'other',
        7: 'no_intervention'
    }
    merged_df['intervention_pecclinical'] = merged_df['intervention_pecclinical'].map(intervention_mapping)
    merged_df.rename(columns={'intervention_pecclinical': 'intervention_type'}, inplace=True)

    # Recode resection_laterality_pecclinical and rename to intervention_laterality:
    # 1 --> left, 2 --> right, 3 --> bilateral
    resection_laterality_mapping = {
        1: 'left',
        2: 'right',
        3: 'bilateral'
    }
    merged_df['resection_laterality_pecclinical'] = merged_df['resection_laterality_pecclinical'].map(resection_laterality_mapping)
    merged_df.rename(columns={'resection_laterality_pecclinical': 'intervention_laterality'}, inplace=True)

    # Map resection lobe codes to names and rename to resection_lobe
    lobe_mapping = {
        1: 'anterior_temporal_lobectomy',
        2: 'other_temporal_lobe', 
        3: 'frontal',
        4: 'parietal',
        5: 'occipital',
        6: 'insula',
        7: 'cingulate',
        8: 'other'
    }

    merged_df['resection_lobe'] = None
    for i in range(1,9):
        col = f'resection_lobe_pecclinical___{i}'
        if col in merged_df.columns:
            mask = merged_df[col] == 1
            merged_df.loc[mask, 'resection_lobe'] = lobe_mapping[i]

    binary_cols = [f'resection_lobe_pecclinical___{i}' for i in range(1,9)]
    merged_df = merged_df.drop(columns=[c for c in binary_cols if c in merged_df.columns])
    merged_df.rename(columns={'resection_lobe_pecclinical': 'resection_lobe'}, inplace=True)

    # Rename resection_histopathology_pecclinical to resection_histopathology
    merged_df.rename(columns={'resection_histopathology_pecclinical': 'resection_histopathology'}, inplace=True)

    # Rename ieeg_resectarea to resection_details
    merged_df.rename(columns={'ieeg_resectarea': 'resection_details'}, inplace=True)

    # Rename engel_class_pecclinical --> engel_followup1 and months_at_followup_1 --> months_followup1
    merged_df.rename(columns={'engel_class_pecclinical': 'engel_followup1'}, inplace=True)
    merged_df.rename(columns={'months_at_followup_1': 'months_followup1'}, inplace=True)

    # Rename engel_class_2_pecclinical --> engel_followup2 and months_at_followup_2 --> months_followup2
    merged_df.rename(columns={'engel_class_2_pecclinical': 'engel_followup2'}, inplace=True)
    merged_df.rename(columns={'months_at_followup_2': 'months_followup2'}, inplace=True)

    # Rename engel_class_3_pecclinical --> engel_followup3 and months_at_followup_3 --> months_followup3
    merged_df.rename(columns={'engel_class_3_pecclinical': 'engel_followup3'}, inplace=True)
    merged_df.rename(columns={'months_at_followup_3': 'months_followup3'}, inplace=True)

    # Recode demog_hemi and rename to resection_hemi
    # 2 --> right, 3 --> left
    resection_hemi_mapping = {
        2: 'right',
        3: 'left'
    }
    merged_df['demog_hemi'] = merged_df['demog_hemi'].map(resection_hemi_mapping)
    merged_df.rename(columns={'demog_hemi': 'resection_hemi'}, inplace=True)

    # Merge resection_hemi into intervention_laterality if intervention_laterality is null
    merged_df['intervention_laterality'] = merged_df['intervention_laterality'].fillna(merged_df['resection_hemi'])
    merged_df = merged_df.drop(columns=['resection_hemi'])

    # Map resection type codes to names
    resection_mapping = {
        1: 'neocortical',
        2: 'anterior_temporal_lobectomy',
        3: 'selective_amygdalohippocampectomy', 
        4: 'corpus_callosotomy',
        5: 'multiple_subpial_transection',
        6: 'radio_surgery',
        7: 'hemispherectomy',
        8: 'therapeutic_brain_stimulation',
        9: 'other'
    }

    merged_df['resection_type'] = None
    for i in range(1,10):
        col = f'demog_resectiontype___{i}'
        if col in merged_df.columns:
            mask = merged_df[col] == 1
            merged_df.loc[mask, 'resection_type'] = resection_mapping[i]
        
    binary_cols = [f'demog_resectiontype___{i}' for i in range(1,10)]
    merged_df = merged_df.drop(columns=[c for c in binary_cols if c in merged_df.columns])
    merged_df.rename(columns={'demog_resectiontype': 'resection_type'}, inplace=True)

    # Merge resection_lobe and resection_type into resection_type - if they are equal or resection_lobe is other, keep resection_type. if they are different, separate with -
    merged_df['resection_type'] = merged_df.apply(
        lambda row: row['resection_type'] if row['resection_lobe'] == 'other' or row['resection_type'] == row['resection_lobe'] else f"{row['resection_type']}-{row['resection_lobe']}",
        axis=1
    )
    # Strip -None if it exists
    merged_df['resection_type'] = merged_df['resection_type'].apply(lambda x: x.strip('-None') if x is not None else x)
    merged_df = merged_df.drop(columns=['resection_lobe'])

    # Recode ablation_laterality_pecclinical (1-->left, 2-->right, 3-->bilateral) and rename to ablation_laterality
    ablation_laterality_mapping = {
        1: 'left',
        2: 'right',
        3: 'bilateral'
    }
    merged_df['ablation_laterality_pecclinical'] = merged_df['ablation_laterality_pecclinical'].map(ablation_laterality_mapping)
    merged_df.rename(columns={'ablation_laterality_pecclinical': 'ablation_laterality'}, inplace=True)

    # Recode ablation_target_pecclinical (1--> mesial_temporal, 2--> other) and rename to ablation_target
    ablation_target_mapping = {
        1: 'mesial_temporal',
        2: 'other'
    }
    merged_df['ablation_target_pecclinical'] = merged_df['ablation_target_pecclinical'].map(ablation_target_mapping)
    merged_df.rename(columns={'ablation_target_pecclinical': 'ablation_target'}, inplace=True)

    # For cases where ablation_target is other, add values from ablation_target_other_pecclinical. then drop ablation_target_other_pecclinical
    merged_df['ablation_target'] = merged_df.apply(
        lambda row: f"{row['ablation_target_other_pecclinical']}" if row['ablation_target'] == 'other' else row['ablation_target'],
        axis=1
    )
    merged_df = merged_df.drop(columns=['ablation_target_other_pecclinical'])

    # Merge ablation_laterality into intervention_laterality if intervention_laterality is null and drop ablation_laterality
    merged_df['intervention_laterality'] = merged_df['intervention_laterality'].fillna(merged_df['ablation_laterality'])
    merged_df = merged_df.drop(columns=['ablation_laterality'])

    # Recode mri_result_pecclinical (1-->Normal, 2-->Nonspecific Abnormality, 3-->Potential Epileptogenic Lesion, 4-->Not Indicated/Not Performed) and rename to mri_result
    if 'mri_result_pecclinical' in merged_df.columns:
        mri_result_mapping = {
            1: 'nonlesional',
            2: 'lesion_incidental',
            3: 'lesion_potential_epileptogenic',
            4: 'not_indicated'
        }
        merged_df['mri_result'] = merged_df['mri_result_pecclinical'].map(mri_result_mapping)
        merged_df = merged_df.drop(columns=['mri_result_pecclinical'])

    # Rename MRI lesion type checkbox columns (keep as 0/1)
    lesion_renames = {
        'mri_lesion_type_pecclinical___1': 'lesion_tbi',
        'mri_lesion_type_pecclinical___2': 'lesion_stroke',
        'mri_lesion_type_pecclinical___3': 'lesion_fcd',
        'mri_lesion_type_pecclinical___4': 'lesion_neoplasia',
        'mri_lesion_type_pecclinical___5': 'lesion_mts',
        'mri_lesion_type_pecclinical___6': 'lesion_other'
    }
    for old_name, new_name in lesion_renames.items():
        if old_name in merged_df.columns:
            merged_df.rename(columns={old_name: new_name}, inplace=True)

    # In cases where intervention_type == no_intervention, if resection_type is not None, set intervention_type to resection
    merged_df['intervention_type'] = merged_df.apply(
        lambda row: 'resection' if row['resection_type'] is not None and row['intervention_type'] == 'no_intervention' else row['intervention_type'],
        axis=1
    )

    # Create intervention_lobe column by the following logic (in order of precedence):
    # If resection_type contains the substrings "temporal", "frontal",or "insular", set intervention_lobe to "temporal", "frontal", or "insular"
    # If ablation_target contains the substring "temporal", "frontal", or "insular", set intervention_lobe to "temporal", "frontal", or "insular"
    # If ablation target contains the substring "hippocampus", set intervention_lobe to "temporal"
    # Else, leave intervention_lobe as None
    def get_intervention_lobe(row):
        # Check resection_type for lobe keywords
        if pd.notnull(row.get('resection_type')):
            resection_type = str(row['resection_type']).lower()
            if 'temporal' in resection_type:
                return 'temporal'
            if 'frontal' in resection_type:
                return 'frontal'
            if 'insular' in resection_type:
                return 'insular'
        # Check ablation_target for lobe keywords
        if pd.notnull(row.get('ablation_target')):
            ablation_target = str(row['ablation_target']).lower()
            if 'temporal' in ablation_target:
                return 'temporal'
            if 'frontal' in ablation_target:
                return 'frontal'
            if 'insular' in ablation_target:
                return 'insular'
            if 'hippocamp' in ablation_target:
                return 'temporal'
        return None

    merged_df['intervention_lobe'] = merged_df.apply(get_intervention_lobe, axis=1)

    # Reorder columns (include optional MRI columns if present)
    base_cols = ['sub', 'sex', 'age', 'age_onset', 'seizure_lateralization', 'seizure_localization', 'intervention_laterality', 'intervention_lobe', 'intervention_type', 'resection_type', 'resection_histopathology', 'resection_details', 'ablation_target']
    mri_cols = ['mri_result', 'lesion_tbi', 'lesion_stroke', 'lesion_fcd', 'lesion_neoplasia', 'lesion_mts', 'lesion_other']
    outcome_cols = ['engel_followup1', 'ilae_category_pecclinical', 'months_followup1', 'engel_followup2', 'ilae_category_2_pecclinical', 'months_followup2', 'engel_followup3', 'ilae_category_3_pecclinical', 'months_followup3']
    reorder_cols = base_cols + [c for c in mri_cols if c in merged_df.columns] + outcome_cols
    merged_df = merged_df[[c for c in reorder_cols if c in merged_df.columns]] 

    # Save output
    if output_file:
        merged_df.to_csv(output_file, index=False)
        print(f"Merged data saved to: {output_file}")
    else:
        print("\nFirst 10 rows of merged data:")
        print(merged_df.head(10))

    return merged_df

if __name__ == "__main__":
    cohort_file = "{project_root()}/derivatives/metadata/subjects_penn_epilepsy.csv"
    redcap_file = "{project_root()}/data/metadata/clinical_redcap.csv"
    output_file = "{project_root()}/derivatives/metadata/clinical_penn_epilepsy.csv"
    
    merged_data = merge_cohort_redcap(cohort_file, redcap_file, output_file)