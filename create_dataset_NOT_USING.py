import os
import pandas as pd


CHOC_CODE_DIR = '/home/jake/Documents/AI_Clinician/choc_code'
FEATURES_TO_USE_DF = pd.read_csv(os.path.join(CHOC_CODE_DIR, 'dataframes', 'features_to_use.csv'))
CHOC_DATA_DIR = '/home/jake/Documents/AI_Clinician/choc_data'

DEMOGRAPHICS_PATH = os.path.join(DATA_DIR, 'clean_data', 'demographics.csv.gz')
DEMOGRAPHICS_DF = pd.read_csv(DEMOGRAPHICS_PATH, compression='gzip')

SEPSIS_DX_PATH = os.path.join(DATA_DIR, 'clean_data', 'ai_icu_primary_dx_sepsis.csv.gz')
SEPSIS_DX_DF = pd.read_csv(SEPSIS_DX_PATH, compression='gzip')

WARD_PATH = os.path.join(DATA_DIR, 'clean_data', 'icu_units.csv.gz')
WARD_DF = pd.read_csv(WARD_PATH, compression='gzip')

COMBINED_DATA_HOURLY_PATH = os.path.join(DATA_DIR, 'combined_data', 'hourly', 
                                         'combined_hourly_features.parquet')
COMBINED_DATA_DAILY_PATH = os.path.join(DATA_DIR, 'combined_data', 'daily',
                                         'combined_daily_features.parquet')

TIME_COL = 'EVENT_END_DT_TM'  # Default time column name
ENCOUNTER_COL = 'MASKED_ENCNTR_ID' 



def load_dataset(type='hourly'):
    if type == 'hourly':
        combined_data_path = COMBINED_DATA_HOURLY_PATH
    elif type == 'daily':
        combined_data_path = COMBINED_DATA_DAILY_PATH
    else:
        raise ValueError("Type must be 'hourly' or 'daily'.")

    if not os.path.exists(combined_data_path):
        raise FileNotFoundError(f"Combined data path {combined_data_path} does not exist.")

    combined_df = pd.read_parquet(combined_data_path)
    
    return combined_df


def select_patients(dataset, cohort='sepsis', encounter_ids = None, ward = None):

    all_patients = set(dataset[ENCOUNTER_COL].unique())    
    n_patients_full_dataset = len(all_patients)

    n_rows_full_dataset = dataset.shape[0]
    
    if cohort == 'sepsis':
              
        # Filter dataset for sepsis patients
        sepsis_patients = set(SEPSIS_DX_DF['MASKED_ENCNTR_ID'].unique())
        
        # sepsis patients must be in all_patients
        sepsis_patients = [pid for pid in sepsis_patients if pid in all_patients]
        
        dataset = dataset[dataset[ENCOUNTER_COL].isin(sepsis_patients)].copy()
        n_rows_cohort = dataset.shape[0]

        n_patients_cohort = dataset[ENCOUNTER_COL].nunique()

        print(f"Number of patients in full dataset: {n_patients_full_dataset}")
        print(f"Number of patients in {cohort} cohort: {n_patients_cohort}\n")
        print(f"Number of rows in full dataset: {n_rows_full_dataset}")
        print(f"Number of rows in {cohort} cohort: {n_rows_cohort}")    

    return dataset


def select_data_by_proportion(dataset, proportion_w_data=0.6, cohort='sepsis'):
    """
    removes features based on the proportion of patients with valid data for 
    a given feature. 
    Note that a number of features do not require valid data, such as drug and 
    ventilation data, since absence of data simply means the patient was not on 
    that drug or not on ventilation. These features are identifed by using
    "carry forward" as the imputation method
    """

    # loop the features_to_use_df
    for i, row in FEATURES_TO_USE_DF.iterrows():
        feature_set = row['feature_set']
        feature_name = row['feature']
        imputation_method = row['imputation_method']

        if imputation_method == 'carry forward':
            continue

        if cohort is None:
            proportion = row['proportion_encounters']
        elif cohort == 'sepsis':
            proportion = row['proportion_sepsis']
        else:
            raise ValueError("Cohort must be 'sepsis' or None.")
        
        if proportion < proportion_w_data:
            # drop the feature from dataset
            print(f"Dropping feature {feature_name} from {cohort} cohort: {proportion:.2f} < {proportion_w_data}")
            dataset.drop(columns=[feature_name], inplace=True)
        
    n_remaining_features = dataset.shape[1] - 2  # Subtract 2 for ENCOUNTER_COL and TIME_COL
    print(f"Number of remaining features after filtering: {n_remaining_features}")

    return dataset


def create_dataset_for_contrastive(type='hourly', cohort='sepsis', 
                    proportion_w_data = 0.6, ward=None, feature_sets = None, 
                    features_to_exclude = None, filename = None):

    """
    Create a dataset by loading the combined data and selecting patients based on the cohort.
    
    Parameters:
    - type: 'hourly' or 'daily'
    - cohort: 'sepsis' or 'all'
    - encounter_ids: list of encounter IDs to filter by
    - ward: specific ward to filter by
    
    Returns:
    - DataFrame containing the selected patients and features.


    For contrastive learning (used to calculate reward), we are using only the following 
    feature sets or features:
    Feature sets:
    - demographics
    - blood_gas
    - labs
    - neuro
    - vitals
    Features:
    - fio2
    - inspiratory_time


    """
    
    feature_sets_to_use = ['demographics', 'blood_gas', 'labs', 'neuro', 'vitals']
    other_features_to_use = ['fio2', 'inspiratory_time']

    dataset = load_dataset(type=type)
    
    dataset = select_patients(dataset, cohort=cohort)

    dataset = select_data_by_proportion(dataset, 
                                      proportion_w_data=proportion_w_data, 
                                      cohort='sepsis')

    # get list of features to keep
    features_to_keep = FEATURES_TO_USE_DF[(FEATURES_TO_USE_DF['feature_set']
                .isin(feature_sets_to_use)) | FEATURES_TO_USE_DF['feature']
                .isin(other_features_to_use)]['feature'].tolist()
    
    # Add ENCOUNTER_COL and TIME_COL to the list of features to keep
    features_to_keep.append(ENCOUNTER_COL)
    features_to_keep.append(TIME_COL)

    # Filter the dataset to keep only the selected features
    # first, get the features taht are actually there
    columns_in_dataset = dataset.columns.tolist()
    features_to_keep = [feature for feature in features_to_keep if feature in columns_in_dataset]

    dataset = dataset[features_to_keep].copy()
    n_remaining_features = dataset.shape[1] - 2  # Subtract 2 for ENCOUNTER_COL and TIME_COL
    print(f"Number of remaining features after filtering: {n_remaining_features}")

    # drop any rows with NaN values anywhere
    dataset.dropna(inplace=True)

    # get the number of remaining rows and patients
    n_rows_remaining = dataset.shape[0]
    n_patients_remaining = dataset[ENCOUNTER_COL].nunique()
    print(f"Number of remaining rows: {n_rows_remaining}")
    print(f"Number of remaining patients: {n_patients_remaining}")


    dataset = dataset.sort_values(by=[ENCOUNTER_COL, TIME_COL])

    dataset_name = f"combined_{type}_features"
    if cohort == 'sepsis':
        dataset_name += '_sepsis'
    if ward is not None:
        dataset_name += f'_{ward}'
    dataset_name += f"_proportion_{proportion_w_data:.2f}"
    dataset_name += '.parquet'

    dataset_path = os.path.join(DATA_DIR, 'combined_data', type, dataset_name)
    dataset.to_parquet(dataset_path, index=False)
    
    return dataset
    

def create_dataset_for_rl(type='hourly', cohort='sepsis', 
                    proportion_w_data = 0.8, ward=None, feature_sets = None, 
                    features_to_exclude = None, filename = "choc_hourly_for_rl_proportion_0.8.parquet"):
    
    """
    need to get observations, actions, rewards, terminations, and next_observations
    """


    dataset = load_dataset(type=type)    
    dataset = select_patients(dataset, cohort=cohort)
    dataset = select_data_by_proportion(dataset, 
                                      proportion_w_data=proportion_w_data, 
                                      cohort='sepsis')

    pass

    return


def main():
    
    type = 'hourly'
    proportion_w_data = 0.8 

    feature_sets = FEATURES_TO_USE_DF['feature_set'].unique().tolist()

    dataset = create_dataset(type='hourly', cohort='sepsis', 
                    proportion_w_data = 0.8, ward=None, feature_sets = None, 
                    features_to_exclude = None, filename = "choc_hourly_for_rl_proportion_0.8.parquet")


    # dataset = create_dataset_for_contrastive(type=type, cohort='sepsis', 
    #                                proportion_w_data = proportion_w_data, ward=None)

    return


if __name__ == '__main__':
    main()



