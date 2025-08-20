import os
import yaml
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import joblib
import glob
import numpy as np
import pandas as pd
import pickle
import shap
import wandb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, GroupKFold, RandomizedSearchCV
from scipy.stats import uniform, randint, loguniform
import xgboost as xgb
# from .evaluation import calculate_metrics
# from .plotting import create_all_plots, create_aggregated_shap_plots
from ..utils.logging_utilities import wandb_init
from ..data.data_processor import DataProcessor
# from .dataset import Dataset


def get_safe_n_jobs():
    """Get safe number of parallel jobs based on environment."""
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        slurm_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))
        # Use half the allocated CPUs for safety
        n_jobs = min(4, max(1, slurm_cpus // 2))
        print(f"SLURM detected: Using n_jobs={n_jobs} (SLURM_CPUS_PER_TASK={slurm_cpus})")
        return n_jobs
    else:
        print("Local environment: Using n_jobs=-1 (all cores)")
        return -1
    

def get_feature_names_from_pipeline(pipeline, dataset):
    """Extract feature names from the fitted pipeline."""
    
    # Get the original feature names from your dataset
    # This depends on your dataset structure
    try:
        # If your dataset has a method to get feature names
        base_feature_names = dataset.get_feature_names()
    except:
        # Fallback: use generic names
        n_features = pipeline.named_steps['scaler'].n_features_in_
        base_feature_names = [f'feature_{i}' for i in range(n_features)]
    
    return base_feature_names



def keep_best_trajectories(
    dataset: Dict[str, np.ndarray],
    frac: float,
    discount: float,
    max_episode_steps: int = 1000,
    ):

    ids_by_trajectories = []
    returns = []
    cur_ids = []
    cur_return = 0
    reward_scale = 1.0
    for i, (reward, done) in enumerate(zip(dataset["rewards"], dataset["terminated"])):
        cur_return += reward_scale * reward
        cur_ids.append(i)
        reward_scale *= discount
        if done == 1.0 or len(cur_ids) == max_episode_steps:
            ids_by_trajectories.append(list(cur_ids))
            returns.append(cur_return)
            cur_ids = []
            cur_return = 0
            reward_scale = 1.0

    sort_ord = np.argsort(returns, axis=0)[::-1].reshape(-1)
    top_trajs = sort_ord[: max(1, int(frac * len(sort_ord)))]
    top_returns = np.array(returns)[top_trajs]
    print(f"mean top returns: {np.mean(top_returns)}")

    order = []
    for i in top_trajs:
        order += ids_by_trajectories[i]
    order = np.array(order)
    dataset["observations"] = dataset["observations"][order]
    dataset["actions"] = dataset["actions"][order]
    dataset["next_observations"] = dataset["next_observations"][order]
    dataset["rewards"] = dataset["rewards"][order]
    dataset["terminated"] = dataset["terminated"][order]

    return dataset


def outer_training_loop(dataset, config_manager): 

    # Get CV configuration from config
    n_outer_folds = config_manager.cv_config["n_outer_folds"]
    n_inner_folds = config_manager.cv_config["n_inner_folds"]
    
    print(f"Using {n_outer_folds} outer folds and {n_inner_folds} inner folds") 

    

    print(f"Number of unique encounter IDs: {len(dataset.encounter_ids())}")

    # Get CV configuration from config
    n_outer_folds = config_manager.cv_config["n_outer_folds"]
    n_inner_folds = config_manager.cv_config["n_inner_folds"]
    
    print(f"Using {n_outer_folds} outer folds and {n_inner_folds} inner folds") 
    
    ######################################### OUTER SPLITS #################################################
       
    # the outer test splits are held out for the final prediction
    # the outer train splits are used for hyperparameter tuning 
    kf_outer = KFold(n_splits=n_outer_folds, 
                     shuffle=True, random_state=config_manager.data_config["seed"])

    target_fold = config_manager.cv_config['fold']
    encounter_col = config_manager.data_config["encounter_col"]
    time_col = config_manager.data_config["time_col"]

    for outer_fold, (outer_train_id, outer_test_id) in enumerate(kf_outer.split(dataset.encounter_ids())):
        if target_fold is not None and outer_fold != target_fold:
            print(f"Skipping outer fold {outer_fold} as it is not the target fold {target_fold}")
            continue

        train_dataset = dataset.create_subset(dataset.encounter_ids()[outer_train_id])
        test_dataset = dataset.create_subset(dataset.encounter_ids()[outer_test_id])
        
        # split the train dataset into train and validation sets
        kf_inner = KFold(n_splits=n_inner_folds, 
                shuffle=True, random_state=config_manager.data_config["seed"])
        
        for inner_fold, (inner_train_id, inner_val_id) in enumerate(kf_inner.split(train_dataset.encounter_ids())): 
            if target_fold is not None and inner_fold != target_fold: # for testing only
                print(f"Skipping inner fold {inner_fold} as it is not the target fold {target_fold}; FOR TESTING ONLY")
                continue

            # Create inner train and validation datasets
            inner_train_dataset = train_dataset.create_subset(train_dataset.encounter_ids()[inner_train_id])
            inner_val_dataset = train_dataset.create_subset(train_dataset.encounter_ids()[inner_val_id])

            # create data_processors from the datasets
            inner_train_processor = DataProcessor(inner_train_dataset)

            pass






        



    


def aggregate_results_across_outer_folds(config):
    """
    This function is used to combine the results across outer folds.
    It will read the predictions and metrics from each outer fold and combine them into a single DataFrame.
    """
    print(f"Aggregating results from experiment directory: {config.experiment_dir}")
    
    n_folds = config.cv_config["n_outer_folds"]

    # load the parquet files for each fold
    all_test_data = []
    
    all_predictions_dfs = []
    all_test_data = []    
    
    
    for fold in range(n_folds):

        # find the folder in the experiment directory that begins with f"fold_{fold}" and ends with some random numbers
        fold_dir = config.experiment_dir / f"fold_{fold}_*"

        # within that folder, find the parquet file that starts with "predictions_fold_" and ends with ".parquet"
        fold_predictions_path = glob.glob(os.path.join(fold_dir, 
                                    f"predictions_fold_{fold}.parquet"))
        fold_df = pd.read_parquet(fold_predictions_path, engine='pyarrow')
        all_predictions_dfs.append(fold_df)

        # Load model and scaler for SHAP
        model_path = fold_dir / f"model_fold_{fold}.joblib"
        scaler_path = fold_dir / f"scaler_fold_{fold}.joblib"

        # Load model and scaler for SHAP
        model_path = glob.glob(os.path.join(fold_dir, 
                                    f"model_fold_{fold}.joblib"))[0]
        scaler_path = glob.glob(os.path.join(fold_dir, 
                                    f"scaler_fold_{fold}.joblib"))[0]

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        # ✅ Store as comprehensive fold data
        fold_data = {
            'fold': fold,
            'model': model,
            'scaler': scaler,
            'predictions_df': fold_df
        }
        
        all_test_data.append(fold_data)  # Reuse this list for comprehensive data            
        print(f"✓ Loaded complete data for fold {fold}")

    # concatenate all predictions DataFrames
    all_predictions_df = pd.concat(all_predictions_dfs, ignore_index=True)
    # sort by encounter_id and time
    all_predictions_df.sort_values(by=['encounter_id', 'time'], inplace=True)
    all_predictions_df.reset_index(drop=True, inplace=True)

    # check if there are any duplicates of encounter_id and time
    duplicates = all_predictions_df[all_predictions_df.
            duplicated(subset=['encounter_id', 'time'], keep=False)]
    if not duplicates.empty:
        # raise an error if there are duplicates
        raise ValueError("Duplicate encounter_id and time found in predictions DataFrame. ")
       
    # extract the true labels, predicted labels, and predicted probabilities
    # Note: y_pred_proba is the continuous prediction for regression, or the probability for classification
    # For multiclass, y_pred_proba_raw will contain the full probability matrix
    # For binary classification, y_pred_proba will be the probability of the positive class
    all_true_labels = all_predictions_df["true_label"].values
    all_predictions = all_predictions_df["predicted_label"].values
    all_pred_proba = all_predictions_df["predicted_proba"].values    

    if config.prediction_type == "discharge_mortality":
        # For multiclass, we need to reconstruct the full probability matrix
        y_pred_proba_raw = all_predictions_df.filter(like='predicted_proba_class_').values
        if y_pred_proba_raw.size == 0:
            y_pred_proba_raw = None
    else:
        y_pred_proba_raw = None

    # calculate overall metrics
    overall_metrics = calculate_metrics(all_true_labels, \
            all_pred_proba, config.model_type, y_pred = all_predictions, y_pred_proba_raw=y_pred_proba_raw)
    print(f"Overall metrics: {overall_metrics}")
    
    with open(config.experiment_dir / "overall_metrics.json", "w") as f:
        serializable_metrics = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                            for k, v in overall_metrics.items()}
        json.dump(serializable_metrics, f, indent=2)

    # make plots
    n_classes = len(np.unique(all_true_labels))
    is_classification = config.model_type == "xgbclassifier"

    # Pass y_pred_proba_raw for multiclass
    ############################# TEMPORARILY DISABLED PLOTS #############################
    # if is_classification and n_classes > 2:
    #     # You'll need to reconstruct the full probability matrix from your saved data
    #     # This depends on how you saved the individual class probabilities
        
    #     create_all_plots(
    #         all_true_labels, 
    #         all_predictions, 
    #         all_pred_proba, 
    #         config.experiment_dir,
    #         is_classification=is_classification,
    #         y_pred_proba_raw=y_pred_proba_raw
    #     )
    # else:
    #     create_all_plots(
    #         all_true_labels, 
    #         all_predictions, 
    #         all_pred_proba, 
    #         config.experiment_dir,
    #         is_classification=is_classification
    #     )

    # save all_predictions_df to a parquet file
    all_predictions_df.to_parquet(config.experiment_dir / "all_predictions.parquet", 
                                  engine='pyarrow', index=False)
    
    # SHAP calculation
    if config.calculate_shap:
        print("\n=== Calculating SHAP Values ===")
        
        shap_results = calculate_shap_for_aggregated_results(config, all_test_data)
            
        shap_save_path = config.experiment_dir / "shap_analysis.pkl"
        with open(shap_save_path, 'wb') as f:
            pickle.dump(shap_results, f)
        print(f"✓ SHAP results saved to {shap_save_path}") 

    
def calculate_shap_for_aggregated_results(config, all_test_data): 
    
    """
    Calculate SHAP values for all models using the fold_data structure.
    
    Args:
        config: Configuration object
        all_test_data: List of fold_data dictionaries with 'fold', 'model', 'scaler', 'predictions_df'
    """
    import time

    # Get parameters from config
    max_samples_per_fold = getattr(config, 'shap_max_samples', None)  # None = use all
    background_samples = getattr(config, 'shap_background_samples', 100)  # Default 100
    
    print(f"SHAP Configuration:")
    print(f"  Max samples per fold: {max_samples_per_fold if max_samples_per_fold else 'All samples'}")
    print(f"  Background samples: {background_samples if background_samples else 'None'}")
       
    dataset = Dataset(config.hospital_config)

    # Recreate the same CV splits as in training
    kf_outer = KFold(n_splits=config.cv_config["n_outer_folds"], 
                     shuffle=True, random_state=config.hospital_config["seed"])

    # Create mapping of fold -> test_encounters
    fold_to_encounters = {}
    fold_to_train_encounters = {}
    for fold, (train_idx, test_idx) in enumerate(kf_outer.split(dataset.encounter_ids())):
        fold_to_encounters[fold] = dataset.encounter_ids()[test_idx]
        fold_to_train_encounters[fold] = dataset.encounter_ids()[train_idx]
    
    all_shap_results = []
    feature_names = None
    background_data = None

    print(f"Processing SHAP for {len(all_test_data)} folds...")
    
    for fold_data in all_test_data:
        fold = fold_data['fold']
        model = fold_data['model'] 
        scaler = fold_data['scaler']
        predictions_df = fold_data['predictions_df']
        
        print(f"  Processing fold {fold}...")

        test_encounters = fold_to_encounters[fold]
        X_test, y_test, _ = dataset.modeling_data(test_encounters)
        
        # Get feature names (only need to do this once)
        if feature_names is None:
            feature_names = X_test.columns.tolist()
            print(f"    Found {len(feature_names)} features")
        
        # ✅ OPTIMIZATION 1: Create background dataset (once per fold)
        if background_samples and background_samples > 0 and background_data is None:
            print(f"    Creating background dataset with {background_samples} samples...")
            train_encounters = fold_to_train_encounters[fold]
            X_bg, _, _ = dataset.modeling_data(train_encounters)
            
            # Sample background data
            if len(X_bg) > background_samples:
                bg_indices = np.random.choice(len(X_bg), size=background_samples, replace=False)
                X_bg_sample = X_bg.iloc[bg_indices]
            else:
                X_bg_sample = X_bg
                
            # Scale background data using the fold's scaler
            background_data = scaler.transform(X_bg_sample)
            print(f"    ✓ Background dataset created: {len(background_data)} samples")
        
        # Scale test data using the fold's scaler
        X_test_scaled = scaler.transform(X_test)
        
        # Verify data alignment
        if len(X_test_scaled) != len(predictions_df):
            raise ValueError(
                f"Data length mismatch: X_test({len(X_test_scaled)}) vs predictions({len(predictions_df)})"
            )

        # ✅ OPTIMIZATION 2: Downsample test data
        n_samples_original = len(X_test_scaled)
        if max_samples_per_fold and n_samples_original > max_samples_per_fold:
            print(f"    Downsampling from {n_samples_original} to {max_samples_per_fold} samples")
            
            # Stratified sampling to maintain class distribution
            sample_indices = np.random.choice(n_samples_original, size=max_samples_per_fold, replace=False)
            X_test_scaled_sample = X_test_scaled[sample_indices]
            y_test_sample = y_test.values[sample_indices]
            n_samples = max_samples_per_fold
        else:
            X_test_scaled_sample = X_test_scaled
            y_test_sample = y_test.values
            n_samples = n_samples_original
        
        print(f"    Calculating SHAP for {n_samples} samples...")

        # Calculate SHAP values using TreeExplainer
        start_time = time.time()
        
        # Verify data alignment
        if len(X_test_scaled) != len(predictions_df):
            print(f"    ⚠️  Data length mismatch: X_test({len(X_test_scaled)}) vs predictions({len(predictions_df)})")
            # Try to align by encounter_id/time if possible
            # This is complex, so skip for now
            continue
               
        if background_data is not None:
            explainer = shap.TreeExplainer(model, background_data)
            print(f"    Using background dataset with {len(background_data)} samples")
        else:
            explainer = shap.TreeExplainer(model)
            print(f"    Using default TreeExplainer (no background)")

        shap_values_fold = explainer.shap_values(X_test_scaled_sample)

        end_time = time.time()
        print(f"    ✓ Fold {fold} SHAP complete in {end_time - start_time:.1f}s ({n_samples} samples)")
               
        # Store results
        fold_result = {
            'fold': fold,
            'shap_values': shap_values_fold,
            'test_data': X_test_scaled_sample,
            'true_labels': y_test_sample,
            'feature_names': feature_names,
            'explainer_expected_value': explainer.expected_value,
            'n_samples': n_samples,
            'n_samples_original': n_samples_original
        }
        
        all_shap_results.append(fold_result)

        
    # Create aggregated SHAP plots
    print("Creating aggregated SHAP plots...")
    create_aggregated_shap_plots(config, all_shap_results, feature_names)

    total_samples = sum(r['n_samples'] for r in all_shap_results)
    total_original = sum(r['n_samples_original'] for r in all_shap_results)
    print(f"✓ SHAP analysis complete: {total_samples} samples processed (out of {total_original} total)")


    return {
        'fold_results': all_shap_results,
        'feature_names': feature_names,
        'n_folds_processed': len(all_shap_results),
        'total_samples': total_samples,
        'total_original_samples': total_original,
        'config': {
            'max_samples_per_fold': max_samples_per_fold,
            'background_samples': background_samples
        }
    }