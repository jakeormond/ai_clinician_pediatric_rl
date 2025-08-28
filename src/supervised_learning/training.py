import os
import yaml
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
from .evaluation import calculate_metrics
from .plotting import create_all_plots, create_aggregated_shap_plots
from ..utils.logging_utilities import wandb_init
from .dataset import Dataset


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


def prepare_param_for_pipeline(config_manager):
    # add "model__" prefix to all parameters in the grid
    if config_manager.search_variant == "grid_search":
        # For grid search, use the exact values from the config
        return {f"model__{k}": v for k, v in config_manager.search_grid.items() if k != "model__n_estimators"}
    
    elif config_manager.search_variant == "random_search":
        pipeline_params = {}
        for param_name, param_config in config_manager.search_grid.items():
            param_key = f"model__{param_name}"
            
            if isinstance(param_config, dict):
                # Handle distribution specifications
                if param_config['type'] == 'randint':
                    pipeline_params[param_key] = randint(param_config['low'], param_config['high'])
                elif param_config['type'] == 'uniform':
                    if 'low' in param_config and 'high' in param_config:
                        pipeline_params[param_key] = uniform(param_config['low'], 
                                                           param_config['high'] - param_config['low'])
                    else:
                        # Old format with loc/scale
                        pipeline_params[param_key] = uniform(param_config['loc'], param_config['scale'])
                elif param_config['type'] == 'loguniform':
                    pipeline_params[param_key] = loguniform(param_config['low'], param_config['high'])
                elif param_config['type'] == 'choice':
                    pipeline_params[param_key] = param_config['values']
                else:
                    raise ValueError(f"Unknown distribution type: {param_config['type']}")
            else:
                # Handle simple list values (for backwards compatibility)
                pipeline_params[param_key] = param_config
        
        return pipeline_params
    
    else:
        raise ValueError(f"Unknown search variant: {config_manager.search_variant}")
        # 

def remove_model_prefix_from_params(best_params):
    """
    Remove the 'model__' prefix from the parameter names.
    This is necessary because the GridSearchCV returns parameters with 'model__' prefix.
    """
    return {k.replace('model__', ''): v for k, v in best_params.items() if k.startswith('model__')}


def outer_training_loop(dataset, config_manager):
   
    n_classes = len(np.unique(dataset.dataframe[config_manager.hospital_config["target_col"]]))

    # prepare the parameters for the pipeline
    pipe_grid = prepare_param_for_pipeline(config_manager)

    print(f"Number of unique encounter IDs: {len(dataset.encounter_ids())}")

    # Get CV configuration from config
    n_outer_folds = config_manager.cv_config["n_outer_folds"]
    n_inner_folds = config_manager.cv_config["n_inner_folds"]
    
    print(f"Using {n_outer_folds} outer folds and {n_inner_folds} inner folds") 
    
    ######################################### OUTER SPLITS #################################################
       
    # the outer test splits are held out for the final prediction
    # the outer train splits are used for hyperparameter tuning 
    kf_outer = KFold(n_splits=n_outer_folds, 
                     shuffle=True, random_state=config_manager.hospital_config["seed"])

    target_fold = config_manager.cv_config['fold']
    encounter_col = config_manager.hospital_config["encounter_col"]
    time_col = config_manager.hospital_config["time_col"]

    for outer_fold, (outer_train_id, outer_test_id) in enumerate(kf_outer.split(dataset.encounter_ids())):
        if target_fold is not None and outer_fold != target_fold:
            print(f"Skipping outer fold {outer_fold} as it is not the target fold {target_fold}")
            continue

        outer_train_encounters = dataset.encounter_ids()[outer_train_id]
        outer_test_encounters = dataset.encounter_ids()[outer_test_id]

        print(f"Outer Loop Fold {outer_fold}: Train outer encounters={len(outer_train_encounters)}, "
              f"Test outer encounters={len(outer_test_encounters)}")
        
        ################################## INNER SPLITS AND HYPERPARAMETER TUNING ##################################
        # the outer train data is used to create inner training and validation splits for hyperparameter tuning
        X_outer_train, y_outer_train, encounters_and_times = dataset.modeling_data(outer_train_encounters)

        # make the groups for the GroupKFold splits
        groups = encounters_and_times[encounter_col].values

        # Inner CV for hyperparameter tuning - Note that GroupKFold is deterministic 
        # (and does not allow setting of a random seed), so we will produce the 
        # same splits every time
        kf_inner = GroupKFold(n_splits=n_inner_folds)

        if config_manager.model_type == "xgbclassifier":
            model = xgb.XGBClassifier(
                eval_metric='mlogloss', 
                verbosity=2
            )
        elif config_manager.model_type == "xgbregressor":
            model = xgb.XGBRegressor(
                eval_metric='rmse', 
                verbosity=2
            )
        else:
            raise ValueError(f"Unsupported model type: {config_manager.model_type}")        
        
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])        

        # grid search for hyper
        if config_manager.model_type == "xgbclassifier":
        # Check number of classes first
            if n_classes == 2:
                scoring_metric = 'average_precision'  # Binary classification
            else:
                scoring_metric = 'f1_macro'  # Multiclass classification
                
        elif config_manager.model_type == "xgbregressor":
            scoring_metric = 'neg_mean_squared_error'
        
        n_jobs = get_safe_n_jobs()
        print(f"Using {n_jobs} processes for hyperparameter tuning")

        if config_manager.search_variant == "grid_search":
            grid_search = GridSearchCV(
                pipe, 
                pipe_grid, 
                cv=kf_inner, 
                scoring=scoring_metric, 
                n_jobs=n_jobs, # use all available CPU cores
                verbose=2
            )
        elif config_manager.search_variant == "random_search":
            grid_search = RandomizedSearchCV(
                pipe, 
                pipe_grid, 
                n_iter=config_manager.n_search_iterations,
                cv=kf_inner, 
                scoring=scoring_metric, 
                n_jobs=n_jobs, # use all available CPU cores
                verbose=2,
                random_state=config_manager.hospital_config["seed"]
            )

        if config_manager.model_type in ["xgbclassifier", "xgbregressor"] and n_classes > 2:
                # Define your class weights
                class_weights = {0: 5.0, 1: 1.0, 2: 1.0}  # Boost class 0
                sample_weights = np.array([class_weights[int(label)] for label in y_outer_train])
        
                print(f"Class distribution: {np.bincount(y_outer_train.astype(int))}")
                print(f"Using class weights: {class_weights}")
                print(f"Sample weights range: [{sample_weights.min():.2f}, {sample_weights.max():.2f}]")    

                # For multiclass, we need to pass the sample weights
                grid_search.fit(X_outer_train, y_outer_train, groups=groups, 
                                        model__sample_weight=sample_weights)
        else:
            # For binary classification or regression, we can fit without sample weights
            grid_search.fit(X_outer_train, y_outer_train, groups=groups)

        # Get best hyperparameters from inner CV
        best_params = remove_model_prefix_from_params(grid_search.best_params_)
        print(f"Best hyperparameters for outer fold {outer_fold}: {best_params}")

        # Retrain model on full outer train set with best hyperparameters
        if config_manager.model_type == "xgbclassifier":
            best_model = xgb.XGBClassifier(**best_params, eval_metric='mlogloss')
        elif config_manager.model_type == "xgbregressor":
            best_model = xgb.XGBRegressor(**best_params, eval_metric='rmse')
        else:
            raise ValueError(f"Unsupported model type: {config_manager.model_type}")
        
        # scale X_outer_train, and output the parameters so they can be used to scale the test set
        scaler = StandardScaler()
        X_outer_train_scaled = scaler.fit_transform(X_outer_train)        
        
        if config_manager.model_type in ["xgbclassifier", "xgbregressor"] and n_classes > 2:
            # For multiclass, we need to pass the sample weights
            best_model.fit(X_outer_train_scaled, y_outer_train, 
                           sample_weight=sample_weights)
        else:
            # For binary classification or regression, we can fit without sample weights
            best_model.fit(X_outer_train_scaled, y_outer_train)

        # Generate predictions on the outer test set
        X_outer_test, y_outer_test, outer_test_encounters_and_times = \
                                dataset.modeling_data(outer_test_encounters)
        X_outer_test_scaled = scaler.transform(X_outer_test)
        
        # get predictions for the outer test set
        if config_manager.model_type == "xgbclassifier":
            y_pred = best_model.predict(X_outer_test_scaled)
            y_pred_proba_raw = best_model.predict_proba(X_outer_test_scaled)

            # Handle multiclass vs binary
            if n_classes == 2:
                # Binary classification: use positive class probabilities
                y_pred_proba = y_pred_proba_raw[:, 1]
            else:
                # Multiclass: use max probability or specific class
                y_pred_proba = y_pred_proba_raw[np.arange(len(y_outer_test)), y_outer_test.astype(int)]
                # Or for specific class: y_pred_proba = y_pred_proba_raw[:, 2]  # Class 2 probabilities
                
        elif config_manager.model_type == "xgbregressor":
            # y_pred = None
            # y_pred_proba = best_model.predict(X_outer_test_scaled)

            y_pred_continuous = best_model.predict(X_outer_test_scaled)

            max_class = n_classes - 1   
            
            y_pred = np.round(np.clip(y_pred_continuous, 0, max_class)).astype(int)  # Round to nearest class
            y_pred_proba = y_pred_continuous  # Use continuous values as "probabilities"

        # calculate metrics on the outer test set
        # calculate metrics on the outer test set
        if config_manager.model_type == "xgbclassifier" and y_pred_proba_raw.shape[1] > 2:
            # Pass full probability matrix for multiclass
            metrics = calculate_metrics(y_outer_test, y_pred_proba, 
                        config_manager.model_type, y_pred_proba_raw=y_pred_proba_raw)
        else:
            # Binary classification or regression
            metrics = calculate_metrics(y_outer_test, y_pred_proba, config_manager.model_type)
        print(f"Outer Fold {outer_fold}: Test metrics: {metrics}")

        # save to the fold directory        
        with open(os.path.join(config_manager.fold_dir, f"metrics_fold_{outer_fold}.json"), "w") as f:
            serializable_metrics = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                          for k, v in metrics.items()}
            json.dump(serializable_metrics, f)

        # create a DataFrame to hold the predictions and true labels
        predictions_df = pd.DataFrame({
            'encounter_id': outer_test_encounters_and_times[encounter_col],
            'time': outer_test_encounters_and_times[time_col],
            'true_label': y_outer_test,
            'predicted_label': y_pred,
            'predicted_proba': y_pred_proba
        })

        # if xgbclassifier and multiclass, we need to save the full probability matrix
        if config_manager.model_type == "xgbclassifier" and y_pred_proba_raw.shape[1] > 2:
            # Add the full probability matrix to the DataFrame
            for i in range(y_pred_proba_raw.shape[1]):
                predictions_df[f'predicted_proba_class_{i}'] = y_pred_proba_raw[:, i]

        # save the predictions to a parquet file
        predictions_df.to_parquet(os.path.join(config_manager.fold_dir, 
            f"predictions_fold_{outer_fold}.parquet"), engine='pyarrow', index=False)

        # save the models
        joblib.dump(scaler, os.path.join(config_manager.fold_dir, 
                                         f"scaler_fold_{outer_fold}.joblib"))
        joblib.dump(best_model, os.path.join(config_manager.fold_dir, 
                                             f"model_fold_{outer_fold}.joblib"))

        # Save hyperparameters
        with open(os.path.join(config_manager.fold_dir, f"hyperparams_fold_{outer_fold}.json"), "w") as f:
            json.dump(best_params, f)


        # RETURN SUMMARY (for calling script to log)
        results = {
            'fold': outer_fold,
            'best_cv_score': grid_search.best_score_,
            'best_params': best_params,
            'test_metrics': metrics,
            'n_train_samples': len(y_outer_train),
            'n_test_samples': len(y_outer_test)
        }

        # Log feature importance if available - NOT IMPLEMENTED YET
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = best_model.feature_importances_
            wandb.log({
                f"fold_{outer_fold}/feature_importance": wandb.Histogram(feature_importance)
            }, step=outer_fold)
        
        break

    if results is None:
        raise ValueError(f"No matching fold found for target_fold={config_manager.cv_config['fold']}")
    
    return results


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

    # convert y_pred_proba_raw to health score if 
    # config.prediction_type == "discharge_mortality" and
    # config.model_type == "xgbclassifier". For 2 class or 
    # regression, this is not needed
    if config.prediction_type == "discharge_mortality" and \
       config.model_type == "xgbclassifier":
        
        # health score is the sum of column probabilities 
        # multiplied by the ith column index (0, 1, 2) 
        health_score = np.sum(y_pred_proba_raw * np.arange(y_pred_proba_raw.shape[1]), axis=1)
        all_predictions_df['health_score'] = health_score

    else:
        pass

    # save all_predictions_df to a parquet file
    all_predictions_df.to_parquet(config.experiment_dir / "all_predictions.parquet", 
                                  engine='pyarrow', index=False)
    
    # also save config 
    with open(config.experiment_dir / "config.pkl", "wb") as f:
        pickle.dump(config, f)

    # calculate overall metrics
    overall_metrics = calculate_metrics(all_true_labels, \
            all_pred_proba, config.model_type, y_pred = all_predictions, y_pred_proba_raw=y_pred_proba_raw)
    print(f"Overall metrics: {overall_metrics}")
    
    with open(config.experiment_dir / "overall_metrics.json", "w") as f:
        serializable_metrics = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                            for k, v in overall_metrics.items()}
        json.dump(serializable_metrics, f, indent=2)

    # plot model results
    plot_model_results(all_predictions_df, config)
    
    # SHAP calculation
    if config.calculate_shap:
        print("\n=== Calculating SHAP Values ===")
        
        shap_results = calculate_shap_for_aggregated_results(config, all_test_data)
            
        shap_save_path = config.experiment_dir / "shap_analysis.pkl"
        with open(shap_save_path, 'wb') as f:
            pickle.dump(shap_results, f)
        print(f"✓ SHAP results saved to {shap_save_path}") 


def plot_model_results(all_predictions_df, config):

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
    
    n_classes = len(np.unique(all_true_labels))
    is_classification = config.model_type == "xgbclassifier"
    
    if is_classification and n_classes > 2:
        # You'll need to reconstruct the full probability matrix from your saved data
        # This depends on how you saved the individual class probabilities
        
        create_all_plots(
            all_true_labels, 
            all_predictions, 
            all_pred_proba, 
            config.experiment_dir,
            is_classification=is_classification,
            y_pred_proba_raw=y_pred_proba_raw
        )
    else:
        create_all_plots(
            all_true_labels, 
            all_predictions, 
            all_pred_proba, 
            config.experiment_dir,
            is_classification=is_classification
        )



def plot_saved_shap_results(shap_results_path):
    """
    Load and plot previously saved SHAP results.
    
    Args:
        shap_results_path: Path to the saved SHAP results pickle file
    """
    with open(shap_results_path, 'rb') as f:
        shap_data = pickle.load(f)
    
    config = shap_data['config']['config']
    all_shap_results = shap_data['fold_results']
    feature_names = shap_data['feature_names']
    
    print(f"Loaded SHAP results from {shap_results_path}")
    print(f"Configuration: {config}")
    print(f"Number of folds in SHAP results: {len(all_shap_results)}")
    print(f"Number of features: {len(feature_names)}")

    # Create aggregated SHAP plots
    print("Creating aggregated SHAP plots...")
    create_aggregated_shap_plots(config, all_shap_results, feature_names)


    
def calculate_shap_for_aggregated_results(config, all_test_data): 
    
    """
    Calculate SHAP values for all models using the fold_data structure.
    
    Args:
        config: Configuration object
        all_test_data: List of fold_data dictionaries with 'fold', 'model', 'scaler', 'predictions_df'
    """
    import time

    # Get parameters from config - WHAT CONFIG ARE THESE SET IN?
    max_samples_per_fold = getattr(config, 'shap_max_samples', None)  # None = use all
    background_samples = getattr(config, 'shap_background_samples', 100)  # Default 100
    
    print(f"SHAP Configuration:")
    print(f"  Max samples per fold: {max_samples_per_fold if max_samples_per_fold else 'All samples'}")
    print(f"  Background samples: {background_samples if background_samples else 'None'}")
       
    dataset = Dataset(config.data_config)

    # Recreate the same CV splits as in training
    kf_outer = KFold(n_splits=config.cv_config["n_outer_folds"], 
                     shuffle=True, random_state=config.data_config["seed"])

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
        X_test, y_test, _ = dataset.modeling_data(test_encounters, split_x_y=True)
        
        # Get feature names (only need to do this once)
        if feature_names is None:
            feature_names = X_test.columns.tolist()
            print(f"    Found {len(feature_names)} features")
        
        # ✅ OPTIMIZATION 1: Create background dataset (once per fold)
        if background_samples and background_samples > 0 and background_data is None:
            print(f"    Creating background dataset with {background_samples} samples...")
            train_encounters = fold_to_train_encounters[fold]
            X_bg, _, _ = dataset.modeling_data(train_encounters, split_x_y=True)
            
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
            'background_samples': background_samples,
            'config': config
        }
    }