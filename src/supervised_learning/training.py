import os
import yaml
import json
import joblib
import glob
import numpy as np
import pandas as pd
import wandb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, GroupKFold
import xgboost as xgb
from .evaluation import calculate_metrics
from .plotting import create_all_plots
from ..utils.logging_utilities import wandb_init


def prepare_param_for_pipeline(param_grid):
    # add "model__" prefix to all parameters in the grid
    return {f"model__{k}": v for k, v in param_grid.items() if k != "model__n_estimators"}


def remove_model_prefix_from_params(best_params):
    """
    Remove the 'model__' prefix from the parameter names.
    This is necessary because the GridSearchCV returns parameters with 'model__' prefix.
    """
    return {k.replace('model__', ''): v for k, v in best_params.items() if k.startswith('model__')}


def outer_training_loop(dataset, config_manager):
   

    # prepare the parameters for the pipeline
    pipe_grid = prepare_param_for_pipeline(config_manager.search_grid)

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
                eval_metric='logloss', 
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
        grid_search = GridSearchCV(
            pipe, 
            pipe_grid, 
            cv=kf_inner, 
            scoring='average_precision', 
            n_jobs=-1, # use all available CPU cores
            verbose=2
        )
        grid_search.fit(X_outer_train, y_outer_train, groups=groups)

        # Get best hyperparameters from inner CV
        best_params = remove_model_prefix_from_params(grid_search.best_params_)
        print(f"Best hyperparameters for outer fold {outer_fold}: {best_params}")

        # Retrain model on full outer train set with best hyperparameters
        if config_manager.model_type == "xgclassifier":
            best_model = xgb.XGBClassifier(**best_params, eval_metric='logloss')
        elif config_manager.model_type == "xgbregressor":
            best_model = xgb.XGBRegressor(**best_params, eval_metric='rmse')
        else:
            raise ValueError(f"Unsupported model type: {config_manager.model_type}")
        
        # scale X_outer_train, and output the parameters so they can be used to scale the test set
        scaler = StandardScaler()
        X_outer_train_scaled = scaler.fit_transform(X_outer_train)        
        
        best_model.fit(X_outer_train_scaled, y_outer_train)

        # Generate predictions on the outer test set
        X_outer_test, y_outer_test, outer_test_encounters_and_times = \
                                dataset.modeling_data(outer_test_encounters)
        X_outer_test_scaled = scaler.transform(X_outer_test)
        
        # get predictions for the outer test set
        if config_manager.model_type == "xgbclassifier":
            y_pred = best_model.predict(X_outer_test_scaled)
            y_pred_proba = best_model.predict_proba(X_outer_test_scaled)[:, 1]
        elif config_manager.model_type == "xgbregressor":
            y_pred = None
            y_pred_proba = best_model.predict(X_outer_test_scaled)

        # calculate metrics on the outer test set
        metrics = calculate_metrics(y_outer_test, y_pred_proba)
        print(f"Outer Fold {outer_fold}: Test metrics: {metrics}")

        # save to the fold directory        
        with open(os.path.join(config_manager.fold_dir, f"metrics_fold_{outer_fold}.json"), "w") as f:
            json.dump(metrics, f)

        # create a DataFrame to hold the predictions and true labels
        predictions_df = pd.DataFrame({
            'encounter_id': outer_test_encounters_and_times[encounter_col],
            'time': outer_test_encounters_and_times[time_col],
            'true_label': y_outer_test,
            'predicted_label': y_pred,
            'predicted_proba': y_pred_proba
        })

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

    return results


def aggregate_results_across_outer_folds(config):
    """
    This function is used to combine the results across outer folds.
    It will read the predictions and metrics from each outer fold and combine them into a single DataFrame.
    """
    print(f"Aggregating results from experiment directory: {config.experiment_dir}")
    
    n_folds = config.cv_config["n_outer_folds"]

    # load the parquet files for each fold
    all_predictions_dfs = []
    for fold in range(n_folds):

        # find the folder in the experiment directory that begins with f"fold_{fold}" and ends with some random numbers
        fold_dir = config.experiment_dir / f"fold_{fold}_*"

        # within that folder, find the parquet file that starts with "predictions_fold_" and ends with ".parquet"
        fold_predictions_path = glob.glob(os.path.join(fold_dir, f"predictions_fold_{fold}.parquet"))
        fold_df = pd.read_parquet(fold_predictions_path, engine='pyarrow')
        all_predictions_dfs.append(fold_df)

    # concatenate all predictions DataFrames
    all_predictions_df = pd.concat(all_predictions_dfs, ignore_index=True)
    all_true_labels = all_predictions_df["true_label"].values
    all_predictions = all_predictions_df["predicted_label"].values
    all_pred_proba = all_predictions_df["predicted_proba"].values    

    # calculate overall metrics
    overall_metrics = calculate_metrics(all_true_labels, all_pred_proba)
    print(f"Overall metrics: {overall_metrics}")
    
    with open(config.experiment_dir / "overall_metrics.json", "w") as f:
        json.dump(overall_metrics, f)

    # plot ROC curve, precision-recall curve, calibration curve, and confusion matrix
    if config.model_type == "xgbclassifier":
        is_classification = True
    elif config.model_type == "xgbregressor":
        is_classification = False
    create_all_plots(all_true_labels, all_predictions, all_pred_proba, \
            config.experiment_dir, is_classification=is_classification)





