import os
import yaml
import json
import joblib
import numpy as np
import pandas as pd
import wandb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, GroupKFold
import xgboost as xgb
from .evaluation import calculate_metrics
from .plotting import plot_roc_curve, plot_precision_recall_curve, plot_calibration_curve, plot_confusion_matrix
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


def outer_training_loop(dataset, config, experiment_dir, model_type):

    if model_type == "xgb":
        with open("config/xgboost_grid_search_test.yaml") as f:
            param_grid = yaml.safe_load(f)
    else: # raise not implemented error for unsupported model types
        raise NotImplementedError(f"Model type {config['model_type']} is not supported.")
    
    # save a copy of param_grid to the experiment directory
    with open(os.path.join(experiment_dir, "param_grid.json"), "w") as f:
        json.dump(param_grid, f)

    # prepare the parameters for the pipeline
    pipe_grid = prepare_param_for_pipeline(param_grid)

    print(f"Number of unique encounter IDs: {len(dataset.encounter_ids())}")

    # Get CV configuration from config
    cv_config = config["cv_config"]
    n_outer_folds = cv_config["n_outer_folds"]
    n_inner_folds = cv_config["n_inner_folds"]
    
    print(f"Using {n_outer_folds} outer folds and {n_inner_folds} inner folds") 
    
    # Log hyperparameter grid to W&B
    if wandb.run is not None:
        wandb.config.update({"param_grid": param_grid})

    ######################################### OUTER SPLITS #################################################
    
    # Initialize list to collect all prediction DataFrames
    all_predictions_dfs = []
    
    # the outer test splits are held out for the final prediction
    # the outer train splits are used for hyperparameter tuning 
    kf_outer = KFold(n_splits=n_outer_folds, shuffle=True, random_state=config["seed"])

    for outer_fold, (outer_train_id, outer_test_id) in enumerate(kf_outer.split(dataset.encounter_ids())):
        outer_train_encounters = dataset.encounter_ids()[outer_train_id]
        outer_test_encounters = dataset.encounter_ids()[outer_test_id]

        print(f"Outer Loop Fold {outer_fold}: Train outer encounters={len(outer_train_encounters)}, "
              f"Test outer encounters={len(outer_test_encounters)}")
        
        ################################## INNER SPLITS AND HYPERPARAMETER TUNING ##################################
        # the outer train data is used to create inner training and validation splits for hyperparameter tuning
        X_outer_train, y_outer_train, encounters_and_times = dataset.modeling_data(outer_train_encounters)

        # make the groups for the GroupKFold splits
        groups = encounters_and_times[config["encounter_col"]].values

        # Inner CV for hyperparameter tuning - Note that GroupKFold is deterministic 
        # (and does not allow setting of a random seed), so we will produce the 
        # same splits every time
        kf_inner = GroupKFold(n_splits=n_inner_folds)

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', xgb.XGBClassifier(eval_metric='logloss', verbosity=1))
        ])
            
        # grid search for hyper
        grid_search = GridSearchCV(pipe, pipe_grid, cv=kf_inner, scoring='average_precision', verbose=2)
        grid_search.fit(X_outer_train, y_outer_train, groups=groups)

        # Get best hyperparameters from inner CV
        best_params = remove_model_prefix_from_params(grid_search.best_params_)
        print(f"Best hyperparameters for outer fold {outer_fold}: {best_params}")

        # Retrain model on full outer train set with best hyperparameters
        # best_model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
        best_model = xgb.XGBClassifier(**best_params, eval_metric='logloss')
        
        # scale X_outer_train, and output the parameters so they can be used to scale the test set
        scaler = StandardScaler()
        X_outer_train_scaled = scaler.fit_transform(X_outer_train)        
        
        best_model.fit(X_outer_train_scaled, y_outer_train)

        # Generate predictions on the outer test set
        X_outer_test, y_outer_test, outer_test_encounters_and_times = \
                                dataset.modeling_data(outer_test_encounters)
        X_outer_test_scaled = scaler.transform(X_outer_test)
        
        # get predictions for the outer test set
        y_pred = best_model.predict(X_outer_test_scaled)
        y_pred_proba = best_model.predict_proba(X_outer_test_scaled)[:, 1]

        # calculate metrics on the outer test set
        metrics = calculate_metrics(y_outer_test, y_pred, y_pred_proba)
        print(f"Outer Fold {outer_fold}: Test metrics: {metrics}")

        with open(os.path.join(experiment_dir, f"metrics_fold_{outer_fold}.json"), "w") as f:
            json.dump(metrics, f)

        # create a DataFrame to hold the predictions and true labels
        predictions_df = pd.DataFrame({
            'encounter_id': outer_test_encounters_and_times[config["encounter_col"]],
            'time': outer_test_encounters_and_times[config["time_col"]],
            'true_label': y_outer_test,
            'predicted_label': y_pred,
            'predicted_proba': y_pred_proba
        })

        # save the predictions to a parquet file
        predictions_df.to_parquet(os.path.join(experiment_dir, f"predictions_fold_{outer_fold}.parquet"), 
                                  engine='pyarrow', index=False)

        # save the models
        joblib.dump(scaler, os.path.join(experiment_dir, f"scaler_fold_{outer_fold}.joblib"))
        joblib.dump(best_model, os.path.join(experiment_dir, f"model_fold_{outer_fold}.joblib"))

        # Save hyperparameters
        with open(os.path.join(experiment_dir, f"hyperparams_fold_{outer_fold}.json"), "w") as f:
            json.dump(best_params, f)

        # Log metrics to W&B
        if wandb.run is not None:
            wandb.log({
                f"fold_{outer_fold}/best_cv_score": grid_search.best_score_,
                f"fold_{outer_fold}/best_params": best_params,
                **{f"fold_{outer_fold}/{k}": v for k, v in metrics.items()}
            }, step=outer_fold)

        # Log feature importance if available - NOT IMPLEMENTED YET
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = best_model.feature_importances_
            wandb.log({
                f"fold_{outer_fold}/feature_importance": wandb.Histogram(feature_importance)
            }, step=outer_fold)

        if outer_fold == 0:
            # create the arrays to hold the predictions and test scores
            all_true_labels = y_outer_test
            all_predictions = y_pred
            all_pred_proba = y_pred_proba
        
        else:
            # append the predictions and test scores to the arrays
            all_true_labels = np.concatenate((all_true_labels, y_outer_test))
            all_predictions = np.concatenate((all_predictions, y_pred))
            all_pred_proba = np.concatenate((all_pred_proba, y_pred_proba))

    # calculate overall metrics
    overall_metrics = calculate_metrics(all_true_labels, all_predictions, all_pred_proba)
    print(f"Overall metrics: {overall_metrics}")
    
    with open(os.path.join(experiment_dir, "overall_metrics.json"), "w") as f:
        json.dump(overall_metrics, f)

    # plot ROC curve, precision-recall curve, calibration curve, and confusion matrix
    plot_roc_curve(all_true_labels, all_pred_proba, experiment_dir)
    plot_precision_recall_curve(all_true_labels, all_pred_proba, experiment_dir)
    plot_calibration_curve(all_true_labels, all_pred_proba, experiment_dir)
    plot_confusion_matrix(all_true_labels, all_predictions, experiment_dir)

    # Log overall metrics to W&B
    if wandb.run is not None:
        wandb.log({f"overall/{k}": v for k, v in overall_metrics.items()})

        # Log plots
        plot_files = ["roc_curve.png", "precision_recall_curve.png", 
                     "calibration_curve.png", "confusion_matrix.png"]
        
        for plot_file in plot_files:
            plot_path = os.path.join(experiment_dir, plot_file)
            if os.path.exists(plot_path):
                wandb.log({plot_file.replace('.png', ''): wandb.Image(plot_path)})
        
        wandb.finish()