import os
import yaml
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, GroupKFold
import xgboost as xgb
from .evaluation import calculate_metrics
from .plotting import plot_roc_curve, plot_precision_recall_curve, plot_calibration_curve, plot_confusion_matrix



def prepare_param_for_pipeline(param_grid):
    # add "model__" prefix to all parameters in the grid
    return {f"model__{k}": v for k, v in param_grid.items() if k != "model__n_estimators"}


def remove_model_prefix_from_params(best_params):
    """
    Remove the 'model__' prefix from the parameter names.
    This is necessary because the GridSearchCV returns parameters with 'model__' prefix.
    """
    return {k.replace('model__', ''): v for k, v in best_params.items() if k.startswith('model__')}


def outer_training_loop(dataset, config, experiment_dir, model_type="xgb"):

    if model_type == "xgb":
        with open("config/xgboost_grid_search_test.yaml") as f:
            param_grid = yaml.safe_load(f)
    
    # save a copy of param_grid to the experiment directory
    with open(os.path.join(experiment_dir, "param_grid.json"), "w") as f:
        json.dump(param_grid, f)

    # prepare the parameters for the pipeline
    pipe_grid = prepare_param_for_pipeline(param_grid)

    print(f"Number of unique encounter IDs: {len(dataset.encounter_ids())}")
    
    
    ######################################### OUTER SPLITS #################################################
    # the outer test splits are held out for the final prediction
    # the outer train splits are used for hyperparameter tuning 
    kf_outer = KFold(n_splits=5, shuffle=True, random_state=config["seed"])

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
        kf_inner = GroupKFold(n_splits=5)

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
    pass