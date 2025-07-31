"""
Prepares data for supervised learning by creating a dataset class.
Will handle cross validation splits and data normalization.
Splits data into X and y (as opposed to the observations, actions,
and rewards in the RL setting).

Create a DataSet object that holds the full dataset, and then 
CVSplit objects that hold the encounter IDs for the splits
and methods to access the data in the DataSet object.
"""

import argparse
import os
import yaml
import json
import joblib
import pandas as pd
import numpy as np
import time
import pyarrow
from sklearn.model_selection import train_test_split
from utilities import load_config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import KFold, GridSearchCV, GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, \
    precision_score, recall_score, f1_score, average_precision_score
import xgboost as xgb

from config_manager import ConfigManager  # Import the config manager


def prepare_param_for_pipeline(param_grid):
    # add "model__" prefix to all parameters in the grid
    return {f"model__{k}": v for k, v in param_grid.items() if k != "model__n_estimators"}

def remove_model_prefix_from_params(best_params):
    """
    Remove the 'model__' prefix from the parameter names.
    This is necessary because the GridSearchCV returns parameters with 'model__' prefix.
    """
    return {k.replace('model__', ''): v for k, v in best_params.items() if k.startswith('model__')}


# plots
def plot_roc_curve(y_true, y_pred_proba, experiment_dir):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(experiment_dir, 'roc_curve.png'))
    plt.close()

def plot_precision_recall_curve(y_true, y_pred_proba, experiment_dir):
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(os.path.join(experiment_dir, 'precision_recall_curve.png'))
    plt.close()

def plot_calibration_curve(y_true, y_pred_proba, experiment_dir):
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt

    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred_proba, n_bins=10)

    plt.figure()
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Calibration Curve")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.xlabel("Mean predicted value")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.savefig(os.path.join(experiment_dir, 'calibration_curve.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, experiment_dir):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], 
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(experiment_dir, 'confusion_matrix.png'))
    plt.close()


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate and return various evaluation metrics.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred), 
        'average_precision': average_precision_score(y_true, y_pred_proba)
    }
    return metrics


# this class holds the complete dataset, the folds can 
# access the data they need using the encounter IDs
# and the methods in this class
class Dataset:
    def __init__(self, 
                 config):
        
        self.target_col = config["target_col"]
        self.encounter_col = config["encounter_col"]
        self.time_col = config["time_col"]
        self.config = config
        
        self._load_dataset()
        self._transform_target()

    def _load_dataset(self):
        self.dataframe = pd.read_parquet(self.config["dataframe_path"])
        features_to_use_df = pd.read_csv(self.config["features_to_use_path"])

         # drop any rows where "do not use" is 1
        features_to_use_df = features_to_use_df[features_to_use_df['do not use'] != 1]

        # drop any rows where "proportion_sepsis" is less than proportion_w_data and
        # imputation_method is "interpolation"
        features_to_use_df = features_to_use_df[
            (features_to_use_df['proportion_sepsis'] >= self.config["proportion_w_data"]) |   
            (features_to_use_df['imputation_method'] != 'interpolation')
        ]

        # drop all features in the action_feature_sets
        for feature_set in self.config["action_feature_sets"]:
            features_to_use_df = features_to_use_df[features_to_use_df['feature_set'] != feature_set]

        # make list of features
        features = features_to_use_df['feature'].tolist()

        # add to these the encounter_col, time_col,
        features_to_keep = [self.config["encounter_col"], self.config["time_col"]] + features

        self.dataframe = self.dataframe[features_to_keep].copy()
        
        n_encounter_ids_before = self.dataframe[self.config["encounter_col"]].nunique()
        n_rows_before = self.dataframe.shape[0]  

        # remove rows with NaN values anywhere in the dataframe
        self.dataframe = self.dataframe.dropna()
        self.dataframe = self.dataframe.reset_index(drop=True)

        n_encounter_ids_after = self.dataframe[self.config["encounter_col"]].nunique()
        n_rows_after = self.dataframe.shape[0]

        print(f"Removed {n_encounter_ids_before - n_encounter_ids_after} "
                f"encounter IDs and {n_rows_before - n_rows_after} rows with NaN values.")
        print(f"Remaining encounter IDs: {n_encounter_ids_after}, Remaining rows: {n_rows_after}")       

        return self
    

    def _transform_target(self):
        """ depending on the type of prediction, we may need to alter
        the target data. For example, if we are predicting mortality,
        we may need to transform all the timepoints to a binary outcome
        (1 if the patient died, 0 if they survived). If we are predicting 
        mortality at a specific timepoint, we may need to transform the target
        to a binary outcome (1 if the patient died at that timepoint, 0 if 
        they survive) only for those specific timepoints.
        """
        # find the encounter_ids that have target_data equal to 1
        target_encounter_ids = self.dataframe[self.dataframe[self.target_col] == 1][self.encounter_col].unique()
        
        if self.config["prediction_time"] == "all":
            # all values for the target_data are 1 for the encounter_ids that have target_data equal to 1
            self.dataframe[self.target_col] = self.dataframe[self.encounter_col].apply(
                lambda x: 1 if x in target_encounter_ids else 0
            )       

        else:
            # not implemented yet, so raise an error
            raise NotImplementedError("Prediction time 'specific' is not implemented yet.")

    def modeling_data(self, encounter_ids):
        """
        Returns the modeling data as a DataFrame.
        This is the data that will be used for training and testing.
        """
        
        modeling_data = self.dataframe[self.dataframe[self.encounter_col].isin(encounter_ids)].copy()
        
        encounters_and_times = modeling_data[[self.encounter_col, self.time_col]]
        
        X_train = modeling_data.drop(columns=[self.encounter_col, self.time_col, self.target_col])
        y_train = modeling_data[self.target_col]
                
        return X_train, y_train, encounters_and_times   
    
    def encounter_ids(self):
        """
        Returns the unique encounter IDs in the dataset.
        """
        return self.dataframe[self.encounter_col].unique()


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

        # Calculate the ratio to upweight the minority positive class
        pos_count = np.sum(y_outer_train == 1)  # positive (minority) count
        neg_count = np.sum(y_outer_train == 0)  # negative (majority) count
        scale_pos_weight = neg_count / pos_count  # This will be > 1

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

    
def main():
    parser = argparse.ArgumentParser(description='Train mortality prediction model')
    parser.add_argument('--hospital', 
                        default='choc', 
                        choices=['choc', 'mimic'], 
                        help='Hospital dataset to use (default: choc)')
    parser.add_argument('--model_type', 
                        default='xgb', 
                        choices=['xgb'], 
                        help='Model type to train (default: xgb)')
    args = parser.parse_args()

    # Initialize config manager
    config_manager = ConfigManager()

    hospital = args.hospital
    config = config_manager.load_hospital_config(hospital)
    # config_path = f"config/{hospital}_supervised.yaml"
    # config = load_config(config_path)

    dataset = Dataset(config)
    print(f"Dataset loaded with {len(dataset.dataframe)} rows and {len(dataset.dataframe.columns)} columns")
          
    # train the model using the outer training loop
    model_type = args.model_type
    experiment_name = f"mortality_{model_type}_grid_search_{time.strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = config_manager.create_experiment_dir(experiment_name)

    print(f"Running on {config_manager.environment} environment")
    print(f"Experiment directory: {experiment_dir}")

    # experiment_dir = os.path.join(EXPERIMENTS_DIR, experiment_name)
    # os.makedirs(experiment_dir, exist_ok=True)
    outer_training_loop(dataset, config, experiment_dir, model_type="xgb")
    
    pass

if __name__ == "__main__":
    main()