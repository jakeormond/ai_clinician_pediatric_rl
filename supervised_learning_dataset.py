"""
Prepares data for supervised learning by creating a dataset class.
Will handle cross validation splits and data normalization.
Splits data into X and y (as opposed to the observations, actions,
and rewards in the RL setting).

Create a DataSet object that holds the full dataset, and then 
CVSplit objects that hold the encounter IDs for the splits
and methods to access the data in the DataSet object.
"""

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
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import xgboost as xgb

EXPERIMENTS_DIR = "/home/jake/Documents/AI_Clinician/experiments"


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

    print(f"Number of unique encounter IDs: {len(dataset.encounter_ids())}")
    
    # split encounter_ids into 5 folds
    kf_outer = KFold(n_splits=5, shuffle=True, random_state=config["seed"])

    for outer_fold, (outer_train_id, outer_test_id) in enumerate(kf_outer.split(dataset.encounter_ids())):
        outer_train_encounters = dataset.encounter_ids()[outer_train_id]
        outer_test_encounters = dataset.encounter_ids()[outer_test_id]

        print(f"Outer Loop Fold {outer_fold}: Train outer encounters={len(outer_train_encounters)}, "
              f"Test outer encounters={len(outer_test_encounters)}")
        
        # Prepare outer train data
        X_outer_train, y_outer_train, _ = dataset.modeling_data(outer_train_encounters)

        # Inner CV for hyperparameter tuning
        kf_inner = KFold(n_splits=5, shuffle=True, random_state=config["seed"])
        
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=1))
        ])
            
        grid_search = GridSearchCV(pipe, param_grid, cv=kf_inner, scoring='accuracy', verbose=2)
        grid_search.fit(X_outer_train, y_outer_train)

        # Get best hyperparameters from inner CV
        best_params = grid_search.best_params_
        print(f"Best hyperparameters for outer fold {outer_fold}: {best_params}")

        # Retrain model on full outer train set with best hyperparameters
        best_model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
        
        # scale X_outer_train, and output the parameters so they can be used to scale the test set
        scaler = StandardScaler()
        X_outer_train_scaled = scaler.fit_transform(X_outer_train)        
        
        best_model.fit(X_outer_train_scaled, y_outer_train)

        # Evaluate on outer test set
        X_outer_test, y_outer_test = dataset.modeling_data(outer_test_encounters)
        X_outer_test_scaled = scaler.transform(X_outer_test)
        test_score = best_model.score(X_outer_test_scaled, y_outer_test)
        print(f"Outer Fold {outer_fold}: Test accuracy={test_score:.4f}")

        # get predictions for the outer test set
        y_pred = best_model.predict(X_outer_test_scaled)
        y_pred_proba = best_model.predict_proba(X_outer_test_scaled)[:, 1]

        # save the predictions
        np.save(os.path.join(experiment_dir, f"predictions_fold_{outer_fold}.npy"), y_pred_proba)

        # save the models
        joblib.dump(scaler, os.path.join(experiment_dir, f"scaler_fold_{outer_fold}.joblib"))
        joblib.dump(best_model, os.path.join(experiment_dir, f"model_fold_{outer_fold}.joblib"))

        # Save hyperparameters
        with open(os.path.join(experiment_dir, f"hyperparams_fold_{outer_fold}.json"), "w") as f:
            json.dump(best_params, f)

        if outer_fold == 0:
            # create the arrays to hold the predictions and test scores
            all_true_labels = np.zeros((len(dataset.encounter_ids()),))
            all_predictions = np.zeros((len(dataset.encounter_ids()),))
            all_pred_proba = np.zeros((len(dataset.encounter_ids()),))
        
        # store the predictions and test scores
        all_true_labels[outer_test_id] = y_outer_test
        all_predictions[outer_test_id] = y_pred
        all_pred_proba[outer_test_id] = y_pred_proba

        # calulate the accuracy socre, roc_auc score, precision score, recall score and f1 score
        # and save to a file
        accuracy = accuracy_score(y_outer_test, y_pred)
        roc_auc = roc_auc_score(y_outer_test, y_pred_proba)
        precision = precision_score(y_outer_test, y_pred)
        recall = recall_score(y_outer_test, y_pred)
        f1 = f1_score(y_outer_test, y_pred)

        metrics = {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        with open(os.path.join(experiment_dir, f"metrics_fold_{outer_fold}.json"), "w") as f:
            json.dump(metrics, f)

    # Save all predictions and true labels for the outer test set
    np.save(os.path.join(experiment_dir, "all_true_labels.npy"), all_true_labels)
    np.save(os.path.join(experiment_dir, "all_predictions.npy"), all_predictions)
    np.save(os.path.join(experiment_dir, "all_pred_proba.npy"), all_pred_proba)

    # calculate overall metrics
    overall_accuracy = accuracy_score(all_true_labels, all_predictions)
    overall_roc_auc = roc_auc_score(all_true_labels, all_pred_proba)
    overall_precision = precision_score(all_true_labels, all_predictions)
    overall_recall = recall_score(all_true_labels, all_predictions)
    overall_f1 = f1_score(all_true_labels, all_predictions)

    overall_metrics = {
        "overall_accuracy": overall_accuracy,
        "overall_roc_auc": overall_roc_auc,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1
    }
    with open(os.path.join(experiment_dir, "overall_metrics.json"), "w") as f:
        json.dump(overall_metrics, f)

    # plot ROC curve, precision-recall curve, calibration curve, and confusion matrix
    plot_roc_curve(all_true_labels, all_pred_proba, experiment_dir)
    plot_precision_recall_curve(all_true_labels, all_pred_proba, experiment_dir)
    plot_calibration_curve(all_true_labels, all_pred_proba, experiment_dir)
    plot_confusion_matrix(all_true_labels, all_predictions, experiment_dir)

    
def main():
    hospital = 'choc'  # or 'mimic'
    config_path = f"config/{hospital}_supervised.yaml"
    config = load_config(config_path)

    dataset = Dataset(config)
    print(f"Dataset loaded with {len(dataset.dataframe)} rows and {len(dataset.dataframe.columns)} columns")
          
    # train the model using the outer training loop
    model_type = "xgb" 
    experiment_name = f"mortality_{model_type}_grid_search_{time.strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = os.path.join(EXPERIMENTS_DIR, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    outer_training_loop(dataset, config, experiment_dir, model_type="xgb")



    
    pass

if __name__ == "__main__":
    main()