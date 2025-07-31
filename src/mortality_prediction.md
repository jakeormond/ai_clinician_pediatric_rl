## Mortality prediction is one candidate mechanism for creating the value function.  The supervised_learning_dataset.py has code to compute mortality prediction from the CHOC data (need to adapt to MIMIC IV). 

The model is trained on all features aside from the actions and mortality. Mortality is transformed such that all timepoints take on the final value (0 for survivors, 1 for non-survivors).

The code uses a nested cross-validation procedure to allow for both a grid search for optimal hyperparameters, as well as the for inference using the trained model at all timesteps. 
1. First, the data is split into 5 folds (we can change this to 10 later). 
   - This is the outer split. 
   - From this, we get 5 different training-test splits
2. From each of the outer 5 training-test splits, we hold out the test fold, and create 5 new folds from the training split.
   - This is the the inner split.
   - Using this inner split, we perform a hyperparameter search using cross-validation.
   - Once the cross-validation procedure ends and optimal hyperparameters are identified, a new model is trained on the entire outer training split.
3. Using the model trained on the outer training split, predictions are then made using the held out outer test split.
   - This procedure is repeated for each of the outer training-test splits. 
    
    

For evaluating the model, ROC curves, precision-recall curves, calibration curves, and confusion matrices are produced. 
1. ROC curve
   - What it shows: The trade-off between True Positive Rate (Sensitivity) and False Positive Rate (1-Specificity) across all classification thresholds.
      - X-axis: False Positive Rate (FPR) = False Positives / (False Positives + True Negatives)
      - Y-axis: True Positive Rate (TPR) = True Positives / (True Positives + False Negatives)
      - Diagonal line: Random classifier performance
      - AUC (Area Under Curve): Single metric summarizing performance (0.5 = random, 1.0 = perfect)
   - For your mortality prediction: Shows how well your model distinguishes between patients who will die vs. survive. Higher AUC = better discrimination.

2. Precision-Recall Curve
   - What it shows: The trade-off between Precision and Recall across all thresholds.
       - X-axis: Recall = True Positives / (True Positives + False Negatives)
       - Y-axis: Precision = True Positives / (True Positives + False Positives)
       - Better for imbalanced data: More informative than ROC when positive class is rare (like mortality)
   - For your mortality prediction: Shows the trade-off between catching all patients who will die (recall) vs. being accurate when predicting death (precision). More relevant than ROC for your imbalanced dataset.

3. Calibration Curve
   - What it shows: How well your predicted probabilities match actual outcomes.
       - X-axis: Mean predicted probability in each bin
       - Y-axis: Actual fraction of positive cases in each bin
       - Diagonal line: Perfect calibration (predicted probabilities = actual frequencies)
       - Above diagonal: Model is under-confident
       - Below diagonal: Model is over-confident
   - For your mortality prediction: If your model predicts 30% mortality risk for a group of patients, ideally ~30% of those patients should actually die. Critical for clinical decision-making.

4. Confusion Matrix
   - What it shows: Breakdown of correct and incorrect predictions at a specific threshold (usually 0.5).
       - True Negatives (TN): Correctly predicted survivors
      - False Positives (FP): Incorrectly predicted deaths (survivors predicted to die)
      - False Negatives (FN): Missed deaths (deaths predicted to survive)
      - True Positives (TP): Correctly predicted deaths
   - For your mortality prediction: Shows the raw counts of each prediction type. FN is particularly concerning (missing patients who will die), while FP might lead to unnecessary interventions.

Why All Four Matter for Mortality Prediction:
   - ROC: Overall discrimination ability
   - PR Curve: Performance with imbalanced data (few deaths vs. many survivors)
   - Calibration: Whether probabilities are trustworthy for clinical decisions
   - Confusion Matrix: Concrete understanding of prediction errors and their clinical implications
