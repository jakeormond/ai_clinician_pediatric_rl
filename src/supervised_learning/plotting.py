
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
from itertools import cycle


# plots
def create_all_plots(y_true, y_pred, y_pred_proba, 
        experiment_dir, is_classification=True, y_pred_proba_raw=None):
    """
    Create all evaluation plots and save them to the experiment directory.
    """
    n_classes = len(np.unique(y_true))        
    
    if is_classification and n_classes == 2:
        # For binary classification, we can use y_pred_proba directly
        plot_roc_curve(y_true, y_pred_proba, experiment_dir)
        plot_precision_recall_curve(y_true, y_pred_proba, experiment_dir)
        plot_calibration_curve(y_true, y_pred_proba, experiment_dir)
        plot_confusion_matrix(y_true, y_pred, experiment_dir)
    
    elif is_classification and n_classes > 2:
        # Multiclass classification
        if y_pred_proba_raw is not None:
            plot_multiclass_roc_curve(y_true, y_pred_proba_raw, experiment_dir)
            plot_multiclass_precision_recall_curve(y_true, y_pred_proba_raw, experiment_dir)
        plot_multiclass_confusion_matrix(y_true, y_pred, experiment_dir)
        # Note: Calibration curve for multiclass is more complex
        #
    
    else:
    # Regression
        plot_regression_plots(y_true, y_pred_proba, experiment_dir)


def plot_multiclass_roc_curve(y_true, y_pred_proba_raw, experiment_dir):
    """Plot ROC curves for multiclass classification using One-vs-Rest approach."""
    
    n_classes = y_pred_proba_raw.shape[1]
    classes = np.unique(y_true)
    
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=classes)
    if n_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])  # For binary case
    
    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba_raw[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba_raw.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    
    # Plot individual class ROC curves
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'Class {classes[i]} (AUC = {roc_auc[i]:.2f})')
    
    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"],
            label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
            color='deeppink', linestyle=':', linewidth=4)
    
    # Plot random classifier line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curves (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(experiment_dir, 'multiclass_roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_multiclass_precision_recall_curve(y_true, y_pred_proba_raw, experiment_dir):
    """Plot Precision-Recall curves for multiclass classification."""
    
    n_classes = y_pred_proba_raw.shape[1]
    classes = np.unique(y_true)
    
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=classes)
    if n_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    
    # Compute Precision-Recall curve for each class
    precision = {}
    recall = {}
    avg_precision = {}
    
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green'])
    
    for i, color in zip(range(n_classes), colors):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba_raw[:, i])
        avg_precision[i] = auc(recall[i], precision[i])
        
        plt.plot(recall[i], precision[i], color=color, lw=2,
                label=f'Class {classes[i]} (AP = {avg_precision[i]:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multiclass Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(experiment_dir, 'multiclass_precision_recall_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_multiclass_confusion_matrix(y_true, y_pred, experiment_dir):
    """Plot confusion matrix for multiclass classification."""
    
    classes = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'Class {c}' for c in classes], 
                yticklabels=[f'Class {c}' for c in classes])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Multiclass Confusion Matrix')
    plt.savefig(os.path.join(experiment_dir, 'multiclass_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_regression_plots(y_true, y_pred, experiment_dir):
    """Create plots for regression models."""
    
    # Residuals plot
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Predicted vs Actual
    axes[0, 0].scatter(y_pred, y_true, alpha=0.6)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    axes[0, 0].set_title('Predicted vs Actual')
    
    # 2. Residuals vs Predicted
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs Predicted')
    
    # 3. Residuals histogram
    axes[1, 0].hist(residuals, bins=30, alpha=0.7)
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residuals Distribution')
    
    # 4. For ordinal regression: confusion matrix style
    if len(np.unique(y_true)) <= 5:  # If few unique values (ordinal)
        y_pred_rounded = np.round(np.clip(y_pred, y_true.min(), y_true.max())).astype(int)
        cm = confusion_matrix(y_true.astype(int), y_pred_rounded)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Predicted (Rounded)')
        axes[1, 1].set_ylabel('Actual')
        axes[1, 1].set_title('Ordinal Confusion Matrix')
    else:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'regression_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, experiment_dir):

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

    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(os.path.join(experiment_dir, 'precision_recall_curve.png'))
    plt.close()

def plot_calibration_curve(y_true, y_pred_proba, experiment_dir):  
    

    # clip values of y_pred_proba to avoid issues with calibration_curve
    y_pred_proba = y_pred_proba.clip(0, 1)
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred_proba, n_bins=10)

    plt.figure()
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Calibration Curve")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.xlabel("Mean predicted value")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.savefig(os.path.join(experiment_dir, f"calibration_curve.png"))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, experiment_dir):

    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], 
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(experiment_dir, 'confusion_matrix.png'))
    plt.close()