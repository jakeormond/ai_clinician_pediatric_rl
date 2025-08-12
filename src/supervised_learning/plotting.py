
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve


# plots
def create_all_plots(y_true, y_pred, y_pred_proba, experiment_dir, is_classification=True):
    """
    Create all evaluation plots and save them to the experiment directory.
    """
    plot_roc_curve(y_true, y_pred_proba, experiment_dir)
    plot_precision_recall_curve(y_true, y_pred_proba, experiment_dir)
    plot_calibration_curve(y_true, y_pred_proba, experiment_dir)
    if is_classification:
        plot_confusion_matrix(y_true, y_pred, experiment_dir)
    else:
        # For regression, you might want to add other plots like residuals, etc.
        pass


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