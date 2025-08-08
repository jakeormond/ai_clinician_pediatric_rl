from sklearn.metrics import (accuracy_score, roc_auc_score, 
                            precision_score, recall_score, 
                            f1_score, average_precision_score,
                            log_loss, brier_score_loss,
                            precision_recall_curve, roc_curve, 
                            confusion_matrix, mean_squared_error,
                            mean_absolute_error, r2_score)  
                           

import numpy as np

def calculate_metrics(y_true, y_pred_proba, model_type):
    """
    Calculate and return various evaluation metrics.
    """

    if model_type == 'xgbclassifier':

        metrics = {
            # 'accuracy': accuracy_score(y_true, y_pred),
            # 'roc_auc': roc_auc_score(y_true, y_pred_proba),
            # 'precision': precision_score(y_true, y_pred),
            # 'recall': recall_score(y_true, y_pred),
            # 'f1_score': f1_score(y_true, y_pred), 
            'average_precision': average_precision_score(y_true, y_pred_proba),
            'log_loss': log_loss(y_true, y_pred_proba),
            'brier_score': brier_score_loss(y_true, y_pred_proba),
            'brier_baseline': np.mean(y_true) * (1 - np.mean(y_true)),
            }
        
        # Brier improvement
        metrics['relative_brier_improvement'] = \
            (metrics['brier_baseline'] - metrics['brier_score']) / metrics['brier_baseline'] if metrics['brier_baseline'] > 0 else 0
        
        # Expected Calibration Error (ECE)
        n_bins = 10
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        binids = np.digitize(y_pred_proba, bin_edges) - 1
        ece = 0.0
        for i in range(n_bins):
            bin_mask = binids == i
            if np.any(bin_mask):
                avg_conf = np.mean(y_pred_proba[bin_mask])
                avg_acc = np.mean(y_true[bin_mask])
                ece += (np.sum(bin_mask) / len(y_true)) * abs(avg_conf - avg_acc)
        metrics['ece'] = ece

    elif model_type == 'xgbregressor':
        metrics = {
            'average_precision': average_precision_score(y_true, y_pred_proba),
            'log_loss': log_loss(y_true, y_pred_proba),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred_proba)),
            'mae': mean_absolute_error(y_true, y_pred_proba),
            'r2_score': r2_score(y_true, y_pred_proba),
            
        }

    return metrics

