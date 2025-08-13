from sklearn.metrics import (accuracy_score, roc_auc_score, 
                            precision_score, recall_score, 
                            f1_score, average_precision_score,
                            log_loss, brier_score_loss,
                            precision_recall_curve, roc_curve, 
                            confusion_matrix, mean_squared_error,
                            mean_absolute_error, r2_score)  
                           

import numpy as np

def calculate_metrics(y_true, y_pred_proba, model_type, y_pred_proba_raw=None):
    """
    Calculate and return various evaluation metrics.
    """
    
    n_classes = len(np.unique(y_true))
    if model_type == 'xgbclassifier':
        
        if n_classes == 2:
            # Binary classification - use positive class probabilities
            y_pred_proba_clipped = np.clip(y_pred_proba, 1e-15, 1-1e-15)

            metrics = {
                'average_precision': average_precision_score(y_true, y_pred_proba),
                'roc_auc': roc_auc_score(y_true, y_pred_proba),
                'log_loss': log_loss(y_true, y_pred_proba),
                'brier_score': brier_score_loss(y_true, y_pred_proba),
                'brier_baseline': np.mean(y_true) * (1 - np.mean(y_true)),
                }
        
            # Brier improvement
            metrics['relative_brier_improvement'] = \
                (metrics['brier_baseline'] - metrics['brier_score']) / metrics['brier_baseline'] if metrics['brier_baseline'] > 0 else 0

            # Expected Calibration Error (ECE) for binary
            metrics['ece'] = calculate_ece_binary(y_pred_proba, y_true)

        else:
            y_pred_true_class_proba = y_pred_proba_raw[np.arange(len(y_true)), y_true.astype(int)]
            y_pred_true_class_proba = np.clip(y_pred_true_class_proba, 1e-15, 1-1e-15)

            # For multiclass, we need different metrics
            y_pred_classes = np.argmax(y_pred_proba_raw, axis=1)
            
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred_classes),
                'log_loss': log_loss(y_true, y_pred_proba_raw) if y_pred_proba_raw is not None else None,
                'f1_macro': f1_score(y_true, y_pred_classes, average='macro'),
                'f1_weighted': f1_score(y_true, y_pred_classes, average='weighted'),
                'mean_true_class_probability': np.mean(y_pred_true_class_proba),  # ✅ This is what you want!
            }

            # Add Expected Calibration Error for multiclass
            if y_pred_proba_raw is not None:
                metrics['ece'] = calculate_ece_multiclass(y_pred_proba_raw, y_true)


            # Add per-class average precision (one-vs-rest)
            if y_pred_proba_raw is not None:
                for class_idx in range(n_classes):
                    y_true_binary = (y_true == class_idx).astype(int)
                    if np.sum(y_true_binary) > 0:  # Only if class exists in test set
                        metrics[f'average_precision_class_{class_idx}'] = average_precision_score(
                            y_true_binary, y_pred_proba_raw[:, class_idx]
                        )

    elif model_type == 'xgbregressor':
        
        if n_classes == 2:
        
            metrics = {
                'average_precision': average_precision_score(y_true, y_pred_proba),
                'log_loss': log_loss(y_true, y_pred_proba),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred_proba)),
                'mae': mean_absolute_error(y_true, y_pred_proba),
                'r2_score': r2_score(y_true, y_pred_proba)           
            }

        else:
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred_proba)),
                'mae': mean_absolute_error(y_true, y_pred_proba),
                'r2_score': r2_score(y_true, y_pred_proba)           
            }            

    
    
    return metrics


def calculate_ece_binary(y_pred_proba, y_true, n_bins=10):
    """Expected Calibration Error for binary classification."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_pred_proba, bin_edges) - 1
    ece = 0.0
    
    for i in range(n_bins):
        bin_mask = binids == i
        if np.any(bin_mask):
            avg_conf = np.mean(y_pred_proba[bin_mask])
            avg_acc = np.mean(y_true[bin_mask])
            ece += (np.sum(bin_mask) / len(y_true)) * abs(avg_conf - avg_acc)
    
    return ece


def calculate_ece_multiclass(y_pred_proba_raw, y_true, n_bins=10):
    """Expected Calibration Error for multiclass classification."""
    # Use confidence (max probability) and accuracy
    confidences = np.max(y_pred_proba_raw, axis=1)
    predictions = np.argmax(y_pred_proba_raw, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(confidences, bin_edges) - 1
    ece = 0.0
    
    for i in range(n_bins):
        bin_mask = binids == i
        if np.any(bin_mask):
            avg_conf = np.mean(confidences[bin_mask])
            avg_acc = np.mean(accuracies[bin_mask])
            ece += (np.sum(bin_mask) / len(y_true)) * abs(avg_conf - avg_acc)
    
    return ece





