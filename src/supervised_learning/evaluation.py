from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, average_precision_score

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
