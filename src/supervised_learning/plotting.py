
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
from itertools import cycle
import shap


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
    class_labels = {0.0: "non-survival", 1.0: "admission", 2.0: "discharge"}
    
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
    # plt.figure(figsize=(10, 8))

    # Target one screen = 1920x1080 px - for data observatory
    dpi = 100
    plt.figure(figsize=(1920/dpi, 1080/dpi), dpi=dpi)
    
    # Plot individual class ROC curves
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green'])
    for i, color in zip(range(n_classes), colors):
        class_val = classes[i]
        class_name = class_labels.get(float(class_val), f'{class_val}')
        plt.plot(fpr[i], tpr[i], color=color, lw=4,
                label=f'{class_name} (AUC = {roc_auc[i]:.2f})')
    
    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"],
            label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
            color='deeppink', linestyle=':', linewidth=4)
    
    # Plot random classifier line
    plt.plot([0, 1], [0, 1], 'k--', lw=4, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=20, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=20, fontweight='bold')
    plt.title('Multiclass ROC Curves (One-vs-Rest)', fontsize=24, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.gca().set_aspect('equal', adjustable='box')  # Force square axes

    plt.tight_layout()    
    
    plt.savefig(os.path.join(experiment_dir, 'multiclass_roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_multiclass_precision_recall_curve(y_true, y_pred_proba_raw, experiment_dir):
    """Plot Precision-Recall curves for multiclass classification."""
    
    n_classes = y_pred_proba_raw.shape[1]
    classes = np.unique(y_true)
    class_labels = {0.0: "non-survival", 1.0: "admission", 2.0: "discharge"}

    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=classes)
    if n_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    
    # Compute Precision-Recall curve for each class
    precision = {}
    recall = {}
    avg_precision = {}
    
    # plt.figure(figsize=(10, 8))
    # Target one screen = 1920x1080 px - for data observatory
    dpi = 100
    plt.figure(figsize=(1920/dpi, 1080/dpi), dpi=dpi)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green'])
    
    for i, color in zip(range(n_classes), colors):
        class_val = classes[i]
        class_name = class_labels.get(float(class_val), f'{class_val}')

        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba_raw[:, i])
        avg_precision[i] = auc(recall[i], precision[i])
        
        plt.plot(recall[i], precision[i], color=color, lw=4,
                label=f'{class_name} (AP = {avg_precision[i]:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall', fontsize=20, fontweight='bold')
    plt.ylabel('Precision', fontsize=20, fontweight='bold')
    plt.title('Multiclass Precision-Recall Curves', fontsize=24, fontweight='bold', pad=20)
    plt.legend(loc="lower left", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.gca().set_aspect('equal', adjustable='box')  # Force square axes

    plt.tight_layout() 
    
    plt.savefig(os.path.join(experiment_dir, 'multiclass_precision_recall_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_multiclass_confusion_matrix(y_true, y_pred, experiment_dir):
    """Plot confusion matrix for multiclass classification."""
    
    classes = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred)

    class_labels = {0.: "non-survival", 1.: "admission", 2.: "discharge"}
    
    # plt.figure(figsize=(8, 6))
    
    # Target one screen = 1920x1080 px - for data observatory
    dpi = 100
    plt.figure(figsize=(1400/dpi, 1080/dpi), dpi=dpi)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'{class_labels[c]}' for c in classes], 
                yticklabels=[f'{class_labels[c]}' for c in classes],
                annot_kws={'size': 16},
                square=True)
    
    plt.xlabel('Predicted', fontsize=18, fontweight='bold')
    plt.ylabel('Actual',  fontsize=18, fontweight='bold')

    plt.xticks(fontsize=16, rotation=45)  # X-axis tick labels
    plt.yticks(fontsize=16, rotation=0) 

    plt.title('Multiclass Confusion Matrix', fontsize=22, fontweight='bold', pad=20)
    
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


def create_aggregated_shap_plots(config, all_shap_results, feature_names, max_display=20):
    """Create SHAP plots aggregated across all folds."""
    
    print(f"Creating SHAP plots for {len(all_shap_results)} folds with {len(feature_names)} features")
    
    # Determine if this is multiclass
    sample_shap = all_shap_results[0]['shap_values']
    is_multiclass = len(sample_shap.shape) == 3 and sample_shap.shape[2] > 1
    
    if is_multiclass:
        _create_multiclass_shap_plots(config, all_shap_results, feature_names, max_display)
    else:
        _create_binary_regression_shap_plots(config, all_shap_results, feature_names, max_display)
    
    print(f"✓ All SHAP plots saved to {config.experiment_dir}")


def _create_multiclass_shap_plots(config, all_shap_results, feature_names, max_display):
    """Create SHAP plots for multiclass models."""
    
    sample_shap = all_shap_results[0]['shap_values']
    n_classes = sample_shap.shape[2] if len(sample_shap.shape) == 3 else 1
    print(f"Multiclass detected: {n_classes} classes")
    
    # Combine data across folds for each class
    combined_shap_by_class = [[] for _ in range(n_classes)]
    combined_data = []
    
    for fold_result in all_shap_results:
        combined_data.append(fold_result['test_data'])
        shap_values_3d = fold_result['shap_values']  # Shape: (n_samples, n_features, n_classes)
        
        # ✅ Extract each class's SHAP values from 3D array
        for class_idx in range(n_classes):
            class_shap = shap_values_3d[:, :, class_idx]  # Shape: (n_samples, n_features)
            combined_shap_by_class[class_idx].append(class_shap)
    
    # Stack arrays
    combined_test_data = np.vstack(combined_data)

    # ✅ CREATE COMBINED 3D ARRAY FOR WATERFALL PLOTS
    combined_shap_3d = []
    for fold_result in all_shap_results:
        combined_shap_3d.append(fold_result['shap_values'])
    combined_shap_3d = np.vstack(combined_shap_3d)  # Shape: (total_samples, n_features, n_classes)
    
    print(f"  Combined SHAP 3D shape: {combined_shap_3d.shape}")

    # ✅ CREATE COMBINED MULTICLASS IMPORTANCE PLOT (3-color bar chart)
    print("  Creating combined multiclass SHAP importance plot...")
    # plt.figure(figsize=(14, 10))
    dpi = 100
    # plt.figure(figsize=(1920/dpi, 1080/dpi), dpi=dpi)
    figsize = (1920/dpi, 1080/dpi)

    # Use the 3D array directly with SHAP - it will automatically create multi-colored bars
    class_names = ["non-survival", "admission", "discharge"]

    shap.summary_plot(combined_shap_3d, combined_test_data,
                    feature_names=feature_names, plot_type="bar", 
                    max_display=max_display, show=False,
                    class_names=class_names)
    
    plt.gca().title.set_fontsize(22)
    plt.gca().title.set_fontweight('bold')
    plt.gca().tick_params(axis='both', labelsize=16)
    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    for label in ax.get_xticklabels():
        label.set_fontsize(16)
    for label in ax.get_yticklabels():
        label.set_fontsize(16)

    plt.gcf().set_size_inches(figsize)
    plt.title(f'SHAP Feature Importance - All Classes Combined (All Folds)', 
             fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(config.experiment_dir, 'shap_importance_all_classes.png'), 
               dpi=dpi, bbox_inches='tight')
    plt.close()
        
    print("    ✓ Combined multiclass plots saved")
    
    # ✅ INDIVIDUAL CLASS PLOTS (keep these for detailed analysis)
    
    for class_idx in range(n_classes):
        combined_shap_class = np.vstack(combined_shap_by_class[class_idx])
        class_name = class_names[class_idx]
        
        print(f"  Creating plots for Class {class_idx} ({len(combined_shap_class)} samples)")
        
        # ✅ Summary plot (beeswarm) - use larger figure for multiclass        
        # plt.figure(figsize=(16, 12))  # Much larger for multiclass
        dpi = 100
        # plt.figure(figsize=(1920/dpi, 1080/dpi), dpi=dpi)
        figsize = (1920/dpi, 1080/dpi)
        shap.summary_plot(combined_shap_class, combined_test_data,
                        feature_names=feature_names, max_display=max_display, 
                        show=False)

        plt.gca().title.set_fontsize(22)
        plt.gca().title.set_fontweight('bold')
        plt.gca().tick_params(axis='both', labelsize=16)
        ax = plt.gca()
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        for label in ax.get_xticklabels():
            label.set_fontsize(16)
        for label in ax.get_yticklabels():
            label.set_fontsize(16)

        plt.gcf().set_size_inches(figsize)
        plt.title(f'SHAP Summary - {class_name} (All Folds, n={len(combined_shap_class)})',
                 fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(config.experiment_dir, f'shap_summary_{class_name}.png'), 
                   dpi=100, bbox_inches='tight')
        plt.close()
        
        # ✅ Feature importance bar plot - use taller figure
        # plt.figure(figsize=(14, 10))
        dpi = 100
        # plt.figure(figsize=(1920/dpi, 1080/dpi), dpi=dpi)
        figsize = (1920/dpi, 1080/dpi)
        shap.summary_plot(combined_shap_class, combined_test_data,
                        feature_names=feature_names, plot_type="bar", 
                        max_display=max_display, show=False)
        
        plt.gca().title.set_fontsize(22)
        plt.gca().title.set_fontweight('bold')
        ax = plt.gca()
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        for label in ax.get_xticklabels():
            label.set_fontsize(16)
        for label in ax.get_yticklabels():
            label.set_fontsize(16)

        plt.gcf().set_size_inches(figsize)
        plt.title(f'SHAP Feature Importance - {class_name} (All Folds)',
                 fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(config.experiment_dir, f'shap_importance_{class_name}.png'), 
                   dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Class {class_idx} plots saved")

    # ✅ ADD WATERFALL PLOTS FOR MULTICLASS
    print("  Creating multiclass waterfall plots...")
    _create_sample_waterfall_plots(config, all_shap_results, combined_shap_3d, 
                                 combined_test_data, feature_names)


def _create_binary_regression_shap_plots(config, all_shap_results, feature_names, max_display):
    """Create SHAP plots for binary/regression models."""
    
    print("Binary/regression detected")
    
    combined_shap = []
    combined_data = []
    
    for fold_result in all_shap_results:
        combined_shap.append(fold_result['shap_values'])
        combined_data.append(fold_result['test_data'])
    
    combined_shap = np.vstack(combined_shap)
    combined_test_data = np.vstack(combined_data)
    
    print(f"  Creating plots for {len(combined_shap)} total samples")
    
    # Summary plot (beeswarm)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(combined_shap, combined_test_data,
                    feature_names=feature_names, max_display=max_display, show=False)
    plt.title(f'SHAP Summary (All Folds, n={len(combined_shap)})')
    plt.tight_layout()
    plt.savefig(os.path.join(config.experiment_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature importance bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(combined_shap, combined_test_data,
                    feature_names=feature_names, plot_type="bar", 
                    max_display=max_display, show=False)
    plt.title(f'SHAP Feature Importance (All Folds)')
    plt.tight_layout()
    plt.savefig(os.path.join(config.experiment_dir, 'shap_importance.png'), bbox_inches='tight')
    plt.close()
    
    print("    ✓ Summary plots saved")
    
    # Create a few sample explanations
    _create_sample_waterfall_plots(config, all_shap_results, combined_shap, 
                                 combined_test_data, feature_names)

def _create_sample_waterfall_plots(config, all_shap_results, combined_shap, 
                                 combined_test_data, feature_names, n_samples=3):
    """Create waterfall plots for individual sample explanations."""
    
    n_sample_explanations = min(n_samples, len(combined_shap))
    explainer_expected_value = all_shap_results[0]['explainer_expected_value']
    
    # ✅ Check if this is multiclass (SHAP values are a list)
    sample_shap = all_shap_results[0]['shap_values']
    is_multiclass = (len(sample_shap.shape) == 3 and sample_shap.shape[2] > 1)

    if is_multiclass:
        # ✅ For 3D array multiclass (n_samples, n_features, n_classes)
        n_classes = sample_shap.shape[2]
        class_names = ["non-survival", "admission", "discharge"]
        print(f"    Creating waterfall plots for {n_classes} classes")
        
        for class_idx in range(n_classes):
            for i in range(n_sample_explanations):
                try:
                    # plt.figure(figsize=(10, 6))
                    dpi = 100
                    # plt.figure(figsize=(1920/dpi, 1080/dpi), dpi=dpi)
                    figsize = (1920/dpi, 1080/dpi)
                    # ✅ Extract SHAP values for this class and sample from 3D array
                    sample_shap_values = combined_shap[i, :, class_idx]  # Shape: (n_features,)
                    
                    print(f"    Sample {i}, {class_names[class_idx]}: shap_values.shape = {sample_shap_values.shape}")
                    
                    # ✅ Create SHAP Explanation object for multiclass
                    shap_explanation = shap.Explanation(
                        values=sample_shap_values,                        # SHAP values for this class and sample
                        base_values=explainer_expected_value[class_idx],  # Expected value for this class  
                        data=combined_test_data[i],                       # Feature values for this sample
                        feature_names=feature_names                       # Feature names
                    )
                    
                    # ✅ Use the Explanation object with waterfall_plot
                    shap.waterfall_plot(shap_explanation, show=False)
                    plt.gca().title.set_fontsize(22)
                    plt.gca().title.set_fontweight('bold')
                    ax = plt.gca()
                    ax.tick_params(axis='x', labelsize=20)
                    ax.tick_params(axis='y', labelsize=20)
                    for label in ax.get_xticklabels():
                        label.set_fontsize(20)
                    for label in ax.get_yticklabels():
                        label.set_fontsize(20)
                    for text in ax.texts:
                        text.set_fontsize(36) 

                    plt.gcf().set_size_inches(figsize)
                    
                    plt.title(f'Sample {i} - {class_names[class_idx]} Prediction Explanation')
                    plt.tight_layout()
                    plt.savefig(os.path.join(config.experiment_dir, f'shap_waterfall_sample_{i}_{class_names[class_idx]}.png'), 
                               dpi=100, bbox_inches='tight')
                    plt.close()
                    
                except Exception as e:
                    print(f"    ⚠️  Could not create waterfall plot for sample {i}, class {class_names[class_idx]}: {e}")
        
        print(f"    ✓ {n_sample_explanations * n_classes} waterfall plots saved")
        

    else:
        # Binary or regression case

        for i in range(n_sample_explanations):
            try:
                plt.figure(figsize=(10, 6))

                # ✅ Create SHAP Explanation object for waterfall plot
                # waterfall_plot expects an Explanation object, not raw arrays
                shap_explanation = shap.Explanation(
                    values=combined_shap[i],           # SHAP values for this sample
                    base_values=explainer_expected_value,  # Expected value
                    data=combined_test_data[i],        # Feature values for this sample
                    feature_names=feature_names        # Feature names
                )
                
                # ✅ Use the Explanation object with waterfall_plot
                shap.waterfall_plot(shap_explanation, show=False)        

                plt.title(f'Sample {i} Prediction Explanation')
                plt.tight_layout()
                plt.savefig(os.path.join(config.experiment_dir, f'shap_waterfall_sample_{i}.png'), 
                        dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"    ⚠️  Could not create waterfall plot for sample {i}: {e}")
        
        print(f"    ✓ {n_sample_explanations} waterfall plots saved")


def create_individual_shap_plots(config, fold, shap_values, test_data, 
                               feature_names, explainer_expected_value, max_display=20):
    """Create SHAP plots for individual folds (optional, for debugging)."""
    
    fold_dir = os.path.join(config.experiment_dir, f'shap_fold_{fold}')
    os.makedirs(fold_dir, exist_ok=True)
    
    if isinstance(shap_values, list):
        # Multiclass
        for class_idx, class_shap in enumerate(shap_values):
            plt.figure(figsize=(10, 6))
            shap.summary_plot(class_shap, test_data, feature_names=feature_names,
                            max_display=max_display, show=False)
            plt.title(f'Fold {fold} - Class {class_idx} SHAP Summary')
            plt.savefig(os.path.join(fold_dir, f'fold_{fold}_class_{class_idx}_summary.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    else:
        # Binary/regression
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, test_data, feature_names=feature_names,
                        max_display=max_display, show=False)
        plt.title(f'Fold {fold} SHAP Summary')
        plt.savefig(os.path.join(fold_dir, f'fold_{fold}_summary.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()