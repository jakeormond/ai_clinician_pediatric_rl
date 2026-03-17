import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
from pathlib import Path



def plot_losses(training_scores, val_scores, test_scores, config_manager):
    loss_keys = list(training_scores.keys())
    # drop mean_Q and std_Q if present
    if 'mean_Q' in loss_keys:
        loss_keys.remove('mean_Q')
    if 'std_Q' in loss_keys:
        loss_keys.remove('std_Q')

    n_subplots = len(loss_keys)
    plt.figure(figsize=(15, 5 * n_subplots))
    for i, key in enumerate(loss_keys):        
        plt.subplot(n_subplots, 1, i + 1)
        epochs_range = np.arange(1, len(training_scores[key]) + 1)

        plt.plot(epochs_range, training_scores[key], color=plt.cm.Blues(0.9), linewidth=2, label='Training Loss')

        val_epochs = np.array(val_scores['epoch']) + 1
        val_losses = val_scores[key]

        if test_scores is None:
            plt.plot(val_epochs, val_losses, color=plt.cm.Oranges(0.9), linewidth=2, label='Validation Loss')      
        
        else:
            plt.plot(val_epochs, val_losses, color=plt.cm.Oranges(0.9), linewidth=2, label='Val Loss')     
            val_loss = test_scores[key]
            # plot val loss as horizontal line
            plt.axhline(y=val_loss, color='green', linestyle='--', linewidth=2, label='Val Loss final')

        if i == len(loss_keys) - 1:
            plt.xlabel('Epochs')
        plt.ylabel(key)
        plt.xticks(np.arange(0, epochs_range[-1] + 1, step=5))
        plt.xlim(-0.5, np.max(epochs_range)+0.5)
        plt.title(f'{key} over Epochs')
        plt.grid(True)
        plt.legend(
            loc='upper right',
            facecolor='white',  # solid white background
            framealpha=1,
            edgecolor='lightgray',
            bbox_to_anchor=(0.98, 0.98),
            fontsize=10,
        )

     # Save the plot
    plot_path = Path(config_manager.experiment_dir) / f"loss_plot.png"
    plt.savefig(plot_path)
    plt.close()


def plot_gradient_norms(grad_norms, config_manager):
    grad_keys = list(grad_norms.keys())
    if 'was_clipped' in grad_keys:
        grad_keys.remove('was_clipped')

    n_subplots = len(grad_keys)
    plt.figure(figsize=(15, 5 * n_subplots))

    for i, key in enumerate(grad_keys):
        plt.subplot(n_subplots, 1, i + 1)
        epochs_range = np.arange(1, len(grad_norms[key]) + 1)

        plt.plot(epochs_range, grad_norms[key], color=plt.cm.viridis(0.9), linewidth=2, label=key)

        if i == len(grad_keys) - 1:
            plt.xlabel('Epochs')
        plt.ylabel('Gradient Norm')
        plt.xticks(np.arange(0, epochs_range[-1] + 1, step=5))
        plt.xlim(-0.5, np.max(epochs_range)+0.5)
        plt.title(f'{key} over Epochs')
        plt.grid(True)
        plt.legend(
            loc='upper right',
            facecolor='white',  # solid white background
            framealpha=1,
            edgecolor='lightgray',
            bbox_to_anchor=(0.98, 0.98),
            fontsize=10,
        )

     # Save the plot
    plot_path = Path(config_manager.experiment_dir) / f"gradient_norms_plot.png"
    plt.savefig(plot_path)
    plt.close()
    

def plot_c_and_brier_over_time(test_scores, config_manager):
    
    metric_keys = ['weighted_C_index', 'ibs_weighted', 'auroc', 'average_precision']
    epochs = [e+1 for e in test_scores['epoch']]
    n_subplots = len(metric_keys)
    
    plt.figure(figsize=(15, 5 * n_subplots))
    for i, key in enumerate(metric_keys):        
        
        # test_scores[key] is a list tensors of len(2), we want to plot the 
        # concatenated columns over time
        metric_values = np.stack([score for score in test_scores[key]], axis=0)
        
        plt.subplot(n_subplots, 1, i + 1)
        
        if key in ['weighted_C_index', 'ibs_weighted']:
            plt.plot(epochs, metric_values[:,0], color=plt.cm.Blues(0.9), linewidth=2, label=f'{key} - Cause 1')
            plt.plot(epochs, metric_values[:,1], color=plt.cm.Oranges(0.9), linewidth=2, label=f'{key} - Cause 2')
        else:
            # for auroc and average_precision, plot single line
            plt.plot(epochs, metric_values, color=plt.cm.Purples(0.9), linewidth=2, label=f'{key}')

            if key == 'average_precision':
                prevalence = test_scores['prevalence']
                plt.plot(epochs, prevalence, color='red', linestyle='--', linewidth=2, label='Prevalence')

        if i == len(metric_keys) - 1:
            plt.xlabel('Epochs')
        plt.ylabel(key)
        plt.xticks(epochs)
        plt.xlim(-0.5, np.max(epochs))
        plt.title(f'{key} over Epochs')
        plt.grid(True)
        plt.legend(
            loc='upper right',
            facecolor='white',  # solid white background
            framealpha=1,
            edgecolor='lightgray',
            bbox_to_anchor=(0.98, 0.98),
            fontsize=10,
        )

     # Save the plot
    plot_path = Path(config_manager.experiment_dir) / f"c_index_and_brier_score.png"
    plt.savefig(plot_path)
    plt.close()



def plot_auroc_prc_curves(roc_curve, pr_curve, prevalence, config_manager):
    plot_roc(roc_curve["fpr"], roc_curve["tpr"], config_manager)
    plot_prc(
        pr_curve["precision"], 
        pr_curve["recall"], 
        prevalence,
        config_manager
        )


def plot_roc(fpr, tpr, config_manager):
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='blue', linewidth=2, label='ROC Curve')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid()

    # Save the plot
    plot_path = Path(config_manager.experiment_dir) / f"roc_curve.png"
    plt.savefig(plot_path)
    plt.close()

def plot_prc(precision, recall, prevalence, config_manager):
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='blue', linewidth=2, label='PRC Curve')
    plt.axhline(y=prevalence, color='red', linestyle='--', label='Prevalence')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid()

    # Save the plot
    plot_path = Path(config_manager.experiment_dir) / f"prc_curve.png"
    plt.savefig(plot_path)
    plt.close()


def plot_losses_cv(training_scores, val_scores, test_scores, outer_fold, config_manager):    

    n_folds = len(training_scores)
    loss_keys = list(training_scores[0].keys())
   
    # if q stats are present, plot them first
    if 'mean_Q' in loss_keys:
        # create figures for mean Q values with std deviation shading
        plot_Q_by_mortality(val_scores, test_scores, outer_fold, config_manager)
        
        # plot snr
        plot_Q_survival_non_survival_snr(val_scores, test_scores, outer_fold, config_manager)

        loss_keys.remove('mean_Q')
        loss_keys.remove('std_Q')
        loss_keys.remove('mean_survival_Q')
        loss_keys.remove('std_survival_Q')
        loss_keys.remove('mean_non_survival_Q')
        loss_keys.remove('std_non_survival_Q')
        loss_keys.remove('survival_non_survival_diff')
        loss_keys.remove('survival_non_survival_snr')

    n_subplots = len(loss_keys)
    # create a figure with n_subplots subplots
    plt.figure(figsize=(15, 5 * n_subplots))
    for i, key in enumerate(loss_keys):        
        plt.subplot(n_subplots, 1, i + 1)
        
        train_max_length = max(len(training_scores[fold][key]) for fold in range(n_folds))
        train_array = np.array([np.pad(training_scores[fold][key], 
                (0, train_max_length - len(training_scores[fold][key])), 
                mode='edge') for fold in range(n_folds)])
        train_min = np.nanmin(train_array, axis=0)
        train_max = np.nanmax(train_array, axis=0)
        train_mean = np.nanmean(train_array, axis=0)
        epochs_range = np.arange(1, train_max_length + 1)

        val_array = np.full_like(train_array, np.nan)
        for fold in range(n_folds):
            val_losses = val_scores[fold][key]
            val_epochs = np.array(val_scores[fold]['epoch'])
            val_array[fold, val_epochs] = val_losses
        
        val_mask = ~np.all(np.isnan(val_array), axis=0)
        val_array = val_array[:, val_mask]
        
        val_mean = np.nanmean(val_array, axis=0)
        val_min  = np.nanmin(val_array, axis=0)
        val_max  = np.nanmax(val_array, axis=0)

        val_epochs_range = epochs_range[val_mask]

        plt.fill_between(epochs_range, train_min, train_max, color=plt.cm.Blues(0.6), alpha=0.25, edgecolor=None)
        plt.plot(epochs_range, train_mean, color=plt.cm.Blues(0.9), linewidth=2, label='Training Loss - Mean')

        plt.fill_between(val_epochs_range, val_min, val_max, color=plt.cm.Oranges(0.6), alpha=0.25, edgecolor=None)
        plt.plot(val_epochs_range, val_mean, color=plt.cm.Oranges(0.9), linewidth=2, label='Validation Loss - Mean')      
        
        
        if test_scores is not None:
            test_loss = test_scores[key]
            # plot test loss as horizontal line
            plt.axhline(y=test_loss, color='green', linestyle='--', linewidth=2, label='Test ' + key)
        if i == len(loss_keys) - 1:
            plt.xlabel('Epochs')
        plt.ylabel(key)
        plt.xticks(np.arange(1, epochs_range[-1] + 1, step=5))
        plt.xlim(0.5, np.max(epochs_range))
        plt.title(f'Fold {outer_fold+1} - {key} over Epochs')
        plt.grid(True)
        plt.legend(
            loc='upper right',
            facecolor='white',  # solid white background
            framealpha=1,
            edgecolor='lightgray',
            bbox_to_anchor=(0.98, 0.98),
            fontsize=10,
        )         

    # Save the plot
    plots_path = Path(config_manager.fold_dir) / f"fold_{outer_fold}_loss_plot.png"
    plt.savefig(plots_path)
    plt.close()

    
def plot_Q_by_mortality(val_scores, test_scores, config_manager, data='val'):
    """
    Single-run version (no CV folds).
    Expects val_scores keys:
      - 'epoch'
      - 'mean_survival_Q', 'std_survival_Q'
      - 'mean_non_survival_Q', 'std_non_survival_Q'
    """
    plt.figure(figsize=(10, 5))

    epochs = np.asarray(val_scores['epoch']) + 1  # 1-based display

    survival_mean = np.asarray(val_scores['mean_survival_Q'], dtype=float)
    survival_std = np.asarray(val_scores['std_survival_Q'], dtype=float)

    non_survival_mean = np.asarray(val_scores['mean_non_survival_Q'], dtype=float)
    non_survival_std = np.asarray(val_scores['std_non_survival_Q'], dtype=float)

    # Plot validation curves + std bands
    plt.fill_between(
        epochs,
        survival_mean - survival_std,
        survival_mean + survival_std,
        color=plt.cm.Blues(0.6),
        alpha=0.25,
        edgecolor=None
    )
    plt.plot(epochs, survival_mean, color=plt.cm.Blues(0.9), linewidth=2, label='Val survival_Q')

    plt.fill_between(
        epochs,
        non_survival_mean - non_survival_std,
        non_survival_mean + non_survival_std,
        color=plt.cm.Oranges(0.6),
        alpha=0.25,
        edgecolor=None
    )
    plt.plot(epochs, non_survival_mean, color=plt.cm.Oranges(0.9), linewidth=2, label='Val non_survival_Q')

    # Optional test horizontal lines
    if test_scores is not None:
        plt.axhline(y=test_scores['mean_survival_Q'], color='blue', linestyle='--', linewidth=2, label='Test survival_Q')
        plt.axhline(y=test_scores['mean_survival_Q'] - test_scores['std_survival_Q'], color='blue', linestyle='--', linewidth=1)
        plt.axhline(y=test_scores['mean_survival_Q'] + test_scores['std_survival_Q'], color='blue', linestyle='--', linewidth=1)

        plt.axhline(y=test_scores['mean_non_survival_Q'], color='orange', linestyle='--', linewidth=2, label='Test non_survival_Q')
        plt.axhline(y=test_scores['mean_non_survival_Q'] - test_scores['std_non_survival_Q'], color='orange', linestyle='--', linewidth=1)
        plt.axhline(y=test_scores['mean_non_survival_Q'] + test_scores['std_non_survival_Q'], color='orange', linestyle='--', linewidth=1)

    plt.xlabel('Epochs')
    plt.ylabel('Q value')
    if len(epochs) > 0:
        plt.xticks(np.arange(1, int(epochs[-1]) + 1, step=5))
        plt.xlim(0.5, np.max(epochs))
    plt.title('Q values by Mortality over Epochs')
    plt.grid(True)
    plt.legend(
        loc='upper left',
        facecolor='white',
        framealpha=1,
        edgecolor='lightgray',
        bbox_to_anchor=(0.02, 0.98),
        fontsize=10,
    )

    plot_path = Path(config_manager.experiment_dir) / f"{data}_Q_by_mortality_plot.png"
    plt.savefig(plot_path)
    plt.close()
    
    
    
    
    
    
    
    # # create a figure with n_subplots subplots
    # plt.figure(figsize=(15, 5))

    # # determine the maximum length of epochs across all folds, as well as the value of the final
    # # epoch for the longest fold
    # max_epoch = max(val_scores[fold]['epoch'][-1] for fold in range(n_folds))

    # # create a vector from 0 to max_epoch
    # epochs = np.arange(0, max_epoch + 1)

    # survival_Q_array = np.full((n_folds, len(epochs)), np.nan)
    # non_survival_Q_array = np.full((n_folds, len(epochs)), np.nan)

    # survival_Q_std_array = np.full((n_folds, len(epochs)), np.nan)
    # non_survival_Q_std_array = np.full((n_folds, len(epochs)), np.nan)

    # for fold in range(n_folds):
    #     epochs_fold = val_scores[fold]['epoch']           
    #     survival_Q = val_scores[fold]['mean_survival_Q']
    #     non_survival_Q = val_scores[fold]['mean_non_survival_Q']

    #     survival_Q_array[fold, epochs_fold] = survival_Q
    #     non_survival_Q_array[fold, epochs_fold] = non_survival_Q 

    #     survival_Q_std = val_scores[fold]['std_survival_Q']
    #     non_survival_Q_std = val_scores[fold]['std_non_survival_Q']

    #     survival_Q_std_array[fold, epochs_fold] = survival_Q_std
    #     non_survival_Q_std_array[fold, epochs_fold] = non_survival_Q_std

    # epochs = [epoch+1 for epoch in epochs]

    # mask = ~np.all(np.isnan(survival_Q_array), axis=0)
    # survival_Q_array = survival_Q_array[:, mask]
    # non_survival_Q_array = non_survival_Q_array[:, mask]
    # epochs = np.array(epochs)[mask]

    # survival_Q_std_array = survival_Q_std_array[:, mask]
    # non_survival_Q_std_array = non_survival_Q_std_array[:, mask]

    # mean_survival_Q = np.nanmean(survival_Q_array, axis=0)
    # mean_survival_Q_std = np.nanmean(survival_Q_std_array, axis=0)

    # mean_non_survival_Q = np.nanmean(non_survival_Q_array, axis=0)
    # mean_non_survival_Q_std = np.nanmean(non_survival_Q_std_array, axis=0)

    # plt.fill_between(epochs, mean_survival_Q - mean_non_survival_Q_std, mean_survival_Q + mean_survival_Q_std, color=plt.cm.Blues(0.6), alpha=0.25, edgecolor=None)
    # plt.plot(epochs, mean_survival_Q, color=plt.cm.Blues(0.9), linewidth=2, label='Val survival_Q - Mean')

    # plt.fill_between(epochs, mean_non_survival_Q - mean_non_survival_Q_std, mean_non_survival_Q + mean_non_survival_Q_std, color=plt.cm.Oranges(0.6), alpha=0.25, edgecolor=None)
    # plt.plot(epochs, mean_non_survival_Q, color=plt.cm.Oranges(0.9), linewidth=2, label='Val non_survival_Q - Mean')

    # plt.axhline(y=test_scores['mean_survival_Q'], color='blue', linestyle='--', linewidth=2, label='Test survival_Q')
    # plt.axhline(y=test_scores['mean_survival_Q'] - test_scores['std_survival_Q'], color='blue', linestyle='--', linewidth=1)
    # plt.axhline(y=test_scores['mean_survival_Q'] + test_scores['std_survival_Q'], color='blue', linestyle='--', linewidth=1)

    # plt.axhline(y=test_scores['mean_non_survival_Q'], color='orange', linestyle='--', linewidth=2, label='Test non_survival_Q')
    # plt.axhline(y=test_scores['mean_non_survival_Q'] - test_scores['std_non_survival_Q'], color='orange', linestyle='--', linewidth=1)
    # plt.axhline(y=test_scores['mean_non_survival_Q'] + test_scores['std_non_survival_Q'], color='orange', linestyle='--', linewidth=1)

    # plt.xlabel('Epochs')
    # plt.ylabel('Q value')
    # plt.xticks(np.arange(1, epochs[-1] + 1, step=5))
    # plt.xlim(0.5, np.max(epochs))
    # plt.title('Q values by Mortality over Epochs')
    # plt.grid(True)
    # plt.legend(
    #     loc='upper left',
    #     facecolor='white',  # solid white background
    #     framealpha=1,
    #     edgecolor='lightgray',
    #     bbox_to_anchor=(0.02, 0.98),
    #     fontsize=10,
    # )

    # # Save the plot
    # plots_path = Path(config_manager.experiment_dir) / f"{data}_Q_by_mortality_plot.png"
    # plt.savefig(plots_path)
    # plt.close()


def plot_Q_survival_non_survival_snr(val_scores, test_scores, outer_fold, config_manager, data='val'):
    
    n_folds = len(val_scores)

    # create a figure with n_subplots subplots
    plt.figure(figsize=(15, 5))

    # determine the maximum length of epochs across all folds, as well as the value of the final
    # epoch for the longest fold
    max_epoch = max(val_scores[fold]['epoch'][-1] for fold in range(n_folds))

    # create a vector from 0 to max_epoch
    epochs = np.arange(0, max_epoch + 1)

    snr_array = np.full((n_folds, len(epochs)), np.nan)

    for fold in range(n_folds):
        epochs_fold = val_scores[fold]['epoch']
        snr = val_scores[fold]['survival_non_survival_snr']

        snr_array[fold, epochs_fold] = snr

    # Compute mean and std across folds
    if n_folds > 1:
        mean_snr = np.nanmean(snr_array, axis=0)
        std_snr = np.nanstd(snr_array, axis=0)

        # drop epochs where all mean_snr are nan
        mask = ~(np.isnan(mean_snr)).all(axis=0)
        mean_snr = mean_snr[mask]
        std_snr = std_snr[mask]
        epochs = epochs[mask]

        # Plotting
        plt.fill_between(epochs, mean_snr - std_snr, mean_snr + std_snr, color='gray', alpha=0.2)
        plt.plot(epochs, mean_snr, color='black', linewidth=2, label='Mean SNR')

    else:
        mean_snr = snr_array[0].flatten()
        # drop epochs where all mean_snr are nan
        mask = ~(np.isnan(mean_snr))
        mean_snr = mean_snr[mask]
        epochs = epochs[mask]
    
        plt.plot(epochs, mean_snr, color='black', linewidth=2, label='Mean SNR')
    
    plt.axhline(y=test_scores['survival_non_survival_snr'], color='red', linestyle='--', linewidth=2, label='Test SNR')

    plt.xlabel('Epochs')
    plt.ylabel('SNR')
    plt.title('SNR over Epochs')
    plt.legend()
    plt.grid()

    # Save the plot
    plots_path = Path(config_manager.experiment_dir) / f"{data}_snr_plot.png"
    plt.savefig(plots_path)
    plt.close()