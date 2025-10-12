import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
from pathlib import Path


def plot_losses(training_scores, val_scores, test_scores, outer_fold, config_manager):    

    n_folds = len(training_scores)
    loss_keys = list(training_scores[0].keys())
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
        
        test_loss = test_scores[key]
        # plot test loss as horizontal line
        plt.axhline(y=test_loss, color='green', linestyle='--', linewidth=2, label='Test ' + key)
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