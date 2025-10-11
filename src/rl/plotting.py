import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
from pathlib import Path


def plot_losses(training_scores, val_scores, test_scores, outer_fold, config_manager):    

    n_folds = len(training_scores)
    loss_keys = list(training_scores[0].keys())
    n_subplots = len(loss_keys)

    # create training colours
    train_color = 'blue'
    norm = Normalize(vmin=0, vmax=n_folds-1)
    cmap = plt.cm.get_cmap('Blues')
    colors_train = [cmap(norm(i)) for i in range(n_folds)]

    val_color = 'orange'
    norm = Normalize(vmin=0, vmax=n_folds-1)
    cmap = plt.cm.get_cmap('Oranges')
    colors_val = [cmap(norm(i)) for i in range(n_folds)]
   
    # create a figure with n_subplots subplots
    plt.figure(figsize=(15, 5 * n_subplots))
    for i, key in enumerate(loss_keys):        
        plt.subplot(n_subplots, 1, i + 1)
        
        max_epochs = 0
        for fold in range(n_folds):
            training_losses = training_scores[fold][key]
            epochs = np.arange(len(training_losses)) + 1 
            if epochs[-1] > max_epochs:
                max_epochs = len(training_losses)
            plt.plot(epochs, training_losses, color=colors_train[fold], alpha=0.6, linewidth=1.5, label=f'Training {key} Fold {fold+1}')
                     
            val_losses = val_scores[fold][key]
            val_epochs = np.array(val_scores[fold]['epoch']) + 1
            if val_epochs[-1] > max_epochs:
                max_epochs = val_epochs[-1] + 1
            plt.plot(val_epochs, val_losses, color=colors_val[fold], alpha=0.3, linewidth=2, label=f'Validation {key} Fold {fold+1}')
        
        mean_train_loss = np.mean([training_scores[fold][key] for fold in range(n_folds)], axis=0)
        plt.plot(mean_train_loss, color=train_color, linewidth=3, label='Mean Training')

        mean_val_loss = np.mean([val_scores[fold][key] for fold in range(n_folds)], axis=0)
        val_epochs = val_scores[0]['epoch'] 
        plt.plot(val_epochs, mean_val_loss, color=val_color, linewidth=3, label='Mean Validation')    
        
        test_loss = test_scores[key]
        # plot test loss as horizontal line
        plt.axhline(y=test_loss, color='green', linestyle='--', linewidth=2, label='Test ' + key)
        plt.xlabel('Epochs')
        plt.ylabel(key)
        plt.xlim(1, max_epochs)
        plt.title(f'Fold {outer_fold+1} - {key} over Epochs')
        plt.legend(
            loc='center left',      # align vertically centered on the left side
            bbox_to_anchor=(-0.25, 0.5),  # (x, y) in axes fraction coords; negative x pushes it left
        )        
        plt.grid(True)

    # Save the plot
    plots_path = Path(config_manager.fold_dir) / f"fold_{outer_fold}_loss_plot.png"
    plt.savefig(plots_path)
    plt.close()