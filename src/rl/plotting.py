import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_losses(training_losses, val_losses, outer_fold, config_manager):    

    loss_keys = list(training_losses.keys())
    n_subplots = len(loss_keys)

    n_epochs = len(training_losses[loss_keys[0]])
    eval_freq = config_manager.training_config['eval_freq']
    val_epochs = np.arange(0, n_epochs, eval_freq)
    if (n_epochs - 1) % eval_freq != 0:
        val_epochs = np.append(val_epochs, n_epochs-1)

    # create a figure with n_subplots subplots
    plt.figure(figsize=(15, 5 * n_subplots))
    for i, key in enumerate(loss_keys):
        plt.subplot(n_subplots, 1, i + 1)
        plt.plot(training_losses[key], label='Training ' + key)
        plt.plot(val_epochs, val_losses[key], label='Validation ' + key)
        plt.xlabel('Epochs')
        plt.ylabel(key)
        plt.title(f'Training and Validation {key} for Fold {outer_fold}')
        plt.legend()
        plt.grid(True)  
    
    # Save the plot
    plots_path = Path(config_manager.fold_dir) / f"fold_{outer_fold}_loss_plot.png"
    plt.savefig(plots_path)
    plt.close()