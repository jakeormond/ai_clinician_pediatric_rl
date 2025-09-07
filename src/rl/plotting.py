import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_losses(training_losses, val_losses, outer_fold, config_manager):    

    plt.figure(figsize=(10, 6))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(list(val_losses.keys()), list(val_losses.values()), label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Losses for Fold {outer_fold}')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plots_path = Path(config_manager.experiment_dir) / f"fold_{outer_fold}_loss_plot.png"
    plt.savefig(plots_path)
    plt.close()