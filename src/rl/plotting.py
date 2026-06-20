from venv import logger

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
from pathlib import Path



def plot_results(training_metrics, plot_dir, config_manager):
    
    # training_epochs = training_metrics.epoch_training_losses['epoch']
    global_step = training_metrics.epoch_training_losses['global_step']
    global_step_diff_mid = ((global_step[1]-1)-global_step[0])/2
    train_epochs_to_global_step = [g - global_step_diff_mid for g in global_step]
    train_global_steps = np.arange(0, global_step[-1] + 1)
    
    val_global_step_indices = training_metrics.val_losses['epoch']
    if val_global_step_indices[0] != -1:
        raise ValueError("Expected the first epoch index for validation losses to be -1, indicating it was logged at the end of the first epoch. Found: " + str(val_global_step_indices[0]))
    val_epochs_to_global_step = [-1] + [global_step[i] for i in val_global_step_indices[1:]]

    x_axes = {
        'train/batch': train_global_steps,
        'train/epoch': train_epochs_to_global_step,
        'val/batch': val_epochs_to_global_step,
        'val/epoch': val_global_step_indices
    }

    all_keys = get_keys(training_metrics)
    
    if 'policy_freq' in config_manager.training_config:
        policy_freq = config_manager.training_config['policy_freq']
    else:
        policy_freq = None

    plot_data(
        training_metrics, 
        x_axes, 
        all_keys, 
        'losses', 
        plot_dir, 
        policy_frequency=policy_freq
    )
    
    if 'q_stats' in all_keys:
        plot_data(
            training_metrics, 
            x_axes, 
            all_keys, 
            'q_stats', 
            plot_dir,
            policy_frequency=policy_freq
        )
    
    if 'advantage_keys' in all_keys:
        plot_data(
            training_metrics, 
            x_axes, 
            all_keys, 
            'advantages', 
            plot_dir,
            policy_frequency=policy_freq
        )
    
    plot_data(
        training_metrics, 
        x_axes, 
        all_keys, 
        'neuron_stats', 
        plot_dir,
        policy_frequency=policy_freq
    )

    plot_data(
        training_metrics, 
        x_axes, 
        all_keys, 
        'grad_norms', 
        plot_dir,
        policy_frequency=policy_freq
    )

    pass



def get_keys(training_metrics):
    all_keys = {}
    
    # loss keys
    loss_keys = list(training_metrics.batch_training_losses.keys())
    new_loss_keys = []
    for key in loss_keys:
        key_truncated = key.split('/batch_')[-1]  # get the part after the last '/'
        new_loss_keys.append(key_truncated)
   
    all_keys['losses'] = new_loss_keys

    if hasattr(training_metrics, 'batch_training_q_stats'):
        # q_stat keys
        q_stat_keys = list(training_metrics.batch_training_q_stats.keys())
        new_q_stat_keys = []
        for key in q_stat_keys:
            key_truncated = key.split('/batch_')[-1]  # get the part after the last '/'
            new_q_stat_keys.append(key_truncated)
    
        # drop batch_n from new_q_stat_keys if present
        if 'batch_n' in new_q_stat_keys:
            new_q_stat_keys.remove('batch_n')

        all_keys['q_stats'] = new_q_stat_keys

    # neuron_stat_keys
    neuron_stat_keys = list(training_metrics.val_neuron_stats.keys())
    # remove 'val/' prefix from neuron_stat_keys if present
    new_neuron_stat_keys = []
    for key in neuron_stat_keys:
        key_truncated = key.split('val/')[-1]  # get the part after the last '/'
        new_neuron_stat_keys.append(key_truncated)
    #remove 'epoch' from new_neuron_stat_keys if present
    if 'epoch' in new_neuron_stat_keys:
        new_neuron_stat_keys.remove('epoch')

    all_keys['neuron_stats'] = new_neuron_stat_keys

    # grad_norm_keys
    grad_norm_keys = list(training_metrics.batch_training_grad_norms.keys())
    new_grad_norm_keys = []
    for key in grad_norm_keys:
        key_truncated = key.split('/batch_')[-1]  # get the part after the last '/'
        new_grad_norm_keys.append(key_truncated)

    all_keys['grad_norms'] = new_grad_norm_keys

    if hasattr(training_metrics, 'val_advantages'):
        # advantage_keys
        # check if training_metrics.val_advantages has keys that start with 'val/' and if so, remove the prefix for the advantage keys
        advantage_keys = list(training_metrics.val_advantages.keys())
        if len(advantage_keys) > 0:
            advantage_keys = [key.split('/val_')[-1] for key in advantage_keys]

            all_keys['advantage_keys'] = advantage_keys
        
    return all_keys



def plot_data(training_metrics, x_axes, all_keys, group, plot_dir, policy_frequency=None):
    if group not in all_keys:
        logger.warning(f"No keys found for group {group}. Skipping plotting for this group.")
        return
    keys = all_keys[group]

    if group == 'losses':
        batch_training_data = training_metrics.batch_training_losses
        epoch_training_data = training_metrics.epoch_training_losses
        val_data = training_metrics.val_losses
        final_val_data = training_metrics.final_val_losses

    elif group == 'q_stats':
        batch_training_data = training_metrics.batch_training_q_stats
        epoch_training_data = training_metrics.epoch_training_q_stats
        val_data = training_metrics.val_q_stats
        final_val_data = training_metrics.final_val_q_stats

    elif group == 'neuron_stats':
        batch_training_data = None
        epoch_training_data = None
        val_data = training_metrics.val_neuron_stats
        final_val_data = training_metrics.final_val_neuron_stats

    elif group == 'grad_norms':
        batch_training_data = training_metrics.batch_training_grad_norms
        epoch_training_data = training_metrics.epoch_training_grad_norms
        val_data = None
        final_val_data = None
    
    elif group == 'advantages':
        batch_training_data = None
        epoch_training_data = None
        val_data = training_metrics.val_advantages
        final_val_data = training_metrics.final_val_advantages


    n_subplots = len(keys)
    n_figures = n_subplots // 3 + int(n_subplots % 3 > 0)
    n_subplots_per_figure = 3

    subplot_counter = 0

    for fig_idx in range(n_figures):    
        plt.figure(figsize=(15, 5 * n_subplots_per_figure))

        for i in range(n_subplots_per_figure):
            if subplot_counter >= n_subplots:
                break
            key = keys[subplot_counter]
            x_axis_label = 'Epochs' if batch_training_data is None else 'Global Steps'

            plt.subplot(n_subplots_per_figure, 1, i + 1)

            # check if batch_training data exists
            if batch_training_data is not None:
                batch_key = f'train/batch_{key}'
                if batch_key in batch_training_data:
                    x_axis_data = x_axes['train/batch']                  
                    batch_data = [item for sublist in batch_training_data[batch_key] for item in sublist]
                    
                    if len(batch_data) != x_axis_data.shape[0]:
                        # if policy_frequency is not None, we can try to adjust x_axis_data by multiplying the number of epochs by the policy frequency
                        if policy_frequency is not None:
                            # subsample x_axis_data by policy_frequency starting
                            # from index policy_frequency-1
                            x_axis_data = x_axis_data[policy_frequency-1::policy_frequency]
                            
                    # if length is still not right, just use x_axis from the right
                    if len(batch_data) != x_axis_data.shape[0]:
                        x_axis_data = x_axis_data[-len(batch_data):]

                    plt.plot(x_axis_data, batch_data, color=plt.cm.Reds(0.6), linewidth=1, label='Batch Training')
            
            # check if epoch_training data exists
            if epoch_training_data is not None:
                epoch_key = f'train/epoch_{key}'
                if epoch_key in epoch_training_data:
                    epoch_data = epoch_training_data[epoch_key]
                    if len(epoch_data) != len(x_axes['train/epoch']):
                        x_axis_data = x_axes['train/epoch'][-len(epoch_data):]
                    else:
                        x_axis_data = x_axes['train/epoch']
                    plt.plot(x_axis_data, epoch_data, color=plt.cm.Reds(0.9), linewidth=2, label='Epoch Training')

            # check if val data exists
            if val_data is not None:
                val_key = f'val/{key}'
                if val_key in val_data:
                    if batch_training_data is None:
                        x_axis_data = x_axes['val/epoch']
                    else: 
                        x_axis_data = x_axes['val/batch']
                    val_data_to_plot = val_data[val_key]
                    plt.plot(x_axis_data, val_data_to_plot, color=plt.cm.Blues(0.9), linewidth=2, label='Validation')      
                
                
                if final_val_data is not None:
                    final_val_key = f'val/{key}'
                    if final_val_key in final_val_data:
                        val_final = final_val_data[final_val_key]
                        plt.axhline(y=val_final, color='blue', linestyle='--', linewidth=2, label='Val final')
    
        
            plt.xlabel(x_axis_label)
            plt.ylabel(key)
            # plt.xticks(np.arange(0, epochs_range[-1] + 1, step=5))
            # plt.xlim(-1, x_axes['val/'][-1] + 1)
            plt.title(f'{key} over {x_axis_label}')
            plt.grid(True)
            plt.legend(
                loc='upper right',
                facecolor='white',  # solid white background
                framealpha=1,
                edgecolor='lightgray',
                bbox_to_anchor=(0.98, 0.98),
                fontsize=10,
            )

            subplot_counter += 1

        # Save the plot
        plot_path = Path(plot_dir) / f"{group}_{fig_idx}.png"
        plt.savefig(plot_path)
        plt.close()







# def plot_advantages(training_metrics, plot_dir):
#     loss_keys = list(training_metrics.batch_training_losses.keys())

#     # remove the 'train/' prefix from loss_keys if present
#     new_loss_keys = []
#     for key in loss_keys:
#         key_truncated = key.split('/batch_')[-1]  # get the part after the last '/'
#         new_loss_keys.append(key_truncated)

#     n_subplots = len(new_loss_keys)
#     n_figures = n_subplots // 3 + int(n_subplots % 3 > 0)
#     n_subplots_per_figure = 3

#     subplot_counter = 0

#     for fig_idx in range(n_figures):    
#         plt.figure(figsize=(15, 5 * n_subplots_per_figure))
        
#         for i in range(n_subplots_per_figure):
#             if subplot_counter >= n_subplots:
#                 break
#             key = new_loss_keys[subplot_counter]

#             # if 'loss' in key:
#             losses = {}           
#             if 'loss' in key:
#                 losses['batch_train'] = training_metrics.batch_training_losses[f'train/batch_{key}']
#                 losses['epoch_train'] = training_metrics.epoch_training_losses[f'train/epoch_{key}']
#                 losses['val'] = training_metrics.val_losses[f'val/{key}']
#                 losses['final_val'] = training_metrics.final_val_losses[f'val/{key}']

#             elif 'lr' in key:
#                 losses['batch_train'] = training_metrics.batch_training_losses[f'train/batch_{key}']
#                 losses['epoch_train'] = training_metrics.epoch_training_losses[f'train/epoch_{key}']

#             else:
#                 logger.warning(f"Key {key} not recognized as loss or learning rate. Skipping.")
#                 continue

#             epochs = training_metrics.epoch_training_losses['epoch']
#             global_step = training_metrics.epoch_training_losses['global_step']
#             global_step_diff_mid = ((global_step[1]-1)-global_step[0])/2
#             epochs_to_global_step = [g - global_step_diff_mid for g in global_step]
#             global_steps = np.arange(0, global_step[-1] + 1)
           
#             plt.subplot(n_subplots_per_figure, 1, i + 1)

#             # batch losses are a list lists, we need to concatenate them for plotting
#             batch_losses = [item for sublist in losses['batch_train'] for item in sublist]
#             # plot against global steps, which are more fine-grained than epochs in light red
#             plt.plot(global_steps, batch_losses, color=plt.cm.Reds(0.6), linewidth=1, label='Batch Training Loss')

#             # epoch losses are already averaged per epoch, so we can plot them directly against epochs in dark red
#             epoch_losses = losses['epoch_train']
#             plt.plot(epochs_to_global_step, epoch_losses, color=plt.cm.Reds(0.9), linewidth=2, label='Epoch Training Loss')

#             if 'val' in losses:
#                 global_step_indices = training_metrics.val_losses['epoch']
#                 if global_step_indices[0] != -1:
#                     raise ValueError("Expected the first epoch index for validation losses to be -1, indicating it was logged at the end of the first epoch. Found: " + str(global_step_indices[0]))
#                 epochs_to_global_step = [-1] + [global_step[i] for i in global_step_indices[1:]]

                 
#                 val_losses = losses['val']
#                 plt.plot(epochs_to_global_step, val_losses, color=plt.cm.Blues(0.9), linewidth=2, label='Validation Loss')      
                
#                 if 'final_val' in losses:                    
#                     val_loss_final = losses['final_val']
#                     # plot val loss as horizontal line
#                     plt.axhline(y=val_loss_final, color='blue', linestyle='--', linewidth=2, label='Val Loss final')

#             if i == n_subplots_per_figure - 1:
#                 plt.xlabel('Epochs')
#             plt.ylabel(key)
#             # plt.xticks(np.arange(0, epochs_range[-1] + 1, step=5))
#             plt.xlim(-1, global_step[-1] + 1)
#             plt.title(f'{key} over Epochs')
#             plt.grid(True)
#             plt.legend(
#                 loc='upper right',
#                 facecolor='white',  # solid white background
#                 framealpha=1,
#                 edgecolor='lightgray',
#                 bbox_to_anchor=(0.98, 0.98),
#                 fontsize=10,
#             )

#             subplot_counter += 1

#         # Save the plot
#         plot_path = Path(plot_dir) / f"loss_plot_{fig_idx}.png"
#         plt.savefig(plot_path)
#         plt.close()





# def plot_neuron_stats(training_metrics, plot_dir):
#     loss_keys = list(training_metrics.batch_training_losses.keys())

#     # remove the 'train/' prefix from loss_keys if present
#     new_loss_keys = []
#     for key in loss_keys:
#         key_truncated = key.split('/batch_')[-1]  # get the part after the last '/'
#         new_loss_keys.append(key_truncated)

#     n_subplots = len(new_loss_keys)
#     n_figures = n_subplots // 3 + int(n_subplots % 3 > 0)
#     n_subplots_per_figure = 3

#     subplot_counter = 0

#     for fig_idx in range(n_figures):    
#         plt.figure(figsize=(15, 5 * n_subplots_per_figure))
        
#         for i in range(n_subplots_per_figure):
#             if subplot_counter >= n_subplots:
#                 break
#             key = new_loss_keys[subplot_counter]

#             # if 'loss' in key:
#             losses = {}           
#             if 'loss' in key:
#                 losses['batch_train'] = training_metrics.batch_training_losses[f'train/batch_{key}']
#                 losses['epoch_train'] = training_metrics.epoch_training_losses[f'train/epoch_{key}']
#                 losses['val'] = training_metrics.val_losses[f'val/{key}']
#                 losses['final_val'] = training_metrics.final_val_losses[f'val/{key}']

#             elif 'lr' in key:
#                 losses['batch_train'] = training_metrics.batch_training_losses[f'train/batch_{key}']
#                 losses['epoch_train'] = training_metrics.epoch_training_losses[f'train/epoch_{key}']

#             else:
#                 logger.warning(f"Key {key} not recognized as loss or learning rate. Skipping.")
#                 continue

#             epochs = training_metrics.epoch_training_losses['epoch']
#             global_step = training_metrics.epoch_training_losses['global_step']
#             global_step_diff_mid = ((global_step[1]-1)-global_step[0])/2
#             epochs_to_global_step = [g - global_step_diff_mid for g in global_step]
#             global_steps = np.arange(0, global_step[-1] + 1)
           
#             plt.subplot(n_subplots_per_figure, 1, i + 1)

#             # batch losses are a list lists, we need to concatenate them for plotting
#             batch_losses = [item for sublist in losses['batch_train'] for item in sublist]
#             # plot against global steps, which are more fine-grained than epochs in light red
#             plt.plot(global_steps, batch_losses, color=plt.cm.Reds(0.6), linewidth=1, label='Batch Training Loss')

#             # epoch losses are already averaged per epoch, so we can plot them directly against epochs in dark red
#             epoch_losses = losses['epoch_train']
#             plt.plot(epochs_to_global_step, epoch_losses, color=plt.cm.Reds(0.9), linewidth=2, label='Epoch Training Loss')

#             if 'val' in losses:
#                 global_step_indices = training_metrics.val_losses['epoch']
#                 if global_step_indices[0] != -1:
#                     raise ValueError("Expected the first epoch index for validation losses to be -1, indicating it was logged at the end of the first epoch. Found: " + str(global_step_indices[0]))
#                 epochs_to_global_step = [-1] + [global_step[i] for i in global_step_indices[1:]]

                 
#                 val_losses = losses['val']
#                 plt.plot(epochs_to_global_step, val_losses, color=plt.cm.Blues(0.9), linewidth=2, label='Validation Loss')      
                
#                 if 'final_val' in losses:                    
#                     val_loss_final = losses['final_val']
#                     # plot val loss as horizontal line
#                     plt.axhline(y=val_loss_final, color='blue', linestyle='--', linewidth=2, label='Val Loss final')

#             if i == n_subplots_per_figure - 1:
#                 plt.xlabel('Epochs')
#             plt.ylabel(key)
#             # plt.xticks(np.arange(0, epochs_range[-1] + 1, step=5))
#             plt.xlim(-1, global_step[-1] + 1)
#             plt.title(f'{key} over Epochs')
#             plt.grid(True)
#             plt.legend(
#                 loc='upper right',
#                 facecolor='white',  # solid white background
#                 framealpha=1,
#                 edgecolor='lightgray',
#                 bbox_to_anchor=(0.98, 0.98),
#                 fontsize=10,
#             )

#             subplot_counter += 1

#         # Save the plot
#         plot_path = Path(plot_dir) / f"loss_plot_{fig_idx}.png"
#         plt.savefig(plot_path)
#         plt.close()























# def plot_validation_scores(training_scores, val_scores, test_scores, plot_dir):
#     loss_keys = list(val_scores.keys())
#     new_loss_keys = []

#     training_keys = list(training_scores.keys())
#     # drop training_keys from loss_keys
#     for key in loss_keys:
#         key_truncated = key.split('/')[-1]  # get the part after the last '/'
#         if key_truncated not in training_keys:
#             new_loss_keys.append(key)
#     loss_keys = new_loss_keys
    
#     # drop mean_Q and std_Q if present
#     if 'mean_Q' in loss_keys:
#         loss_keys.remove('mean_Q')
#     if 'std_Q' in loss_keys:
#         loss_keys.remove('std_Q')

#     n_subplots = len(loss_keys)
#     n_figures = n_subplots // 3 + int(n_subplots % 3 > 0)
#     n_subplots_per_figure = 3

#     subplot_counter = 0

#     for fig_idx in range(n_figures):    
#         plt.figure(figsize=(15, 5 * n_subplots_per_figure))
        
#         for i in range(n_subplots_per_figure):
#             if subplot_counter >= n_subplots:
#                 break
#             key = loss_keys[subplot_counter]

#             plt.subplot(n_subplots_per_figure, 1, i + 1)
#             epochs_range = np.arange(1, len(val_scores[key]) + 1)

#             # check if training_scores[key] is a list of lists (one per fold) or a single list
#             if isinstance(val_scores[key][0], list):
#                 val_scores_4plot = [item for sublist in val_scores[key] for item in sublist]
#             else:
#                 val_scores_4plot = val_scores[key]

#             val_range = np.linspace(epochs_range[0], epochs_range[-1], len(val_scores_4plot) + 1)[:-1]

#             plt.plot(val_range, val_scores_4plot, color=plt.cm.Blues(0.9), linewidth=2, label='Validation Score')

#             val_epochs = np.array(val_scores['epoch']) + 1
#             val_key = f"val/{key}"
#             if val_key in val_scores:
#                 # val_losses = val_scores[val_key]

#                 # plt.plot(val_epochs, val_losses, color=plt.cm.Oranges(0.9), linewidth=2, label='Validation Loss')      
                
#                 if test_scores is not None and key in test_scores:
#                     val_loss_final = test_scores[key]
#                     # plot val loss as horizontal line
#                     plt.axhline(y=val_loss_final, color='green', linestyle='--', linewidth=2, label='Val Loss final')

#             if i == n_subplots_per_figure - 1:
#                 plt.xlabel('Epochs')
#             plt.ylabel(key)
#             plt.xticks(np.arange(0, epochs_range[-1] + 1, step=5))
#             plt.xlim(-0.5, np.max(epochs_range)+0.5)
#             plt.title(f'{key} over Epochs')
#             plt.grid(True)
#             plt.legend(
#                 loc='upper right',
#                 facecolor='white',  # solid white background
#                 framealpha=1,
#                 edgecolor='lightgray',
#                 bbox_to_anchor=(0.98, 0.98),
#                 fontsize=10,
#             )

#             subplot_counter += 1

#         # Save the plot
#         plot_path = Path(plot_dir) / f"validation_scores_plot_{fig_idx}.png"
#         plt.savefig(plot_path)
#         plt.close()



# def plot_gradient_norms(grad_norms, plot_dir):
#     grad_keys = list(grad_norms.keys())
#     if 'was_clipped' in grad_keys:
#         grad_keys.remove('was_clipped')

#     n_subplots = len(grad_keys)
#     plt.figure(figsize=(15, 5 * n_subplots))

#     for i, key in enumerate(grad_keys):
#         plt.subplot(n_subplots, 1, i + 1)
#         epochs_range = np.arange(1, len(grad_norms[key]) + 1)

#         if isinstance(grad_norms[key][0], list):
#             grad_norms_4plot = [item for sublist in grad_norms[key] for item in sublist]
#         else:
#             grad_norms_4plot = grad_norms[key]

#         training_range = np.linspace(epochs_range[0], epochs_range[-1], len(grad_norms_4plot) + 1)[:-1]
#         plt.plot(training_range, grad_norms_4plot, color=plt.cm.viridis(0.9), linewidth=2, label=key)

#         if i == len(grad_keys) - 1:
#             plt.xlabel('Epochs')
#         plt.ylabel('Gradient Norm')
#         plt.xticks(np.arange(0, epochs_range[-1] + 1, step=5))
#         plt.xlim(-0.5, np.max(epochs_range)+0.5)
#         plt.title(f'{key} over Epochs')
#         plt.grid(True)
#         plt.legend(
#             loc='upper right',
#             facecolor='white',  # solid white background
#             framealpha=1,
#             edgecolor='lightgray',
#             bbox_to_anchor=(0.98, 0.98),
#             fontsize=10,
#         )

#      # Save the plot
#     plot_path = Path(plot_dir) / f"gradient_norms_plot.png"
#     plt.savefig(plot_path)
#     plt.close()
    

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

    
def plot_Q_by_mortality(val_scores, test_scores, plot_dir, data='val'):
    """
    Single-run version (no CV folds).
    Expects val_scores keys:
      - 'epoch'
      - 'mean_survival_Q', 'std_survival_Q'
      - 'mean_non_survival_Q', 'std_non_survival_Q'
    """
    plt.figure(figsize=(10, 5))

    epochs = np.asarray(val_scores['epoch']) + 1  # 1-based display

    survival_mean = np.asarray(val_scores['val/mean_survival_Q'], dtype=float)
    survival_std = np.asarray(val_scores['val/std_survival_Q'], dtype=float)

    non_survival_mean = np.asarray(val_scores['val/mean_non_survival_Q'], dtype=float)
    non_survival_std = np.asarray(val_scores['val/std_non_survival_Q'], dtype=float)

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

    plot_path = Path(plot_dir) / f"{data}_Q_by_mortality_plot.png"
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