import argparse
import json
import os
import pickle
from pathlib import Path
from turtle import pd
import matplotlib.pyplot as plt
import pandas as pd

import torch
import numpy as np
from ..config_manager import ConfigManager   
from .models.basic_q import Critic
from .models.td3_bc import Actor as TD3Actor
from .models.conservative_q_learning import Actor as CQLActor
from .models.conservative_q_learning import Critic as CQLCritic
from .models.iql import Actor as IQLActor
from .models.iql import Critic as IQLCritic
from .models.iql import ValueNetwork as IQLValue
from .training import create_processor_from_shards
from .replay_buffer import RLDataset
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats

from ..setup_environment import load_paths


# rl_models_path = "/home/jake/Documents/AI_Clinician/rl_models/" 


def load_configuration(config_manager_path):
    if not os.path.exists(config_manager_path):
        print(f"Config manager not found in: {config_manager_path}, skipping evaluation.")
        raise FileNotFoundError(f"Config manager not found in: {config_manager_path}")
    else:
        print(f"Loading config manager from: {config_manager_path}")
        with open(config_manager_path, "rb") as cfg_file:
            config_manager = pickle.load(cfg_file)
    
    # set paths 
    config_manager.paths, config_manager.environment = load_paths(config_manager.hospital)
    if config_manager.environment == "local":
        config_manager.config_dir = Path(config_manager.paths['config_dir'])
    config_manager.set_data()
    config_manager.reset_cohort(config_manager.cohort)

    return config_manager


def create_test_rl_dataset(train_processor, test_processor, config_manager, hyperparams):
    action_names=config_manager.data_config['actions']
    n_step = hyperparams['n_step']
    gamma = hyperparams['discount']
    use_compressed_latents = hyperparams['use_compressed_latents']

    scaler = train_processor.create_rl_dataset(
        states="model_outputs", 
        scaler=None, 
        action_names=action_names, 
        use_compressed_latents=use_compressed_latents,
        n_step=n_step,
        gamma=gamma
    ) 

    _ = test_processor.create_rl_dataset(
        states="model_outputs",
        scaler=scaler,
        action_names=action_names,
        use_compressed_latents=use_compressed_latents,
        n_step=n_step,
        gamma=gamma
    )

    return train_processor, test_processor, scaler


def load_sarsa_critic(config_manager = None, model_path = None):
    reward_type = config_manager.reward_type
    discount = config_manager.discount
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_path is None and config_manager is not None:
        model_path = os.path.join(config_manager.paths['rl_models_dir'], config_manager.model_type, 
                        f"split_{config_manager.split_num}", config_manager.cohort, config_manager.actions, 
                        reward_type, f"final_model_discount_{discount}",)
                                              
        # get folder inside model_path
        model_folder = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))][0]
        model_path = os.path.join(model_path, model_folder)

    configuration_path = os.path.join(model_path, "config_manager.pkl")
    sarsa_configuration = load_configuration(configuration_path)
    
    final_model_path = os.path.join(model_path, "final_model.pt")

    # load results.json from model_folder, which has state_dim 
    results_path = os.path.join(model_path, "results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
            hidden_dim = results["hyperparameters"]["hidden_dim"]

    sarsa_critic = Critic(512,2, hidden_dim=hidden_dim).to(device)
    sarsa_critic.to(device)
    checkpoint = torch.load(final_model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['critic_state_dict'] 
    sarsa_critic.load_state_dict(state_dict)
    sarsa_critic.eval()

    # load hyperparameters from results.json
    results_path = os.path.join(model_path, "results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)  
    sarsa_hyperparams = results["hyperparameters"]
        
    return sarsa_critic, sarsa_configuration, sarsa_hyperparams


def load_actor(config_manager, model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_path is None:
        model_path = os.path.join(config_manager.paths['rl_models_dir'], config_manager.rl_type, 
                        f"split_{config_manager.split_num}", config_manager.cohort, config_manager.actions, "final_model")
    
        # get folder inside model_path
        model_folder = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))][0]
        model_path = os.path.join(model_path, model_folder)

    # configuration file is in the parent directory of the model_path
    configuration_path = os.path.join(os.path.dirname(model_path), "config_manager.pkl")
    actor_configuration = load_configuration(configuration_path)

    final_model_path = os.path.join(model_path, "final_model.pt")

    # load results.json from model_folder, which has state_dim 
    results_path = os.path.join(model_path, "results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
            hyperparams = results["hyperparameters"]
            # hidden_dim = hyperparams["hidden_dim"]

    if actor_configuration.model_type == "td3_bc":
        actor = TD3Actor(512, 2).to(device)
    elif actor_configuration.model_type == "conservative_q_learning":
        actor = CQLActor(512, hyperparams["hidden_dim"]).to(device)
        critic = CQLCritic(512, 2, hidden_dim=hyperparams["hidden_dim"]).to(device)
    elif actor_configuration.model_type == "iql":
        actor = IQLActor(512, hyperparams["hidden_dim"]).to(device)
        critic = IQLCritic(512, 2, hidden_dim=hyperparams["hidden_dim"]).to(device)

    checkpoint = torch.load(final_model_path, map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()

    # load critic as well
    critic.load_state_dict(checkpoint['critic_state_dict'])
    critic.eval()

    return actor, critic, actor_configuration, hyperparams, model_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RL models on SARSA CHOC")

    # Keyword Arguments (Match your launch.json)
    parser.add_argument("--hospital", type=str, default="choc")
    parser.add_argument("--rl_type", type=str, choices=["td3_bc", "conservative_q_learning", "bc", "implicit_q_learning"], required=True)
    parser.add_argument("--experiment_type", type=str, default="model_evaluation")
    parser.add_argument("--model_type", type=str, default="basic_q")
    parser.add_argument("--mode", type=str, default="single_split")
    parser.add_argument("--split_num", type=int, default=0)
    parser.add_argument("--cohort", type=str, default="sepsis")
    parser.add_argument("--actions", type=str, default="vis-intake")
    parser.add_argument("--reward_type", type=str, default="health_score_change_only")
    parser.add_argument("--discount", type=float, default=0.975)
    
    # The Flag (Action: store_true means if it's in the args, it becomes True)
    parser.add_argument("--testing", action="store_true", help="Run in test mode")

    return parser.parse_args()


def generate_actions(rl_actor, states, batch_size=1024, device="cuda" if torch.cuda.is_available() else "cpu"):
    
    loader = DataLoader(states, batch_size=1024, shuffle=False)

    all_rl_actions = []
    with torch.no_grad():        
        for batch in loader:
            batch_states = batch.to(device)
            rl_actions = rl_actor.act(batch_states, device=device)
            all_rl_actions.append(rl_actions)

    all_rl_actions = np.concatenate(all_rl_actions, axis=0)
    # convert back to torch tensor
    all_rl_actions = torch.from_numpy(all_rl_actions).float().to(device)
    return all_rl_actions


def calculate_advantage(sarsa_critic, states, clinician_actions, rl_actions, device="cuda" if torch.cuda.is_available() else "cpu"):
    states = states.to(device)
    clinician_actions = clinician_actions.to(device)
    rl_actions = rl_actions.to(device)

    with torch.no_grad():
        q1_clinician, q2_clinician = sarsa_critic(states, clinician_actions)
        q1_rl, q2_rl = sarsa_critic(states, rl_actions)

    q_clinician = torch.min(q1_clinician, q2_clinician)
    q_rl = torch.min(q1_rl, q2_rl)

    advantage = q_rl - q_clinician
    mean_advantage = advantage.mean().item()
    print(f"Mean Advantage of RL Actions over Clinician Actions: {mean_advantage:.4f}")
    
    return advantage.cpu().numpy().flatten(), q_clinician.cpu().numpy().flatten(), q_rl.cpu().numpy().flatten()


def plot_advantage(advantage, title=None, save_dir=None):
    # plot as kde plot
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(7, 4))
    sns.kdeplot(advantage, fill=True)
    plt.title(f"Advantage Distribution of {title} over Clinician Actions")
    plt.xlabel("Advantage")
    plt.ylabel("Density")
    plt.xlim(-.1, .2)

    # plot a vertical line at mean advantage
    mean_advantage = np.mean(advantage)
    plt.axvline(mean_advantage, color='red', linestyle='--', label=f"Mean Advantage: {mean_advantage:.4f}")
    plt.legend()

    # plot a vertical line at median advantage
    median_advantage = np.median(advantage)
    plt.axvline(median_advantage, color='green', linestyle='--', label=f"Median Advantage: {median_advantage:.4f}")
    plt.legend()
    
    results_dir = os.path.join(save_dir, f"advantage_plot_{title.replace(' ', '_')}.png")
    plt.savefig(results_dir)
    plt.close()


def plot_real_and_noise_advantage(advantage, noise_advantage, title=None, save_dir=None):
    # plot as kde plot
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(7, 4))
    sns.kdeplot(advantage, fill=True)

    sns.kdeplot(noise_advantage, fill=True, color='orange', alpha=0.5)

    plt.title(f"Advantage Distribution of {title} over Clinician Actions")
    plt.xlabel("Advantage")
    plt.ylabel("Density")
    plt.legend(["Real Advantage", "Noise Advantage"])
    plt.xlim(-.1, .2)

    # plot a vertical line at mean advantage
    mean_advantage = np.mean(advantage)
    plt.axvline(mean_advantage, color='red', linestyle='--', label=f"Real mean: {mean_advantage:.4f}")
    plt.legend()

    # plot a vertical line at median advantage
    median_advantage = np.median(advantage)
    plt.axvline(median_advantage, color='green', linestyle='--', label=f"Real median: {median_advantage:.4f}")
    plt.legend()

    # plot a vertical line at mean noise advantage
    mean_noise_advantage = np.mean(noise_advantage)
    plt.axvline(mean_noise_advantage, color='orange', linestyle='--', label=f"Noise mean: {mean_noise_advantage:.4f}")
    plt.legend()

    # plot a vertical line at median noise advantage
    median_noise_advantage = np.median(noise_advantage)
    plt.axvline(median_noise_advantage, color='purple', linestyle='--', label=f"Noise median: {median_noise_advantage:.4f}")
    plt.legend()

    results_dir = os.path.join(save_dir, f"advantage_plot_{title.replace(' ', '_')}.png")
    plt.savefig(results_dir)
    plt.close()

    advantage_stats = {
        "model directory": save_dir,
        "Real Mean Advantage": mean_advantage,
        "Real Median Advantage": median_advantage,
        "Noise Mean Advantage": mean_noise_advantage,
        "Noise Median Advantage": median_noise_advantage
    }
    return advantage_stats



def plot_advantage_against_clinician_q_values(advantage, q_clinician, model_folder, config_manager, critic_type="sarsa"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    os.makedirs(model_folder, exist_ok=True)

    advantage = np.asarray(advantage)
    q_clinician = np.asarray(q_clinician)
    finite_mask = np.isfinite(advantage) & np.isfinite(q_clinician)

    if finite_mask.sum() < 2:
        print(
            f"Skipping plot_advantage_against_clinician_q_values: only {finite_mask.sum()} finite points."
        )
        return

    advantage = advantage[finite_mask]
    q_clinician = q_clinician[finite_mask]
    
    plt.figure(figsize=(6, 4))

    # 1. Plot the data with a regression line
    # 'scatter_kws' thins out the points so the line is visible
    sns.regplot(x=q_clinician, y=advantage, 
                scatter_kws={'alpha':0.1, 's':2}, 
                line_kws={'color':'red', 'label':'Linear Trend'})

    # 2. Calculate Pearson correlation for the title
    try:
        r, p = stats.pearsonr(q_clinician, advantage)
    except ValueError:
        r, p = np.nan, np.nan

    plt.title(f"Advantage vs Clinician Q-Values ({config_manager.model_type.upper()})\n"
                f"Pearson R: {r:.3f}, p-value: {p:.3e}")
    plt.xlabel(f"Clinician Q-Values (Evaluated by {critic_type.upper()})")
    plt.ylabel("RL Policy Advantage")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    results_dir = os.path.join(model_folder, f"advantage_vs_clinician_q_values_{critic_type}.png")
    plt.savefig(results_dir, dpi=300, bbox_inches='tight')
    print(f"Saved advantage-vs-clinician plot to: {results_dir}")
    plt.close()
    # plt.show()



def plot_binned_advantage(advantage, q_clinician, model_folder, config_manager, critic_type="sarsa"):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({
        'Clinician_Q': q_clinician,
        'Advantage': advantage
    })

    # Divide Clinician Q-values into 10 equal-sized bins
    df['Q_Bin'] = pd.qcut(df['Clinician_Q'], q=10, labels=False, duplicates='drop')
    
    # Calculate the mean Q-value for each bin to use as the X-axis center
    bin_centers = df.groupby('Q_Bin')['Clinician_Q'].mean()
    df['Bin_Center'] = df['Q_Bin'].map(bin_centers)

    plt.figure(figsize=(8, 4))
    
    # Draw a line plot with 95% confidence intervals
    sns.lineplot(data=df, x='Bin_Center', y='Advantage', marker='o', color='teal', linewidth=2.5)
    
    # Add a horizontal line at y=0 (Neutral advantage)
    plt.axhline(0, color='black', linestyle='--', alpha=0.6)
    
    plt.title(f"Mean Policy Advantage by Clinician Performance ({config_manager.model_type.upper()})")
    plt.xlabel("Clinician Q-Value (Binned Deciles)")
    plt.ylabel("Mean RL Advantage")
    plt.grid(True, alpha=0.2)

    save_path = os.path.join(model_folder, f"binned_advantage_trend_{critic_type}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    # plt.show()


def plot_binned_advantage_by_mortality(advantage, q_clinician, mortality, model_folder, config_manager, critic_type="sarsa"):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({
        'Clinician_Q': q_clinician,
        'Advantage': advantage,
        'Mortality': mortality
    })

    # Divide Clinician Q-values into 10 equal-sized bins
    df['Q_Bin'] = pd.qcut(df['Clinician_Q'], q=10, labels=False, duplicates='drop')
    
    # Calculate the mean Q-value for each bin to use as the X-axis center
    bin_centers = df.groupby('Q_Bin')['Clinician_Q'].mean()
    df['Bin_Center'] = df['Q_Bin'].map(bin_centers)

    # create figure with 2 subplots: one for survivors and one for non-survivors
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, mortality_status in zip(axes, [0, 1]):
        subset = df[df['Mortality'] == mortality_status]
        sns.lineplot(data=subset, x='Bin_Center', y='Advantage', marker='o', ax=ax, color='teal', linewidth=2.5)
        ax.axhline(0, color='black', linestyle='--', alpha=0.6)
        ax.set_title(f"{'Survivors' if mortality_status == 0 else 'Non-Survivors'}")
        ax.set_xlabel("Clinician Q-Value (Binned Deciles)")
        ax.set_ylabel("Mean RL Advantage")
        ax.grid(True, alpha=0.2)
         
    save_path = os.path.join(model_folder, f"binned_advantage_trend_by_mortality_{critic_type}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    # plt.show()


def plot_actions_by_q_clinician_by_mortality(
        advantage, q_clinician, mortality, clinician_actions, rl_actions,
        model_folder):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({
        'Clinician_Q': q_clinician,
        'Advantage': advantage,
        'Mortality': mortality
    })

    # Divide Clinician Q-values into 10 equal-sized bins
    df['Q_Bin'] = pd.qcut(df['Clinician_Q'], q=10, labels=False, duplicates='drop')
    
    # Calculate the mean Q-value for each bin to use as the X-axis center
    bin_centers = df.groupby('Q_Bin')['Clinician_Q'].mean()
    df['Bin_Center'] = df['Q_Bin'].map(bin_centers)

    # create figure with 2 subplots: one for survivors and one for non-survivors
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, mortality_status in zip(axes, [0, 1]):
        subset = df[df['Mortality'] == mortality_status]
        sns.lineplot(data=subset, x='Bin_Center', y='Advantage', marker='o', ax=ax, color='teal', linewidth=2.5)
        ax.axhline(0, color='black', linestyle='--', alpha=0.6)
        ax.set_title(f"{'Survivors' if mortality_status == 0 else 'Non-Survivors'}")
        ax.set_xlabel("Clinician Q-Value (Binned Deciles)")
        ax.set_ylabel("Mean RL Advantage")
        ax.grid(True, alpha=0.2)
         
    save_path = os.path.join(model_folder, f"binned_advantage_trend_by_mortality_{critic_type}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    # plt.show()



# def rescue_rate(advantage, q_clinician, model_folder, critic_type="sarsa"):
# # Create the mask for that 10th bin (lowest performance)
#     bottom_10_mask = q_clinician < 9.0 

#     # What % of these 'deteriorating' cases did the AI improve upon?
#     improvements = advantage[bottom_10_mask] > 0
#     rescue_rate = 100 * improvements.mean().item()

#     print(f"In the bottom 10% of cases, RL proposed a better action than the clinician {rescue_rate:.1f}% of the time.")


def plot_journal_advantage(advantage, q_clinician, model_folder, critic_type="sarsa"):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.DataFrame({'Q_Clinician': q_clinician, 'Advantage': advantage})
    # Use deciles
    df['Decile'] = pd.qcut(df['Q_Clinician'], 10, labels=False, duplicates='drop')
    
    plt.figure(figsize=(6, 4))
    sns.set_style("whitegrid")
    
    # Plotting the mean advantage per decile with 95% Confidence Interval
    ax = sns.lineplot(data=df, x='Decile', y='Advantage', 
                      marker='o', markersize=8, linewidth=3, color='#2c3e50')
    
    # Highlight the 'Rescue' zone
    plt.axvspan(-0.5, 0.5, color='red', alpha=0.1, label='Critical/Rescue Zone')
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    
    # Custom Labels for Deciles
    plt.xticks(range(10), [f"D{i+1}" for i in range(10)])
    
    plt.title("RL Policy Value-Add Across Patient Severity Deciles", fontsize=15, pad=20)
    plt.xlabel("Patient Outcome Decile (D1 = Highest Risk of Mortality)", fontsize=12)
    plt.ylabel("Mean RL Advantage (SARSA Jury Score)", fontsize=12)
    
    # Annotate the 10th bin
    plt.annotate(f'Significant Improvement\n(+{df[df.Decile==0].Advantage.mean():.3f})', 
                 xy=(0, df[df.Decile==0].Advantage.mean()), 
                 xytext=(1.5, 0.04),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, f"journal_advantage_deciles_{critic_type}.png"), dpi=300)
    plt.close()



def plot_publication_deciles(advantage, q_clinician, model_folder, critic_type="sarsa"):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    # 1. Prepare Data
    df = pd.DataFrame({'Q': q_clinician, 'Advantage': advantage})
    # Sort into deciles (D1 = lowest Q/highest risk)
    df['Decile'] = pd.qcut(df['Q'], 10, labels=[f"D{i+1}" for i in range(10)])
    
    # 2. Calculate Stats per Decile
    stats_list = []
    for name, group in df.groupby('Decile', observed=True):
        mean = group['Advantage'].mean()
        sem = stats.sem(group['Advantage'])
        ci95 = sem * 1.96
        stats_list.append({
            'Decile': name,
            'Mean_Adv': mean,
            'CI_Lower': mean - ci95,
            'CI_Upper': mean + ci95,
            'Count': len(group)
        })
    stats_df = pd.DataFrame(stats_list)

    # 3. Plotting
    plt.figure(figsize=(10, 6))
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("ticks")

    # Draw the main trend line with error bars
    plt.errorbar(stats_df['Decile'], stats_df['Mean_Adv'], 
                 yerr=[stats_df['Mean_Adv'] - stats_df['CI_Lower'], 
                       stats_df['CI_Upper'] - stats_df['Mean_Adv']],
                 fmt='o-', color='#1a5276', ecolor='#a9cce3', 
                 capsize=5, linewidth=2, markersize=8, label='Mean Advantage (95% CI)')

    # Add baseline
    plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    
    # Highlight the Critical Decile
    plt.gca().get_xticklabels()[0].set_color('red') 
    plt.gca().get_xticklabels()[0].set_weight('bold')

    plt.title("Clinical Value-Add by Patient Risk Decile", fontsize=14, pad=15)
    plt.ylabel("Policy Advantage ($\Delta Q$)", fontsize=12)
    plt.xlabel("Patient Severity Deciles (D1 = Highest Mortality Risk)", fontsize=12)
    
    # Clean up spines
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{model_folder}/publication_decile_stats_{critic_type}.png", dpi=300)
    plt.close()
    
    return stats_df


def run_decile_statistics(advantage, q_clinician):
    import pandas as pd
    from scipy import stats
    df = pd.DataFrame({'Adv': advantage, 'Q': q_clinician})
    df['Decile'] = pd.qcut(df['Q'], 10, labels=[f"D{i+1}" for i in range(10)])
    
    results = []
    # We apply Bonferroni correction here (alpha/10)
    alpha_corrected = 0.05 / 10
    
    for name, group in df.groupby('Decile', observed=True):
        # 1. One-sample t-test: Is mean Advantage > 0?
        t_stat, p_val = stats.ttest_1samp(group['Adv'], 0)
        
        results.append({
            'Decile': name,
            'Mean Advantage': group['Adv'].mean(),
            'P-Value': p_val,
            'Significant?': p_val < alpha_corrected,
            '95% CI': stats.t.interval(0.95, len(group['Adv'])-1, 
                                       loc=group['Adv'].mean(), 
                                       scale=stats.sem(group['Adv']))
        })

    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df



def get_trajectory_data(processor, traj_start_inds, stay_id, as_tensor=True):

    stay_ids = processor.dynamic_deephit_dataset["stay_ids"]
    # get index of stay_id in stay_ids
    stay_idx = np.where(stay_ids == stay_id)[0][0]
    indices = np.arange(traj_start_inds[stay_idx], traj_start_inds[stay_idx+1])
    states = processor.rl_dataset["states"][indices]
    actions = processor.rl_dataset["actions"][indices]
    rewards = processor.rl_dataset["rewards"][indices]
    next_states = processor.rl_dataset["next_states"][indices]
    next_actions = processor.rl_dataset["next_actions"][indices]
    dones = processor.rl_dataset["dones"][indices]
    mortalities = processor.rl_dataset["mortality"][indices]

    if not as_tensor:
        return states, actions, rewards, next_states, next_actions, dones, mortalities, indices

    # convert to torch tensors
    states = torch.as_tensor(states, dtype=torch.float32).to('cuda')
    actions = torch.as_tensor(actions, dtype=torch.float32).to('cuda')
    rewards = torch.as_tensor(rewards, dtype=torch.float32).to('cuda')
    next_states = torch.as_tensor(next_states, dtype=torch.float32).to('cuda')
    next_actions = torch.as_tensor(next_actions, dtype=torch.float32).to('cuda')
    dones = torch.as_tensor(dones, dtype=torch.float32).to('cuda')
    mortalities = torch.as_tensor(mortalities, dtype=torch.float32).to('cuda')

    return states, actions, rewards, next_states, next_actions, dones, mortalities, indices



def generate_model_outputs(
        test_processor, 
        batch_size=1024, 
        model=None,
        actions_to_use=None,
        model_type="critic",
        reduction='mean',
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
    """
    model types are critic, actor, value_network. If critic is provided, it will generate q_values and q_values_shuffled.
    """
    
    # get the actions and get their standard deviation
    all_actions = test_processor.rl_dataset["actions"]
    action_dim = all_actions.shape[1]
    
    action_std = torch.as_tensor(all_actions.std(axis=0), dtype=torch.float32).to(device)
    print(f"Action std: {action_std}")    
    
    # loop through trajectories in test_processor.rl_dataset and evaluate the model
    traj_end_inds = np.where(test_processor.rl_dataset["dones"] == 1)[0]
    traj_start_inds = np.concatenate(([0], traj_end_inds + 1))
    stay_ids = test_processor.dynamic_deephit_dataset["stay_ids"]
    
    # create storage
    if model_type == "critic":
        model_outputs = torch.zeros(test_processor.rl_dataset["states"].shape[0], dtype=torch.float32).to(device)
    elif model_type == "actor":
        model_outputs = torch.zeros(test_processor.rl_dataset["states"].shape[0], action_dim, dtype=torch.float32).to(device)
    elif model_type == "value_network":
        model_outputs = torch.zeros(test_processor.rl_dataset["states"].shape[0], dtype=torch.float32).to(device)

    for stay_id in stay_ids:
        states, traj_actions, rewards, next_states, next_actions, dones, mortalities, indices = \
            get_trajectory_data(test_processor, traj_start_inds, stay_id, as_tensor=True) 
        
        if actions_to_use is not None:
            # use the provided actions instead of the ones from the dataset
            actions = actions_to_use[indices].to(device)
        else:
            actions = traj_actions
        
        n_batches = int(np.ceil(states.shape[0] / batch_size))
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, states.shape[0])
            states_batch = states[start_idx:end_idx]
            actions_batch = actions[start_idx:end_idx]

            # run inference
            with torch.no_grad():
                if model_type == "critic":
                    q1, q2 = model(states_batch, actions_batch)
                    if reduction == 'mean':
                        model_outputs[indices[start_idx:end_idx]] = ((q1 + q2) / 2.0).flatten()
                    elif reduction == 'min':
                        model_outputs[indices[start_idx:end_idx]] = torch.min(q1, q2).flatten()
                elif model_type == "actor":
                    actor_actions = model(states_batch)
                    model_outputs[indices[start_idx:end_idx]] = actor_actions[0]
                    # probabilities of actions will have to be implemented here
                elif model_type == "value_network":
                    v_values = model(states_batch)
                    model_outputs[indices[start_idx:end_idx]] = v_values.flatten()

    return model_outputs


@torch.no_grad()
def compute_paired_advantage_and_noise_floor(critic, actor, test_processor,
                                              reduction='mean', n_shuffles=100,
                                              seed=42, device='cuda'):
    
    q_clin = generate_model_outputs(
        test_processor, 
        model=critic,
        actions_to_use=None,
        model_type="critic",
        reduction=reduction
    ).cpu().numpy().ravel()

    actions = generate_model_outputs(
        test_processor,
        model=actor,
        actions_to_use=None,
        model_type="actor"
    )

    q_policy = generate_model_outputs(
        test_processor,
        model=critic,
        actions_to_use=actions,
        model_type="critic",
        reduction=reduction
    ).cpu().numpy().ravel()

    # real advantage
    advantage_real = q_policy - q_clin

    # generate noise floor by shuffling clinician actions and evaluating critic
    clinician_actions = test_processor.rl_dataset["actions"]
    rng = np.random.default_rng(seed)
    all_noise_adv = []
    for _ in range(n_shuffles):
        shuffled_actions = clinician_actions[torch.randperm(clinician_actions.size(0))]
        q_shuffled = generate_model_outputs(
            test_processor,
            model=critic,
            actions_to_use=shuffled_actions,
            model_type="critic",
            reduction=reduction
        ).cpu().numpy().ravel()
        all_noise_adv.append(q_shuffled - q_clin)
    
    advantage_noise = np.concatenate(all_noise_adv)
    
    return advantage_real, advantage_noise, q_policy, q_clin


def find_calibrated_threshold_relative(real_headroom, noise_headroom,
                                         min_relative_gap=0.5, n_thresholds=500,
                                         stable_window=20):
    """
    Find the threshold where the real distribution's CCDF exceeds the
    noise-floor distribution's CCDF by at least `min_relative_gap` as a
    RELATIVE (not absolute) amount. Robust on a log-scale CCDF, where
    absolute gaps shrink in the tail even when the relative separation
    is large and real.

    min_relative_gap: e.g. 0.5 means "real curve is at least 50% higher
                       than noise curve at this threshold"
    stable_window:    require the gap to hold for this many consecutive
                       threshold steps, so a single noisy crossing isn't
                       mistaken for genuine divergence
    """
    max_val = max(real_headroom.max(), noise_headroom.max())
    thresholds = np.linspace(0, max_val, n_thresholds)

    real_ccdf  = np.array([(real_headroom  > t).mean() for t in thresholds])
    noise_ccdf = np.array([(noise_headroom > t).mean() for t in thresholds])

    with np.errstate(divide='ignore', invalid='ignore'):
        relative_excess = np.where(
            noise_ccdf > 0,
            (real_ccdf - noise_ccdf) / noise_ccdf,
            np.where(real_ccdf > 0, np.inf, 0.0)
        )

    candidate_idx = np.where(relative_excess >= min_relative_gap)[0]

    if len(candidate_idx) == 0:
        return None, thresholds, real_ccdf, noise_ccdf, relative_excess

    for idx in candidate_idx:
        window_end = min(idx + stable_window, len(relative_excess))
        if window_end - idx == stable_window and \
           (relative_excess[idx:window_end] >= min_relative_gap).all():
            return thresholds[idx], thresholds, real_ccdf, noise_ccdf, relative_excess

    # Fall back to first crossing if no fully stable window found
    threshold = thresholds[candidate_idx[0]]
    return threshold, thresholds, real_ccdf, noise_ccdf, relative_excess


def plot_noise_floor_comparison(real_dist, noise_dist, label="headroom",
                                  min_relative_gap=0.5, stable_window=20,
                                  save_dir=None):

    threshold, thresholds, real_ccdf, noise_ccdf, relative_excess = \
        find_calibrated_threshold_relative(
            real_dist, noise_dist,
            min_relative_gap=min_relative_gap, stable_window=stable_window
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: CCDF comparison with calibrated threshold marked ---
    ax = axes[0]
    for d, series_label, color in [
        (real_dist,  'Real (e.g. policy vs clinician)', 'tab:blue'),
        (noise_dist, 'Noise floor (random baseline)', 'tab:red'),
    ]:
        d_sorted = np.sort(d)
        ccdf = 1.0 - np.arange(len(d_sorted)) / len(d_sorted)
        ax.plot(d_sorted, ccdf, lw=2, label=series_label, color=color)

    ax.set_yscale('log')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Fraction of samples > threshold')
    ax.set_title(f'Noise floor calibration — {label}')

    if threshold is not None:
        ax.axvline(threshold, color='k', ls=':', lw=1.5,
                   label=f'Calibrated threshold = {threshold:.4f}')
        pct_above = (real_dist > threshold).mean() * 100
        ax.annotate(f'{pct_above:.2f}% of samples\nabove threshold',
                    xy=(threshold, max((real_dist > threshold).mean(), 1e-5)),
                    xytext=(15, 15), textcoords='offset points', fontsize=8,
                    arrowprops=dict(arrowstyle='->', lw=0.8))

    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    # --- Right: relative excess, showing where it stabilises ---
    ax2 = axes[1]
    ax2.plot(thresholds, relative_excess, lw=2, color='tab:purple')
    ax2.axhline(min_relative_gap, color='gray', ls='--', lw=1,
                label=f'Min relative gap = {min_relative_gap}')
    if threshold is not None:
        ax2.axvline(threshold, color='k', ls=':', lw=1.5,
                    label=f'Calibrated threshold = {threshold:.4f}')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Relative excess: (real − noise) / noise')
    ax2.set_title('Relative divergence between real and noise-floor CCDFs')
    finite_excess = relative_excess[np.isfinite(relative_excess)]
    if finite_excess.size > 0:
        ax2.set_ylim(0, np.nanpercentile(finite_excess, 95) * 1.2)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"noise_floor_{label}.png"),
                    dpi=150, bbox_inches='tight')
        
    plt.close(fig)

    return fig, axes, threshold




 
def plot_actions_by_q_clinician_by_mortality(
    advantage, q_clinician, mortality, clinician_actions, policy_actions,
    save_dir=None, n_q_bins=5, deadband=0.02, action_names=("fluid", "vaso")
):
    """
    For each (mortality x Q-bin) cell, compare clinician vs policy dosing,
    and show the advantage distribution and signed direction of change.
 
    advantage:          (N,) array — per-timestep advantage (e.g. SARSA-evaluated)
    q_clinician:        (N,) array — per-timestep Q(s, a_clinician), used to bin severity
    mortality:          (N,) array — 0 = survivor, 1 = non-survivor
    clinician_actions:  (N, 2) array — clinician fluid/vaso doses
    policy_actions:     (N, 2) array — policy fluid/vaso doses
    n_q_bins:           number of quantile bins to split q_clinician into, PER mortality group
    deadband:           +/- range around zero treated as "about the same" for direction analysis
    """
    advantage = np.asarray(advantage).ravel()
    q_clinician = np.asarray(q_clinician).ravel()
    mortality = np.asarray(mortality).ravel()
    clinician_actions = np.asarray(clinician_actions)
    policy_actions = np.asarray(policy_actions)
 
    os.makedirs(save_dir, exist_ok=True) if save_dir else None
 
    mortality_groups = [("Survivors", mortality == 0), ("Non-survivors", mortality == 1)]
 
    # ------------------------------------------------------------------
    # Build per-cell summary table (printed + returned), bins computed
    # SEPARATELY within each mortality group so bins reflect that group's
    # own Q-distribution rather than a shared, potentially mismatched range
    # ------------------------------------------------------------------
    summary_rows = []
    for group_label, group_mask in mortality_groups:
        q_group = q_clinician[group_mask]
        if q_group.size < n_q_bins * 5:
            continue  # not enough data to bin meaningfully
 
        bin_edges = np.quantile(q_group, np.linspace(0, 1, n_q_bins + 1))
        bin_edges[-1] += 1e-8  # ensure max value falls into last bin
        bin_idx_group = np.digitize(q_group, bin_edges[1:-1], right=True)
 
        adv_group = advantage[group_mask]
        clin_group = clinician_actions[group_mask]
        pol_group = policy_actions[group_mask]
 
        for b in range(n_q_bins):
            cell_mask = bin_idx_group == b
            n_cell = cell_mask.sum()
            if n_cell < 5:
                continue
 
            row = {
                "mortality_group": group_label,
                "q_bin": b,
                "q_bin_range": (bin_edges[b], bin_edges[b + 1]),
                "n": int(n_cell),
                "mean_advantage": float(adv_group[cell_mask].mean()),
            }
 
            for d, name in enumerate(action_names):
                clin_d = clin_group[cell_mask, d]
                pol_d = pol_group[cell_mask, d]
                diff_d = pol_d - clin_d
 
                row[f"{name}_clinician_mean"] = float(clin_d.mean())
                row[f"{name}_policy_mean"] = float(pol_d.mean())
                row[f"{name}_clinician_pct_nonzero"] = float((clin_d > 1e-4).mean() * 100)
                row[f"{name}_policy_pct_nonzero"] = float((pol_d > 1e-4).mean() * 100)
 
                clin_nonzero = clin_d[clin_d > 1e-4]
                pol_nonzero = pol_d[pol_d > 1e-4]
                row[f"{name}_clinician_mean_given_nonzero"] = (
                    float(clin_nonzero.mean()) if clin_nonzero.size > 0 else float("nan")
                )
                row[f"{name}_policy_mean_given_nonzero"] = (
                    float(pol_nonzero.mean()) if pol_nonzero.size > 0 else float("nan")
                )
 
                row[f"{name}_pct_more"] = float((diff_d > deadband).mean() * 100)
                row[f"{name}_pct_less"] = float((diff_d < -deadband).mean() * 100)
                row[f"{name}_pct_same"] = float((np.abs(diff_d) <= deadband).mean() * 100)
                row[f"{name}_mean_diff"] = float(diff_d.mean())
 
            summary_rows.append(row)
 
    # Print a readable summary table
    print(f"{'Group':<14} {'Qbin':<5} {'N':<7} {'AdvMean':<9} "
          + "".join([f"{n}_clin{'':<3}{n}_pol{'':<3}{n}_%more{'':<2}{n}_%less{'':<3}"
                      for n in action_names]))
    for row in summary_rows:
        line = (f"{row['mortality_group']:<14} {row['q_bin']:<5} {row['n']:<7} "
                 f"{row['mean_advantage']:<9.4f} ")
        for name in action_names:
            line += (f"{row[f'{name}_clinician_mean']:<9.3f}"
                      f"{row[f'{name}_policy_mean']:<9.3f}"
                      f"{row[f'{name}_pct_more']:<8.1f}"
                      f"{row[f'{name}_pct_less']:<8.1f}")
        print(line)
 
    # ------------------------------------------------------------------
    # FIGURE 1 — dosing comparison (clinician vs policy mean dose, and
    # % nonzero) across Q-bins, one row per action dimension, one column
    # per mortality group
    # ------------------------------------------------------------------
    n_actions = len(action_names)
    fig1, axes1 = plt.subplots(n_actions, len(mortality_groups),
                                figsize=(6 * len(mortality_groups), 4 * n_actions),
                                squeeze=False)
 
    for col, (group_label, _) in enumerate(mortality_groups):
        group_rows = [r for r in summary_rows if r["mortality_group"] == group_label]
        if not group_rows:
            continue
        bins = [r["q_bin"] for r in group_rows]
 
        for row_i, name in enumerate(action_names):
            ax = axes1[row_i, col]
            clin_means = [r[f"{name}_clinician_mean"] for r in group_rows]
            pol_means = [r[f"{name}_policy_mean"] for r in group_rows]
 
            width = 0.35
            x = np.arange(len(bins))
            ax.bar(x - width / 2, clin_means, width, label="Clinician", color="tab:blue")
            ax.bar(x + width / 2, pol_means, width, label="Policy", color="tab:orange")
            ax.set_xticks(x)
            ax.set_xticklabels([f"Q-bin {b}" for b in bins], fontsize=8)
            ax.set_ylabel(f"Mean {name} dose")
            ax.set_title(f"{group_label} — {name}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
 
    plt.tight_layout()
    if save_dir:
        fig1.savefig(os.path.join(save_dir, "actions_by_qbin_mortality_dosing.png"),
                      dpi=150, bbox_inches='tight')
 
    # ------------------------------------------------------------------
    # FIGURE 2 — direction of change (% more / less / same) across Q-bins
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(n_actions, len(mortality_groups),
                                figsize=(6 * len(mortality_groups), 4 * n_actions),
                                squeeze=False)
 
    for col, (group_label, _) in enumerate(mortality_groups):
        group_rows = [r for r in summary_rows if r["mortality_group"] == group_label]
        if not group_rows:
            continue
        bins = [r["q_bin"] for r in group_rows]
 
        for row_i, name in enumerate(action_names):
            ax = axes2[row_i, col]
            pct_more = [r[f"{name}_pct_more"] for r in group_rows]
            pct_less = [r[f"{name}_pct_less"] for r in group_rows]
            pct_same = [r[f"{name}_pct_same"] for r in group_rows]
 
            x = np.arange(len(bins))
            ax.bar(x, pct_less, label="Less", color="tab:red")
            ax.bar(x, pct_same, bottom=pct_less, label="Same", color="tab:gray")
            ax.bar(x, pct_more,
                   bottom=[a + b for a, b in zip(pct_less, pct_same)],
                   label="More", color="tab:green")
            ax.set_xticks(x)
            ax.set_xticklabels([f"Q-bin {b}" for b in bins], fontsize=8)
            ax.set_ylabel("% of timesteps")
            ax.set_title(f"{group_label} — {name} direction of change")
            ax.legend(fontsize=8, loc='lower right')
            ax.grid(True, alpha=0.3, axis='y')
 
    plt.tight_layout()
    if save_dir:
        fig2.savefig(os.path.join(save_dir, "actions_by_qbin_mortality_direction.png"),
                     dpi=150, bbox_inches='tight')
 
    # ------------------------------------------------------------------
    # FIGURE 3 — advantage by Q-bin, both groups overlaid, to anchor the
    # dosing/direction findings against where advantage actually concentrates
    # ------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    for group_label, color in [("Survivors", "tab:blue"), ("Non-survivors", "tab:orange")]:
        group_rows = [r for r in summary_rows if r["mortality_group"] == group_label]
        if not group_rows:
            continue
        bins = [r["q_bin"] for r in group_rows]
        adv = [r["mean_advantage"] for r in group_rows]
        ax3.plot(bins, adv, marker='o', label=group_label, color=color)
 
    ax3.set_xlabel("Q-bin (0 = lowest clinician Q, within mortality group)")
    ax3.set_ylabel("Mean advantage")
    ax3.set_title("Advantage by Q-bin and mortality group")
    ax3.axhline(0, color='k', lw=0.5)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
 
    plt.tight_layout()
    if save_dir:
        fig3.savefig(os.path.join(save_dir, "advantage_by_qbin_mortality.png"),
                     dpi=150, bbox_inches='tight')
 
    return summary_rows, (fig1, fig2, fig3)




def evaluate(
        config_manager=None, 
        sarsa_critic=None, 
        sarsa_configuration=None,
        model_path=None,
        test_processor=None
    ):
    """
    1. calculate advantage of RL policy over clinician actions using the SARSA critic.
    2. calculate advantage of RL policy over clinician actions using the RL critic.
    3. TO IMPLEMENT calculatate advantage of RL policy over clinician actions using fitted Q evaluation.

    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config_manager is None:
    # parse arguments
        args = parse_args()

        # create configuration
        config_manager = ConfigManager(args)

    # 1. Load the "Jury" (SARSA Critic)
    # Using the exact same architecture as your training
    if sarsa_critic is None:
        sarsa_critic, sarsa_configuration = load_sarsa_critic(config_manager, model_path=model_path)
   
    # 2. Load the RL Policies (Actors)
    actor, critic, actor_configuration, actor_hyperparams, model_folder = load_actor(config_manager, model_path=model_path)
    
    batch_size = actor_hyperparams["batch_size"]

    # calculate advantage of RL policy over clinician actions using the SARSA critic.
    advantage_sarsa_real, advantage_sarsa_noise, q_policy_sarsa, q_clin_sarsa = compute_paired_advantage_and_noise_floor(
        sarsa_critic, actor, test_processor,
        reduction='mean', n_shuffles=100)
    
    # plot advantage
    sarsa_advantage_results = plot_real_and_noise_advantage(advantage_sarsa_real, advantage_sarsa_noise, title="SARSA", save_dir=model_folder)     
    plot_noise_floor_comparison(advantage_sarsa_real, advantage_sarsa_noise, label="SARSA Advantage", save_dir=model_folder)
          
    # calculate advantage of RL policy over clinician actions using the RL critic.
    advantage_rl_real, advantage_rl_noise, q_policy_rl, q_clin_rl = compute_paired_advantage_and_noise_floor(
        critic, actor, test_processor,
        reduction='min', n_shuffles=100)
    
    # plot advantage
    rl_advantage_results = plot_real_and_noise_advantage(advantage_rl_real, advantage_rl_noise, title="RL", save_dir=model_folder)     
    plot_noise_floor_comparison(advantage_rl_real, advantage_rl_noise, label="RL Advantage", save_dir=model_folder)

    # # calculate non-parametric t-test to see if advantage is significantly greater than 0       
    # t_stat, p_value = stats.ttest_1samp(advantage, 0, alternative='greater')
    # print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

    plot_advantage_against_clinician_q_values(advantage_sarsa_real, q_clin_sarsa, model_folder, config_manager, critic_type="sarsa")

    plot_binned_advantage(advantage_sarsa_real, q_clin_sarsa, model_folder, config_manager, critic_type="sarsa")    
    plot_binned_advantage_by_mortality(advantage_sarsa_real, q_clin_sarsa, test_processor.rl_dataset["mortality"], model_folder, config_manager, critic_type="sarsa")
    stats_summary = plot_publication_deciles(advantage_sarsa_real, q_clin_sarsa, model_folder, critic_type="sarsa")

    # plot actions by binned advantage by mortality
    actions = generate_model_outputs(
        test_processor,
        model=actor,
        actions_to_use=None,
        model_type="actor"
    ).cpu().numpy()

    plot_actions_by_q_clinician_by_mortality(
        advantage_sarsa_real, q_clin_sarsa, test_processor.rl_dataset["mortality"], 
        test_processor.rl_dataset["actions"], actions, model_folder)
    
    


    # rescue_rate(advantage_sarsa_real, q_clin_sarsa, model_folder, critic_type="sarsa")

    plot_journal_advantage(advantage_sarsa_real, q_clin_sarsa, model_folder, critic_type="sarsa")

    plot_publication_deciles(advantage_sarsa_real, q_clin_sarsa, model_folder, critic_type="sarsa")

    run_decile_statistics(advantage_sarsa_real, q_clin_sarsa)
    return sarsa_advantage_results, rl_advantage_results

def evaluate_multiple_models(rl_model_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # parse arguments
    args = parse_args()

    # create configuration
    config_manager = ConfigManager(args)

    # 1. Load the "Jury" (SARSA Critic)
    # Using the exact same architecture as your training
    sarsa_critic, sarsa_configuration, sarsa_hyperparams = load_sarsa_critic(config_manager)

    # load the data
    train_processor, _, test_processor = create_processor_from_shards(sarsa_configuration, normalize=True, set_to_return='all')

    # process the reward
    health_score_median = train_processor.process_reward(config_manager, sarsa_hyperparams)
    test_processor.process_reward(config_manager, sarsa_hyperparams, health_score_median=health_score_median)
    
    # create rl dataset
    batch_size = 1024
    train_processor, test_processor, _ = create_test_rl_dataset(train_processor, test_processor, config_manager, sarsa_hyperparams)

    # get the model directories
    rl_path = os.path.join(config_manager.paths['rl_models_dir'], config_manager.rl_type, 
                    f"split_{config_manager.split_num}", config_manager.cohort, config_manager.actions,
                    config_manager.reward_type, f"discount_{config_manager.discount}")
    rl_folders = [f for f in os.listdir(rl_path) if os.path.isdir(os.path.join(rl_path, f))]
    if "final_model" in rl_folders or "*_run" in rl_folders:
        rl_folders.remove("final_model")
        rl_folders = [f for f in rl_folders if not f.endswith("_run")]

    sarsa_advantage_results =[]
    rl_advantage_results = []

    for rl_folder in rl_folders:       
        model_path = os.path.join(rl_path, rl_folder)
        
        # load the configuration
        configuration_path = os.path.join(model_path, "config_manager.pkl")
        model_cfg = load_configuration(configuration_path)

        # find directories within model_path that begin with "local" or "unknown"
        model_subfolders = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f)) and (f.startswith("local") or f.startswith("unknown"))]

        for subfolder in model_subfolders:
            if rl_model_dir is not None:
                if subfolder != rl_model_dir:
                    continue

            model_subfolder_path = os.path.join(model_path, subfolder)
            
            print(f"Evaluating model in folder: {model_subfolder_path}")

            sarsa_advantage_result, rl_advantage_result = evaluate(
                config_manager=model_cfg, 
                sarsa_critic=sarsa_critic, 
                sarsa_configuration=sarsa_configuration,
                model_path=model_subfolder_path,
                test_processor=test_processor
                )

            sarsa_advantage_results.append(sarsa_advantage_result)
            rl_advantage_results.append(rl_advantage_result)

    # convert results to dataframe and save as csv
    sarsa_advantage_df = pd.DataFrame(sarsa_advantage_results)
    rl_advantage_df = pd.DataFrame(rl_advantage_results)

    sarsa_advantage_df.to_csv(os.path.join(rl_path, 'sarsa_advantage_results.csv'), index=False)
    rl_advantage_df.to_csv(os.path.join(rl_path, 'rl_advantage_results.csv'), index=False)


def main():
    # evaluate()
    # evaluate_multiple_models()
    evaluate_multiple_models(rl_model_dir="unknown_cx3-20-5_20260619-025044_7ac09ff8")  # Set to None to evaluate all models, or specify a model directory name


if __name__ == "__main__":
    main()