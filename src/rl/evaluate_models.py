import argparse
import json
import os

import torch
import numpy as np
from ..config_manager import ConfigManager   
from .models.basic_q import Critic
from .models.td3_bc import Actor as TD3Actor
from .models.conservative_q_learning import Actor as CQLActor
# from .training import create_processor_from_shards
from .replay_buffer import RLDataset
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats


# rl_models_path = "/home/jake/Documents/AI_Clinician/rl_models/" 


def load_sarsa_critic(config_manager, model_path = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_path is None:
        model_path = os.path.join(config_manager.paths['rl_models_dir'], config_manager.model_type, 
                        f"split_{config_manager.split_num}", config_manager.cohort, config_manager.actions, "final_model")
    
        # get folder inside model_path
        model_folder = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))][0]
    
    final_model_path = os.path.join(model_path, model_folder, "final_model.pt")

    # load results.json from model_folder, which has state_dim 
    results_path = os.path.join(model_path, model_folder, "results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
            hidden_dim = results["hyperparameters"]["hidden_dim"]

    sarsa_critic = Critic(512,2, hidden_dim=hidden_dim).to(device)
    sarsa_critic.load_state_dict(torch.load(final_model_path, weights_only=True))
    sarsa_critic.eval()
    return sarsa_critic


def load_actor(config_manager, model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_path is None:
        model_path = os.path.join(config_manager.paths['rl_models_dir'], config_manager.rl_type, 
                        f"split_{config_manager.split_num}", config_manager.cohort, config_manager.actions, "final_model")
    
        # get folder inside model_path
        model_folder = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))][0]
        model_path = os.path.join(model_path, model_folder)

    final_model_path = os.path.join(model_path, "final_model.pt")

    # load results.json from model_folder, which has state_dim 
    results_path = os.path.join(model_path, "results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
            hyperparams = results["hyperparameters"]
            # hidden_dim = hyperparams["hidden_dim"]

    if config_manager.rl_type == "td3_bc":
        actor = TD3Actor(512, 2).to(device)
    elif config_manager.rl_type == "conservative_q_learning":
        actor = CQLActor(512, 2).to(device)

    checkpoint = torch.load(final_model_path, map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()
    return actor, hyperparams, model_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RL models on SARSA CHOC")

    # Keyword Arguments (Match your launch.json)
    parser.add_argument("--hospital", type=str, default="choc")
    parser.add_argument("--rl_type", type=str, choices=["td3_bc", "conservative_q_learning", "bc"], required=True)
    parser.add_argument("--experiment_type", type=str, default="model_evaluation")
    parser.add_argument("--model_type", type=str, default="basic_q")
    parser.add_argument("--mode", type=str, default="single_split")
    parser.add_argument("--split_num", type=int, default=0)
    parser.add_argument("--cohort", type=str, default="sepsis")
    parser.add_argument("--actions", type=str, default="vis-intake")
    
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

    advantage = (q_rl - q_clinician) / 2
    mean_advantage = advantage.mean().item()
    print(f"Mean Advantage of RL Actions over Clinician Actions: {mean_advantage:.4f}")
    
    return advantage.cpu().numpy().flatten(), q_clinician.cpu().numpy().flatten(), q_rl.cpu().numpy().flatten()


def plot_advantage(advantage, model_folder, config_manager):
    # plot as kde plot
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(7, 4))
    sns.kdeplot(advantage, fill=True)
    plt.title(f"Advantage Distribution of {config_manager.rl_type.upper()} over Clinician Actions")
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
    
    results_dir = os.path.join(model_folder, "advantage_plot.png")
    plt.savefig(results_dir)
    plt.close()
    # plt.show()


def plot_advantage_against_clinician_q_values(advantage, q_clinician, model_folder, config_manager):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    
    plt.figure(figsize=(6, 4))

    # 1. Plot the data with a regression line
    # 'scatter_kws' thins out the points so the line is visible
    sns.regplot(x=q_clinician, y=advantage, 
                scatter_kws={'alpha':0.1, 's':2}, 
                line_kws={'color':'red', 'label':'Linear Trend'})

    # 2. Calculate Pearson correlation for the title
    r, p = stats.pearsonr(q_clinician, advantage)

    plt.title(f"Advantage vs Clinician Q-Values ({config_manager.rl_type.upper()})\n"
                f"Pearson R: {r:.3f}, p-value: {p:.3e}")
    plt.xlabel("Clinician Q-Values (Evaluated by SARSA)")
    plt.ylabel("RL Policy Advantage")
    plt.grid(True, alpha=0.3)

    results_dir = os.path.join(model_folder, "advantage_vs_clinician_q_values.png")
    plt.savefig(results_dir)
    plt.close()
    # plt.show()



def plot_binned_advantage(advantage, q_clinician, model_folder, config_manager):
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
    
    plt.title(f"Mean Policy Advantage by Clinician Performance ({config_manager.rl_type.upper()})")
    plt.xlabel("Clinician Q-Value (Binned Deciles)")
    plt.ylabel("Mean RL Advantage")
    plt.grid(True, alpha=0.2)

    save_path = os.path.join(model_folder, "binned_advantage_trend.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    # plt.show()


def rescue_rate(advantage, q_clinician):
# Create the mask for that 10th bin (lowest performance)
    bottom_10_mask = q_clinician < 9.0 

    # What % of these 'deteriorating' cases did the AI improve upon?
    improvements = advantage[bottom_10_mask] > 0
    rescue_rate = 100 * improvements.mean().item()

    print(f"In the bottom 10% of cases, RL proposed a better action than the clinician {rescue_rate:.1f}% of the time.")


def plot_journal_advantage(advantage, q_clinician, model_folder):
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
    plt.savefig(os.path.join(model_folder, "journal_advantage_deciles.png"), dpi=300)
    plt.close()



def plot_publication_deciles(advantage, q_clinician, model_folder):
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
    plt.savefig(f"{model_folder}/publication_decile_stats.png", dpi=300)
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


def evaluate(config_manager=None, sarsa_critic=None, model_path=None):
    from .training import create_processor_from_shards
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config_manager is None:
    # parse arguments
        args = parse_args()

        # create configuration
        config_manager = ConfigManager(args)

    # 1. Load the "Jury" (SARSA Critic)
    # Using the exact same architecture as your training
    if sarsa_critic is None:
        sarsa_critic = load_sarsa_critic(config_manager, model_path=model_path)
   
    # 2. Load the RL Policies (Actors)
    actor, hyperparams, model_folder = load_actor(config_manager, model_path=model_path)
    
    # load data
    train_processor, _, test_processor = create_processor_from_shards(config_manager, normalize=True, set_to_return='all')
    train_processor.process_reward(config_manager, hyperparams) # not sure this line is necessary???
    test_processor.process_reward(config_manager, hyperparams)

    # get the scaler for the dynamic deephit states
    action_names = config_manager.data_config['actions']
    scaler = train_processor.create_rl_dataset(states="model_outputs", scaler=None, action_names=action_names) # states argumen
    test_processor.create_rl_dataset(states="model_outputs", scaler=scaler, action_names=action_names)
    test_dataset = RLDataset(test_processor)

    batch_size = hyperparams["batch_size"]

    rl_actions = generate_actions(actor, test_dataset.states, batch_size=batch_size, device=device)

    advantage, q_clinician, q_rl = calculate_advantage(
        sarsa_critic, 
        test_dataset.states, 
        test_dataset.actions, 
        rl_actions, 
        device
        )   

    # plot advantage
    plot_advantage(advantage, model_folder, config_manager)
    
    # calculate non-parametric t-test to see if advantage is significantly greater than 0       
    t_stat, p_value = stats.ttest_1samp(advantage, 0, alternative='greater')
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

    
    plot_advantage_against_clinician_q_values(advantage, q_clinician, model_folder, config_manager)

    plot_binned_advantage(advantage, q_clinician, model_folder, config_manager)    

    stats_summary = plot_publication_deciles(advantage, q_clinician, model_folder)

    rescue_rate(advantage, q_clinician)

    plot_journal_advantage(advantage, q_clinician, model_folder)

    plot_publication_deciles(advantage, q_clinician, model_folder)

    run_decile_statistics(advantage, q_clinician)


def evaluate_multiple_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # parse arguments
    args = parse_args()

    # create configuration
    config_manager = ConfigManager(args)

    # 1. Load the "Jury" (SARSA Critic)
    # Using the exact same architecture as your training
    sarsa_critic = load_sarsa_critic(config_manager)

    # get the model directories
    rl_path = os.path.join(config_manager.paths['rl_models_dir'], config_manager.rl_type, 
                    f"split_{config_manager.split_num}", config_manager.cohort, config_manager.actions)
    rl_folders = [f for f in os.listdir(rl_path) if os.path.isdir(os.path.join(rl_path, f))]
    if "final_model" in rl_folders:
        rl_folders.remove("final_model")

    for rl_folder in rl_folders:
        model_path = os.path.join(rl_path, rl_folder)
        
        # check if advantage_plot.png already exists in the model_path, if it does, skip evaluation for that model
        advantage_plot_path = os.path.join(model_path, "advantage_plot.png")
        if os.path.exists(advantage_plot_path):
            print(f"Advantage plot already exists for model in folder: {model_path}, skipping evaluation.")
            continue
        
        print(f"Evaluating model in folder: {model_path}")
        evaluate(config_manager, sarsa_critic=sarsa_critic, model_path=model_path)





def main():
    # evaluate()
    evaluate_multiple_models()


if __name__ == "__main__":
    main()