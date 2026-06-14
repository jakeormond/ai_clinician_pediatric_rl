import argparse
import json
import os
from typing_extensions import final

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from ..config_manager import ConfigManager   
from .models.basic_q import Critic
from .training import create_processor_from_shards
from .replay_buffer import RLDataset
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from sklearn.metrics import roc_auc_score
from pathlib import Path
import pickle
from .training import create_buffer
from .models.shared_components import TrainerUtils
from ..setup_environment import load_paths
import config_manager


RL_MODELS_PATH = "/home/jake/Documents/AI_Clinician/models/rl_models/choc" 


def load_sarsa_critic(config_manager = None, model_path = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_path is None and config_manager is not None:
        model_path = os.path.join(config_manager.paths['rl_models_dir'], config_manager.model_type, 
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
            hidden_dim = results["hyperparameters"]["hidden_dim"]

    sarsa_critic = Critic(512,2, hidden_dim=hidden_dim).to(device)
    sarsa_critic.load_state_dict(torch.load(final_model_path, weights_only=True))
    sarsa_critic.eval()
    return sarsa_critic



    return actor, hyperparams, model_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RL models on SARSA CHOC")

    # Keyword Arguments (Match your launch.json)
    parser.add_argument("--hospital", type=str, default="choc")
    parser.add_argument("--experiment_type", type=str, default="reinforcement_learning")
    parser.add_argument("--experiment_variant", type=str, default="hyperparameter_search_evaluation")
    parser.add_argument("--model_type", type=str, default="basic_q")
    parser.add_argument("--mode", type=str, default="single_split")
    parser.add_argument("--split_num", type=int, default=0)
    parser.add_argument("--cohort", type=str, default="sepsis")
    parser.add_argument("--actions", type=str, default="vis-intake")
    
    # The Flag (Action: store_true means if it's in the args, it becomes True)
    parser.add_argument("--testing", action="store_true", help="Run in test mode")

    return parser.parse_args()










@torch.no_grad()
def hyperparameter_search_evaluation(args=None):
    
    search_results = []
    
    models_dir = os.path.join(RL_MODELS_PATH, 'basic_q', 
        f"split_{args.split_num}", args.cohort, args.actions)

    # get the model directories
    folder_paths = []
    search_folders = [f for f in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, f))]
    
    search_num = 0
    
    for f in search_folders:
        # load the config_manager.pkl file from the model folder
        config_manager_path = os.path.join(models_dir, f, "config_manager.pkl")
        if not os.path.exists(config_manager_path):
            print(f"Config manager not found in folder: {f}, skipping evaluation.")
            continue
        else:
            print(f"Loading config manager from folder: {f}")
            with open(config_manager_path, "rb") as cfg_file:
                config_manager = pickle.load(cfg_file)
        
        # set paths 
        config_manager.paths, config_manager.environment = load_paths(config_manager.hospital)
        if config_manager.environment == "local":
            config_manager.config_dir = Path(config_manager.paths['config_dir'])
        config_manager.set_data()
        config_manager.reset_cohort(config_manager.cohort)


        if search_num == 0:
            # load data
            train_processor, val_processor, _ = create_processor_from_shards(config_manager, normalize=True, set_to_return='all')
        
        inner_folders = [inner_f for inner_f in os.listdir(os.path.join(models_dir, f)) 
                         if os.path.isdir(os.path.join(models_dir, f, inner_f))]
        
        for inner_f in inner_folders:
            model_path = os.path.join(models_dir, f, inner_f, "final_model.pt")
            
            print (f"Evaluating model: {model_path}")

            print(f"Evaluating model in folder: {model_path}")
        
            # load hyperprameters from results.json
            results_path = os.path.join(models_dir, f, inner_f, "results.json")
            with open(results_path, "r") as file:
                results = json.load(file)
                hyperparams = results["hyperparameters"]
                        
            # process the reward
            health_score_median = train_processor.process_reward(config_manager, hyperparams)
            val_processor.process_reward(config_manager, hyperparams, health_score_median=health_score_median)
            
            # Create replay buffers - THESE CALLS CREATE THE RL DATASET IN THE PROCESSORS!!!!!!!!!!!!!!!!!!!!!!!!
            batch_size = 1024
            training_buffer, scaler = create_buffer(
                train_processor, 
                batch_size, 
                hyperparams,
                action_names=config_manager.data_config['actions'],
            )
   
            # create val buffer
            val_buffer, _ = create_buffer(
                val_processor,
                batch_size, 
                hyperparams,
                scaler=scaler, 
                action_names=config_manager.data_config['actions'],
                shuffle=False
            )
                        
            state_dim = val_processor.rl_dataset["states"].shape[1]
            action_dim = val_processor.rl_dataset["actions"].shape[1]

            sarsa_model = Critic(
                state_dim, 
                action_dim, 
                hidden_dim=hyperparams['hidden_dim'], 
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sarsa_model.to(device)
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            state_dict = checkpoint['critic_state_dict']
            sarsa_model.load_state_dict(state_dict)
            sarsa_model.eval()
        
    
            q1_values = []
            q2_values = []
            target_q_values = []
            mortality_values = []
            for batch_idx, batch in enumerate(val_buffer):
                state, action, reward, next_state, next_action, done, mortality = TrainerUtils._unpack_batch(batch, 'cuda')
                not_done = 1 - done
                mortality_values.append(mortality.detach().cpu().numpy())

                q1, q2 = sarsa_model(state, action)
                
                # Compute the target Q value
                target_Q1, target_Q2 = sarsa_model(next_state, next_action)
                target_Q = (target_Q1 + target_Q2) / 2.0
                # target_Q = torch.min(target_Q1, target_Q2)  # TD3 style: take the minimum of the two critics to mitigate overestimation bias
                target_Q = reward + not_done * hyperparams['discount'] * target_Q	

                q1_values.append(q1.detach())
                q2_values.append(q2.detach())
                target_q_values.append(target_Q.detach())

            q1_values = torch.cat(q1_values, dim=0)
            q2_values = torch.cat(q2_values, dim=0)
            target_q_values = torch.cat(target_q_values, dim=0)

            current_q_values = (q1_values + q2_values) / 2.0

            # bellman residual calculation
            bellman_residual = F.mse_loss(current_q_values, target_q_values)
            normalized_bellman_residual = bellman_residual / (target_Q.var() + 1e-8) 

            # Q1-Q2 uncertainty calculation
            disagreement = F.mse_loss(q1_values, q2_values)
            mean_Q   = (q1_values + q2_values) / 2.0
            q_var    = mean_Q.var()
            normalized_uncertainty = (disagreement / (q_var + 1e-8)).item()

            # calculate auc for current_q_values and mortality_values
            mortality_values = np.concatenate(mortality_values, axis=0)
            y_true = 1 - mortality_values  # Assuming mortality=1 means death, we want to predict survival
            y_score = current_q_values.detach().cpu().numpy().ravel()         
            auc = roc_auc_score(y_true, y_score)
            print(f"Model: {model_path}, AUC: {auc:.4f}")

            search_results.append({
                "model_path": model_path,
                "discount": hyperparams['discount'],
                "learning_rate": hyperparams['learning_rate'],
                "tau": hyperparams['tau'],
                "hidden_dim": hyperparams['hidden_dim'],
                "bellman_residual": bellman_residual.item(),
                "normalized_bellman_residual": normalized_bellman_residual.item(),
                "disagreement": disagreement.item(),
                "normalized_uncertainty": normalized_uncertainty,
                "auc": auc
            })
        
    # concatenate search_results into a dataframe and save to csv
    df = pd.DataFrame(search_results)
    df.to_csv(os.path.join(models_dir, "hyperparameter_search_evaluation_results.csv"), index=False)    

    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # parse arguments
    args = parse_args()

    if args.experiment_variant == "hyperparameter_search_evaluation":
        hyperparameter_search_evaluation(args)


if __name__ == "__main__":
    main()