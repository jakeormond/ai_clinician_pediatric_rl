
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import asdict
from pathlib import Path
import os
import time


import pyrallis

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb


from ...config_manager import ConfigManager  # Import the config manager
from ...supervised_learning.dataset import Dataset
from ...utils.logging_utilities import wandb_init  # Add this import

# from src.data.data_processor import DataProcessor
# from src.rl.replay_buffer import ReplayBuffer


TensorBatch = List[torch.Tensor]


def keep_best_trajectories(dataprocessor, config_manager):

    fraction_to_keep = config_manager.frac
    discount = config_manager.training_config['discount']
   
    # check that the rewards are not terminal only, i.e. only -1, 0, +1
    if set(np.unique(dataprocessor.rl_dataset["rewards"])).issubset({-1, 0, 1}):
        raise ValueError("Rewards appear to be terminal only (-1, 0, +1). Use keep_positive_trajectories instead.")
    
    ids_by_trajectories = []
    returns = []
    cur_ids = []
    cur_return = 0
    reward_scale = 1.0

    for i, (reward, done) in enumerate(zip(dataprocessor.rl_dataset["rewards"], 
                                           dataprocessor.rl_dataset["terminated"])):
        cur_return += reward_scale * reward
        cur_ids.append(i)
        reward_scale *= discount     
        
        if done == 1.0:
            if len(cur_ids) <= config_manager.training_config['max_traj_len']:
                # don't use trajectories that are too long
                ids_by_trajectories.append(list(cur_ids))
                returns.append(cur_return)
            
            cur_ids = []
            cur_return = 0
            reward_scale = 1.0

    sort_ord = np.argsort(returns, axis=0)[::-1].reshape(-1)
    top_trajs = sort_ord[: max(1, int(fraction_to_keep * len(sort_ord)))]
    top_returns = np.array(returns)[top_trajs]
    print(f"mean top returns: {np.mean(top_returns)}")

    order = []
    for i in top_trajs:
        order += ids_by_trajectories[i]
    order = np.array(order)
    dataprocessor.rl_dataset["observations"] = dataprocessor.rl_dataset["observations"][order]
    dataprocessor.rl_dataset["actions"] = dataprocessor.rl_dataset["actions"][order]
    dataprocessor.rl_dataset["next_observations"] = dataprocessor.rl_dataset["next_observations"][order]
    dataprocessor.rl_dataset["rewards"] = dataprocessor.rl_dataset["rewards"][order]
    dataprocessor.rl_dataset["terminated"] = dataprocessor.rl_dataset["terminated"][order]

    return


def keep_positive_trajectories(dataprocessor, config_manager):
    """
    This is for the behavioural cloning using terminal rewards only. In
    this case, it's simply a matter of keeping only survival trajectories;
    non-survival trajectories are discarded.
    """
    max_traj_len = config_manager.training_config['max_traj_len']

    ids_by_trajectories = []
    returns = []
    cur_ids = []
    cur_return = 0
    reward_scale = 1.0

    for i, (reward, done) in enumerate(zip(dataprocessor.rl_dataset["rewards"], 
                                           dataprocessor.rl_dataset["terminated"])):
        cur_return += reward_scale * reward
        cur_ids.append(i)
        reward_scale *= dataprocessor.discount     
        
        if done == 1.0:
            if reward < 0:
                cur_ids = []
                continue
            
            elif reward > 0:
                if len(cur_ids) <= max_traj_len:
                    # don't use trajectories that are too long
                    ids_by_trajectories.append(list(cur_ids))
                returns.append(cur_return)
            
            cur_ids = []
            
    order = [item for sublist in ids_by_trajectories for item in sublist]    
    
    dataprocessor.rl_dataset["observations"] = dataprocessor.rl_dataset["observations"][order]
    dataprocessor.rl_dataset["actions"] = dataprocessor.rl_dataset["actions"][order]
    dataprocessor.rl_dataset["next_observations"] = dataprocessor.rl_dataset["next_observations"][order]
    dataprocessor.rl_dataset["rewards"] = dataprocessor.rl_dataset["rewards"][order]
    dataprocessor.rl_dataset["terminated"] = dataprocessor.rl_dataset["terminated"][order]

    return


class Actor(nn.Module):
    def __init__(
            self, 
            state_dim: int, 
            action_dim: int, 
            hidden_dim: List[int], 
            max_action: float,
            action_mode: str
            ):
        
        super(Actor, self).__init__()

        # Build network dynamically from hidden_dim list
        layers = []
        input_dim = state_dim

        # Add hidden layers
        for h_dim in hidden_dim:
            layers.extend([
                nn.Linear(input_dim, h_dim),
                nn.ReLU()
            ])
            input_dim = h_dim

        # Add final output layer
        layers.extend([
            nn.Linear(input_dim, action_dim),
            nn.Sigmoid()  # Keep your sigmoid activation
        ])

        self.net = nn.Sequential(*layers)
        self.max_action = max_action
        self.action_mode = action_mode

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if self.action_mode == 'binary_only':
            return self.net(state)  # Outputs in [0,1] range for binary actions
        elif self.action_mode == 'continuous_only':
            return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()
    

class MixedActionActor(nn.Module):
    def __init__(
            self, 
            state_dim: int, 
            dose_actions: int, 
            binary_actions: int, 
            hidden_dim: List[int], 
            max_action: float):
        
        super().__init__()
        
        # Build network dynamically from hidden_dim list
        layers = []
        input_dim = state_dim

        # Add hidden layers
        for h_dim in hidden_dim:
            layers.extend([
                nn.Linear(input_dim, h_dim),
                nn.ReLU()
            ])
            input_dim = h_dim

        # Create the shared network (this was missing!)
        self.shared_net = nn.Sequential(*layers)
        
        # Separate heads for different action types
        self.dose_head = nn.Sequential(
            nn.Linear(input_dim, dose_actions),
            nn.Sigmoid()  # Outputs in [0,1] range for doses
        )
        
        self.binary_head = nn.Sequential(
            nn.Linear(input_dim, binary_actions), 
            nn.Sigmoid()  # Outputs probabilities for binary actions
        )
        
        self.max_action = max_action
        self.dose_actions = dose_actions

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Shared features
        features = self.shared_net(state)
        
        # Different processing for different action types
        dose_outputs = self.dose_head(features) * self.max_action  # Scale doses
        binary_outputs = self.binary_head(features)  # Keep as probabilities
        
        # Concatenate outputs
        return torch.cat([dose_outputs, binary_outputs], dim=1)
    
    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        features = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dose_outputs = self.dose_head(self.shared_net(features)) * self.max_action
        binary_outputs = self.binary_head(self.shared_net(features))
        return torch.cat([dose_outputs, binary_outputs], dim=1).cpu().data.numpy().flatten()
    
class BC:
    def __init__(
        self,
        max_action: np.ndarray,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        device: str = "CUDA" if torch.cuda.is_available() else "CPU",
        action_mode: str = "mixed",
        dose_actions: int = 2,  #  Number of dose actions
        binary_actions: int = 2
    ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.max_action = max_action
        self.discount = discount
        self.action_mode = action_mode
        self.dose_actions = dose_actions
        self.binary_actions = binary_actions

        self.total_it = 0
        self.device = device

    
    def _get_state_and_action(self, batch: TensorBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        state, action, _, _, _ = batch
        
        state = state.to(self.device)
        action = action.to(self.device)
        
        return state, action
    

    def _compute_actor_loss(self, pi: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
      
        if self.action_mode == "binary_only":
            binary_loss = self._compute_binary_loss(pi, action)
            return binary_loss

        elif self.action_mode == "continuous_only":
            dose_loss = self._compute_dose_loss(pi, action)
            return dose_loss
    
        elif self.action_mode == "mixed":
            dose_predictions = pi[:, :self.dose_actions]      # First 2 actions (doses)
            binary_predictions = pi[:, self.dose_actions:] 

            dose_targets = action[:, :self.dose_actions]
            binary_targets = action[:, self.dose_actions:]

            dose_loss = self._compute_dose_loss(dose_predictions, dose_targets)
            binary_loss = self._compute_binary_loss(binary_predictions, binary_targets)

            # Combine losses (you can weight them differently)
            dose_weight = 1.0
            binary_weight = 1.0
            actor_loss = dose_weight * dose_loss + binary_weight * binary_loss

            return actor_loss, dose_loss, binary_loss
            
    def _compute_binary_loss(self, binary_predictions: torch.Tensor, binary_targets: torch.Tensor):
        binary_loss = F.binary_cross_entropy(binary_predictions, binary_targets)
        return binary_loss

    def _compute_dose_loss(self, dose_predictions: torch.Tensor, dose_targets: torch.Tensor):
        dose_loss = F.mse_loss(dose_predictions, dose_targets)
        return dose_loss

    
    def train(self, batch: TensorBatch) -> Dict[str, float]:
        
        log_dict = {}
        self.total_it += 1

        state, action = self._get_state_and_action(batch)

        # generate actions
        pi = self.actor(state)

        # Compute actor loss
        if self.action_mode in ["binary_only", "continuous_only"]:
            actor_loss = self._compute_actor_loss(pi, action)
            log_dict["actor_loss"] = actor_loss.item()

        elif self.action_mode == "mixed":
            actor_loss, dose_loss, binary_loss = self._compute_actor_loss(pi, action)        
            # Log individual losses for monitoring
            log_dict["actor_loss"] = actor_loss.item()
            log_dict["dose_loss"] = dose_loss.item() 
            log_dict["binary_loss"] = binary_loss.item()
        
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return log_dict
    

    def evaluate(self, batch: TensorBatch) -> Dict[str, float]:
        """
        Evaluate model with appropriate metrics for each action type.
        
        Args:
            replay_buffer: ReplayBuffer to sample evaluation data from
            batch_size: Number of samples to use for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.actor.eval()  # Set to evaluation mode
        
        state, action = self._get_state_and_action(batch)  
        
        with torch.no_grad():
            # Get predictions
            pi = self.actor(state)
            # check if pi contains NaNs or Infs
            if torch.isnan(pi).any() or torch.isinf(pi).any():
                raise ValueError("Actor network output contains NaNs or Infs.")
            
            if self.action_mode == "binary_only":
                overall_metrics = self._evaluate_binary(pi, action)
            
            elif self.action_mode == "continuous_only":
                overall_metrics = self._evaluate_continuous(pi, action)

            elif self.action_mode == "mixed":            
            
                # Split predictions and targets
                dose_pred = pi[:, :self.dose_actions]
                binary_pred = pi[:, self.dose_actions:]
                dose_target = action[:, :self.dose_actions]
                binary_target = action[:, self.dose_actions:]

                binary_metrics = self._evaluate_binary(binary_pred, binary_target)
                continuous_metrics = self._evaluate_continuous(dose_pred, dose_target)

                # Combine metrics
                actor_loss = (binary_metrics["eval/binary_bce(loss)"] + 
                            continuous_metrics["eval/dose_mse(loss)"])
                
                overall_metrics = {actor_loss: actor_loss, **binary_metrics, **continuous_metrics}
            
            return overall_metrics, action.cpu().numpy(), pi.cpu().numpy()

    
    def _evaluate_binary(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        binary_loss = self._compute_binary_loss(predictions, targets)

        # Binary accuracy (threshold at 0.5)
        binary_pred_classes = (predictions > 0.5).float()
        binary_accuracy = (binary_pred_classes == targets).float().mean()
        
        # Per-action accuracy for binary actions
        binary_acc_per_action = []
        for i in range(self.binary_actions):
            acc = (binary_pred_classes[:, i] == targets[:, i]).float().mean()
            binary_acc_per_action.append(acc) 

        binary_metrics = {# Overall metrics               
            # Binary action metrics
            "eval/binary_bce(loss)": binary_loss.item(),
            "eval/binary_accuracy": binary_accuracy.item(),
            "eval/binary_acc_action_0": binary_acc_per_action[0].item(),
            "eval/binary_acc_action_1": binary_acc_per_action[1].item()
        }

        return binary_metrics


    def _evaluate_continuous(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        dose_loss = self._compute_dose_loss(predictions, targets)
       
        # Dose action metrics (continuous regression)
        dose_mae = F.l1_loss(predictions, targets)

        dose_metrics = {# Overall metrics           
            # Dose action metrics
            "eval/dose_mse(loss)": dose_loss.item(),
            "eval/dose_mae": dose_mae.item(),
            "eval/dose_rmse": dose_loss.item() ** 0.5,        
        }

        return dose_metrics    
    
    
    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.total_it = state_dict["total_it"]


def get_trainer(config_manager, state_dim, action_dim, max_action, hyperparams):

    learning_rate = float(hyperparams['learning_rate'])

    action_mode = config_manager.action_mode

    if action_mode == 'binary_only' or action_mode == 'continuous_only':
        actor = Actor(
            state_dim=state_dim, 
            action_dim=action_dim, 
            hidden_dim=hyperparams['hidden_dim'], 
            max_action = 1.0,
            action_mode = action_mode  # max_action is 1.0 for binary
            ).to(config_manager.training_config['device'])
        
    elif action_mode == 'mixed':
        actor = MixedActionActor(
            state_dim=state_dim, 
            dose_actions=action_dim//2, 
            binary_actions=action_dim//2, 
            hidden_dim=hyperparams['hidden_dim'], 
            max_action = max_action
            ).to(config_manager.training_config['device'])
    
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), 
        lr=learning_rate
        )
    
    if action_mode == 'binary_only':
        dose_actions = 0
        binary_actions = action_dim
    elif action_mode == 'continuous_only':
        dose_actions = action_dim
        binary_actions = 0
    elif action_mode == 'mixed':
        dose_actions = action_dim // 2
        binary_actions = action_dim // 2
    
    
    trainer = BC(
        max_action=max_action,
        actor=actor,
        actor_optimizer=actor_optimizer,
        discount=config_manager.training_config['discount'],
        device=config_manager.training_config['device'],
        action_mode = action_mode,
        dose_actions=dose_actions,
        binary_actions=binary_actions
    )

    return trainer