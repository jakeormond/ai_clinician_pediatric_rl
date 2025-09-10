
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


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: List[int], max_action: float):
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

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()
    

class MixedActionActor(nn.Module):
    def __init__(self, state_dim: int, dose_actions: int, 
                binary_actions: int, hidden_dim: List[int], max_action: float):
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

class BC:
    def __init__(
        self,
        max_action: np.ndarray,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        device: str = "cpu",
        dose_actions: int = 2,  #  Number of dose actions
        binary_actions: int = 2
    ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.max_action = max_action
        self.discount = discount
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
      
        # Split predictions and targets
        dose_predictions = pi[:, :self.dose_actions]      # First 2 actions (doses)
        binary_predictions = pi[:, self.dose_actions:]   # Last 2 actions (binary)

        dose_targets = action[:, :self.dose_actions]
        binary_targets = action[:, self.dose_actions:]

        # Different loss functions for different action types
        dose_loss = F.mse_loss(dose_predictions, dose_targets)

        binary_loss = F.binary_cross_entropy(binary_predictions, binary_targets)

        # Combine losses (you can weight them differently)
        dose_weight = 1.0
        binary_weight = 1.0
        actor_loss = dose_weight * dose_loss + binary_weight * binary_loss

        return actor_loss, dose_loss, binary_loss

    
    def train(self, batch: TensorBatch) -> Dict[str, float]:
        
        log_dict = {}
        self.total_it += 1

        state, action = self._get_state_and_action(batch)

        # generate actions
        pi = self.actor(state)

        # Compute actor loss
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

            # Compute actor loss
            actor_loss, dose_loss, binary_loss = self._compute_actor_loss(pi, action)
            
            # Split predictions and targets
            dose_pred = pi[:, :self.dose_actions]
            binary_pred = pi[:, self.dose_actions:]
            dose_target = action[:, :self.dose_actions]
            binary_target = action[:, self.dose_actions:]
            
            # Dose action metrics (continuous regression)
            dose_mae = F.l1_loss(dose_pred, dose_target)
            
            # Binary action metrics (binary classification)
            # Clamp to avoid numerical issues with BCE
            # binary_pred_clamped = torch.clamp(binary_pred, 1e-7, 1-1e-7)
            # binary_bce = F.binary_cross_entropy(binary_pred_clamped, binary_target).item()
            
            # Binary accuracy (threshold at 0.5)
            binary_pred_classes = (binary_pred > 0.5).float()
            binary_accuracy = (binary_pred_classes == binary_target).float().mean()
            
            # Per-action accuracy for binary actions
            binary_acc_per_action = []
            for i in range(self.binary_actions):
                acc = (binary_pred_classes[:, i] == binary_target[:, i]).float().mean()
                binary_acc_per_action.append(acc)   
                        
        self.actor.train()  # Set back to training mode

        overall_metrics = {# Overall metrics
            "eval/total_loss": actor_loss.item(),
            
            # Dose action metrics
            "eval/dose_mse": dose_loss.item(),
            "eval/dose_mae": dose_mae.item(),
            "eval/dose_rmse": dose_loss.item() ** 0.5,
            
            # Binary action metrics
            "eval/binary_bce": binary_loss.item(),
            "eval/binary_accuracy": binary_accuracy.item(),
            "eval/binary_acc_action_0": binary_acc_per_action[0].item(),
            "eval/binary_acc_action_1": binary_acc_per_action[1].item()
        }

        preds_and_targets = {
            "dose_pred": dose_pred.cpu().numpy(),
            "dose_target": dose_target.cpu().numpy(),
            "binary_pred": binary_pred.cpu().numpy(),
            "binary_target": binary_target.cpu().numpy()
        }
        
        return overall_metrics, action.cpu().numpy(), pi.cpu().numpy()

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
