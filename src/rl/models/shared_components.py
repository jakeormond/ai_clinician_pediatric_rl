import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class TrainerUtils:
    @staticmethod
    def _unpack_batch(batch: List[torch.Tensor], device: str) -> Tuple[torch.Tensor, ...]:
        """Unpacks a batch and moves tensors to specified device."""
        if len(batch) == 7: # for basic_q (SARSA)
            state, action, reward, next_state, next_action, done, mortality = batch
            # unsqueeze reward and done if necessary
            if reward.dim() == 1:
                reward = reward.unsqueeze(1)
            if done.dim() == 1:
                done = done.unsqueeze(1)
            if mortality.dim() == 1:
                mortality = mortality.unsqueeze(1)
            
            state = state.to(device)
            action = action.to(device)
            reward = reward.to(device)
            next_state = next_state.to(device)
            next_action = next_action.to(device)
            done = done.to(device)
            mortality = mortality.to(device)
            
            return state, action, reward, next_state, next_action, done, mortality
        else:
            state, action, reward, next_state, done, mortality = batch

            # unsqueeze reward and done if necessary
            if reward.dim() == 1:
                reward = reward.unsqueeze(1)
            if done.dim() == 1:
                done = done.unsqueeze(1)
            if mortality.dim() == 1:
                mortality = mortality.unsqueeze(1)
            
            state = state.to(device)
            action = action.to(device)
            reward = reward.to(device)
            next_state = next_state.to(device)
            done = done.to(device)
            mortality = mortality.to(device)
            
            return state, action, reward, next_state, done, mortality

    @staticmethod
    def _compute_binary_loss(binary_predictions: torch.Tensor, binary_targets: torch.Tensor) -> torch.Tensor:
        """Compute BCE loss for binary actions."""
        return F.binary_cross_entropy(binary_predictions, binary_targets)

    @staticmethod
    def _compute_dose_loss(dose_predictions: torch.Tensor, dose_targets: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss for continuous actions."""
        return F.mse_loss(dose_predictions, dose_targets)

    @staticmethod
    def _compute_actor_loss(pi: torch.Tensor, 
                          action: torch.Tensor, 
                          action_mode: str,
                          dose_actions: int = None) -> torch.Tensor:
        """Compute actor loss based on action mode."""
        if action_mode == "binary_only":
            return TrainerUtils._compute_binary_loss(pi, action)
        
        elif action_mode == "continuous_only":
            return TrainerUtils._compute_dose_loss(pi, action)
        
        elif action_mode == "mixed":
            dose_predictions = pi[:, :dose_actions]
            binary_predictions = pi[:, dose_actions:]
            dose_targets = action[:, :dose_actions]
            binary_targets = action[:, dose_actions:]
            
            dose_loss = TrainerUtils._compute_dose_loss(dose_predictions, dose_targets)
            binary_loss = TrainerUtils._compute_binary_loss(binary_predictions, binary_targets)

            # Combine losses (you can weight them differently)
            dose_weight = 1.0
            binary_weight = 1.0
            actor_loss = dose_weight * dose_loss + binary_weight * binary_loss

            return actor_loss, dose_loss, binary_loss
        
        else:
            raise ValueError(f"Unknown action_mode: {action_mode}")






class Actor(nn.Module):
    def __init__(
            self, 
            state_dim: int, 
            action_dim: int, 
            hidden_dim: List[int], 
            max_action: float,
            action_mode: str,
            dropout: float = 0.0  # Add dropout argument
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
                if dropout == 0.0 else nn.Dropout(dropout)  # Add dropout layer if dropout > 0
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
            max_action: float,
            dropout: float = 0.0  # Add dropout argument
            ):
        
        super().__init__()
        
        # Build network dynamically from hidden_dim list
        layers = []
        input_dim = state_dim

        # Add hidden layers
        for h_dim in hidden_dim:
            layers.extend([
                nn.Linear(input_dim, h_dim),
                nn.ReLU()
                if dropout == 0.0 else nn.Dropout(dropout)  # Add dropout layer if dropout > 0
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
    