# PVP imlpementation adapted from original base CQL code
# original source: https://github.com/young-geng/CQL/tree/934b0e8354ca431d6c083c4e3a29df88d4b0a24d
# https://arxiv.org/pdf/2006.04779.pdf

"""
Some notes from Filippo:
Working with negative rewards:
-2 options, set q_min manually or set zero_min_reward to True to automatically shift rewards so that the minimum reward in the dataset is 0.
    -q_min should be a baseline "bad" value for a random policy essentially, but you can think of it as a hyperparameter
-you can judge from the learning curves whether your pvp_alpha value is too big or too small
    -if the value for pvp_alpha is too small then the critic loss explodes and the actor loss goes very negative
-the general pattern i found, especially for pvp_alpha, is you want to make it as small as possible up to where the losses blow up, 
    basically reach a critical value where the regularisaiton is small but the training is still stable


From gemini: 
Actor uses TahhGaussian for forward pass, this makes action distribution bounded 
between -1 and 1. Note: I've rescaled the actions accordingly, they will need to 
be rescaled for evaluation!
    - it also allows for reparameterization, which is important for backprop 
    through the sampling process

Alpha is a temperature parameter that controls the strength of the entropy regularization in the policy loss.
It is learned (loss calculated, step is taken), and then alpha is used in policy loss.
    -alpha loss is designed to keep entropy close to target entropy, which is a hyperparameter that we set.

Policy loss aims to maximize expected Q value of the actions taken by the current policy, while also 
maximizing entropy (encouraging exploration).
    -it is the entropy calculation that is scaled by alpha

Q loss provides a standard Temporal Difference (TD) learning signal for the critic, but also includes a 
regularization term that penalizes the Q-values of out-of-distribution actions (the PVP term).
    -Calculate TD target using the target critic and the next state and action
    -If backup_entropy is True, it subtracts alpha * next_log_pi from the target. This ensures 
    that actions that are both high-reward and high-entropy (uncertain/exploratory) are valued more.
    -Filippo says to use q_min 0, and zero the rewards. I will do this for now. Alternative would be:    
    q_min calculated from the minimum reward and discount factor.
    -He also says use 
"""

import os
import math
import random
from turtle import done
import uuid
from copy import deepcopy
import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal, TanhTransform, TransformedDistribution

from .shared_components import TrainerUtils # Actor, MixedActionActor, 




TensorBatch = List[torch.Tensor]

class Actor(nn.Module): # adapted from TanhGaussianPolicy
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        hidden_dim: List[int] = [512, 256, 128], 
        dropout: float = 0.1,
        negative_slope: float = 0.1,
        log_std_multiplier: float = 1.0,
        log_std_offset: float = -1.0,
        orthogonal_init: bool = False,
        no_tanh: bool = False,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        layers = []
        input_dim = state_dim

        for h_dim in hidden_dim:
            layers.append(nn.Linear(input_dim, h_dim))
            # Critical: Always keep the activation, then apply dropout
            layers.append(nn.LeakyReLU(negative_slope)) 
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = h_dim

        # 2. Add the Final Output Layer
        # No activation or dropout here—this allows the network to output 
        # the full range of values required for 2 * action_dim.
        layers.append(nn.Linear(input_dim, 2 * action_dim))
        
        self.base_network = nn.Sequential(*layers)

        # Initialize weights
        # Initialize backbone for LeakyReLU signal flow
        for m in self.base_network[:-1]:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, 
                    a=negative_slope, 
                    mode='fan_in',
                    nonlinearity='leaky_relu'
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize the final output layer differently
        final_layer = self.base_network[-1]
        if isinstance(final_layer, nn.Linear):
            # Xavier/Glorot is the standard for linear outputs
            nn.init.xavier_normal_(final_layer.weight)
            if final_layer.bias is not None:
                # Often helpful to initialize bias to 0 
                # or a small negative value for log_std
                nn.init.zeros_(final_layer.bias)

        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    
    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        repeat: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if repeat is not None:
            state = extend_and_repeat(state, 1, repeat)
        base_network_output = self.base_network(state)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        return self.max_action * actions, log_probs
    

    def log_prob(                     # COPILOT SAYING THIS ISN'T PROPER logPI(a|s)
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        if action.ndim == 3:
            state = extend_and_repeat(state, 1, action.shape[1])
        base_network_output = self.base_network(state)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        _, log_probs = self.tanh_gaussian(mean, log_std, False)
        return log_probs


    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        with torch.no_grad():
            actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()


class ReparameterizedTanhGaussian(nn.Module):
    def __init__(
        self, log_std_min: float = -20.0, log_std_max: float = 2.0, no_tanh: bool = False
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh

    def log_prob(
        self, mean: torch.Tensor, log_std: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    def forward(
        self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(action_distribution.log_prob(action_sample), dim=-1)

        return action_sample, log_prob



class Critic(nn.Module): # FullyConnectedQFunction
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        batch_norm=False,
        layer_norm=True,    
        dropout=0.1, 
        orthogonal_init: bool = False,
        hidden_dim=512, 
        negative_slope=0.1,
    ):
        super(Critic, self).__init__()

        # standardize the incoming embeddings
        # self.state_norm = nn.LayerNorm(state_dim + action_dim) # if cql loss stays close to zero, we may
        self.state_norm = nn.LayerNorm(state_dim)
        # want to remove action_dim from this normalization, but for now we can keep it since actions are also log-min-maxed and should be on a similar scale to states	
        self.hidden_dim = hidden_dim

        self.hidden_dim2 = hidden_dim // 2  # Example of a second hidden layer with half the units of the first
        self.hidden_dim3 = self.hidden_dim2 // 2  # Example of a third hidden layer with half the units of the second

        self.negative_slope = negative_slope

        def create_net(self, state_dim, action_dim, batch_norm, layer_norm, dropout):
            if batch_norm:
                print("Using batch normalization in critic")
                q_net = nn.Sequential(
                    nn.Linear(state_dim + action_dim, self.hidden_dim),
                    nn.BatchNorm1d(self.hidden_dim), # Normalize internal activations
                    nn.LeakyReLU(negative_slope=self.negative_slope),
                    nn.Dropout(p=dropout),
                    nn.Linear(self.hidden_dim, self.hidden_dim2),
                    nn.BatchNorm1d(self.hidden_dim2), # Normalize internal activations
                    nn.LeakyReLU(negative_slope=self.negative_slope),
                    nn.Dropout(p=dropout),
                    nn.Linear(self.hidden_dim2, self.hidden_dim3),
                    nn.BatchNorm1d(self.hidden_dim3), # Normalize internal activations
                    nn.LeakyReLU(negative_slope=self.negative_slope),
                    # nn.Dropout(p=dropout),
                )

            elif layer_norm:     
                q_net = nn.Sequential(
                    nn.Linear(state_dim + action_dim, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim), # Normalize internal activations
                    nn.LeakyReLU(negative_slope=self.negative_slope),
                    nn.Dropout(p=dropout),
                    
                    nn.Linear(self.hidden_dim, self.hidden_dim2),
                    nn.LayerNorm(self.hidden_dim2), # Normalize internal activations
                    nn.LeakyReLU(negative_slope=self.negative_slope),
                    nn.Dropout(p=dropout),
                    
                    nn.Linear(self.hidden_dim2, self.hidden_dim3),
                    nn.LayerNorm(self.hidden_dim3), # Normalize internal activations
                    nn.LeakyReLU(negative_slope=self.negative_slope),
                    # nn.Dropout(p=dropout),    
                    
                    nn.Linear(self.hidden_dim3, 1)
                )
            init_module_weights(q_net, orthogonal_init)
            return q_net
        
        # note for layer norm, we may want to apply it only to the state dimensions, 
        # but for now we can apply it to the concatenated state-action input since 
        # both are log-min-maxed and should be on a similar scale
		# Q1 architecture
        self.q1_net = create_net(self, state_dim, action_dim, batch_norm, layer_norm, dropout)
        self.q2_net = create_net(self, state_dim, action_dim, batch_norm, layer_norm, dropout)

        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 1. THE FINAL HEAD (Output = 1)
            if m.out_features == 1:
                # Squeezing to near-zero is good, but 3e-3 is VERY tight for a 3-layer net.
                # Let's use a slightly wider window to give the gradient a "handle."
                nn.init.uniform_(m.weight, -1e-2, 1e-2) 
                if m.bias is not None:
                    nn.init.constant_(m.bias, -0.2) # this starts model in pessimistic state
                                                    # but if Q values collapse, we may need to make this 0.0
            
            # 2. THE HIDDEN LAYERS
            else:
                # CRITICAL: Match 'a' to your negative_slope (0.1)
                # 'fan_in' is correct for preserving signal in the forward pass
                nn.init.kaiming_normal_(
                    m.weight, 
                    a=self.negative_slope,  # <--- Change this from 0.01 to 0.1
                    mode='fan_in', 
                    nonlinearity='leaky_relu'
                )
                
                if m.bias is not None:
                    # Keep this at 0. With LayerNorm, non-zero biases can 
                    # actually introduce unnecessary shift.
                    nn.init.constant_(m.bias, 0.05)      

        elif isinstance(m, nn.LayerNorm):
            # Standard identity transform - keep as is
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        multiple_actions = False
        batch_size = state.shape[0]
        
        if action.ndim == 3 and state.ndim == 2:
            multiple_actions = True
            state = extend_and_repeat(state, 1, action.shape[1]).reshape(
                -1, state.shape[-1]
            )
            action = action.reshape(-1, action.shape[-1])       

        norm_state = self.state_norm(state) # ai says only apply to state, fix in other algos   
        sa = torch.cat([norm_state, action], 1)

        # normalize states after concatenation
        # sa = self.state_norm(sa)

        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)

        return q1, q2
    


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


def init_module_weights(module: torch.nn.Sequential, orthogonal_init: bool = False):
    # Specific orthgonal initialization for inner layers
    # If orthogonal init is off, we do not change default initialization
    if orthogonal_init:
        for submodule in module[:-1]:
            if isinstance(submodule, nn.Linear):
                nn.init.orthogonal_(submodule.weight, gain=np.sqrt(2))
                nn.init.constant_(submodule.bias, 0.0)

    # Lasy layers should be initialzied differently as well
    if orthogonal_init:
        nn.init.orthogonal_(module[-1].weight, gain=1e-2)
    else:
        nn.init.xavier_uniform_(module[-1].weight, gain=1e-2)

    nn.init.constant_(module[-1].bias, 0.0)



def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def return_reward_range(dataset: Dict, max_episode_steps: int) -> Tuple[float, float]:
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(
    dataset: Dict,
    env_name: str,
    max_episode_steps: int = 1000,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
):
    if reward_scale > 0:          # normalise based on returns
        if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
            min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
            dataset["rewards"] /= max_ret - min_ret
            dataset["rewards"] *= max_episode_steps
    elif reward_scale == 0:       # naive 0-1 reward normalization
        max_reward = dataset["rewards"].max()
        min_reward = dataset["rewards"].min()
        reward_scale = 1/(max_reward-min_reward)
    dataset["rewards"] = dataset["rewards"] * reward_scale + reward_bias



def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


class ContinuousPVP:
    def __init__(
        self,
        max_action: np.ndarray,

        actor: nn.Module = None,
		actor_optimizer: torch.optim.Optimizer = None,
		actor_scheduler: torch.optim.lr_scheduler._LRScheduler = None,

        critic: nn.Module = None,
		critic_optimizer: torch.optim.Optimizer = None,
		critic_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        
        alpha_optimizer: torch.optim.Optimizer = None,

        device: str = "CUDA" if torch.cuda.is_available() else "CPU",

        grad_norm_max_actor = None,
        grad_norm_max_critic = None,

        target_entropy: float = 0.,

        batch_norm: bool = False,
        zero_min_reward: bool = True,
        importance_sampling: bool = False,
        discount: float = 0.99,
        alpha_multiplier: float = 1.0,
        use_automatic_entropy_tuning: bool = True,
        backup_entropy: bool = False,

        q_optim: str = "adam",

        alpha_optim: str = "sgd",
        policy_lr: bool = 3e-4,
        qf_lr: bool = 3e-4,
        alpha_lr: bool = 3e-4,
        adam_eps: float = 1e-8,
        soft_target_update_rate: float = 5e-3,
        bc_steps=0,
        target_update_period: int = 1,
        pvp_n_actions: int = 10,
        q_min: float = 0.0, 
        min_reward: float = 0.0,
        pvp_alpha: float = 10.0,
        next_state_reg: bool = True,
        fixed_actor: bool = False,
        ncp_samples: bool = False,
        ood_action_var: float = 0.2,

        reward_scaling: float = 1.0,

        ema_alpha: float = 0.99,  # Add ema_alpha argument for moving
    ):

        self.max_action = max_action

        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.actor_scheduler = actor_scheduler
        
        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.critic_scheduler = critic_scheduler  
        
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
        else:
            self.log_alpha = None
        
        self.alpha_optimizer = alpha_optimizer    

        self.device = device  
        
        self.discount = discount
        self.batch_norm = batch_norm
        
        # q_min = min_reward / (1 - discount) if zero_min_reward else min_reward
        self.q_min = q_min
        self.min_reward = min_reward

        self.q_min = q_min
        self.pvp_alpha = pvp_alpha
        self.target_entropy = target_entropy
        self.alpha_multiplier = alpha_multiplier
        
        self.backup_entropy = backup_entropy

        self.reward_scaling = reward_scaling

        self.alpha_optim = alpha_optim
        self.q_optim = q_optim
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.alpha_lr = alpha_lr
        self.adam_eps = adam_eps
        self.soft_target_update_rate = soft_target_update_rate
        self.bc_steps = bc_steps
        self.target_update_period = target_update_period
        self.pvp_n_actions = pvp_n_actions
        self._device = device
        self.zero_min_reward = zero_min_reward
        self.importance_sampling = importance_sampling
        self.grad_norm_max_actor = grad_norm_max_actor
        self.grad_norm_max_critic = grad_norm_max_critic
        self.fixed_actor = fixed_actor
        self.ncp_samples = ncp_samples
        self.ood_action_var = ood_action_var
        self.next_state_reg = next_state_reg
        self.total_it = 0

        self.critic_clip_fraction_ema = 0.0
        self.actor_clip_fraction_ema = 0.0
        self.ema_alpha = ema_alpha
        # if self.critic_1 is not None:
        #     self.target_critic_1 = deepcopy(self.critic_1).to(device)
        #     self.target_critic_2 = deepcopy(self.critic_2).to(device)

        # copy the critic network to create the target critic
        self.target_critic = copy.deepcopy(self.critic)
        self.target_critic.eval()  # Target networks start in eval mode to disable Dropout

        self.total_it = 0

    def update_target_network(self, soft_target_update_rate: float):
        soft_update(self.target_critic, self.critic, soft_target_update_rate)

    def _alpha_and_alpha_loss(self, observations: torch.Tensor, log_pi: torch.Tensor):
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha() * (log_pi + self.target_entropy).detach()
            ).mean()
            alpha = self.log_alpha().exp() * self.alpha_multiplier
        else:
            alpha_loss = observations.new_tensor(0.0)
            alpha = observations.new_tensor(self.alpha_multiplier)
        return alpha, alpha_loss

    def _policy_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        new_actions: torch.Tensor,
        alpha: torch.Tensor,
        log_pi: torch.Tensor,
    ) -> torch.Tensor:
        
        # if self.batch_norm:
        #     self.critic.eval()
        self.critic.eval()
        q_new_actions = torch.min(*self.critic(observations, new_actions))

        # if self.batch_norm:
        #     self.critic.train()
        self.critic.train()

        policy_critic_loss = -q_new_actions.mean()
        policy_entropy_loss = alpha*log_pi.mean()
        policy_loss = (alpha * log_pi - q_new_actions).mean()
        
        return policy_loss, policy_critic_loss, policy_entropy_loss

    def _q_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        alpha: torch.Tensor,
        log_dict: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q1_predicted, q2_predicted = self.critic(observations, actions)

        new_next_actions, next_log_pi = self.actor(next_observations)
        target_q_values = torch.min(
            *self.target_critic(next_observations, new_next_actions),
        )

        if self.backup_entropy:
            target_q_values = target_q_values - alpha * next_log_pi

        # target_q_values = target_q_values.unsqueeze(-1)
        td_target = rewards + (1.0 - dones) * self.discount * target_q_values.detach()
        # td_target = td_target.squeeze(-1)
        qf1_loss = F.mse_loss(q1_predicted, td_target.detach())
        qf2_loss = F.mse_loss(q2_predicted, td_target.detach())

        # PVP regularisation term
        batch_size = actions.shape[0]

         # policy-based sampling
        pvp_ood_actions, pvp_ood_log_pis = self.actor(observations.repeat((self.pvp_n_actions,1)))
        pvp_ood_actions = pvp_ood_actions.detach().reshape(batch_size, self.pvp_n_actions, -1)
        
        pvp_ood_q1, pvp_ood_q2 = self.critic(observations, pvp_ood_actions)

        if not self.next_state_reg:
            pvp_qf1_ood_loss = torch.mean((pvp_ood_q1 - self.q_min)**2, dim=1).mean()
            pvp_qf2_ood_loss = torch.mean((pvp_ood_q2 - self.q_min)**2, dim=1).mean()
            
        else: # Filippo says to use next_state_reg
            next_q1, next_q2 = self.critic(next_observations, new_next_actions.detach())
            
            pvp_qf1_ood_loss = torch.mean((pvp_ood_q1 - self.q_min)**2, dim=1).mean() + ((next_q1-self.q_min)**2).mean()
            pvp_qf2_ood_loss = torch.mean((pvp_ood_q2 - self.q_min)**2, dim=1).mean() + ((next_q2-self.q_min)**2).mean()
   

        # Total loss
        qf_loss = qf1_loss + qf2_loss + pvp_qf1_ood_loss*self.pvp_alpha + pvp_qf2_ood_loss*self.pvp_alpha

        log_dict.update(
            dict(
                qf_loss=qf_loss.item(),
                qf1_loss=qf1_loss.item(),
                qf2_loss=qf2_loss.item(),
                alpha=alpha.item(),
                average_qf1=q1_predicted.mean().item(),
                average_qf2=q2_predicted.mean().item(),
                average_target_q=target_q_values.mean().item(),
                pvp_ood_q1=pvp_ood_q1.mean().item(),
                pvp_ood_q2=pvp_ood_q2.mean().item(),
                pvp_ood_q1_std = pvp_ood_q1.std().item(),
                pvp_ood_q2_std = pvp_ood_q2.std().item(),
                pvp_qf1_ood_loss=self.pvp_alpha*pvp_qf1_ood_loss.mean().item(),
                pvp_qf2_ood_loss=self.pvp_alpha*pvp_qf2_ood_loss.mean().item(),
                pvp_ood_actions=pvp_ood_actions.mean().item()
            )
        )
   
        return qf_loss

    def _get_grad_norm(self, model: nn.Module) -> float:
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def train(self, batch: TensorBatch) -> Dict[str, float]:
 
        self.total_it += 1

        # Sample replay buffer 
        state, action, reward, next_state, next_action, done, mortality = TrainerUtils._unpack_batch(batch, self.device)

        # zero reward if specified
        reward = reward * self.reward_scaling
        if self.zero_min_reward:
            reward = reward - self.min_reward

		# Rescale actions from [0, 1] to [-1, 1]
        action = 2.0 * action - 1.0
        next_action = 2.0 * next_action - 1.0

        # compute the target Q value
        treat_survival_as_truncated = True
        if treat_survival_as_truncated:
                # make done 0 where mortality is 0, since we want to treat survival as truncated rather than terminal
            done = done * mortality
		
        not_done = 1 - done # not done should be ued for calculating Q values - check code, I think Filippo uses not_done

        new_actions, log_pi = self.actor(state)

        # alpha loss
        alpha, alpha_loss = self._alpha_and_alpha_loss(state, log_pi)

        """ Policy loss """
        policy_loss, policy_critic_loss, policy_entropy_loss = self._policy_loss(
            state, action, new_actions, alpha, log_pi
        )
        log_dict = dict(
            log_pi=log_pi.mean().item(),
            policy_loss=policy_loss.item(),
            policy_critic_loss=policy_critic_loss.item(),
            policy_entropy_loss=policy_entropy_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha=alpha.item(),
        )

        """ Q function loss """
        qf_loss = self._q_loss(
            state, action, next_state, reward, done, alpha, log_dict
        )        

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()            

            self.alpha_grad_norm = torch.cat([param.grad.detach().flatten() for param in self.log_alpha.parameters() if param.grad is not None]).norm()
            grad_norm = {"alpha_grad_norm": self.alpha_grad_norm}
            
            # need to implement gradient clipping
            
            self.alpha_optimizer.step()

        self.actor_optimizer.zero_grad()

        # policy loss
        policy_loss.backward()

        total_actor_grad_norm = self._get_grad_norm(self.actor)
        grad_norm['actor_grad_norm'] = total_actor_grad_norm
        self.policy_grad_norm = total_actor_grad_norm
        # self.policy_grad_norm = torch.cat([param.grad.detach().flatten() for param in self.actor.parameters() if param.grad is not None]).norm()

        grad_norm['actor_was_clipped'] = 1.0 if total_actor_grad_norm > self.grad_norm_max_actor else 0.0
        
        # update the moving average of the clipping binary
        self.actor_clip_fraction_ema = self.ema_alpha * self.actor_clip_fraction_ema + (1 - self.ema_alpha) * grad_norm['actor_was_clipped']
        grad_norm['actor_clip_fraction_ema'] = self.actor_clip_fraction_ema
        if self.grad_norm_max_actor is not None:
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), 
                self.grad_norm_max_actor
            )
        
        self.actor_optimizer.step()
        if self.actor_scheduler is not None:
            self.actor_scheduler.step()
            log_dict.update(lr_actor=self.actor_optimizer.param_groups[0]['lr'])

        # critic loss
        self.critic_optimizer.zero_grad()
        qf_loss.backward(retain_graph=True)

        total_critic_grad_norm = self._get_grad_norm(self.critic)
        grad_norm['critic_grad_norm'] = total_critic_grad_norm
        self.critic_grad_norm = total_critic_grad_norm

        grad_norm['critic_was_clipped'] = 1.0 if total_critic_grad_norm > self.grad_norm_max_critic else 0.0
        self.critic_clip_fraction_ema = self.ema_alpha * self.critic_clip_fraction_ema + (1 - self.ema_alpha) * grad_norm['critic_was_clipped']
        grad_norm['critic_clip_fraction_ema'] = self.critic_clip_fraction_ema
        if self.grad_norm_max_critic is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm_max_critic)

        self.critic_optimizer.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()
            log_dict.update(lr_critic=self.critic_optimizer.param_groups[0]['lr'])

        if self.total_it % self.target_update_period == 0:
            self.update_target_network(self.soft_target_update_rate)

        return log_dict, grad_norm


    def actor_forward_for_sarsa(self, batch: TensorBatch, rescale=True) -> torch.Tensor:
        state, _, _, _, _, _, _ = TrainerUtils._unpack_batch(batch, self.device)
               
        action = 2.0 * action - 1.0
        new_actions, log_pi = self.actor(state)
        
        if rescale:
            new_actions = (new_actions / 2.0) + 0.5  # Rescale back to [0, 1]        
        
        return new_actions
    
    
    
    def evaluate(self, batch: TensorBatch) -> Dict[str, float]:
        self.actor.eval()  # This is correct and important!
        self.critic.eval()  # Ensure critic is also in eval mode

        state, action, reward, next_state, next_action, done, mortality = TrainerUtils._unpack_batch(batch, self.device)
        
        reward = reward * self.reward_scaling
        # zero reward if specified
        reward = reward.clone()  # Clone to avoid in-place operations   
        if self.zero_min_reward:
            reward = reward - self.min_reward

        # Rescale actions from [0, 1] to [-1, 1]
        action = 2.0 * action - 1.0
        next_action = 2.0 * next_action - 1.0

        # compute the target Q value
        treat_survival_as_truncated = True
        if treat_survival_as_truncated:
                # make done 0 where mortality is 0, since we want to treat survival as truncated rather than terminal
            done = done * mortality
		
        not_done = 1 - done

        with torch.no_grad():
            new_actions, log_pi = self.actor(state)

            # alpha loss
            alpha, alpha_loss = self._alpha_and_alpha_loss(state, log_pi)

            """ Policy loss """
            policy_loss, policy_critic_loss, policy_entropy_loss = self._policy_loss(
                state, action, new_actions, alpha, log_pi
            )
            log_dict = dict(
                log_pi=log_pi.mean().item(),
                policy_loss=policy_loss.item(),
                policy_critic_loss=policy_critic_loss.item(),
                policy_entropy_loss=policy_entropy_loss.item(),
                alpha_loss=alpha_loss.item(),
                alpha=alpha.item(),
            )
            """ Q function loss """
            qf_loss = self._q_loss(
                state, action, next_state, reward, done, alpha, log_dict
            )
            log_dict.update(qf_loss=qf_loss.item())           
        
        return log_dict, action, new_actions


    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic": self.critic.state_dict(),
			"critic_optimizer": self.critic_optimizer.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "sac_log_alpha": self.log_alpha,
            "sac_log_alpha_optim": self.alpha_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict=state_dict["actor"])
        self.critic.load_state_dict(state_dict=state_dict["critic"])

        self.target_critic.load_state_dict(state_dict=state_dict["target_critic"])

        self.critic_optimizer.load_state_dict(
            state_dict=state_dict["critic_optimizer"]
        )
        self.actor_optimizer.load_state_dict(state_dict=state_dict["actor_optimizer"])

        self.log_alpha = state_dict["sac_log_alpha"]
        self.alpha_optimizer.load_state_dict(
            state_dict=state_dict["sac_log_alpha_optim"]
        )

        self.total_it = state_dict["total_it"]



def get_trainer(config_manager, state_dim, action_dim, max_action, min_reward, hyperparams):
    learning_rate_critic = float(hyperparams['learning_rate_critic'])
    learning_rate_actor = float(hyperparams['learning_rate_actor'])
    # weight_decay = float(hyperparams.get('weight_decay', 0.01)) # New hyperparam
    # device = config_manager.training_config['device']
    adam_eps = 1e-8
    
    actor = Actor(
        state_dim=state_dim, 
        action_dim=action_dim, 
        max_action=max_action,
        # hidden_dim=hyperparams['hidden_dim'], 
        log_std_multiplier=hyperparams['policy_log_std_multiplier'],
        orthogonal_init=hyperparams['orthogonal_init'],
        dropout=hyperparams['dropout']
    ).to(config_manager.training_config['device'])

    if hyperparams['actor_optim'] == "adam":
        actor_optimizer = torch.optim.Adam(
            actor.parameters(), 
            lr=learning_rate_actor,
            eps=adam_eps,
        )
    elif hyperparams['actor_optim'] == "sgd":
        actor_optimizer = torch.optim.SGD(
            actor.parameters(), 
            lr=learning_rate_actor,
        )

    actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        actor_optimizer, 
        T_max=hyperparams['total_steps']//hyperparams['target_update_period'], # we step the actor scheduler every policy_freq steps, so T_max should be total_steps // policy_freq
        eta_min=learning_rate_actor * .10
    )

    critic = Critic(
        state_dim, 
        action_dim,
        orthogonal_init=hyperparams['orthogonal_init'],        
        batch_norm=hyperparams['batch_norm'], # - i think I can keep using layer_norm
    ).to(config_manager.training_config['device'])

    if hyperparams['q_optim'] == "adam":
        critic_optimizer = torch.optim.Adam(
            critic.parameters(),
            lr=learning_rate_critic,
            eps=adam_eps,
        )
    elif hyperparams['q_optim'] == "sgd":
         critic_optimizer = torch.optim.SGD(
            critic.parameters(),
            lr=learning_rate_critic,
        )

    critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        critic_optimizer, 
        T_max=hyperparams['total_steps'],
        eta_min=learning_rate_critic * .10
    )

    log_alpha = Scalar(0.0)
    
    if hyperparams['alpha_optim'] == "adam":
        alpha_optimizer = torch.optim.Adam(
            log_alpha.parameters(),
            lr=hyperparams['learning_rate_alpha'], 
            eps=adam_eps
    ) 
    elif hyperparams['alpha_optim'] == "sgd":
        alpha_optimizer = torch.optim.SGD(
            log_alpha.parameters(),
            lr=hyperparams['learning_rate_alpha'], 
        )

    kwargs = {
        "device": config_manager.training_config['device'],

        "critic": critic,
        "critic_optimizer": critic_optimizer,
        "critic_scheduler": critic_scheduler,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "actor_scheduler": actor_scheduler,
        "alpha_optimizer": alpha_optimizer,
        "discount": hyperparams['discount'],

        "max_action": max_action,
        "min_reward": min_reward,
        "zero_min_reward": hyperparams['zero_min_reward'],

        "soft_target_update_rate": hyperparams['tau'],
        "device": config_manager.training_config['device'],
        "batch_norm": hyperparams['batch_norm'],
        
        "importance_sampling": hyperparams['importance_sampling'],
        "target_entropy": hyperparams['target_entropy'],  # -np.prod(env.action_space.shape).item(),
        "alpha_multiplier": hyperparams['alpha_multiplier'],
        "use_automatic_entropy_tuning": hyperparams['use_automatic_entropy_tuning'],
        "backup_entropy": hyperparams['backup_entropy'],
                
        "policy_lr": hyperparams['learning_rate_actor'],
        "qf_lr": hyperparams['learning_rate_critic'],
        "alpha_lr": hyperparams['learning_rate_alpha'],

        "bc_steps": hyperparams['bc_steps'],
        "grad_norm_max_actor": hyperparams['grad_norm_max_actor'],
        "grad_norm_max_critic": hyperparams['grad_norm_max_critic'],
        "next_state_reg": hyperparams['next_state_reg'],
        "target_update_period": hyperparams['target_update_period'],
        "pvp_n_actions": hyperparams['pvp_n_actions'],
        "pvp_alpha": hyperparams['pvp_alpha'],
        # "q_min": hyperparams['q_min'],
    }

    trainer = ContinuousPVP(
        **kwargs,
    )

    return trainer




