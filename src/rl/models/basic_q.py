from typing import Dict, List, Tuple
import copy
from ... import config_manager
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from sklearn.metrics import roc_auc_score
import pandas as pd

from .shared_components import TrainerUtils


TensorBatch = List[torch.Tensor]

"""
this is designed for 1. testing that the pipeline works, and 2. testing whether
intermediate rewards help learning

FQE UPDATE: this file now also supports Fitted Q Evaluation (FQE). FQE is
algorithmically IDENTICAL to SARSA in every respect except one: the
Bellman target's next_action is no longer the clinician's real, logged
next action -- it is the FIXED, FROZEN target policy's action at
next_state (here, a trained CQL actor, evaluated deterministically, never
updated during FQE training). Everything else (critic architecture, loss,
target-network Polyak averaging, discounting) is unchanged. This is
implemented as a single conditional substitution in train()/evaluate(),
controlled by whether a target_policy_actor was passed to Q_Learner.
"""

class Critic(nn.Module):
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        dropout=0.1, 
        hidden_dim=64, 
        negative_slope=0.1
    ):
        
        super(Critic, self).__init__()

        # standardize the incoming embeddings
        self.hidden_dim = hidden_dim

        self.negative_slope = negative_slope

        self.state_norm = nn.LayerNorm(state_dim) #

        self.q_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(negative_slope=negative_slope),
                nn.Dropout(p=dropout),

                nn.Linear(hidden_dim, hidden_dim//2),
                nn.LeakyReLU(negative_slope=negative_slope),

                nn.Linear(hidden_dim//2, 1)
            ) for _ in range(2)
        ])
        
        self.apply(self._init_weights)

        
    
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

    
    def forward(self, state, action):

        norm_state = self.state_norm(state)
        sa = torch.cat([norm_state, action], dim=-1)  # (B, 40+2=42)

        q1 = self.q_heads[0](sa)
        q2 = self.q_heads[1](sa)

        return q1, q2
    

    @torch.no_grad()
    def get_neuron_stats(self, state, action):
        # Ensure we are in eval mode to avoid Dropout interference
        self.eval()

        stats = {}
        leaky_idx = 0

        # Helper function to trace activations layer-by-layer through an nn.Sequential block
        def trace_sequential_block(sequential_block, current_input, block_name):
            nonlocal leaky_idx
            x = current_input
            for layer in sequential_block:
                x = layer(x)
                if isinstance(layer, nn.LeakyReLU):
                    # 1. Percent of inputs utilizing the negative slope in this batch
                    pct_negative = (x < 0).float().mean().item() * 100
                    stats[f"{block_name}_neg_regime_L{leaky_idx}_pct"] = pct_negative
                    
                    # 2. True Inactive/Dead Neurons: Variance across the batch is essentially 0
                    neuron_vars = torch.var(x, dim=0)
                    pct_dead = (neuron_vars < 1e-6).float().mean().item() * 100
                    stats[f"{block_name}_dead_neurons_L{leaky_idx}_pct"] = pct_dead 

                    leaky_idx += 1            
            return x

        norm_state = self.state_norm(state)
        sa = torch.cat([norm_state, action], dim=-1)
        _ = trace_sequential_block(self.q_heads[0], sa, "q_head")       
            
        return stats


    def get_grad_norm(self):
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm




def get_q_stats(current_Q1, current_Q2, mortality, random_action_sensitivity, zero_action_sensitivity) -> Dict[str, float]:
    
    # create current_Q as the mean of current_Q1 and current_Q2
    current_Q = (current_Q1 + current_Q2) / 2.0
    mean_Q = current_Q.mean().item()
    variance_Q = current_Q.var().item()
    batch_n = current_Q.shape[0]
    uncertainty_Q = F.mse_loss(current_Q1, current_Q2).item() # simple measure of uncertainty based on disagreement between the two critics

    # random_sensitivity
    positive_random_sensitivity = random_action_sensitivity[random_action_sensitivity > 0]
    proportion_positive_sensitivity = positive_random_sensitivity.shape[0] / random_action_sensitivity.shape[0]
    mean_positive_random_sensitivity = positive_random_sensitivity.abs().mean().item() if positive_random_sensitivity.shape[0] > 0 else 0.0
    mean_random_sensitivity = random_action_sensitivity.abs().mean().item()
    variance_random_sensitivity = random_action_sensitivity.var().item()
    max_random_sensitivity = random_action_sensitivity.max().item()

    # zero_action_sensitivity
    fluid_value = zero_action_sensitivity['non_zero_fluid_sensitivity']
    mean_fluid_zero_sensitivity = fluid_value.abs().mean().item() if isinstance(fluid_value, torch.Tensor) and fluid_value.numel() > 0 else float('nan')
    # mean_fluid_zero_sensitivity = zero_action_sensitivity['non_zero_fluid_sensitivity'].abs().mean().item() if len(zero_action_sensitivity['non_zero_fluid_sensitivity'].shape) > 0 else float('nan') # focus on fluid sensitivity for now, since vaso sensitivity is often very low due to many zero vaso actions

    vaso_value = zero_action_sensitivity['non_zero_vaso_sensitivity']
    mean_vaso_zero_sensitivity = vaso_value.abs().mean().item() if isinstance(vaso_value, torch.Tensor) and vaso_value.numel() > 0 else float('nan')
    # mean_vaso_zero_sensitivity = zero_action_sensitivity['non_zero_vaso_sensitivity'].abs().mean().item() if len(zero_action_sensitivity['non_zero_vaso_sensitivity'].shape) > 0 else float('nan')

    # masks and counts
    survival_mask = (mortality == 0)
    non_survival_mask = (mortality == 1)

    survival_n = survival_mask.sum().item()
    non_survival_n = non_survival_mask.sum().item()
    
    # only compute subgroup mean/variance together when n > 2
    if survival_n > 2:
        survival_vals = current_Q[survival_mask]
        survival_Q = survival_vals.mean().item()
        survival_variance_Q = survival_vals.var().item()
    else:
        survival_Q = float('nan')
        survival_variance_Q = float('nan')
        survival_n = 0

    if non_survival_n > 2:
        non_survival_vals = current_Q[non_survival_mask]
        non_survival_Q = non_survival_vals.mean().item()
        non_survival_variance_Q = non_survival_vals.var().item()
    else:
        non_survival_Q = float('nan')
        non_survival_variance_Q = float('nan')
        non_survival_n = 0

    if survival_n >= 2 and non_survival_n >= 2:
        # calculate signal to noise ratio of the difference between survival and non-survival Q values
        signal = abs(survival_Q - non_survival_Q)
        pooled_variance = (survival_variance_Q + non_survival_variance_Q) / 2.0
        noise = pooled_variance ** 0.5
        signal_to_noise = signal / noise if noise > 0 else float('inf')

        # also calculate AUC of using Q value to predict mortality, as an alternative measure of signal to noise
        auc_Q = roc_auc_score(1-mortality.detach().cpu().numpy(), current_Q.detach().cpu().numpy())

    else:
        signal_to_noise = float('nan')
        auc_Q = float('nan')

        
    q_stats = {
        # "batch_n": batch_n,
        "mean_Q": mean_Q,
        "variance_Q": variance_Q,
        "uncertainty_Q": uncertainty_Q,                
        # "survival_n": survival_n,
        "survival_Q": survival_Q,
        "survival_variance_Q": survival_variance_Q,
        # "non_survival_n": non_survival_n,
        "non_survival_Q": non_survival_Q,
        "non_survival_variance_Q": non_survival_variance_Q,
        "signal_to_noise": signal_to_noise,
        "auc_Q": auc_Q,
        "proportion_positive_random_sensitivity": proportion_positive_sensitivity,
        "mean_positive_random_sensitivity": mean_positive_random_sensitivity,
        "mean_random_sensitivity": mean_random_sensitivity,
        "variance_random_sensitivity": variance_random_sensitivity,
        "max_random_sensitivity": max_random_sensitivity,
        "abs_mean_fluid_zero_sensitivity": mean_fluid_zero_sensitivity,
        "abs_mean_vaso_zero_sensitivity": mean_vaso_zero_sensitivity,
    }
    return q_stats


def get_q_stats_sarsa(q_clinician, q_rl, mortality) -> Dict[str, float]:

    # masks and counts
    survival_mask = (mortality == 0)
    non_survival_mask = (mortality == 1)

    batch_n = q_clinician.shape[0]
    survival_n = survival_mask.sum().item()
    non_survival_n = non_survival_mask.sum().item()

    advantage = q_rl - q_clinician

    if survival_n > 2:
        q_clinician_survival = q_clinician[survival_mask].mean().item()
        q_rl_survival = q_rl[survival_mask].mean().item()
        advantage_survival = advantage[survival_mask]
        advantage_survival_mean = advantage_survival.mean().item()
        variance_advantage_survival = advantage_survival.var().item()
    else:
        q_clinician_survival = float('nan')
        q_rl_survival = float('nan')
        advantage_survival_mean = float('nan')
        variance_advantage_survival = float('nan')
        survival_n = 0

    if non_survival_n > 2:
        q_clinician_non_survival = q_clinician[non_survival_mask].mean().item()
        q_rl_non_survival = q_rl[non_survival_mask].mean().item()
        advantage_non_survival = advantage[non_survival_mask]
        advantage_non_survival_mean = advantage_non_survival.mean().item()
        variance_advantage_non_survival = advantage_non_survival.var().item()
    else:
        q_clinician_non_survival = float('nan')
        q_rl_non_survival = float('nan')
        advantage_non_survival_mean = float('nan')
        variance_advantage_non_survival = float('nan')
        non_survival_n = 0

    advantage_mean = advantage.mean().item()
    advantage_variance = advantage.var().item()

    q_stats_sarsa = {
        "batch_n": batch_n,
        "sarsa_advantage_mean": advantage_mean,
        "sarsa_advantage_variance": advantage_variance,
        # "survival_n": survival_n,
        "sarsa_advantage_survival_mean": advantage_survival_mean,
        "sarsa_variance_advantage_survival": variance_advantage_survival,            
        "sarsa_q_clinician_survival": q_clinician_survival,
        "sarsa_q_rl_survival": q_rl_survival,
        # "non_survival_n": non_survival_n,
        "sarsa_advantage_non_survival_mean": advantage_non_survival_mean,
        "sarsa_variance_advantage_non_survival": variance_advantage_non_survival,
        "sarsa_q_clinician_non_survival": q_clinician_non_survival,
        "sarsa_q_rl_non_survival": q_rl_non_survival,            
    }
    return q_stats_sarsa



class Q_Learner(object):
    def __init__(
        self,
        critic: nn.Module = None,
        critic_optimizer: torch.optim.Optimizer = None,	
        discount: float = 0.99,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        tau: float = 0.005,
        grad_clip: float = 1.0,
        reward_scaling: float = 1.0,
        reward_type: str = None,  # can be 'terminal' or 'health_score'
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        evaluator=False,
        target_policy_actor: nn.Module = None,  # FQE: if set, next_action is
            # generated from THIS FIXED, FROZEN actor instead of the
            # clinician's real logged next_action. None (default) = SARSA,
            # unchanged behavior. Never trained -- set to eval() and its
            # parameters are never touched by this class's optimizer.
    ):
        
        self.total_it = 0
        self.device = device

        self.critic = critic.to(self.device)
        self.reward_type = reward_type  # can be 'terminal' or 'health_score'

        # --- FQE: fixed target policy, frozen, eval mode, no gradients ---
        self.target_policy_actor = target_policy_actor
        if self.target_policy_actor is not None:
            self.target_policy_actor = self.target_policy_actor.to(self.device)
            self.target_policy_actor.eval()
            for p in self.target_policy_actor.parameters():
                p.requires_grad_(False)
        
        if not evaluator:
            self.discount = discount
            self.reward_scaling = reward_scaling

            self.critic_optimizer = critic_optimizer
            self.critic_target = copy.deepcopy(self.critic)
            self.grad_clip = grad_clip
            self.tau = tau

            self.critic_optimizer = critic_optimizer
            self.scheduler = scheduler
            
            # for calculating moving average of gradient clipping binary
            self.clip_fraction_ema = 0.0
            self.ema_alpha = 0.99 # Closer to 1.0 means more smoothing

    def _get_next_action(self, next_state, next_action_from_batch):
        """
        SARSA (target_policy_actor is None): returns the clinician's real,
        logged next_action, unchanged.
        FQE (target_policy_actor is set): returns the FIXED target policy's
        deterministic action at next_state instead -- this single swap is
        the entire, complete difference between SARSA and FQE.
        """
        if self.target_policy_actor is None:
            return next_action_from_batch
        with torch.no_grad():
            actor_output = self.target_policy_actor(next_state, deterministic=True)
            return actor_output[0]  # CQL actor returns (actions, log_prob, (gate_lp, mag_lp)[, dist])

            
    def target_q(self, reward, not_done, next_state, next_action): # need to add next action to replay buffer
        with torch.no_grad():            
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = (target_Q1 + target_Q2) / 2.0
            # target_Q = torch.min(target_Q1, target_Q2)  # TD3 style: take the minimum of the two critics to mitigate overestimation bias
            target_Q = reward + not_done * self.discount * target_Q			

        return target_Q
    
    @torch.no_grad()
    def random_action_values(self, states, actions, clinician_q):

        random_actions = torch.rand_like(actions).to(self.device)  # (B, action_dim)
        random_q1, random_q2 = self.critic(states, random_actions)
        random_q = (random_q1 + random_q2) / 2.0  # average of the two critics for stability

        # Per-state action sensitivity — how much do random actions deviate from behavioural?
        action_sensitivity = random_q - clinician_q # (B,) — varies across actions, not states

        # print(f"Mean relative action sensitivity: {action_sensitivity.mean():.4f}")
        # print(f"Median relative action sensitivity: {action_sensitivity.median():.4f}")


        return action_sensitivity

    @torch.no_grad()
    def zero_action_values(self, states, actions, clinician_q):

        zero_actions = torch.zeros_like(actions).to(self.device)  # (B, action_dim)
        zero_q1, zero_q2 = self.critic(states, zero_actions)
        zero_q = (zero_q1 + zero_q2) / 2.0  # average of the two critics for stability

        # Per-state action sensitivity — how much do zero actions deviate from behavioural?
        action_sensitivity = zero_q - clinician_q  # (B,) — varies across actions, not states

        # we want to specifically look at times when fluids or vaso are non-zero in the clinician action, 
        # since that is when we should see some deviation in Q values for zero actions
        non_zero_fluid_mask = (actions[:, 0] != 0)  # Assuming action[:, 0] corresponds to fluids
        non_zero_vaso_mask = (actions[:, 1] != 0)   # Assuming action[:, 1] corresponds to vaso

        action_sensitivity = {
            'non_zero_fluid_sensitivity': action_sensitivity[non_zero_fluid_mask] if non_zero_fluid_mask.sum() > 0 else float('nan'),
            'non_zero_vaso_sensitivity': action_sensitivity[non_zero_vaso_mask] if non_zero_vaso_mask.sum() > 0 else float('nan')
        }

        return action_sensitivity


    def train(self, batch: TensorBatch):
        self.total_it += 1

        # sample the replay buffer
        state, action, reward, next_state, next_action, done, mortality = TrainerUtils._unpack_batch(batch, self.device)

        # FQE: swap the clinician's real next_action for the fixed target
        # policy's action at next_state. No-op (returns next_action
        # unchanged) when target_policy_actor is None, i.e. plain SARSA.
        next_action = self._get_next_action(next_state, next_action)

        reward = reward * self.reward_scaling

        if self.reward_type != 'terminal': # using monte carlo returns  
            # compute the target Q value
            treat_survival_as_truncated = True
            if treat_survival_as_truncated:
                # make done 0 where mortality is 0, since we want to treat survival as truncated rather than terminal
                done = done * mortality
            
            not_done = 1 - done
        
        self.critic.train()          # Dropout ACTIVE for the network we are updating
        self.critic_target.eval()
        
        with torch.no_grad():
            if self.reward_type == 'terminal': # using monte carlo returns
                target_Q = reward
            else: # using bootstrapped returns
                target_Q = self.target_q(reward, not_done, next_state, next_action)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # use Huber loss instead
        critic_loss = (F.smooth_l1_loss(current_Q1, target_Q) + F.smooth_l1_loss(current_Q2, target_Q)) / 2.0
        
        losses = {"critic_loss": critic_loss.item()}        

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # capture gradient norms
        grad_norm = {"raw_grad_norm": self.critic.get_grad_norm()}
        
        # determine whether to clip gradients based on grad_clip value
        grad_norm['was_clipped'] = 1.0 if (self.grad_clip and grad_norm['raw_grad_norm'] > self.grad_clip) else 0.0

        # update the moving average of the clipping binary
        self.clip_fraction_ema = self.ema_alpha * self.clip_fraction_ema + (1 - self.ema_alpha) * grad_norm['was_clipped']
        grad_norm['clip_fraction_ema'] = self.clip_fraction_ema
        
        # Gradient clipping if specified
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), 
                self.grad_clip
            )
        
        self.critic_optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
            losses.update(lr_critic=self.critic_optimizer.param_groups[0]['lr'])

        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            # get Q statistics
            self.critic.eval() 
            current_Q = (current_Q1 + current_Q2) / 2.0 
            random_action_sensitivity = self.random_action_values(state, action, current_Q)
            zero_action_sensitivity = self.zero_action_values(state, action, current_Q)
            q_stats = get_q_stats(current_Q1, current_Q2, mortality, random_action_sensitivity, zero_action_sensitivity)

        return losses, q_stats, grad_norm
        

    def evaluate(self, batch: TensorBatch) -> Dict[str, float]:
        self.critic.eval()  # Ensure critic is also in eval mode
        
        state, action, reward, next_state, next_action, done, mortality = TrainerUtils._unpack_batch(batch, self.device)

        # FQE: same swap as in train() -- see _get_next_action docstring.
        next_action = self._get_next_action(next_state, next_action)
        
        reward = reward * self.reward_scaling
        
        treat_survival_as_truncated = True
        if treat_survival_as_truncated:
                # make done 0 where mortality is 0, since we want to treat survival as truncated rather than terminal
            done = done * mortality        
        
        not_done = 1 - done

        with torch.no_grad():
            # compute the target Q value
            if self.reward_type == 'terminal': # using monte carlo returns
                target_Q = reward
            else: # using bootstrapped returns  
                target_Q = self.target_q(reward, not_done, next_state, next_action)

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            # critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            # use Huber loss instead
            critic_loss = (F.smooth_l1_loss(current_Q1, target_Q) + F.smooth_l1_loss(current_Q2, target_Q)) / 2.0
            
            losses = {"critic_loss": critic_loss.item()}
            
            # ADD THIS LINE:
            neuron_stats = self.critic.get_neuron_stats(state, action)

            # get Q statistics
            current_Q = (current_Q1 + current_Q2) / 2.0  
            random_action_sensitivity = self.random_action_values(state, action, current_Q)
            zero_action_sensitivity = self.zero_action_values(state, action, current_Q)
            q_stats = get_q_stats(current_Q1, current_Q2, mortality, random_action_sensitivity, zero_action_sensitivity)
        
        return losses, q_stats, neuron_stats
	

    def critic_forward_for_sarsa(self, batch: TensorBatch, action: torch.Tensor=None):
        self.critic.eval()  # Ensure critic is in eval mode

        if action is None:
            state, action, _, _, _, done, mortality = TrainerUtils._unpack_batch(batch, self.device)
        else:
            state, _, _, _, _, done, mortality = TrainerUtils._unpack_batch(batch, self.device)

        with torch.no_grad():
            current_Q1, current_Q2 = self.critic(state, action)
        
        current_Q = torch.min(current_Q1, current_Q2)  # TD3 style: take the minimum of the two critics to mitigate overestimation bias

        return current_Q


    def evaluate_sarsa(self, batch: TensorBatch, mortality: torch.Tensor, rl_actions: torch.Tensor) -> Dict[str, float]:
        
        self.critic.eval()  # Ensure critic is also in eval mode
        with torch.no_grad():        
            # generate q values from clinician actions
            q_clinician = self.critic_forward_for_sarsa(batch)
            # generate q values from rl actions
            q_rl = self.critic_forward_for_sarsa(batch, rl_actions)
        
        q_stats_sarsa = get_q_stats_sarsa(q_clinician, q_rl, mortality)

        return q_stats_sarsa



def get_adamw_optimizer(model, lr, weight_decay):
    """
    Sets up AdamW with weight decay only on Linear weights, 
    excluding biases and LayerNorm parameters.
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check for keywords that should NOT be decayed
        # Biases and LayerNorm 'weight' (which is a scale) and 'bias' (shift)
        if any(nd in name.lower() for nd in ['bias', 'norm']):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    return torch.optim.AdamW(optim_groups, lr=lr)


def get_trainer(config_manager, state_dim, action_dim, hyperparams, actor=None):

    learning_rate = float(hyperparams['learning_rate'])
    weight_decay = float(hyperparams.get('weight_decay', 0.01)) # New hyperparam
    device = config_manager.training_config['device']
    
    critic = Critic(
        state_dim, 
        action_dim, 
        dropout=hyperparams.get('dropout', 0.1),
        hidden_dim=hyperparams.get('hidden_dim', 512)
    ).to(device)  

    critic_optimizer = get_adamw_optimizer(critic, learning_rate, weight_decay)

    # 2. Define the Scheduler 
    # Using StepLR to "step down" the intensity every 3 epochs
    # scheduler = torch.optim.lr_scheduler.StepLR(critic_optimizer, step_size=10, gamma=0.5)
    critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		critic_optimizer, 
		T_max=hyperparams['total_steps'],
		eta_min=learning_rate * 0.1  # Final learning rate will be 10% of initial LR 
	)

    # --- FQE: `actor` is loaded and passed in EXTERNALLY (by training.py's
    # create_trainer / load_actor, via cql_with_logprobs.load_trained_actor)
    # -- this function does no path resolution of its own. actor=None
    # (the default, and what happens for every non-"fqe" model_type) means
    # ordinary SARSA; a real actor object here means FQE. ---
    trainer = Q_Learner(
        critic=critic,
        critic_optimizer=critic_optimizer,
        discount=hyperparams.get('discount', config_manager.training_config['discount']),
        device=device,
        tau=hyperparams.get('tau', config_manager.training_config['tau']),  
        grad_clip=hyperparams['grad_clip'], 
        reward_scaling=hyperparams.get('reward_scaling', 1.0),
        reward_type=config_manager.reward_type,
        scheduler=critic_scheduler,
        target_policy_actor=actor,
    )

    return trainer


def get_evaluator(state_dim, action_dim, config_manager):

    reward_type = config_manager.reward_type
    cohort = config_manager.cohort
    unit = config_manager.unit
    reward_cohort = config_manager.reward_cohort
    
    model_path = os.path.join(config_manager.paths['rl_models_dir'], 
        'basic_q', f'split_{config_manager.split_num}', cohort, 
        config_manager.actions)

    if unit.lower() == 'picu':
        if reward_cohort.lower() == 'picu':
            model_path = os.path.join(model_path, 
            "picu_patients_health_score_from_picu_dataset")
        elif reward_cohort.lower() == 'restricted':
            model_path = os.path.join(model_path, 
            "picu_patients_health_score_from_restricted_dataset")
        else:
            raise ValueError(f"Unknown reward_cohort: {reward_cohort}. Must be 'picu' or 'restricted'.")
    else:
        raise ValueError(f"Unknown unit: {unit}. Must be 'picu'.")

    if config_manager.include_actions_in_state == 'True':
        model_path = os.path.join(model_path, "with_actions")

        # currently just using the best run from the hypeparameter sweep (could switch to 
        # proper seed, but doesn't matter, this is just for online evaluation)

        if reward_type == 'terminal':             
                # load sorted_runs_health_score.csv
                sorted_runs_path = os.path.join(model_path, "sorted_runs_terminal.csv")
                if os.path.exists(sorted_runs_path):
                    sorted_runs = pd.read_csv(sorted_runs_path)
                    # get the last row, which has the best run
                    best_run = sorted_runs.iloc[0]
                    best_run_folder = best_run['run_dir']
                    model_path = os.path.join(model_path, best_run_folder)
        
        elif reward_type == 'health_score':
            # load sorted_runs_health_score.csv
            sorted_runs_path = os.path.join(model_path, "sorted_runs_health_score.csv")
            if os.path.exists(sorted_runs_path):
                sorted_runs = pd.read_csv(sorted_runs_path)
                # get the last row, which has the best run
                best_run = sorted_runs.iloc[0]
                best_run_folder = best_run['run_dir']
                model_path = os.path.join(model_path, best_run_folder)
            else:
                #### not using ######
                raise ValueError("Health score scale is not 0.0, but this is not supported in the current implementation.")
        
    else:
        model_path = os.path.join(model_path, "without_actions")    
        # we need to find the right model because we only have the five seeds here
        # find the folder inside model_path that begins with config_manager.hospital
        inner_folders = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f)) and f.startswith(config_manager.hospital)]
        if len(inner_folders) == 0 or len(inner_folders) > 1:
            raise ValueError(f"Could not find a unique model folder for hospital {config_manager.hospital} in {model_path}. Found: {inner_folders}")
        model_path = os.path.join(model_path, inner_folders[0])
        # find the folders inside that begin "unknown" or "local"
        inner_folders = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f)) and (f.startswith("unknown") or f.startswith("local"))]
        seed = config_manager.training_config['seed']
        for f in inner_folders:
            # load results.json from the folder and check if the seed matches
            results_path = os.path.join(model_path, f, "results.json")
            if os.path.exists(results_path):
                with open(results_path, "r") as file:
                    results = json.load(file)
                    if results["hyperparameters"]["seed"] == seed:
                        model_path = os.path.join(model_path, f)
                        break


    critic, hyperparams = load_critic(state_dim, action_dim,model_path=model_path)

    evaluator = Q_Learner(
        critic=critic,
        discount=hyperparams['discount'],
    )

    return evaluator

def load_critic(state_dim, action_dim, config_manager = None, model_path = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if model_path is None and config_manager is not None:
    #     model_path = os.path.join(config_manager.paths['rl_models_dir'], config_manager.model_type, 
    #                     f"split_{config_manager.split_num}", config_manager.cohort, config_manager.actions, "final_model")
    
    #     # get folder inside model_path
    #     model_folder = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))][0]
    #     model_path = os.path.join(model_path, model_folder)

    final_model_path = os.path.join(model_path, "final_model.pt")

    # load results.json from model_folder, which has state_dim 
    results_path = os.path.join(model_path, "results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
            hidden_dim = results["hyperparameters"]["hidden_dim"]

    critic = Critic(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
    checkpoint = torch.load(final_model_path, map_location=device, weights_only=False)
    critic.load_state_dict(checkpoint['critic_state_dict'])
    critic.eval()

    hyperparameters = results["hyperparameters"]

    return critic, hyperparameters