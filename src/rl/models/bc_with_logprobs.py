import typing
from typing import Dict, List, Tuple
import copy
import os
import json
# import config_manager
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from sklearn.metrics import roc_auc_score

from .shared_components import TrainerUtils # Actor, MixedActionActor, 

import logging
logger = logging.getLogger(__name__)

"""
Gemini is suggesting training only the Actor for first 1000 steps, claims this can allow can prevent the policy
from "overreacting" to initial, noisy Critic estimates. 
"""


TensorBatch = List[torch.Tensor]

class Actor(nn.Module):
	def __init__(
		self,
		state_dim: int,
		hidden_dim: int = 64,
		log_std_min: float = -2,
		log_std_max: float = 1., # -1.5, # 0.
		action_dim: int = 2,
		dropout: float = 0.1
	):
		super().__init__()
		self.log_std_min = log_std_min
		self.log_std_max = log_std_max

		# trunk — learns a joint patient representation
		self.trunk = nn.Sequential(
			nn.Linear(state_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.LayerNorm(hidden_dim), 
		)
		self.fluid_mean    = nn.Linear(hidden_dim, 1)		
		self.vaso_mean    = nn.Linear(hidden_dim, 1)
	
		self.fluid_log_std = nn.Linear(hidden_dim, 1)
		self.vaso_log_std = nn.Linear(hidden_dim, 1)

		self._init_weights()


	def _init_weights(self):
		for m in self.trunk:
			if isinstance(m, nn.Linear):
				nn.init.orthogonal_(m.weight, gain=1.0)
				nn.init.zeros_(m.bias)

		nn.init.zeros_(self.fluid_mean.weight)
		self.fluid_mean.bias.data.fill_(-0.405)

		nn.init.zeros_(self.vaso_mean.weight)
		self.vaso_mean.bias.data.fill_(-3.89)

		for layer in [self.fluid_log_std, self.vaso_log_std]:
			nn.init.zeros_(layer.weight)
			layer.bias.data.fill_(-2.25)


	def forward(self, state: torch.Tensor, deterministic: bool = False, 
			 return_dist=False, sigmoid_correction=True):
		z = self.trunk(state)
		# --- Fluid ---
		# f_mean = torch.sigmoid(self.fluid_mean(z))          # (0, 1)
		f_raw_mean = self.fluid_mean(z)       # (0, 1)
		# --- Vasopressor ---
		# v_mean = torch.sigmoid(self.vaso_mean(z))            # (0, 1)
		v_raw_mean = self.vaso_mean(z)            # (0, 1)

		f_log_std = torch.clamp(self.fluid_log_std(z), self.log_std_min, self.log_std_max)
		v_log_std = torch.clamp(self.vaso_log_std(z), self.log_std_min, self.log_std_max)

		f_std = f_log_std.exp()
		v_std = v_log_std.exp()

	
		if not deterministic:
			# Sample in unconstrained space, then squash with sigmoid
			# f_dist = torch.distributions.Normal(
			# 	torch.logit(f_mean.clamp(1e-6, 1 - 1e-6)), f_std
			# )
			f_dist = torch.distributions.Normal(f_raw_mean, f_std)

			# v_dist = torch.distributions.Normal(
			# 	torch.logit(v_mean.clamp(1e-6, 1 - 1e-6)), v_std
			# )
			v_dist = torch.distributions.Normal(v_raw_mean, v_std)

			f_raw = f_dist.rsample()
			v_raw = v_dist.rsample()

			f_action = torch.sigmoid(f_raw)
			v_action = torch.sigmoid(v_raw)

			if sigmoid_correction:
				# Log-prob with sigmoid change-of-variables correction
				f_log_prob = f_dist.log_prob(f_raw) - torch.log(
					f_action * (1 - f_action) + 1e-6
				)
				v_log_prob = v_dist.log_prob(v_raw) - torch.log(
					v_action * (1 - v_action) + 1e-6
				)

			else:
				f_log_prob = f_dist.log_prob(f_raw)
				v_log_prob = v_dist.log_prob(v_raw) 

			log_prob    = (f_log_prob + v_log_prob).squeeze(-1)
			actions     = torch.cat([f_action, v_action], dim=-1)
			# mean_actions = torch.cat([f_mean, v_mean], dim=-1)
		else:
			f_mean = torch.sigmoid(f_raw_mean)
			v_mean = torch.sigmoid(v_raw_mean)
			actions = torch.cat([f_mean, v_mean], dim=-1)
			log_prob = None
		
		if return_dist:
			return actions, log_prob, (f_mean, v_mean, f_std, v_std)

		return actions, log_prob


	def log_prob_of_action(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
		"""
		Evaluate the log-density of a GIVEN action under the current policy,
		as opposed to forward()'s stochastic branch, which samples a fresh
		action and returns its own log-prob. Needed for importance-sampling-based
		evaluation (WIS) and as a building block for entropy_loss().

		"""
		z = self.trunk(state)

		# f_mean = torch.sigmoid(self.fluid_mean(z)).clamp(1e-6, 1 - 1e-6)
		# v_mean = torch.sigmoid(self.vaso_mean(z)).clamp(1e-6, 1 - 1e-6)

		f_raw_mean = self.fluid_mean(z)          
		v_raw_mean = self.vaso_mean(z)  

		f_log_std = torch.clamp(self.fluid_log_std(z), self.log_std_min, self.log_std_max)
		v_log_std = torch.clamp(self.vaso_log_std(z), self.log_std_min, self.log_std_max)
		f_std = f_log_std.exp()
		v_std = v_log_std.exp()

		# Clamp the GIVEN action away from exactly 0/1 before inverting the
		# sigmoid — logit(0) and logit(1) are -inf/+inf, and real clinical
		# doses can legitimately be exactly 0 (e.g. no vasopressor given)
		a = action.clamp(1e-6, 1 - 1e-6)
		f_action = a[:, 0:1]
		v_action = a[:, 1:2]

		f_raw = torch.logit(f_action)
		v_raw = torch.logit(v_action)

		f_dist = torch.distributions.Normal(f_raw_mean, f_std)
		v_dist = torch.distributions.Normal(v_raw_mean, v_std)

		# Sigmoid-Jacobian correction, evaluated at the GIVEN action
		f_log_prob = f_dist.log_prob(f_raw) - torch.log(f_action * (1 - f_action) + 1e-6)
		v_log_prob = v_dist.log_prob(v_raw) - torch.log(v_action * (1 - v_action) + 1e-6)

		return (f_log_prob + v_log_prob).squeeze(-1)
	
	@torch.no_grad()
	def get_neuron_stats(self, state):
		self.eval()
		stats = {}
		relu_idx = 0

		def trace_sequential_block(sequential_block, current_input, block_name):
			nonlocal relu_idx
			x = current_input
			for layer in sequential_block:
				x = layer(x)
				if isinstance(layer, nn.ReLU):
					pct_zero = (x == 0).float().mean().item() * 100
					stats[f"{block_name}_zero_activation_L{relu_idx}_pct"] = pct_zero

					neuron_vars = torch.var(x, dim=0)
					pct_dead = (neuron_vars < 1e-6).float().mean().item() * 100
					stats[f"{block_name}_dead_neurons_L{relu_idx}_pct"] = pct_dead

					relu_idx += 1
			return x

		_ = trace_sequential_block(self.trunk, state, "trunk")
		return stats


def get_q_stats(current_Q1, current_Q2, actor_Q, mortality):
	
	# create current_Q as the mean of current_Q1 and current_Q2
	current_Q_mean = (current_Q1 + current_Q2) / 2.0
	current_Q_min = torch.min(current_Q1, current_Q2)
	mean_Q = current_Q_mean.mean().item()
	variance_Q = current_Q_mean.var().item()
	batch_n = current_Q_mean.shape[0]
	uncertainty_Q = F.mse_loss(current_Q1, current_Q2).item() # simple measure of uncertainty based on disagreement between the two critics

	# masks and counts
	survival_mask = (mortality == 0)
	non_survival_mask = (mortality == 1)

	survival_n = survival_mask.sum().item()
	non_survival_n = non_survival_mask.sum().item()
	
	# only compute subgroup mean/variance together when n > 2
	if survival_n > 2:
		survival_vals = current_Q_mean[survival_mask]
		survival_Q = survival_vals.mean().item()
		survival_variance_Q = survival_vals.var().item()
	else:
		survival_Q = float('nan')
		survival_variance_Q = float('nan')
		survival_n = 0

	if non_survival_n > 2:
		non_survival_vals = current_Q_mean[non_survival_mask]
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
		auc_Q = roc_auc_score(1-mortality.detach().cpu().numpy(), current_Q_mean.detach().cpu().numpy())

	else:
		signal_to_noise = float('nan')
		auc_Q = float('nan')

	q_stats = {
		"batch_n": batch_n,
		"mean_Q": mean_Q,
		"variance_Q": variance_Q,
		"uncertainty_Q": uncertainty_Q,                
		"survival_n": survival_n,
		"survival_Q": survival_Q,
		"survival_variance_Q": survival_variance_Q,
		"non_survival_n": non_survival_n,
		"non_survival_Q": non_survival_Q,
		"non_survival_variance_Q": non_survival_variance_Q,
		"signal_to_noise": signal_to_noise,
		"auc_Q": auc_Q
	}

	

	if actor_Q is not None:
		advantage_Q = actor_Q - current_Q_min # how much better the actor's action is compared to the current Q estimate, on average. This can help us understand if the policy is improving over time.
		advantage_Q_mean = (actor_Q - current_Q_min).mean().item() # how much better the actor's action is compared to the current Q estimate, on average. This can help us understand if the policy is improving over time.
		survival_advantage_Q = advantage_Q[survival_mask].mean().item() if survival_n > 0 else float('nan')
		non_survival_advantage_Q = advantage_Q[non_survival_mask].mean().item() if non_survival_n > 0 else float('nan')
		
		q_stats.update({
			"advantage_Q_mean": advantage_Q_mean,
			"survival_advantage_Q": survival_advantage_Q,
			"non_survival_advantage_Q": non_survival_advantage_Q
		})	
				
	return q_stats



class BC:
	def __init__(
		self,
		
		use_smoothing: bool = False,

		actor: nn.Module = None,
		actor_optimizer: torch.optim.Optimizer = None,
		actor_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
	
		temp: float = 1.0,         # Temperature for the LogSumExp
		num_samples: int = 10,   # How many random actions to sample for the penalty

		device: str = "CUDA" if torch.cuda.is_available() else "CPU",
		action_mode: str = "continuous",
		dose_actions: int = 2,  #  Number of dose actions
		binary_actions: int = 0,		
		grad_clip: float = None,  # Add grad_clip argument
		ema_alpha: float = 0.99,  # Add ema_alpha argument for moving


	):
		
		self.use_smoothing = use_smoothing

		self.actor = actor
		self.actor_optimizer = actor_optimizer
		self.actor_scheduler = actor_scheduler
		
		self.device = device
		self.action_mode = action_mode
		self.dose_actions = dose_actions
		self.binary_actions = binary_actions
		self.grad_clip = grad_clip		

		# for calculating moving average of gradient clipping binary
		self.actor_clip_fraction_ema = 0.0
		self.ema_alpha = ema_alpha # Closer to 1.0 means more smoothing
		
		self.total_it = 0

		self.utils = TrainerUtils()

		
	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		actions, _ = self.actor(state, deterministic=True)
		return actions.cpu().data.numpy().flatten()


	def train(self, batch: TensorBatch) -> Tuple[Dict[str, float], Dict[str, float]]:
		self.total_it += 1

		# Sample replay buffer 
		state, action, reward, next_state, next_action, done, mortality = TrainerUtils._unpack_batch(batch, self.device)
		
		if self.use_smoothing is True:
			with torch.no_grad():
				# add noise to states
				state_std = state.std(dim=0, keepdim=True)
				noise_scale = state_std * 0.04 + 1e-6
				state_noise = torch.randn_like(state) * noise_scale
				smoothed_states = state + state_noise
				
				# add noise to actions
				critic_action_noise = (torch.randn_like(action) * 0.04).clamp(-0.1, 0.1)
				smoothed_actions = action + critic_action_noise
				is_zero_dose = (action < 1e-4)
				smoothed_actions = torch.where(is_zero_dose, torch.zeros_like(smoothed_actions), smoothed_actions)			
				smoothed_actions = smoothed_actions.clamp(0, 1.0) # Ensure actions remain in valid range after smoothing

		# Compute actor loss
		self.actor.train() # Dropout ACTIVE for the network we are updating
		
		# Sample actions from the stochastic actor
		log_prob = self.actor.log_prob_of_action(state, action) # log probability of the action taken by the doctor under the current policy
		
		actor_loss = -log_prob.mean() # negative log likelihood of the action taken by the doctor under the current policy
		losses = {
			"actor_loss": actor_loss.item(),
		}

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		
		# clip actor gradients if specified
		total_actor_grad_norm = self._get_grad_norm(self.actor)

		grad_norm = {}
		grad_norm['actor_grad_norm'] = total_actor_grad_norm

		if self.grad_clip is not None:				
			grad_norm['actor_was_clipped'] = 1.0 if total_actor_grad_norm > self.grad_clip else 0.0

			# update the moving average of the clipping binary
			self.actor_clip_fraction_ema = self.ema_alpha * self.actor_clip_fraction_ema + (1 - self.ema_alpha) * grad_norm['actor_was_clipped']
			grad_norm['actor_clip_fraction_ema'] = self.actor_clip_fraction_ema

			torch.nn.utils.clip_grad_norm_(
				self.actor.parameters(), 
				self.grad_clip
			)
		self.actor_optimizer.step()
		if self.actor_scheduler is not None:
			self.actor_scheduler.step()
			# log_dict.update(lr_actor=self.actor_optimizer.param_groups[0]['lr'])
			losses.update(lr_actor=self.actor_optimizer.param_groups[0]['lr'])
		
		self.actor.eval()
		with torch.no_grad():
			_, _, (f_mean, v_mean, f_std, v_std) = self.actor(
				state, deterministic=True, return_dist=True, 
				sigmoid_correction=True
			)
			
			fluid_mse = F.mse_loss(f_mean, action[:, 0:1]).item()
			vaso_mse = F.mse_loss(v_mean, action[:, 1:2]).item()

		losses["fluid_mean_pred"] = f_mean.mean().item()
		losses["vaso_mean_pred"] = v_mean.mean().item()
		losses["fluid_std_mean"] = f_std.mean().item()
		losses["vaso_std_mean"] = v_std.mean().item()
		losses["fluid_mse"] = fluid_mse
		losses["vaso_mse"] = vaso_mse

		# return log_dict, grad_norm
		return losses, _, grad_norm


	def actor_forward_for_sarsa(self, batch: TensorBatch) -> torch.Tensor:
		state, _, _, _, _, _, _ = TrainerUtils._unpack_batch(batch, self.device)

		self.actor.eval()  # Set actor to eval mode to disable Dropout
		with torch.no_grad():
			new_actions, _ = self.actor(state, deterministic=True)  # Get the "best" action without noise or log probabilities
		
		return new_actions


	def _get_grad_norm(self, model: nn.Module) -> float:
		total_norm = 0
		for p in model.parameters():
			if p.grad is not None:
				param_norm = p.grad.detach().data.norm(2)
				total_norm += param_norm.item() ** 2
		return total_norm ** 0.5
	

	def evaluate(self, batch: TensorBatch) -> Dict[str, float]:
		self.actor.eval()  # This is correct and important!
		
		state, action, reward, next_state, next_action, done, mortality = TrainerUtils._unpack_batch(batch, self.device)
		
		with torch.no_grad():
			# Sample actions from the stochastic actor
			log_prob = self.actor.log_prob_of_action(state, action) # log probability of the action taken by the doctor under the current policy
			
			actor_loss = -log_prob.mean() # negative log likelihood of the action taken by the doctor under the current policy
			losses = {
				"actor_loss": actor_loss.item(),
			}
			
			# ADD THIS LINE:
			neuron_stats = self.actor.get_neuron_stats(state)
	
			_, _, (f_mean, v_mean, f_std, v_std) = self.actor(
				state, deterministic=True, return_dist=True, 
				sigmoid_correction=True
			)

			fluid_mse = F.mse_loss(f_mean, action[:, 0:1]).item()
			vaso_mse = F.mse_loss(v_mean, action[:, 1:2]).item()
	
			losses["fluid_mean_pred"] = f_mean.mean().item()
			losses["vaso_mean_pred"] = v_mean.mean().item()
			losses["fluid_std_mean"] = f_std.mean().item()
			losses["vaso_std_mean"] = v_std.mean().item()
			losses["fluid_mse"] = fluid_mse
			losses["vaso_mse"] = vaso_mse

		# return log_dict, action, pi
		return losses, _, neuron_stats

	
	def state_dict(self) -> Dict[str, torch.Tensor]:
		return {
			"actor": self.actor.state_dict(),
			"actor_optimizer": self.actor_optimizer.state_dict(),
			"critic": self.critic.state_dict(),
			"critic_optimizer": self.critic_optimizer.state_dict(),
			"actor_target": self.actor_target.state_dict(),
			"critic_target": self.critic_target.state_dict()
			
		}


	######################## currently unused ###############################
	def save(self, filename):
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

	######################## currently unused ###############################
	def load(self, filename):
		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)


def get_trainer(config_manager, state_dim, action_dim, hyperparams):

	learning_rate_actor = float(hyperparams['learning_rate_actor'])
	weight_decay = float(hyperparams.get('weight_decay', 0.01)) # New hyperparam
	device = config_manager.training_config['device']

	
	actor = Actor(
		state_dim=state_dim, 
		action_dim=action_dim, 
		# hidden_dim=hyperparams['hidden_dim'], 
		dropout=hyperparams['dropout']
	).to(config_manager.training_config['device'])
	
	actor_optimizer = torch.optim.AdamW(
		actor.parameters(), 
		lr=learning_rate_actor,
		weight_decay=weight_decay
		)
	
	actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		actor_optimizer, 
		T_max=hyperparams['total_steps']//hyperparams['policy_freq'], # we step the actor scheduler every policy_freq steps, so T_max should be total_steps // policy_freq
		eta_min=learning_rate_actor * .10
	)
	
	
	trainer = BC(               # LIST OF PARAMETERS NEEDS TO BE UPDATED
		actor=actor,
		actor_optimizer=actor_optimizer,
		actor_scheduler=actor_scheduler,
		device=config_manager.training_config['device'],
		action_mode=config_manager.action_mode,

		grad_clip=hyperparams['grad_clip'],


		use_smoothing=hyperparams['use_smoothing'],

		# ratio_of_fluid_to_vaso=config_manager.training_config['ratio_of_fluid_to_vaso']
	)

	return trainer
