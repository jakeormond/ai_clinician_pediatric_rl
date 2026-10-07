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
import pandas as pd
import pickle
import math

from .shared_components import TrainerUtils # Actor, MixedActionActor, 

import logging
logger = logging.getLogger(__name__)

"""
Gemini is suggesting training only the Actor for first 1000 steps, claims this can allow can prevent the policy
from "overreacting" to initial, noisy Critic estimates. 

Claude: Hurdle-model changes to Actor + BC.
 
Core idea: instead of one squashed Gaussian per action trying to represent
both "P(dose=0)" and "how much, given dose>0" at once, split each action
into:
  1. a Bernoulli GATE head:      P(dose > 0 | state)
  2. a magnitude head (as before): Normal-in-logit-space, but its loss is
     now only evaluated/backpropagated on timesteps where dose > 0.
 
This stops the ~96%-zero vaso targets (and ~55%-zero fluid targets) from
dragging the magnitude mean/std toward the zero cluster. It also fixes the
std-floor bug: the old log_std bias (-2.25) sat outside the clamp range
[-2, 1], so torch.clamp zeroed its gradient from step 0. Fixed below.
"""


TensorBatch = List[torch.Tensor]

class Actor(nn.Module):
	def __init__(
		self,
		state_dim: int,
		hidden_dim: int = 64,
		log_std_min: float = -2,
		log_std_max: float = 1.,
		dropout: float = 0.1,
		zero_threshold: float = 1e-4,
		gate_temperature: float = 1.0,   # relaxation temperature for differentiable gate sampling
	):
		super().__init__()
		self.log_std_min = log_std_min
		self.log_std_max = log_std_max
		self.zero_threshold = zero_threshold
		self.gate_smooth = 0.02
		self.gate_temperature = gate_temperature
 
		# --- single shared trunk (matches BC actor) ---
		self.trunk = nn.Sequential(
			nn.Linear(state_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
		)
		self.fluid_mean = nn.Linear(hidden_dim, 1)
		self.vaso_mean = nn.Linear(hidden_dim, 1)
		self.fluid_gate = nn.Linear(hidden_dim, 1)
		self.vaso_gate = nn.Linear(hidden_dim, 1)
		self.fluid_log_std = nn.Linear(hidden_dim, 1)
		self.vaso_log_std = nn.Linear(hidden_dim, 1)
 
		self._init_weights()


	def _init_weights(self):
		for m in self.trunk:
			if isinstance(m, nn.Linear):
				nn.init.orthogonal_(m.weight, gain=1.0)
				nn.init.zeros_(m.bias)
 
		nn.init.zeros_(self.fluid_mean.weight)
		self.fluid_mean.bias.data.fill_(0.0)
		nn.init.zeros_(self.vaso_mean.weight)
		self.vaso_mean.bias.data.fill_(0.0)
 
		for layer in [self.fluid_log_std, self.vaso_log_std]:
			nn.init.zeros_(layer.weight)
			layer.bias.data.fill_(-1.0)   # -1.0 gives std ≈ 0.37, which is a reasonable starting point for exploration

		# match known base rates — same values as the BC actor
		nn.init.zeros_(self.fluid_gate.weight)
		self.fluid_gate.bias.data.fill_(0.78)    # sigmoid(0.78) ≈ 0.686 fluid nonzero rate
		nn.init.zeros_(self.vaso_gate.weight)
		self.vaso_gate.bias.data.fill_(-2.98)    # sigmoid(-2.98) ≈ 0.048 vaso nonzero rate
 

	def _dists(self, state):
		z = self.trunk(state)

		f_gate_logit = self.fluid_gate(z)
		v_gate_logit = self.vaso_gate(z)

		f_raw_mean = self.fluid_mean(z)
		v_raw_mean = self.vaso_mean(z)

		f_log_std = torch.clamp(self.fluid_log_std(z), self.log_std_min, self.log_std_max)
		v_log_std = torch.clamp(self.vaso_log_std(z), self.log_std_min, self.log_std_max)
		f_std = f_log_std.exp()
		v_std = v_log_std.exp()

		return f_gate_logit, v_gate_logit, f_raw_mean, v_raw_mean, f_std, v_std
	

	def _relaxed_gate_sample(self, gate_logit, hard=False):
		"""
		Differentiable Bernoulli sample via Gumbel-sigmoid (binary Concrete
		distribution). Gradient flows into gate_logit through this sample.
 
		hard=True applies a straight-through estimator: forward pass uses a
		hard 0/1 value, but gradient flows as if it were the soft/relaxed
		sample (standard straight-through trick). Use hard=False during
		normal training so downstream Q(s,a) sees the true continuous
		relaxation; hard=True if you specifically need a binary-valued
		action while still training through it (e.g. matching a hard
		zero_threshold in the critic's expected input distribution).

		NOTE THAT CURRENTLY HARD IS NEVER SET TO FALSE IN TRAINING
		"""
		u = torch.rand_like(gate_logit).clamp(1e-6, 1 - 1e-6)
		gumbel_noise = torch.log(u) - torch.log(1 - u)

		# with temperature low enough, this effectively acts like a bernoulli sample, producing vanishing gradients
		soft = torch.sigmoid((gate_logit + gumbel_noise) / self.gate_temperature)
 
		if not hard:
			return soft
 
		hard_sample = (soft > 0.5).float() # PyTorch has no defined gradient for >, so this contributes exactly 0 gradient.
		# straight-through: forward uses hard_sample, backward uses soft's gradient, even though it doesn't contribute to 
		# the forward pass. This is a standard trick for discrete relaxations.
		return hard_sample + soft - soft.detach()
	

	def forward(self, state: torch.Tensor, deterministic: bool = False,
			 return_dist=False, sigmoid_correction=True, hard_gate: bool = False,
			 return_components: bool = False) -> Tuple[torch.Tensor, torch.Tensor, typing.Optional[Dict[str, torch.Tensor]], typing.Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]]:
		f_gate_logit, v_gate_logit, f_raw_mean, v_raw_mean, f_std, v_std = self._dists(state)
		f_p = torch.sigmoid(f_gate_logit) 
		v_p = torch.sigmoid(v_gate_logit)
 
		if not deterministic: # training path: sample from the distribution
			# differentiable gate sample — gradient flows into fluid_gate/vaso_gate
			f_fire = self._relaxed_gate_sample(f_gate_logit, hard=hard_gate) 
			v_fire = self._relaxed_gate_sample(v_gate_logit, hard=hard_gate) 

			# create distributions for the magnitude component using mean and std, then sample from them
			f_dist = torch.distributions.Normal(f_raw_mean, f_std)
			v_dist = torch.distributions.Normal(v_raw_mean, v_std)
			f_raw = f_dist.rsample()
			v_raw = v_dist.rsample()
			f_mag = torch.sigmoid(f_raw)
			v_mag = torch.sigmoid(v_raw)
 
			f_action = f_fire * f_mag
			v_action = v_fire * v_mag
			actions = torch.cat([f_action, v_action], dim=-1)
 
			# entropy term: use the *soft* probability (not the relaxed sample)
			# for a stable log-prob estimate of the gate component — this is
			# the standard choice for the concrete-distribution entropy proxy
			f_gate_lp = f_fire * torch.log(f_p + 1e-8) + (1 - f_fire) * torch.log(1 - f_p + 1e-8)
			v_gate_lp = v_fire * torch.log(v_p + 1e-8) + (1 - v_fire) * torch.log(1 - v_p + 1e-8)
 
			if sigmoid_correction:
				f_mag_lp = f_dist.log_prob(f_raw) - torch.log(f_mag * (1 - f_mag) + 1e-6)
				v_mag_lp = v_dist.log_prob(v_raw) - torch.log(v_mag * (1 - v_mag) + 1e-6)
			else:
				f_mag_lp = f_dist.log_prob(f_raw)
				v_mag_lp = v_dist.log_prob(v_raw)
 
			# note: f_fire is now continuous-valued (soft gate), so this
			# naturally down-weights the magnitude log-prob's contribution
			# in proportion to how "off" the gate currently is — a soft
			# analogue of the hard mask used in BC.
			gate_lp = (f_gate_lp + v_gate_lp).squeeze(-1)
			mag_lp = (f_fire * f_mag_lp + v_fire * v_mag_lp).squeeze(-1)
			log_prob = (f_gate_lp + f_fire * f_mag_lp + v_gate_lp + v_fire * v_mag_lp).squeeze(-1)

			if return_components:
				components = {
					"f_fire": f_fire, "v_fire": v_fire,
					"f_mag_lp": f_mag_lp, "v_mag_lp": v_mag_lp,
					"f_gate_logit": f_gate_logit, "v_gate_logit": v_gate_logit,
				}

		else:
			f_mean_mag = torch.sigmoid(f_raw_mean)
			v_mean_mag = torch.sigmoid(v_raw_mean)
			# deterministic (eval / target network) path: no gradient needed,
			# hard threshold is fine and matches BC's convention.
			f_action = (f_p > 0.5).float() * f_mean_mag
			v_action = (v_p > 0.5).float() * v_mean_mag
			actions = torch.cat([f_action, v_action], dim=-1)
			gate_lp = None
			mag_lp = None
			log_prob = None

			if return_components:
				raise ValueError(
					"return_components=True is not supported with deterministic=True: "
					"f_fire/v_fire/f_mag_lp/v_mag_lp are only meaningful for the "
					"stochastic sampling path. Call with deterministic=False if you "
					"need these components."
				)

		if return_dist:
			result = (actions, log_prob, (gate_lp, mag_lp), (f_p, v_p, f_raw_mean, v_raw_mean, f_std, v_std))
		else:
			result = (actions, log_prob, (gate_lp, mag_lp))

		if return_components:
			return result + (components,)
		return result
 
		

	def log_prob_of_action(self, state: torch.Tensor, action: torch.Tensor, return_components=False) -> torch.Tensor:
		"""
		Hurdle log-likelihood of a GIVEN (dataset) action. Used for WIS eval
		and/or a CQL+BC regularization term. (The original version of this
		actor referenced a nonexistent self.trunk here — fixed now that
		there's a single real trunk to call.)
		"""
		f_gate_logit, v_gate_logit, f_raw_mean, v_raw_mean, f_std, v_std = self._dists(state)
 
		f_dose = action[:, 0:1]
		v_dose = action[:, 1:2]
 
		f_fired = (f_dose > self.zero_threshold).float()
		v_fired = (v_dose > self.zero_threshold).float()
 
		f_fired_smooth = f_fired * (1 - 2 * self.gate_smooth) + self.gate_smooth
		v_fired_smooth = v_fired * (1 - 2 * self.gate_smooth) + self.gate_smooth
 
		f_gate_lp = -F.binary_cross_entropy_with_logits(f_gate_logit, f_fired_smooth, reduction='none')
		v_gate_lp = -F.binary_cross_entropy_with_logits(v_gate_logit, v_fired_smooth, reduction='none')
 
		f_a = f_dose.clamp(1e-4, 1 - 1e-4)
		v_a = v_dose.clamp(1e-4, 1 - 1e-4)
		f_raw = torch.logit(f_a)
		v_raw = torch.logit(v_a)
 
		f_dist = torch.distributions.Normal(f_raw_mean, f_std)
		v_dist = torch.distributions.Normal(v_raw_mean, v_std)
 
		f_mag_lp = f_dist.log_prob(f_raw) - torch.log(f_a * (1 - f_a) + 1e-6)
		v_mag_lp = v_dist.log_prob(v_raw) - torch.log(v_a * (1 - v_a) + 1e-6)
 
		total_lp = (
			f_gate_lp + f_fired * f_mag_lp
			+ v_gate_lp + v_fired * v_mag_lp
		)
		total_lp = total_lp.squeeze(-1)
 
		if return_components:
			components = {
				"fluid_gate_lp": f_gate_lp.squeeze(-1),
				"vaso_gate_lp": v_gate_lp.squeeze(-1),
				"fluid_mag_lp": f_mag_lp.squeeze(-1),
				"vaso_mag_lp": v_mag_lp.squeeze(-1),
				"fluid_fired": f_fired.squeeze(-1),
				"vaso_fired": v_fired.squeeze(-1),
			}
			return total_lp, components
 
		return total_lp
	

	@torch.no_grad()
	def log_prob_for_wis(self, state, action, bc_actor: nn.Module):
		"""
		Constructs an evaluation density around the CQL policy's DETERMINISTIC
		action (mean, hard-gated), using BC's per-state fitted std as the
		bandwidth. This is purely for WIS — never used in training. The std
		values are borrowed from BC (already a well-calibrated MLE fit to real
		doctor variability), not re-derived from CQL's own training dynamics,
		which is exactly the decoupling we want.
		"""
		bc_actor.eval()  # ensure BC actor is in eval mode to avoid Dropout interference

		f_gate_logit, v_gate_logit, f_raw_mean, v_raw_mean, _, _ = self._dists(state)
		f_p = torch.sigmoid(f_gate_logit)
		v_p = torch.sigmoid(v_gate_logit)

		# pull BC's per-state std for this same batch of states
		_, _, _, (_, _, _, _, bc_f_std, bc_v_std) = bc_actor(
			state, return_dist=True
		)

		f_dose, v_dose = action[:, 0:1], action[:, 1:2]
		f_fired_actual = (f_dose > self.zero_threshold).float()
		v_fired_actual = (v_dose > self.zero_threshold).float()

		f_gate_lp = f_fired_actual * torch.log(f_p + 1e-8) + (1 - f_fired_actual) * torch.log(1 - f_p + 1e-8)
		v_gate_lp = v_fired_actual * torch.log(v_p + 1e-8) + (1 - v_fired_actual) * torch.log(1 - v_p + 1e-8)

		f_a = f_dose.clamp(1e-4, 1 - 1e-4)
		v_a = v_dose.clamp(1e-4, 1 - 1e-4)

		f_dist = torch.distributions.Normal(f_raw_mean, bc_f_std)
		v_dist = torch.distributions.Normal(v_raw_mean, bc_v_std)

		f_mag_lp = f_dist.log_prob(torch.logit(f_a)) - torch.log(f_a * (1 - f_a) + 1e-6)
		v_mag_lp = v_dist.log_prob(torch.logit(v_a)) - torch.log(v_a * (1 - v_a) + 1e-6)

		total_lp = f_gate_lp + f_fired_actual * f_mag_lp + v_gate_lp + v_fired_actual * v_mag_lp
		return total_lp.squeeze(-1)




class Critic(nn.Module):
	def __init__(
		self, 
		state_dim, 
		action_dim, 
		dropout=0.1, 
		hidden_dim=64, # 512 
		negative_slope=0.1,
	):		
		super(Critic, self).__init__()

		self.hidden_dim = hidden_dim
		# self.hidden_dim2 = hidden_dim // 2  # Example of a second hidden layer with half the units of the first
		# self.hidden_dim3 = self.hidden_dim2 // 2  # Example of a third hidden layer with half the units of the second

		self.negative_slope = negative_slope

		# standardize the incoming embeddings
		self.state_norm = nn.LayerNorm(state_dim) # if cql loss stays close to zero, we may
		
		# State + action fusion — no need to compress first,
        # just fuse directly into a small hidden space
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
	def get_neuron_stats(self, state, action): # not currently used, make sure to use eval when implemented
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
	

	def get_all_q_values(self, state, actions):
		"""
		Evaluates Q(s, a) for a batch of states and multiple actions per state.
		state: [batch_size, state_dim]
		actions: [batch_size, num_samples, action_dim]
		"""
		batch_size, num_samples, action_dim = actions.shape
		state_dim = state.shape[-1]

		# Expand state to match the number of samples: [batch_size, num_samples, state_dim]
		state = state.unsqueeze(1).repeat(1, num_samples, 1)

		# Flatten to pseudo-batch: [batch_size * num_samples, dim]
		flat_state = state.view(-1, state_dim)
		flat_actions = actions.view(-1, action_dim)

		# Get Q-values
		q1, q2 = self.forward(flat_state, flat_actions)

		# Reshape back to [batch_size, num_samples]
		return q1.view(batch_size, num_samples), q2.view(batch_size, num_samples) 
	

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
		
		# log_dict.update({
		#     "actor_Q": actor_Q.mean().item(),
		#     "advantage_Q": advantage_Q_mean,
		#     "survival_advantage_Q": survival_advantage_Q,
		#     "non_survival_advantage_Q": non_survival_advantage_Q
		# })
			
	return q_stats



class CQL:
	def __init__(
		self,
		
		# use_smoothing: bool = False,

		actor: nn.Module = None,
		actor_optimizer: torch.optim.Optimizer = None,
		actor_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
		critic: nn.Module = None,
		critic_optimizer: torch.optim.Optimizer = None,
		critic_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
		discount: float = 0.99,
		policy_noise: float = 0.2,
		noise_clip: float = 0.05,
		policy_freq: int = 2,

		reward_type: str = None,  # can be 'terminal' or 'health_score'

		cql_alpha: float = 2.0,  # Use the alpha arg as the CQL conservative weight
		temp: float = 1.0,         # Temperature for the LogSumExp
		num_samples: int = 10,   # How many random actions to sample for the penalty

		actor_loss_q_scale: float = 50.0,   # Weight for the Q loss in the actor update
		actor_loss_gate_entropy_scale: float = 0.0, # Weight for the entropy loss in the actor update
		# actor_loss_mag_entropy_scale: float = 5.e-4, # Weight for the entropy loss in the actor update		
		target_std_fluid: float = 0.5,    # desired raw (pre-sigmoid) std for fluid magnitude
		target_std_vaso: float = 0.5,     # desired raw (pre-sigmoid) std for vaso magnitude
		initial_alpha_mag: float = 5e-4,  # starting value for the learned entropy-temperature
		alpha_mag_lr: float = 1e-3,       # learning rate for the dual-variable optimizer
		fluid_mean_reg_scale: float = 0.05,  # Weight for the fluid mean regularization term
		vaso_mean_reg_scale: float = 0.05,  # Weight for the vaso mean regularization term

		# Vaso protection (None = off). With both off this reproduces the no_floor_bound runs exactly.
		vaso_mean_floor: typing.Optional[float] = None,   # raw-logit floor for vaso magnitude mean; None = off
		vaso_mean_floor_scale: float = 0.05,

		vaso_gate_logit_bound: typing.Optional[float] = None,   # |vaso gate logit| bound; None = off
		vaso_gate_logit_bound_scale: float = 0.05,

		device: str = "CUDA" if torch.cuda.is_available() else "CPU",
		action_mode: str = "continuous",
		dose_actions: int = 2,  #  Number of dose actions
		binary_actions: int = 0,		
		tau: float = 0.005,
		grad_clip: float = None,  # Add grad_clip argument
		ema_alpha: float = 0.99,  # Add ema_alpha argument for moving
		reward_scaling: float = 1.0,

		ratio_of_fluid_to_vaso: float = 1.0,  # Add ratio_of_fluid_to_vaso argument

		gate_temp_start: float = 1.0,   # e.g. 1.0
		gate_temp_end: float = 0.3,       # e.g. 0.3
		total_steps: int = 100000

	):
		
		# self.use_smoothing = use_smoothing

		self.actor = actor
		self.actor_optimizer = actor_optimizer
		self.actor_scheduler = actor_scheduler
		self.critic = critic
		self.critic_optimizer = critic_optimizer
		self.critic_scheduler = critic_scheduler

		self.discount = discount
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.cql_alpha = cql_alpha  # Add cql_alpha as an argument and assign it to the instance variable
		self.temp = temp         # Add temp as an argument and assign it to the instance variable
		self.num_samples = num_samples   # Add num_samples as an argument and assign it to the instance variable

		self.reward_type = reward_type  # can be 'terminal' or 'health_score'
	
		self.actor_loss_q_scale = actor_loss_q_scale
		self.actor_loss_gate_entropy_scale = actor_loss_gate_entropy_scale
		# self.actor_loss_mag_entropy_scale = actor_loss_mag_entropy_scale
		self.fluid_mean_reg_scale = fluid_mean_reg_scale
		self.vaso_mean_reg_scale = vaso_mean_reg_scale

		self.vaso_mean_floor = vaso_mean_floor
		self.vaso_mean_floor_scale = vaso_mean_floor_scale

		self.vaso_gate_logit_bound = vaso_gate_logit_bound
		self.vaso_gate_logit_bound_scale = vaso_gate_logit_bound_scale

		self.device = device
		self.action_mode = action_mode
		self.dose_actions = dose_actions
		self.binary_actions = binary_actions
		self.tau = tau
		self.grad_clip = grad_clip		

		# for calculating moving average of gradient clipping binary
		self.critic_clip_fraction_ema = 0.0
		self.actor_clip_fraction_ema = 0.0
		self.ema_alpha = ema_alpha # Closer to 1.0 means more smoothing
		
		self.reward_scaling = reward_scaling

		self.ratio_of_fluid_to_vaso = ratio_of_fluid_to_vaso

		self.total_it = 0

		self.utils = TrainerUtils()

		# copy the actor network to create the target actor
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_target.eval()  # Target networks start in eval mode to disable Dropout

		# copy the critic network to create the target critic
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_target.eval()  # Target networks start in eval mode to disable Dropout

		# gate temp
		# in CQL.__init__, alongside other hyperparams
		self.gate_temp_start = gate_temp_start   # e.g. 1.0
		self.gate_temp_end = gate_temp_end       # e.g. 0.3
		self.gate_temp_anneal_steps = total_steps //  2  # e.g. total_steps // 2

		# --- automatic magnitude-entropy tuning (SAC-style dual variables) ---
		# Target differential entropy for a Gaussian with the desired raw-space
		# std: H = 0.5 * log(2*pi*e*sigma^2). Fixed constants, not learned.
		self.target_entropy_mag_fluid = 0.5 * np.log(2 * np.pi * np.e * target_std_fluid ** 2)
		self.target_entropy_mag_vaso = 0.5 * np.log(2 * np.pi * np.e * target_std_vaso ** 2)

		self.log_alpha_mag_fluid = torch.tensor(
			np.log(initial_alpha_mag), dtype=torch.float32, device=self.device, requires_grad=True
		)
		self.log_alpha_mag_vaso = torch.tensor(
			np.log(initial_alpha_mag), dtype=torch.float32, device=self.device, requires_grad=True
		)
		self.alpha_mag_optimizer = torch.optim.Adam(
			[self.log_alpha_mag_fluid, self.log_alpha_mag_vaso], lr=alpha_mag_lr
		)


	def _update_gate_temperature(self):
		frac = min(1.0, self.total_it / self.gate_temp_anneal_steps)
		# exponential decay is the more standard choice for Gumbel-softmax schedules
		new_temp = self.gate_temp_start * (self.gate_temp_end / self.gate_temp_start) ** frac
		self.actor.gate_temperature = new_temp
		self.actor_target.gate_temperature = new_temp   # keep target actor consistent too
		return new_temp

	# def select_action(self, state):
	# 	state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
	# 	actions, _ = self.actor(state, deterministic=True)
	# 	return actions.cpu().data.numpy().flatten()


	def compute_actor_loss(self,pi: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
		"""
		Refined BC Loss for sparse clinical actions (Fluids/Vasopressors).
		"""
		############# MACRO LOSS #############
		fluid_volume_loss = (pi[:, 0].mean() - action[:, 0].mean()) ** 2
		vaso_volume_loss  = (pi[:, 1].mean()  - action[:, 1].mean()) ** 2		
		vaso_volume_loss_weight = self.ratio_of_fluid_to_vaso
		
		total_loss = fluid_volume_loss + vaso_volume_loss_weight*vaso_volume_loss
		actor_dict = {
			"total_bc_loss": total_loss,
			"fluid_volume_loss": fluid_volume_loss,
			"vaso_volume_loss": vaso_volume_loss,
			# "mse_fluid_loss": micro_fluid_loss,
			# "mse_vaso_loss": micro_vaso_loss
		}
		
		return actor_dict
		

	def target_q(self, action, next_state, reward, not_done, noise=True):
		with torch.no_grad():
			# Select action according to policy and add clipped noise
			next_action, _, _ = self.actor_target(next_state, deterministic=True)
		
			if noise:
				noise = (torch.randn_like(next_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
				next_action = (next_action + noise).clamp(0, 1.0)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)

			# Bellman equation
			target_Q = reward + not_done * self.discount * target_Q			
			
			return target_Q


	def sample_zero_occupancy_actions(self, batch_size, num_samples):
		"""
		For each of batch_size*num_samples draws, pick a random zero-occupancy
		bin and sample an action uniformly within its bounds.
		"""
		n_bins = self.zero_occupancy_action_pool.shape[0]
		bin_idx = torch.randint(0, n_bins, (batch_size, num_samples), device=self.device)
		bins = self.zero_occupancy_action_pool[bin_idx]  # (batch, num_samples, 4)

		f_lo, f_hi, v_lo, v_hi = bins[..., 0], bins[..., 1], bins[..., 2], bins[..., 3]
		f_sample = f_lo + torch.rand_like(f_lo) * (f_hi - f_lo)
		v_sample = v_lo + torch.rand_like(v_lo) * (v_hi - v_lo)

		return torch.stack([f_sample, v_sample], dim=-1)  # (batch, num_samples, 2)


	def train(self, batch: TensorBatch) -> Tuple[Dict[str, float], Dict[str, float]]:
		self.total_it += 1

		# this is for logging, gate temp is used in the actor's forward pass to sample the gates
		current_gate_temp = self._update_gate_temperature()

		# Sample replay buffer 
		state, action, reward, next_state, next_action, done, mortality = TrainerUtils._unpack_batch(batch, self.device)
		
		# scaling is just 1 for now, no need to adjust
		reward = reward * self.reward_scaling
		
		########################## CRITIC TRAINING ##########################
		# compute the target Q value
		if self.reward_type != 'terminal': # using monte carlo returns  
			treat_survival_as_truncated = True # i.e. survival is not terminal. 
			if treat_survival_as_truncated:
				# make done 0 where mortality is 0, since we want to treat survival as truncated rather than terminal
				done = done * mortality		
			not_done = 1 - done

		self.critic.train()          # Dropout ACTIVE for the network we are updating
		
		# get target Q for bellman loss
		with torch.no_grad(): # uses target actor for next action
			if self.reward_type == 'terminal': # using monte carlo returns
				# is 1 where mortality is 1, else 0. 
				target_Q = reward
			else:
				target_Q = self.target_q(action, next_state, reward, not_done, noise=True)

		# Get current Q estimates - for bellman loss and cql loss
		# current_Q1, current_Q2 = self.critic(state, action)
		current_Q1, current_Q2 = self.critic(state, action)

		# 2. CQL CONSERVATIVE LOSS - sample random actions to penalize them
		# We are penalize 4 types of actions:
		# 1. random actions created by shuffling the batch actions
		# 2. current actions sampled from the actor
		# 3. next actions sampled from the actor for the next states
		# 4. actions with vasopressor dose sampled from a uniform distribution between 0.01 and 1, 
		# with fluid dose equal to 0. This is because vaso never given without fluid.
		batch_size = state.shape[0]

		random_indices = torch.randint(0, batch_size, (batch_size * self.num_samples,), device=self.device)
		random_actions = action[random_indices].view(batch_size, self.num_samples, -1)

		with torch.no_grad():
			curr_actions, curr_log_probs, _ = self.actor(
				state.repeat_interleave(self.num_samples, dim=0), hard_gate=True
			)
			curr_actions = curr_actions.view(batch_size, self.num_samples, -1)
			curr_log_probs = curr_log_probs.view(batch_size, self.num_samples).clamp(min=-10, max=0)

			# sample next_actions from the actor for the next states
			next_actions, next_log_probs, _ = self.actor(
				next_state.repeat_interleave(self.num_samples, dim=0), hard_gate=True
			)
			next_actions = next_actions.view(batch_size, self.num_samples, -1)
			next_log_probs = next_log_probs.view(batch_size, self.num_samples).clamp(min=-10, max=0)

		# create actions with vasopressor dose sampled from a uniform distribution between 0.01 and 1,
		# with fluid dose equal to 0. This is because vaso never given without fluid.
		zero_fluid_actions = torch.zeros(batch_size, self.num_samples, 1, device=self.device)
		vaso_uniform_actions = torch.FloatTensor(batch_size, self.num_samples, 1).uniform_(0.01, 1.0).to(self.device)
		vaso_only_actions = torch.cat([zero_fluid_actions, vaso_uniform_actions], dim=-1)

		# zero_occupancy_actions
		zero_occupancy_actions = self.sample_zero_occupancy_actions(batch_size, self.num_samples)

		# flatten samples and get q values for each type of action
		def get_q_for_samples(s, a_samples, critic_net):
			# s: [batch, dim], a_samples: [batch, num_samples, act_dim]
			s_rep = s.unsqueeze(1).repeat(1, self.num_samples, 1).view(-1, s.shape[-1])
			a_flat = a_samples.reshape(-1, a_samples.shape[-1])
			q1, q2 = critic_net(s_rep, a_flat)
			return q1.view(batch_size, self.num_samples), q2.view(batch_size, self.num_samples)
		
		q1_rand, q2_rand = get_q_for_samples(state, random_actions, self.critic)
		q1_curr, q2_curr = get_q_for_samples(state, curr_actions, self.critic)
		q1_next, q2_next = get_q_for_samples(next_state, next_actions, self.critic) 
		q1_vaso_only, q2_vaso_only = get_q_for_samples(state, vaso_only_actions, self.critic)
		q1_zero_occupancy, q2_zero_occupancy = get_q_for_samples(state, zero_occupancy_actions, self.critic)

		# Inside the training loop, after calculating q1_rand, q1_curr, etc.
		with torch.no_grad():
			# The 'Gap' is the difference between what the Critic thinks of its 
			# own chosen actions vs. what it thinks of the doctor's actions.
			q_gap1 = q1_curr.mean() - current_Q1.mean()
			q_gap2 = q2_curr.mean() - current_Q2.mean()
		q_gap = (q_gap1 + q_gap2) / 2.0
			
		# combine all the q values for the logsumexp calculation
		cat_q1 = torch.cat([
			q1_rand, 
			q1_curr, 
			q1_next,
			q1_vaso_only,
			q1_zero_occupancy
		], dim=1)
		cat_q2 = torch.cat([
			q2_rand, 
			q2_curr, 
			q2_next,
			q2_vaso_only,
			q2_zero_occupancy
		], dim=1)

		num_samples = cat_q1.shape[1]

		# logsumexp acts as a soft maximum, with a smooth, differentiable contribution from every sample,
		# temperature-scaled to control the "softness" of the maximum. Lower temp = sharper max, higher temp = smoother average.
		# As max sharpens, gradient concentrates on the highest Q values
		scaled_lse1 = torch.logsumexp(cat_q1 / self.temp, dim=1).mean() * self.temp
		scaled_lse2 = torch.logsumexp(cat_q2 / self.temp, dim=1).mean() * self.temp

		# Subtract the temperature-scaled sample density offset because logsumexp is biased upwards by 
		# the number of samples. Important so cql_alpha has consistnt meaning across different batch sizes and numbers of samples.
		cql_offset = self.temp * np.log(num_samples)
		cql_loss1 = (scaled_lse1 - cql_offset) - current_Q1.mean()
		cql_loss2 = (scaled_lse2 - cql_offset) - current_Q2.mean()

		# combined critic loss from Bellman and CQL
		bellman_loss = (F.smooth_l1_loss(current_Q1, target_Q) + F.smooth_l1_loss(current_Q2, target_Q)) / 2.0
		cql_loss = cql_loss1 + cql_loss2	
		current_alpha = self.cql_alpha  # Use the fixed cql_alpha value for now
		critic_loss = bellman_loss + current_alpha * cql_loss			
					
		losses = {
			"bellman_loss": bellman_loss.item(),
			"cql_loss": cql_loss.item(),
			"current_alpha": current_alpha,
			"critic_loss": critic_loss.item(), 
			"q_gap": q_gap.item(),
			"q_rand": np.mean([q1_rand.mean().item(), q2_rand.mean().item()]),
			"q_curr": np.mean([q1_curr.mean().item(), q2_curr.mean().item()]),
			"q_next": np.mean([q1_next.mean().item(), q2_next.mean().item()]),
			"q_vaso_only": np.mean([q1_vaso_only.mean().item(), q2_vaso_only.mean().item()]),
			"q_zero_occupancy": np.mean([q1_zero_occupancy.mean().item(), q2_zero_occupancy.mean().item()]),
		}
				
		# OPTIMIZE CRITIC
		self.critic_optimizer.zero_grad()
		critic_loss.backward()

		# capture gradient norms
		total_critic_grad_norm = self._get_grad_norm(self.critic)
		grad_norm = {"critic_grad_norm": total_critic_grad_norm}

		# gradient clipping
		if self.grad_clip is not None:
			# determine whether to clip gradients based on grad_clip value
			grad_norm['critic_was_clipped'] = 1.0 if (self.grad_clip and total_critic_grad_norm > self.grad_clip) else 0.0
			
			# update the moving average of the clipping binary
			self.critic_clip_fraction_ema = self.ema_alpha * self.critic_clip_fraction_ema + (1 - self.ema_alpha) * grad_norm['critic_was_clipped']
			grad_norm['critic_clip_fraction_ema'] = self.critic_clip_fraction_ema
			
			# Gradient clipping if specified		
			torch.nn.utils.clip_grad_norm_(
				self.critic.parameters(), 
				self.grad_clip
			)

		self.critic_optimizer.step()
		if self.critic_scheduler is not None:
			self.critic_scheduler.step()
		# log_dict.update(lr_critic=self.critic_optimizer.param_groups[0]['lr'])
		losses.update(lr_critic=self.critic_optimizer.param_groups[0]['lr'])

		# update critic target networks with soft update
		with torch.no_grad():
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.mul_(1.0 - self.tau)
				target_param.add_(param.data * self.tau)


		##################### ACTOR TRAINING #####################
		# Delayed policy updates, as actor is chasing a moving target (i.e. the critic). 
		if self.total_it % self.policy_freq != 0:
			self.actor.eval() # Dropout INACTIVE for the network we are evaluating
			with torch.no_grad():
				pi, _, _ = self.actor(state, deterministic=True)
				Q1_pi, Q2_pi = self.critic(state, pi)
			Q_pi = torch.min(Q1_pi, Q2_pi)
			q_stats = get_q_stats(current_Q1, current_Q2, Q_pi, mortality)			
		
		else:
			# Compute actor loss
			self.actor.train() # Dropout ACTIVE for the network we are updating
			
			# Sample actions from the stochastic actor
			pi, log_prob, (gate_lp, mag_lp), (f_p, v_p, f_raw_mean, v_raw_mean, f_std, v_std), components = self.actor(
				state, deterministic=False, return_dist=True, sigmoid_correction=False,
				hard_gate=True, return_components=True
			)
			Q1_pi, Q2_pi = self.critic(state, pi)
			Q_pi = torch.min(Q1_pi, Q2_pi)

			q_stats = get_q_stats(current_Q1, current_Q2, Q_pi, mortality)
					
			# Standard SAC-style or TD3-style actor loss
			# Note: We usually keep a small BC component for safety in medical RL
			# but the lmbda logic is usually replaced by pure Q maximization in CQL
			with torch.no_grad():
				current_Q = torch.min(current_Q1, current_Q2)
			
			# 1. Calculate raw advantage
			raw_advantage = Q_pi - current_Q

			# 2. Calculate exponential weights using only the positive advantage landscape
			# We filter the std calculation so the denominator isn't warped by huge negative outliers
			positive_mask = raw_advantage > 0
			if positive_mask.any():
				beta = raw_advantage[positive_mask].std() * 2.0 + 1e-6
			else:
				beta = 1e-6

			actor_loss_q = -Q_pi.mean() # we can also try just maximizing Q_pi directly without the weights, since the weights are mostly for interpretability and analysis of the advantage landscape
		
			actor_loss_gate_ent = gate_lp.mean()

			# Smooth weight: 1 at f_p=0.5, decaying smoothly toward (but never exactly
			# reaching) 0 as f_p -> 0 or 1. Avoids a hard cutoff that creates an
			# artificial population boundary (spike-plus-decay) at whatever threshold
			# is chosen.
			f_gate_uncertainty_weight = 4 * f_p * (1 - f_p)  # parabola: 1 at p=0.5, 0 at p=0 or p=1
			f_mean_saturation_penalty = (f_gate_uncertainty_weight * f_raw_mean.pow(2)).mean()
			
			# For vaso, we use a more traditional approach with a tapering function
			v_gate_uncertainty_weight = (0.5 - (v_p - 0.5).abs()).clamp(min=0)
			v_mean_saturation_penalty = (v_gate_uncertainty_weight * v_raw_mean.pow(2)).mean()

			# Floor on the vaso magnitude mean (NOT gate-weighted). The gate-weighted
			# penalty above vanishes when the vaso gate is near-closed (p ~ 0.02), and the
			# magnitude head only gets Q-gradient on fired samples, so v_raw_mean can
			# drift to ~-12 where sigmoid'(x) ~ 3e-6 and neither head can recover.
			# One-sided hinge: zero above the floor, so it doesn't bias doses in the
			# normal range. None -> disabled (reproduces previous runs exactly).
			if self.vaso_mean_floor is not None:
				v_mean_floor_penalty = torch.relu(self.vaso_mean_floor - v_raw_mean).pow(2).mean()
			else:
				v_mean_floor_penalty = torch.zeros((), device=v_raw_mean.device)

			# Two-sided bound on the vaso GATE logit. With actor_loss_gate_entropy_scale = 0
			# nothing stops gate logits running to extremes (p > 0.99 / < 0.01), where the
			# straight-through gradient ~ p(1-p) vanishes and the gate stops following the
			# critic. Zero inside [-bound, bound] (no bias toward 0.5); only resists leaving it.
			# None -> disabled (reproduces previous runs exactly).
			v_gate_logit_flat = components["v_gate_logit"].squeeze(-1)
			if self.vaso_gate_logit_bound is not None:
				v_gate_bound_penalty = torch.relu(v_gate_logit_flat.abs() - self.vaso_gate_logit_bound).pow(2).mean()
			else:
				v_gate_bound_penalty = torch.zeros((), device=v_gate_logit_flat.device)

			# --- per-drug magnitude entropy, masked mean over FIRED samples only ---
			# (mag_lp is exactly 0 for non-fired samples via the fire-mask multiply,
			# so a plain batch mean would dilute proportionally to firing rate --
			# masking correctly reflects the actual conditional Gaussian's entropy.)
			eps = 1e-6
			f_fire_flat = components["f_fire"].squeeze(-1)
			v_fire_flat = components["v_fire"].squeeze(-1)
			f_mag_lp_flat = components["f_mag_lp"].squeeze(-1)
			v_mag_lp_flat = components["v_mag_lp"].squeeze(-1)

			f_mag_lp_mean = (f_fire_flat * f_mag_lp_flat).sum() / (f_fire_flat.sum() + eps)
			v_mag_lp_mean = (v_fire_flat * v_mag_lp_flat).sum() / (v_fire_flat.sum() + eps)

			alpha_mag_fluid = self.log_alpha_mag_fluid.exp().detach()
			alpha_mag_vaso = self.log_alpha_mag_vaso.exp().detach()

			actor_loss = (
				self.actor_loss_q_scale * actor_loss_q +
				self.actor_loss_gate_entropy_scale * actor_loss_gate_ent +
				alpha_mag_fluid * f_mag_lp_mean +
				alpha_mag_vaso * v_mag_lp_mean +
				self.fluid_mean_reg_scale * f_mean_saturation_penalty +
				self.vaso_mean_reg_scale * v_mean_saturation_penalty + 
				self.vaso_mean_floor_scale * v_mean_floor_penalty +
				self.vaso_gate_logit_bound_scale * v_gate_bound_penalty
			)


			# every 500 iterations, logg actor_loss_q, actor_loss_bc, and actor_loss_ent separately to see how they are contributing to the total loss
			if self.total_it % 500 == 0:
				logging.info(
					f"Iteration {self.total_it}: actor_loss_q={actor_loss_q.item():.4f}, \
					actor_loss_gate_ent={actor_loss_gate_ent.item():.4f}, \
					raw_advantage_mean={raw_advantage.mean().item():.4f}")		 

									
			# Final combined update
			self.actor_optimizer.zero_grad()
			actor_loss.backward()

			# --- update the learned entropy temperatures toward their target entropy ---
			alpha_mag_fluid_loss = -(self.log_alpha_mag_fluid * (f_mag_lp_mean.detach() + self.target_entropy_mag_fluid))
			alpha_mag_vaso_loss = -(self.log_alpha_mag_vaso * (v_mag_lp_mean.detach() + self.target_entropy_mag_vaso))
			alpha_mag_loss = alpha_mag_fluid_loss + alpha_mag_vaso_loss

			self.alpha_mag_optimizer.zero_grad()
			alpha_mag_loss.backward()
			self.alpha_mag_optimizer.step()

			losses.update({
				"actor_loss_total": actor_loss.item(),
				"actor_loss_q": actor_loss_q.item(),
				# "actor_loss_bc": actor_loss_bc.item(),
				"fluid_fire_prob_mean": f_p.mean().item(),
				"fluid_hard_fire_rate": (f_p > 0.5).float().mean().item(),
				"vaso_fire_prob_mean": v_p.mean().item(),
				"vaso_hard_fire_rate": (v_p > 0.5).float().mean().item(),
				# "fluid_volume_loss": actor_dict['fluid_volume_loss'].item(),
				# "vaso_volume_loss": actor_dict['vaso_volume_loss'].item(),
				"fluid_mean": pi[:, 0].mean().item(),
				"fluid_non_zero_mean": pi[pi[:, 0] > 0, 0].mean().item() if (pi[:, 0] > 0).any() else float('nan'),
				"vaso_mean": pi[:, 1].mean().item(),
				"vaso_non_zero_mean": pi[pi[:, 1] > 0, 1].mean().item() if (pi[:, 1] > 0).any() else float('nan'),
				"fluid_std": f_std.mean().item(),
				"vaso_std": v_std.mean().item(),
				"fluid_std_std": f_std.std().item(),
				"vaso_std_std": v_std.std().item(),
				"actor_raw_advantage_mean": raw_advantage.mean().item(),
				"actor_loss_gate_ent": actor_loss_gate_ent.item(),
				"gate_temperature": current_gate_temp,
				"f_mean_saturation_penalty": f_mean_saturation_penalty.item(),
				"v_mean_saturation_penalty": v_mean_saturation_penalty.item(),
				"v_mean_floor_penalty": v_mean_floor_penalty.item(),
				"v_gate_bound_penalty": v_gate_bound_penalty.item(),
				"v_gate_logit_abs_p95": v_gate_logit_flat.detach().abs().quantile(0.95).item(),
				"v_gate_beyond_bound_frac": ((v_gate_logit_flat.detach().abs() > self.vaso_gate_logit_bound).float().mean().item()
											 if self.vaso_gate_logit_bound is not None else float('nan')),
				"v_raw_mean_median": v_raw_mean.median().item(),				
				"alpha_mag_fluid": alpha_mag_fluid.item(),
				"alpha_mag_vaso": alpha_mag_vaso.item(),
				"f_mag_lp_mean": f_mag_lp_mean.item(),
				"v_mag_lp_mean": v_mag_lp_mean.item(),
				"alpha_mag_loss": alpha_mag_loss.item(),
			})

			# clip actor gradients if specified
			total_actor_grad_norm = self._get_grad_norm(self.actor)
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

			# with torch.no_grad():
			# 	self.actor.fluid_log_std.clamp_(self.actor.log_std_min, self.actor.log_std_max)
			# 	self.actor.vaso_log_std.clamp_(self.actor.log_std_min, self.actor.log_std_max)

			
			with torch.no_grad():
				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
					target_param.mul_(1.0 - self.tau)
					target_param.add_(param.data * self.tau)
						
		# return log_dict, grad_norm
		return losses, q_stats, grad_norm
	

	def actor_forward_for_sarsa(self, batch: TensorBatch) -> torch.Tensor:
		state, _, _, _, _, _, _ = TrainerUtils._unpack_batch(batch, self.device)

		self.actor.eval()  # Set actor to eval mode to disable Dropout
		with torch.no_grad():
			new_actions, _, _ = self.actor(state, deterministic=True)  # Get the "best" action without noise or log probabilities
		
		return new_actions


	def _get_grad_norm(self, model: nn.Module) -> float:
		total_norm = 0
		for p in model.parameters():
			if p.grad is not None:
				param_norm = p.grad.detach().data.norm(2)
				total_norm += param_norm.item() ** 2
		return total_norm ** 0.5
	

	@torch.no_grad()
	def estimate_v0(self, start_states: torch.Tensor, batch_size: int = 256) -> float:
		"""
		Estimate the value of the initial state under the current policy.
		NEED TO CONFIRM THIS IS GETTING START STATES
		"""
		self.actor.eval(); 
		self.critic.eval()
		start_states = torch.as_tensor(start_states, dtype=torch.float32, device=self.device)
		q_values = []
		for i in range(0, len(start_states), batch_size):
			batch_states = start_states[i:i+batch_size]
			actions, _, _ = self.actor(batch_states, deterministic=True)
			q1, q2 = self.critic(batch_states, actions)
			q_batch = torch.min(q1, q2)
			q_values.append(q_batch.detach().cpu().numpy())

		q_values = np.concatenate(q_values, axis=0)
		v0_mean = q_values.mean()
		v0_sem = q_values.std() / np.sqrt(len(q_values))

		return v0_mean, v0_sem

	
	@torch.no_grad()
	def _stochastic_vs_deterministic_q_check(self, state, q_stoch=None, q_det=None):
		if q_stoch is None:
			pi_stoch, _, _ = self.actor(state, deterministic=False, hard_gate=True)
			q_stoch = torch.min(*self.critic(state, pi_stoch))
		if q_det is None:
			pi_det, _, _ = self.actor(state, deterministic=True)
			q_det = torch.min(*self.critic(state, pi_det))

		return {
			"q_stoch_mean": q_stoch.mean().item(),
			"q_det_mean": q_det.mean().item(),
			"q_stoch_minus_det": (q_stoch - q_det).mean().item(),
		}


	@torch.no_grad()
	def _gate_incentive_diagnostic(self, state, action, tiny_dose_fluid=0.0005, tiny_dose_vaso=0.001):
		"""
		Compares Q(action with dose forced to exactly 0) vs Q(action with dose
		forced to a tiny nonzero value), holding the OTHER dose fixed at its
		real observed value. Computed only on states where the doctor's actual
		action for that dose was ~0, since that's the population where a
		spurious "always fire a little" incentive would show up.

		Positive gap (Q_zero - Q_tiny > 0) = critic correctly prefers true zero.
		Negative gap = critic prefers a tiny dose over zero — the incentive
		that would push the actor's gate toward saturation.
		"""
		results = {}

		tiny_dose = {0: tiny_dose_fluid, 1: tiny_dose_vaso}

		for dose_idx, name in [(0, "fluid"), (1, "vaso")]:
			near_zero_mask = action[:, dose_idx] < self.actor.zero_threshold
			if near_zero_mask.sum() < 2:
				results[f"{name}_gate_incentive_gap"] = float('nan')
				results[f"{name}_gate_incentive_n"] = 0
				continue

			s = state[near_zero_mask]
			a_real = action[near_zero_mask]

			a_zero = a_real.clone()
			a_zero[:, dose_idx] = 0.0

			a_tiny = a_real.clone()
			a_tiny[:, dose_idx] = tiny_dose[dose_idx]

			q1_zero, q2_zero = self.critic(s, a_zero)
			q1_tiny, q2_tiny = self.critic(s, a_tiny)

			q_zero = torch.min(q1_zero, q2_zero)
			q_tiny = torch.min(q1_tiny, q2_tiny)

			gap = (q_zero - q_tiny).mean().item()

			results[f"{name}_gate_incentive_gap"] = gap
			results[f"{name}_gate_incentive_n"] = int(near_zero_mask.sum().item())

		return results


	def evaluate(self, batch: TensorBatch) -> Dict[str, float]:
		self.actor.eval()  # This is correct and important!
		self.critic.eval()  # Ensure critic is also in eval mode
		
		state, action, reward, next_state, next_action, done, mortality = TrainerUtils._unpack_batch(batch, self.device)

		reward = reward * self.reward_scaling

		# compute the target Q value
		if self.reward_type != 'terminal': # using monte carlo returns
			treat_survival_as_truncated = True
			if treat_survival_as_truncated:
					# make done 0 where mortality is 0, since we want to treat survival as truncated rather than terminal
				done = done * mortality			
			not_done = 1 - done

		with torch.no_grad():
			# compute the target Q value
			if self.reward_type == 'terminal': # using monte carlo returns
				# is 1 where mortality is 1, else 0. 
				target_Q = reward
			else:
				target_Q = self.target_q(action, next_state, reward, not_done, noise=False)

			# Get current Q estimates
			current_Q1, current_Q2 = self.critic(state, action) # critic value of action taken by doctor
			bellman_loss = (F.smooth_l1_loss(current_Q1, target_Q) + F.smooth_l1_loss(current_Q2, target_Q)) / 2.0

			losses = {
				"bellman_loss": bellman_loss.item(),
			}

			gate_incentive = self._gate_incentive_diagnostic(state, action)
			losses.update(gate_incentive)

			# 3. Actor Actions and their Critic Values
            # We use deterministic=True to see the "best" action the model recommends
			pi, _, _, (f_p, v_p, f_raw_mean, v_raw_mean, f_std, v_std) = self.actor(
				state, deterministic=True, return_dist=True
			)
			Q_pi_1, Q_pi_2 = self.critic(state, pi)
			Q_pi = torch.min(Q_pi_1, Q_pi_2)
					
			current_Q = torch.min(current_Q1, current_Q2)
			
			raw_advantage = Q_pi - current_Q	

			# get stochastic pi for comparison
			pi_stoch, _, _ = self.actor(state, deterministic=False, hard_gate=True)
			q_stoch = torch.min(*self.critic(state, pi_stoch))
			stoch_det_q_stats = self._stochastic_vs_deterministic_q_check(state, q_stoch=q_stoch, q_det=Q_pi)
			losses.update(stoch_det_q_stats)

			# 2. Calculate exponential weights using only the positive advantage landscape
			# We filter the std calculation so the denominator isn't warped by huge negative outliers
			positive_mask = raw_advantage > 0
			if positive_mask.any():
				beta = raw_advantage[positive_mask].std() * 2.0 + 1e-6
			else:
				beta = 1e-6

			actor_loss_q = -Q_pi.mean() # we can also try just maximizing Q_pi directly without the weights, since the weights are mostly for interpretability and analysis of the advantage landscape

				
			# ADD THIS LINE:
			neuron_stats = self.critic.get_neuron_stats(state, action)
	
			# get Q statistics
			q_stats = get_q_stats(current_Q1, current_Q2, Q_pi, mortality)

			losses.update({
				"actor_loss_q": actor_loss_q.item(),
				"actor_raw_advantage_mean": raw_advantage.mean().item(),
				# "actor_loss_bc": actor_loss_bc.item(), 
				"fluid_fire_prob_mean": f_p.mean().item(),
				"fluid_hard_fire_rate": (f_p > 0.5).float().mean().item(),
				"f_p_deviation_mean": (f_p - 0.5).abs().mean().item(),
    			"vaso_fire_prob_mean": v_p.mean().item(),
				"vaso_hard_fire_rate": (v_p > 0.5).float().mean().item(),
				"v_p_deviation_mean": (v_p - 0.5).abs().mean().item(),
				"fluid_mean": pi[:, 0].mean().item(),
				"fluid_non_zero_mean": pi[pi[:, 0] > 0, 0].mean().item() if (pi[:, 0] > 0).any() else float('nan'),
				"vaso_mean": pi[:, 1].mean().item(),
				"vaso_non_zero_mean": pi[pi[:, 1] > 0, 1].mean().item() if (pi[:, 1] > 0).any() else float('nan'),
			})
		# return log_dict, action, pi
		return losses, q_stats, neuron_stats

	
	def state_dict(self) -> Dict[str, torch.Tensor]:
		return {
			"actor": self.actor.state_dict(),
			"actor_optimizer": self.actor_optimizer.state_dict(),
			"critic": self.critic.state_dict(),
			"critic_optimizer": self.critic_optimizer.state_dict(),
			"actor_target": self.actor_target.state_dict(),
			"critic_target": self.critic_target.state_dict(),
			"log_alpha_mag_fluid": self.log_alpha_mag_fluid.detach().cpu(),
			"log_alpha_mag_vaso": self.log_alpha_mag_vaso.detach().cpu(),
			"alpha_mag_optimizer": self.alpha_mag_optimizer.state_dict(),			
		}


	######################## currently unused ###############################
	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

	######################## currently unused ###############################
	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)


def get_trainer(config_manager, state_dim, action_dim, hyperparams):

	learning_rate_critic = float(hyperparams['learning_rate_critic'])
	learning_rate_actor = float(hyperparams['learning_rate_actor'])
	weight_decay = float(hyperparams.get('weight_decay', 0.01)) # New hyperparam
	device = config_manager.training_config['device']
	
	actor = Actor(
		state_dim=state_dim, 
		hidden_dim=hyperparams['actor_hidden_dim'], 
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
	
	critic = Critic(
		state_dim, 
		action_dim,
		hidden_dim=hyperparams['critic_hidden_dim'],
		dropout=hyperparams['dropout']
	).to(config_manager.training_config['device'])

	critic_optimizer = torch.optim.AdamW(
		critic.parameters(),
		lr=learning_rate_critic,
		weight_decay=weight_decay
	)

	critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		critic_optimizer, 
		T_max=hyperparams['total_steps'],
		eta_min=learning_rate_critic * .10
	)

	trainer = CQL(               # LIST OF PARAMETERS NEEDS TO BE UPDATED
		actor=actor,
		actor_optimizer=actor_optimizer,
		actor_scheduler=actor_scheduler,
		critic=critic,
		critic_optimizer=critic_optimizer,
		critic_scheduler=critic_scheduler,
		discount=hyperparams['discount'],
		device=config_manager.training_config['device'],
		action_mode=config_manager.action_mode,
		tau=hyperparams['tau'],
		policy_noise=hyperparams['policy_noise'],
		noise_clip=hyperparams['noise_clip'],
		grad_clip=hyperparams['grad_clip'],
		policy_freq=hyperparams['policy_freq'],

		cql_alpha=hyperparams['cql_alpha'],
		temp=hyperparams['temp'],

		# use_smoothing=hyperparams['use_smoothing'],

		# actor_loss_bc_scale=hyperparams['actor_loss_bc_scale'],
		actor_loss_q_scale=hyperparams['actor_loss_q_scale'],
		# actor_loss_entropy_scale=hyperparams['actor_loss_entropy_scale'],
		actor_loss_gate_entropy_scale=hyperparams['actor_loss_gate_entropy_scale'],

		target_std_fluid=hyperparams.get('target_std_fluid', 0.6721),
		target_std_vaso=hyperparams.get('target_std_vaso', 0.6342),
		initial_alpha_mag=hyperparams.get('initial_alpha_mag', 5e-4),
		alpha_mag_lr=hyperparams.get('alpha_mag_lr', 1e-3),


		fluid_mean_reg_scale= hyperparams.get('fluid_mean_reg_scale', 0.05),
		vaso_mean_reg_scale= hyperparams.get('vaso_mean_reg_scale', 0.05),

		vaso_mean_floor=hyperparams.get('vaso_mean_floor', None),
		vaso_mean_floor_scale=hyperparams.get('vaso_mean_floor_scale', 0.05),

		vaso_gate_logit_bound=hyperparams.get('vaso_gate_logit_bound', None),
		vaso_gate_logit_bound_scale=hyperparams.get('vaso_gate_logit_bound_scale', 0.05),

		num_samples=hyperparams['num_samples'],

		reward_type=hyperparams['reward_type'],

		ratio_of_fluid_to_vaso=config_manager.training_config['ratio_of_fluid_to_vaso'],

		gate_temp_start=hyperparams['gate_temp_start'],
		gate_temp_end=hyperparams['gate_temp_end'],
		total_steps=hyperparams['total_steps']
	)

	return trainer



def load_trained_actor(state_dim, action_dim, config_manager):
	"""
	Loads a trained CQL actor for use as another model's fixed target
	policy (e.g. FQE's target_policy_actor). Called generically by
	training.py's load_actor() when config_manager.actor_type ==
	"cql_with_logprobs".

	Path/run resolution is driven directly by config_manager -- NOT by
	any hyperparams dict entries:

	  config_manager.search_variant == "random_search":
	      Used while tuning FQE's OWN hyperparameters. This does NOT
	      simply take the best row of the (LR/hidden_dim/etc.) CQL
	      hyperparameter search's own sorted_runs csv, because that
	      search was run BEFORE the final cql_alpha value was chosen
	      (via the separate alpha-sweep/knee-detection analysis) -- so
	      its "best" row need not use the alpha you actually settled on.
	      Instead:
	        1. Read the FINAL cql_alpha from the pickled config_manager
	           found inside the "fixed_alpha" folder (this is the
	           config_manager that WAS run with the final, chosen alpha).
	        2. Search "cql_alpha_sweep" (the sibling folder holding every
	           individually-tested alpha value from that sweep) for the
	           run whose results.json hyperparameters record that same
	           cql_alpha, and load ITS actor.

	  config_manager.search_variant == "best_hyperparameters":
	      The final, fixed-alpha, multi-seed run stage. Looks inside
	      "fixed_alpha" for the run whose saved checkpoint's
	      training_config['seed'] matches
	      config_manager.training_config['seed'], and loads that run's
	      actor -- pairing each FQE seed with the CQL seed of the same
	      number automatically.

	Both "fixed_alpha" and "cql_alpha_sweep" are subfolders of
	"health_score_change_only", itself a subfolder of the same
	cohort/unit/reward_cohort base path used elsewhere in this project
	(see get_evaluator in basic_q.py for the equivalent basic_q pattern).

	NOTE -- please verify against your actual directory layout/checkpoint
	contents before relying on this:
	  - assumes each run's checkpoint (final_model.pt) was saved via
	    train_with_hyperparameters's 'cql_with_logprobs' branch, i.e.
	    contains 'actor_state_dict' and 'hyperparameters' (with
	    'actor_hidden_dim' and, for cql_alpha_sweep runs, 'cql_alpha')
	  - assumes results.json's hyperparameters dict also records
	    'cql_alpha' (falls back to a top-level 'cql_alpha' key if not
	    nested, but verify which your save_results() actually produces)
	  - assumes exactly one config_manager.pkl exists somewhere under
	    "fixed_alpha" (searched recursively, first match used) recording
	    the final chosen cql_alpha as a `.cql_alpha` attribute
	"""
	device = torch.device(config_manager.training_config.get('device', "cuda" if torch.cuda.is_available() else "cpu"))

	cohort = config_manager.cohort
	unit = config_manager.unit
	reward_cohort = config_manager.reward_cohort

	base_path = os.path.join(config_manager.paths['rl_models_dir'],
		'cql_with_logprobs', f'split_{config_manager.split_num}', cohort,
		config_manager.actions)

	if unit.lower() == 'picu':
		if reward_cohort.lower() == 'picu':
			base_path = os.path.join(base_path,
			"picu_patients_health_score_from_picu_dataset")
		elif reward_cohort.lower() == 'restricted':
			base_path = os.path.join(base_path,
			"picu_patients_health_score_from_restricted_dataset")
		else:
			raise ValueError(f"Unknown reward_cohort: {reward_cohort}. Must be 'picu' or 'restricted'.")
	else:
		raise ValueError(f"Unknown unit: {unit}. Must be 'picu'.")

	health_score_change_only_path = os.path.join(base_path, "health_score_change_only")
	fixed_alpha_path = os.path.join(health_score_change_only_path, "fixed_alpha")

	def _load_actor_from_checkpoint(checkpoint):
		hidden_dim = checkpoint['hyperparameters']['actor_hidden_dim']
		actor = Actor(state_dim=state_dim, hidden_dim=hidden_dim).to(device)
		actor.load_state_dict(checkpoint['actor_state_dict'])
		actor.eval()
		for p in actor.parameters():
			p.requires_grad_(False)
		return actor

	if config_manager.search_variant == "random_search":
		# 1. find the final chosen cql_alpha, from fixed_alpha's own config_manager.pkl
		if not os.path.isdir(fixed_alpha_path):
			raise FileNotFoundError(f"fixed_alpha directory not found: {fixed_alpha_path}")

		config_pkl_path = None
		for root, dirs, files in os.walk(fixed_alpha_path):
			if "config_manager.pkl" in files:
				config_pkl_path = os.path.join(root, "config_manager.pkl")
				break
		if config_pkl_path is None:
			raise FileNotFoundError(f"No config_manager.pkl found anywhere under {fixed_alpha_path}")

		with open(config_pkl_path, "rb") as f:
			fixed_alpha_config_manager = pickle.load(f)
		target_cql_alpha = fixed_alpha_config_manager.cql_alpha
		print(f"[load_trained_actor] Final cql_alpha (from {config_pkl_path}): {target_cql_alpha}")

		# 2. search cql_alpha_sweep for the run matching that alpha
		sweep_path = os.path.join(health_score_change_only_path, "cql_alpha_sweep")
		if not os.path.isdir(sweep_path):
			raise FileNotFoundError(f"cql_alpha_sweep directory not found: {sweep_path}")

		matched_run_path = None
		for run_folder in sorted(os.listdir(sweep_path)):
			run_path = os.path.join(sweep_path, run_folder)
			for inner_run_folder in sorted(os.listdir(run_path)):
				inner_run_path = os.path.join(run_path, inner_run_folder)
				results_path = os.path.join(inner_run_path, "results.json")
				if not os.path.isfile(results_path):
					continue
				with open(results_path, "r") as f:
					results = json.load(f)
				run_cql_alpha = results.get("hyperparameters", {}).get("cql_alpha", results.get("cql_alpha"))
				if run_cql_alpha is not None and math.isclose(run_cql_alpha, target_cql_alpha, rel_tol=1e-6, abs_tol=1e-9):
					matched_run_path = inner_run_path
					break

		if matched_run_path is None:
			raise FileNotFoundError(
				f"No run under {sweep_path} matched cql_alpha={target_cql_alpha} "
				f"(checked results.json in each subfolder)."
			)
		print(f"[load_trained_actor] Matched cql_alpha_sweep run: {matched_run_path}")

		final_model_path = os.path.join(matched_run_path, "final_model.pt")
		checkpoint = torch.load(final_model_path, map_location=device, weights_only=False)
		return _load_actor_from_checkpoint(checkpoint)

	elif config_manager.search_variant == "best_hyperparameters":
		target_seed = config_manager.training_config['seed']
	
		if not os.path.isdir(fixed_alpha_path):
			raise FileNotFoundError(f"fixed_alpha directory not found: {fixed_alpha_path}")

		for run_folder in sorted(os.listdir(fixed_alpha_path)):
			run_path = os.path.join(fixed_alpha_path, run_folder)
			for inner_run_folder in sorted(os.listdir(run_path)):
				inner_run_path = os.path.join(run_path, inner_run_folder)
				# load results.json to check the seed
				results_path = os.path.join(inner_run_path, "results.json")
				if not os.path.isfile(results_path):
					continue
				with open(results_path, "r") as f:
					results = json.load(f)
				run_seed = results.get("hyperparameters", {}).get("seed")
				if run_seed == target_seed:
					final_model_path = os.path.join(inner_run_path, "final_model.pt")
					checkpoint = torch.load(final_model_path, map_location=device, weights_only=False)
					return _load_actor_from_checkpoint(checkpoint)

		raise FileNotFoundError(
			f"No run with training_config['seed']=={target_seed} found under {fixed_alpha_path}."
		)

	else:
		raise ValueError(
			f"Unknown config_manager.search_variant='{config_manager.search_variant}', "
			f"expected 'random_search' or 'best_hyperparameters'"
		)