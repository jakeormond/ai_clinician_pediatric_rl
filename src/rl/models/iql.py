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
		log_std_min: float = -3,
		log_std_max: float = 0.,
		action_dim: int = 2,
		dropout: float = 0.1
	):
		super().__init__()
		self.log_std_min = log_std_min
		self.log_std_max = log_std_max

		# Shared trunk — learns a joint patient representation
		self.trunk = nn.Sequential(
			nn.Linear(state_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),
			# nn.Linear(hidden_dim, hidden_dim),
			# nn.LayerNorm(hidden_dim),
			# nn.ReLU(),
		)

		# Separate heads — each action has its own mean/log_std stream
		# This prevents the high-gradient fluid head from swamping vasopressors
		self.fluid_mean    = nn.Linear(hidden_dim, 1)
		# self.fluid_log_std = nn.Linear(hidden_dim, 1)

		self.vaso_mean    = nn.Linear(hidden_dim, 1)
		# self.vaso_log_std = nn.Linear(hidden_dim, 1)

		# log_std as learned scalar parameters — NOT network outputs
		# Gradient from Q only reaches the mean heads, not these
		self.fluid_log_std = nn.Parameter(torch.tensor([-1.0]))
		self.vaso_log_std  = nn.Parameter(torch.tensor([-2.0]))

		self._init_weights()


	def _init_weights(self):
		for m in self.trunk:
			if isinstance(m, nn.Linear):
				nn.init.orthogonal_(m.weight, gain=1.0)
				nn.init.zeros_(m.bias)

		# Fluid head — initialise near 0.4 (your observed mean)
		# sigmoid(0.4) ≈ 0.6, so bias = logit(0.4) ≈ -0.405
		nn.init.zeros_(self.fluid_mean.weight)
		self.fluid_mean.bias.data.fill_(-0.405)

		# Vasopressor head — initialise near 0.02 (sparse baseline)
		# logit(0.02) ≈ -3.89; pull mean toward near-zero actions
		nn.init.zeros_(self.vaso_mean.weight)
		self.vaso_mean.bias.data.fill_(-3.89)

		# for layer in [self.fluid_log_std, self.vaso_log_std]:
		# 	nn.init.zeros_(layer.weight)
		# 	layer.bias.data.fill_(-1.0)  # moderate initial std


	def forward(self, state: torch.Tensor, deterministic: bool = False):
		z = self.trunk(state)

		# --- Fluid ---
		f_mean = torch.sigmoid(self.fluid_mean(z))          # (0, 1)
		# f_log_std = torch.clamp(
		# 	self.fluid_log_std(z), self.log_std_min, self.log_std_max
		# )
		f_log_std = torch.clamp(self.fluid_log_std, self.log_std_min, self.log_std_max)


		# --- Vasopressor ---
		v_mean = torch.sigmoid(self.vaso_mean(z))            # (0, 1)
		v_log_std = torch.clamp(
			self.vaso_log_std, self.log_std_min, self.log_std_max
		)


		f_std = f_log_std.exp()
		v_std = v_log_std.exp()
	
		if not deterministic:
			# Sample in unconstrained space, then squash with sigmoid
			f_dist = torch.distributions.Normal(
				torch.logit(f_mean.clamp(1e-6, 1 - 1e-6)), f_std
			)
			v_dist = torch.distributions.Normal(
				torch.logit(v_mean.clamp(1e-6, 1 - 1e-6)), v_std
			)

			f_raw = f_dist.rsample()
			v_raw = v_dist.rsample()

			f_action = torch.sigmoid(f_raw)
			v_action = torch.sigmoid(v_raw)

			# Log-prob with sigmoid change-of-variables correction
			# log p(a) = log p(u) - sum log sigmoid'(u)
			f_log_prob = f_dist.log_prob(f_raw) + torch.log(
				f_action * (1 - f_action) + 1e-6
			)
			v_log_prob = v_dist.log_prob(v_raw) + torch.log(
				v_action * (1 - v_action) + 1e-6
			)

			log_prob    = (f_log_prob + v_log_prob).squeeze(-1)
			actions     = torch.cat([f_action, v_action], dim=-1)
			# mean_actions = torch.cat([f_mean, v_mean], dim=-1)
		else:
			actions = torch.cat([f_mean, v_mean], dim=-1)
			log_prob = None

		return actions, log_prob



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
		
		# 2. Action Processing Pathway (Shared across both critics)
		# self.action_projector = nn.Sequential(
		# 	nn.Linear(action_dim, 8), # 32
		# 	nn.LeakyReLU(negative_slope=self.negative_slope)
		# )


		# # 3. State Compression Pathway (Shared or individual, individual is safer for twin critics)
		# # Reduces 768 down to 64 dimensions safely away from the action
		# self.state_projectors = nn.ModuleList([
		# 		nn.Sequential(
		# 		nn.Linear(state_dim, self.hidden_dim),
		# 		nn.LayerNorm(self.hidden_dim),
		# 		nn.LeakyReLU(negative_slope=self.negative_slope),
		# 		nn.Dropout(p=dropout),
		# 		nn.Linear(self.hidden_dim, 64),
		# 		nn.LeakyReLU(negative_slope=self.negative_slope)
		# 	)
		# 	for _ in range(2) # Two separate state projectors for Q1 and Q2
		# ])
		
		# 4. Q-Value Heads (Separate for Q1 and Q2)
		# self.q_heads = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(64 + 32, self.hidden_dim2),
        #         nn.LeakyReLU(negative_slope=self.negative_slope),
        #         nn.Dropout(p=dropout),
                
        #         nn.Linear(self.hidden_dim2, self.hidden_dim3),
        #         nn.LeakyReLU(negative_slope=self.negative_slope),
                
        #         nn.Linear(self.hidden_dim3, 1)
        #     ) for _ in range(2)
        # ])

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
		# # 1. Normalize the massive state latents cleanly
		# norm_state = self.state_norm(state)

		# # 2. Process the action features once
		# h_action = self.action_projector(action)

		# # 3. Pass through the twin pathways dynamically
		# # Twin 1
		# h_state1 = self.state_projectors[0](norm_state)
		# q1 = self.q_heads[0](torch.cat([h_state1, h_action], dim=-1))
		
		# # Twin 2
		# h_state2 = self.state_projectors[1](norm_state)
		# q2 = self.q_heads[1](torch.cat([h_state2, h_action], dim=-1))

		# return q1, q2
	
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

		# Wrap in no_grad to save memory since we're just extracting diagnostics
		# with torch.no_grad():
		# 	# 1. Trace the State Normalization + State Projector 1 (Representative of Twin 1)
		# 	norm_state = self.state_norm(state)
		# 	h_state1 = trace_sequential_block(self.state_projectors[0], norm_state, "state_proj")
			
		# 	# 2. Trace the Action Projector
		# 	h_action = trace_sequential_block(self.action_projector, action, "action_proj")
			
		# 	# 3. Concatenate and trace through the Q1 Head
		# 	x_joint = torch.cat([h_state1, h_action], dim=-1)
		# 	_ = trace_sequential_block(self.q_heads[0], x_joint, "q_head")

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
	

class ValueNetwork(nn.Module):
    def __init__(
        self,
        state_dim,
        dropout=0.1,
        hidden_dim=64,
        negative_slope=0.1,
    ):
        super(ValueNetwork, self).__init__()
        self.negative_slope = negative_slope

        self.state_norm = nn.LayerNorm(state_dim)

        # Single network — no twin, since V doesn't need the
        # overestimation-bias machinery that motivates twin Q-critics
        self.v_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(p=dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=negative_slope),

            nn.Linear(hidden_dim // 2, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if m.out_features == 1:
                nn.init.uniform_(m.weight, -1e-2, 1e-2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, -0.2)
            else:
                nn.init.kaiming_normal_(
                    m.weight,
                    a=self.negative_slope,
                    mode='fan_in',
                    nonlinearity='leaky_relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.05)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, state):
        norm_state = self.state_norm(state)
        return self.v_head(norm_state)



def get_q_stats(current_Q1, current_Q2, actor_Q, value, mortality):
	
	# create current_Q as the mean of current_Q1 and current_Q2
	current_Q = (current_Q1 + current_Q2) / 2.0
	mean_Q = current_Q.mean().item()
	variance_Q = current_Q.var().item()
	batch_n = current_Q.shape[0]
	uncertainty_Q = F.mse_loss(current_Q1, current_Q2).item() # simple measure of uncertainty based on disagreement between the two critics

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
		survival_value = value[survival_mask].mean().item()
	else:
		survival_Q = float('nan')
		survival_variance_Q = float('nan')
		survival_n = 0
		survival_value = float('nan')

	if non_survival_n > 2:
		non_survival_vals = current_Q[non_survival_mask]
		non_survival_Q = non_survival_vals.mean().item()
		non_survival_variance_Q = non_survival_vals.var().item()
		non_survival_value = value[non_survival_mask].mean().item()
	else:
		non_survival_Q = float('nan')
		non_survival_variance_Q = float('nan')
		non_survival_n = 0
		non_survival_value = float('nan')

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
		"auc_Q": auc_Q,
		"survival_value": survival_value,
		"non_survival_value": non_survival_value
	}

	# log_dict.update({
	# 	"batch_n": batch_n,
	# 	"mean_Q": mean_Q,
	# 	"variance_Q": variance_Q,
	# 	"uncertainty_Q": uncertainty_Q,
	# 	"survival_n": survival_n,
	# 	"survival_Q": survival_Q,
	# 	"survival_variance_Q": survival_variance_Q,
	# 	"non_survival_n": non_survival_n,
	# 	"non_survival_Q": non_survival_Q,
	# 	"non_survival_variance_Q": non_survival_variance_Q
	# })

	if actor_Q is not None:
		advantage_Q = actor_Q - current_Q # how much better the actor's action is compared to the current Q estimate, on average. This can help us understand if the policy is improving over time.
		advantage_Q_mean = (actor_Q - current_Q).mean().item() # how much better the actor's action is compared to the current Q estimate, on average. This can help us understand if the policy is improving over time.
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



class IQL:
	def __init__(
		self,
		actor: nn.Module = None,
		actor_optimizer: torch.optim.Optimizer = None,
		actor_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
		
		critic: nn.Module = None,
		critic_optimizer: torch.optim.Optimizer = None,
		critic_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
		
		value_net: nn.Module = None,
		value_net_optimizer: torch.optim.Optimizer = None,
		value_net_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
		
		discount: float = 0.975,
		expectile: float = 0.7,
		awr_temperature: float = 1.0,

		policy_freq: int = 1,

		device: str = "CUDA" if torch.cuda.is_available() else "CPU",
		
		action_mode: str = "continuous",
		dose_actions: int = 2,  #  Number of dose actions
		binary_actions: int = 0,		
		tau: float = 0.005,
		grad_clip: float = None,  # Add grad_clip argument
		ema_alpha: float = 0.99,  # Add ema_alpha argument for moving
		reward_scaling: float = 1.0,
		use_monte_carlo_terminal_returns: bool = False,

		ratio_of_fluid_to_vaso: float = 1.0  # Add ratio_of_fluid_to_vaso argument

	):
		
		self.actor = actor
		self.actor_optimizer = actor_optimizer
		self.actor_scheduler = actor_scheduler
		
		self.critic = critic
		self.critic_optimizer = critic_optimizer
		self.critic_scheduler = critic_scheduler

		self.value_net = value_net
		self.value_net_optimizer = value_net_optimizer
		self.value_net_scheduler = value_net_scheduler

		self.discount = discount
		self.expectile = expectile
		self.awr_temperature = awr_temperature
		
		self.policy_freq = policy_freq

		self.device = device
		self.action_mode = action_mode
		self.dose_actions = dose_actions
		self.binary_actions = binary_actions
		self.tau = tau
		self.grad_clip = grad_clip		

		# for calculating moving average of gradient clipping binary
		self.critic_clip_fraction_ema = 0.0
		self.actor_clip_fraction_ema = 0.0
		self.value_clip_fraction_ema = 0.0
		self.ema_alpha = ema_alpha # Closer to 1.0 means more smoothing
		
		self.reward_scaling = reward_scaling

		self.use_monte_carlo_terminal_returns = use_monte_carlo_terminal_returns

		self.ratio_of_fluid_to_vaso = ratio_of_fluid_to_vaso

		self.total_it = 0

		self.train_actor_flag = False

		self.utils = TrainerUtils()



	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		actions, _ = self.actor(state, deterministic=True)
		return actions.cpu().data.numpy().flatten()


	@staticmethod
	def expectile_loss(diff, expectile=0.7):
		"""
		Asymmetric loss: penalises underestimation (diff > 0) less than
		overestimation (diff < 0) when expectile > 0.5, pushing V(s) toward
		something close to max_a Q(s,a) without ever evaluating out-of-distribution actions.
		"""
		weight = torch.where(diff > 0, expectile, 1 - expectile)
		return weight * diff.pow(2)
	
		
	def compute_actor_loss(self,pi: torch.Tensor, action: torch.Tensor, exp_weights: torch.Tensor) -> torch.Tensor:
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
		}
		
		return actor_dict



	def train(self, batch: TensorBatch) -> Tuple[Dict[str, float], Dict[str, float]]:
		self.total_it += 1

		# Sample replay buffer 
		state, action, reward, next_state, next_action, done, mortality = TrainerUtils._unpack_batch(batch, self.device)
		
		# for troubleshootng, let's make a copy of done
		done_copy = done.clone()
		
		
		# with torch.no_grad():
		# 	# add noise to states
		# 	state_std = state.std(dim=0, keepdim=True)
		# 	noise_scale = state_std * 0.04 + 1e-6
		# 	state_noise = torch.randn_like(state) * noise_scale
		# 	smoothed_states = state + state_noise

		# 	next_state_noise = torch.randn_like(next_state) * noise_scale
		# 	smoothed_next_states = next_state + next_state_noise
			
		# 	# add noise to actions
		# 	critic_action_noise = (torch.randn_like(action) * 0.04).clamp(-0.1, 0.1)
		# 	smoothed_actions = action + critic_action_noise
		# 	is_zero_dose = (action < 1e-4)
		# 	smoothed_actions = torch.where(is_zero_dose, torch.zeros_like(smoothed_actions), smoothed_actions)			
		# 	smoothed_actions = smoothed_actions.clamp(0, 1.0) # Ensure actions remain in valid range after smoothing

		reward = reward * self.reward_scaling
		
		# compute the target Q value
		treat_survival_as_truncated = True
		if treat_survival_as_truncated:
			# make done 0 where mortality is 0, since we want to treat survival as truncated rather than terminal
			done = done * mortality		
		not_done = 1 - done

		# ---- 1. Value network update (expectile regression against dataset action's Q) ----
		self.value_net.train()
		self.critic.eval()  # Set critic to eval mode to disable Dropout
		with torch.no_grad():
			q1, q2 = self.critic(state, action)
			# q1, q2 = self.critic(smoothed_states, smoothed_actions)
			q_min = torch.min(q1, q2)

		v = self.value_net(state)
		# v = self.value_net(smoothed_states)
		v_loss = self.expectile_loss(q_min - v, expectile=self.expectile).mean()
		
		self.value_net_optimizer.zero_grad()
		v_loss.backward()
		value_grad_norm = self._get_grad_norm(self.value_net)
		grad_norm = {"value_grad_norm": value_grad_norm}
		if self.grad_clip is not None:
			grad_norm['value_was_clipped'] = 1.0 if value_grad_norm > self.grad_clip else 0.0
			self.value_clip_fraction_ema = self.ema_alpha * self.value_clip_fraction_ema + (1 - self.ema_alpha) * grad_norm['value_was_clipped']
			grad_norm['value_clip_fraction_ema'] = self.value_clip_fraction_ema
			torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.grad_clip)
		self.value_net_optimizer.step()
		if self.value_net_scheduler is not None:
			self.value_net_scheduler.step()
		
		# ---- 2. Critic update (bootstraps from V(s'), not from a sampled next action) ----	
		self.critic.train()        # Dropout ACTIVE for the network we are updating
		self.value_net.eval()  # Set value net to eval mode to disable Dropout
		with torch.no_grad(): # uses target actor for next action
			if self.use_monte_carlo_terminal_returns:
				# is 1 where mortality is 1, else 0. 
				target_Q = -(mortality - 1).float()
			else:
				
				next_v = self.value_net(next_state)   # no target net needed for V itself
				# next_v = self.value_net(smoothed_next_states)   # no target net needed for V itself
				target_Q = reward + not_done * self.discount * next_v

				# for trouble shooting, let's check V at terminal states
				terminal_survival_mask = (done_copy == 1) & (mortality == 0)
				terminal_non_survival_mask = (done_copy == 1) & (mortality == 1)
				if terminal_survival_mask.any():
					terminal_survival_v = next_v[terminal_survival_mask].mean().item()
				else:
					terminal_survival_v = float('nan')

				if terminal_non_survival_mask.any():
					terminal_non_survival_v = next_v[terminal_non_survival_mask].mean().item()
				else:
					terminal_non_survival_v = float('nan')


		# Get current Q estimates>
		current_Q1, current_Q2 = self.critic(state, action)
		# current_Q1, current_Q2 = self.critic(smoothed_states, smoothed_actions)

		critic_loss = (F.smooth_l1_loss(current_Q1, target_Q) + F.smooth_l1_loss(current_Q2, target_Q)) / 2.0

		losses = {
			"v_loss": v_loss.item(),
			"critic_loss": critic_loss.item(),			
		}

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		total_critic_grad_norm = self._get_grad_norm(self.critic)
		grad_norm["critic_grad_norm"] = total_critic_grad_norm
		if self.grad_clip is not None:
			grad_norm['critic_was_clipped'] = 1.0 if total_critic_grad_norm > self.grad_clip else 0.0
			self.critic_clip_fraction_ema = self.ema_alpha * self.critic_clip_fraction_ema + (1 - self.ema_alpha) * grad_norm['critic_was_clipped']
			grad_norm['critic_clip_fraction_ema'] = self.critic_clip_fraction_ema
			torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
		self.critic_optimizer.step()
		if self.critic_scheduler is not None:
			self.critic_scheduler.step()
		losses["lr_critic"] = self.critic_optimizer.param_groups[0]['lr']

		# ---- 3. Actor update (advantage-weighted regression — delayed, like CQL) ----
		self.critic.eval()  # Set critic to eval mode to disable Dropout
		self.value_net.eval()  # Set value net to eval mode to disable Dropout
		with torch.no_grad():
			q1_eval, q2_eval = self.critic(state, action)
			# q1_eval, q2_eval = self.critic(smoothed_states, smoothed_actions)
			q_min = torch.min(q1_eval, q2_eval)
			v = self.value_net(state)
			# v = self.value_net(smoothed_states)
			advantage_qv = q_min - v

		# Delayed policy updates
		if self.total_it % self.policy_freq != 0 or self.train_actor_flag is False:
			self.actor.eval()  # Set actor to eval mode to disable Dropout
			with torch.no_grad():
				pi, _ = self.actor(state, deterministic=True)
				# pi, _ = self.actor(smoothed_states, deterministic=True)
				Q1_actor, Q2_actor = self.critic(state, pi)
				# Q1_actor, Q2_actor = self.critic(smoothed_states, pi)

			Q_actor = (Q1_actor + Q2_actor) / 2.0 

			q_stats = get_q_stats(current_Q1, current_Q2, Q_actor, v, mortality)
			q_stats["advantage_qv_mean"] = advantage_qv.mean().item()	
			q_stats["terminal_survival_v"] = terminal_survival_v
			q_stats["terminal_non_survival_v"] = terminal_non_survival_v
	
		
		else:
			# Compute actor loss
			self.actor.train() # Dropout ACTIVE for the network we are updating
			# Sample actions from the stochastic actor
			pi, _ = self.actor(state, deterministic=True)
			# pi, _ = self.actor(smoothed_states, deterministic=True)
			
			exp_weight = torch.exp(advantage_qv * self.awr_temperature).clamp(max=100.0).detach()

			# Weighted regression toward the dataset action, weighted by advantage —
			# this REPLACES the Q-maximisation gradient CQL used. The actor never
			# differentiates through the critic in IQL.
			actor_raw_sq_error = (pi - action).pow(2)
			actor_loss = (exp_weight * actor_raw_sq_error).mean()

			losses["actor_loss"] = actor_loss.item()
			losses["actor_raw_sq_error"] = actor_raw_sq_error.mean().item()
			losses["exp_weight_mean"] = exp_weight.mean().item()
			losses["exp_weight_max"] = exp_weight.max().item()
			losses["exp_weight_pct_at_ceiling"] = (exp_weight >= 99.9).float().mean().item()
			
			with torch.no_grad():
				Q1_actor, Q2_actor = self.critic(state, pi)
				# Q1_actor, Q2_actor = self.critic(smoothed_states, pi)
			Q_actor = (Q1_actor + Q2_actor) / 2.0
			
			q_stats = get_q_stats(current_Q1, current_Q2, Q_actor, v, mortality)
			q_stats["advantage_qv_mean"] = advantage_qv.mean().item()
			q_stats["terminal_survival_v"] = terminal_survival_v
			q_stats["terminal_non_survival_v"] = terminal_non_survival_v
			
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			total_actor_grad_norm = self._get_grad_norm(self.actor)
			grad_norm['actor_grad_norm'] = total_actor_grad_norm
			if self.grad_clip is not None:
				grad_norm['actor_was_clipped'] = 1.0 if total_actor_grad_norm > self.grad_clip else 0.0
				self.actor_clip_fraction_ema = self.ema_alpha * self.actor_clip_fraction_ema + (1 - self.ema_alpha) * grad_norm['actor_was_clipped']
				grad_norm['actor_clip_fraction_ema'] = self.actor_clip_fraction_ema
				torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
			self.actor_optimizer.step()
			if self.actor_scheduler is not None:
				self.actor_scheduler.step()
				losses["lr_actor"] = self.actor_optimizer.param_groups[0]['lr']

		return losses, q_stats, grad_norm
			
			

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
			actions, _ = self.actor(batch_states, deterministic=True)
			q1, q2 = self.critic(batch_states, actions)
			q_batch = torch.min(q1, q2)
			q_values.append(q_batch.detach().cpu().numpy())

		q_values = np.concatenate(q_values, axis=0)
		v0_mean = q_values.mean()
		v0_sem = q_values.std() / np.sqrt(len(q_values))

		return v0_mean, v0_sem


	def evaluate(self, batch: TensorBatch) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
		self.actor.eval()
		self.critic.eval()
		self.value_net.eval()

		state, action, reward, next_state, next_action, done, mortality = TrainerUtils._unpack_batch(batch, self.device)
		reward = reward * self.reward_scaling

		treat_survival_as_truncated = True
		if treat_survival_as_truncated:
			done = done * mortality
		not_done = 1 - done

		with torch.no_grad():
			if self.use_monte_carlo_terminal_returns:
				target_Q = -(mortality - 1).float()
			else:
				next_v = self.value_net(next_state)
				target_Q = reward + not_done * self.discount * next_v

			current_Q1, current_Q2 = self.critic(state, action)
			critic_loss = (F.smooth_l1_loss(current_Q1, target_Q) + F.smooth_l1_loss(current_Q2, target_Q)) / 2.0

			v = self.value_net(state)
			q_min = torch.min(current_Q1, current_Q2)
			v_loss = self.expectile_loss(q_min - v, expectile=self.expectile).mean()
			advantage_qv = q_min - v

			pi, _ = self.actor(state, deterministic=True)
			Q1_actor, Q2_actor = self.critic(state, pi)
			Q_actor = torch.min(Q1_actor, Q2_actor)

			# Actor loss on held-out data, computed the same way as train(),
			# but never used to update parameters — purely diagnostic
			exp_weight = torch.exp(advantage_qv * self.awr_temperature).clamp(max=100.0)
			actor_loss = (exp_weight * (pi - action).pow(2)).mean()

			losses = {
				"critic_loss": critic_loss.item(),
				"v_loss": v_loss.item(),
				"actor_loss": actor_loss.item(),
				"exp_weight_mean": exp_weight.mean().item(),
				"exp_weight_max": exp_weight.max().item(),
				"exp_weight_pct_at_ceiling": (exp_weight >= 99.9).float().mean().item()
			}

			neuron_stats = self.critic.get_neuron_stats(state, action)
			q_stats = get_q_stats(current_Q1, current_Q2, Q_actor, v, mortality)
			q_stats["advantage_qv_mean"] = advantage_qv.mean().item()

		return losses, q_stats, neuron_stats

	
	def set_actor_active(self, active: bool):
		self.train_actor_flag = active


	def state_dict(self) -> Dict[str, torch.Tensor]:
		return {
			"actor": self.actor.state_dict(),
			"actor_optimizer": self.actor_optimizer.state_dict(),
			
			"critic": self.critic.state_dict(),
			"critic_optimizer": self.critic_optimizer.state_dict(),
			
			"value_net": self.value_net.state_dict(),
			"value_net_optimizer": self.value_net_optimizer.state_dict(),		
			
		}


	######################## currently unused ###############################
	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

		torch.save(self.value_net.state_dict(), filename + "_value_net")
		torch.save(self.value_net_optimizer.state_dict(), filename + "_value_net_optimizer")

	######################## currently unused ###############################
	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))

		self.value_net.load_state_dict(torch.load(filename + "_value_net"))
		self.value_net_optimizer.load_state_dict(torch.load(filename + "_value_net_optimizer"))




def get_trainer(config_manager, state_dim, action_dim, hyperparams):

	learning_rate_critic = float(hyperparams['learning_rate_critic'])
	learning_rate_actor  = float(hyperparams['learning_rate_actor'])
	learning_rate_value  = float(hyperparams.get('learning_rate_value', learning_rate_critic))
	weight_decay = float(hyperparams.get('weight_decay', 0.01))
	device = config_manager.training_config['device']

	hidden_dim = hyperparams['hidden_dim']
	dropout = hyperparams['dropout']

	value_dropout = hyperparams['value_dropout']
	value_weight_decay = float(hyperparams['value_weight_decay'])

	actor = Actor(
		state_dim=state_dim, 
		action_dim=action_dim, 
		dropout=dropout
	).to(config_manager.training_config['device'])
	
	actor_optimizer = torch.optim.AdamW(
		actor.parameters(), 
		lr=learning_rate_actor,
		weight_decay=weight_decay
		)
	
	# Actor only updates every policy_freq steps, AND only once unfrozen —
	# T_max should reflect the number of ACTOR steps actually expected,
	# i.e. (total_steps - actor_warmup_steps) // policy_freq, not the full
	# total_steps // policy_freq, since the actor never sees any steps
	# before warmup completes
	actor_warmup_steps = 40000 # this is just a guess, since warm up is calculated in training.py 
	# using convergence of v loss and critic loss
	actor_total_steps = max(
		(hyperparams['total_steps'] - actor_warmup_steps) // hyperparams['policy_freq'],
		1
	)
	
	actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		actor_optimizer,
		T_max=actor_total_steps,
		eta_min=learning_rate_actor * 0.10
	)
	

	critic = Critic(
		state_dim, 
		action_dim,
		hidden_dim=hidden_dim,
		dropout=dropout
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


	value_net = ValueNetwork(
		state_dim,
		hidden_dim=hidden_dim,
		dropout=value_dropout,
	).to(device)

	value_net_optimizer = torch.optim.AdamW(
		value_net.parameters(),
		lr=learning_rate_value,
		weight_decay=value_weight_decay
	)

	value_net_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		value_net_optimizer,
		T_max=hyperparams['total_steps'],
		eta_min=learning_rate_value * 0.10
	)

	trainer = IQL(
		actor=actor,
		actor_optimizer=actor_optimizer,
		actor_scheduler=actor_scheduler,
		critic=critic,
		critic_optimizer=critic_optimizer,
		critic_scheduler=critic_scheduler,
		value_net=value_net,
		value_net_optimizer=value_net_optimizer,
		value_net_scheduler=value_net_scheduler,

		discount=hyperparams['discount'],
		expectile=hyperparams['expectile'],
		awr_temperature=hyperparams['awr_temperature'],
		policy_freq=hyperparams['policy_freq'],

		device=device,
		action_mode=config_manager.action_mode,
		tau=hyperparams['tau'],
		grad_clip=hyperparams['grad_clip'],
		reward_scaling=hyperparams.get('reward_scaling', 1.0),
		use_monte_carlo_terminal_returns=hyperparams['monte_carlo_terminal_returns'],

	)


	return trainer

