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

"""
Gemini is suggesting training only the Actor for first 1000 steps, claims this can allow can prevent the policy
from "overreacting" to initial, noisy Critic estimates. 
"""


TensorBatch = List[torch.Tensor]

class Actor(nn.Module):
	def __init__(
			self, 
			state_dim: int, 
			action_dim: int, 
			hidden_dim: List[int] = [512, 256, 128], 
			dropout: float = 0.1,
			negative_slope: float = 0.1,
			log_std_min: float = -20,
            log_std_max: float = 2
		):
		
		super(Actor, self).__init__()
		self.log_std_min = log_std_min
		self.log_std_max = log_std_max

		layers = []
		input_dim = state_dim

		for h_dim in hidden_dim:
			layers.append(nn.Linear(input_dim, h_dim))
			# Critical: Always keep the activation, then apply dropout
			layers.append(nn.LeakyReLU(negative_slope)) 
			if dropout > 0:
				layers.append(nn.Dropout(dropout))
			input_dim = h_dim

		self.backbone = nn.Sequential(*layers)

		# Two heads: one for the Mean, one for the Log Standard Deviation
		self.mu_head = nn.Linear(input_dim, action_dim)
		self.log_std_head = nn.Linear(input_dim, action_dim)

		# Initialize weights
		# Initialize backbone for LeakyReLU signal flow
		for m in self.backbone:
			if isinstance(m, nn.Linear):
				nn.init.kaiming_normal_(
					m.weight, 
					a=negative_slope, 
					mode='fan_in',
					nonlinearity='leaky_relu'
				)
		
		# We use a very small range so the Sigmoid starts around 0.5 
		# (the middle of your log-min-maxed range).
		nn.init.uniform_(self.mu_head.weight, -3e-3, 3e-3)
		nn.init.constant_(self.mu_head.bias, 0.0)

		nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)
		nn.init.constant_(self.log_std_head.bias, 0.0)


	def forward(self, state: torch.Tensor, deterministic: bool = False, with_log_prob: bool = True):
		features = self.backbone(state)
		mu = self.mu_head(features)
		log_std = self.log_std_head(features)

		# Clip log_std for numerical stability
		log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
		std = torch.exp(log_std)

		dist = Normal(mu, std)

		if deterministic:
			x_t = mu
		else:
			x_t = dist.rsample() # Reparameterization trick (allows gradients)

		# Apply Tanh to bound [-1, 1], then scale to [0, 1] for your environment
		y_t = torch.tanh(x_t)
		action = (y_t + 1.0) / 2.0 # Transformation to [0, 1]

		if with_log_prob:
			# We must adjust log_prob because of the Tanh and Scaling transformations
			# 1. Standard log_prob from the Normal dist
			log_prob = dist.log_prob(x_t)
			# 2. Correction for Tanh: log(1 - tanh(x)^2)
			log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
			# 3. Correction for [0, 1] scaling (multiplying by 0.5 means adding log(2) to log_prob)
			log_prob -= torch.log(torch.tensor(0.5, device=state.device))
			log_prob = log_prob.sum(dim=-1, keepdim=True)
		else:
			log_prob = None

		return action, log_prob
	

	@torch.no_grad()
	def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
		self.eval() 
		state = torch.as_tensor(state, device=device, dtype=torch.float32)
		if state.ndim == 1:
			state = state.unsqueeze(0)
		
		# 1. We call forward with deterministic=True to get the 'mean' action
		# 2. We set with_log_prob=False because we don't need it for inference
		action, _ = self.forward(state, deterministic=True, with_log_prob=False)
		
		action = action.cpu().numpy()
		
		# Keep your clinical thresholding
		action[action < 1e-3] = 0.0
		return action


class Critic(nn.Module):
	def __init__(
		self, 
		state_dim, 
		action_dim, 
		dropout=0.1, 
		hidden_dim=512, 
		negative_slope=0.1,
	):
		
		super(Critic, self).__init__()

		# standardize the incoming embeddings
		self.state_norm = nn.LayerNorm(state_dim + action_dim) # if cql loss stays close to zero, we may
		# want to remove action_dim from this normalization, but for now we can keep it since actions are also log-min-maxed and should be on a similar scale to states	
		self.hidden_dim = hidden_dim

		self.hidden_dim2 = hidden_dim // 2  # Example of a second hidden layer with half the units of the first
		self.hidden_dim3 = self.hidden_dim2 // 2  # Example of a third hidden layer with half the units of the second

		self.negative_slope = negative_slope

		# Q1 architecture
		self.q1_net = nn.Sequential(
			nn.Linear(state_dim + action_dim, self.hidden_dim),
			nn.LayerNorm(self.hidden_dim), # Normalize internal activations
			nn.LeakyReLU(negative_slope=self.negative_slope),
			nn.Dropout(p=dropout),
			
			nn.Linear(self.hidden_dim, self.hidden_dim2),
			nn.LeakyReLU(negative_slope=self.negative_slope),
			nn.Dropout(p=dropout),
			
			nn.Linear(self.hidden_dim2, self.hidden_dim3),
			nn.LeakyReLU(negative_slope=self.negative_slope),
			# nn.Dropout(p=dropout),    
			
			nn.Linear(self.hidden_dim3, 1)
		)

		# Q2 architecture
		self.q2_net = nn.Sequential(
			nn.Linear(state_dim + action_dim, self.hidden_dim),
			nn.LayerNorm(self.hidden_dim), # Normalize internal activations 
			nn.LeakyReLU(negative_slope=self.negative_slope),
			nn.Dropout(p=dropout),
			
			nn.Linear(self.hidden_dim, self.hidden_dim2),
			nn.LeakyReLU(negative_slope=self.negative_slope),
			nn.Dropout(p=dropout),
			
			nn.Linear(self.hidden_dim2, self.hidden_dim3),
			nn.LeakyReLU(negative_slope=self.negative_slope),
			# nn.Dropout(p=dropout),            
			
			nn.Linear(self.hidden_dim3, 1)
		)

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

		sa = torch.cat([state, action], 1)

		# normalize states after concatenation
		sa = self.state_norm(sa)
		
		q1 = self.q1_net(sa)
		q2 = self.q2_net(sa)

		return q1, q2
	
	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)
		sa = self.state_norm(sa)
		q1 = self.q1_net(sa)
		return q1	
    

	@torch.no_grad()
	def get_neuron_stats(self, state, action): # not currently used, make sure to use eval when implemented
		# Ensure we are in eval mode to avoid Dropout interference
		self.eval()
		
		# Concatenate input
		x_init = torch.cat([state, action], dim=-1)
		
		stats = {}
		# We only need to check q1_net to get a representative sample of activity
		x = x_init
		leaky_idx = 0
		for layer in self.q1_net:
			x = layer(x)
			if isinstance(layer, nn.LeakyReLU):
				# 1. Percent of inputs utilizing the negative slope in this batch
				pct_negative = (x < 0).float().mean().item() * 100
				stats[f"negative_regime_L{leaky_idx}_pct"] = pct_negative
				
				# 2. True Inactive/Dead Neurons: Variance across the batch is essentially 0
				# (dim=0 calculates variance across the batch for each individual neuron)
				neuron_vars = torch.var(x, dim=0)
				pct_dead = (neuron_vars < 1e-6).float().mean().item() * 100
				stats[f"dead_neurons_L{leaky_idx}_pct"] = pct_dead 

				leaky_idx += 1            
				
		return stats


	# def get_grad_norm(self):
	# 	total_norm = 0
	# 	for p in self.parameters():
	# 		if p.grad is not None:
	# 			param_norm = p.grad.detach().norm(2)
	# 			total_norm += param_norm.item() ** 2
	# 	total_norm = total_norm ** 0.5
	# 	return total_norm
	

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



class CQL:
	def __init__(
		self,
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
		beta: float = 1.0,

		cql_alpha: float = 2.0,  # Use the alpha arg as the CQL conservative weight
		temp: float = 1.0,         # Temperature for the LogSumExp
		num_samples: int = 10,   # How many random actions to sample for the penalty

		device: str = "CUDA" if torch.cuda.is_available() else "CPU",
		action_mode: str = "continuous",
		dose_actions: int = 2,  #  Number of dose actions
		binary_actions: int = 0,		
		tau: float = 0.005,
		grad_clip: float = None,  # Add grad_clip argument
		ema_alpha: float = 0.99,  # Add ema_alpha argument for moving
		reward_scaling: float = 1.0,
	):
		
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
		self.beta = beta

		self.cql_alpha = cql_alpha  # Add cql_alpha as an argument and assign it to the instance variable
		self.temp = temp         # Add temp as an argument and assign it to the instance variable
		self.num_samples = num_samples   # Add num_samples as an argument and assign it to the instance variable
	
		self.cql_alpha = cql_alpha  #

		self.num_samples = 10   # How many random actions to sample for the penalty

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

		self.total_it = 0

		self.utils = TrainerUtils()

		# copy the actor network to create the target actor
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_target.eval()  # Target networks start in eval mode to disable Dropout

		# copy the critic network to create the target critic
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_target.eval()  # Target networks start in eval mode to disable Dropout


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		return self.actor(state).cpu().data.numpy().flatten()


	def compute_actor_loss(self,pi: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
		"""
		Refined BC Loss for sparse clinical actions (Fluids/Vasopressors).
		"""
		# 1. Standard MSE for the base BC loss
		mse_loss = F.mse_loss(pi, action, reduction='none') # MSE loss penalizes large deviations, gemini thinks this is best for the actor
		
		# 2. Create a 'Clinical Importance' weight
		# We weight time-steps where the doctor actually intervened (action > 0)
		# This prevents the model from just learning to 'do nothing'
		weights = torch.where(action > 0, 5.0, 1.0) # You can tune this weight (5.0) based on how much you want to encourage matching the doctor's interventions
		# we will probably want to implement different weights for fluids vs vasopressors in the mixed action setting, but for now we can just apply the same weight to all non-zero actions

		weighted_bc_loss = (mse_loss * weights).mean()

		# 3. Optional: Add a small L1 penalty to keep the policy sparse
		# (Only if you find the AI is 'micro-dosing' constantly)
		l1_sparse = pi.abs().mean() * 1e-3 # this should prevent many small non-zero actions, while still allowing for 
										# large actions when clinically indicated. Tune the 1e-3 coefficient as needed.
		
		total_bc_loss = weighted_bc_loss + l1_sparse

		actor_dict = {
			"total_bc_loss": total_bc_loss.item(),
			"bc_mse_only": mse_loss.mean().item(),
			"weighted_bc_loss": weighted_bc_loss.item(),
			"l1_sparse": l1_sparse.item(),
			"pi_mean_val": pi.mean().item()
		}
		
		return actor_dict
		

	def target_q(self, action, next_state, reward, not_done, noise=True):
		with torch.no_grad():
			# Select action according to policy and add clipped noise
			next_action, _ = self.actor_target(next_state, deterministic=True)
		
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)

			next_action = (next_action + noise).clamp(0, 1.0) # Using 1.0 since our actor outputs [0,1]

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)

			# Bellman equation
			target_Q = reward + not_done * self.discount * target_Q			
			
			return target_Q


	def train(self, batch: TensorBatch) -> Tuple[Dict[str, float], Dict[str, float]]:
		self.total_it += 1

		# Sample replay buffer 
		state, action, reward, next_state, next_action, done, mortality = TrainerUtils._unpack_batch(batch, self.device)
		
		reward = reward * self.reward_scaling
		
		# compute the target Q value
		treat_survival_as_truncated = True
		if treat_survival_as_truncated:
			# make done 0 where mortality is 0, since we want to treat survival as truncated rather than terminal
			done = done * mortality
		
		not_done = 1 - done

		self.critic.train()          # Dropout ACTIVE for the network we are updating
		
		with torch.no_grad(): # uses target actor for next action
			target_Q = self.target_q(action, next_state, reward, not_done, noise=True)

		# Get current Q estimates>
		current_Q1, current_Q2 = self.critic(state, action)

		# 2. CQL CONSERVATIVE LOSS
        # We need to sample random actions to penalize them
		batch_size = state.shape[0]

		with torch.no_grad():
			# Sample random actions uniformly [0, 1]
			random_actions = torch.FloatTensor(batch_size, self.num_samples, self.dose_actions).uniform_(0, 1).to(self.device)
			
			# Sample current actions from our Stochastic Actor
			# We sample multiple times to get a good estimate of the current policy's Q-values
			curr_actions, curr_log_probs = self.actor(
				state.repeat_interleave(self.num_samples, dim=0)
			)
			curr_actions = curr_actions.view(batch_size, self.num_samples, -1)

			# sample next_actions from the actor for the next states
			next_actions, next_log_probs = self.actor(
				next_state.repeat_interleave(self.num_samples, dim=0)
			)
			next_actions = next_actions.view(batch_size, self.num_samples, -1)
			
		# Evaluate these actions in the Critic
		# (Using a helper or flattening/unflattening)
		def get_q_for_samples(s, a_samples, critic_net):
			# s: [batch, dim], a_samples: [batch, num_samples, act_dim]
			s_rep = s.unsqueeze(1).repeat(1, self.num_samples, 1).view(-1, s.shape[-1])
			a_flat = a_samples.reshape(-1, a_samples.shape[-1])
			q1, q2 = critic_net(s_rep, a_flat)
			return q1.view(batch_size, self.num_samples), q2.view(batch_size, self.num_samples)
		
		q1_rand, q2_rand = get_q_for_samples(state, random_actions, self.critic)
		q1_curr, q2_curr = get_q_for_samples(state, curr_actions, self.critic)
		q1_next, q2_next = get_q_for_samples(next_state, next_actions, self.critic) 

		# Inside the training loop, after calculating q1_rand, q1_curr, etc.
		with torch.no_grad():
			# The 'Gap' is the difference between what the Critic thinks of its 
			# own chosen actions vs. what it thinks of the doctor's actions.
			q_gap1 = q1_curr.mean() - current_Q1.mean()
			q_gap2 = q2_curr.mean() - current_Q2.mean()
			
		# 3. CONSTRUCT THE CQL PENALTY (LogSumExp)
		# We concatenate the Q-values of all sampled actions
		cat_q1 = torch.cat([q1_rand, q1_curr, q1_next, current_Q1], dim=1)
		cat_q2 = torch.cat([q2_rand, q2_curr, q2_next, current_Q2], dim=1)

		cql_loss1 = torch.logsumexp(cat_q1 / self.temp, dim=1).mean() * self.temp - current_Q1.mean()
		cql_loss2 = torch.logsumexp(cat_q2 / self.temp, dim=1).mean() * self.temp - current_Q2.mean()

		# 4. COMBINED CRITIC LOSS
		bellman_loss = (F.smooth_l1_loss(current_Q1, target_Q) + F.smooth_l1_loss(current_Q2, target_Q)) / 2.0
		cql_loss = cql_loss1 + cql_loss2
		# dynamic warm up for cql_alpha
		if self.total_it < 5000:
			current_alpha = 0.0
		elif 5000 <= self.total_it < 15000:
			# Linearly scale alpha from 0.0 to your target_alpha over 10,000 steps
			progress = (self.total_it - 5000) / 10000
			current_alpha = progress * self.cql_alpha
		else:
			current_alpha = self.cql_alpha

		critic_loss = bellman_loss + current_alpha * cql_loss			
					
		losses = {
			"bellman_loss": bellman_loss.item(),
			"cql_loss": cql_loss.item(),
			"current_alpha": current_alpha,
			"critic_loss": critic_loss.item(), 
		}
				
		# log_dict = {
		# 	"cql_q_gap": (q_gap1 + q_gap2).item() / 2.0,
		# 	"cql_q_rand": (q1_rand.mean() + q2_rand.mean()).item() / 2.0,
    	# 	"cql_q_data": (current_Q1.mean() + current_Q2.mean()).item() / 2.0,
		# 	"cql_q_next": (q1_next.mean() + q2_next.mean()).item() / 2.0,
		# 	"bellman_loss": bellman_loss.item(),
		# 	"cql_loss": current_alpha * (cql_loss1 + cql_loss2).item(),
		# 	"critic_loss": critic_loss.item()}

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()

		# capture gradient norms
		total_critic_grad_norm = self._get_grad_norm(self.critic)
		grad_norm = {"critic_grad_norm": total_critic_grad_norm}
		
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


		# Delayed policy updates
		if self.total_it % self.policy_freq != 0:
			with torch.no_grad():
				pi, log_prob = self.actor(state)
				Q1_pi, Q2_pi = self.critic(state, pi)
			Q_pi = torch.min(Q1_pi, Q2_pi)
			q_stats = get_q_stats(current_Q1, current_Q2, Q_pi, mortality)

			
		
		else:
			# Compute actor loss
			self.actor.train() # Dropout ACTIVE for the network we are updating
			
			# Sample actions from the stochastic actor
			pi, log_prob = self.actor(state)
			Q1_pi, Q2_pi = self.critic(state, pi)
			Q_pi = torch.min(Q1_pi, Q2_pi)

			q_stats = get_q_stats(current_Q1, current_Q2, Q_pi, mortality)
					
			# Standard SAC-style or TD3-style actor loss
			# Note: We usually keep a small BC component for safety in medical RL
			# but the lmbda logic is usually replaced by pure Q maximization in CQL
			actor_loss_q = -Q_pi.mean()

			# BC Loss - compute actor loss using TrainerUtils function - the "anchor"
			actor_dict = self.compute_actor_loss(pi, action)
			actor_loss_bc = actor_dict['total_bc_loss'] # this is the combined BC loss that includes both the weighted MSE and the sparsity penalty
			losses.update(actor_dict) # add the actor_dict components to the losses dict for logging

			# add a small entropy bonus
			ent_alpha = 0.05
			actor_loss_ent = ent_alpha * log_prob.mean() if log_prob is not None else 0.0

			actor_loss = actor_loss_q 
			# actor_loss = actor_loss_q + self.beta * actor_loss_bc + actor_loss_ent # i think CQL doesn't normally include a BC 
			# component, but gemini says keep if for medical, probably use a very small beta
			losses.update({
				"actor_loss_total": actor_loss.item(),
				"actor_loss_q": actor_loss_q.item(),
				"actor_loss_ent": actor_loss_ent.item()
			})
						
			# Final combined update
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			
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

			# # Log the actor loss and its components
			# log_dict.update({
			# 	"actor_loss": actor_loss.item(),
			# 	"actor_loss_q": actor_loss_q.item(),
			# 	"actor_loss_bc": actor_loss_bc.item(),
			# 	"pi_mean_val": pi.mean().item(),
			# 	"log_prob_mean": log_prob.mean().item() if log_prob is not None else float('nan'),
			# 	"Q_mean": Q.mean().item(),
			# 	"actor_loss_ent": actor_loss_ent.item()})
			
			# log_dict.update(actor_dict)
			
			with torch.no_grad():
				for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
					target_param.mul_(1.0 - self.tau)
					target_param.add_(param.data * self.tau)

				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
					target_param.mul_(1.0 - self.tau)
					target_param.add_(param.data * self.tau)
						
		# return log_dict, grad_norm
		return losses, q_stats, grad_norm
	

	def actor_forward_for_sarsa(self, batch: TensorBatch) -> torch.Tensor:
		state, _, _, _, _, _, _ = TrainerUtils._unpack_batch(batch, self.device)

		self.actor.eval()  # Set actor to eval mode to disable Dropout
		with torch.no_grad():
			new_actions, _ = self.actor(state) 
		
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
			actions, _ = self.actor(batch_states, deterministic=True, with_log_prob=False)
			q1, q2 = self.critic(batch_states, actions)
			q_batch = torch.min(q1, q2)
			q_values.append(q_batch.detach().cpu().numpy())

		q_values = np.concatenate(q_values, axis=0)
		v0_mean = q_values.mean()
		v0_sem = q_values.std() / np.sqrt(len(q_values))

		return v0_mean, v0_sem

	def evaluate(self, batch: TensorBatch) -> Dict[str, float]:
		self.actor.eval()  # This is correct and important!
		self.critic.eval()  # Ensure critic is also in eval mode
		
		state, action, reward, next_state, next_action, done, mortality = TrainerUtils._unpack_batch(batch, self.device)
		
		reward = reward * self.reward_scaling

		# compute the target Q value
		treat_survival_as_truncated = True
		if treat_survival_as_truncated:
				# make done 0 where mortality is 0, since we want to treat survival as truncated rather than terminal
			done = done * mortality
		
		not_done = 1 - done

		with torch.no_grad():
			# compute the target Q value
			target_Q = self.target_q(action, next_state, reward, not_done, noise=False)

			# Get current Q estimates
			current_Q1, current_Q2 = self.critic(state, action) # critic value of action taken by doctor
			bellman_loss = (F.smooth_l1_loss(current_Q1, target_Q) + F.smooth_l1_loss(current_Q2, target_Q)) / 2.0

			# 3. Actor Actions and their Critic Values
            # We use deterministic=True to see the "best" action the model recommends
			pi, _ = self.actor(state, deterministic=True)
			Q_pi_1, Q_pi_2 = self.critic(state, pi)
			Q_pi = torch.min(Q_pi_1, Q_pi_2)

			eval_q_gap = (Q_pi - ((current_Q1 + current_Q2)/2.0)).mean().item()

			# 4. BC Loss (How much we deviate from the doctor)
			actor_dict = self.compute_actor_loss(pi, action)


			# ADD THIS LINE:
			neuron_stats = self.critic.get_neuron_stats(state, action)
	
			losses = {
				"bellman_loss": bellman_loss.item(),
			}
			losses.update(actor_dict)

			# get Q statistics
			q_stats = get_q_stats(current_Q1, current_Q2, Q_pi, mortality)

			# log_dict = {
			# 	"bellman_loss": bellman_loss.item(),
			# 	"pi_Q_mean": Q_pi.mean().item(),
			# 	"current_Q_mean": ((current_Q1 + current_Q2) / 2.0).mean().item(),
			# 	"eval_q_gap": eval_q_gap,
			# 	"actor_loss_bc": actor_loss_bc.item()
			# }

			# log_dict = self.get_q_stats(current_Q1, current_Q2, Q_pi, mortality, log_dict)
			# log_dict.update(actor_dict)
			   
		# return log_dict, action, pi
		return losses, q_stats, neuron_stats

	
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
	
	hidden_dim = hyperparams['hidden_dim']
	dropout = hyperparams['dropout']
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
		beta=hyperparams['beta'],
		cql_alpha=hyperparams['cql_alpha'],
		temp=hyperparams['temp'],
		num_samples=hyperparams['num_samples'],
		
	)

	return trainer


def load_trained_actor(state_dim, action_dim, config_manager = None, model_path = None):
	"""
	used to load a trained actor for train of Fitted Q Evaluation (FQE) or for deployment. 
	"""
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
			dropout = results["hyperparameters"]["dropout"]

	actor = Actor(
		state_dim = state_dim, 
		action_dim = action_dim, 
		hidden_dim = hidden_dim
	).to(config_manager.training_config['device'])

	actor.load_state_dict(torch.load(final_model_path, weights_only=True))
	actor.eval()

	return actor