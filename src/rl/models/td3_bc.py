import typing
from typing import Dict, List, Tuple
import copy
# import config_manager
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
			negative_slope: float = 0.1
			):
		
		super(Actor, self).__init__()

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
		self.output_head = nn.Linear(input_dim, action_dim)
		self.sigmoid = nn.Sigmoid()

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
		nn.init.uniform_(self.output_head.weight, -3e-3, 3e-3)
		nn.init.constant_(self.output_head.bias, 0.0)

	def forward(self, state: torch.Tensor) -> torch.Tensor:
		# Returns values in [0, 1] for both Vasopressors and Fluids
		return self.sigmoid(self.output_head(self.backbone(state)))

	@torch.no_grad()
	def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
		self.eval() # Ensure dropout is OFF during inference
		state = torch.as_tensor(state, device=device, dtype=torch.float32)
		if state.ndim == 1:
			state = state.unsqueeze(0)
		
		action = self.forward(state).cpu().numpy().flatten()
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
			scheduler: torch.optim.lr_scheduler._LRScheduler = None
		):
		
		super(Critic, self).__init__()

		# standardize the incoming embeddings
		self.state_norm = nn.LayerNorm(state_dim + action_dim)
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
					nn.init.constant_(m.bias, -0.2)
			
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
		# Concatenate input
		x_init = torch.cat([state, action], dim=-1)
		
		stats = {}
		# We only need to check q1_net to get a representative sample of activity
		x = x_init
		leaky_idx = 0
		for layer in self.q1_net:
			x = layer(x)
			if isinstance(layer, nn.LeakyReLU):
				# (x <= 0) is 1 for inactive, 0 for active
				pct_inactive = (x <= 0).float().mean().item() * 100
				stats[f"inactive_L{leaky_idx}_pct"] = pct_inactive
				leaky_idx += 1
				
		return stats


	def get_grad_norm(self):
		total_norm = 0
		for p in self.parameters():
			if p.grad is not None:
				param_norm = p.grad.detach().norm(2)
				total_norm += param_norm.item() ** 2
		total_norm = total_norm ** 0.5
		return total_norm
	

class TD3_BC(object):
	def __init__(
		self,
		max_action: np.ndarray,
		actor: nn.Module = None,
		actor_optimizer: torch.optim.Optimizer = None,
		critic: nn.Module = None,
		critic_optimizer: torch.optim.Optimizer = None,	
		discount: float = 0.99,
		policy_noise: float = 0.2,
		noise_clip: float = 0.05,
		policy_freq: int = 2,
		alpha: float = 2.5,
		device: str = "CUDA" if torch.cuda.is_available() else "CPU",
		action_mode: str = "continuous",
		dose_actions: int = 2,  #  Number of dose actions
		binary_actions: int = 0,		
		tau: float = 0.005,
		grad_clip: float = None,  # Add grad_clip argument
		ema_alpha: float = 0.99,  # Add ema_alpha argument for moving
		reward_scaling: float = 1.0,
		scheduler: torch.optim.lr_scheduler._LRScheduler = None

	):
		
		self.max_action = max_action
		self.actor = actor
		self.actor_optimizer = actor_optimizer
		self.critic = critic
		self.critic_optimizer = critic_optimizer		
		self.discount = discount
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.alpha = alpha
		self.device = device
		self.action_mode = action_mode
		self.dose_actions = dose_actions
		self.binary_actions = binary_actions
		self.tau = tau
		self.grad_clip = grad_clip

		self.running_lmbda = 1.0

		self.scheduler = scheduler
		
		# for calculating moving average of gradient clipping binary
		self.clip_fraction_ema = 0.0
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
		l1_sparse = pi.abs().mean() * 1e-3 # this should prevent many small non-zero actions, while still allowing for large actions when clinically indicated. Tune the 1e-3 coefficient as needed.
		
		total_bc_loss = weighted_bc_loss + l1_sparse

		actor_dict = {
			"total_bc_loss": total_bc_loss.item(),
			"bc_mse_only": mse_loss.mean().item(),
			"weighted_bc_loss": weighted_bc_loss.item(),
			"l1_sparse": l1_sparse.item(),
			"pi_mean_val": pi.mean().item()
		}
		
		return total_bc_loss, actor_dict
		

	def target_q(self, action, next_state, reward, not_done, noise=True):
		with torch.no_grad():
			# Select action according to policy and add clipped noise
			if noise:
				noise = (
					torch.randn_like(action) * self.policy_noise
				).clamp(-self.noise_clip, self.noise_clip)
			else:
				noise = torch.zeros_like(action)

			next_action = (
				self.actor_target(next_state) + noise
			).clamp(0, self.max_action) # action must be min maxed (or log min maxed) so min is 0.

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q			
			
			return target_Q


	def get_q_stats(self, current_Q1, current_Q2, actor_Q, mortality, log_dict) -> Tuple[float, float]:
		
		# create current_Q as the mean of current_Q1 and current_Q2
		current_Q = (current_Q1 + current_Q2) / 2.0
		mean_Q = current_Q.mean().item()
		variance_Q = current_Q.var().item()
		batch_n = current_Q.shape[0]
		uncertainty_Q = F.mse_loss(current_Q1, current_Q2).item() # simple measure of uncertainty based on disagreement between the two critics

		if actor_Q is not None:
			advantage_Q = (actor_Q - current_Q).mean().item() # how much better the actor's action is compared to the current Q estimate, on average. This can help us understand if the policy is improving over time.
		else:
			advantage_Q = None

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

		log_dict.update({
			"batch_n": batch_n,
			"mean_Q": mean_Q,
			"variance_Q": variance_Q,
			"uncertainty_Q": uncertainty_Q,
			"actor_Q": actor_Q.mean().item() if actor_Q is not None else None,
			"advantage_Q": advantage_Q,
			"survival_n": survival_n,
			"survival_Q": survival_Q,
			"survival_variance_Q": survival_variance_Q,
			"non_survival_n": non_survival_n,
			"non_survival_Q": non_survival_Q,
			"non_survival_variance_Q": non_survival_variance_Q
		})
		return log_dict


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
		
		with torch.no_grad():
			target_Q = self.target_q(action, next_state, reward, not_done, noise=True)

		# Get current Q estimates>
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		# critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		critic_loss = (F.smooth_l1_loss(current_Q1, target_Q) + F.smooth_l1_loss(current_Q2, target_Q)) / 2.0
		
		log_dict = {"critic_loss": critic_loss.item()}

		# get Q statistics
		log_dict = self.get_q_stats(current_Q1, current_Q2, None, mortality, log_dict)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()		

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

		# Delayed policy updates
		if self.total_it % self.policy_freq != 0:
			log_dict = self.get_q_stats(current_Q1, current_Q2, None, mortality, log_dict)
		else:
			# Compute actor loss
			self.actor.train() # Dropout ACTIVE for the network we are updating
			pi = self.actor(state)
			Q = self.critic.Q1(state, pi)

			log_dict = self.get_q_stats(current_Q1, current_Q2, Q, mortality, log_dict)
			
			# calculate the standard TD3-BC lambda coefficient
			lmbda_base = self.alpha/Q.abs().mean().detach()

			# warmup logic
			# linear ramp-up of the Q-gradient influence over 5000 iterations
			warmup_steps = 5000
			warmup_factor = min(1.0, self.total_it / warmup_steps)
			active_lmbda = lmbda_base * warmup_factor

			# running lmbda
			self.running_lmbda = 0.95 * self.running_lmbda + 0.05 * active_lmbda.item()

			# BC Loss - compute actor loss using TrainerUtils function - the "anchor"
			actor_loss_bc, actor_dict = self.compute_actor_loss(pi, action)
			
			# Q-Loss (The "Incentive")
			actor_loss_q = -self.running_lmbda * Q.mean()
			
			actor_loss = actor_loss_q + actor_loss_bc
			
			# Gradient Ratio Logging ---
			# We calculate the norms of the gradients from each component separately
			self.actor_optimizer.zero_grad()
			
			# Get gradient norm for BC
			actor_loss_bc.backward(retain_graph=True)
			grad_norm_bc = self._get_grad_norm(self.actor)
			
			self.actor_optimizer.zero_grad()

			# Get gradient norm for Q
			actor_loss_q.backward(retain_graph=True)
			grad_norm_q = self._get_grad_norm(self.actor)

			# Final combined update
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Log the ratio
			log_dict.update({
				"actor_loss": actor_loss.item(),
				"bc_grad_norm": grad_norm_bc,
				"q_grad_norm": grad_norm_q,
				"grad_ratio_q_bc": grad_norm_q / (grad_norm_bc + 1e-6),
				"warmup_factor": warmup_factor
			})
			
			log_dict.update(actor_dict)
			
			with torch.no_grad():
				for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
					target_param.mul_(1.0 - self.tau)
					target_param.add_(param.data * self.tau)

				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
					target_param.mul_(1.0 - self.tau)
					target_param.add_(param.data * self.tau)
						
		return log_dict, grad_norm
	
	def _get_grad_norm(self, model: nn.Module) -> float:
		total_norm = 0
		for p in model.parameters():
			if p.grad is not None:
				param_norm = p.grad.data.norm(2)
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

		q_values = []
		for i in range(0, len(start_states), batch_size):
			batch_states = start_states[i:i+batch_size]
			actions = self.actor(batch_states)
			q1, q2 = self.critic(batch_states, actions)
			q_batch = torch.min(q1, q2)
			q_values.append(q_batch.cpu().numpy())

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

			# Compute critic loss
			# critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
			critic_loss = (F.smooth_l1_loss(current_Q1, target_Q) + F.smooth_l1_loss(current_Q2, target_Q)) / 2.0			
			log_dict = {"critic_loss": critic_loss.item()}

			# Compute actor loss
			pi = self.actor(state)
			Q = self.critic.Q1(state, pi) # critic value of action suggested by the actor
			# lmbda = self.alpha/Q.abs().mean().detach()		
			
			log_dict = self.get_q_stats(current_Q1, current_Q2, Q, mortality, log_dict)
						
			# compute actor loss using TrainerUtils function
			actor_loss_prelim, actor_dict = self.compute_actor_loss(pi, action)

			if self.total_it == 0:
				current_lmbda = self.alpha / (Q.abs().mean().detach() + 1e-6)
			else:
				current_lmbda = self.running_lmbda

			actor_loss = -current_lmbda * Q.mean() + actor_loss_prelim
			log_dict.update(actor_dict)
			log_dict["actor_loss"] = actor_loss.item()
			log_dict["current_lmbda"] = current_lmbda.item()

		return log_dict, action, pi

	
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


def get_trainer(config_manager, state_dim, action_dim, max_action, hyperparams):

	learning_rate = float(hyperparams['learning_rate'])
	weight_decay = float(hyperparams.get('weight_decay', 0.01)) # New hyperparam
	device = config_manager.training_config['device']

	# action_mode = config_manager.action_mode FOR NOW, WE CAN ASSUME CONTINUOUS

	# if action_mode == 'binary_only' or action_mode == 'continuous_only':
	# 	actor = Actor(
	# 		state_dim=state_dim, 
	# 		action_dim=action_dim, 
	# 		hidden_dim=hyperparams['hidden_dim'], 
	# 		dropout=hyperparams['dropout'],
	# 		max_action = 1.0,
	# 		action_mode = action_mode  # max_action is 1.0 for binary
	# 		).to(config_manager.training_config['device'])
		
	# elif action_mode == 'mixed':
	# 	actor = MixedActionActor(
	# 		state_dim=state_dim, 
	# 		dose_actions=action_dim//2, 
	# 		binary_actions=action_dim//2, 
	# 		hidden_dim=hyperparams['hidden_dim'], 
	# 		dropout=hyperparams['dropout'],
	# 		max_action = max_action
	# 		).to(config_manager.training_config['device'])
		
	actor = Actor(
		state_dim=state_dim, 
		action_dim=action_dim, 
		# hidden_dim=hyperparams['hidden_dim'], 
		dropout=hyperparams['dropout']
	).to(config_manager.training_config['device'])
	
	actor_optimizer = torch.optim.Adam(
		actor.parameters(), 
		lr=learning_rate,
		weight_decay=weight_decay
		)
	
	critic = Critic(state_dim, action_dim).to(config_manager.training_config['device'])

	critic_optimizer = torch.optim.Adam(
		critic.parameters(),
		lr=learning_rate,
		weight_decay=weight_decay
	)

	critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		critic_optimizer, 
		T_max=hyperparams['total_steps'],
		eta_min=1e-6
	)

	trainer = TD3_BC(               # LIST OF PARAMETERS NEEDS TO BE UPDATED
		max_action=max_action,
		actor=actor,
		actor_optimizer=actor_optimizer,
		critic=critic,
		critic_optimizer=critic_optimizer,
		discount=hyperparams['discount'],
		device=config_manager.training_config['device'],
		action_mode=config_manager.action_mode,
		tau=hyperparams['tau'],
		policy_noise=hyperparams['policy_noise'],
		noise_clip=hyperparams['noise_clip'],
		policy_freq=hyperparams['policy_freq'],
		alpha=hyperparams['alpha'],
		scheduler=critic_scheduler
	)

	# if action_mode == 'binary_only' or action_mode == 'continuous_only':
	# 	trainer = TD3_BC(
	# 		max_action=max_action,
	# 		actor=actor,
	# 		actor_optimizer=actor_optimizer,
	# 		critic=critic,
	# 		critic_optimizer=critic_optimizer,
	# 		discount=config_manager.training_config['discount'],
	# 		device=config_manager.training_config['device'],
	# 		action_mode=action_mode,
	# 		tau=config_manager.training_config['tau'],	
	# 		policy_noise=config_manager.training_config['policy_noise'],
	# 		noise_clip=config_manager.training_config['noise_clip'],
	# 		policy_freq=config_manager.training_config['policy_freq'],
	# 		alpha=config_manager.training_config['alpha']		
	# 	)
	
	# elif action_mode == 'mixed':
	# 	trainer = TD3_BC(
	# 			max_action=max_action,
	# 			actor=actor,
	# 			actor_optimizer=actor_optimizer,
	# 			critic=critic,
	# 			critic_optimizer=critic_optimizer,
	# 			discount=config_manager.training_config['discount'],
	# 			device=config_manager.training_config['device'],
	# 			action_mode=action_mode,
	# 			dose_actions=action_dim//2,
	# 			binary_actions=action_dim//2,
	# 			tau=config_manager.training_config['tau'],	
	# 			policy_noise=config_manager.training_config['policy_noise'],
	# 			noise_clip=config_manager.training_config['noise_clip'],
	# 			policy_freq=config_manager.training_config['policy_freq'],
	# 			alpha=config_manager.training_config['alpha']		
	# 	)
	return trainer