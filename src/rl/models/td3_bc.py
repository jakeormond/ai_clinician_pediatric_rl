import typing
from typing import Dict, List, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .shared_components import Actor, MixedActionActor, TrainerUtils


TensorBatch = List[torch.Tensor]


# class Actor(nn.Module):
# 	"""
# 	This is the implementation from CORL
# 	"""
# 	def __init__(self, state_dim, action_dim, max_action):
# 		super(Actor, self).__init__()

# 		self.l1 = nn.Linear(state_dim, 256)
# 		self.l2 = nn.Linear(256, 256)
# 		self.l3 = nn.Linear(256, action_dim)
		
# 		self.max_action = max_action
		

# 	def forward(self, state):
# 		a = F.relu(self.l1(state))
# 		a = F.relu(self.l2(a))
# 		return self.max_action * torch.tanh(self.l3(a))


# class MixedActionActor(nn.Module):
# 	"""
# 	This is the same actor from behavioural_cloning.py. I've copied it here
# 	as changes will likely be made specific to TD3+BC training.
# 	"""
# 	def __init__(self, state_dim: int, dose_actions: int, 
# 			binary_actions: int, hidden_dim: List[int], max_action: float):
# 		super().__init__()

# 		# Build network dynamically from hidden_dim list
# 		layers = []
# 		input_dim = state_dim

# 		# Add hidden layers
# 		for h_dim in hidden_dim:
# 			layers.extend([
# 				nn.Linear(input_dim, h_dim),
# 				nn.ReLU()
# 			])
# 			input_dim = h_dim

# 		# Create the shared network (this was missing!)
# 		self.shared_net = nn.Sequential(*layers)

# 		# Separate heads for different action types
# 		self.dose_head = nn.Sequential(
# 			nn.Linear(input_dim, dose_actions),
# 			nn.Sigmoid()  # Outputs in [0,1] range for doses
# 		)

# 		self.binary_head = nn.Sequential(
# 			nn.Linear(input_dim, binary_actions), 
# 			nn.Sigmoid()  # Outputs probabilities for binary actions
# 		)

# 		self.max_action = max_action
# 		self.dose_actions = dose_actions

# 	def forward(self, state: torch.Tensor) -> torch.Tensor:
# 		# Shared features
# 		features = self.shared_net(state)

# 		# Different processing for different action types
# 		dose_outputs = self.dose_head(features) * self.max_action  # Scale doses
# 		binary_outputs = self.binary_head(features)  # Keep as probabilities

# 		# Concatenate outputs
# 		return torch.cat([dose_outputs, binary_outputs], dim=1)

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


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
		noise_clip: float = 0.5,
		policy_freq: int = 2,
		alpha: float = 2.5,
        device: str = "CUDA" if torch.cuda.is_available() else "CPU",
        action_mode: str = "mixed",
        dose_actions: int = 2,  #  Number of dose actions
        binary_actions: int = 2,		
		tau: float = 0.005,
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

		self.total_it = 0

		self.utils = TrainerUtils()

		# copy the actor network to create the target actor
		self.actor_target = copy.deepcopy(self.actor)

		# copy the critic network to create the target critic
		self.critic_target = copy.deepcopy(self.critic)



	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		return self.actor(state).cpu().data.numpy().flatten()


	def compute_actor_loss(self, pi: torch.Tensor, action: torch.Tensor) -> Dict[str, float]:
		log_dict = {}
		if self.action_mode in ["binary_only", "continuous_only"]:
			
			# actor_loss = self._compute_actor_loss(pi, action)
			actor_loss = TrainerUtils._compute_actor_loss(
				pi=pi, 
				action=action, 
				action_mode=self.actor.action_mode,
				dose_actions=self.dose_actions if hasattr(self, 'dose_actions') else None
			)

			log_dict["actor_loss_prelim"] = actor_loss.item()

		elif self.action_mode == "mixed":
			actor_loss, dose_loss, binary_loss = TrainerUtils._compute_actor_loss(
				pi=pi, 
				action=action, 
				action_mode=self.actor.action_mode,
				dose_actions=self.dose_actions if hasattr(self, 'dose_actions') else None
			)     
			
			# Log individual losses for monitoring
			log_dict["actor_loss_prelim"] = actor_loss.item()
			log_dict["dose_loss"] = dose_loss.item() 
			log_dict["binary_loss"] = binary_loss.item()

		return actor_loss, log_dict



	def target_q(self, action, next_state, reward, not_done):
		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q			
			
			return target_Q
		

	def train(self, batch: TensorBatch) -> Dict[str, float]:
		self.total_it += 1

		# Sample replay buffer 
		state, action, reward, next_state, done = TrainerUtils._unpack_batch(batch, self.device)
		not_done = 1 - done
		
		# compute the target Q value
		target_Q = self.target_q(action, next_state, reward, not_done)

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		log_dict = {"critic_loss": critic_loss.item()}

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor loss
			pi = self.actor(state)
			Q = self.critic.Q1(state, pi)
			lmbda = self.alpha/Q.abs().mean().detach()

			# compute actor loss using TrainerUtils function
			actor_loss_prelim, actor_dict = self.compute_actor_loss(pi, action)
			actor_loss = -lmbda * Q.mean() + actor_loss_prelim
			log_dict.update(actor_dict)
			log_dict["actor_loss"] = actor_loss.item()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		return log_dict
	

	@torch.no_grad()
	def estimate_v0(self, start_states: torch.Tensor, batch_size: int = 256) -> float:
		"""
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
		
		state, action, reward, next_state, done = TrainerUtils._unpack_batch(batch, self.device)
		not_done = 1 - done

		with torch.no_grad():
			# compute the target Q value
			target_Q = self.target_q(action, next_state, reward, not_done)

			# Get current Q estimates
			current_Q1, current_Q2 = self.critic(state, action)

			# Compute critic loss
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
			log_dict = {"critic_loss": critic_loss.item()}
		
			# Compute actor loss
			pi = self.actor(state)
			Q = self.critic.Q1(state, pi)
			lmbda = self.alpha/Q.abs().mean().detach()

			# compute actor loss using TrainerUtils function
			actor_loss_prelim, actor_dict = self.compute_actor_loss(pi, action)
			actor_loss = -lmbda * Q.mean() + actor_loss_prelim
			log_dict.update(actor_dict)
			log_dict["actor_loss"] = actor_loss.item()

		return log_dict, action, pi


	def state_dict(self) -> Dict[str, torch.Tensor]:
		return {
			"actor": self.actor.state_dict(),
			"actor_optimizer": self.actor_optimizer.state_dict(),
			"critic": self.critic.state_dict(),
			"critic_optimizer": self.critic_optimizer.state_dict(),
			"actor_target": self.actor_target.state_dict(),
			"critic_target": self.critic_target.state_dict(),
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

	action_mode = config_manager.action_mode

	if action_mode == 'binary_only' or action_mode == 'continuous_only':
		actor = Actor(
			state_dim=state_dim, 
			action_dim=action_dim, 
			hidden_dim=hyperparams['hidden_dim'], 
			dropout=hyperparams['dropout'],
			max_action = 1.0,
			action_mode = action_mode  # max_action is 1.0 for binary
			).to(config_manager.training_config['device'])
		
	elif action_mode == 'mixed':
		actor = MixedActionActor(
			state_dim=state_dim, 
			dose_actions=action_dim//2, 
			binary_actions=action_dim//2, 
			hidden_dim=hyperparams['hidden_dim'], 
			dropout=hyperparams['dropout'],
			max_action = max_action
			).to(config_manager.training_config['device'])
		
	
	actor_optimizer = torch.optim.Adam(
		actor.parameters(), 
		lr=learning_rate
		)
	
	critic = Critic(state_dim, action_dim).to(config_manager.training_config['device'])

	critic_optimizer = torch.optim.Adam(
		critic.parameters(),
		lr=learning_rate
	)

	if action_mode == 'binary_only' or action_mode == 'continuous_only':
		trainer = TD3_BC(
			max_action=max_action,
			actor=actor,
			actor_optimizer=actor_optimizer,
			critic=critic,
			critic_optimizer=critic_optimizer,
			discount=config_manager.training_config['discount'],
			device=config_manager.training_config['device'],
			action_mode=action_mode,
			tau=config_manager.training_config['tau'],	
			policy_noise=config_manager.training_config['policy_noise'],
			noise_clip=config_manager.training_config['noise_clip'],
			policy_freq=config_manager.training_config['policy_freq'],
			alpha=config_manager.training_config['alpha']		
		)
	
	elif action_mode == 'mixed':
		trainer = TD3_BC(
				max_action=max_action,
				actor=actor,
				actor_optimizer=actor_optimizer,
				critic=critic,
				critic_optimizer=critic_optimizer,
				discount=config_manager.training_config['discount'],
				device=config_manager.training_config['device'],
				action_mode=action_mode,
				dose_actions=action_dim//2,
				binary_actions=action_dim//2,
				tau=config_manager.training_config['tau'],	
				policy_noise=config_manager.training_config['policy_noise'],
				noise_clip=config_manager.training_config['noise_clip'],
				policy_freq=config_manager.training_config['policy_freq'],
				alpha=config_manager.training_config['alpha']		
		)
	return trainer