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
	
	
	def sample(self, state):
		f_gate_logit, v_gate_logit, f_raw_mean, v_raw_mean, f_std, v_std = self._dists(state)
		f_p = torch.sigmoid(f_gate_logit)
		v_p = torch.sigmoid(v_gate_logit)

		f_fire = torch.bernoulli(f_p)
		v_fire = torch.bernoulli(v_p)

		f_mag = torch.sigmoid(Normal(f_raw_mean, f_std).sample())
		v_mag = torch.sigmoid(Normal(v_raw_mean, v_std).sample())

		return torch.cat([f_fire * f_mag, v_fire * v_mag], dim=-1)
	

	def forward(self, state: torch.Tensor, return_dist=False):
		f_gate_logit, v_gate_logit, f_raw_mean, v_raw_mean, f_std, v_std = self._dists(state)
		f_p = torch.sigmoid(f_gate_logit) 
		v_p = torch.sigmoid(v_gate_logit) 
		
		f_mean_mag = torch.sigmoid(f_raw_mean)
		v_mean_mag = torch.sigmoid(v_raw_mean)

		f_action = (f_p > 0.5).float() * f_mean_mag
		v_action = (v_p > 0.5).float() * v_mean_mag
		actions = torch.cat([f_action, v_action], dim=-1)
		gate_lp = None
		mag_lp = None
		log_prob = None
 
		if return_dist:
			return actions, log_prob, (gate_lp, mag_lp), (f_p, v_p, f_raw_mean, v_raw_mean, f_std, v_std)
 
		return actions, log_prob, (gate_lp, mag_lp)


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



class BC:
	def __init__(
		self,
		
		use_smoothing: bool = False,

		actor: nn.Module = None,
		actor_optimizer: torch.optim.Optimizer = None,
		actor_scheduler: torch.optim.lr_scheduler._LRScheduler = None,

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
		actions, _, _ = self.actor(state)
		return actions.cpu().data.numpy().flatten()


	def train(self, batch: TensorBatch) -> Tuple[Dict[str, float], Dict[str, float]]:
		self.total_it += 1

		# Sample replay buffer 
		state, action, reward, next_state, next_action, done, mortality = TrainerUtils._unpack_batch(batch, self.device)
		

		# Compute actor loss
		self.actor.train() # Dropout ACTIVE for the network we are updating
		
		# Sample actions from the stochastic actor
		log_prob, components = self.actor.log_prob_of_action(state, action, return_components=True) # log probability of the action taken by the doctor under the current policy
		
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
			_, _, _, (f_p, v_p, f_raw_mean, v_raw_mean, f_std, v_std) = self.actor(
				state, return_dist=True
			)
			f_mag = torch.sigmoid(f_raw_mean)
			v_mag = torch.sigmoid(v_raw_mean)

		# fired refers to clinician dose binary
		f_fired_flat, v_fired_flat = components["fluid_fired"], components["vaso_fired"]  # (batch,)
		f_fired, v_fired = f_fired_flat.unsqueeze(-1), v_fired_flat.unsqueeze(-1)          # (batch,1), for MSE only

		# f_p is the gate logits (<0.5 = no dose, >0.5 = dose)
		# gate accuracy — did we predict "dose given" correctly?
		fluid_gate_acc = ((f_p > 0.5).float() == f_fired).float().mean().item()
		vaso_gate_acc  = ((v_p > 0.5).float() == v_fired).float().mean().item()

		# magnitude MSE computed ONLY on the subset that actually got a dose —
		# this is now the honest number, since previously mixing in the zeros
		# made the "mse" mostly measure how well you predicted zero.
		if f_fired_flat.sum() > 0:
			fluid_mse = (F.mse_loss(f_mag, action[:, 0:1], reduction='none') * f_fired).sum() / f_fired.sum()
			fluid_mag_loss = -(f_fired_flat * components["fluid_mag_lp"]).sum().item() / f_fired.sum().item()
			fluid_bias = ((f_mag - action[:, 0:1]) * f_fired).sum() / f_fired.sum()
			fluid_mag_pred_nonzero = (f_mag * f_fired).sum() / f_fired.sum()
			fluid_mag_pred_all = (f_mag * f_fired).mean()  # mean over all samples, including zeros
		else:
			fluid_mse = torch.tensor(float('nan'))
			fluid_mag_loss = torch.tensor(float('nan'))
			fluid_bias = torch.tensor(float('nan'))
			fluid_mag_pred_nonzero = torch.tensor(float('nan'))
			fluid_mag_pred_all = torch.tensor(float('nan'))

		if v_fired_flat.sum() > 0:
			vaso_mse = (F.mse_loss(v_mag, action[:, 1:2], reduction='none') * v_fired).sum() / v_fired.sum()
			vaso_mag_loss = -(v_fired_flat * components["vaso_mag_lp"]).sum().item() / v_fired.sum().item()
			vaso_bias = ((v_mag - action[:, 1:2]) * v_fired).sum() / v_fired.sum()
			vaso_mag_pred_nonzero = (v_mag * v_fired).sum() / v_fired.sum()
			vaso_mag_pred_all = (v_mag * v_fired).mean()  # mean over all samples, including zeros
		else:
			vaso_mse = torch.tensor(float('nan'))
			vaso_mag_loss = torch.tensor(float('nan'))
			vaso_bias = torch.tensor(float('nan'))
			vaso_mag_pred_nonzero = torch.tensor(float('nan'))
			vaso_mag_pred_all = torch.tensor(float('nan'))

		losses["fluid_gate_acc"] = fluid_gate_acc
		losses["vaso_gate_acc"] = vaso_gate_acc
		losses["fluid_mag_pred_nonzero"] = fluid_mag_pred_nonzero.item()
		losses["vaso_mag_pred_nonzero"] = vaso_mag_pred_nonzero.item()
		losses["fluid_mag_pred_all"] = fluid_mag_pred_all.item()
		losses["vaso_mag_pred_all"] = vaso_mag_pred_all.item()
		losses["fluid_std_mean"] = f_std.mean().item()
		losses["vaso_std_mean"] = v_std.mean().item()
		losses["fluid_mse_nonzero"] = fluid_mse.item()
		losses["vaso_mse_nonzero"] = vaso_mse.item()	
		losses["fluid_mag_loss"] = fluid_mag_loss if isinstance(fluid_mag_loss, float) else fluid_mag_loss.item()
		losses["vaso_mag_loss"] = vaso_mag_loss if isinstance(vaso_mag_loss, float) else vaso_mag_loss.item()
		losses["fluid_bias_nonzero"] = fluid_bias.item()
		losses["vaso_bias_nonzero"] = vaso_bias.item()
		# diagnostics — don't affect the gradient, just logging
		losses["fluid_gate_loss"] = -components["fluid_gate_lp"].mean().item()
		losses["vaso_gate_loss"] = -components["vaso_gate_lp"].mean().item()


		# return log_dict, grad_norm
		return losses, None, grad_norm


	def actor_forward_for_sarsa(self, batch: TensorBatch) -> torch.Tensor:
		state, _, _, _, _, _, _ = TrainerUtils._unpack_batch(batch, self.device)

		self.actor.eval()  # Set actor to eval mode to disable Dropout
		with torch.no_grad():
			new_actions, _, _ = self.actor(state)  # Get the "best" action without noise or log probabilities
		
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
			log_prob, components = self.actor.log_prob_of_action(state, action, return_components=True) # log probability of the action taken by the doctor under the current policy
			
			actor_loss = -log_prob.mean() # negative log likelihood of the action taken by the doctor under the current policy
			losses = {
				"actor_loss": actor_loss.item(),
			}
			
			# ADD THIS LINE:
			neuron_stats = self.actor.get_neuron_stats(state)
	
			_, _, _, (f_p, v_p, f_raw_mean, v_raw_mean, f_std, v_std) = self.actor(
				state, return_dist=True
			)
			f_mag = torch.sigmoid(f_raw_mean)
			v_mag = torch.sigmoid(v_raw_mean)

			f_fired_flat, v_fired_flat = components["fluid_fired"], components["vaso_fired"]  # (batch,)
			f_fired, v_fired = f_fired_flat.unsqueeze(-1), v_fired_flat.unsqueeze(-1)          # (batch,1), for MSE only

			# gate accuracy — did we predict "dose given" correctly?
			fluid_gate_acc = ((f_p > 0.5).float() == f_fired).float().mean().item()
			vaso_gate_acc  = ((v_p > 0.5).float() == v_fired).float().mean().item()

			# magnitude MSE computed ONLY on the subset that actually got a dose —
			# this is now the honest number, since previously mixing in the zeros
			# made the "mse" mostly measure how well you predicted zero.
			if f_fired_flat.sum() > 0:
				fluid_mse = (F.mse_loss(f_mag, action[:, 0:1], reduction='none') * f_fired).sum() / f_fired.sum()
				fluid_mag_loss = -(f_fired_flat * components["fluid_mag_lp"]).sum().item() / f_fired.sum().item()
				fluid_bias = ((f_mag - action[:, 0:1]) * f_fired).sum() / f_fired.sum()
				fluid_mag_pred_nonzero = (f_mag * f_fired).sum() / f_fired.sum()
				fluid_mag_pred_all = (f_mag * f_fired).mean()
			else:
				fluid_mse = torch.tensor(float('nan'))
				fluid_mag_loss = torch.tensor(float('nan'))
				fluid_bias = torch.tensor(float('nan'))
				fluid_mag_pred_nonzero = torch.tensor(float('nan'))
				fluid_mag_pred_all = torch.tensor(float('nan'))

			if v_fired_flat.sum() > 0:
				vaso_mse = (F.mse_loss(v_mag, action[:, 1:2], reduction='none') * v_fired).sum() / v_fired.sum()
				vaso_mag_loss = -(v_fired_flat * components["vaso_mag_lp"]).sum().item() / v_fired.sum().item()
				vaso_bias = ((v_mag - action[:, 1:2]) * v_fired).sum() / v_fired.sum()
				vaso_mag_pred_nonzero = (v_mag * v_fired).sum() / v_fired.sum()
				vaso_mag_pred_all = (v_mag * v_fired).mean()
			else:
				vaso_mse = torch.tensor(float('nan'))
				vaso_mag_loss = torch.tensor(float('nan'))
				vaso_bias = torch.tensor(float('nan'))
				vaso_mag_pred_nonzero = torch.tensor(float('nan'))
				vaso_mag_pred_all = torch.tensor(float('nan'))

			losses["fluid_gate_acc"] = fluid_gate_acc
			losses["vaso_gate_acc"] = vaso_gate_acc
			losses["fluid_mag_pred_nonzero"] = fluid_mag_pred_nonzero.item()
			losses["vaso_mag_pred_nonzero"] = vaso_mag_pred_nonzero.item()
			losses["fluid_mag_pred_all"] = fluid_mag_pred_all.item()
			losses["vaso_mag_pred_all"] = vaso_mag_pred_all.item()
			losses["fluid_std_mean"] = f_std.mean().item()
			losses["vaso_std_mean"] = v_std.mean().item()
			losses["fluid_mse_nonzero"] = fluid_mse.item()
			losses["vaso_mse_nonzero"] = vaso_mse.item()	
			losses["fluid_mag_loss"] = fluid_mag_loss if isinstance(fluid_mag_loss, float) else fluid_mag_loss.item()
			losses["vaso_mag_loss"] = vaso_mag_loss if isinstance(vaso_mag_loss, float) else vaso_mag_loss.item()
			losses["fluid_bias_nonzero"] = fluid_bias.item()
			losses["vaso_bias_nonzero"] = vaso_bias.item() 
			# diagnostics — don't affect the gradient, just logging
			losses["fluid_gate_loss"] = -components["fluid_gate_lp"].mean().item()
			losses["vaso_gate_loss"] = -components["vaso_gate_lp"].mean().item()
		
		# return log_dict, action, pi
		return losses, None, neuron_stats

	
	def state_dict(self) -> Dict[str, torch.Tensor]:
		return {
			"actor": self.actor.state_dict(),
			"actor_optimizer": self.actor_optimizer.state_dict(),			
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
		hidden_dim=hyperparams['hidden_dim'], 
		dropout=hyperparams['dropout']
	).to(config_manager.training_config['device'])
	
	actor_optimizer = torch.optim.AdamW(
		actor.parameters(), 
		lr=learning_rate_actor,
		weight_decay=weight_decay
		)
	
	actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		actor_optimizer, 
		T_max=hyperparams['total_steps'], 
		eta_min=learning_rate_actor * .10
	)
	
	
	trainer = BC(               # LIST OF PARAMETERS NEEDS TO BE UPDATED
		actor=actor,
		actor_optimizer=actor_optimizer,
		actor_scheduler=actor_scheduler,
		device=config_manager.training_config['device'],
		action_mode=config_manager.action_mode,
		grad_clip=hyperparams['grad_clip'],
	)

	return trainer
