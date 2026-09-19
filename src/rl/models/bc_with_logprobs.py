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

UPDATE (v3): this version is built directly from the ORIGINAL, confirmed-
clean baseline (v2 -- no entropy-tuning dual variables), NOT from the
intermediate version that ported CQL's entropy-tuning machinery wholesale.
That port was found to actively cause vaso's magnitude std to hit its
clamp CEILING in ~45-50% of states -- a regression, not a fix, since BC's
pure-MLE objective never had the Jensen's-inequality-driven variance-
collapse problem that entropy-tuning was designed to solve for CQL's
stochastic policy-gradient loss. The entropy-tuning dual variables
(log_alpha_mag_fluid/vaso, target_entropy_mag_*) are therefore REMOVED
entirely in this version.

Two other additions from that same work ARE kept, since they address
genuine, distinct problems and don't share entropy-tuning's failure mode:

  1. Mean-saturation regularization: weighted by GATE UNCERTAINTY
     (4*p*(1-p), broadly active near p=0.5), not masked to only "fired"
     states the way the entropy term was -- doesn't share entropy-tuning's
     sparse-signal fragility for rare-firing actions like vaso.

  2. Fluid gate class-weighting: addresses a real, distinct problem (BCE
     class imbalance from fluid's ~90/10 fire-rate split) that was
     already present even in the original v2 baseline -- confirmed via
     evaluate_bc_final_seeds.check_gate_calibration_by_seed, which showed
     BC assigning only ~0.19-0.27 probability to the real "no fluid"
     decision at states where that was correct (vs. vaso's well-
     calibrated ~0.94-0.98 in the analogous case), contributing a median
     ~33-39% (p90 ~73-82%) of each trajectory's total gate-related
     log-likelihood magnitude -- large enough to matter for WIS. Applied
     to the LOSS only, never to log_prob_of_action's reported likelihood,
     which must stay a genuine, correctly-specified probability.
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
		fluid_gate_hidden: int = 0,      # 0 = single linear layer (original). >0 = adds one hidden
			# layer of this width (ReLU) before the final fluid gate logit -- gives the fluid
			# gate more capacity to separate ambiguous zero/nonzero states, independent of the
			# shared trunk. Vaso's gate is untouched (already well-calibrated).
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

		if fluid_gate_hidden > 0:
			self.fluid_gate = nn.Sequential(
				nn.Linear(hidden_dim, fluid_gate_hidden),
				nn.ReLU(),
				nn.Linear(fluid_gate_hidden, 1),
			)
		else:
			self.fluid_gate = nn.Linear(hidden_dim, 1)
		self.vaso_gate = nn.Linear(hidden_dim, 1)

		self.fluid_log_std = nn.Linear(hidden_dim, 1)
		self.vaso_log_std = nn.Linear(hidden_dim, 1)

		# --- post-hoc gate calibration (Platt scaling), fit AFTER normal
		# training via calibrate_fluid_gate() below. Identity by default
		# (scale=1, bias=0) -- has NO EFFECT unless calibrate_fluid_gate()
		# has been called. Stored as buffers, not nn.Parameters, since
		# they're fit by a small separate procedure, not by the main
		# optimizer/backward pass used for everything else.
		self.register_buffer("fluid_gate_calib_scale", torch.tensor(1.0))
		self.register_buffer("fluid_gate_calib_bias", torch.tensor(0.0))
 
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
		if isinstance(self.fluid_gate, nn.Linear):
			nn.init.zeros_(self.fluid_gate.weight)
			self.fluid_gate.bias.data.fill_(0.78)    # sigmoid(0.78) ≈ 0.686 fluid nonzero rate
		else:
			# MLP gate head: zero-init the FINAL layer only (so it starts at
			# the same base-rate prior), let the hidden layer use its default init
			nn.init.zeros_(self.fluid_gate[-1].weight)
			self.fluid_gate[-1].bias.data.fill_(0.78)
		nn.init.zeros_(self.vaso_gate.weight)
		self.vaso_gate.bias.data.fill_(-2.98)    # sigmoid(-2.98) ≈ 0.048 vaso nonzero rate


	def _dists(self, state, apply_calibration=False):
		z = self.trunk(state)

		f_gate_logit = self.fluid_gate(z)
		v_gate_logit = self.vaso_gate(z)

		if apply_calibration:
			f_gate_logit = f_gate_logit * self.fluid_gate_calib_scale + self.fluid_gate_calib_bias

		f_raw_mean = self.fluid_mean(z)
		v_raw_mean = self.vaso_mean(z)

		f_log_std = torch.clamp(self.fluid_log_std(z), self.log_std_min, self.log_std_max)
		v_log_std = torch.clamp(self.vaso_log_std(z), self.log_std_min, self.log_std_max)
		f_std = f_log_std.exp()
		v_std = v_log_std.exp()

		return f_gate_logit, v_gate_logit, f_raw_mean, v_raw_mean, f_std, v_std


	@torch.no_grad()
	def calibrate_fluid_gate(self, states, actions, n_steps=300, lr=0.05, zero_threshold=1e-4):
		"""
		Post-hoc Platt scaling: fits fluid_gate_calib_scale/bias (a single
		scale + bias applied to the raw fluid gate logit) by minimizing BCE
		against REAL fired/not-fired labels on a held-out set (val, not
		train -- same reasoning as any other data-derived calibration
		constant in this project). Call this ONCE, after normal training is
		otherwise complete, on a frozen model. Does not touch trunk/gate
		weights or any other parameter -- purely a post-hoc correction.

		states: (n, state_dim) held-out states
		actions: (n, 2) held-out real actions (for the true fluid fired/not label)
		"""
		self.eval()
		with torch.enable_grad():
			f_gate_logit_raw = self.fluid_gate(self.trunk(states)).detach()  # frozen, no grad to the network
			fired = (actions[:, 0:1] > zero_threshold).float()

			scale = torch.tensor(1.0, requires_grad=True, device=states.device)
			bias = torch.tensor(0.0, requires_grad=True, device=states.device)
			optimizer = torch.optim.Adam([scale, bias], lr=lr)

			for _ in range(n_steps):
				optimizer.zero_grad()
				calibrated_logit = f_gate_logit_raw * scale + bias
				loss = F.binary_cross_entropy_with_logits(calibrated_logit, fired)
				loss.backward()
				optimizer.step()

		self.fluid_gate_calib_scale.copy_(scale.detach())
		self.fluid_gate_calib_bias.copy_(bias.detach())
		print(f"Fitted fluid gate calibration: scale={scale.item():.4f}, bias={bias.item():.4f}, "
			  f"final BCE={loss.item():.4f}")
	
	
	def sample(self, state, apply_calibration=False):
		f_gate_logit, v_gate_logit, f_raw_mean, v_raw_mean, f_std, v_std = self._dists(state, apply_calibration=apply_calibration)
		f_p = torch.sigmoid(f_gate_logit)
		v_p = torch.sigmoid(v_gate_logit)

		f_fire = torch.bernoulli(f_p)
		v_fire = torch.bernoulli(v_p)

		f_mag = torch.sigmoid(Normal(f_raw_mean, f_std).sample())
		v_mag = torch.sigmoid(Normal(v_raw_mean, v_std).sample())

		return torch.cat([f_fire * f_mag, v_fire * v_mag], dim=-1)
	

	def forward(self, state: torch.Tensor, return_dist=False, apply_calibration=False):
		f_gate_logit, v_gate_logit, f_raw_mean, v_raw_mean, f_std, v_std = self._dists(state, apply_calibration=apply_calibration)
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


	def log_prob_of_action(self, state: torch.Tensor, action: torch.Tensor, return_components=False,
							apply_calibration=False) -> torch.Tensor:
		"""
		Hurdle log-likelihood of a GIVEN (dataset) action. Used for WIS eval
		and/or a CQL+BC regularization term. (The original version of this
		actor referenced a nonexistent self.trunk here — fixed now that
		there's a single real trunk to call.)

		NOTE: this is the TRUE, correctly-specified log-likelihood -- it is
		NEVER reweighted by the training-time class weight/focal loss (see
		BC.train()'s separate, class-weighted LOSS, which uses these same
		per-sample components but is a distinct quantity used only for the
		gradient, not reported here).

		apply_calibration: if True, applies the post-hoc Platt-scaling
		correction fit by calibrate_fluid_gate() (identity/no-op if that
		method has never been called on this actor). Default False, so
		existing behavior (including any already-trained/saved models) is
		completely unaffected unless explicitly requested. Set True here
		specifically when computing the density used for WIS, once you've
		fit and validated a calibration.
		"""
		f_gate_logit, v_gate_logit, f_raw_mean, v_raw_mean, f_std, v_std = self._dists(state, apply_calibration=apply_calibration)
 
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
				"f_p": torch.sigmoid(f_gate_logit).squeeze(-1),
				"v_p": torch.sigmoid(v_gate_logit).squeeze(-1),
				"f_raw_mean": f_raw_mean.squeeze(-1),
				"v_raw_mean": v_raw_mean.squeeze(-1),
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

		fluid_mean_reg_scale: float = 0.05,
		vaso_mean_reg_scale: float = 0.05,

		fluid_gate_neg_class_weight: float = 6.0,   # updated default -- see tuning history in module docstring

		fluid_gate_loss_type: str = "class_weighted",  # "class_weighted" or "focal"
		fluid_gate_focal_gamma: float = 2.0,           # only used if fluid_gate_loss_type=="focal"

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

		self.fluid_mean_reg_scale = fluid_mean_reg_scale
		self.vaso_mean_reg_scale = vaso_mean_reg_scale
		self.fluid_gate_neg_class_weight = fluid_gate_neg_class_weight
		self.fluid_gate_loss_type = fluid_gate_loss_type
		self.fluid_gate_focal_gamma = fluid_gate_focal_gamma

		# for calculating moving average of gradient clipping binary
		self.actor_clip_fraction_ema = 0.0
		self.ema_alpha = ema_alpha # Closer to 1.0 means more smoothing
		
		self.total_it = 0

		self.utils = TrainerUtils()

		
	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		actions, _, _ = self.actor(state)
		return actions.cpu().data.numpy().flatten()


	def _compute_fluid_gate_weight(self, f_fired, fluid_gate_lp):
		"""
		Shared by train()/evaluate(). Computes the per-sample weight applied
		to the fluid gate's log-probability in the LOSS (never in the
		reported log_prob itself -- see module docstring).

		fluid_gate_loss_type == "class_weighted": weight = 1 if fired,
			else fluid_gate_neg_class_weight (uniform reweighting of the
			whole negative class -- the version validated across 5 seeds).

		fluid_gate_loss_type == "focal": additionally multiplies by
			(1 - p_t)^gamma, where p_t = exp(fluid_gate_lp) is the
			probability BC assigns to the TRUE label for that sample.
			This down-weights already-easy/correct predictions (p_t close
			to 1) and concentrates gradient on ambiguous cases, rather
			than reweighting the whole class uniformly regardless of how
			easy any individual example already is.
		"""
		class_weight = torch.where(
			f_fired.bool(),
			torch.ones_like(f_fired),
			self.fluid_gate_neg_class_weight * torch.ones_like(f_fired),
		)

		if self.fluid_gate_loss_type == "class_weighted":
			return class_weight
		elif self.fluid_gate_loss_type == "focal":
			p_t = fluid_gate_lp.exp().clamp(1e-6, 1 - 1e-6)
			focal_term = (1 - p_t).pow(self.fluid_gate_focal_gamma)
			return class_weight * focal_term
		else:
			raise ValueError(f"Unknown fluid_gate_loss_type='{self.fluid_gate_loss_type}', "
							  f"use 'class_weighted' or 'focal'")


	def train(self, batch: TensorBatch) -> Tuple[Dict[str, float], Dict[str, float]]:
		self.total_it += 1

		state, action, reward, next_state, next_action, done, mortality = TrainerUtils._unpack_batch(batch, self.device)
		

		self.actor.train() # Dropout ACTIVE for the network we are updating
		
		log_prob, components = self.actor.log_prob_of_action(state, action, return_components=True)
		
		nll_loss = -log_prob.mean() # TRUE, unweighted -- monitoring only

		f_p = components["f_p"]
		v_p = components["v_p"]
		f_raw_mean = components["f_raw_mean"]
		v_raw_mean = components["v_raw_mean"]

		f_gate_uncertainty_weight = 4 * f_p * (1 - f_p)  # smooth parabola: 1 at p=0.5, 0 at p=0 or 1
		v_gate_uncertainty_weight = 4 * v_p * (1 - v_p)
		f_mean_saturation_penalty = (f_gate_uncertainty_weight * f_raw_mean.pow(2)).mean()
		v_mean_saturation_penalty = (v_gate_uncertainty_weight * v_raw_mean.pow(2)).mean()

		f_fired = components["fluid_fired"]
		v_fired = components["vaso_fired"]

		fluid_gate_weight = self._compute_fluid_gate_weight(f_fired, components["fluid_gate_lp"])
		weighted_nll_loss = -(
			(fluid_gate_weight * components["fluid_gate_lp"]).mean()
			+ (f_fired * components["fluid_mag_lp"]).mean()
			+ components["vaso_gate_lp"].mean()
			+ (v_fired * components["vaso_mag_lp"]).mean()
		)

		actor_loss = (
			weighted_nll_loss
			+ self.fluid_mean_reg_scale * f_mean_saturation_penalty
			+ self.vaso_mean_reg_scale * v_mean_saturation_penalty
		)

		losses = {
			"actor_loss": actor_loss.item(),
			"nll_loss": nll_loss.item(),
			"weighted_nll_loss": weighted_nll_loss.item(),
			"f_mean_saturation_penalty": f_mean_saturation_penalty.item(),
			"v_mean_saturation_penalty": v_mean_saturation_penalty.item(),
		}

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		
		total_actor_grad_norm = self._get_grad_norm(self.actor)

		grad_norm = {}
		grad_norm['actor_grad_norm'] = total_actor_grad_norm

		if self.grad_clip is not None:				
			grad_norm['actor_was_clipped'] = 1.0 if total_actor_grad_norm > self.grad_clip else 0.0

			self.actor_clip_fraction_ema = self.ema_alpha * self.actor_clip_fraction_ema + (1 - self.ema_alpha) * grad_norm['actor_was_clipped']
			grad_norm['actor_clip_fraction_ema'] = self.actor_clip_fraction_ema

			torch.nn.utils.clip_grad_norm_(
				self.actor.parameters(), 
				self.grad_clip
			)
		self.actor_optimizer.step()
		if self.actor_scheduler is not None:
			self.actor_scheduler.step()
			losses.update(lr_actor=self.actor_optimizer.param_groups[0]['lr'])
		
		self.actor.eval()
		with torch.no_grad():
			_, _, _, (f_p, v_p, f_raw_mean, v_raw_mean, f_std, v_std) = self.actor(
				state, return_dist=True
			)
			f_mag = torch.sigmoid(f_raw_mean)
			v_mag = torch.sigmoid(v_raw_mean)

		f_fired_flat, v_fired_flat = components["fluid_fired"], components["vaso_fired"]
		f_fired, v_fired = f_fired_flat.unsqueeze(-1), v_fired_flat.unsqueeze(-1)

		fluid_gate_acc = ((f_p > 0.5).float() == f_fired).float().mean().item()
		vaso_gate_acc  = ((v_p > 0.5).float() == v_fired).float().mean().item()

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
		losses["fluid_gate_loss"] = -components["fluid_gate_lp"].mean().item()
		losses["vaso_gate_loss"] = -components["vaso_gate_lp"].mean().item()


		return losses, None, grad_norm


	def actor_forward_for_sarsa(self, batch: TensorBatch) -> torch.Tensor:
		state, _, _, _, _, _, _ = TrainerUtils._unpack_batch(batch, self.device)

		self.actor.eval()
		with torch.no_grad():
			new_actions, _, _ = self.actor(state)
		
		return new_actions


	def _get_grad_norm(self, model: nn.Module) -> float:
		total_norm = 0
		for p in model.parameters():
			if p.grad is not None:
				param_norm = p.grad.detach().data.norm(2)
				total_norm += param_norm.item() ** 2
		return total_norm ** 0.5
	

	def evaluate(self, batch: TensorBatch) -> Dict[str, float]:
		self.actor.eval()
		
		state, action, reward, next_state, next_action, done, mortality = TrainerUtils._unpack_batch(batch, self.device)
		
		with torch.no_grad():
			log_prob, components = self.actor.log_prob_of_action(state, action, return_components=True)
			
			nll_loss = -log_prob.mean() # TRUE, unweighted -- monitoring only

			f_p = components["f_p"]
			v_p = components["v_p"]
			f_raw_mean = components["f_raw_mean"]
			v_raw_mean = components["v_raw_mean"]

			f_gate_uncertainty_weight = 4 * f_p * (1 - f_p)
			v_gate_uncertainty_weight = 4 * v_p * (1 - v_p)
			f_mean_saturation_penalty = (f_gate_uncertainty_weight * f_raw_mean.pow(2)).mean()
			v_mean_saturation_penalty = (v_gate_uncertainty_weight * v_raw_mean.pow(2)).mean()

			f_fired_c = components["fluid_fired"]
			v_fired_c = components["vaso_fired"]

			fluid_gate_weight = self._compute_fluid_gate_weight(f_fired_c, components["fluid_gate_lp"])
			weighted_nll_loss = -(
				(fluid_gate_weight * components["fluid_gate_lp"]).mean()
				+ (f_fired_c * components["fluid_mag_lp"]).mean()
				+ components["vaso_gate_lp"].mean()
				+ (v_fired_c * components["vaso_mag_lp"]).mean()
			)

			actor_loss = (
				weighted_nll_loss
				+ self.fluid_mean_reg_scale * f_mean_saturation_penalty
				+ self.vaso_mean_reg_scale * v_mean_saturation_penalty
			)

			losses = {
				"actor_loss": actor_loss.item(),
				"nll_loss": nll_loss.item(),
				"weighted_nll_loss": weighted_nll_loss.item(),
				"f_mean_saturation_penalty": f_mean_saturation_penalty.item(),
				"v_mean_saturation_penalty": v_mean_saturation_penalty.item(),
			}
			
			neuron_stats = self.actor.get_neuron_stats(state)
	
			_, _, _, (f_p, v_p, f_raw_mean, v_raw_mean, f_std, v_std) = self.actor(
				state, return_dist=True
			)
			f_mag = torch.sigmoid(f_raw_mean)
			v_mag = torch.sigmoid(v_raw_mean)

			f_fired_flat, v_fired_flat = components["fluid_fired"], components["vaso_fired"]
			f_fired, v_fired = f_fired_flat.unsqueeze(-1), v_fired_flat.unsqueeze(-1)

			fluid_gate_acc = ((f_p > 0.5).float() == f_fired).float().mean().item()
			vaso_gate_acc  = ((v_p > 0.5).float() == v_fired).float().mean().item()

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
			losses["fluid_gate_loss"] = -components["fluid_gate_lp"].mean().item()
			losses["vaso_gate_loss"] = -components["vaso_gate_lp"].mean().item()
		
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
	weight_decay = float(hyperparams.get('weight_decay', 0.01))
	device = config_manager.training_config['device']

	
	actor = Actor(
		state_dim=state_dim, 
		hidden_dim=hyperparams['hidden_dim'], 
		dropout=hyperparams['dropout'],
		fluid_gate_hidden=hyperparams.get('fluid_gate_hidden', 0),
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
	
	
	trainer = BC(
		actor=actor,
		actor_optimizer=actor_optimizer,
		actor_scheduler=actor_scheduler,
		device=config_manager.training_config['device'],
		action_mode=config_manager.action_mode,
		grad_clip=hyperparams['grad_clip'],

		fluid_mean_reg_scale=hyperparams.get('fluid_mean_reg_scale', 0.05),
		vaso_mean_reg_scale=hyperparams.get('vaso_mean_reg_scale', 0.05),
		fluid_gate_neg_class_weight=hyperparams.get('fluid_gate_neg_class_weight', 6.0),
		fluid_gate_loss_type=hyperparams.get('fluid_gate_loss_type', 'class_weighted'),
		fluid_gate_focal_gamma=hyperparams.get('fluid_gate_focal_gamma', 2.0),

	)

	return trainer