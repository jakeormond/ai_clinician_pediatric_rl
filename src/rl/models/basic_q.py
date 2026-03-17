import typing
from typing import Dict, List, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .shared_components import TrainerUtils


TensorBatch = List[torch.Tensor]

"""
this is designed for 1. testing that the pipeline works, and 2. testing whether
intermediate rewards help learning
"""

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        
        super(Critic, self).__init__()

        # standardize the incoming embeddings
        self.state_norm = nn.LayerNorm(state_dim)

        # Q1 architecture
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.LayerNorm(256), # Normalize internal activations
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )

        # Q2 architecture
        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )

        # # Q1 architecture
        # self.l1 = nn.Linear(state_dim + action_dim, 256)
        # self.l2 = nn.Linear(256, 256)
        # self.l3 = nn.Linear(256, 1)

		# # Q2 architecture
        # self.l4 = nn.Linear(state_dim + action_dim, 256)
        # self.l5 = nn.Linear(256, 256)
        # self.l6 = nn.Linear(256, 1)

        self.apply(self._init_weights)
        
    
    def _init_weights(self, m):
        # Initialize hidden layers with Xavier/Glorot for stability
        if isinstance(m, nn.Linear):
            # Check if this is the final output layer (it has 1 output)
            if m.out_features == 1:
                # The "Conservative" init: squeeze initial Q-values to near zero
                nn.init.uniform_(m.weight, -3e-3, 3e-3)
                nn.init.uniform_(m.bias, -3e-3, 3e-3)
            else:
                # Hidden layers: Xavier Uniform for gradient stability
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1) # Helps prevent "Dead ReLU" issues early on
        
        elif isinstance(m, nn.LayerNorm):
            # LayerNorm should start as an identity transform
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)    
    
    def forward(self, state, action):

        # normalize states before concatenation
        state = self.state_norm(state)

        sa = torch.cat([state, action], 1)

        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)

        # q1 = F.relu(self.l1(sa))
        # q1 = F.relu(self.l2(q1))
        # q1 = self.l3(q1)

        # q2 = F.relu(self.l4(sa))
        # q2 = F.relu(self.l5(q2))
        # q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = self.q1_net(sa)

        # q1 = F.relu(self.l1(sa))
        # q1 = F.relu(self.l2(q1))
        # q1 = self.l3(q1)
        return q1
	

def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


class Q_Learner(object):
    def __init__(
        self,
        critic: nn.Module = None,
        critic_optimizer: torch.optim.Optimizer = None,	
        discount: float = 0.99,
        device: str = "CUDA" if torch.cuda.is_available() else "CPU",
        tau: float = 0.005,
        grad_clip: float = 1.0,
    ):
        
        self.total_it = 0

        self.device = device

        self.critic = critic.to(self.device)
        self.critic_optimizer = critic_optimizer

        self.critic_target = copy.deepcopy(self.critic)

        self.discount = discount

        self.tau = tau

        self.grad_clip = grad_clip

        # for calculating moving average of gradient clipping binary
        self.clip_fraction_ema = 0.0
        self.ema_alpha = 0.99 # Closer to 1.0 means more smoothing
		
        # copy the critic network to create the target critic
        self.critic_target = copy.deepcopy(self.critic)
		
    def target_q(self, reward, not_done, next_state, next_action): # need to add next action to replay buffer
        with torch.no_grad():            
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = (target_Q1 + target_Q2) / 2.0
            target_Q = reward + not_done * self.discount * target_Q			

        return target_Q
    


    def get_q_stats(self, current_Q1, current_Q2, mortality, log_dict) -> Tuple[float, float]:
         
            # create current_Q as the mean of current_Q1 and current_Q2
            current_Q = (current_Q1 + current_Q2) / 2.0
            mean_Q = current_Q.mean().item()
            variance_Q = current_Q.var().item()
            batch_n = current_Q.shape[0]

            # masks and counts
            survival_mask = (mortality == 0)
            non_survival_mask = (mortality == 1)

            survival_n = survival_mask.sum().item()
            non_survival_n = non_survival_mask.sum().item()
            
            # # calculate the survival_Q and non_survival_Q as the mean of current_Q where mortality is 0 and 1 respectively            
            # survival_Q = current_Q[mortality == 0].mean().item() if (mortality == 0).any() else float('nan')
            # survival_variance_Q = current_Q[mortality == 0].var().item() if (mortality == 0).any() else float('nan')           
            # survival_n = (mortality == 0).sum().item()
            # non_survival_Q = current_Q[mortality == 1].mean().item() if (mortality == 1).any() else float('nan')
            # non_survival_variance_Q = current_Q[mortality == 1].var().item() if (mortality == 1).any() else float('nan')
            # non_survival_n = (mortality == 1).sum().item()
            
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
                "survival_n": survival_n,
                "survival_Q": survival_Q,
                "survival_variance_Q": survival_variance_Q,
                "non_survival_n": non_survival_n,
                "non_survival_Q": non_survival_Q,
                "non_survival_variance_Q": non_survival_variance_Q
            })
            return log_dict


    def train(self, batch: TensorBatch):
        self.total_it += 1

        # sample the replay buffer
        state, action, reward, next_state, next_action, done, mortality = TrainerUtils._unpack_batch(batch, self.device)

        # compute the target Q value
        treat_survival_as_truncated = True
        if treat_survival_as_truncated:
             # make done 0 where mortality is 0, since we want to treat survival as truncated rather than terminal
            done = done * mortality
        
        not_done = 1 - done
        target_Q = self.target_q(reward, not_done, next_state, next_action)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # use Huber loss instead
        critic_loss = F.smooth_l1_loss(current_Q1, target_Q) + F.smooth_l1_loss(current_Q2, target_Q)
        
        log_dict = {"critic_loss": critic_loss.item()}

        # get Q statistics
        log_dict = self.get_q_stats(current_Q1, current_Q2, mortality, log_dict)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # capture gradient norms
        grad_norm = {"raw_grad_norm": get_grad_norm(self.critic)}
        
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

        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return log_dict, grad_norm
    

    def evaluate(self, batch: TensorBatch) -> Dict[str, float]:
        self.critic.eval()  # Ensure critic is also in eval mode
        
        state, action, reward, next_state, next_action, done, mortality = TrainerUtils._unpack_batch(batch, self.device)
        not_done = 1 - done

        with torch.no_grad():
            # compute the target Q value
            target_Q = self.target_q(reward, not_done, next_state, next_action)

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            # critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            # use Huber loss instead
            critic_loss = F.smooth_l1_loss(current_Q1, target_Q) + F.smooth_l1_loss(current_Q2, target_Q)
            
            log_dict = {"critic_loss": critic_loss.item()}
            
            # get Q statistics
            log_dict = self.get_q_stats(current_Q1, current_Q2, mortality, log_dict)
        
        return log_dict, None, None
	
    
    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "critic": self.critic.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }
    

def get_trainer(config_manager, state_dim, action_dim, _, hyperparams):

    learning_rate = float(hyperparams['learning_rate'])
    
    critic = Critic(state_dim, action_dim).to(config_manager.training_config['device'])

    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=learning_rate
    )

    trainer = Q_Learner(
            critic=critic,
            critic_optimizer=critic_optimizer,
            discount=config_manager.training_config['discount'],
            device=config_manager.training_config['device'],
            tau=config_manager.training_config['tau'],	
            grad_clip = hyperparams['grad_clip']
        )

    return trainer