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
    def __init__(self, state_dim, action_dim, dropout=0.1, hidden_dim=512, negative_slope=0.1):
        
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
    

    @torch.no_grad()
    def get_neuron_stats(self, state, action):
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



class Q_Learner(object):
    def __init__(
        self,
        critic: nn.Module = None,
        critic_optimizer: torch.optim.Optimizer = None,	
        discount: float = 0.99,
        device: str = "CUDA" if torch.cuda.is_available() else "CPU",
        tau: float = 0.005,
        grad_clip: float = 1.0,
        reward_scaling: float = 1.0,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None
    ):
        
        self.total_it = 0

        self.device = device

        self.critic = critic.to(self.device)
        self.critic_optimizer = critic_optimizer
        
        self.critic_target = copy.deepcopy(self.critic)

        self.discount = discount

        self.tau = tau

        self.grad_clip = grad_clip

        self.reward_scaling = reward_scaling

        self.critic_optimizer = critic_optimizer
        self.scheduler = scheduler
        
        # for calculating moving average of gradient clipping binary
        self.clip_fraction_ema = 0.0
        self.ema_alpha = 0.99 # Closer to 1.0 means more smoothing
		
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

        reward = reward * self.reward_scaling

        # compute the target Q value
        treat_survival_as_truncated = True
        if treat_survival_as_truncated:
             # make done 0 where mortality is 0, since we want to treat survival as truncated rather than terminal
            done = done * mortality
        
        not_done = 1 - done
        
        self.critic.train()          # Dropout ACTIVE for the network we are updating
        self.critic_target.eval()
        
        with torch.no_grad():
            target_Q = self.target_q(reward, not_done, next_state, next_action)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # use Huber loss instead
        critic_loss = (F.smooth_l1_loss(current_Q1, target_Q) + F.smooth_l1_loss(current_Q2, target_Q)) / 2.0
        
        log_dict = {"critic_loss": critic_loss.item()}

        # get Q statistics
        log_dict = self.get_q_stats(current_Q1, current_Q2, mortality, log_dict)

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

        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return log_dict, grad_norm
    

    def evaluate(self, batch: TensorBatch) -> Dict[str, float]:
        self.critic.eval()  # Ensure critic is also in eval mode
        
        state, action, reward, next_state, next_action, done, mortality = TrainerUtils._unpack_batch(batch, self.device)
        reward = reward * self.reward_scaling
        not_done = 1 - done

        with torch.no_grad():
            # compute the target Q value
            target_Q = self.target_q(reward, not_done, next_state, next_action)

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            # critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            # use Huber loss instead
            critic_loss = (F.smooth_l1_loss(current_Q1, target_Q) + F.smooth_l1_loss(current_Q2, target_Q)) / 2.0
            
            log_dict = {"critic_loss": critic_loss.item()}
            
            # ADD THIS LINE:
            neuron_stats = self.critic.get_neuron_stats(state, action)
            # Merge stats into your log_dict
            log_dict.update(neuron_stats)

            # get Q statistics
            log_dict = self.get_q_stats(current_Q1, current_Q2, mortality, log_dict)
        
        return log_dict, None, None
	
    
    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "critic": self.critic.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }
    

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


def get_trainer(config_manager, state_dim, action_dim, _, hyperparams):

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
    scheduler = torch.optim.lr_scheduler.StepLR(critic_optimizer, step_size=10, gamma=0.5)

    # trainer = Q_Learner(
    #         critic=critic,
    #         critic_optimizer=critic_optimizer,
    #         discount=config_manager.training_config['discount'],
    #         device=config_manager.training_config['device'],
    #         tau=config_manager.training_config['tau'],	
    #         grad_clip = hyperparams['grad_clip']
    #     )
    
    trainer = Q_Learner(
        critic=critic,
        critic_optimizer=critic_optimizer,
        discount=hyperparams.get('discount', config_manager.training_config['discount']),
        device=device,
        tau=hyperparams.get('tau', config_manager.training_config['tau']),  
        grad_clip=hyperparams['grad_clip'], 
        reward_scaling=hyperparams.get('reward_scaling', 1.0),
        scheduler=scheduler
    )

    return trainer