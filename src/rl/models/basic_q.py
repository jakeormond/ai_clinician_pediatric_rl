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
	
    
class Q_Learner(object):
    def __init__(
        self,
        critic: nn.Module = None,
        critic_optimizer: torch.optim.Optimizer = None,	
        discount: float = 0.99,
        device: str = "CUDA" if torch.cuda.is_available() else "CPU",
        tau: float = 0.005,
    ):
        
        self.total_it = 0

        self.device = device

        self.critic = critic.to(self.device)
        self.critic_optimizer = critic_optimizer

        self.critic_target = copy.deepcopy(self.critic)

        self.discount = discount

        self.tau = tau
		
        # copy the critic network to create the target critic
        self.critic_target = copy.deepcopy(self.critic)
		
    def target_q(self, reward, not_done, next_state, next_action): # need to add next action to replay buffer
        with torch.no_grad():            
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q			

        return target_Q
    

    def train(self, batch: TensorBatch):
        self.total_it += 1

        # sample the replay buffer
        state, action, reward, next_state, next_action, done, _ = TrainerUtils._unpack_batch(batch, self.device)
        not_done = 1 - done

        # compute the target Q value
        target_Q = self.target_q(reward, not_done, next_state, next_action)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        
        ########## check critic and target critic outputs ##########
        target_Q1, target_Q2 = self.critic_target(state, action)

        ############## check next state and next action ##############
        next_target1, next_target2 = self.critic_target(next_state, next_action)

        next_1, next_2 = self.critic(next_state, next_action)


        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        log_dict = {"critic_loss": critic_loss.item()}

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return log_dict
    

    def evaluate(self, batch: TensorBatch) -> Dict[str, float]:
        self.critic.eval()  # Ensure critic is also in eval mode
        
        state, action, reward, next_state, next_action, done, mortality = TrainerUtils._unpack_batch(batch, self.device)
        not_done = 1 - done

        with torch.no_grad():
            # compute the target Q value
            target_Q = self.target_q(reward, not_done, next_state, next_action)

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # create current_Q as the mean of current_Q1 and current_Q2
            current_Q = (current_Q1 + current_Q2) / 2.0
            # calculate the survival_Q and non_survival_Q as the mean of current_Q where mortality is 0 and 1 respectively
            survival_Q = current_Q[mortality == 0].mean().item() if (mortality == 0).any() else float('nan')
            non_survival_Q = current_Q[mortality == 1].mean().item() if (mortality == 1).any() else float('nan')
            
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            log_dict = {"critic_loss": critic_loss.item(), "survival_Q": survival_Q, "non_survival_Q": non_survival_Q}
        
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
        )

    return trainer