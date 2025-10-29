from typing import Dict, List, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...rl.models.shared_components import TrainerUtils


TensorBatch = List[torch.Tensor]


class DynamicDeepHitMulti(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_time_bins: int,
        num_risks: int,
        dropout: float = 0.3,
        rnn_type: str = "GRU",
    ):
        """
        Args:
            input_dim: number of covariates per time step
            hidden_dim: RNN hidden size
            num_layers: number of RNN layers
            num_time_bins: number of discrete time bins
            num_risks: number of competing risks (e.g. 2 for death, discharge)
            dropout: dropout rate
            rnn_type: 'GRU' or 'LSTM'
        """
        super().__init__()

        # --- Temporal encoder ---
        if rnn_type.upper() == "LSTM":
            self.rnn = nn.LSTM(
                input_size=feature_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            self.rnn = nn.GRU(
                input_size=feature_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

        # --- Event-specific heads ---
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_time_bins)
            )
            for _ in range(num_risks)
        ])

        self.num_risks = num_risks

        def forward(self, x, mask=None):
            """
            Args:
                x: [batch, seq_len, input_dim]
                mask: [batch, seq_len] with 1 for valid timesteps, 0 for padding
            Returns:
                out: list of [batch, num_time_bins] tensors, one per risk
            """
            # --- Pack padded sequences if mask provided ---
            if mask is not None:
                lengths = mask.sum(dim=1).cpu()
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths, batch_first=True, enforce_sorted=False
                )
                h_packed, _ = self.rnn(packed)
                h, _ = nn.utils.rnn.pad_packed_sequence(h_packed, batch_first=True)
            else:
                h, _ = self.rnn(x)

            # --- Extract last valid hidden state per sequence ---
            if mask is not None:
                idx = (mask.sum(dim=1) - 1).long().unsqueeze(1).unsqueeze(2)
                idx = idx.expand(-1, 1, h.size(2))
                h_last = h.gather(1, idx).squeeze(1)
            else:
                h_last = h[:, -1, :]

            # --- Event-specific hazard outputs ---
            outputs = []
            for head in self.heads:
                logits = head(h_last)             # [batch, num_time_bins]
                probs = F.softmax(logits, dim=-1) # per-risk discrete-time probabilities
                outputs.append(probs)

            return outputs  # list of tensors, len=num_risks
        


class DynamicDeepHitTrainer:
    def __init__(
        self,
        model,
        optimizer,
        device,
        num_risks,
        num_time_bins,
        loss_fn=None,
        grad_clip=None,
        scheduler=None,
        verbose=True,
    ):
        """
        Args:
            model: DynamicDeepHitMulti instance
            optimizer: torch.optim optimizer
            device: 'cuda' or 'cpu'
            num_risks: number of competing risks
            num_time_bins: discretized time horizon
            loss_fn: custom loss function (optional)
            grad_clip: gradient clipping value (optional)
            scheduler: lr scheduler (optional)
            verbose: print progress
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.num_risks = num_risks
        self.num_time_bins = num_time_bins
        self.loss_fn = loss_fn
        self.grad_clip = grad_clip
        self.scheduler = scheduler
        self.verbose = verbose

        self.time_bins = self.make_mixed_resolution_time_bins(
            max_time_hours=3000,
            num_bins=num_time_bins
        )


    def train_epoch(self, batch: TensorBatch):

        pass




    def get_trainer(config_manager, feature_dim, hyperparams):

        learning_rate = float(hyperparams['learning_rate'])

        dynamic_deephit_model = DynamicDeepHitMulti(feature_dim, action_dim).to(config_manager.training_config['device'])

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
    

def make_mixed_resolution_time_bins(max_time_hours=3000, base=1.2):
    """
    Create non-uniform time bins spaced approximately logarithmically.

    Args:
        max_time_hours: maximum survival time in hours
        num_bins: total number of discrete time bins
        base: log scale growth rate (1.1–1.3 typical)
    Returns:
        bin_edges: np.array of length (num_bins + 1)
        bin_centers: np.array of length (num_bins)
    """
    # create hourly time bins from 0 to 71 hours
    early_bins = np.arange(0, 71)  # 0 to 71 hours, we start the late bins from 72 hours

    # make logarithmically spaced bins from 72 hours to max_time_hours
    late_bins = []
    current_edge = 72.0
    while True:
        late_bins.append(current_edge)
        if current_edge >= max_time_hours:
            break
        current_edge *= base
    
    late_bins = np.array(late_bins)

    # combine early and late bins
    all_bins = np.unique(np.concatenate((early_bins, late_bins)))

    return all_bins
