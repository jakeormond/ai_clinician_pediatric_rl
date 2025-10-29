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



def deephit_loss(preds, y_time, y_event, mask=None, alpha=0.1, eps=1e-8):
    """
    Dynamic-DeepHit loss for competing risks (multi-risk, discrete-time)

    Args:
        preds: list of tensors [batch, num_bins] per risk (hazard probabilities, already softmaxed)
        y_time: [batch] integer time bin of event/censoring (0-based)
        y_event: [batch] risk label (0=censored, 1..R)
        mask: optional [batch] or [batch, seq_len] for padded sequences (1=valid, 0=padded)
        alpha: weight for ranking loss
        eps: numerical stability

    Returns:
        scalar loss tensor
    """

    device = y_time.device
    batch_size = y_time.size(0)
    num_risks = len(preds)
    num_bins = preds[0].size(1)

    # --- Convert hazards to survival probabilities per bin ---
    # h_r(t) = hazard probability for risk r at bin t
    # S_r(t) = survival up to bin t = prod_{k=0}^{t} (1 - h_r(k))
    S = []
    for h in preds:
        S_r = torch.cumprod(1 - h + eps, dim=1)
        S.append(S_r)

    # --- Log-likelihood term ---
    ll = torch.zeros(batch_size, device=device)
    for r_idx, h_r in enumerate(preds):
        # mask for samples with this event
        is_event_r = (y_event == (r_idx + 1))  # 1-indexed in y_event
        # log prob of observed event
        idx = y_time[is_event_r]
        if idx.numel() > 0:
            ll[is_event_r] += -torch.log(h_r[is_event_r, idx] + eps)

        # for censored or other-risk events: survival up to time
        mask_surv = (y_event != (r_idx + 1))
        idx_surv = y_time[mask_surv]
        if idx_surv.numel() > 0:
            ll[mask_surv] += -torch.log(S[r_idx][mask_surv, idx_surv] + eps)

    # --- Ranking loss ---
    # Encourage earlier events to have higher predicted risk
    ranking = torch.zeros(1, device=device)
    for r_idx, h_r in enumerate(preds):
        S_r = S[r_idx]
        # pairwise: i,j such that y_time[i] < y_time[j] and both uncensored for this risk
        for i in range(batch_size):
            for j in range(batch_size):
                if y_event[i] == r_idx + 1 and y_event[j] == r_idx + 1:
                    if y_time[i] < y_time[j]:
                        ranking += F.relu(S_r[j, y_time[j]] - S_r[i, y_time[i]])

    ranking = ranking / (batch_size * batch_size + eps)

    # --- Combine ---
    loss = ll.mean() + alpha * ranking
    return loss



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
    early_bins = np.arange(0, 72)  # 0 to 71 hours, we start the late bins from 72 hours

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
