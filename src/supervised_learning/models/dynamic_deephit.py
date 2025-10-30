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
        attn_hidden_dim: int,
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
        self.feature_dim = feature_dim
        self.hidden_size = hidden_dim
        self.attn_hidden_dim = attn_hidden_dim
        self.num_layers = num_layers
        self.num_time_bins = num_time_bins
        self.num_risks = num_risks

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

        # temporal attention
        self.attn = TemporalAttention(hidden_size=self.hidden_size, attn_size=attn_hidden_dim)

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

        # # --- Extract last valid hidden state per sequence ---
        # if mask is not None:
        #     idx = (mask.sum(dim=1) - 1).long().unsqueeze(1).unsqueeze(2)
        #     idx = idx.expand(-1, 1, h.size(2))
        #     h_last = h.gather(1, idx).squeeze(1)
        # else:
        #     h_last = h[:, -1, :]

        # pass hidden states through temporal attention
        device = h.device
        batch_size, max_len, _ = h.shape
        lengths = lengths.to(device)
        new_mask = torch.arange(max_len, device=device).expand(batch_size, max_len) < lengths.unsqueeze(1)
        h_last, attn_weights = self.attn(h, new_mask)  # [batch_size, hidden_dim]

        # --- Event-specific hazard outputs ---
        outputs = []
        for head in self.heads:
            logits = head(h_last)             # [batch, num_time_bins]
            probs = F.softmax(logits, dim=-1) # per-risk discrete-time probabilities
            outputs.append(probs)

        return outputs  # list of tensors, len=num_risks
        

class TemporalAttention(nn.Module):
    """
    Simple temporal attention over RNN hidden states.
    Computes scalar score for each time step and returns:
      - context: [B, H]  (weighted sum of h over time)
      - attn_weights: [B, T]
    Masking ensures padded timesteps receive zero weight.
    """
    def __init__(self, hidden_size: int, attn_size: int = None):
        super().__init__()
        self.hidden_size = hidden_size
        if attn_size is None:
            attn_size = self.hidden_size // 2 if self.hidden_size >= 4 else self.hidden_size
        self.attn_size = attn_size
        self.W = nn.Linear(self.hidden_size, self.attn_size, bias=True)
        self.v = nn.Linear(self.attn_size, 1, bias=False)  #
        # optionally add dropout / layernorm if desired

    def forward(self, h: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h: [B, T, H] RNN outputs / hidden states (batch_first)
        mask: [B, T] boolean or float (1 for valid, 0 for pad)
        returns: context [B, H], attn_weights [B, T]
        """
        # compute intermediate: [B, T, attn_hidden]
        u = torch.tanh(self.W(h))                # [B, T, A]
        scores = self.v(u).squeeze(-1)           # [B, T]

        # mask: convert to float (1.0 valid, 0.0 pad)
        if mask.dtype != torch.bool:
            mask_bool = mask.bool()
        else:
            mask_bool = mask
        
        # set scores of padded positions to a large negative number before softmax
        # numerical stable softmax: subtract max per row
        scores_masked = scores.masked_fill(~mask_bool, float("-1e9"))
        # compute normalized attention
        attn_weights = F.softmax(scores_masked, dim=-1)   # [B, T], padded positions ~0

        # weighted sum => context
        attn_weights_unsq = attn_weights.unsqueeze(-1)    # [B, T, 1]
        context = torch.sum(attn_weights_unsq * h, dim=1) # [B, H]

        return context, attn_weights


class DynamicDeepHitTrainer:
    def __init__(
        self,
        model,
        optimizer,
        device = "CUDA" if torch.cuda.is_available() else "CPU",
        num_risks = 2,
        num_time_bins = 100,
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



    def train(self, batch: TensorBatch):
        # unpack batch
        X = batch[0].to(self.device)      # [batch_size, n_ts, n_features]
        T = batch[1].to(self.device)      # [batch_size]
        E = batch[2].to(self.device)      # [batch_size]
        mask = batch[3].to(self.device) 

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass
        # Model should output: [batch_size, num_risks, num_time_bins]
        outputs = self.model(X, mask)

        # Compute loss
        loss = self.loss_fn(outputs, T, E, mask)

        # Gradient clipping if specified
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.grad_clip
            )

        # Update weights
        self.optimizer.step()

        log_dict = {"dynamic_deephit_loss": loss.item()}

        return log_dict



def get_trainer(config_manager, feature_dim, hyperparams):

    learning_rate = float(hyperparams['learning_rate'])
    n_time_bins = len(config_manager.time_bins)

    num_risks = 2 # hardcoded for death vs discharge

    dynamic_deephit_model = DynamicDeepHitMulti(
        feature_dim,
        hyperparams['hidden_dim'],
        hyperparams['attn_hidden_dim'],
        hyperparams['n_layers'],
        n_time_bins,
        num_risks,
        dropout=hyperparams['dropout'],
        rnn_type="GRU",
    ).to(config_manager.training_config['device'])

    deephit_optimizer = torch.optim.Adam(
        dynamic_deephit_model.parameters(),
        lr=learning_rate
    )

    trainer = DynamicDeepHitTrainer(
            dynamic_deephit_model,
            deephit_optimizer,
            device=config_manager.training_config['device'],
            num_risks=num_risks,
            num_time_bins=n_time_bins,
            loss_fn=dynamic_deephit_loss,
            grad_clip = None,
            scheduler=None,
            verbose=True                
        )

    return trainer



def dynamic_deephit_loss(preds, y_time, y_event, mask=None, alpha=0.1, eps=1e-8):
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
