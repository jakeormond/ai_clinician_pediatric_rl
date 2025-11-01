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
        predict_covariates: list = None
    ):
        """
        Args:
            input_dim: number of covariates per time step
            hidden_dim: RNN hidden size
            num_layers: number of RNN layers
            num_time_bins: number of discrete time bins
            num_risks: number of competing risks (e.g. 2 for death, discharge)
            dropout: dropout rate
            rnn_type: 'GRU' or 'LSTM',
            predict_covariates: list of indices of covariates to predict for regularization
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_dim
        self.attn_hidden_dim = attn_hidden_dim
        self.num_layers = num_layers
        self.num_time_bins = num_time_bins
        self.num_risks = num_risks
        self.predict_covariates = predict_covariates if predict_covariates else []

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

        # prediction head for step-ahead covariate prediction 
        if len(self.predict_covariates) > 0:
            self.covariate_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, len(self.predict_covariates))
            )
        else:
            self.covariate_predictor = None


        # temporal attention
        self.attn = TemporalAttention(
            hidden_size=self.hidden_size, 
            feature_size=self.feature_dim, 
            attn_size=attn_hidden_dim
            )

        # --- Event-specific heads ---
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_time_bins)
            )
            for _ in range(num_risks)
        ])
        

    def forward(self, x, mask=None, return_predictions=False):
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

        # --- NEW: Generate step-ahead predictions from GRU hidden states ---
        predictions = None
        if self.prediction_head is not None and return_predictions:
            # Use hidden states from timesteps 0 to seq_len-2 to predict timesteps 1 to seq_len-1
            # h: [batch, seq_len, hidden_dim]
            h_for_pred = h[:, :-1, :]  # [batch, seq_len-1, hidden_dim]
            predictions = self.prediction_head(h_for_pred)  # [batch, seq_len-1, num_predict_covariates]
        
        # ----temporal attention--------
        #  pass hidden states through temporal attention
        device = h.device
        batch_size, max_len, _ = h.shape
        lengths = lengths.to(device)
        new_mask = torch.arange(max_len, device=device).expand(batch_size, max_len) < lengths.unsqueeze(1)
        context, attn_weights = self.attn(h, new_mask)  # [batch_size, hidden_dim]

        # --- Event-specific hazard outputs ---
        last_measurement = x[:, -1, :]      # [batch, 80] - just the last timestep

        # Input to cause-specific subnetworks
        subnetwork_input = torch.cat([context, last_measurement], dim=-1)
        
        logits = torch.stack([head(subnetwork_input) for head in self.heads], dim=1)
        B, K, T = logits.shape
        logits_flat = logits.view(B, -1)                # [B, K*T]
        probs_flat = F.softmax(logits_flat, dim=-1)     # each row sums to 1
        probs = probs_flat.view(B, K, T)   

        if return_predictions:
            return logits, probs, predictions  # [B, K, T], [B, K, T], [B, seq_len-1, num_predict_covariates]
        else:
            return logits, probs  # [B, K, T], [B, K, T]
        

class TemporalAttention(nn.Module):
    """
    Simple temporal attention over RNN hidden states.
    Computes scalar score for each time step and returns:
      - context: [B, H]  (weighted sum of h over time)
      - attn_weights: [B, T]
    Masking ensures padded timesteps receive zero weight.
    """
    def __init__(self, hidden_size: int, feature_size: int, attn_size: int = None):
        super().__init__()
        self.hidden_size = hidden_size
        if attn_size is None:
            attn_size = self.hidden_size // 2 if self.hidden_size >= 4 else self.hidden_size
        self.attn_size = attn_size
        self.feature_size = feature_size
        
        self.W_h = nn.Linear(self.hidden_size, self.attn_size, bias=False)
        self.W_x = nn.Linear(self.feature_size, self.attn_size, bias=False)
        self.v = nn.Linear(self.attn_size, 1, bias=False)  #
        self.b = nn.Parameter(torch.zeros(attn_size))
        # optionally add dropout / layernorm if desired

    def forward(self, h: torch.Tensor, x: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h: [B, T, H] RNN outputs / hidden states (batch_first)
        mask: [B, T] boolean or float (1 for valid, 0 for pad)
        returns: context [B, H], attn_weights [B, T]
        """
        # compute intermediate: [B, T, attn_hidden]
        Wh = self.W_h(h)    # [B, T, A]
        Wx = self.W_x(x)    # [B, T, A]
        
        u = torch.tanh(Wh + Wx + self.b)                # [B, T, A]
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
        time_to_event = batch[1].to(self.device)      # [batch_size]
        event_label = batch[2].to(self.device)      # [batch_size]
        mask = batch[3].to(self.device) 

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass
        # Model should output: [batch_size, num_risks, num_time_bins]
        logits, _ = self.model(X, mask)

        # Compute loss
        loss = self.loss_fn(logits, time_to_event, event_label, mask)

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



def dynamic_deephit_loss(
        logits, 
        time_to_event, 
        event_label, 
        mask, 
        no_censoring=True,
        alpha=0.1, 
        eps=1e-8
        ):
    """
    Dynamic-DeepHit loss for competing risks (multi-risk, discrete-time)

    Args:
        logits: tensor (B, K, T)  -- raw model outputs
        time_to_event: [batch] integer time bin of event/censoring (0-based)
        event_label: [batch] risk label (0=censored, 1..R)
        mask: optional [batch] or [batch, seq_len] for padded sequences (1=valid, 0=padded)
        alpha: weight for ranking loss
        eps: numerical stability

    Returns:
        scalar loss tensor
    """
    B, K, T = logits.shape
    device = logits.device

    # subtract 1 from event_label if no censoring
    event_label = event_label -1 if no_censoring else event_label   

    # --- 1) Likelihood loss (use log_softmax over flattened K*T) ----------------
    logits_flat = logits.view(B, K * T)                # (B, K*T)
    logp_flat = F.log_softmax(logits_flat, dim=1)      # (B, K*T)
    logp = logp_flat.view(B, K, T)                     # (B, K, T)
    # gather log probability for observed (k_obs, t_obs)
    idx = (event_label * T + time_to_event).unsqueeze(1) # (B,1) - this gives the index of the actual event label - time combination
    logp_flat_sel = logp_flat.gather(1, idx.long()).squeeze(1) # (B,) - take the log prob for the actual event label - time combination
    lik_loss = - logp_flat_sel.mean() # the negative of which can be used as the loss (log prob close to 0 mean highs high prob, 
    # log prob very negative means low prob   

    # --- 2) Ranking loss --------------------------------------------------------
    # compute cumulative incidence S_k(t) = sum_{s<=t} p_k(s)
    p = logp.exp()                                     # (B,K,T) - probabilities
    S = torch.cumsum(p, dim=2)                         # (B,K,T) - cumulative incidence for each event class

    # We'll form pairwise differences only for pairs where i has event k and t_i < t_j
    # Create index tensors for pairwise operations
    # We vectorize by creating all pairs (i,j) in batch: shape (B,B)
    # mask_ij_k: True where pair (i,j) is comparable for cause k.
    # To limit complexity, we optionally sample pairs randomly.
    # Build matrices for t and k comparisons:
    t_i = time_to_event.view(B, 1)   # (B,1)
    t_j = time_to_event.view(1, B)   # (1,B)
    k_i = event_label.view(B, 1)   # (B,1)
    k_j = event_label.view(1, B)   # (1,B)

    # comparable if k_i == k (i experienced cause k), and t_i < t_j (i earlier than j)
    # We'll loop over causes to get S_k at t_i for i and at t_i for j (note: compare at time t_i)
    rank_loss = torch.tensor(0.0, device=device)
    total_pairs = 0

    for k in range(K):
        # mask of i samples with event==k
        mask_i_k = (k_obs == k)                 # (B,)
        if mask_i_k.sum() == 0:
            continue

        # For cause k, build pair mask: i in mask_i_k AND t_i < t_j
        # Broadcasting:
        mask_i_k_mat = mask_i_k.view(B, 1)      # (B,1)
        mask_time = (t_i < t_j)                 # (B,B)
        pair_mask = mask_i_k_mat & mask_time    # (B,B) -- candidate pairs (i,j)

        # Optionally sample a subset of pairs
        if pair_sample_frac < 1.0:
            # get indices of true entries
            i_idx, j_idx = torch.nonzero(pair_mask, as_tuple=True)
            n_pairs = i_idx.shape[0]
            if n_pairs == 0:
                continue
            n_sample = max(1, int(n_pairs * pair_sample_frac))
            perm = torch.randperm(n_pairs, device=device)[:n_sample]
            i_idx_s = i_idx[perm]; j_idx_s = j_idx[perm]
            # compute differences using these index lists
            S_i = S[i_idx_s, k, t_obs[i_idx_s]]           # S_k(t_i | x_i)
            S_j = S[j_idx_s, k, t_obs[i_idx_s]]           # S_k(t_i | x_j)  note same t_i
            diff = S_i - S_j
            # logistic pairwise loss: log(1 + exp(-diff))
            rank_loss += torch.log1p(torch.exp(-diff)).sum()
            total_pairs += n_sample
        else:
            # vectorized over all pairs
            # we need S_k(t_i | x_i) and S_k(t_i | x_j) for all (i,j)
            # build S_i matrix: (B,B) with row i giving S_k(t_i | x_i)
            t_i_idx = t_obs.view(B, 1).expand(B, B)     # (B,B)
            # S_k at times for all samples: S[:,k,:] shape (B, T)
            S_k = S[:, k, :]                            # (B, T)
            # gather S_k(t_i | x_i) as matrix: S_k[i, t_i] repeated across columns
            S_i_vals = S_k.gather(1, t_obs.view(B,1)).expand(B, B)  # (B,B)
            # gather S_k(t_i | x_j): for each row i, we want S_k[j, t_i]
            # create index matrix t_i_idx and use gather on S_k.transpose?
            # easiest: S_k_ti_for_all_j = S_k[:, t_i] but we need per-column different t_i.
            # We'll use advanced indexing:
            # For pair (i,j): S_k[j, t_i] -> S_k[ j_idx, t_i[i] ]
            i_idx, j_idx = torch.nonzero(pair_mask, as_tuple=True)
            if i_idx.numel() == 0:
                continue
            S_i = S_k[i_idx, t_obs[i_idx]]              # (n_pairs,)
            S_j = S_k[j_idx, t_obs[i_idx]]              # (n_pairs,)  # note t_i used
            diff = S_i - S_j
            rank_loss += torch.log1p(torch.exp(-diff)).sum()
            total_pairs += i_idx.shape[0]


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


def predition_loss(predictions, x, mask, predict_covariates, continuous_indices=None):
    """
    Compute the step-ahead prediction loss (L3 in DeepHit paper).
    
    Args:
        predictions: [batch, seq_len-1, num_predict_covariates] - predicted next-step values
        x: [batch, seq_len, feature_dim] - original input sequences
        mask: [batch, seq_len] - 1 for valid timesteps, 0 for padding
        predict_covariates: list of covariate indices being predicted
        continuous_indices: list of indices (within predict_covariates) that are continuous
                          (if None, assumes all are continuous)
    
    Returns:
        loss: scalar prediction loss
    """
    batch_size, seq_len_minus_1, num_pred = predictions.shape

    # Extract the actual next-step values
    x_next = x[:, 1:, :]  # [batch, seq_len-1, feature_dim]
    x_next_pred_covs = x_next[:, :, predict_covariates]  # [batch, seq_len-1, num_predict_covariates]
    
    # Mask for valid predictions (both current and next timestep must be valid)
    # mask: [batch, seq_len]
    mask_current = mask[:, :-1]  # [batch, seq_len-1]
    mask_next = mask[:, 1:]      # [batch, seq_len-1]
    valid_mask = mask_current * mask_next  # [batch, seq_len-1]
    valid_mask = valid_mask.unsqueeze(-1)  # [batch, seq_len-1, 1]

    # Compute loss based on covariate type
    if continuous_indices is None:
        # All continuous - MSE
        loss = F.mse_loss(predictions, x_next_pred_covs, reduction='none')  # [batch, seq_len-1, num_pred]
    else:
        # Mixed continuous and binary
        loss = torch.zeros_like(predictions)
        
        # Continuous covariates - MSE
        for i in continuous_indices:
            loss[:, :, i] = (predictions[:, :, i] - x_next_pred_covs[:, :, i]) ** 2
        
        # Binary covariates - Binary cross-entropy
        binary_indices = [i for i in range(num_pred) if i not in continuous_indices]
        for i in binary_indices:
            pred_i = torch.sigmoid(predictions[:, :, i])  # Apply sigmoid for binary
            target_i = x_next_pred_covs[:, :, i]
            # Binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
            loss[:, :, i] = -(target_i * torch.log(pred_i + 1e-8) + 
                             (1 - target_i) * torch.log(1 - pred_i + 1e-8))
    
    # Apply mask and compute mean
    masked_loss = loss * valid_mask
    total_loss = masked_loss.sum()
    num_valid = valid_mask.sum()
    
    return total_loss / (num_valid + 1e-8)


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
