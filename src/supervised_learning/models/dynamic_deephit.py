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
        predict_covariates: list = None,
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

        # ------ note that h comes out padded at the end not the beginning for shorter windows -------
        # generate right-padded mask
        device = h.device
        batch_size, max_len, _ = h.shape
        lengths = lengths.to(device)
        new_mask = torch.arange(max_len, device=device).expand(batch_size, max_len) < lengths.unsqueeze(1)  
        
        # --- NEW: Generate step-ahead predictions from GRU hidden states ---
        predictions = None
        if self.covariate_predictor is not None and return_predictions:
            # Use hidden states from timesteps 0 to seq_len-2 to predict timesteps 1 to seq_len-1
            # h: [batch, seq_len, hidden_dim]
            h_for_pred = h[:, :-1, :]  # [batch, seq_len-1, hidden_dim]
            predictions = self.covariate_predictor(h_for_pred)  # [batch, seq_len-1, num_predict_covariates]
            # apply mask to predictions
            pred_mask = new_mask[:, 1:] # need to remove as many predictions as zeros in the mask  
            predictions = predictions * pred_mask.unsqueeze(-1).float()  # Zero out padded predictions

        # ----temporal attention--------
        #  pass hidden states through temporal attention
        
        # move padding from left to right for x
        feature_dim = x.shape[2]
        x_right_padded = torch.zeros((batch_size, max_len, feature_dim), device=device, dtype=x.dtype)
        for i in range(batch_size):
            L = lengths[i].long()
            # Copy the last L timesteps (the valid ones) to the front of the new array
            x_right_padded[i, :L] = x[i, -L:]        
        
        context, attn_weights = self.attn(h, x_right_padded, new_mask)  # [batch_size, hidden_dim]

        # --- Event-specific hazard outputs ---
        # Note that here we use the left padded data so the last measurement corresponds to the last time step
        last_measurement = x[:, -1, :]      # [batch, 80] - just the last timestep

        # Input to cause-specific subnetworks
        subnetwork_input = torch.cat([context, last_measurement], dim=-1)
        
        logits = torch.stack([head(subnetwork_input) for head in self.heads], dim=1)
        B, K, T = logits.shape
        logits_flat = logits.view(B, -1)                # [B, K*T]
        probs_flat = F.softmax(logits_flat, dim=-1)     # each row sums to 1
        probs = probs_flat.view(B, K, T)   

        if return_predictions:
            return logits, probs, predictions, x_right_padded, new_mask  # [B, K, T], [B, K, T], [B, seq_len-1, num_predict_covariates]
        else:
            return logits, probs, _, x_right_padded, new_mask  # [B, K, T], [B, K, T]
        

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
        grad_clip=1.0,
        alpha_rank = None,
        alpha_pred = None,
        scheduler=None,
        verbose=True,
        physiological_feature_type = None
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
        self.alpha_rank = alpha_rank
        self.alpha_pred = alpha_pred
        self.scheduler = scheduler
        self.verbose = verbose
        self.physiological_feature_type = physiological_feature_type


    def train(self, batch: TensorBatch):

        self.model.train() # set model to train mode

        # unpack batch
        X = batch[0].to(self.device)      # [batch_size, n_ts, n_features]
        time_to_event = batch[1].to(self.device)      # [batch_size]
        event_label = batch[2].to(self.device)      # [batch_size]
        mask = batch[3].to(self.device) 

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass
        # Model should output: [batch_size, num_risks, num_time_bins]
        logits, _, predictions, x_right_padded , new_mask = self.model(X, mask=mask, return_predictions=True)

        # Compute log loss and ranking loss
        loss, lik_loss, rank_loss, pred_loss = self.loss_fn(
            logits, 
            predictions, 
            time_to_event, 
            event_label, 
            x_right_padded,
            new_mask,
            alpha_rank = self.alpha_rank,
            alpha_pred = self.alpha_pred,
            predict_covariates=self.model.predict_covariates,
            physiological_feature_type=self.physiological_feature_type
            )

        # Backward pass - compute gradients
        loss.backward()

        # Gradient clipping if specified
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.grad_clip
            )

        # Update weights
        self.optimizer.step()

        loss_dict = {
            "combined_loss": loss.item(),
            "likelihood_loss": lik_loss.item(),
            "ranking_loss": rank_loss.item(),
            "prediction_loss": pred_loss.item()
            }

        return loss_dict, logits
    

    def evaluate(self, batch: TensorBatch):
        self.model.eval()  # set model to eval mode
        
        # unpack batch
        X = batch[0].to(self.device)      # [batch_size, n_ts, n_features]
        time_to_event = batch[1].to(self.device)      # [batch_size]
        event_label = batch[2].to(self.device)      # [batch_size]
        mask = batch[3].to(self.device) 

        # Forward pass
        with torch.no_grad():
            logits, _, predictions, x_right_padded , new_mask = self.model(X, mask=mask, return_predictions=True)

            # Compute log loss and ranking loss
            loss, lik_loss, rank_loss, pred_loss = self.loss_fn(
                logits, 
                predictions, 
                time_to_event, 
                event_label, 
                x_right_padded,
                new_mask,
                predict_covariates=self.model.predict_covariates,
                physiological_feature_type=self.physiological_feature_type
                )

        # set model back to train mode
        self.model.train()

        loss_dict = {
            "combined_loss": loss.item(),
            "likelihood_loss": lik_loss.item(),
            "ranking_loss": rank_loss.item(),
            "prediction_loss": pred_loss.item()
            }

        return loss_dict, logits


def get_trainer(config_manager, feature_dim, hyperparams, train_processor):

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
        predict_covariates=train_processor.physiological_feature_ind,
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
            grad_clip = hyperparams['grad_clip'],
            alpha_rank = hyperparams['alpha_rank'],
            alpha_pred = hyperparams['alpha_pred'],
            scheduler=None,
            verbose=True, 
            physiological_feature_type = train_processor.physiological_feature_type                
        )

    return trainer



def dynamic_deephit_loss(
        logits, 
        predictions,
        time_to_event, 
        event_label,
        x_right_padded, 
        mask, # right padded
        predict_covariates: list = [],
        physiological_feature_type: list = [],
        no_censoring=True,
        pair_sample_frac = 1.0,
        alpha_rank= None, 
        alpha_pred = None,
        sigma=0.1,
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

        NOTES: SIGMA of 0.01 leads to all NaNs in predictions - why?


    Returns:
        scalar loss tensor
    """
    
    lik_loss = log_likelihood_loss(logits, time_to_event, event_label, no_censoring)
    rank_loss = ranking_loss(logits, time_to_event, event_label, no_censoring, sigma)
    
    pred_loss = prediction_loss(
            x_right_padded,
            predictions, 
            mask, 
            predict_covariates, 
            physiological_feature_type
            ) 
      
    # --- Combine ---
    loss = lik_loss + alpha_rank * rank_loss + alpha_pred * pred_loss
    return loss, lik_loss, rank_loss, pred_loss


def log_likelihood_loss(logits, time_to_event, event_label, no_censoring=True):
    B, K, T = logits.shape

    # subtract 1 from event_label if no censoring
    event_label = event_label -1 if no_censoring else event_label   

    # --- 1) Likelihood loss (use log_softmax over flattened K*T) ----------------
    logits_flat = logits.view(B, K * T)                # (B, K*T)
    logp_flat = F.log_softmax(logits_flat, dim=1)      # (B, K*T)
    idx = (event_label * T + time_to_event).unsqueeze(1) # (B,1) - this gives the index of the actual event label - time combination
    logp_flat_sel = logp_flat.gather(1, idx.long()).squeeze(1) # (B,) - take the log prob for the actual event label - time combination
    lik_loss = - logp_flat_sel.mean() # the negative of which can be used as the loss (log prob close to 0 mean highs high prob, 
    # log prob very negative means low prob   
    return lik_loss


def ranking_loss(logits, time_to_event, event_label, no_censoring=True, sigma=0.2):

    B, K, T = logits.shape
    device = logits.device

    # subtract 1 from event_label if no censoring
    event_label = event_label -1 if no_censoring else event_label   
    
    # --- 2) Ranking loss --------------------------------------------------------
    # compute cumulative incidence S_k(t) = sum_{s<=t} p_k(s)
    logits_flat = logits.view(B, K * T)                # (B, K*T)
    logp_flat = F.log_softmax(logits_flat, dim=1)      # (B, K*T)
    logp = logp_flat.view(B, K, T)     
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

    # comparable if k_i == k (i experienced cause k), and t_i < t_j (i earlier than j)
    # We'll loop over causes to get S_k at t_i for i and at t_i for j (note: compare at time t_i)
    rank_loss = torch.tensor(0.0, device=device)
    total_pairs = 0

    for k in range(K):
        # mask of i samples with event==k
        mask_i_k = (event_label == k)                 # (B,)
        if mask_i_k.sum() == 0:
            continue

        # For cause k, build pair mask: i in mask_i_k AND t_i < t_j
        # Broadcasting:
        mask_i_k_row = mask_i_k.view(B, 1)      # (B,1) - for rows
        mask_i_k_col = mask_i_k.view(1, B)           
        mask_time = (t_i < t_j)                 # (B,B) -- true where t_i < t_j
        pair_mask = mask_i_k_row & mask_i_k_col & mask_time    # (B,B) -- candidate pairs (i,j)

        num_pairs = pair_mask.sum().item()
        if num_pairs == 0:
            continue

        # Gather CIF values at t_i for all samples
        # cif_k has shape (B, T), we want to gather at position t_i for each sample
        cif_k = S[:, k, :]  # (B, T)

        # Create gathering indices: for each sample i, get cif at all t_i values
        # We need to broadcast this operation
        t_i_expand = t_i.expand(B, B)  # (B, B) - each row i has t_i repeated
        t_i_expand = t_i_expand.long()

        # Gather: for each (i,j) pair, get CIF_j at time t_i
        # This is a bit tricky - we need to use advanced indexing
        batch_indices = torch.arange(B, device=device).view(1, B).expand(B, B)
        cif_at_ti = cif_k[batch_indices, t_i_expand]  # (B, B)

        # Now cif_at_ti[i, j] = CIF_k(t_i | X_j)
        # We want: risk_i = CIF_k(t_i | X_i) and risk_j = CIF_k(t_i | X_j)

        risk_i = torch.diagonal(cif_at_ti, 0).unsqueeze(1)  # (B, 1) - diagonal elements
        risk_j = cif_at_ti  # (B, B)

        # Compute pairwise ranking loss: η(risk_i, risk_j) = exp(-(risk_i - risk_j) / sigma)
        # We want risk_i > risk_j, so we penalize when risk_i <= risk_j
        pairwise_loss = torch.exp(-(risk_i - risk_j) / sigma)  # (B, B)

        # Apply pair mask and sum
        masked_loss = pairwise_loss * pair_mask.float()
        rank_loss += masked_loss.sum()
        total_pairs += num_pairs

    if total_pairs > 0:
        rank_loss = rank_loss / total_pairs
    else:
        rank_loss = torch.tensor(0.0, device=device)
    
    return rank_loss


def prediction_loss(X, predictions, mask, predict_covariates, continuous_indices=None):
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
    x_next = X[:, 1:, predict_covariates]  # [batch, seq_len-1, feature_dim]
    mask_next = mask[:, 1:]      # [batch, seq_len-1]
    valid_mask = mask_next.unsqueeze(-1)  # [batch, seq_len-1, 1]

    # Compute loss based on covariate type
    binary_indices = [i for i, val in enumerate(continuous_indices) if val == 'binary']
    continuous_indices = [i for i, val in enumerate(continuous_indices) if val == 'continuous']

    # continuous - MSE
    continuous_loss = F.mse_loss(
        predictions[:,:,continuous_indices], 
        x_next[:,:,continuous_indices], reduction='none')  # [batch, seq_len-1, num_pred]

    # cross-entropy for binary
    target_binary = x_next[:,:,binary_indices]
    binary_loss = F.binary_cross_entropy_with_logits(
        predictions[:,:,binary_indices], 
        target_binary, reduction='none')  # [batch, seq_len-1, num

    # Combine losses
    loss = torch.cat([continuous_loss, binary_loss], dim=2)  
    
    # Apply mask and compute mean
    masked_loss = loss * valid_mask
    total_loss = masked_loss.sum()
    num_valid = valid_mask.sum() * loss.shape[2]  # multiply by num_pred
    
    return total_loss / (num_valid + 1e-8)


def make_mixed_resolution_time_bins(max_time_hours=3000):
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
    # create hourly time bins from 0 to 72 hours
    
    # first_day = np.arange(0, 25, 6) 
    # first_2_days = np.arange(24, 1+ 48, 12) 
    # first_2days_to_week = np.arange(48, 1+ 24*7, 24) 
    # week1_to_week2 = np.arange(168, 1 + 168*2.2, 48)
    # week2_to_rest = np.array((360, max_time_hours))

    # time_indices = np.concatenate((first_day, first_2_days, first_2days_to_week, week1_to_week2, week2_to_rest))
    # all_bins = np.sort(np.unique(time_indices))

    # create every 4 hour bins from 0 to 72 hours
    all_bins = np.arange(0, 73, 12)

    return all_bins
