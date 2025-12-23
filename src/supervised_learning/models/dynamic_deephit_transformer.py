from typing import Dict, List, Tuple
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

TensorBatch = List[torch.Tensor]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)



class TransformerDeepHitMulti(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        d_model: int, # Replaces hidden_dim
        nhead: int,
        num_encoder_layers: int, # Replaces num_layers
        num_time_bins: int,
        num_risks: int,
        dropout: float = 0.3,
        dim_feedforward: int = 2048, # Standard Transformer parameter
        max_seq_len: int = 768, # For Positional Encoding and padding
        predict_covariates: list = None,
    ):
        """
        Args:
            feature_dim: number of covariates per time step (input_size)
            d_model: Transformer feature dimension (replaces hidden_dim)
            nhead: number of attention heads
            num_encoder_layers: number of Transformer Encoder layers
            num_time_bins: number of discrete time bins
            num_risks: number of competing risks (e.g. 2 for death, discharge)
            dropout: dropout rate
            max_seq_len: maximum length of the input window (e.g., 768)
            predict_covariates: list of indices of covariates to predict for regularization
        """
        super().__init__()

        self.feature_dim = int(feature_dim)
        self.d_model = int(d_model)
        self.num_time_bins = int(num_time_bins)
        self.num_risks = int(num_risks)
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.predict_covariates = predict_covariates if predict_covariates else []

        # Pre-calculate the fixed-size Causal Mask
        self.register_buffer('causal_mask', self.generate_square_subsequent_mask(max_seq_len))
        
        # --- 1. Input Embedding ---
        # Map input features to the Transformer's model dimension (d_model)
        self.input_projection = nn.Linear(self.feature_dim, self.d_model)

        # --- 2. Positional Encoding ---
        # Crucial for time-series to embed sequential order
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)

        # --- 3. Transformer Encoder Stack ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model) # Final LayerNorm after the stack
        )

        # --- 4. Output Pooling (Replaces Attention) ---
        # The GRU version used an external Temporal Attention layer. 
        # Here, the Self-Attention *inside* the Transformer handles the temporal weighting.
        # We will use the final timestep's output (index -1) from the Transformer.
        
        # --- 5. Covariate Predictor (If used) ---
        if len(self.predict_covariates) > 0:
            self.covariate_predictor = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.d_model, len(self.predict_covariates))
            )
        else:
            self.covariate_predictor = None

        # --- 6. Event-specific heads ---
        # Input to the heads is the final timestep's output (size d_model)
        # We only use the d_model output here, as we are predicting conditional survival
        # from the final timestep, and the Transformer output at this position 
        # already summarizes the relevant context window.
        head_input_dim = self.d_model 

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_input_dim, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.d_model, num_time_bins)
            )
            for _ in range(num_risks)
        ])
        
        self._initialize_weights()

    def _initialize_weights(self):
        # Optional: Initialize weights for better convergence
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generates an upper-triangular matrix of -inf, which prevents attention to future positions."""
        # PyTorch helper function creates a matrix where T means blocked
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    
    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, feature_dim]
            mask: Boolean mask of shape [batch_size, seq_len] where True indicates padded (ignored) elements.
        """
        # Ensure we only use the part of the pre-calculated mask relevant to the batch sequence length
        batch_size, seq_len, _ = x.shape
        
        # 1. Project and encode features
        x = self.input_projection(x) # [B, T, feature_dim] -> [B, T, d_model]
        
        # 2. Add positional encoding
        x = self.pos_encoder(x)
        
        # 3. Transformer Encoder
        # The attention mask (src_key_padding_mask) needs to be passed to the encoder.
        # PyTorch Transformer expects mask to be a boolean tensor where True means to IGNORE the key.
        if padding_mask is not None:
            # Mask should be True where padding exists (you should ensure your padding logic creates this mask)
            output = self.transformer_encoder(
                x, 
                mask=self.causal_mask[:seq_len, :seq_len], # Causal Mask
                src_key_padding_mask=padding_mask
            ) # [B, T, d_model]
        else:
            output = self.transformer_encoder(
                x,
                mask=self.causal_mask[:seq_len, :seq_len] # Causal Mask
            )
        
        # generate predictions for each timestep
        # 5. Covariate Prediction (If enabled)
        covariate_pred = None
        if self.covariate_predictor:
            # 2. Slice: Use outputs from 0 to T-1 to predict covariates at 1 to T
            # We take all timesteps except the last one for the input
            h_for_pred = output[:, :-1, :]

            # 3. Generate predictions
            covariate_pred = self.covariate_predictor(h_for_pred)
            
            # 4. Handle Masking
            # If you have a padding mask [batch, seq_len], align it to the predictions
            if padding_mask is not None:
                # Shift the mask to match the t+1 targets
                pred_mask = padding_mask[:, 1:]
                # Note: In PyTorch, padding masks are often True for 'pad' and False for 'data'
                # Adjust the logic below to match your mask's boolean logic:
                covariate_pred = covariate_pred * (~pred_mask).unsqueeze(-1).float()
            
        # 4. Process all timesteps
        # Flatten the output to treat every timestep of every sequence as a separate prediction point
        # Output shape is now [B * T, d_model]
        all_timesteps_rep = output.reshape(batch_size * seq_len, self.d_model)
            
        # 6. Event-specific heads
        risk_logits = []
        for head in self.heads:
            # Apply the heads to the flattened representation
            logits = head(all_timesteps_rep) # [B * T, num_time_bins]
            
            # Reshape the logits back to sequence form for loss calculation
            logits = logits.reshape(batch_size, seq_len, self.num_time_bins) # [B, T, num_time_bins]
            risk_logits.append(logits)
            
        return risk_logits, covariate_pred


class TransformerDeepHitTrainer:
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
        alpha_class = None,
        sigma = None,
        scheduler=None,
        verbose=True,
        physiological_feature_type = None,
        l1_params: list = [],
        l1_lambda: float = 0.0,
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
        self.alpha_class = alpha_class
        self.alpha_rank = alpha_rank
        self.alpha_pred = alpha_pred
        self.sigma = sigma
        self.scheduler = scheduler
        self.verbose = verbose
        self.physiological_feature_type = physiological_feature_type
        self.l1_params = l1_params
        self.l1_lambda = float(l1_lambda)

    def train(self, batch: TensorBatch):

        self.model.train() # set model to train mode

        # unpack batch
        # X = batch[0].to(self.device)      # [batch_size, n_ts, n_features]
        X = batch['features'].to(self.device)
        # time_to_event = batch[1].to(self.device)      # [batch_size]
        time_to_event = batch['time_to_event'].to(self.device)
        # event_label = batch[2].to(self.device)      # [batch_size]
        event_label = batch['event'].to(self.device)
        # mask = batch[3].to(self.device)
        mask = batch['mask'].to(self.device)

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass
        # risk_logits: list of tensors, where each tensor is [B, T_cap, num_time_bins]
        # covariate_pred: None (or [B, T_cap, num_covariates] if implemented)
        risk_logits, covariate_pred = self.model(X, padding_mask=mask)
        
        # We need to restructure the risk_logits for the loss function:
        # 1. Stack the list of risk tensors into a single tensor: [B, T_cap, num_risks, num_time_bins]
        logits = torch.stack(risk_logits, dim=2)

        # 2. Reshape to [B * T_cap, num_risks, num_time_bins] for the loss function
        #    (Assuming your existing loss function expects a flattened batch/time dimension)
        B, T_cap, R, D = logits.shape
        logits_flat = logits.reshape(B * T_cap, R, D)

        # 3. Create flattened versions of the targets and masks
        time_to_event_flat = time_to_event.reshape(B * T_cap)
        event_label_flat = event_label.reshape(B * T_cap)
        mask_flat = mask.reshape(B * T_cap) # This is the key for excluding padded steps

        # Compute log loss and ranking loss
        loss, lik_loss, class_loss, rank_loss, pred_loss = self.loss_fn(
            logits, 
            covariate_pred, 
            time_to_event_flat, 
            event_label_flat, 
            mask_flat, # The new mask is the key to exclude padded/invalid steps
            alpha_class = self.alpha_class,
            alpha_rank = self.alpha_rank,
            alpha_pred = self.alpha_pred,
            sigma = self.sigma,
            predict_covariates=self.model.predict_covariates,
            physiological_feature_type=self.physiological_feature_type
            )

        if self.l1_lambda > 0.0 and len(self.l1_params) > 0:
            l1_lambda = self.l1_lambda  # set your value during initialization
            l1_penalty = sum(torch.sum(torch.abs(p)) for p in self.l1_params)
            loss = loss + l1_lambda * l1_penalty

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
            "classification_loss": class_loss.item(),
            "ranking_loss": rank_loss.item(),
            "prediction_loss": pred_loss.item()
            }

        return loss_dict, logits
    

    def evaluate(self, batch: TensorBatch):
        self.model.eval()  # set model to eval mode
        
        # unpack batch
        # X = batch[0].to(self.device)      # [batch_size, n_ts, n_features]
        X = batch['features'].to(self.device)
        # time_to_event = batch[1].to(self.device)      # [batch_size]
        time_to_event = batch['time_to_event'].to(self.device)
        # event_label = batch[2].to(self.device)      # [batch_size]
        event_label = batch['event'].to(self.device)
        # mask = batch[3].to(self.device)
        mask = batch['mask'].to(self.device)

        # Forward pass
        with torch.no_grad():
            risk_logits, covariate_pred = self.model(X, padding_mask=mask)

            # We need to restructure the risk_logits for the loss function:
            # 1. Stack the list of risk tensors into a single tensor: [B, T_cap, num_risks, num_time_bins]
            logits = torch.stack(risk_logits, dim=2)
            
            # Compute log loss and ranking loss
            loss, lik_loss, class_loss, rank_loss, pred_loss = self.loss_fn(
                logits, 
                covariate_pred, 
                time_to_event,                  
                event_label, 
                X,
                mask, # The new mask is the key to exclude padded/invalid steps
                alpha_class = self.alpha_class,
                alpha_rank = self.alpha_rank,
                alpha_pred = self.alpha_pred,
                sigma = self.sigma,
                predict_covariates=self.model.predict_covariates,
                physiological_feature_type=self.physiological_feature_type
                )

        # set model back to train mode
        self.model.train()

        loss_dict = {
            "combined_loss": loss.item(),
            "likelihood_loss": lik_loss.item(),
            "classification_loss": class_loss.item(),
            "ranking_loss": rank_loss.item(),
            "prediction_loss": pred_loss.item()
            }

        return loss_dict, logits


def get_trainer(config_manager, feature_dim, hyperparams, train_processor):

    learning_rate = float(hyperparams['learning_rate'])
    n_time_bins = len(config_manager.time_bins) - 1

    num_risks = 2 # hardcoded for death vs discharge

    dynamic_deephit_model = TransformerDeepHitMulti(
        feature_dim=feature_dim,
        d_model=hyperparams['d_model'],               # e.g., 128 or 256
        nhead=hyperparams['nhead'],                   # e.g., 4 or 8
        num_encoder_layers=hyperparams['n_layers'],   # e.g., 3 or 6
        num_time_bins=n_time_bins,
        num_risks=num_risks,
        dropout=hyperparams['dropout'],
        dim_feedforward=hyperparams.get('dim_feedforward', 512), # Usually 4 * d_model
        max_seq_len=train_processor.window_length,
        predict_covariates=train_processor.physiological_feature_ind,       
    ).to(config_manager.training_config['device'])

    # Collect L1 parameters: classification head weights only
    l1_params = [
        p for name, p in dynamic_deephit_model.named_parameters()
        if name.startswith("heads.") and name.endswith(".weight")
    ]

    deephit_optimizer = torch.optim.AdamW(
        dynamic_deephit_model.parameters(),
        lr=learning_rate,
        weight_decay = float(hyperparams['weight_decay'])
    )

    trainer = TransformerDeepHitTrainer(
            dynamic_deephit_model,
            deephit_optimizer,
            device=config_manager.training_config['device'],
            num_risks=num_risks,
            num_time_bins=n_time_bins,
            loss_fn=dynamic_deephit_loss,
            grad_clip = hyperparams['grad_clip'],
            alpha_class = hyperparams['alpha_class'], 
            alpha_rank = hyperparams['alpha_rank'],
            alpha_pred = hyperparams['alpha_pred'],
            sigma = hyperparams['sigma'],
            scheduler=None,
            verbose=True, 
            physiological_feature_type = train_processor.physiological_feature_type,
            l1_params=l1_params,          # ← ADD THIS
            l1_lambda=hyperparams['l1_lambda'],                
        )

    return trainer



def dynamic_deephit_loss(
        logits, 
        predictions,
        time_to_event, 
        event_label,
        X, # might not need this anymore, check!
        mask, # right padded
        predict_covariates: list = [],
        physiological_feature_type: list = [],
        no_censoring=True,
        pair_sample_frac = 0.5,
        alpha_class= None,
        alpha_rank= None, 
        alpha_pred = None,
        sigma=None,
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
    # before flattening tensors, deal with predictions, as they have one less timestep per window
    targets_shifted = X[:, 1:, :]  # Shifted targets for prediction loss
    mask_shifted = mask[:, 1:]    # Shifted mask for prediction loss
    valid_preds = predictions[~mask_shifted]  # Filter out padded predictions
    valid_targets = targets_shifted[~mask_shifted]  # Corresponding valid targets
    valid_targets = valid_targets[:, predict_covariates]  # Select only predicted covariates

    # flatten tensors where needed
    # 2. Reshape to [B * T_cap, num_risks, num_time_bins] for the loss function
    #    (Assuming your existing loss function expects a flattened batch/time dimension)
    B, T_cap, R, D = logits.shape
    logits_flat = logits.reshape(B * T_cap, R, D)
    
    # 3. Create flattened versions of the targets and masks
    time_to_event_flat = time_to_event.reshape(B * T_cap)
    event_label_flat = event_label.reshape(B * T_cap)
    mask_flat = mask.reshape(B * T_cap) # This is the key for excluding padded steps

    # --- 1. Filter Padded Timesteps ---
    # We only want to compute loss for timesteps that are NOT padding.
    # mask is False for valid steps.
    valid_indices = ~mask_flat

    # Filter all relevant tensors to keep only the valid "active" hours
    v_logits = logits_flat[valid_indices]    
    v_time_to_event = time_to_event_flat[valid_indices]
    v_event_label = event_label_flat[valid_indices]

    # --- 2. Compute DeepHit Component Losses ---
    # Because we filtered, these functions now see a "batch" of size N_valid
    lik_loss = log_likelihood_loss(v_logits, v_time_to_event, v_event_label, no_censoring)
    classification_loss = event_classification_loss(v_logits, v_event_label, no_censoring)
    
    # Ranking loss evaluates pairs. With multi-timestep, it compares risk 
    # across different patients AND different hours within the stay.
    rank_loss = ranking_loss(v_logits, v_time_to_event, v_event_label, no_censoring=no_censoring, sigma=sigma)

    # --- 3. Auxiliary Prediction Loss ---
    # If you are doing step-ahead covariate prediction, you need to filter 
    # the feature tensors as well. 
    
    v_predictions = predictions[valid_indices]
    
    if alpha_pred > 0 and len(predict_covariates) > 0 and x_right_padded is not None:
        # Flatten x_right_padded to [B * T, feature_dim] to match mask
        x_flat = x_right_padded.view(-1, x_right_padded.shape[-1])
        v_x = x_flat[valid_indices]
        
        pred_loss = prediction_loss(
                v_x,
                v_predictions, 
                torch.ones_like(v_event_label), # Already filtered, so all are valid (1)
                predict_covariates, 
                physiological_feature_type
                ) 
    else:
        pred_loss = torch.tensor(0.0, device=logits.device)

    # --- 4. Combine ---
    loss = lik_loss + alpha_class * classification_loss + alpha_rank * rank_loss + alpha_pred * pred_loss
    
    return loss, lik_loss, classification_loss, rank_loss, pred_loss



def log_likelihood_loss(logits, time_to_event, event_label, no_censoring=True):
    B, K, T = logits.shape # note that T is num_time_bins here, that may not be consistent with the model naming where T may be sequence length

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


def event_classification_loss(logits, event_label, no_censoring=True):

    # subtract 1 from event_label if no censoring
    event_label = event_label -1 if no_censoring else event_label   
    
    # aggregate over time to get event-level logits
    # event_logits = logits.sum(dim=2)           # [B, K]
    event_logits = torch.logsumexp(logits, dim=2)  # [B, K]
    # event_logp = F.log_softmax(event_logits, dim=1)
    # classification_loss = F.nll_loss(event_logp, event_label.long())                      
    classification_loss = F.cross_entropy(event_logits, event_label.long())
    
    return classification_loss


def ranking_loss(logits, time_to_event, event_label, no_censoring=True, sigma=None, pair_sample_frac=0.5):

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
        event_indices = torch.where(mask_i_k)[0]      # indices of samples with event k
        
        if len(event_indices) == 0:
            continue

        if len(event_indices) > 12000:
            event_indices = event_indices[torch.randperm(len(event_indices))[:12000]]
        n_event = len(event_indices)

        # Gather CIF values at t_i for all samples
        # cif_k has shape (B, T), we want to gather at position t_i for each sample
        cif_k = S[event_indices, k, :]  # (B, T)

        # Build matrices for t and k comparisons:
        t_i_k = t_i[event_indices]   # (N_k,1)
        t_j_k = t_j[:, event_indices]
        mask_time = (t_i_k < t_j_k)  
        num_pairs = mask_time.sum().item()
        if num_pairs == 0:
            continue

        # Create gathering indices: for each sample i, get cif at all t_i values
        # We need to broadcast this operation
        t_i_expand = t_i_k.expand(n_event, n_event)  # (B, B) - each row i has t_i repeated
        t_i_expand = t_i_expand.long()

        # Gather: for each (i,j) pair, get CIF_j at time t_i
        # This is a bit tricky - we need to use advanced indexing
        event_indices_expanded = torch.arange(n_event, device=device).view(1, n_event).expand(n_event, n_event)
        cif_at_ti = cif_k[event_indices_expanded, t_i_expand]  # (B, B)

        risk_i = torch.diagonal(cif_at_ti, 0).unsqueeze(1)  # (B, 1) - diagonal elements
        risk_j = cif_at_ti  # (B, B)

        # Compute pairwise ranking loss: η(risk_i, risk_j) = exp(-(risk_i - risk_j) / sigma)
        # We want risk_i > risk_j, so we penalize when risk_i <= risk_j
        # pairwise_loss = torch.exp(-(risk_i - risk_j) / sigma)  # (B, B)
        pairwise_loss = torch.log(1 + torch.exp(-(risk_i - risk_j))) 
        
        # apply mask to loss
        masked_loss = pairwise_loss * mask_time.float()
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

