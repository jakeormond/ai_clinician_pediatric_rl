from typing import Dict, List, Tuple
import numpy as np
import math
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F

TensorBatch = List[torch.Tensor]
from torch.optim.lr_scheduler import OneCycleLR

from src.supervised_learning.dynamic_deephit_evaluation import calculate_c_index


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


class LayerNormChannels(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape: [B, C, T]
        x = x.transpose(1, 2)          # [B, T, C]
        x = self.norm(x)
        return x.transpose(1, 2)       # [B, C, T] back for the next CNN layer


class ResidualCausalStem_V1(nn.Module): # this causes stripiness because of dilated convolution aliasing
    def __init__(self, feature_dim, d_model, kernel_size, dilation, dropout):
        super().__init__()
        # 1. Path A: The Convolutional Trend Extractor
        causal_padding = (kernel_size - 1) * dilation
        self.pad = nn.ConstantPad1d((causal_padding, 0), 0)
        self.conv = nn.Conv1d(
            feature_dim, d_model, kernel_size, 
            dilation=dilation, bias=False
        )
        
        # 2. Path B: The Raw Feature Projection (The Skip Connection)
        self.raw_proj = nn.Linear(feature_dim, d_model)
        
        # 3. Post-Combination Layers
        self.norm = LayerNormChannels(d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, F]
        
        # Trend Extraction
        x_conv = x.transpose(1, 2) # [B, F, T]
        trends = self.conv(self.pad(x_conv)).transpose(1, 2) # [B, T, d_model]
        
        # Raw State Projection
        raw_state = self.raw_proj(x) # [B, T, d_model]
        
        # The Residual Step: Add them together
        # This keeps the "raw" value and "trend" value in the same embedding
        out = trends + raw_state
        
        # Final refinement
        out = out.transpose(1, 2)
        out = self.dropout(self.relu(self.norm(out)))
        return out.transpose(1, 2)



class ResidualCausalStem(nn.Module):
    def __init__(self, feature_dim, d_model, kernel_size, dilation, dropout):
        super().__init__()
        # 1. Path A: Local Feature Mixing (Dilation=1)
        # This prevents the "checkerboard" effect by ensuring no timesteps are skipped
        self.local_conv = nn.Sequential(
            nn.ConstantPad1d((2, 0), 0), # (Left Padding, Right Padding)
            nn.Conv1d(feature_dim, d_model, kernel_size=3, dilation=1, bias=False),
            nn.ReLU()
        )
        
        # 2. Path B: The Dilated Trend Extractor (Dilation=d)
        causal_padding = (kernel_size - 1) * dilation
        self.pad_dilated = nn.ConstantPad1d((causal_padding, 0), 0)
        self.dilated_conv = nn.Conv1d(
            d_model, d_model, kernel_size, 
            dilation=dilation, bias=False
        )
        
        # 3. Path C: The Skip Connection (Projection)
        self.raw_proj = nn.Linear(feature_dim, d_model)
        
        self.norm = nn.LayerNorm(d_model) # Simplified Norm
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, F]
        
        # Step 1: Raw State Projection
        raw_state = self.raw_proj(x) # [B, T, d_model]
        
        # Step 2: Causal Trend Extraction
        x_conv = x.transpose(1, 2) # [B, F, T]
        
        # First pass: local mixing (fills the "dilation gaps")
        feat_mixed = self.local_conv(x_conv) # [B, d_model, T]
        
        # Second pass: dilated trend
        trends = self.dilated_conv(self.pad_dilated(feat_mixed))
        trends = trends.transpose(1, 2) # [B, T, d_model]
        
        # Step 3: Combined Residual
        # Now every timestep in the kernel window contributes to 'trends'
        out = self.norm(trends + raw_state)
        return self.dropout(out)


class FeatureDropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        # 1. Automatic Deactivation:
        # If the model is in .eval() mode OR dropout is 0, skip the logic
        if not self.training or self.p == 0:
            return x
        
        # 2. Training Logic:
        batch_size, _, num_features = x.shape
        # Create a mask: 1 for 'keep', 0 for 'drop'
        mask = torch.bernoulli(torch.full((batch_size, 1, num_features), 1 - self.p)).to(x.device)
        
        # Standard dropout scaling (divide by 1-p) so that the 
        # total energy of the signal stays consistent between train and eval
        return x * mask * (1.0 / (1.0 - self.p))



class TransformerDeepHitMulti(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        d_model: int, # Replaces hidden_dim
        nhead: int,
        num_encoder_layers: int, # Replaces num_layers
        ff_factor: int, # dim_feedforward = ff_factor * d_model
        num_time_bins: int,
        num_targets: int,
        dropout: float = 0.3,
        feature_dropout_p: float = 0.0, # Probability of dropping each feature (column) during training
        max_seq_len: int = 768, # For Positional Encoding and padding
        predict_covariates: list = None,
        prediction_time_horizon: int = 4, # How many steps ahead to predict 
        cnn_kernel_size: int = 5,
        cnn_dilation: int = 2, # Dilation=2 with kernel=5 sees a 9-hour window
    ):
        """
        Args:
            feature_dim: number of covariates per time step (input_size)
            d_model: Transformer feature dimension (replaces hidden_dim)
            nhead: number of attention heads
            num_encoder_layers: number of Transformer Encoder layers
            num_time_bins: number of discrete time bins
            num_targets: number of competing risks (e.g. 2 for death, discharge)
            dropout: dropout rate
            max_seq_len: maximum length of the input window (e.g., 768)
            predict_covariates: list of indices of covariates to predict for regularization
        """
        super().__init__()

        self.feature_dim = int(feature_dim)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_encoder_layers = int(num_encoder_layers)
        self.ff_factor = int(ff_factor)
        self.num_time_bins = int(num_time_bins)
        self.num_targets = int(num_targets)
        self.dropout = dropout
        self.feature_dropout_p = feature_dropout_p
        self.max_seq_len = max_seq_len
        self.predict_covariates = predict_covariates if predict_covariates else []
        self.prediction_time_horizon = prediction_time_horizon

        # Pre-calculate the fixed-size Causal Mask
        self.register_buffer('causal_mask', self.generate_square_subsequent_mask(max_seq_len))
        
        # feature dropout
        # 1. Feature Dropout (The Column Dropout we discussed)
        self.feature_dropout = FeatureDropout(p=feature_dropout_p)

        # ---- NEW: CNN Embedding Stem ---
        self.embedding_stem = ResidualCausalStem(
            feature_dim=self.feature_dim,
            d_model=self.d_model,
            kernel_size=cnn_kernel_size,
            dilation=cnn_dilation,
            dropout=dropout
        )        
        
        # --- 1. Input Embedding ---
        # Map input features to the Transformer's model dimension (d_model)
        self.input_projection = nn.Linear(self.feature_dim, self.d_model)

        # --- 2. Positional Encoding ---
        # Crucial for time-series to embed sequential order
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)

        # --- 3. Transformer Encoder Stack ---
        dim_feedforward = self.ff_factor * self.d_model  # Common choice is 4 times d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.nhead, 
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
            for _ in range(num_targets)
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

        # FEATURE DROPOUT
        # Apply Feature (Column) Dropout first
        x = self.feature_dropout(x)
        
        # --- NEW: 1. Process with CNN Stem ---  

        x = self.embedding_stem(x) # [B, T, d_model]
        
        # 1. Project and encode features
        # x = self.input_projection(x) # [B, T, feature_dim] -> [B, T, d_model]
        
        # 2. Add positional encoding
        # x = self.pos_encoder(x)
        
        # 3. Transformer Encoder
        # The attention mask (src_key_padding_mask) needs to be passed to the encoder.
        # PyTorch Transformer expects mask to be a boolean tensor where True means to IGNORE the key.
        causal_mask = self.causal_mask[:seq_len, :seq_len].to(x.device).bool()
        if padding_mask is not None:
            # Mask should be True where padding exists (you should ensure your padding logic creates this mask)
            output = self.transformer_encoder(
                x, 
                mask=causal_mask, # Causal Mask
                # src_key_padding_mask=padding_mask
                src_key_padding_mask=padding_mask.bool() if padding_mask is not None else None
            ) # [B, T, d_model]
        else:
            output = self.transformer_encoder(
                x,
                mask=causal_mask # Causal Mask
            )
        
        # generate predictions for each timestep
        # 5. Covariate Prediction (If enabled)
        covariate_pred = None
        if self.covariate_predictor:
            # 2. Slice: Use outputs from 0 to T-1 to predict covariates at 1 to T
            # We take all timesteps except the last one for the input
            h_for_pred = output[:, :-self.prediction_time_horizon, :]

            # 3. Generate predictions
            covariate_pred = self.covariate_predictor(h_for_pred)
            
            # 4. Handle Masking - masking handled in trainer/loss function
        #     # If you have a padding mask [batch, seq_len], align it to the predictions
        #     if padding_mask is not None:
        #         # Shift the mask to match the t+1 targets
        #         pred_mask = padding_mask[:, :-self.prediction_time_horizon]
        #         # Note: In PyTorch, padding masks are often True for 'pad' and False for 'data'
        #         # Adjust the logic below to match your mask's boolean logic:
        #         covariate_pred = covariate_pred * (~pred_mask).unsqueeze(-1).float()
            
        # # 4. Process all timesteps
        # # Flatten the output to treat every timestep of every sequence as a separate prediction point
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

            # if any nans in logits, raise error
            if torch.isnan(logits).any():
                raise ValueError("NaN values found in risk_logits during forward pass.")
            
        # return the output for the final timestep only, as that's what we use for prediction and evaluation    
        return risk_logits, covariate_pred, output[:, -1, :] # return the final timestep's output for potential use in loss functions

    @torch.no_grad()
    def get_neuron_stats(self, x: torch.Tensor, padding_mask: torch.Tensor = None):
        self.eval()
        stats = {}

        # 1. Forward pass to get the pooled latent embedding vector (final timestep)
        # shape: [Batch, d_model]
        if padding_mask is not None:
            _, _, latent_features = self(x, padding_mask)
        else:
            _, _, latent_features = self(x)

        # 2. Check for Token Uniformity (Is our d_model embedding space alive?)
        # Calculate variance across the batch for each hidden dimension
        latent_vars = torch.var(latent_features, dim=0)

        # If variance across the batch is near 0, the transformer is outputting
        # the exact same embedding for every patient (Representational Collapse)
        pct_dead_dimensions = (latent_vars < 1e-5).float().mean().item() * 100
        stats["latent_space_dead_dim_pct"] = pct_dead_dimensions

        # Average variance across the embedding space (higher is generally healthier)
        stats["latent_space_avg_variance"] = latent_vars.mean().item()

        # 3. Audit the event-specific heads (these DO use ReLU!)
        for head_idx, head in enumerate(self.heads):
            head_x = latent_features
            relu_idx = 0
            for layer in head:
                head_x = layer(head_x)
                if isinstance(layer, nn.ReLU):
                    # For standard ReLU, being inactive means outputting exactly 0
                    pct_inactive = (head_x == 0).float().mean().item() * 100
                    stats[f"head_{head_idx}_relu_{relu_idx}_inactive_pct"] = pct_inactive

                    # Check for dead neurons in the head's dense mapping
                    layer_vars = torch.var(head_x, dim=0)
                    pct_dead = (layer_vars < 1e-6).float().mean().item() * 100
                    stats[f"head_{head_idx}_layer_{relu_idx}_dead_pct"] = pct_dead
                    relu_idx += 1

        # 4. Audit the covariate predictor, if present (also uses ReLU)
        if self.covariate_predictor is not None:
            head_x = latent_features
            relu_idx = 0
            for layer in self.covariate_predictor:
                head_x = layer(head_x)
                if isinstance(layer, nn.ReLU):
                    pct_inactive = (head_x == 0).float().mean().item() * 100
                    stats[f"covariate_predictor_relu_{relu_idx}_inactive_pct"] = pct_inactive

                    layer_vars = torch.var(head_x, dim=0)
                    pct_dead = (layer_vars < 1e-6).float().mean().item() * 100
                    stats[f"covariate_predictor_layer_{relu_idx}_dead_pct"] = pct_dead
                    relu_idx += 1

        return stats
    
    def get_grad_norm(self):
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm


# def get_grad_norm(model):
#     total_norm = 0
#     for p in model.parameters():
#         if p.grad is not None:
#             param_norm = p.grad.detach().data.norm(2)
#             total_norm += param_norm.item() ** 2
#     total_norm = total_norm ** 0.5
#     return total_norm


class TransformerDeepHitTrainer:
    def __init__(
        self,
        model,
        optimizer,
        device = "CUDA" if torch.cuda.is_available() else "CPU",
        num_targets = 2,
        num_time_bins = 100,
        loss_fn=None,
        grad_clip=1.0,
        alpha_rank = None,
        alpha_pred = None,
        alpha_class = None,
        sigma = None,
        scheduler=None,
        g_sigma_scheduler=None,
        logit_scale = 15.0,

        log_grad_norms_freq = 50,

        verbose=True,
        physiological_feature_type = None,
        l1_params: list = [],
        l1_lambda: float = 0.0,

        alpha_logit_l2: float = 0.0,

        final_timestep_only: bool = True,

        ema_alpha: float = 0.99,  # Add ema_alpha argument for moving
    ):
        """
        Args:
            model: DynamicDeepHitMulti instance
            optimizer: torch.optim optimizer
            device: 'cuda' or 'cpu'
            num_targets: number of competing risks
            num_time_bins: discretized time horizon
            loss_fn: custom loss function (optional)
            grad_clip: gradient clipping value (optional)
            scheduler: lr scheduler (optional)
            verbose: print progress
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.num_targets = num_targets
        self.num_time_bins = num_time_bins
        self.loss_fn = loss_fn
        self.grad_clip = grad_clip
        self.alpha_class = alpha_class
        self.alpha_rank = alpha_rank
        self.alpha_pred = alpha_pred

        self.alpha_logit_l2 = alpha_logit_l2

        self.sigma = sigma
        self.scheduler = scheduler
        self.g_sigma_scheduler = g_sigma_scheduler

        self.logit_scale = logit_scale

        self.verbose = verbose

        self.clip_fraction_ema = 0.0  # Initialize EMA for clipping fraction
        self.ema_alpha = ema_alpha

        self.physiological_feature_type = physiological_feature_type
        self.l1_params = l1_params
        self.l1_lambda = float(l1_lambda)
        self.final_timestep_only = final_timestep_only

        self.log_grad_norms_freq = log_grad_norms_freq  # 0 = off, N = every N steps


        self.total_it = 0


    def _per_loss_grad_norms(self, lik_loss, class_loss, rank_loss, pred_loss):
        """Computes grad norm contribution of each loss term w.r.t. model parameters,
        without populating .grad (safe to call before the real backward pass)."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        terms = {
            'lik': lik_loss,
            'class': self.alpha_class * class_loss,
            'rank': self.alpha_rank * rank_loss,
            'pred': self.alpha_pred * pred_loss,
        }
        norms = {}
        for name, term in terms.items():
            if term.requires_grad:
                grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
                total_norm = sum(g.detach().norm(2).item() ** 2 for g in grads if g is not None) ** 0.5
                norms[f"{name}_grad_norm"] = total_norm
            else:
                norms[f"{name}_grad_norm"] = 0.0
        return norms


    def train(self, batch: TensorBatch, epoch: int):
        self.total_it += 1

        self.model.train() # set model to train mode

        # unpack batch
        # X = batch[0].to(self.device)      # [batch_size, n_ts, n_features]
        X = batch['features'].to(self.device)
        # time_to_event = batch[1].to(self.device)      # [batch_size]
        time_to_event = batch['time_to_event'].to(self.device)
        # event_label = batch[2].to(self.device)      # [batch_size]
        event_label = batch['event'].to(self.device)
        # mask = batch[3].to(self.device)
        # mask = batch['mask'].to(self.device)
        # no longer using a mask for the transformer, all timesteps are valid (padded with data, not zeros)
        # mask is only for loss calculation, and is True where data is valid
        mask = batch['loss_mask'].to(self.device)

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass
        # risk_logits: list of tensors, where each tensor is [B, T_cap, num_time_bins]
        # covariate_pred: None (or [B, T_cap, num_covariates] if implemented)
        risk_logits, covariate_pred, _ = self.model(X, padding_mask=None)
        
        # We need to restructure the risk_logits for the loss function:
        # 1. Stack the list of risk tensors into a single tensor: [B, T_cap, num_targets, num_time_bins]
        logits = torch.stack(risk_logits, dim=2) 

        # Compute log loss and ranking loss
        loss, lik_loss, class_loss, rank_loss, pred_loss, logit_l2_penalty = self.loss_fn(
            logits, 
            covariate_pred, 
            time_to_event,                  
            event_label, 
            X,
            alpha_class = self.alpha_class,
            alpha_rank = self.alpha_rank,
            alpha_pred = self.alpha_pred,
            sigma = self.sigma,
            alpha_logit_l2 = self.alpha_logit_l2,
            g_sigma_scheduler = self.g_sigma_scheduler,
            logit_scale = self.logit_scale,
            epoch = epoch,
            predict_covariates=self.model.predict_covariates,
            predict_horizon=self.model.prediction_time_horizon,
            physiological_feature_type=self.physiological_feature_type,
            mask = mask,
            final_timestep_only = self.final_timestep_only
            )
        
        grad_norm_breakdown = {}
        if self.log_grad_norms_freq and (self.total_it % self.log_grad_norms_freq == 0):
            grad_norm_breakdown = self._per_loss_grad_norms(lik_loss, class_loss, rank_loss, pred_loss)

        # calculate c index here for logging if needed
        if self.final_timestep_only:
            logits = logits[:, -1:, :, :]  # [B, 1, K, D]
            time_to_event = time_to_event[:, -1:]  # [B, 1]
        
        B, T_cap, K, D = logits.shape
        
        # reshape logits to [B*T_cap, K, num_time_bins] for c-index calculation
        logits_c_index = logits.reshape(-1, K, D)
        tte_c_index = time_to_event.reshape(-1)
        # expand event_label to [B*T_cap] 
        event_label_c_index = event_label.unsqueeze(1).expand(-1, T_cap).reshape(-1) - 1   

        c_index_results = calculate_c_index(logits_c_index, tte_c_index, event_label_c_index)
    
        if self.l1_lambda > 0.0 and len(self.l1_params) > 0:
            l1_lambda = self.l1_lambda  # set your value during initialization
            
            l1_penalty = 0
            for name, p in self.model.named_parameters():
                if "heads" in name and "weight" in name:
                    l1_penalty += torch.sum(torch.abs(p))
            
            # l1_penalty = sum(torch.sum(torch.abs(p)) for p in self.l1_params)
            
            loss = loss + l1_lambda * l1_penalty

        # Backward pass - compute gradients
        loss.backward()

        # 1. Capture the raw norm for logging
        # raw_grad_norm = get_grad_norm(self.model)
        grad_norm = {"raw_grad_norm": self.model.get_grad_norm()}
        if self.log_grad_norms_freq and (self.total_it % self.log_grad_norms_freq == 0):
            grad_norm.update(grad_norm_breakdown)

        # determine whether to clip gradients based on grad_clip value
        grad_norm['was_clipped'] = 1.0 if (self.grad_clip and grad_norm['raw_grad_norm'] > self.grad_clip) else 0.0


        # update the moving average of the clipping binary
        self.clip_fraction_ema = self.ema_alpha * self.clip_fraction_ema + (1 - self.ema_alpha) * grad_norm['was_clipped']
        grad_norm['clip_fraction_ema'] = self.clip_fraction_ema
        
        # Gradient clipping if specified
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.grad_clip
            )

        # # Gradient clipping if specified
        # if self.grad_clip is not None:
        #     torch.nn.utils.clip_grad_norm_(
        #         self.model.parameters(), 
        #         self.grad_clip
        #     )

        # Update weights
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        loss_dict = {
            "combined_loss": loss.item(),
            "likelihood_loss": lik_loss.item(),
            "classification_loss": class_loss.item(),
            "ranking_loss": rank_loss.item(),
            "prediction_loss": pred_loss.item(),
            "logit_l2_penalty": logit_l2_penalty.item(),
            "weighted_multi_C": c_index_results['C_weighted_multi'],
            }

        
        return loss_dict, logits, grad_norm
    

    def evaluate(self, batch: TensorBatch, epoch: int):
        self.model.eval()  # set model to eval mode
        
        # unpack batch
        # X = batch[0].to(self.device)      # [batch_size, n_ts, n_features]
        X = batch['features'].to(self.device)
        # time_to_event = batch[1].to(self.device)      # [batch_size]
        time_to_event = batch['time_to_event'].to(self.device)
        # event_label = batch[2].to(self.device)      # [batch_size]
        event_label = batch['event'].to(self.device)
        # mask = batch[3].to(self.device)
        # no loger using a mask for the transformer, all timesteps are valid (padded with data, not zeros)
        # mask = batch['mask'].to(self.device)
        # mask is only for loss calculation, and is True where data is valid
        mask = batch['loss_mask'].to(self.device)

        # Forward pass
        with torch.no_grad():
            risk_logits, covariate_pred, _ = self.model(X, padding_mask=None)
            # We need to restructure the risk_logits for the loss function:
            # 1. Stack the list of risk tensors into a single tensor: [B, T_cap, num_targets, num_time_bins]
            logits = torch.stack(risk_logits, dim=2)          
            
            # Compute log loss and ranking loss
            loss, lik_loss, class_loss, rank_loss, pred_loss, logit_l2_penalty = self.loss_fn(
                logits, 
                covariate_pred, 
                time_to_event,                  
                event_label, 
                X,
                alpha_class = self.alpha_class,
                alpha_rank = self.alpha_rank,
                alpha_pred = self.alpha_pred,
                alpha_logit_l2 = self.alpha_logit_l2,
                sigma = self.sigma,
                g_sigma_scheduler = self.g_sigma_scheduler,
                logit_scale = self.logit_scale,
                epoch = epoch,
                predict_covariates=self.model.predict_covariates,
                predict_horizon=self.model.prediction_time_horizon,
                physiological_feature_type=self.physiological_feature_type,
                mask = mask,
                final_timestep_only = self.final_timestep_only
                )
            
            # calculate c index here for logging if needed
            if self.final_timestep_only:
                logits = logits[:, -1:, :, :]  # [B, 1, K, D]
                time_to_event = time_to_event[:, -1:]  # [B, 1]
            
            B, T_cap, K, D = logits.shape

            # reshape logits to [B*T_cap, K, num_time_bins] for c-index calculation
            logits_c_index = logits.reshape(-1, K, D)
            tte_c_index = time_to_event.reshape(-1)
            # expand event_label to [B*T_cap] 
            event_label_c_index = event_label.unsqueeze(1).expand(-1, T_cap).reshape(-1) - 1   

            c_index_results = calculate_c_index(logits_c_index, tte_c_index, event_label_c_index)

            neuron_stats = self.model.get_neuron_stats(X, padding_mask=None)

            if self.final_timestep_only:
                logits = logits[:, -1:, :, :]  # [B, 1, K, D]

        # set model back to train mode
        self.model.train()

        loss_dict = {
            "combined_loss": loss.item(),
            "likelihood_loss": lik_loss.item(),
            "classification_loss": class_loss.item(),
            "ranking_loss": rank_loss.item(),
            "prediction_loss": pred_loss.item(),
            "logit_l2_penalty": logit_l2_penalty.item(),
            "weighted_multi_C": c_index_results['C_weighted_multi'],
            }

        return loss_dict, logits, neuron_stats


def get_trainer(
        config_manager, 
        feature_dim, 
        hyperparams, 
        training_dataset,
        num_targets = 2,
        ):

    learning_rate = float(hyperparams['learning_rate'])
    n_time_bins = len(config_manager.time_bins) - 1

    dynamic_deephit_model = TransformerDeepHitMulti(
        feature_dim=feature_dim+1, # +1 for the local time feature we add in the replay buffer
        d_model=hyperparams['d_model'],               # e.g., 128 or 256
        nhead=hyperparams['nhead'],                   # e.g., 4 or 8
        num_encoder_layers=hyperparams['num_encoder_layers'],   # e.g., 3 or 6
        num_time_bins=n_time_bins,
        num_targets=num_targets,
        dropout=hyperparams['dropout'],
        feature_dropout_p=hyperparams['feature_dropout_p'],
        ff_factor=hyperparams['ff_factor'],
        max_seq_len=config_manager.training_config['window_length'],
        predict_covariates=training_dataset.physiological_feature_ind
    ).to(config_manager.training_config['device'])

    # # Collect L1 parameters: classification head weights only
    # l1_params = [
    #     p for name, p in dynamic_deephit_model.named_parameters()
    #     if name.startswith("heads.") and name.endswith(".weight")
    # ]

    deephit_optimizer = torch.optim.AdamW(
        dynamic_deephit_model.parameters(),
        lr=learning_rate,
        weight_decay = float(hyperparams['weight_decay'])
    )

    n_epochs = config_manager.n_epochs
    batch_size = int(hyperparams['batch_size'])
    # dataset_size = train_processor.dynamic_deephit_dataset['event'].shape[0] * \
    #     train_processor.dynamic_deephit_dataset['event'].shape[1]  # total number of timesteps across all patients
    dataset_size = training_dataset.n_samples  # n windows we sample on each epoch
    steps_per_epoch = (dataset_size + batch_size - 1) // batch_size
    total_steps = n_epochs * steps_per_epoch

    pct_start = hyperparams['pct_start']
    div_factor = hyperparams['div_factor']
    scheduler = OneCycleLR(
        deephit_optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,  # you'll need to pass this
        pct_start=pct_start, 
        div_factor=div_factor,
        anneal_strategy='cos'
    )

    g_sigma_scheduler = G_SigmaScheduler(
        start_sigma=hyperparams['start_sigma'],
        end_sigma=hyperparams['end_sigma'],
        total_epochs=n_epochs
    )

    trainer = TransformerDeepHitTrainer(
            dynamic_deephit_model,
            deephit_optimizer,
            device=config_manager.training_config['device'],
            num_targets=num_targets,
            num_time_bins=n_time_bins,
            loss_fn=dynamic_deephit_loss,
            grad_clip = hyperparams['grad_clip'],
            alpha_class = hyperparams['alpha_class'], 
            alpha_rank = hyperparams['alpha_rank'],
            alpha_pred = hyperparams['alpha_pred'],
            alpha_logit_l2 = hyperparams['alpha_logit_l2'],
            sigma = hyperparams['sigma'],
            scheduler=scheduler,
            g_sigma_scheduler=g_sigma_scheduler,
            logit_scale = hyperparams['logit_scale'],
            log_grad_norms_freq = hyperparams['log_grad_norms_freq'],
            verbose=True, 
            physiological_feature_type = training_dataset.physiological_feature_type,
            # l1_params=l1_params,          # ← ADD THIS
            l1_lambda=hyperparams['l1_lambda'],   
            final_timestep_only = config_manager.final_timestep_only        
        )

    return trainer



def dynamic_deephit_loss(
        logits, 
        predictions,
        time_to_event, 
        event_label,
        X, # might not need this anymore, check!
        predict_covariates: list = [],
        predict_horizon=None,
        physiological_feature_type: list = [],
        no_censoring=True,
        pair_sample_frac = 0.5,
        alpha_class= None,
        alpha_rank= None, 
        alpha_pred = None,
        alpha_logit_l2 = 0.0,   
        sigma=None,
        mask=None,
        eps=1e-8,
        g_sigma_scheduler=None,
        logit_scale=15.0,
        epoch=None,
        final_timestep_only = True,
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

    Note that mask logic is opposite what it would be for the model itself - True indicates valid data here.
    We're not using a mask for the model, because we're padding with data, not zeros.

    """
    # before flattening tensors, deal with predictions, as they have one less timestep per window
    targets_shifted = X[:, predict_horizon:, :]  # Shifted targets for prediction loss
    mask_shifted = mask[:, :-predict_horizon]    # Shifted mask for prediction loss
    valid_preds = predictions[mask_shifted]  # Filter out padded predictions
    valid_targets = targets_shifted[mask_shifted]  # Corresponding valid targets
    valid_targets = valid_targets[:, predict_covariates]  # Select only predicted covariates

    # flatten tensors where needed
    # 2. Reshape to [B * T_cap, num_targets, num_time_bins] for the loss function
    #    (Assuming your existing loss function expects a flattened batch/time dimension)
    if final_timestep_only:
        # only keep the final timestep's logits for each patient
        logits = logits[:, -1:, :, :]  # [B, 1, R, D]
        time_to_event = time_to_event[:, -1:]  # [B]
        event_label = event_label      # [B]
        mask = mask[:, -1:]            # [B, 1]
    
    B, T_cap, R, D = logits.shape
    logits_flat = logits.reshape(B * T_cap, R, D)
    
    # 3. Create flattened versions of the targets and masks
    time_to_event_flat = time_to_event.reshape(B * T_cap)
    # if event_label has shape [B], we need to expand it to [B, T_cap] first
    if event_label.dim() == 1:
        event_label = event_label.unsqueeze(1).expand(-1, T_cap)  # [B, T_cap]
   
    event_label_flat = event_label.reshape(B * T_cap)
    mask_flat = mask.reshape(B * T_cap) # This is the key for excluding padded steps

    # --- 1. Filter Padded Timesteps ---
    # We only want to compute loss for timesteps that are NOT padding.
    # mask is False for valid steps.
    valid_indices = mask_flat

    # Filter all relevant tensors to keep only the valid "active" hours
    v_logits = logits_flat[valid_indices]    
    v_time_to_event = time_to_event_flat[valid_indices]
    v_event_label = event_label_flat[valid_indices]

    # --- Logit L2 penalty (structural fix for unbounded logit growth as sigma anneals) ---
    logit_l2_penalty = v_logits.pow(2).mean()

    # --- 2. Compute DeepHit Component Losses ---
    # Because we filtered, these functions now see a "batch" of size N_valid
    # lik_loss = log_likelihood_loss(v_logits, v_time_to_event, v_event_label, no_censoring)
    lik_loss = gaussian_log_likelihood_loss(v_logits, v_time_to_event, 
                                            v_event_label, no_censoring=True, 
                                            g_sigma_scheduler=g_sigma_scheduler, 
                                            epoch=epoch, logit_scale=logit_scale)

    
    classification_loss = event_classification_loss(v_logits, v_event_label, no_censoring)
    
    # Ranking loss evaluates pairs. With multi-timestep, it compares risk 
    # across different patients AND different hours within the stay.
    rank_loss = ranking_loss(v_logits, v_time_to_event, v_event_label, no_censoring=no_censoring, sigma=sigma)

    # --- 3. Auxiliary Prediction Loss ---
    # If you are doing step-ahead covariate prediction, you need to filter 
    # the feature tensors as well.      
    if alpha_pred > 0 and len(predict_covariates) > 0:
        pred_loss = prediction_loss(
                valid_targets,
                valid_preds,  
                physiological_feature_type
                ) 
    else:
        pred_loss = torch.tensor(0.0, device=logits.device)

    # --- 4. Combine ---
    # loss = lik_loss + alpha_class * classification_loss + alpha_rank * rank_loss + alpha_pred * pred_loss
    loss = lik_loss + alpha_class * classification_loss + alpha_rank * rank_loss + \
        alpha_pred * pred_loss + alpha_logit_l2 * logit_l2_penalty
    
    return loss, lik_loss, classification_loss, rank_loss, pred_loss, logit_l2_penalty


class G_SigmaScheduler:
    def __init__(self, start_sigma=2.0, end_sigma=0.5, total_epochs=100):
        self.start_g_sigma = start_sigma
        self.end_g_sigma = end_sigma
        self.total_epochs = total_epochs

    def get_sigma(self, epoch):
        # Exponential decay is usually most effective
        # sigma = start * (end/start) ^ (epoch/total)
        if epoch >= self.total_epochs:
            return self.end_g_sigma
        
        if epoch <= 0:
            return self.start_g_sigma

        sigma = self.start_g_sigma * (self.end_g_sigma / self.start_g_sigma) ** (epoch / self.total_epochs)
        return sigma



def gaussian_log_likelihood_loss(logits, time_to_event, event_label, 
                                 g_sigma_scheduler=None, epoch=None, 
                                 no_censoring=True, logit_scale=15.0):
    """
    logits: (B, K, T) - K event types, T time bins
    time_to_event: (B,) - Long tensor of ground truth bins
    event_label: (B,) - Long tensor of event types (1 to K)
    sigma: Float - Controls the 'spread' of the reward. 
                   Higher sigma = more credit for near misses.
    """
    B, K, T = logits.shape
    g_sigma = g_sigma_scheduler.get_sigma(epoch) if g_sigma_scheduler and epoch is not None else 1.0
    
    # Adjust labels if 1-indexed
    event_label = event_label - 1 if no_censoring else event_label   
    
    # 1. Create a coordinate grid for the T time bins (0, 1, 2, ..., T-1)
    # Shape: (T,)
    time_bins = torch.arange(T, device=logits.device).float()
    
    # 2. Calculate the Gaussian "bump" for the correct event head
    # We want (B, T) where each row is a Gaussian centered at time_to_event[i]
    # Formula: exp(-0.5 * ((bin - target) / sigma)^2)
    diff = time_bins.view(1, T) - time_to_event.view(B, 1).float()
    target_dist = torch.exp(-0.5 * (diff / g_sigma)**2)
    
    # Normalize so each row sums to 1.0 (to make it a valid probability distribution)
    target_dist = target_dist / (target_dist.sum(dim=1, keepdim=True) + 1e-8)
    
    # 3. Place these bumps into a full (B, K, T) target tensor
    # Initialize with zeros
    full_target = torch.zeros_like(logits)
    
    # Scatter the Gaussian bumps into the correct event channel (k) for each batch
    # This ensures the model is penalized for picking the wrong event type AND the wrong time
    for i in range(B):
        full_target[i, event_label[i], :] = target_dist[i]
    
    # 4. Calculate Loss
    logits = logit_scale * torch.tanh(logits / logit_scale)
    # Flatten both to (B, K*T)
    full_target_flat = full_target.view(B, K * T)
    logp_flat = F.log_softmax(logits.view(B, K * T), dim=1)
    
    # Cross Entropy with soft targets: -sum(target * log_prob)
    # This naturally rewards 'close' predictions because log_prob will be 
    # weighted by the Gaussian values in target_dist.
    lik_loss = -(full_target_flat * logp_flat).sum(dim=1).mean()
    
    return lik_loss


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
        pairwise_loss = torch.log(1 + torch.exp(-(risk_i - risk_j)/sigma)) 
        
        # apply mask to loss
        masked_loss = pairwise_loss * mask_time.float()
        rank_loss += masked_loss.sum()
        total_pairs += num_pairs
        
    if total_pairs > 0:
        rank_loss = rank_loss / total_pairs
    else:
        rank_loss = torch.tensor(0.0, device=device)
    
    return rank_loss


def prediction_loss(X, predictions, physiological_feature_type=None):
    """
    Compute the step-ahead prediction loss (L3 in DeepHit paper).
    
    Args:
        predictions: [batch * seq_len-1, num_predict_covariates] - predicted next-step values
        X: [batch * seq_len-1, feature_dim] - original input sequences
        continuous_indices: list of indices (within predict_covariates) that are continuous
                          (if None, assumes all are continuous)
    
    Returns:
        loss: scalar prediction loss

    NOTE: NEED TO IMPLEMENT PROPER ORDINAL LOSS
    """
    # Compute loss based on covariate type
    binary_indices = [i for i, val in enumerate(physiological_feature_type) if val == 'binary']
    continuous_indices = [i for i, val in enumerate(physiological_feature_type) if val in ['continuous', 'ordinal']]

    # continuous - MSE
    if len(continuous_indices) > 0:
        continuous_loss = F.mse_loss(
            predictions[:, continuous_indices],
            X[:, continuous_indices],
            reduction='none'
        )  # [N, num_cont]
    else:
        continuous_loss = torch.zeros((predictions.shape[0], 0), device=predictions.device, dtype=predictions.dtype)

    # binary - BCE with logits
    if len(binary_indices) > 0:
        target_binary = X[:, binary_indices]
        binary_loss = F.binary_cross_entropy_with_logits(
            predictions[:, binary_indices],
            target_binary,
            reduction='none'
        )  # [N, num_bin]
    else:
        binary_loss = torch.zeros((predictions.shape[0], 0), device=predictions.device, dtype=predictions.dtype)

    # Combine along feature dimension (last dim)
    loss = torch.cat([continuous_loss, binary_loss], dim=-1)
    
    # Apply mask and compute mean   
    total_loss = loss.sum()
    n_timesteps = loss.shape[0] * loss.shape[1]
    
    return total_loss / (n_timesteps + 1e-8)

