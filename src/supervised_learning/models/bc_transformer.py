"""
Behavioural cloning algorithm based on dynamic deephit implementation
"""

from typing import Dict, List, Tuple
import numpy as np
import math
from supervised_learning.models.dynamic_deephit_transformer import TransformerDeepHitTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F

TensorBatch = List[torch.Tensor]
from torch.optim.lr_scheduler import OneCycleLR

class LayerNormChannels(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape: [B, C, T]
        x = x.transpose(1, 2)          # [B, T, C]
        x = self.norm(x)
        return x.transpose(1, 2)       # [B, C, T] back for the next CNN layer



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



class BehaviouralCloneTransformer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        d_model: int, # Replaces hidden_dim
        nhead: int,
        num_encoder_layers: int, # Replaces num_layers
        ff_factor: int, # dim_feedforward = ff_factor * d_model
        dropout: float = 0.3,
        feature_dropout_p: float = 0.0, # Probability of dropping each feature (column) during training
        n_actions: int = 2, # number of target actions to predict
        max_seq_len: int = 72, # For Positional Encoding and padding

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
        self.nhead = int(nhead)
        self.num_encoder_layers = int(num_encoder_layers)
        self.ff_factor = int(ff_factor)

        self.n_actions = int(n_actions)
        
        self.dropout = dropout
        self.feature_dropout_p = feature_dropout_p
        self.max_seq_len = max_seq_len

        # Pre-calculate the fixed-size Causal Mask
        self.register_buffer('causal_mask', self.generate_square_subsequent_mask(max_seq_len))
       
        # feature dropout
        # 1. Feature Dropout (The Column Dropout we discussed)
        self.feature_dropout = FeatureDropout(p=feature_dropout_p)

        # --- 1. Input Embedding ---
        # Map input features to the Transformer's model dimension (d_model)
        self.input_projection = nn.Linear(self.feature_dim, self.d_model)

        # --- 2. Transformer Encoder Stack ---
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

        self.policy_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.d_model, 1),
                nn.Sigmoid()  # Guarantees the continuous dosage outputs sit strictly between 0 and 1
            )    
            for _ in range(self.n_actions)
        ])
        
        self._initialize_weights()

    def _initialize_weights(self):
        # Optional: Initialize weights for better convergence
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    
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
        # 1. Input Projection
        x = self.input_projection(x) # [B, T, d_model]

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
                src_key_padding_mask=~padding_mask.bool() if padding_mask is not None else None
            ) # [B, T, d_model]
        else:
            output = self.transformer_encoder(
                x,
                mask=causal_mask # Causal Mask
            )
        
        # # # 4. Process all timesteps
        # # # Flatten the output to treat every timestep of every sequence as a separate prediction point
        # # Output shape is now [B * T, d_model]
        # all_timesteps_rep = output.reshape(batch_size * seq_len, self.d_model)
        
        # 4. Pooling: Extract the final timestep's latent embedding
        # This vector represents the accumulated history up to hour T
        latent_features = output[:, -1, :] # Shape: [B, d_model]
                   
        # 5. Process through the regression heads to predict continuous actions
        predicted_actions = []
        for head in self.policy_heads:
            # Each head outputs shape [B, 1] due to the final Linear(d_model, 1)
            pred_dose = head(latent_features) 
            predicted_actions.append(pred_dose)

        # Concatenate the list of [B, 1] tensors into a single [B, action_dims] tensor
        predicted_actions = torch.cat(predicted_actions, dim=-1)


        # Return both the bounded predictions (for the loss) and the raw features (for the Critic)
        return predicted_actions, latent_features
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generates an upper-triangular matrix of -inf, which prevents attention to future positions."""
        # PyTorch helper function creates a matrix where T means blocked
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

        
    @torch.no_grad()
    def get_neuron_stats(self, x: torch.Tensor, padding_mask: torch.Tensor = None):
        self.eval()
        stats = {}
        
        # 1. Forward pass to get the pooled latent embedding vector
        # shape: [Batch, d_model] (e.g., [B, 128])
        _, latent_features = self(x, padding_mask)
        
        # 2. Check for Token Uniformity (Is our 128-D embedding space alive?)
        # Calculate variance across the batch for each hidden dimension
        latent_vars = torch.var(latent_features, dim=0)
        
        # If variance across the batch is near 0, the transformer is outputting 
        # the exact same embedding for every patient (Representational Collapse)
        pct_dead_dimensions = (latent_vars < 1e-5).float().mean().item() * 100
        stats["latent_space_dead_dim_pct"] = pct_dead_dimensions
        
        # Average variance across the embedding space (higher is generally healthier)
        stats["latent_space_avg_variance"] = latent_vars.mean().item()

        # 3. Audit the final Dosing Regression Heads (These DO use ReLU!)
        # You can loop through your multi-action policy heads cleanly
        for head_idx, head in enumerate(self.policy_heads):
            # We can trace through the internal layers of each Sequential head
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
                    
        return stats
    

    def get_grad_norm(self):
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm


class BCTransformerTrainer:
    def __init__(
        self,
        model,
        optimizer,
        device = "CUDA" if torch.cuda.is_available() else "CPU",
        loss_fn=None,
        grad_clip=1.0,
        scheduler=None,
        verbose=True,
        l1_params: list = [],
        l1_lambda: float = 0.0,
        ema_alpha: float = 0.99,  # Add ema_alpha argument for moving
        final_timestep_only: bool = True,
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
        self.loss_fn = loss_fn
        self.grad_clip = grad_clip
        self.scheduler = scheduler
        self.verbose = verbose
        self.l1_params = l1_params
        self.l1_lambda = float(l1_lambda)
        self.clip_fraction_ema = 0.0  # Initialize EMA for clipping fraction
        self.ema_alpha = ema_alpha
        self.final_timestep_only = final_timestep_only

        self.total_it = 0

    def train(self, batch: TensorBatch, epoch: int):
        self.total_it += 1

        self.model.train() # set model to train mode

        # unpack batch
        X = batch['features'].to(self.device)
        
        actions = batch['target_actions'].to(self.device)
        
        mask = batch['loss_mask'].to(self.device)

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass
        # risk_logits: list of tensors, where each tensor is [B, T_cap, num_time_bins]
        # covariate_pred: None (or [B, T_cap, num_covariates] if implemented)
        predicted_actions, latent_features = self.model(
            X, 
            padding_mask=mask
        )
        
        criterion = nn.MSELoss()

        loss = criterion(predicted_actions, actions)       

        # Backward pass - compute gradients
        loss.backward()

        # capture gradient norms
        grad_norm = {"raw_grad_norm": self.model.get_grad_norm()}

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

        # Update weights
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        loss_dict = {
            "mse": loss.item(),
        }

        
        return loss_dict, None, grad_norm
    

    def evaluate(self, batch: TensorBatch):
        self.model.eval()  # set model to eval mode
        
        # unpack batch
        X = batch['features'].to(self.device)
        
        actions = batch['target_actions'].to(self.device)
        
        mask = batch['loss_mask'].to(self.device)

        # Forward pass
        with torch.no_grad():
            # Forward pass
            # risk_logits: list of tensors, where each tensor is [B, T_cap, num_time_bins]
            # covariate_pred: None (or [B, T_cap, num_covariates] if implemented)
            predicted_actions, latent_features = self.model(
                X, 
                padding_mask=mask
            )
            
            criterion = nn.MSELoss()

            loss = criterion(predicted_actions, actions)     

            neuron_stats = self.model.get_neuron_stats(X, padding_mask=mask)
                
        # set model back to train mode
        self.model.train()

        loss_dict = {
            "mse": loss.item()
            }

        return loss_dict, None, neuron_stats


def get_trainer(
        config_manager, 
        feature_dim, 
        hyperparams, 
        num_targets=2
        ):

    learning_rate = float(hyperparams['learning_rate'])

    bc_model = BehaviouralCloneTransformer(
        feature_dim=feature_dim+1, # +1 for the local time feature we add in the replay buffer
        d_model=hyperparams['d_model'],               # e.g., 128 or 256
        nhead=hyperparams['nhead'],                   # e.g., 4 or 8
        num_encoder_layers=hyperparams['n_layers'],   # e.g., 3 or 6
        n_actions=num_targets,     # e.g., 2 for binary actions, or more for multi-class
        dropout=hyperparams['dropout'],
        feature_dropout_p=hyperparams['feature_dropout_p'],
        ff_factor=hyperparams['ff_factor'],
        max_seq_len=config_manager.training_config['window_length'],
    ).to(config_manager.training_config['device'])

    # # Collect L1 parameters: classification head weights only
    # l1_params = [
    #     p for name, p in dynamic_deephit_model.named_parameters()
    #     if name.startswith("heads.") and name.endswith(".weight")
    # ]

    bc_optimizer = torch.optim.AdamW(
        bc_model.parameters(),
        lr=learning_rate,
        weight_decay = float(hyperparams['weight_decay'])
    )

    n_epochs = config_manager.n_epochs
    batch_size = int(hyperparams['batch_size'])


    # dataset_size = training_dataset.n_samples  # n windows we sample on each epoch
    # dataset_size = hyperparams['total_steps']
    # steps_per_epoch = (dataset_size + batch_size - 1) // batch_size
    # total_steps = n_epochs * steps_per_epoch
    total_steps = int(hyperparams['total_steps'])

    pct_start = hyperparams['pct_start']
    div_factor = hyperparams['div_factor']
    scheduler = OneCycleLR(
        bc_optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,  # you'll need to pass this
        pct_start=pct_start, 
        div_factor=div_factor,
        anneal_strategy='cos'
    )

    trainer = BCTransformerTrainer(
            bc_model,
            bc_optimizer,
            device=config_manager.training_config['device'],
            loss_fn=bc_loss,
            grad_clip = hyperparams['grad_clip'],
            scheduler=scheduler,
            verbose=True, 
            # l1_params=l1_params,          # ← ADD THIS
            l1_lambda=hyperparams['l1_lambda'],   
        )

    return trainer



def bc_loss(
        predicted_actions, 
        clinician_actions
        ):
    """
    MSE loss between predicted continuous actions and clinician actions
    
    Args:
        predicted_actions: tensor (B, T, A)  -- predicted continuous actions
        clinician_actions: tensor (B, T, A)  -- clinician actions
        mask: optional [batch] or [batch, seq_len] for padded sequences (1=valid, 0=padded)
        alpha: weight for ranking loss
        eps: numerical stability



    Returns:
        scalar loss tensor

    Note that mask logic is opposite what it would be for the model itself - True indicates valid data here.
    We're not using a mask for the model, because we're padding with data, not zeros.

    """
    criterion = nn.MSELoss()
    loss = criterion(predicted_actions, clinician_actions)    
   
    return loss 


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

