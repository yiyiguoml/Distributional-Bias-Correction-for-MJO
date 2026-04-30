"""
DBC (Deep Bias Correction) Transformer model without metadata embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NonCrossingQuantileHead(nn.Module):
    """
    Quantile head that ensures non-crossing quantiles.
    Outputs base + cumulative positive increments.
    """
    def __init__(self, input_dim, n_quantiles, min_increment=1e-4):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.min_increment = min_increment

        # Base quantile (lowest)
        self.base_layer = nn.Linear(input_dim, 2)  # 2 for RMM1 and RMM2

        # Increments between quantiles
        self.increment_layer = nn.Linear(input_dim, 2 * (n_quantiles - 1))

    def forward(self, x):
        """
        Args:
            x: (B, input_dim) features

        Returns:
            quantiles: (B, 2, n_quantiles) non-crossing quantiles
        """
        B = x.shape[0]

        # Base quantile
        base = self.base_layer(x)  # (B, 2)

        # Positive increments
        increments_raw = self.increment_layer(x)  # (B, 2*(n_quantiles-1))
        increments = F.softplus(increments_raw) + self.min_increment
        increments = increments.view(B, 2, self.n_quantiles - 1)

        # Cumulative sum for non-crossing
        quantiles = torch.zeros(B, 2, self.n_quantiles, device=x.device)
        quantiles[:, :, 0] = base

        for i in range(1, self.n_quantiles):
            quantiles[:, :, i] = quantiles[:, :, i-1] + increments[:, :, i-1]

        return quantiles


class DBCTransformerNoMeta(nn.Module):
    """
    Transformer for DBC (Deep Bias Correction) WITHOUT metadata embeddings.
    Only uses sequence input (rmm1, rmm2, amplitude).
    """
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=256,
                 dropout=0.1, n_quantiles=7, max_seq_len=64):
        super().__init__()
        self.d_model = d_model
        self.n_quantiles = n_quantiles
        self.max_seq_len = max_seq_len

        # Input projection (3 features: rmm1, rmm2, amplitude)
        self.input_proj = nn.Linear(3, d_model)

        # Positional encoding
        self.pos_encoder = nn.Embedding(max_seq_len, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        # Non-crossing quantile head
        self.quantile_head = NonCrossingQuantileHead(d_model, n_quantiles)

    def forward(self, rmm1, rmm2, amp, mask):
        """
        Args:
            rmm1: (B, T) RMM1 forecast sequence
            rmm2: (B, T) RMM2 forecast sequence
            amp: (B, T) Amplitude sequence
            mask: (B, T) Attention mask (1 = valid, 0 = padding)

        Returns:
            quantiles: (B, 2, n_quantiles) predicted quantiles for RMM1 and RMM2
        """
        B, T = rmm1.shape

        # Stack inputs
        x = torch.stack([rmm1, rmm2, amp], dim=-1)  # (B, T, 3)
        x = self.input_proj(x)  # (B, T, d_model)

        # Add positional encoding
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_encoder(pos)

        # Attention mask (True = ignore)
        attn_mask = (mask == 0)

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Get last valid position for each sequence
        seq_lens = mask.sum(dim=1).long() - 1
        batch_indices = torch.arange(B, device=x.device)
        x_last = x[batch_indices, seq_lens]  # (B, d_model)

        # Output projection
        features = self.output_proj(x_last)

        # Get quantile predictions
        quantile_predictions = self.quantile_head(features)  # (B, 2, n_quantiles)

        return quantile_predictions
