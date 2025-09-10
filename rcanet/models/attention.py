"""Attention mechanisms for RCANet.

Implements Row Attention and Column Attention modules for dual-axis
transformer architecture on tabular data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RowAttention(nn.Module):
    """Row Attention module for capturing inter-sample relationships.
    
    This module applies attention across rows (samples) to capture dependencies
    between different instances in the dataset.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
        temperature: Temperature scaling for attention weights
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        temperature: float = 1.0
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.temperature = temperature
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Position encoding for row positions
        self.pos_encoding = PositionalEncoding(d_model, max_len=5000)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of Row Attention.
        
        Args:
            x: Input tensor of shape (batch_size, n_samples, n_features, d_model)
            mask: Optional attention mask
            
        Returns:
            Tuple of (attended_output, attention_weights)
        """
        batch_size, n_samples, d_model = x.shape
        
        # For row attention, we attend across samples
        # Input is already in the right format: (batch_size, n_samples, d_model)
        x_reshaped = x
        
        # Add positional encoding
        x_pos = self.pos_encoding(x_reshaped)
        
        # Apply multi-head attention across samples (rows)
        attended, attn_weights = self._multi_head_attention(
            x_pos, x_pos, x_pos, mask
        )
        
        # Output is already in the right format: (batch_size, n_samples, d_model)
        # No reshaping needed
        attended = attended
        
        # Residual connection and layer norm
        output = self.layer_norm(x + attended)
        
        return output, attn_weights
    
    def _multi_head_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-head attention computation."""
        batch_size, seq_len, d_model = query.shape
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(self.d_k) * self.temperature)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear projection
        output = self.w_o(attended)
        
        return output, attn_weights


class ColumnAttention(nn.Module):
    """Column Attention module for capturing cross-feature dependencies.
    
    This module applies attention across columns (features) to model
    relationships between different attributes.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
        feature_interaction: Whether to use feature interaction mechanism
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        feature_interaction: bool = True
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.feature_interaction = feature_interaction
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Feature interaction mechanism
        if feature_interaction:
            self.feature_gate = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, d_model),
                nn.Sigmoid()
            )
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of Column Attention.
        
        Args:
            x: Input tensor of shape (batch_size, n_samples, n_features, d_model)
            mask: Optional attention mask
            
        Returns:
            Tuple of (attended_output, attention_weights)
        """
        batch_size, n_samples, d_model = x.shape
        
        # For column attention, we attend across the feature dimension
        # Input: (batch_size, n_samples, d_model)
        # We keep the same format but attend differently
        x_reshaped = x
        
        # Apply multi-head attention across features (columns)
        attended, attn_weights = self._multi_head_attention(
            x_reshaped, x_reshaped, x_reshaped, mask
        )
        
        # Feature interaction gating
        if self.feature_interaction:
            gate = self.feature_gate(attended)
            attended = attended * gate
        
        # Output is already in the right format: (batch_size, n_samples, d_model)
        # No reshaping needed
        attended = attended
        
        # Residual connection and layer norm
        output = self.layer_norm(x + attended)
        
        return output, attn_weights
    
    def _multi_head_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-head attention computation."""
        batch_size, seq_len, d_model = query.shape
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear projection
        output = self.w_o(attended)
        
        return output, attn_weights


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        return x + self.pe[:x.size(1), :].transpose(0, 1)