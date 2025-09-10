"""Hierarchical aggregation module for fusing row and column attention pathways.

This module combines the outputs from row and column attention mechanisms
to create joint row-column representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class HierarchicalAggregation(nn.Module):
    """Hierarchical aggregation module for fusing attention pathways.
    
    This module combines row and column attention outputs through multiple
    fusion strategies including cross-attention, gated fusion, and adaptive
    weighting mechanisms.
    
    Args:
        d_model: Model dimension
        fusion_type: Type of fusion ('cross_attention', 'gated', 'adaptive')
        n_heads: Number of attention heads for cross-attention
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        d_model: int,
        fusion_type: str = 'adaptive',
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.fusion_type = fusion_type
        self.n_heads = n_heads
        
        if fusion_type == 'cross_attention':
            self._init_cross_attention()
        elif fusion_type == 'gated':
            self._init_gated_fusion()
        elif fusion_type == 'adaptive':
            self._init_adaptive_fusion()
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
            
        # Common components
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
    def _init_cross_attention(self):
        """Initialize cross-attention fusion components."""
        assert self.d_model % self.n_heads == 0
        self.d_k = self.d_model // self.n_heads
        
        # Cross-attention between row and column representations
        self.row_to_col_attn = nn.MultiheadAttention(
            self.d_model, self.n_heads, dropout=0.1, batch_first=True
        )
        self.col_to_row_attn = nn.MultiheadAttention(
            self.d_model, self.n_heads, dropout=0.1, batch_first=True
        )
        
        # Fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(2) / 2)
        
    def _init_gated_fusion(self):
        """Initialize gated fusion components."""
        # Gating mechanism
        self.row_gate = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.Sigmoid()
        )
        self.col_gate = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.Sigmoid()
        )
        
        # Interaction layer
        self.interaction = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.Tanh()
        )
        
    def _init_adaptive_fusion(self):
        """Initialize adaptive fusion components."""
        # Adaptive weighting network
        self.weight_net = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, 2),
            nn.Softmax(dim=-1)
        )
        
        # Feature enhancement
        self.row_enhance = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU()
        )
        self.col_enhance = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU()
        )
        
        # Cross-modal interaction
        self.cross_modal = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model)
        )
        
    def forward(
        self,
        row_output: torch.Tensor,
        col_output: torch.Tensor,
        row_attn_weights: Optional[torch.Tensor] = None,
        col_attn_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass of hierarchical aggregation.
        
        Args:
            row_output: Output from row attention (batch_size, n_samples, n_features, d_model)
            col_output: Output from column attention (batch_size, n_samples, n_features, d_model)
            row_attn_weights: Attention weights from row attention
            col_attn_weights: Attention weights from column attention
            
        Returns:
            Tuple of (fused_output, fusion_info)
        """
        if self.fusion_type == 'cross_attention':
            return self._cross_attention_fusion(row_output, col_output)
        elif self.fusion_type == 'gated':
            return self._gated_fusion(row_output, col_output)
        elif self.fusion_type == 'adaptive':
            return self._adaptive_fusion(row_output, col_output)
            
    def _cross_attention_fusion(
        self,
        row_output: torch.Tensor,
        col_output: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Cross-attention based fusion."""
        batch_size, n_samples, d_model = row_output.shape
        
        # Input is already in the right format for cross-attention
        row_flat = row_output
        col_flat = col_output
        
        # Cross-attention between row and column representations
        row_to_col, row_to_col_weights = self.row_to_col_attn(
            row_flat, col_flat, col_flat
        )
        col_to_row, col_to_row_weights = self.col_to_row_attn(
            col_flat, row_flat, row_flat
        )
        
        # Weighted fusion
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = weights[0] * row_to_col + weights[1] * col_to_row
        
        # Reshape back
        # Output is already in the right format: (batch_size, n_samples, d_model)
        # No reshaping needed
        
        # Apply output projection
        output = self.output_proj(fused)
        output = self.layer_norm(output + row_output + col_output)
        
        fusion_info = {
            'fusion_weights': weights,
            'row_to_col_weights': row_to_col_weights,
            'col_to_row_weights': col_to_row_weights
        }
        
        return output, fusion_info
        
    def _gated_fusion(
        self,
        row_output: torch.Tensor,
        col_output: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Gated fusion mechanism."""
        # Concatenate row and column outputs
        concat_features = torch.cat([row_output, col_output], dim=-1)
        
        # Compute gates
        row_gate = self.row_gate(concat_features)
        col_gate = self.col_gate(concat_features)
        
        # Apply gates
        gated_row = row_output * row_gate
        gated_col = col_output * col_gate
        
        # Interaction term
        interaction = self.interaction(concat_features)
        
        # Combine
        fused = gated_row + gated_col + interaction
        
        # Apply output projection
        output = self.output_proj(fused)
        output = self.layer_norm(output)
        
        fusion_info = {
            'row_gate': row_gate.mean().item(),
            'col_gate': col_gate.mean().item()
        }
        
        return output, fusion_info
        
    def _adaptive_fusion(
        self,
        row_output: torch.Tensor,
        col_output: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Adaptive fusion with learned weighting."""
        # Enhance individual representations
        enhanced_row = self.row_enhance(row_output)
        enhanced_col = self.col_enhance(col_output)
        
        # Concatenate for weight computation
        concat_features = torch.cat([enhanced_row, enhanced_col], dim=-1)
        
        # Compute adaptive weights
        weights = self.weight_net(concat_features)  # (batch, samples, features, 2)
        
        # Apply weights
        weighted_row = enhanced_row * weights[..., 0:1]
        weighted_col = enhanced_col * weights[..., 1:2]
        
        # Cross-modal interaction
        cross_modal = self.cross_modal(concat_features)
        
        # Combine all components
        fused = weighted_row + weighted_col + cross_modal
        
        # Apply output projection
        output = self.output_proj(fused)
        output = self.layer_norm(output + row_output + col_output)
        
        fusion_info = {
            'adaptive_weights': weights.mean(dim=(0, 1, 2)),  # Average weights
            'cross_modal_magnitude': cross_modal.abs().mean().item()
        }
        
        return output, fusion_info


class MultiScaleFusion(nn.Module):
    """Multi-scale fusion for handling different granularities of attention.
    
    This module applies fusion at multiple scales to capture both local
    and global dependencies in the tabular data.
    """
    
    def __init__(self, d_model: int, scales: list = [1, 2, 4]):
        super().__init__()
        self.d_model = d_model
        self.scales = scales
        
        # Scale-specific fusion modules
        self.scale_fusions = nn.ModuleList([
            HierarchicalAggregation(d_model, fusion_type='adaptive')
            for _ in scales
        ])
        
        # Scale combination
        self.scale_combiner = nn.Sequential(
            nn.Linear(d_model * len(scales), d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(
        self,
        row_output: torch.Tensor,
        col_output: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Multi-scale fusion forward pass."""
        batch_size, n_samples, d_model = row_output.shape
        scale_outputs = []
        scale_info = {}
        
        for i, (scale, fusion_module) in enumerate(zip(self.scales, self.scale_fusions)):
            # Apply pooling for different scales
            if scale > 1:
                # Average pooling for larger scales
                kernel_size = min(scale, n_samples)
                stride = max(1, kernel_size // 2)
                
                row_pooled = F.avg_pool2d(
                    row_output.permute(0, 3, 1, 2),
                    kernel_size=(kernel_size, 1),
                    stride=(stride, 1),
                    padding=(kernel_size//2, 0)
                ).permute(0, 2, 3, 1)
                
                col_pooled = F.avg_pool2d(
                    col_output.permute(0, 3, 1, 2),
                    kernel_size=(kernel_size, 1),
                    stride=(stride, 1),
                    padding=(kernel_size//2, 0)
                ).permute(0, 2, 3, 1)
            else:
                row_pooled = row_output
                col_pooled = col_output
                
            # Apply fusion at this scale
            scale_output, info = fusion_module(row_pooled, col_pooled)
            
            # Upsample back to original size if needed
            if scale_output.shape[1] != n_samples:
                scale_output = F.interpolate(
                    scale_output.permute(0, 3, 1, 2),
                    size=(n_samples, n_features),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)
                
            scale_outputs.append(scale_output)
            scale_info[f'scale_{scale}'] = info
            
        # Combine all scales
        combined = torch.cat(scale_outputs, dim=-1)
        output = self.scale_combiner(combined)
        
        return output, scale_info