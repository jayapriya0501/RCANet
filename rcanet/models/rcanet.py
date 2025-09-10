"""RCANet: Row-Column Attention Networks for Tabular Data.

Main architecture implementation that combines row attention, column attention,
and hierarchical fusion for enhanced representation learning on tabular data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from .attention import RowAttention, ColumnAttention
from .fusion import HierarchicalAggregation, MultiScaleFusion
from .contrastive import ContrastiveLoss, RowColumnContrastive


class RCANet(nn.Module):
    """Row-Column Attention Network for tabular data.
    
    A dual-axis transformer architecture that models bidirectional interactions
    between rows and columns in tabular datasets through specialized attention
    mechanisms and hierarchical fusion.
    
    Args:
        input_dim: Input feature dimension
        d_model: Model dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        fusion_type: Type of fusion ('cross_attention', 'gated', 'adaptive')
        use_contrastive: Whether to use contrastive learning
        dropout: Dropout probability
        activation: Activation function
        layer_norm_eps: Layer normalization epsilon
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        fusion_type: str = 'adaptive',
        use_contrastive: bool = True,
        dropout: float = 0.1,
        activation: str = 'relu',
        layer_norm_eps: float = 1e-5,
        max_seq_len: int = 1000,
        use_multi_scale: bool = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.use_contrastive = use_contrastive
        self.use_multi_scale = use_multi_scale
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model, eps=layer_norm_eps),
            nn.Dropout(dropout)
        )
        
        # Feature type embeddings (for different data types)
        self.feature_type_embedding = nn.Embedding(10, d_model)  # Support up to 10 feature types
        
        # RCANet layers
        self.layers = nn.ModuleList([
            RCANetLayer(
                d_model=d_model,
                n_heads=n_heads,
                fusion_type=fusion_type,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                use_multi_scale=use_multi_scale
            )
            for _ in range(n_layers)
        ])
        
        # Output layers
        self.output_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Contrastive learning components
        if use_contrastive:
            self.contrastive_loss = RowColumnContrastive(
                d_model=d_model,
                temperature=0.1,
                projection_dim=128
            )
            
        # Classification/regression heads (can be customized)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # Binary classification by default
        )
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
                
    def forward(
        self,
        x: torch.Tensor,
        feature_types: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_contrastive: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of RCANet.
        
        Args:
            x: Input tensor of shape (batch_size, n_samples, n_features)
            feature_types: Feature type indices (batch_size, n_features)
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            return_contrastive: Whether to compute contrastive loss
            
        Returns:
            Dictionary containing model outputs
        """
        batch_size, n_samples, n_features = x.shape
        
        # Input embedding - reshape to (batch * samples, features) for linear layer
        x_reshaped = x.view(-1, n_features)  # (batch * samples, features)
        x_embedded = self.input_embedding(x_reshaped)  # (batch * samples, d_model)
        x_embedded = x_embedded.view(batch_size, n_samples, self.d_model)  # (batch, samples, d_model)
        
        # Add feature type embeddings if provided
        if feature_types is not None:
            type_emb = self.feature_type_embedding(feature_types)  # (batch, features, d_model)
            type_emb = type_emb.unsqueeze(1).expand(-1, n_samples, -1, -1)
            x_embedded = x_embedded + type_emb
            
        # Pass through RCANet layers
        hidden_states = x_embedded
        all_attention_weights = []
        all_fusion_info = []
        
        for layer in self.layers:
            layer_output = layer(
                hidden_states,
                mask=mask,
                return_attention=return_attention
            )
            
            hidden_states = layer_output['hidden_states']
            
            if return_attention:
                all_attention_weights.append({
                    'row_attention': layer_output['row_attention'],
                    'col_attention': layer_output['col_attention']
                })
                all_fusion_info.append(layer_output['fusion_info'])
                
        # Final layer normalization
        hidden_states = self.output_norm(hidden_states)
        
        # Global pooling for classification
        pooled_output = torch.mean(hidden_states, dim=1)  # (batch, d_model)
        
        # Classification output
        logits = self.classifier(pooled_output)
        
        # Prepare output dictionary
        outputs = {
            'logits': logits,
            'hidden_states': hidden_states,
            'pooled_output': pooled_output
        }
        
        # Add attention weights if requested
        if return_attention:
            outputs['attention_weights'] = all_attention_weights
            outputs['fusion_info'] = all_fusion_info
            
        # Compute contrastive loss if requested
        if return_contrastive and self.use_contrastive:
            # Extract row and column representations
            row_repr = torch.mean(hidden_states, dim=2)  # (batch, samples, d_model)
            col_repr = torch.mean(hidden_states, dim=1)  # (batch, features, d_model)
            
            contrastive_output = self.contrastive_loss(row_repr, col_repr)
            outputs['contrastive_loss'] = contrastive_output
            
        return outputs
        
    def get_attention_maps(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Get attention maps for visualization."""
        with torch.no_grad():
            outputs = self.forward(x, return_attention=True, **kwargs)
            
        attention_maps = {
            'row_attention_maps': [],
            'col_attention_maps': []
        }
        
        for layer_attn in outputs['attention_weights']:
            attention_maps['row_attention_maps'].append(layer_attn['row_attention'])
            attention_maps['col_attention_maps'].append(layer_attn['col_attention'])
            
        return attention_maps
        
    def extract_features(self, x: torch.Tensor, layer_idx: int = -1, **kwargs) -> torch.Tensor:
        """Extract features from a specific layer."""
        with torch.no_grad():
            outputs = self.forward(x, **kwargs)
            
        if layer_idx == -1:
            return outputs['hidden_states']
        else:
            # Would need to modify forward to return intermediate states
            raise NotImplementedError("Layer-specific feature extraction not implemented")


class RCANetLayer(nn.Module):
    """Single RCANet layer with row attention, column attention, and fusion."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        fusion_type: str = 'adaptive',
        dropout: float = 0.1,
        activation: str = 'relu',
        layer_norm_eps: float = 1e-5,
        use_multi_scale: bool = False
    ):
        super().__init__()
        
        # Row and column attention modules
        self.row_attention = RowAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )
        
        self.col_attention = ColumnAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Fusion module
        if use_multi_scale:
            self.fusion = MultiScaleFusion(d_model=d_model)
        else:
            self.fusion = HierarchicalAggregation(
                d_model=d_model,
                fusion_type=fusion_type,
                n_heads=n_heads,
                dropout=dropout
            )
            
        # Feed-forward network
        self.ffn = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_model * 4,
            dropout=dropout,
            activation=activation
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of RCANet layer."""
        # Row attention
        row_output, row_attn_weights = self.row_attention(x, mask)
        
        # Column attention
        col_output, col_attn_weights = self.col_attention(x, mask)
        
        # Fusion
        fused_output, fusion_info = self.fusion(row_output, col_output)
        
        # First residual connection and normalization
        x = self.norm1(x + fused_output)
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        
        # Second residual connection and normalization
        output = self.norm2(x + ffn_output)
        
        # Prepare output
        layer_output = {'hidden_states': output}
        
        if return_attention:
            layer_output.update({
                'row_attention': row_attn_weights,
                'col_attention': col_attn_weights,
                'fusion_info': fusion_info
            })
            
        return layer_output


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of FFN."""
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class RCANetConfig:
    """Configuration class for RCANet."""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        fusion_type: str = 'adaptive',
        use_contrastive: bool = True,
        dropout: float = 0.1,
        activation: str = 'relu',
        layer_norm_eps: float = 1e-5,
        max_seq_len: int = 1000,
        use_multi_scale: bool = False,
        num_classes: int = 2,
        task_type: str = 'classification'  # 'classification' or 'regression'
    ):
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.fusion_type = fusion_type
        self.use_contrastive = use_contrastive
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.max_seq_len = max_seq_len
        self.use_multi_scale = use_multi_scale
        self.num_classes = num_classes
        self.task_type = task_type
        
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return self.__dict__
        
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'RCANetConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


def create_rcanet_model(
    config: Union[RCANetConfig, Dict],
    pretrained: bool = False,
    pretrained_path: Optional[str] = None
) -> RCANet:
    """Factory function to create RCANet model.
    
    Args:
        config: Model configuration
        pretrained: Whether to load pretrained weights
        pretrained_path: Path to pretrained model
        
    Returns:
        RCANet model instance
    """
    if isinstance(config, dict):
        config = RCANetConfig.from_dict(config)
        
    model = RCANet(
        input_dim=config.input_dim,
        d_model=config.hidden_dim,
        n_layers=config.num_layers,
        n_heads=config.num_heads,
        fusion_type=config.fusion_strategy,
        use_contrastive=config.use_contrastive,
        dropout=config.dropout,
        activation=config.activation,
        layer_norm_eps=getattr(config, 'layer_norm_eps', 1e-5),
        max_seq_len=config.max_sequence_length,
        use_multi_scale=getattr(config, 'use_multi_scale', False)
    )
    
    # Adjust classifier for task type
    if config.task_type == 'classification':
        model.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
        )
    elif config.task_type == 'regression':
        model.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
    if pretrained and pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
    return model