"""Visualization utilities for RCANet.

Provides functions for visualizing attention maps, model architecture,
training progress, and data analysis.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Rectangle
    from matplotlib.colors import LinearSegmentedColormap
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("Matplotlib/Seaborn not available. Install for visualization support.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    warnings.warn("Plotly not available. Install for interactive visualization support.")


def plot_attention_maps(
    attention_weights: torch.Tensor,
    feature_names: Optional[List[str]] = None,
    sample_indices: Optional[List[int]] = None,
    attention_type: str = 'both',  # 'row', 'column', 'both'
    max_samples: int = 4,
    figsize: Tuple[int, int] = (20, 12),
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """Plot attention heatmaps for row and column attention.
    
    Args:
        attention_weights: Attention weights [batch, heads, seq, seq] or dict with 'row' and 'col'
        feature_names: Names of features
        sample_indices: Specific samples to plot
        attention_type: Type of attention to plot
        max_samples: Maximum number of samples to plot
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure or None if plotting not available
    """
    if not PLOTTING_AVAILABLE:
        warnings.warn("Plotting not available. Install matplotlib and seaborn.")
        return None
        
    # Handle different input formats
    if isinstance(attention_weights, dict):
        row_attention = attention_weights.get('row')
        col_attention = attention_weights.get('col')
    else:
        # Assume single attention tensor
        row_attention = attention_weights
        col_attention = attention_weights
        
    # Convert to numpy
    if isinstance(row_attention, torch.Tensor):
        row_attention = row_attention.detach().cpu().numpy()
    if isinstance(col_attention, torch.Tensor):
        col_attention = col_attention.detach().cpu().numpy()
        
    batch_size, n_heads, seq_len, _ = row_attention.shape
    
    # Select samples to plot
    if sample_indices is None:
        sample_indices = list(range(min(max_samples, batch_size)))
    else:
        sample_indices = sample_indices[:max_samples]
        
    n_samples = len(sample_indices)
    
    # Determine subplot layout
    if attention_type == 'both':
        n_cols = n_heads * 2  # Row and column for each head
        subplot_titles = []
        for head in range(n_heads):
            subplot_titles.extend([f'Row Head {head}', f'Col Head {head}'])
    else:
        n_cols = n_heads
        subplot_titles = [f'Head {head}' for head in range(n_heads)]
        
    # Create subplots
    fig, axes = plt.subplots(n_samples, n_cols, figsize=figsize)
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
        
    for i, sample_idx in enumerate(sample_indices):
        col_idx = 0
        
        for head in range(n_heads):
            if attention_type in ['row', 'both']:
                ax = axes[i, col_idx] if n_samples > 1 else axes[col_idx]
                
                # Plot row attention heatmap
                attention_map = row_attention[sample_idx, head]
                
                im = ax.imshow(attention_map, cmap='Blues', aspect='auto', vmin=0, vmax=1)
                
                # Set labels
                if feature_names and len(feature_names) == seq_len:
                    ax.set_xticks(range(0, seq_len, max(1, seq_len // 10)))
                    ax.set_yticks(range(0, seq_len, max(1, seq_len // 10)))
                    if seq_len <= 20:
                        ax.set_xticklabels([feature_names[i] for i in range(0, seq_len, max(1, seq_len // 10))], 
                                         rotation=45, ha='right')
                        ax.set_yticklabels([feature_names[i] for i in range(0, seq_len, max(1, seq_len // 10))])
                        
                ax.set_title(f'Sample {sample_idx}, Row Head {head}')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                col_idx += 1
                
            if attention_type in ['column', 'both']:
                ax = axes[i, col_idx] if n_samples > 1 else axes[col_idx]
                
                # Plot column attention heatmap
                attention_map = col_attention[sample_idx, head]
                
                im = ax.imshow(attention_map, cmap='Reds', aspect='auto', vmin=0, vmax=1)
                
                # Set labels
                if feature_names and len(feature_names) == seq_len:
                    ax.set_xticks(range(0, seq_len, max(1, seq_len // 10)))
                    ax.set_yticks(range(0, seq_len, max(1, seq_len // 10)))
                    if seq_len <= 20:
                        ax.set_xticklabels([feature_names[i] for i in range(0, seq_len, max(1, seq_len // 10))], 
                                         rotation=45, ha='right')
                        ax.set_yticklabels([feature_names[i] for i in range(0, seq_len, max(1, seq_len // 10))])
                        
                ax.set_title(f'Sample {sample_idx}, Col Head {head}')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                col_idx += 1
                
    plt.suptitle('RCANet Attention Maps', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig


def plot_attention_statistics(
    attention_weights: Union[torch.Tensor, Dict[str, torch.Tensor]],
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """Plot attention statistics and distributions.
    
    Args:
        attention_weights: Attention weights tensor or dict
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure or None if plotting not available
    """
    if not PLOTTING_AVAILABLE:
        warnings.warn("Plotting not available. Install matplotlib and seaborn.")
        return None
        
    # Handle different input formats
    if isinstance(attention_weights, dict):
        row_attention = attention_weights.get('row')
        col_attention = attention_weights.get('col')
        has_both = row_attention is not None and col_attention is not None
    else:
        row_attention = attention_weights
        col_attention = None
        has_both = False
        
    # Convert to numpy
    if isinstance(row_attention, torch.Tensor):
        row_attention = row_attention.detach().cpu().numpy()
    if col_attention is not None and isinstance(col_attention, torch.Tensor):
        col_attention = col_attention.detach().cpu().numpy()
        
    # Create subplots
    n_rows = 2 if has_both else 1
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
        
    def plot_attention_stats(attention, axes_row, title_prefix):
        """Plot statistics for a single attention type."""
        # Flatten attention for statistics
        attention_flat = attention.reshape(-1, attention.shape[-1])
        
        # 1. Attention entropy distribution
        entropies = []
        for att in attention_flat:
            att_norm = att + 1e-8
            att_norm = att_norm / np.sum(att_norm)
            entropy = -np.sum(att_norm * np.log(att_norm))
            entropies.append(entropy)
            
        axes_row[0].hist(entropies, bins=50, alpha=0.7, edgecolor='black')
        axes_row[0].set_title(f'{title_prefix} Attention Entropy')
        axes_row[0].set_xlabel('Entropy')
        axes_row[0].set_ylabel('Frequency')
        
        # 2. Max attention distribution
        max_attention = np.max(attention, axis=-1).flatten()
        axes_row[1].hist(max_attention, bins=50, alpha=0.7, edgecolor='black')
        axes_row[1].set_title(f'{title_prefix} Max Attention')
        axes_row[1].set_xlabel('Max Attention Value')
        axes_row[1].set_ylabel('Frequency')
        
        # 3. Attention variance distribution
        attention_var = np.var(attention, axis=-1).flatten()
        axes_row[2].hist(attention_var, bins=50, alpha=0.7, edgecolor='black')
        axes_row[2].set_title(f'{title_prefix} Attention Variance')
        axes_row[2].set_xlabel('Variance')
        axes_row[2].set_ylabel('Frequency')
        
        # 4. Average attention per position
        avg_attention = np.mean(attention, axis=(0, 1, 2))
        positions = range(len(avg_attention))
        axes_row[3].bar(positions, avg_attention)
        axes_row[3].set_title(f'{title_prefix} Avg Attention per Position')
        axes_row[3].set_xlabel('Position')
        axes_row[3].set_ylabel('Average Attention')
        
    # Plot row attention statistics
    plot_attention_stats(row_attention, axes[0], 'Row')
    
    # Plot column attention statistics if available
    if has_both:
        plot_attention_stats(col_attention, axes[1], 'Column')
        
    plt.suptitle('Attention Statistics', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig


def plot_feature_importance(
    feature_importance: Dict[str, float],
    top_k: int = 20,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """Plot feature importance scores.
    
    Args:
        feature_importance: Dictionary of feature names and importance scores
        top_k: Number of top features to display
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure or None if plotting not available
    """
    if not PLOTTING_AVAILABLE:
        warnings.warn("Plotting not available. Install matplotlib and seaborn.")
        return None
        
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_k]
    
    features, scores = zip(*top_features)
    
    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, scores, alpha=0.8)
    
    # Color bars based on importance
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()  # Top feature at the top
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_k} Feature Importance')
    
    # Add value labels on bars
    for i, (feature, score) in enumerate(top_features):
        ax.text(score + 0.01 * max(scores), i, f'{score:.3f}', 
                va='center', fontsize=10)
                
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig


def plot_model_architecture(
    config: Dict[str, Any],
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """Plot RCANet model architecture diagram.
    
    Args:
        config: Model configuration dictionary
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure or None if plotting not available
    """
    if not PLOTTING_AVAILABLE:
        warnings.warn("Plotting not available. Install matplotlib and seaborn.")
        return None
        
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define colors
    input_color = '#E8F4FD'
    attention_color = '#B3D9FF'
    fusion_color = '#66B2FF'
    output_color = '#1A75FF'
    
    # Input layer
    input_rect = Rectangle((1, 10), 8, 1, facecolor=input_color, edgecolor='black', linewidth=2)
    ax.add_patch(input_rect)
    ax.text(5, 10.5, f"Input Layer\n(dim: {config.get('input_dim', 'N')})", 
            ha='center', va='center', fontsize=12, weight='bold')
    
    # Row and Column Attention branches
    row_rect = Rectangle((0.5, 7.5), 3.5, 2, facecolor=attention_color, edgecolor='black', linewidth=2)
    ax.add_patch(row_rect)
    ax.text(2.25, 8.5, f"Row Attention\n{config.get('num_heads', 8)} heads\n{config.get('num_layers', 4)} layers", 
            ha='center', va='center', fontsize=10, weight='bold')
    
    col_rect = Rectangle((6, 7.5), 3.5, 2, facecolor=attention_color, edgecolor='black', linewidth=2)
    ax.add_patch(col_rect)
    ax.text(7.75, 8.5, f"Column Attention\n{config.get('num_heads', 8)} heads\n{config.get('num_layers', 4)} layers", 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # Fusion layer
    fusion_rect = Rectangle((2.5, 5), 5, 1.5, facecolor=fusion_color, edgecolor='black', linewidth=2)
    ax.add_patch(fusion_rect)
    ax.text(5, 5.75, f"Hierarchical Fusion\n({config.get('fusion_strategy', 'cross_attention')})", 
            ha='center', va='center', fontsize=11, weight='bold')
    
    # Contrastive learning (if enabled)
    if config.get('use_contrastive', False):
        contrastive_rect = Rectangle((0.5, 2.5), 3, 1.5, facecolor='#FFE6CC', edgecolor='black', linewidth=2)
        ax.add_patch(contrastive_rect)
        ax.text(2, 3.25, f"Contrastive\nLearning", ha='center', va='center', fontsize=10, weight='bold')
        
        # Arrow from fusion to contrastive
        ax.arrow(3.5, 5, -1, -1, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Output layer
    output_rect = Rectangle((6.5, 2.5), 3, 1.5, facecolor=output_color, edgecolor='black', linewidth=2)
    ax.add_patch(output_rect)
    task_type = config.get('task_type', 'regression')
    output_dim = config.get('output_dim', 1)
    ax.text(8, 3.25, f"Output Layer\n{task_type}\n(dim: {output_dim})", 
            ha='center', va='center', fontsize=10, weight='bold', color='white')
    
    # Arrows
    # Input to attention branches
    ax.arrow(3, 10, -0.5, -0.3, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(7, 10, 0.5, -0.3, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Attention branches to fusion
    ax.arrow(2.25, 7.5, 1.25, -1.8, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(7.75, 7.5, -1.25, -1.8, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Fusion to output
    ax.arrow(6.5, 5.75, 1.5, -1.8, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Title
    ax.text(5, 11.5, 'RCANet Architecture', ha='center', va='center', 
            fontsize=16, weight='bold')
    
    # Add configuration details
    config_text = f"""Configuration:
• Hidden dim: {config.get('hidden_dim', 256)}
• Dropout: {config.get('dropout', 0.1)}
• Activation: {config.get('activation', 'gelu')}
• Fusion strategy: {config.get('fusion_strategy', 'cross_attention')}"""
    
    ax.text(0.5, 1, config_text, ha='left', va='top', fontsize=9, 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig


def plot_data_distribution(
    data: np.ndarray,
    feature_names: Optional[List[str]] = None,
    categorical_mask: Optional[np.ndarray] = None,
    max_features: int = 20,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """Plot data distribution for features.
    
    Args:
        data: Data array [samples, features]
        feature_names: Names of features
        categorical_mask: Boolean mask indicating categorical features
        max_features: Maximum number of features to plot
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure or None if plotting not available
    """
    if not PLOTTING_AVAILABLE:
        warnings.warn("Plotting not available. Install matplotlib and seaborn.")
        return None
        
    n_samples, n_features = data.shape
    n_features_to_plot = min(max_features, n_features)
    
    # Select features to plot (first n_features_to_plot)
    selected_features = list(range(n_features_to_plot))
    
    # Create subplots
    n_cols = 4
    n_rows = (n_features_to_plot + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    for i, feature_idx in enumerate(selected_features):
        if i >= len(axes):
            break
            
        ax = axes[i]
        feature_data = data[:, feature_idx]
        
        # Determine if categorical
        is_categorical = (categorical_mask is not None and 
                         feature_idx < len(categorical_mask) and 
                         categorical_mask[feature_idx])
        
        if is_categorical or len(np.unique(feature_data)) <= 10:
            # Plot as categorical (bar plot)
            unique_values, counts = np.unique(feature_data, return_counts=True)
            ax.bar(range(len(unique_values)), counts, alpha=0.7)
            ax.set_xticks(range(len(unique_values)))
            ax.set_xticklabels([str(v) for v in unique_values], rotation=45)
            ax.set_ylabel('Count')
        else:
            # Plot as continuous (histogram)
            ax.hist(feature_data, bins=30, alpha=0.7, edgecolor='black')
            ax.set_ylabel('Frequency')
            
        # Set title
        if feature_names and feature_idx < len(feature_names):
            title = feature_names[feature_idx]
        else:
            title = f'Feature {feature_idx}'
            
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        
    # Hide unused subplots
    for i in range(len(selected_features), len(axes)):
        axes[i].set_visible(False)
        
    plt.suptitle(f'Data Distribution (showing {n_features_to_plot}/{n_features} features)', 
                 fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig


def create_interactive_attention_plot(
    attention_weights: torch.Tensor,
    feature_names: Optional[List[str]] = None,
    sample_idx: int = 0,
    head_idx: int = 0
) -> Optional['go.Figure']:
    """Create interactive attention heatmap using Plotly.
    
    Args:
        attention_weights: Attention weights [batch, heads, seq, seq]
        feature_names: Names of features
        sample_idx: Sample index to plot
        head_idx: Attention head index to plot
        
    Returns:
        Plotly figure or None if Plotly not available
    """
    if not PLOTLY_AVAILABLE:
        warnings.warn("Plotly not available. Install plotly for interactive visualization.")
        return None
        
    # Convert to numpy
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
        
    # Extract attention map for specific sample and head
    attention_map = attention_weights[sample_idx, head_idx]
    
    # Create feature labels
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(attention_map.shape[0])]
        
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=attention_map,
        x=feature_names,
        y=feature_names,
        colorscale='Blues',
        showscale=True,
        hoverongaps=False,
        hovertemplate='From: %{y}<br>To: %{x}<br>Attention: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Attention Map - Sample {sample_idx}, Head {head_idx}',
        xaxis_title='To Feature',
        yaxis_title='From Feature',
        width=800,
        height=800
    )
    
    return fig


def save_attention_analysis(
    attention_weights: Union[torch.Tensor, Dict[str, torch.Tensor]],
    feature_names: Optional[List[str]] = None,
    output_dir: str = 'attention_analysis',
    sample_indices: Optional[List[int]] = None
):
    """Save comprehensive attention analysis plots.
    
    Args:
        attention_weights: Attention weights tensor or dict
        feature_names: Names of features
        output_dir: Directory to save plots
        sample_indices: Specific samples to analyze
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot attention maps
    fig1 = plot_attention_maps(
        attention_weights,
        feature_names=feature_names,
        sample_indices=sample_indices,
        save_path=os.path.join(output_dir, 'attention_maps.png')
    )
    if fig1:
        plt.close(fig1)
        
    # Plot attention statistics
    fig2 = plot_attention_statistics(
        attention_weights,
        save_path=os.path.join(output_dir, 'attention_statistics.png')
    )
    if fig2:
        plt.close(fig2)
        
    print(f"Attention analysis saved to {output_dir}")