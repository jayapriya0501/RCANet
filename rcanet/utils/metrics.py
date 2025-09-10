"""Evaluation metrics for RCANet models.

Provides comprehensive evaluation metrics for both classification
and regression tasks, along with visualization utilities.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
from sklearn.preprocessing import label_binarize
import warnings

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("Matplotlib/Seaborn not available. Install for plotting support.")


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = 'weighted',
    labels: Optional[List[str]] = None
) -> Dict[str, float]:
    """Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        average: Averaging strategy for multi-class metrics
        labels: Class labels (optional)
        
    Returns:
        Dictionary of classification metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # ROC AUC and PR AUC (if probabilities provided)
    if y_prob is not None:
        try:
            n_classes = len(np.unique(y_true))
            
            if n_classes == 2:
                # Binary classification
                if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    y_prob_positive = y_prob[:, 1]
                else:
                    y_prob_positive = y_prob.flatten()
                    
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob_positive)
                metrics['pr_auc'] = average_precision_score(y_true, y_prob_positive)
                
            elif n_classes > 2:
                # Multi-class classification
                if average == 'macro':
                    metrics['roc_auc'] = roc_auc_score(
                        y_true, y_prob, multi_class='ovr', average='macro'
                    )
                elif average == 'weighted':
                    metrics['roc_auc'] = roc_auc_score(
                        y_true, y_prob, multi_class='ovr', average='weighted'
                    )
                    
                # PR AUC for multi-class (one-vs-rest)
                y_true_binarized = label_binarize(y_true, classes=np.unique(y_true))
                if y_true_binarized.shape[1] > 1:
                    pr_aucs = []
                    for i in range(y_true_binarized.shape[1]):
                        pr_auc = average_precision_score(
                            y_true_binarized[:, i], y_prob[:, i]
                        )
                        pr_aucs.append(pr_auc)
                    metrics['pr_auc'] = np.mean(pr_aucs)
                    
        except Exception as e:
            warnings.warn(f"Could not compute ROC/PR AUC: {e}")
            
    # Per-class metrics (if labels provided)
    if labels is not None:
        precision_per_class = precision_score(
            y_true, y_pred, average=None, zero_division=0
        )
        recall_per_class = recall_score(
            y_true, y_pred, average=None, zero_division=0
        )
        f1_per_class = f1_score(
            y_true, y_pred, average=None, zero_division=0
        )
        
        for i, label in enumerate(labels):
            if i < len(precision_per_class):
                metrics[f'precision_{label}'] = precision_per_class[i]
                metrics[f'recall_{label}'] = recall_per_class[i]
                metrics[f'f1_{label}'] = f1_per_class[i]
                
    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of regression metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
    
    # MAPE (handle division by zero)
    try:
        metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
    except:
        # Manual calculation with zero handling
        mask = y_true != 0
        if np.sum(mask) > 0:
            metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            metrics['mape'] = float('inf')
            
    # Additional metrics
    residuals = y_true - y_pred
    metrics['mean_residual'] = np.mean(residuals)
    metrics['std_residual'] = np.std(residuals)
    
    # Median-based metrics (robust to outliers)
    metrics['median_ae'] = np.median(np.abs(residuals))
    
    # Max error
    metrics['max_error'] = np.max(np.abs(residuals))
    
    return metrics


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str = 'auto',
    y_prob: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None
) -> Dict[str, float]:
    """Compute appropriate metrics based on task type.
    
    Args:
        y_true: True values/labels
        y_pred: Predicted values/labels
        task_type: 'classification', 'regression', or 'auto'
        y_prob: Predicted probabilities (for classification)
        labels: Class labels (for classification)
        
    Returns:
        Dictionary of computed metrics
    """
    # Auto-detect task type
    if task_type == 'auto':
        # Check if targets are integers and have limited unique values
        unique_values = len(np.unique(y_true))
        if (y_true.dtype in [np.int32, np.int64] or 
            (unique_values <= 20 and np.all(y_true == y_true.astype(int)))):
            task_type = 'classification'
        else:
            task_type = 'regression'
            
    if task_type == 'classification':
        # Convert predictions to integers if needed
        if y_pred.dtype in [np.float32, np.float64]:
            if y_prob is None and y_pred.ndim == 2:
                # Predictions are probabilities
                y_prob = y_pred
                y_pred = np.argmax(y_pred, axis=1)
            else:
                # Round predictions to nearest integer
                y_pred = np.round(y_pred).astype(int)
                
        return compute_classification_metrics(y_true, y_pred, y_prob, labels=labels)
        
    elif task_type == 'regression':
        # Ensure predictions are float
        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()
        return compute_regression_metrics(y_true, y_pred)
        
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def compute_attention_metrics(
    attention_weights: torch.Tensor,
    ground_truth_attention: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """Compute attention-specific metrics.
    
    Args:
        attention_weights: Computed attention weights [batch, heads, seq, seq]
        ground_truth_attention: Ground truth attention (optional)
        
    Returns:
        Dictionary of attention metrics
    """
    metrics = {}
    
    # Convert to numpy
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
        
    # Attention entropy (measure of attention distribution)
    # Higher entropy = more distributed attention
    attention_flat = attention_weights.reshape(-1, attention_weights.shape[-1])
    entropies = []
    
    for att in attention_flat:
        # Add small epsilon to avoid log(0)
        att_norm = att + 1e-8
        att_norm = att_norm / np.sum(att_norm)
        entropy = -np.sum(att_norm * np.log(att_norm))
        entropies.append(entropy)
        
    metrics['attention_entropy_mean'] = np.mean(entropies)
    metrics['attention_entropy_std'] = np.std(entropies)
    
    # Attention sparsity (measure of how concentrated attention is)
    # Lower values = more sparse attention
    max_attention = np.max(attention_weights, axis=-1)
    metrics['attention_max_mean'] = np.mean(max_attention)
    metrics['attention_max_std'] = np.std(max_attention)
    
    # Attention variance (measure of attention variability)
    attention_var = np.var(attention_weights, axis=-1)
    metrics['attention_variance_mean'] = np.mean(attention_var)
    metrics['attention_variance_std'] = np.std(attention_var)
    
    # If ground truth attention is provided
    if ground_truth_attention is not None:
        if isinstance(ground_truth_attention, torch.Tensor):
            ground_truth_attention = ground_truth_attention.detach().cpu().numpy()
            
        # Attention alignment (correlation with ground truth)
        correlations = []
        for i in range(attention_weights.shape[0]):
            for j in range(attention_weights.shape[1]):
                att_pred = attention_weights[i, j].flatten()
                att_true = ground_truth_attention[i, j].flatten()
                
                corr = np.corrcoef(att_pred, att_true)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
                    
        if correlations:
            metrics['attention_correlation_mean'] = np.mean(correlations)
            metrics['attention_correlation_std'] = np.std(correlations)
            
        # Attention MSE
        metrics['attention_mse'] = np.mean((attention_weights - ground_truth_attention) ** 2)
        
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = 'Confusion Matrix',
    figsize: Tuple[int, int] = (8, 6)
) -> Optional[plt.Figure]:
    """Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        normalize: Whether to normalize the matrix
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure or None if plotting not available
    """
    if not PLOTTING_AVAILABLE:
        warnings.warn("Plotting not available. Install matplotlib and seaborn.")
        return None
        
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use seaborn heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    plt.tight_layout()
    return fig


def plot_regression_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = 'Regression Results',
    figsize: Tuple[int, int] = (12, 4)
) -> Optional[plt.Figure]:
    """Plot regression results.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure or None if plotting not available
    """
    if not PLOTTING_AVAILABLE:
        warnings.warn("Plotting not available. Install matplotlib and seaborn.")
        return None
        
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Scatter plot: predicted vs actual
    axes[0].scatter(y_true, y_pred, alpha=0.6)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0].set_xlabel('True Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title('Predicted vs Actual')
    
    # Residuals plot
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residuals Plot')
    
    # Residuals histogram
    axes[2].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Residuals')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Residuals Distribution')
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def plot_attention_maps(
    attention_weights: torch.Tensor,
    feature_names: Optional[List[str]] = None,
    sample_indices: Optional[List[int]] = None,
    max_samples: int = 4,
    figsize: Tuple[int, int] = (15, 10)
) -> Optional[plt.Figure]:
    """Plot attention heatmaps.
    
    Args:
        attention_weights: Attention weights [batch, heads, seq, seq]
        feature_names: Names of features
        sample_indices: Specific samples to plot
        max_samples: Maximum number of samples to plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure or None if plotting not available
    """
    if not PLOTTING_AVAILABLE:
        warnings.warn("Plotting not available. Install matplotlib and seaborn.")
        return None
        
    # Convert to numpy
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
        
    batch_size, n_heads, seq_len, _ = attention_weights.shape
    
    # Select samples to plot
    if sample_indices is None:
        sample_indices = list(range(min(max_samples, batch_size)))
    else:
        sample_indices = sample_indices[:max_samples]
        
    n_samples = len(sample_indices)
    
    # Create subplots
    fig, axes = plt.subplots(n_samples, n_heads, figsize=figsize)
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    if n_heads == 1:
        axes = axes.reshape(-1, 1)
        
    for i, sample_idx in enumerate(sample_indices):
        for head in range(n_heads):
            ax = axes[i, head] if n_samples > 1 else axes[head]
            
            # Plot attention heatmap
            attention_map = attention_weights[sample_idx, head]
            
            im = ax.imshow(attention_map, cmap='Blues', aspect='auto')
            
            # Set labels
            if feature_names and len(feature_names) == seq_len:
                ax.set_xticks(range(seq_len))
                ax.set_yticks(range(seq_len))
                ax.set_xticklabels(feature_names, rotation=45, ha='right')
                ax.set_yticklabels(feature_names)
            else:
                ax.set_xticks(range(0, seq_len, max(1, seq_len // 10)))
                ax.set_yticks(range(0, seq_len, max(1, seq_len // 10)))
                
            ax.set_title(f'Sample {sample_idx}, Head {head}')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
    plt.suptitle('Attention Maps')
    plt.tight_layout()
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> Optional[plt.Figure]:
    """Plot training history.
    
    Args:
        history: Training history dictionary
        metrics: Specific metrics to plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure or None if plotting not available
    """
    if not PLOTTING_AVAILABLE:
        warnings.warn("Plotting not available. Install matplotlib and seaborn.")
        return None
        
    if metrics is None:
        # Auto-select important metrics
        metrics = []
        for key in history.keys():
            if any(metric in key.lower() for metric in ['loss', 'accuracy', 'f1', 'r2', 'mse']):
                metrics.append(key)
                
    if not metrics:
        metrics = list(history.keys())[:6]  # Limit to 6 metrics
        
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten()
    elif n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
        
    for i, metric in enumerate(metrics):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        if metric in history:
            epochs = range(1, len(history[metric]) + 1)
            ax.plot(epochs, history[metric], label=metric, linewidth=2)
            
            # Plot validation metric if available
            val_metric = f'val_{metric}'
            if val_metric in history:
                ax.plot(epochs, history[val_metric], label=val_metric, linewidth=2, linestyle='--')
                
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} History')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
    # Hide unused subplots
    for i in range(len(metrics), len(axes)):
        axes[i].set_visible(False)
        
    plt.suptitle('Training History')
    plt.tight_layout()
    return fig


class MetricsTracker:
    """Track and compute metrics during training."""
    
    def __init__(self, task_type: str = 'auto'):
        self.task_type = task_type
        self.reset()
        
    def reset(self):
        """Reset all stored predictions and targets."""
        self.predictions = []
        self.targets = []
        self.probabilities = []
        
    def update(
        self,
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
        probabilities: Optional[Union[torch.Tensor, np.ndarray]] = None
    ):
        """Update with new predictions and targets."""
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        if probabilities is not None and isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.detach().cpu().numpy()
            
        self.predictions.append(predictions)
        self.targets.append(targets)
        
        if probabilities is not None:
            self.probabilities.append(probabilities)
            
    def compute(self) -> Dict[str, float]:
        """Compute metrics from all stored predictions and targets."""
        if not self.predictions or not self.targets:
            return {}
            
        all_predictions = np.concatenate(self.predictions, axis=0)
        all_targets = np.concatenate(self.targets, axis=0)
        
        all_probabilities = None
        if self.probabilities:
            all_probabilities = np.concatenate(self.probabilities, axis=0)
            
        return compute_metrics(
            all_targets,
            all_predictions,
            task_type=self.task_type,
            y_prob=all_probabilities
        )