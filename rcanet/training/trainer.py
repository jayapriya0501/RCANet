"""Training utilities and trainer class for RCANet.

Provides comprehensive training functionality including optimization,
scheduling, logging, and evaluation for RCANet models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import (
    StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau,
    OneCycleLR, CosineAnnealingWarmRestarts
)
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from collections import defaultdict
import warnings

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    warnings.warn("TensorBoard not available. Install tensorboard for logging support.")

from ..models.rcanet import RCANet
from ..models.contrastive import ContrastiveLoss
from ..utils.metrics import compute_metrics


class RCANetTrainer:
    """Comprehensive trainer for RCANet models.
    
    Handles training, validation, logging, checkpointing, and evaluation
    for RCANet models with support for various optimization strategies.
    
    Args:
        model: RCANet model to train
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        test_loader: Test data loader (optional)
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        device: Device to train on
        log_dir: Directory for logging (optional)
        checkpoint_dir: Directory for checkpoints (optional)
        patience: Early stopping patience
        min_delta: Minimum change for early stopping
        save_best_only: Whether to save only the best model
        verbose: Verbosity level
    """
    
    def __init__(
        self,
        model: RCANet,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        log_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        patience: int = 10,
        min_delta: float = 1e-4,
        save_best_only: bool = True,
        verbose: int = 1
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Set device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Set default criterion if not provided
        if criterion is None:
            # Determine task type from model config
            if hasattr(model, 'config') and hasattr(model.config, 'task_type'):
                if model.config.task_type == 'classification':
                    criterion = nn.CrossEntropyLoss()
                else:
                    criterion = nn.MSELoss()
            else:
                criterion = nn.MSELoss()  # Default to regression
                
        self.criterion = criterion
        
        # Set default optimizer if not provided
        if optimizer is None:
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.optimizer = optimizer
        
        self.scheduler = scheduler
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_metric = -float('inf')
        self.patience_counter = 0
        self.training_history = defaultdict(list)
        
        # Early stopping
        self.patience = patience
        self.min_delta = min_delta
        self.save_best_only = save_best_only
        
        # Logging
        self.verbose = verbose
        self.log_dir = Path(log_dir) if log_dir else None
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        
        # Create directories
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
        # Initialize tensorboard writer
        self.writer = None
        if TENSORBOARD_AVAILABLE and self.log_dir:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            
        # Contrastive learning
        self.use_contrastive = hasattr(model, 'contrastive_head') and model.contrastive_head is not None
        if self.use_contrastive:
            self.contrastive_loss = ContrastiveLoss()
            
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_main_loss = 0.0
        total_contrastive_loss = 0.0
        total_samples = 0
        
        predictions = []
        targets = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    X, y = batch
                    X, y = X.to(self.device), y.to(self.device)
                else:
                    X = batch[0].to(self.device)
                    y = batch[1].to(self.device) if len(batch) > 1 else None
            else:
                X = batch.to(self.device)
                y = None
                
            batch_size = X.shape[0]
            total_samples += batch_size
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.use_contrastive:
                outputs = self.model(X, return_attention=True, return_contrastive=True)
                main_output = outputs['output']
                row_embeddings = outputs.get('row_embeddings')
                col_embeddings = outputs.get('col_embeddings')
            else:
                main_output = self.model(X)
                
            # Compute main loss
            if y is not None:
                main_loss = self.criterion(main_output, y)
            else:
                # Unsupervised case
                main_loss = torch.tensor(0.0, device=self.device)
                
            total_main_loss += main_loss.item() * batch_size
            
            # Compute contrastive loss
            contrastive_loss = torch.tensor(0.0, device=self.device)
            if self.use_contrastive and row_embeddings is not None and col_embeddings is not None:
                contrastive_loss = self.contrastive_loss(
                    row_embeddings, col_embeddings, y
                )
                total_contrastive_loss += contrastive_loss.item() * batch_size
                
            # Total loss
            loss = main_loss + 0.1 * contrastive_loss  # Weight contrastive loss
            total_loss += loss.item() * batch_size
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Store predictions and targets for metrics
            if y is not None:
                predictions.append(main_output.detach().cpu())
                targets.append(y.detach().cpu())
                
            # Log batch progress
            if self.verbose >= 2 and batch_idx % 100 == 0:
                print(f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
                
        # Compute epoch metrics
        avg_loss = total_loss / total_samples
        avg_main_loss = total_main_loss / total_samples
        avg_contrastive_loss = total_contrastive_loss / total_samples
        
        metrics = {
            'loss': avg_loss,
            'main_loss': avg_main_loss,
            'contrastive_loss': avg_contrastive_loss
        }
        
        # Compute additional metrics if targets available
        if predictions and targets:
            all_predictions = torch.cat(predictions, dim=0)
            all_targets = torch.cat(targets, dim=0)
            
            # Determine task type
            if all_targets.dtype in [torch.long, torch.int]:
                task_type = 'classification'
            else:
                task_type = 'regression'
                
            additional_metrics = compute_metrics(
                all_predictions.numpy(),
                all_targets.numpy(),
                task_type=task_type
            )
            metrics.update(additional_metrics)
            
        return metrics
        
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch.
        
        Returns:
            Dictionary containing validation metrics
        """
        if self.val_loader is None:
            return {}
            
        self.model.eval()
        
        total_loss = 0.0
        total_main_loss = 0.0
        total_contrastive_loss = 0.0
        total_samples = 0
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 2:
                        X, y = batch
                        X, y = X.to(self.device), y.to(self.device)
                    else:
                        X = batch[0].to(self.device)
                        y = batch[1].to(self.device) if len(batch) > 1 else None
                else:
                    X = batch.to(self.device)
                    y = None
                    
                batch_size = X.shape[0]
                total_samples += batch_size
                
                # Forward pass
                if self.use_contrastive:
                    outputs = self.model(X, return_attention=True, return_contrastive=True)
                    main_output = outputs['output']
                    row_embeddings = outputs.get('row_embeddings')
                    col_embeddings = outputs.get('col_embeddings')
                else:
                    main_output = self.model(X)
                    
                # Compute main loss
                if y is not None:
                    main_loss = self.criterion(main_output, y)
                else:
                    main_loss = torch.tensor(0.0, device=self.device)
                    
                total_main_loss += main_loss.item() * batch_size
                
                # Compute contrastive loss
                contrastive_loss = torch.tensor(0.0, device=self.device)
                if self.use_contrastive and row_embeddings is not None and col_embeddings is not None:
                    contrastive_loss = self.contrastive_loss(
                        row_embeddings, col_embeddings, y
                    )
                    total_contrastive_loss += contrastive_loss.item() * batch_size
                    
                # Total loss
                loss = main_loss + 0.1 * contrastive_loss
                total_loss += loss.item() * batch_size
                
                # Store predictions and targets
                if y is not None:
                    predictions.append(main_output.cpu())
                    targets.append(y.cpu())
                    
        # Compute epoch metrics
        avg_loss = total_loss / total_samples
        avg_main_loss = total_main_loss / total_samples
        avg_contrastive_loss = total_contrastive_loss / total_samples
        
        metrics = {
            'val_loss': avg_loss,
            'val_main_loss': avg_main_loss,
            'val_contrastive_loss': avg_contrastive_loss
        }
        
        # Compute additional metrics
        if predictions and targets:
            all_predictions = torch.cat(predictions, dim=0)
            all_targets = torch.cat(targets, dim=0)
            
            # Determine task type
            if all_targets.dtype in [torch.long, torch.int]:
                task_type = 'classification'
            else:
                task_type = 'regression'
                
            additional_metrics = compute_metrics(
                all_predictions.numpy(),
                all_targets.numpy(),
                task_type=task_type
            )
            
            # Add 'val_' prefix to metrics
            val_metrics = {f'val_{k}': v for k, v in additional_metrics.items()}
            metrics.update(val_metrics)
            
        return metrics
        
    def train(
        self,
        epochs: int,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """Train the model for specified number of epochs.
        
        Args:
            epochs: Number of epochs to train
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Training history dictionary
        """
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
            
        start_time = time.time()
        
        for epoch in range(self.current_epoch, epochs):
            epoch_start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # Update training history
            for key, value in epoch_metrics.items():
                self.training_history[key].append(value)
                
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('val_loss', train_metrics['loss']))
                else:
                    self.scheduler.step()
                    
            # Log metrics
            if self.writer:
                for key, value in epoch_metrics.items():
                    self.writer.add_scalar(key, value, epoch)
                    
                # Log learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('learning_rate', current_lr, epoch)
                
            # Print progress
            if self.verbose >= 1:
                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.2f}s):")
                
                for key, value in epoch_metrics.items():
                    print(f"  {key}: {value:.4f}")
                    
                if self.scheduler:
                    print(f"  lr: {self.optimizer.param_groups[0]['lr']:.6f}")
                print()
                
            # Early stopping and checkpointing
            val_loss = val_metrics.get('val_loss', train_metrics['loss'])
            val_metric = val_metrics.get('val_accuracy', val_metrics.get('val_r2', -val_loss))
            
            # Check for improvement
            improved = False
            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                improved = True
                
            if val_metric > self.best_val_metric + self.min_delta:
                self.best_val_metric = val_metric
                improved = True
                
            if improved:
                self.patience_counter = 0
                if self.checkpoint_dir:
                    self.save_checkpoint(
                        epoch,
                        is_best=True,
                        metrics=epoch_metrics
                    )
            else:
                self.patience_counter += 1
                
            # Save regular checkpoint
            if self.checkpoint_dir and not self.save_best_only:
                self.save_checkpoint(
                    epoch,
                    is_best=False,
                    metrics=epoch_metrics
                )
                
            # Early stopping
            if self.patience_counter >= self.patience:
                if self.verbose >= 1:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                break
                
            self.current_epoch = epoch + 1
            
        total_time = time.time() - start_time
        if self.verbose >= 1:
            print(f"Training completed in {total_time:.2f}s")
            
        # Close tensorboard writer
        if self.writer:
            self.writer.close()
            
        return dict(self.training_history)
        
    def evaluate(
        self,
        data_loader: Optional[DataLoader] = None,
        return_predictions: bool = False
    ) -> Dict[str, float]:
        """Evaluate the model on test data.
        
        Args:
            data_loader: Data loader to evaluate on (uses test_loader if None)
            return_predictions: Whether to return predictions
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if data_loader is None:
            data_loader = self.test_loader
            
        if data_loader is None:
            raise ValueError("No data loader provided for evaluation")
            
        self.model.eval()
        
        predictions = []
        targets = []
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    X, y = batch
                    X, y = X.to(self.device), y.to(self.device)
                else:
                    X = batch.to(self.device)
                    y = None
                    
                batch_size = X.shape[0]
                total_samples += batch_size
                
                # Forward pass
                output = self.model(X)
                
                if y is not None:
                    loss = self.criterion(output, y)
                    total_loss += loss.item() * batch_size
                    
                    predictions.append(output.cpu())
                    targets.append(y.cpu())
                else:
                    predictions.append(output.cpu())
                    
        # Compute metrics
        metrics = {}
        
        if targets:
            all_predictions = torch.cat(predictions, dim=0)
            all_targets = torch.cat(targets, dim=0)
            
            # Determine task type
            if all_targets.dtype in [torch.long, torch.int]:
                task_type = 'classification'
            else:
                task_type = 'regression'
                
            metrics = compute_metrics(
                all_predictions.numpy(),
                all_targets.numpy(),
                task_type=task_type
            )
            
            metrics['test_loss'] = total_loss / total_samples
            
        if return_predictions:
            all_predictions = torch.cat(predictions, dim=0)
            return metrics, all_predictions.numpy()
        else:
            return metrics
            
    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        metrics: Optional[Dict[str, float]] = None
    ):
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_metric': self.best_val_metric,
            'training_history': dict(self.training_history),
            'model_config': self.model.config.__dict__ if hasattr(self.model, 'config') else None
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        if metrics:
            checkpoint['metrics'] = metrics
            
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            
        if self.verbose >= 2:
            print(f"Checkpoint saved: {checkpoint_path}")
            
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.current_epoch = checkpoint.get('epoch', 0) + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_metric = checkpoint.get('best_val_metric', -float('inf'))
        
        if 'training_history' in checkpoint:
            self.training_history = defaultdict(list, checkpoint['training_history'])
            
        if self.verbose >= 1:
            print(f"Checkpoint loaded from {checkpoint_path}")
            print(f"Resuming from epoch {self.current_epoch}")
            
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training process."""
        summary = {
            'total_epochs': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'best_val_metric': self.best_val_metric,
            'final_lr': self.optimizer.param_groups[0]['lr'],
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        # Add final metrics
        if self.training_history:
            for key, values in self.training_history.items():
                if values:
                    summary[f'final_{key}'] = values[-1]
                    
        return summary


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = 'adamw',
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    **kwargs
) -> optim.Optimizer:
    """Create optimizer for model.
    
    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd', 'rmsprop')
        learning_rate: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments
        
    Returns:
        Configured optimizer
    """
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
            **{k: v for k, v in kwargs.items() if k != 'momentum'}
        )
    elif optimizer_type == 'rmsprop':
        return optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = 'cosine',
    **kwargs
) -> Optional[Any]:
    """Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler
        **kwargs: Additional scheduler arguments
        
    Returns:
        Configured scheduler or None
    """
    if scheduler_type is None or scheduler_type.lower() == 'none':
        return None
        
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == 'step':
        return StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type == 'exponential':
        return ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.95)
        )
    elif scheduler_type == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 100),
            eta_min=kwargs.get('eta_min', 0)
        )
    elif scheduler_type == 'plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10),
            verbose=kwargs.get('verbose', True)
        )
    elif scheduler_type == 'onecycle':
        return OneCycleLR(
            optimizer,
            max_lr=kwargs.get('max_lr', 1e-2),
            total_steps=kwargs.get('total_steps', 1000)
        )
    elif scheduler_type == 'cosine_restart':
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get('T_0', 10),
            T_mult=kwargs.get('T_mult', 2)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")