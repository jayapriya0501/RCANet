"""Contrastive learning objectives for RCANet.

Implements contrastive pre-training that aligns row-level and column-level
embeddings to capture complementary relational cues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict


class ContrastiveLoss(nn.Module):
    """Contrastive loss for aligning row and column representations.
    
    This loss encourages the model to learn complementary representations
    by aligning row-level and column-level embeddings through contrastive learning.
    
    Args:
        temperature: Temperature parameter for contrastive loss
        projection_dim: Dimension of projection heads
        symmetric: Whether to use symmetric contrastive loss
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        projection_dim: int = 128,
        symmetric: bool = True
    ):
        super().__init__()
        self.temperature = temperature
        self.symmetric = symmetric
        
        # Projection heads for row and column representations
        self.row_projector = ProjectionHead(projection_dim)
        self.col_projector = ProjectionHead(projection_dim)
        
    def forward(
        self,
        row_embeddings: torch.Tensor,
        col_embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute contrastive loss.
        
        Args:
            row_embeddings: Row-level embeddings (batch_size, n_samples, d_model)
            col_embeddings: Column-level embeddings (batch_size, n_features, d_model)
            labels: Optional labels for supervised contrastive learning
            
        Returns:
            Dictionary containing loss components
        """
        # Project embeddings
        row_proj = self.row_projector(row_embeddings)
        col_proj = self.col_projector(col_embeddings)
        
        # Compute contrastive losses
        if labels is not None:
            # Supervised contrastive learning
            row_loss = self._supervised_contrastive_loss(row_proj, labels)
            col_loss = self._supervised_contrastive_loss(col_proj, labels)
            alignment_loss = self._alignment_loss(row_proj, col_proj)
        else:
            # Self-supervised contrastive learning
            row_loss = self._self_supervised_loss(row_proj)
            col_loss = self._self_supervised_loss(col_proj)
            alignment_loss = self._cross_modal_alignment(row_proj, col_proj)
            
        total_loss = row_loss + col_loss + alignment_loss
        
        return {
            'total_loss': total_loss,
            'row_loss': row_loss,
            'col_loss': col_loss,
            'alignment_loss': alignment_loss
        }
        
    def _supervised_contrastive_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Supervised contrastive loss."""
        batch_size = embeddings.shape[0]
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.transpose(-2, -1))
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create label mask
        labels = labels.unsqueeze(1)
        label_mask = torch.eq(labels, labels.transpose(0, 1)).float()
        
        # Remove diagonal elements
        mask = torch.eye(batch_size, device=embeddings.device)
        label_mask = label_mask * (1 - mask)
        
        # Compute loss
        exp_sim = torch.exp(similarity_matrix) * (1 - mask)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=-1, keepdim=True))
        
        # Mean over positive pairs
        loss = -(label_mask * log_prob).sum() / (label_mask.sum() + 1e-8)
        
        return loss
        
    def _self_supervised_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Self-supervised contrastive loss using data augmentation."""
        batch_size, seq_len, dim = embeddings.shape
        
        # Create augmented views through dropout and noise
        aug1 = self._augment_embeddings(embeddings)
        aug2 = self._augment_embeddings(embeddings)
        
        # Normalize
        aug1 = F.normalize(aug1, dim=-1)
        aug2 = F.normalize(aug2, dim=-1)
        
        # Compute similarities
        pos_sim = torch.sum(aug1 * aug2, dim=-1) / self.temperature
        
        # Negative similarities (within batch)
        neg_sim1 = torch.matmul(aug1, aug2.transpose(-2, -1)) / self.temperature
        neg_sim2 = torch.matmul(aug2, aug1.transpose(-2, -1)) / self.temperature
        
        # Compute loss
        logits1 = torch.cat([pos_sim.unsqueeze(-1), neg_sim1], dim=-1)
        logits2 = torch.cat([pos_sim.unsqueeze(-1), neg_sim2], dim=-1)
        
        labels = torch.zeros(batch_size, seq_len, dtype=torch.long, device=embeddings.device)
        
        loss1 = F.cross_entropy(logits1.view(-1, logits1.size(-1)), labels.view(-1))
        loss2 = F.cross_entropy(logits2.view(-1, logits2.size(-1)), labels.view(-1))
        
        return (loss1 + loss2) / 2
        
    def _alignment_loss(
        self,
        row_proj: torch.Tensor,
        col_proj: torch.Tensor
    ) -> torch.Tensor:
        """Alignment loss between row and column representations."""
        # Global pooling to get instance-level representations
        row_global = torch.mean(row_proj, dim=1)  # (batch_size, proj_dim)
        col_global = torch.mean(col_proj, dim=1)  # (batch_size, proj_dim)
        
        # Normalize
        row_global = F.normalize(row_global, dim=-1)
        col_global = F.normalize(col_global, dim=-1)
        
        # Positive pairs: same instance
        pos_sim = torch.sum(row_global * col_global, dim=-1) / self.temperature
        
        # Negative pairs: different instances
        neg_sim_row = torch.matmul(row_global, col_global.transpose(0, 1)) / self.temperature
        neg_sim_col = torch.matmul(col_global, row_global.transpose(0, 1)) / self.temperature
        
        # Compute contrastive loss
        batch_size = row_global.shape[0]
        labels = torch.arange(batch_size, device=row_global.device)
        
        loss_row = F.cross_entropy(neg_sim_row, labels)
        loss_col = F.cross_entropy(neg_sim_col, labels)
        
        return (loss_row + loss_col) / 2
        
    def _cross_modal_alignment(
        self,
        row_proj: torch.Tensor,
        col_proj: torch.Tensor
    ) -> torch.Tensor:
        """Cross-modal alignment for self-supervised learning."""
        batch_size = row_proj.shape[0]
        
        # Create cross-modal positive and negative pairs
        row_flat = row_proj.view(batch_size, -1)
        col_flat = col_proj.view(batch_size, -1)
        
        # Normalize
        row_flat = F.normalize(row_flat, dim=-1)
        col_flat = F.normalize(col_flat, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(row_flat, col_flat.transpose(0, 1)) / self.temperature
        
        # Labels for alignment (diagonal should be positive)
        labels = torch.arange(batch_size, device=row_proj.device)
        
        # Symmetric loss
        loss_r2c = F.cross_entropy(similarity, labels)
        loss_c2r = F.cross_entropy(similarity.transpose(0, 1), labels)
        
        return (loss_r2c + loss_c2r) / 2
        
    def _augment_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to embeddings."""
        # Dropout augmentation
        augmented = F.dropout(embeddings, p=0.1, training=True)
        
        # Add small amount of noise
        noise = torch.randn_like(augmented) * 0.01
        augmented = augmented + noise
        
        return augmented


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 128):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through projection head."""
        original_shape = x.shape
        
        # Flatten for batch norm
        x_flat = x.view(-1, x.shape[-1])
        projected = self.projection(x_flat)
        
        # Reshape back
        projected = projected.view(*original_shape[:-1], projected.shape[-1])
        
        return projected


class RowColumnContrastive(nn.Module):
    """Specialized contrastive learning for row-column interactions.
    
    This module implements contrastive learning specifically designed for
    tabular data, focusing on row-column relationships.
    """
    
    def __init__(
        self,
        d_model: int,
        temperature: float = 0.1,
        projection_dim: int = 128,
        use_momentum: bool = True,
        momentum: float = 0.999
    ):
        super().__init__()
        self.temperature = temperature
        self.use_momentum = use_momentum
        self.momentum = momentum
        
        # Projection heads
        self.row_projector = ProjectionHead(d_model, output_dim=projection_dim)
        self.col_projector = ProjectionHead(d_model, output_dim=projection_dim)
        
        if use_momentum:
            # Momentum encoders
            self.row_projector_m = ProjectionHead(d_model, output_dim=projection_dim)
            self.col_projector_m = ProjectionHead(d_model, output_dim=projection_dim)
            
            # Initialize momentum encoders
            self._init_momentum_encoders()
            
        # Queue for negative samples
        self.register_buffer('row_queue', torch.randn(projection_dim, 65536))
        self.register_buffer('col_queue', torch.randn(projection_dim, 65536))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        
        # Normalize queues
        self.row_queue = F.normalize(self.row_queue, dim=0)
        self.col_queue = F.normalize(self.col_queue, dim=0)
        
    def _init_momentum_encoders(self):
        """Initialize momentum encoders."""
        for param_q, param_k in zip(
            self.row_projector.parameters(), 
            self.row_projector_m.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
            
        for param_q, param_k in zip(
            self.col_projector.parameters(),
            self.col_projector_m.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
            
    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update of momentum encoders."""
        for param_q, param_k in zip(
            self.row_projector.parameters(),
            self.row_projector_m.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
            
        for param_q, param_k in zip(
            self.col_projector.parameters(),
            self.col_projector_m.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
            
    @torch.no_grad()
    def _dequeue_and_enqueue(self, row_keys: torch.Tensor, col_keys: torch.Tensor):
        """Update the queue with new keys."""
        batch_size = row_keys.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # Replace the keys at ptr (dequeue and enqueue)
        self.row_queue[:, ptr:ptr + batch_size] = row_keys.T
        self.col_queue[:, ptr:ptr + batch_size] = col_keys.T
        
        ptr = (ptr + batch_size) % self.row_queue.shape[1]
        self.queue_ptr[0] = ptr
        
    def forward(
        self,
        row_embeddings: torch.Tensor,
        col_embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for row-column contrastive learning."""
        # Global pooling
        row_global = torch.mean(row_embeddings, dim=1)
        col_global = torch.mean(col_embeddings, dim=1)
        
        # Project embeddings
        row_q = F.normalize(self.row_projector(row_global), dim=-1)
        col_q = F.normalize(self.col_projector(col_global), dim=-1)
        
        if self.use_momentum:
            # Update momentum encoders
            self._momentum_update()
            
            # Compute keys with momentum encoders
            with torch.no_grad():
                row_k = F.normalize(self.row_projector_m(row_global), dim=-1)
                col_k = F.normalize(self.col_projector_m(col_global), dim=-1)
                
            # Compute contrastive loss with queue
            loss_row = self._contrastive_loss_with_queue(row_q, row_k, self.row_queue.clone().detach())
            loss_col = self._contrastive_loss_with_queue(col_q, col_k, self.col_queue.clone().detach())
            
            # Cross-modal contrastive loss
            loss_cross = self._cross_modal_loss_with_queue(row_q, col_k, self.col_queue.clone().detach())
            loss_cross += self._cross_modal_loss_with_queue(col_q, row_k, self.row_queue.clone().detach())
            loss_cross /= 2
            
            # Update queue
            self._dequeue_and_enqueue(row_k, col_k)
            
        else:
            # Simple contrastive loss without momentum
            loss_row = self._simple_contrastive_loss(row_q)
            loss_col = self._simple_contrastive_loss(col_q)
            loss_cross = self._simple_cross_modal_loss(row_q, col_q)
            
        total_loss = loss_row + loss_col + loss_cross
        
        return {
            'total_loss': total_loss,
            'row_loss': loss_row,
            'col_loss': loss_col,
            'cross_modal_loss': loss_cross
        }
        
    def _contrastive_loss_with_queue(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        queue: torch.Tensor
    ) -> torch.Tensor:
        """Contrastive loss with momentum queue."""
        # Positive logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # Negative logits
        l_neg = torch.einsum('nc,ck->nk', [q, queue])
        
        # Logits
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        
        # Labels
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)
        
        return F.cross_entropy(logits, labels)
        
    def _cross_modal_loss_with_queue(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        queue: torch.Tensor
    ) -> torch.Tensor:
        """Cross-modal contrastive loss with queue."""
        # Positive logits (cross-modal)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # Negative logits (same modality)
        l_neg = torch.einsum('nc,ck->nk', [q, queue])
        
        # Logits
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        
        # Labels
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)
        
        return F.cross_entropy(logits, labels)
        
    def _simple_contrastive_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Simple contrastive loss without queue."""
        batch_size = embeddings.shape[0]
        
        # Similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Mask for positive pairs (diagonal)
        mask = torch.eye(batch_size, device=embeddings.device).bool()
        
        # Labels
        labels = torch.arange(batch_size, device=embeddings.device)
        
        return F.cross_entropy(sim_matrix, labels)
        
    def _simple_cross_modal_loss(
        self,
        row_embeddings: torch.Tensor,
        col_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Simple cross-modal contrastive loss."""
        batch_size = row_embeddings.shape[0]
        
        # Cross-modal similarity
        sim_matrix = torch.matmul(row_embeddings, col_embeddings.T) / self.temperature
        
        # Labels (diagonal should be positive)
        labels = torch.arange(batch_size, device=row_embeddings.device)
        
        return F.cross_entropy(sim_matrix, labels)