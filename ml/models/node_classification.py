"""
Node classification head for predicting attributes like part-of-speech tags.

This module implements the node classification component of the heterogeneous GNN,
enabling the prediction of node attributes such as POS tags.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)

class NodeClassificationHead(nn.Module):
    """Node classification head for predicting node attributes."""
    
    def __init__(self, 
                 embedding_dim: int, 
                 num_classes: int,
                 hidden_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 task_name: str = "pos_tags"):
        """
        Initialize the node classification head.
        
        Args:
            embedding_dim: Dimension of node embeddings
            num_classes: Number of output classes (e.g., number of POS tags)
            hidden_dim: Hidden dimension for intermediate layers (if None, no hidden layer is used)
            dropout: Dropout probability
            task_name: Name of the classification task (used for logging and output keys)
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.task_name = task_name
        
        # Classifier network
        if hidden_dim is None:
            # Simple linear classifier
            self.classifier = nn.Linear(embedding_dim, num_classes)
        else:
            # MLP classifier with one hidden layer
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                embeddings: torch.Tensor, 
                node_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for node classification.
        
        Args:
            embeddings: Node embeddings
            node_indices: Optional indices of nodes to classify (if None, classify all nodes)
            
        Returns:
            Logits for class probabilities
        """
        if node_indices is not None:
            # Only predict for specified nodes
            embedding_subset = embeddings[node_indices]
        else:
            # Predict for all nodes
            embedding_subset = embeddings
        
        # Apply dropout and classification layer
        embedding_subset = self.dropout(embedding_subset)
        logits = self.classifier(embedding_subset)
        
        return logits
    
    def predict(self, 
                embeddings: torch.Tensor, 
                node_indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict class probabilities and labels for nodes.
        
        Args:
            embeddings: Node embeddings
            node_indices: Optional indices of nodes to classify (if None, classify all nodes)
            
        Returns:
            Tuple of (predicted_labels, class_probabilities)
        """
        # Get logits
        logits = self.forward(embeddings, node_indices)
        
        # Convert to probabilities and predicted labels
        probs = F.softmax(logits, dim=1)
        pred_labels = torch.argmax(probs, dim=1)
        
        return pred_labels, probs
    
    def loss(self, 
             embeddings: torch.Tensor, 
             labels: torch.Tensor,
             node_indices: Optional[torch.Tensor] = None,
             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the cross-entropy loss for node classification.
        
        Args:
            embeddings: Node embeddings
            labels: Ground truth labels
            node_indices: Optional indices of nodes to classify
            mask: Optional mask for valid labels (1 for valid, 0 for invalid)
            
        Returns:
            Cross-entropy loss value
        """
        # Get logits
        logits = self.forward(embeddings, node_indices)
        
        # Extract labels for the specified nodes
        if node_indices is not None:
            target = labels[node_indices]
        else:
            target = labels
        
        # Apply mask if provided
        if mask is not None:
            # Create a mask for valid labels (where mask=1)
            if node_indices is not None:
                valid_mask = mask[node_indices]
            else:
                valid_mask = mask
            
            # Filter out invalid labels (e.g., -1)
            valid_indices = torch.where(valid_mask)[0]
            logits = logits[valid_indices]
            target = target[valid_indices]
        
        # Return zero loss if no valid labels
        if len(target) == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, target)
        
        return loss
    
    def evaluate(self, 
                embeddings: torch.Tensor, 
                labels: torch.Tensor,
                node_indices: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Evaluate the model's performance on node classification.
        
        Args:
            embeddings: Node embeddings
            labels: Ground truth labels
            node_indices: Optional indices of nodes to classify
            mask: Optional mask for valid labels (1 for valid, 0 for invalid)
            
        Returns:
            Dictionary with evaluation metrics (accuracy, etc.)
        """
        # Get predictions
        pred_labels, _ = self.predict(embeddings, node_indices)
        
        # Extract labels for the specified nodes
        if node_indices is not None:
            target = labels[node_indices]
        else:
            target = labels
        
        # Apply mask if provided
        if mask is not None:
            # Create a mask for valid labels (where mask=1)
            if node_indices is not None:
                valid_mask = mask[node_indices]
            else:
                valid_mask = mask
            
            # Filter out invalid labels (e.g., -1)
            valid_indices = torch.where(valid_mask)[0]
            pred_labels = pred_labels[valid_indices]
            target = target[valid_indices]
        
        # Return zero metrics if no valid labels
        if len(target) == 0:
            return {
                f"{self.task_name}_accuracy": 0.0,
                f"{self.task_name}_num_samples": 0
            }
        
        # Compute accuracy
        correct = (pred_labels == target).sum().item()
        total = len(target)
        accuracy = correct / total
        
        # Return metrics
        return {
            f"{self.task_name}_accuracy": accuracy,
            f"{self.task_name}_num_samples": total
        }
    
    def predict_proba_for_classes(self, 
                                 embeddings: torch.Tensor,
                                 node_indices: torch.Tensor) -> Dict[int, float]:
        """
        Predict class probabilities for specific nodes.
        
        Args:
            embeddings: Node embeddings
            node_indices: Indices of nodes to classify
            
        Returns:
            Dictionary mapping class index to probability for each node
        """
        # Get logits and convert to probabilities
        logits = self.forward(embeddings, node_indices)
        probs = F.softmax(logits, dim=1)
        
        # Create a dictionary for each node's class probabilities
        result = {}
        for i, node_idx in enumerate(node_indices):
            node_probs = {cls_idx: prob.item() for cls_idx, prob in enumerate(probs[i])}
            result[node_idx.item()] = node_probs 