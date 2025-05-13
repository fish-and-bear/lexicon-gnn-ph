"""
Graph encoder for lexical knowledge graphs with self-supervised pretraining.

This module implements the graph encoder that combines the core HGNN with
pretraining capabilities based on HGMAE (Heterogeneous Graph Masked Autoencoder).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

from .hgnn import HeterogeneousGNN

logger = logging.getLogger(__name__)

class GraphEncoder(nn.Module):
    """
    Graph encoder for lexical knowledge graphs with self-supervised pretraining.
    
    This class wraps the core HGNN and adds pretraining functionality
    using a masked autoencoder approach.
    """
    
    def __init__(self, 
                 in_dim: int, 
                 hidden_dim: int, 
                 out_dim: int,
                 rel_names: List[str],
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 mask_rate: float = 0.3,
                 decoder_layers: int = 1):
        """
        Initialize the graph encoder.
        
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden feature dimension
            out_dim: Output feature dimension
            rel_names: List of relation types in the graph
            num_layers: Number of GNN layers
            dropout: Dropout probability
            mask_rate: Masking rate for pretraining
            decoder_layers: Number of layers in the decoder
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.rel_names = rel_names
        self.mask_rate = mask_rate
        
        # Encoder: core HGNN
        self.encoder = HeterogeneousGNN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            rel_names=rel_names,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Decoder: simpler HGNN for reconstruction
        self.decoder = HeterogeneousGNN(
            in_dim=out_dim,
            hidden_dim=hidden_dim,
            out_dim=in_dim,  # Reconstruct original feature dim
            rel_names=rel_names,
            num_layers=decoder_layers,
            dropout=dropout
        )
        
        # Feature reconstruction head
        self.feature_decoder = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
        )
        
        # Edge prediction head
        self.edge_predictor = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(rel_names))
        )
    
    def forward(self, g, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the encoder only (for downstream tasks).
        
        Args:
            g: DGL heterogeneous graph
            features: Dictionary of node features
            
        Returns:
            Node embeddings
        """
        # Just use the encoder for the forward pass
        return self.encoder(g, features)
    
    def encode(self, g, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode the graph into node embeddings.
        
        Args:
            g: DGL heterogeneous graph
            features: Dictionary of node features
            
        Returns:
            Node embeddings
        """
        return self.encoder(g, features)
    
    def mask_features(self, 
                     features: torch.Tensor, 
                     mask_rate: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mask input features for self-supervised pretraining.
        
        Args:
            features: Input node features
            mask_rate: Masking rate (if None, use self.mask_rate)
            
        Returns:
            Tuple of (masked_features, mask)
        """
        mask_rate = mask_rate if mask_rate is not None else self.mask_rate
        
        # Create a mask tensor
        num_nodes = features.shape[0]
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=features.device)
        
        # Randomly select nodes to mask
        num_masked = int(num_nodes * mask_rate)
        masked_indices = random.sample(range(num_nodes), num_masked)
        mask[masked_indices] = True
        
        # Create masked features (replace masked nodes with zeros)
        masked_features = features.clone()
        masked_features[mask] = 0.0
        
        return masked_features, mask
    
    def mask_edges(self, 
                  g: dgl.DGLGraph, 
                  mask_rate: Optional[float] = None) -> Tuple[dgl.DGLGraph, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Mask edges for self-supervised pretraining.
        
        Args:
            g: DGL heterogeneous graph
            mask_rate: Masking rate (if None, use self.mask_rate)
            
        Returns:
            Tuple of (masked_graph, masked_edges)
        """
        mask_rate = mask_rate if mask_rate is not None else self.mask_rate
        
        # Create a new graph with masked edges
        masked_graph = dgl.DGLGraph()
        masked_graph.add_nodes(g.num_nodes())
        
        # Dictionary to store masked edges
        masked_edges = {}
        
        # Process each relation type
        for rel in g.etypes:
            # Get edges for this relation
            src, dst = g.edges(etype=rel)
            num_edges = len(src)
            
            # Randomly select edges to mask
            num_masked = int(num_edges * mask_rate)
            masked_indices = random.sample(range(num_edges), num_masked)
            mask = torch.zeros(num_edges, dtype=torch.bool)
            mask[masked_indices] = True
            
            # Add non-masked edges to the new graph
            non_masked_src = src[~mask]
            non_masked_dst = dst[~mask]
            masked_graph.add_edges(non_masked_src, non_masked_dst, etype=rel)
            
            # Store masked edges
            masked_src = src[mask]
            masked_dst = dst[mask]
            masked_edges[rel] = (masked_src, masked_dst)
        
        return masked_graph, masked_edges
    
    def pretraining_step(self, 
                        g: dgl.DGLGraph, 
                        features: Dict[str, torch.Tensor], 
                        mask_features_rate: float = 0.3,
                        mask_edges_rate: float = 0.3) -> Dict[str, torch.Tensor]:
        """
        Perform a pretraining step with both feature and edge masking.
        
        Args:
            g: DGL heterogeneous graph
            features: Dictionary of node features
            mask_features_rate: Rate of nodes to mask features
            mask_edges_rate: Rate of edges to mask
            
        Returns:
            Dictionary with loss values
        """
        device = next(iter(features.values())).device
        
        # Concatenate features if needed
        if isinstance(features, dict):
            x = torch.cat([features[k] for k in sorted(features.keys())], dim=1)
        else:
            x = features
        
        # 1. Mask features
        masked_features, feature_mask = self.mask_features(x, mask_features_rate)
        
        # 2. Mask edges
        masked_graph, masked_edges = self.mask_edges(g, mask_edges_rate)
        
        # 3. Encode with masked inputs
        node_embeddings = self.encoder(masked_graph, masked_features)
        
        # 4. Decode for feature reconstruction
        reconstructed_features = self.feature_decoder(node_embeddings)
        
        # 5. Compute feature reconstruction loss (only for masked nodes)
        feature_loss = F.mse_loss(reconstructed_features[feature_mask], x[feature_mask])
        
        # 6. Edge prediction loss
        edge_loss = 0.0
        num_edge_types = 0
        
        for rel, (src, dst) in masked_edges.items():
            if len(src) == 0:
                continue
                
            # Get embeddings for edge endpoints
            src_emb = node_embeddings[src]
            dst_emb = node_embeddings[dst]
            
            # Concatenate embeddings
            edge_emb = torch.cat([src_emb, dst_emb], dim=1)
            
            # Predict edge type
            edge_preds = self.edge_predictor(edge_emb)
            
            # One-hot encode the true edge type
            rel_idx = self.rel_names.index(rel)
            rel_labels = torch.full((len(src),), rel_idx, dtype=torch.long, device=device)
            
            # Calculate cross entropy loss
            rel_loss = F.cross_entropy(edge_preds, rel_labels)
            edge_loss += rel_loss
            num_edge_types += 1
        
        if num_edge_types > 0:
            edge_loss /= num_edge_types
        
        # 7. Combine losses
        total_loss = feature_loss + edge_loss
        
        return {
            'total_loss': total_loss,
            'feature_loss': feature_loss,
            'edge_loss': edge_loss
        }
    
    def pretrain(self, 
                g: dgl.DGLGraph, 
                features: Dict[str, torch.Tensor],
                num_epochs: int = 100,
                lr: float = 0.001,
                mask_features_rate: float = 0.3,
                mask_edges_rate: float = 0.3,
                device: str = 'cuda'):
        """
        Pretrain the encoder using masked feature and edge reconstruction.
        
        Args:
            g: DGL heterogeneous graph
            features: Dictionary of node features
            num_epochs: Number of pretraining epochs
            lr: Learning rate
            mask_features_rate: Rate of nodes to mask features
            mask_edges_rate: Rate of edges to mask
            device: Device to train on ('cuda' or 'cpu')
            
        Returns:
            Dictionary of training history
        """
        # Move model to device
        self.to(device)
        
        # Move features to device
        features = {k: v.to(device) for k, v in features.items()}
        
        # Create optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # Training history
        history = {
            'total_loss': [],
            'feature_loss': [],
            'edge_loss': []
        }
        
        # Training loop
        for epoch in range(num_epochs):
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            losses = self.pretraining_step(g, features, mask_features_rate, mask_edges_rate)
            
            # Backward pass
            losses['total_loss'].backward()
            
            # Update parameters
            optimizer.step()
            
            # Record history
            for k, v in losses.items():
                history[k].append(v.item())
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs}: "
                    f"total_loss={losses['total_loss']:.4f}, "
                    f"feature_loss={losses['feature_loss']:.4f}, "
                    f"edge_loss={losses['edge_loss']:.4f}"
                )
        
        logger.info(f"Pretraining completed: final loss={history['total_loss'][-1]:.4f}")
        return history 