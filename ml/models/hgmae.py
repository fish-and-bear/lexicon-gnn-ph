"""
Heterogeneous Graph Masked Autoencoder (HGMAE) for self-supervised pre-training.

This module implements HGMAE pre-training for heterogeneous graphs, with specific
adaptations for lexical knowledge graphs as described in the paper:
"Multi-Relational Graph Neural Networks for Automated Knowledge Graph Enhancement 
in Low-Resource Philippine Languages"

References:
- HGMAE (Tian et al., 2023)
- GraphGPS (Rampasek et al., 2022)
- Exphormer (Shirzad et al., 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import dgl.function as fn
from typing import Dict, List, Tuple, Optional, Union, Any, Set
import logging
import numpy as np
import random
from copy import deepcopy

from ml.models.hgnn import HeterogeneousGNN, GraphGPSLayer, RelationalGraphConv

logger = logging.getLogger(__name__)


class HGMAE(nn.Module):
    """
    Heterogeneous Graph Masked Autoencoder for self-supervised pre-training.
    Implements node feature and structure masking+reconstruction for heterogeneous graphs.
    """
    
    def __init__(self, 
                 in_dim: int, 
                 hidden_dim: int,
                 out_dim: int,
                 rel_names: List[str],
                 node_types: List[str] = ["word", "definition", "etymology"],
                 mask_rate: float = 0.3,
                 feature_mask_rate: float = 0.3,
                 edge_mask_rate: float = 0.3,
                 metapath_mask: bool = True,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 residual: bool = True,
                 layer_norm: bool = True,
                 num_bases: int = 8,
                 sparsity: float = 0.9):
        """
        Initialize HGMAE model for heterogeneous graph pre-training.
        
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden dimension
            out_dim: Output dimension
            rel_names: List of relation names in the graph
            node_types: List of node types in the graph
            mask_rate: Ratio of nodes to mask
            feature_mask_rate: Ratio of features to mask
            edge_mask_rate: Ratio of edges to mask
            metapath_mask: Whether to use metapath-based masking
            num_layers: Number of GNN layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            residual: Whether to use residual connections
            layer_norm: Whether to use layer normalization
            num_bases: Number of bases for relation weight decomposition
            sparsity: Sparsity factor for sparse attention
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.rel_names = rel_names
        self.node_types = node_types
        
        self.mask_rate = mask_rate
        self.feature_mask_rate = feature_mask_rate
        self.edge_mask_rate = edge_mask_rate
        self.metapath_mask = metapath_mask
        
        # Encoder: Heterogeneous GNN with GraphGPS layers
        self.encoder = HeterogeneousGNN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            rel_names=rel_names,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            residual=residual,
            layer_norm=layer_norm,
            num_bases=num_bases,
            sparsity=sparsity
        )
        
        # Create feature decoders (for each node type)
        self.feature_decoders = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(out_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, in_dim)
            ) for node_type in node_types
        })
        
        # Create edge decoders for structure reconstruction (relation prediction)
        self.edge_decoder = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(rel_names))
        )
        
        # Masking token embeddings (different for each node type)
        self.mask_tokens = nn.ParameterDict({
            node_type: nn.Parameter(torch.zeros(1, in_dim))
            for node_type in node_types
        })
        
        # Initialize mask tokens
        for node_type in node_types:
            nn.init.normal_(self.mask_tokens[node_type], mean=0.0, std=0.02)
    
    def encode(self, g, feats_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encode node features using heterogeneous GNN.
        
        Args:
            g: Heterogeneous graph
            feats_dict: Dictionary of node features per type
            
        Returns:
            Dictionary of node embeddings per type
        """
        return self.encoder.encode(g, feats_dict)
    
    def decode_features(self, embeddings_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Decode node embeddings back to features.
        
        Args:
            embeddings_dict: Dictionary of node embeddings per type
            
        Returns:
            Dictionary of reconstructed features per type
        """
        reconstructed = {}
        for node_type, embeddings in embeddings_dict.items():
            if node_type in self.feature_decoders:
                reconstructed[node_type] = self.feature_decoders[node_type](embeddings)
            else:
                logger.warning(f"No decoder for node type: {node_type}")
        return reconstructed
    
    def decode_edges(self, 
                    src_embeddings: torch.Tensor, 
                    dst_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Decode edge probabilities from node embeddings.
        
        Args:
            src_embeddings: Source node embeddings
            dst_embeddings: Destination node embeddings
            
        Returns:
            Edge type logits
        """
        # Concatenate source and destination embeddings
        edge_features = torch.cat([src_embeddings, dst_embeddings], dim=1)
        
        # Get edge type predictions
        edge_logits = self.edge_decoder(edge_features)
        
        return edge_logits
    
    def apply_feature_masks(self, 
                          feats_dict: Dict[str, torch.Tensor], 
                          mask_rates: Optional[Dict[str, float]] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Apply feature masking to input features.
        
        Args:
            feats_dict: Dictionary of node features per type
            mask_rates: Optional dictionary of mask rates per node type 
            
        Returns:
            Tuple of (masked_feats_dict, mask_dict) where mask_dict contains boolean masks
        """
        if mask_rates is None:
            mask_rates = {node_type: self.feature_mask_rate for node_type in self.node_types}
        
        masked_feats_dict = {}
        mask_dict = {}
        
        for node_type, feats in feats_dict.items():
            num_nodes = feats.size(0)
            feat_dim = feats.size(1)
            
            # For each node, mask a percentage of its features
            # Create a mask tensor where True means the feature is masked
            mask = torch.zeros(num_nodes, feat_dim, dtype=torch.bool, device=feats.device)
            
            for i in range(num_nodes):
                # For linguistic data, we want to mask features in blocks rather than randomly
                # This better preserves the semantic structure of the embeddings
                if feat_dim > 100:  # For high-dim features like XLM-R embeddings
                    # Mask in continuous chunks to preserve embedding semantics
                    mask_size = int(feat_dim * mask_rates.get(node_type, self.feature_mask_rate))
                    if mask_size > 0:
                        start_idx = random.randint(0, feat_dim - mask_size)
                        mask[i, start_idx:start_idx + mask_size] = True
                else:
                    # For smaller feature vectors, use random masking
                    mask[i] = torch.rand(feat_dim, device=feats.device) < mask_rates.get(node_type, self.feature_mask_rate)
            
            # Apply the mask: replace masked features with mask token
            masked_feats = feats.clone()
            masked_feats[mask] = self.mask_tokens[node_type].expand(mask.sum().item(), -1)
            
            masked_feats_dict[node_type] = masked_feats
            mask_dict[node_type] = mask
        
        return masked_feats_dict, mask_dict
    
    def apply_structure_masks(self, 
                             g: dgl.DGLGraph,
                             mask_rate: float = None) -> Tuple[dgl.DGLGraph, Dict[Tuple[str, str, str], torch.Tensor]]:
        """
        Apply edge masking to input graph.
        
        Args:
            g: Heterogeneous graph
            mask_rate: Edge mask rate
            
        Returns:
            Tuple of (masked_graph, edge_masks) where edge_masks is a dictionary of 
            masked edge indices keyed by canonical edge type (src_type, rel_type, dst_type)
        """
        if mask_rate is None:
            mask_rate = self.edge_mask_rate
        
        # Create a new graph to avoid modifying the input
        masked_g = dgl.heterograph({})
        edge_masks = {}
        
        # Process each edge type
        for canonical_etype in g.canonical_etypes:
            src_type, rel_type, dst_type = canonical_etype
            
            if g.num_edges(etype=rel_type) == 0:
                continue  # Skip edge types with no edges
                
            # Get all edges of this type
            src_nodes, dst_nodes = g.edges(etype=rel_type)
            
            # Determine how many edges to mask
            num_edges = len(src_nodes)
            num_masked = int(num_edges * mask_rate)
            
            if self.metapath_mask and num_masked > 0:
                # Metapath-aware masking: mask edges belonging to the same metapath
                # This is especially important for lexical graphs where patterns like
                # translation-cognate-translation form meaningful paths
                # For simplicity, we'll group edges by source node and mask random groups
                
                # Group edges by source node
                src_groups = {}
                for i, (src, dst) in enumerate(zip(src_nodes, dst_nodes)):
                    src_id = src.item()
                    if src_id not in src_groups:
                        src_groups[src_id] = []
                    src_groups[src_id].append(i)
                
                # Shuffle the groups and select enough to reach num_masked
                group_keys = list(src_groups.keys())
                random.shuffle(group_keys)
                
                # Select groups until we reach desired number of masked edges
                mask_indices = []
                for key in group_keys:
                    indices = src_groups[key]
                    if len(mask_indices) + len(indices) <= num_masked:
                        mask_indices.extend(indices)
                    else:
                        # Take what we need from this group
                        needed = num_masked - len(mask_indices)
                        if needed > 0:
                            mask_indices.extend(indices[:needed])
                        break
            else:
                # Simple random masking
                mask_indices = random.sample(range(num_edges), num_masked) if num_masked > 0 else []
                
            # Convert to tensor
            mask = torch.zeros(num_edges, dtype=torch.bool, device=g.device)
            if mask_indices:
                mask[mask_indices] = True
            
            # Store masked edge indices
            edge_masks[canonical_etype] = mask
            
            # Add edges to masked graph (excluding masked edges)
            keep_mask = ~mask
            if keep_mask.any():
                if src_type not in masked_g.ntypes:
                    masked_g.add_nodes(g.num_nodes(src_type), ntype=src_type)
                if dst_type not in masked_g.ntypes:
                    masked_g.add_nodes(g.num_nodes(dst_type), ntype=dst_type)
                masked_g.add_edges(
                    src_nodes[keep_mask], 
                    dst_nodes[keep_mask], 
                    etype=(src_type, rel_type, dst_type)
                )
        
        return masked_g, edge_masks
    
    def compute_feature_loss(self, 
                           pred_feats_dict: Dict[str, torch.Tensor],
                           orig_feats_dict: Dict[str, torch.Tensor],
                           mask_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute feature reconstruction loss.
        
        Args:
            pred_feats_dict: Dictionary of predicted features per type
            orig_feats_dict: Dictionary of original features per type
            mask_dict: Dictionary of feature masks per type
            
        Returns:
            Feature reconstruction loss
        """
        loss = 0.0
        total_masked = 0
        
        for node_type in self.node_types:
            if node_type not in pred_feats_dict or node_type not in orig_feats_dict or node_type not in mask_dict:
                continue
                
            pred = pred_feats_dict[node_type]
            orig = orig_feats_dict[node_type]
            mask = mask_dict[node_type]
            
            if not mask.any():
                continue  # Skip if no features were masked
                
            # Compute cosine similarity loss for continuous features 
            # (better for embeddings than MSE, scaled to [0, 2])
            pred_masked = pred[mask].view(-1, pred.size(-1))
            orig_masked = orig[mask].view(-1, orig.size(-1))
            
            # Normalize vectors for cosine similarity
            pred_masked_norm = F.normalize(pred_masked, p=2, dim=1)
            orig_masked_norm = F.normalize(orig_masked, p=2, dim=1)
            
            # 2 - 2 * cosine similarity = 2 * (1 - cosine similarity)
            batch_loss = 2 - 2 * (pred_masked_norm * orig_masked_norm).sum(dim=1)
            
            # Average loss for this node type
            type_loss = batch_loss.mean()
            loss += type_loss
            total_masked += 1
        
        return loss / max(1, total_masked)
    
    def compute_edge_loss(self, 
                        pred_logits: torch.Tensor,
                        true_rels: torch.Tensor) -> torch.Tensor:
        """
        Compute edge prediction loss.
        
        Args:
            pred_logits: Predicted relation logits
            true_rels: True relation indices
            
        Returns:
            Edge prediction loss (cross entropy)
        """
        return F.cross_entropy(pred_logits, true_rels)
    
    def forward(self, g: dgl.DGLGraph, feats_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Forward pass with masking and reconstruction.
        
        Args:
            g: Heterogeneous graph
            feats_dict: Dictionary of node features per type
            
        Returns:
            Dictionary with losses and predictions
        """
        # 1. Apply feature and structure masking
        masked_feats_dict, feat_masks = self.apply_feature_masks(feats_dict)
        masked_g, edge_masks = self.apply_structure_masks(g)
        
        # Track edge info for reconstruction
        masked_edges = {
            rel_type: {
                'src': [],
                'dst': [],
                'rel': []
            }
            for rel_type in self.rel_names
        }
        
        for (src_type, rel_type, dst_type), mask in edge_masks.items():
            if not mask.any():
                continue
                
            # Get source and destination nodes for masked edges
            src_nodes, dst_nodes = g.edges(etype=rel_type)
            masked_src = src_nodes[mask]
            masked_dst = dst_nodes[mask]
            
            # Store for edge reconstruction
            rel_idx = self.rel_names.index(rel_type)
            masked_edges[rel_type]['src'].extend(masked_src.tolist())
            masked_edges[rel_type]['dst'].extend(masked_dst.tolist())
            masked_edges[rel_type]['rel'].extend([rel_idx] * len(masked_src))
        
        # 2. Encode masked graph
        node_embeddings = self.encode(masked_g, masked_feats_dict)
        
        # 3. Reconstruct features
        reconstructed_feats = self.decode_features(node_embeddings)
        
        # 4. Compute feature reconstruction loss
        feat_recon_loss = self.compute_feature_loss(
            reconstructed_feats,
            feats_dict,
            feat_masks
        )
        
        # 5. Reconstruct edges (prepare inputs)
        edge_recon_loss = 0.0
        num_edge_types = 0
        
        for rel_type in self.rel_names:
            if not masked_edges[rel_type]['src']:
                continue  # Skip if no edges were masked for this relation
                
            # Get embeddings for source and destination nodes
            src_nodes = masked_edges[rel_type]['src']
            dst_nodes = masked_edges[rel_type]['dst']
            true_rels = torch.tensor(masked_edges[rel_type]['rel'], device=g.device)
            
            # Get embeddings for these nodes by node type
            # This assumes the format (node_type, relation, node_type) for canonical edge types
            src_type, _, dst_type = g.to_canonical_etype(rel_type)
            
            if src_type not in node_embeddings or dst_type not in node_embeddings:
                continue
                
            src_embeds = node_embeddings[src_type][src_nodes]
            dst_embeds = node_embeddings[dst_type][dst_nodes]
            
            # Predict relation
            edge_logits = self.decode_edges(src_embeds, dst_embeds)
            
            # Compute edge reconstruction loss
            rel_loss = self.compute_edge_loss(edge_logits, true_rels)
            edge_recon_loss += rel_loss
            num_edge_types += 1
        
        # Average edge reconstruction loss across relation types
        if num_edge_types > 0:
            edge_recon_loss /= num_edge_types
        
        # 6. Compute total loss (weighted sum of feature and edge reconstruction losses)
        # Default weights: 0.7 for feature reconstruction, 0.3 for edge reconstruction
        feat_weight = 0.7
        edge_weight = 0.3
        
        total_loss = feat_weight * feat_recon_loss
        if num_edge_types > 0:
            total_loss += edge_weight * edge_recon_loss
        
        # Return all relevant information
        return {
            'total_loss': total_loss,
            'feat_recon_loss': feat_recon_loss,
            'edge_recon_loss': edge_recon_loss if num_edge_types > 0 else torch.tensor(0.0, device=g.device),
            'node_embeddings': node_embeddings,
            'reconstructed_feats': reconstructed_feats
        }


def pretrain_hgmae(model, g, feats_dict, optimizer, epochs=100, device='cuda'):
    """
    Pre-train a HGMAE model on a heterogeneous graph.
    
    Args:
        model: HGMAE model
        g: Heterogeneous graph
        feats_dict: Dictionary of node features per type
        optimizer: PyTorch optimizer
        epochs: Number of training epochs
        device: Device to run training on
        
    Returns:
        Pre-trained model
    """
    logger.info(f"Pre-training HGMAE for {epochs} epochs")
    model.train()
    
    # Move graph and features to device
    g = g.to(device)
    feats_dict = {
        k: v.to(device) for k, v in feats_dict.items()
    }
    
    best_loss = float('inf')
    patience = 10  # Early stopping patience
    patience_counter = 0
    
    for epoch in range(epochs):
        # Forward pass
        optimizer.zero_grad()
        output = model(g, feats_dict)
        
        # Backward pass
        loss = output['total_loss']
        loss.backward()
        optimizer.step()
        
        # Log progress
        if (epoch + 1) % 10 == 0:
            feat_loss = output['feat_recon_loss'].item()
            edge_loss = output['edge_recon_loss'].item()
            total_loss = loss.item()
            
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                       f"Total Loss = {total_loss:.4f}, "
                       f"Feature Loss = {feat_loss:.4f}, "
                       f"Edge Loss = {edge_loss:.4f}")
        
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            # Save best model
            best_state = deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                # Restore best model
                model.load_state_dict(best_state)
                break
    
    logger.info("Pre-training complete")
    return model 