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
                 original_feat_dims: Dict[str, int],
                 hidden_dim: int, 
                 out_dim: int,
                 node_types: List[str],
                 rel_names: List[str],
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 1,
                 dropout: float = 0.1,
                 feature_mask_rate: float = 0.3,
                 edge_mask_rate: float = 0.3,
                 num_heads: int = 8,
                 residual: bool = True,
                 layer_norm: bool = True,
                 num_bases: int = 8,
                 hgnn_sparsity: float = 0.9):
        """
        Initialize the graph encoder.
        
        Args:
            original_feat_dims: Dict mapping node type to its original feature dimension.
            hidden_dim: Hidden dimension for GNN layers (common internal dimension).
            out_dim: Output feature dimension of the main encoder GNN.
            node_types: List of all unique node type names.
            rel_names: List of relation types in the graph.
            num_encoder_layers: Number of GNN layers in the main encoder.
            num_decoder_layers: Number of GNN layers in the reconstruction decoder.
            dropout: Dropout probability.
            feature_mask_rate: Rate for masking features within nodes.
            edge_mask_rate: Rate for masking edges.
            num_heads: Number of attention heads for HGNN's GraphGPSLayer.
            residual: Whether HGNN's GraphGPSLayer uses residual connections.
            layer_norm: Whether HGNN's GraphGPSLayer uses layer normalization.
            num_bases: Number of bases for HGNN's RelationalGraphConv.
            hgnn_sparsity: Sparsity factor for HGNN's GraphGPSLayer.
        """
        super().__init__()
        
        self.original_feat_dims = original_feat_dims
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.node_types = sorted(list(set(node_types)))
        self.rel_names = rel_names
        self.feature_mask_rate = feature_mask_rate
        self.edge_mask_rate = edge_mask_rate
        
        # Encoder: core HGNN
        self.encoder = HeterogeneousGNN(
            original_feat_dims=self.original_feat_dims,
            hidden_dim=self.hidden_dim,
            out_dim=self.out_dim,
            node_types=self.node_types,
            rel_names=self.rel_names,
            num_layers=num_encoder_layers,
            dropout=dropout,
            num_heads=num_heads,
            residual=residual,
            layer_norm=layer_norm,
            num_bases=num_bases,
            sparsity=hgnn_sparsity
        )
        
        # Decoder: simpler HGNN for reconstruction of original features
        self.feature_reconstruction_decoder = HeterogeneousGNN(
            original_feat_dims={ntype: self.out_dim for ntype in self.node_types},
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            node_types=self.node_types,
            rel_names=self.rel_names,
            num_layers=num_decoder_layers,
            dropout=dropout,
            num_heads=num_heads,
            residual=residual,
            layer_norm=layer_norm,
            num_bases=num_bases,
            sparsity=hgnn_sparsity
        )
        
        # Feature reconstruction heads (one for each node type)
        self.feature_reconstruction_heads = nn.ModuleDict()
        for ntype in self.node_types:
            if ntype in self.original_feat_dims:
                self.feature_reconstruction_heads[ntype] = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim // 2 if self.hidden_dim > 1 else 1),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim // 2 if self.hidden_dim > 1 else 1, self.original_feat_dims[ntype])
                )
        
        # Edge prediction head (operates on encoder's output embeddings: self.out_dim)
        self.edge_predictor = nn.Sequential(
            nn.Linear(self.out_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, len(self.rel_names))
        )

        # Masking token embeddings (scalar learnable token for each node type)
        self.mask_value_tokens = nn.ParameterDict({
            node_type: nn.Parameter(torch.zeros(1))
            for node_type in self.node_types if node_type in self.original_feat_dims
        })
        for ntype in self.mask_value_tokens:
            nn.init.normal_(self.mask_value_tokens[ntype], mean=0.0, std=0.02)

    def forward(self, g: dgl.DGLGraph, features_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the encoder only (for downstream tasks).
        
        Args:
            g: DGL heterogeneous graph.
            features_dict: Dictionary of node features {node_type: Tensor}.
        Returns:
            Node embeddings_dict: Dictionary of output node embeddings {node_type: Tensor}.
        """
        return self.encoder(g, features_dict)
    
    def encode(self, g: dgl.DGLGraph, features_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encode the graph into node embeddings.
        
        Args:
            g: DGL heterogeneous graph
            features_dict: Dictionary of node features {node_type: Tensor}
            
        Returns:
            Node embeddings_dict: Dictionary of output node embeddings {node_type: Tensor}
        """
        return self.encoder(g, features_dict)
    
    def apply_feature_masks(self, 
                          feats_dict: Dict[str, torch.Tensor], 
                          mask_rate_override: Optional[float] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Mask input features for self-supervised pretraining.
        Operates on a dictionary of features.
        
        Args:
            feats_dict: Dictionary of node features {node_type: Tensor}
            mask_rate_override: Rate for masking features within nodes (if None, use self.feature_mask_rate)
            
        Returns:
            Tuple of (masked_feats_dict, feature_masks_dict)
        """
        effective_mask_rate = mask_rate_override if mask_rate_override is not None else self.feature_mask_rate
        
        masked_feats_dict = {}
        feature_masks_dict = {}
        
        for ntype, feats in feats_dict.items():
            if ntype not in self.original_feat_dims:
                masked_feats_dict[ntype] = feats
                feature_masks_dict[ntype] = torch.zeros_like(feats, dtype=torch.bool, device=feats.device)
                continue

            num_nodes, feat_dim = feats.shape
            
            node_feature_mask = torch.rand(num_nodes, feat_dim, device=feats.device) < effective_mask_rate
            
            masked_feats = feats.clone()
            masked_feats[node_feature_mask] = self.mask_value_tokens[ntype]
            
            masked_feats_dict[ntype] = masked_feats
            feature_masks_dict[ntype] = node_feature_mask
            
        return masked_feats_dict, feature_masks_dict
    
    def mask_edges(self, 
                  g: dgl.DGLGraph, 
                  mask_rate_override: Optional[float] = None) -> Tuple[dgl.DGLGraph, Dict[Tuple[str, str, str], torch.Tensor]]:
        """
        Similar to HGMAE.apply_structure_masks
        
        Args:
            g: DGL heterogeneous graph
            mask_rate_override: Rate for masking edges (if None, use self.edge_mask_rate)
            
        Returns:
            Tuple of (masked_graph, edge_masks_info)
        """
        effective_mask_rate = mask_rate_override if mask_rate_override is not None else self.edge_mask_rate
        
        current_formats = g.formats().get('created', ['coo'])
        if not current_formats: current_formats = ['coo']
        masked_g = g.formats(current_formats)

        for canonical_etype in g.canonical_etypes:
            if g.num_edges(etype=canonical_etype) > 0:
                all_eids = g.edges(etype=canonical_etype, form='eid')
                masked_g.remove_edges(all_eids, etype=canonical_etype)

        edge_masks_info = {}

        for canonical_etype in g.canonical_etypes:
            src_type, rel_name, dst_type = canonical_etype
            
            if g.num_edges(etype=rel_name) == 0:
                continue
                
            src_nodes, dst_nodes = g.edges(etype=canonical_etype)
            num_edges = len(src_nodes)
            
            if num_edges == 0:
                edge_masks_info[canonical_etype] = torch.empty(0, dtype=torch.bool, device=g.device)
                continue

            num_masked = int(num_edges * effective_mask_rate)
            mask_indices = random.sample(range(num_edges), num_masked)
            
            edge_mask = torch.zeros(num_edges, dtype=torch.bool, device=g.device)
            if mask_indices:
                edge_mask[mask_indices] = True
            
            edge_masks_info[canonical_etype] = edge_mask
            
            keep_mask = ~edge_mask
            if keep_mask.any():
                masked_g.add_edges(src_nodes[keep_mask], dst_nodes[keep_mask], etype=canonical_etype)
        
        return masked_g, edge_masks_info

    def compute_feature_reconstruction_loss(self, 
                                           reconstructed_feats_dict: Dict[str, torch.Tensor],
                                           original_feats_dict: Dict[str, torch.Tensor],
                                           feature_masks_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = 0.0
        num_node_types_in_loss = 0
        for ntype in self.node_types:
            if ntype not in reconstructed_feats_dict or \
               ntype not in original_feats_dict or \
               ntype not in feature_masks_dict:
                continue

            pred_feats = reconstructed_feats_dict[ntype]
            orig_feats = original_feats_dict[ntype]
            mask = feature_masks_dict[ntype]

            if not mask.any():
                continue

            loss += F.mse_loss(pred_feats[mask], orig_feats[mask])
            num_node_types_in_loss += 1
        
        return loss / max(1, num_node_types_in_loss)

    def pretraining_step(self, 
                        g: dgl.DGLGraph, 
                        features_dict: Dict[str, torch.Tensor], 
                        mask_features_rate_override: Optional[float] = None,
                        mask_edges_rate_override: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Perform a pretraining step with both feature and edge masking.
        
        Args:
            g: DGL heterogeneous graph
            features_dict: Dictionary of node features {node_type: Tensor}
            mask_features_rate_override: Rate for masking features within nodes (if None, use self.feature_mask_rate)
            mask_edges_rate_override: Rate for masking edges (if None, use self.edge_mask_rate)
            
        Returns:
            Dictionary with loss values
        """
        # 1. Mask features and edges
        masked_features_dict, feature_masks_dict = self.apply_feature_masks(features_dict, mask_features_rate_override)
        masked_graph, edge_masks_info = self.mask_edges(g, mask_edges_rate_override)
        
        # 2. Encode with masked inputs using the main encoder
        encoded_embeddings_dict = self.encoder(masked_graph, masked_features_dict)
        
        # 3. Decode for feature reconstruction
        decoded_hidden_embeddings_dict = self.feature_reconstruction_decoder(masked_graph, encoded_embeddings_dict)

        #    Then, per-type heads project from self.hidden_dim to original_feat_dims.
        reconstructed_features_dict = {}
        for ntype, hidden_emb in decoded_hidden_embeddings_dict.items():
            if ntype in self.feature_reconstruction_heads:
                reconstructed_features_dict[ntype] = self.feature_reconstruction_heads[ntype](hidden_emb)
        
        # 4. Compute feature reconstruction loss
        feature_loss = self.compute_feature_reconstruction_loss(
            reconstructed_features_dict, features_dict, feature_masks_dict
        )
        
        # 5. Edge prediction loss
        edge_loss = torch.tensor(0.0, device=g.device)
        num_valid_edge_types_for_loss = 0
        
        all_masked_src_emb = []
        all_masked_dst_emb = []
        all_masked_rel_labels = []

        rel_name_to_idx = {name: i for i, name in enumerate(self.rel_names)}

        for canonical_etype, edge_mask in edge_masks_info.items():
            if not edge_mask.any():
                continue

            src_type, rel_name, dst_type = canonical_etype
            original_src_nodes, original_dst_nodes = g.edges(etype=canonical_etype)
            
            masked_src_nodes_for_type = original_src_nodes[edge_mask]
            masked_dst_nodes_for_type = original_dst_nodes[edge_mask]

            if len(masked_src_nodes_for_type) == 0:
                continue

            if not (src_type in encoded_embeddings_dict and dst_type in encoded_embeddings_dict):
                logger.warning(f"Embeddings for {src_type} or {dst_type} not found in encoder output. Skipping edge loss for {canonical_etype}.")
                continue

            src_emb = encoded_embeddings_dict[src_type][masked_src_nodes_for_type]
            dst_emb = encoded_embeddings_dict[dst_type][masked_dst_nodes_for_type]
            
            all_masked_src_emb.append(src_emb)
            all_masked_dst_emb.append(dst_emb)
            
            rel_idx = rel_name_to_idx.get(rel_name)
            if rel_idx is None:
                logger.error(f"Relation name {rel_name} not in self.rel_names mapping!")
                continue
            all_masked_rel_labels.extend([rel_idx] * len(masked_src_nodes_for_type))
            num_valid_edge_types_for_loss +=1

        if all_masked_src_emb:
            all_masked_src_emb_cat = torch.cat(all_masked_src_emb, dim=0)
            all_masked_dst_emb_cat = torch.cat(all_masked_dst_emb, dim=0)
            all_masked_rel_labels_cat = torch.tensor(all_masked_rel_labels, dtype=torch.long, device=g.device)

            edge_emb_for_predictor = torch.cat([all_masked_src_emb_cat, all_masked_dst_emb_cat], dim=1)
            
            edge_preds_logits = self.edge_predictor(edge_emb_for_predictor)
            
            if edge_preds_logits.shape[0] > 0 :
                edge_loss = F.cross_entropy(edge_preds_logits, all_masked_rel_labels_cat)
        
        # 6. Combine losses
        feat_loss_weight = 0.7
        edge_loss_weight = 0.3
        total_loss = feat_loss_weight * feature_loss
        if num_valid_edge_types_for_loss > 0 :
             total_loss += edge_loss_weight * edge_loss
        
        return {
            'total_loss': total_loss,
            'feature_loss': feature_loss,
            'edge_loss': edge_loss
        }
    
    def pretrain(self, 
                g: dgl.DGLGraph, 
                features_dict: Dict[str, torch.Tensor],
                num_epochs: int = 100,
                lr: float = 0.001,
                mask_features_rate_override: Optional[float] = None,
                mask_edges_rate_override: Optional[float] = None,
                device: Optional[str] = None):
        """
        Pretrain the encoder using masked feature and edge reconstruction.
        
        Args:
            g: DGL heterogeneous graph
            features_dict: Dictionary of node features {node_type: Tensor}
            num_epochs: Number of pretraining epochs
            lr: Learning rate
            mask_features_rate_override: Rate for masking features within nodes (if None, use self.feature_mask_rate)
            mask_edges_rate_override: Rate for masking edges (if None, use self.edge_mask_rate)
            device: Device to train on ('cuda' or 'cpu')
            
        Returns:
            Dictionary of training history
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        g = g.to(device)
        features_dict = {k: v.to(device) for k, v in features_dict.items()}
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        history = {'total_loss': [], 'feature_loss': [], 'edge_loss': []}
        
        logger.info(f"Starting pretraining on device: {device}")
        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()
            
            losses = self.pretraining_step(
                g, features_dict, 
                mask_features_rate_override, mask_edges_rate_override
            )
            
            if losses['total_loss'].requires_grad:
                losses['total_loss'].backward()
                optimizer.step()
            else:
                logger.warning(f"Epoch {epoch+1}: Total loss does not require grad. Skipping backward/step.")

            for k, v_loss in losses.items():
                history[k].append(v_loss.item())
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs}: "
                    f"Total Loss={history['total_loss'][-1]:.4f}, "
                    f"Feature Loss={history['feature_loss'][-1]:.4f}, "
                    f"Edge Loss={history['edge_loss'][-1]:.4f}"
                )
        
        logger.info(f"Pretraining completed: final total_loss={history['total_loss'][-1]:.4f}")
        return history 