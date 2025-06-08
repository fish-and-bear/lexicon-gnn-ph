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
import time
import sys # Added for sys.stdout.flush()

# Import autocast and GradScaler
from torch.cuda.amp import autocast, GradScaler

from ml.models.hgnn import HeterogeneousGNN, GraphGPSLayer, RelationalGraphConv

logger = logging.getLogger(__name__)

# Fix the deprecated PyTorch function warning
if hasattr(torch.utils._pytree, 'register_pytree_node'):
    register_func = torch.utils._pytree.register_pytree_node
else:
    register_func = torch.utils._pytree._register_pytree_node

# Use the appropriate function to avoid the warning 
if '_register_pytree_node' in str(torch.utils._pytree.__dict__):
    logger.info("Using legacy PyTorch pytree registration method")
else:
    logger.info("Using modern PyTorch pytree registration method")

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
                 original_feat_dims: Dict[str, int],
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
                 sparsity: float = 0.9,
                 pretrain_feat_loss_weight: float = 0.7,
                 pretrain_edge_loss_weight: float = 0.3):
        """
        Initialize HGMAE model for heterogeneous graph pre-training.
        
        Args:
            in_dim: Input feature dimension for the GNN encoder (after any initial projection)
            hidden_dim: Hidden dimension for GNN and decoders
            out_dim: Output dimension for GNN (input to decoders)
            rel_names: List of relation names in the graph
            original_feat_dims: Dictionary mapping node type to its original feature dimension
            node_types: List of node types in the graph
            mask_rate: Ratio of nodes to mask (currently not used for feature masking logic)
            feature_mask_rate: Ratio of features to mask within each node
            edge_mask_rate: Ratio of edges to mask
            metapath_mask: Whether to use metapath-based masking
            num_layers: Number of GNN layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            residual: Whether to use residual connections
            layer_norm: Whether to use layer normalization
            num_bases: Number of bases for relation weight decomposition
            sparsity: Sparsity factor for sparse attention
            pretrain_feat_loss_weight: Weight for feature reconstruction loss during pre-training.
            pretrain_edge_loss_weight: Weight for edge reconstruction loss during pre-training.
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.rel_names = rel_names
        self.node_types = node_types
        self.original_feat_dims = original_feat_dims
        
        self.mask_rate = mask_rate
        self.feature_mask_rate = feature_mask_rate
        self.edge_mask_rate = edge_mask_rate
        self.metapath_mask = metapath_mask
        self.pretrain_feat_loss_weight = pretrain_feat_loss_weight
        self.pretrain_edge_loss_weight = pretrain_edge_loss_weight
        
        # Encoder: Heterogeneous GNN
        self.encoder = HeterogeneousGNN(
            original_feat_dims=self.original_feat_dims,
            hidden_dim=self.hidden_dim,
            out_dim=self.out_dim,
            node_types=self.node_types,
            rel_names=self.rel_names,
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
                nn.Linear(hidden_dim, self.original_feat_dims[node_type])
            ) for node_type in self.node_types if node_type in self.original_feat_dims
        })
        
        # Create edge decoders for structure reconstruction (relation prediction)
        self.edge_decoder = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(rel_names))
        )
        
        # Masking token embeddings (scalar learnable token for each node type)
        self.mask_tokens = nn.ParameterDict({
            node_type: nn.Parameter(torch.zeros(1))
            for node_type in self.node_types
        })
        
        # Initialize mask tokens
        for node_type in self.node_types:
            nn.init.normal_(self.mask_tokens[node_type], mean=0.0, std=0.02)

        # Reconstruction loss function
        self.recon_loss_fn = nn.MSELoss()
    
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
            masked_feats[mask] = self.mask_tokens[node_type]
            
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
        
        # Clone the original graph using formats() to get a copy of structure and features.
        # This ensures a valid starting graph on the correct device.
        current_formats = g.formats()['created'] # Get list of already created formats (e.g., ['coo'])
        if not current_formats: # Fallback if for some reason no formats are listed as created
            current_formats = ['coo']
        masked_g = g.formats(current_formats) # Request a clone with these formats

        # Remove all existing edges from the cloned graph
        # We will add back only the unmasked edges later.
        for canonical_etype in g.canonical_etypes:
            if g.num_edges(etype=canonical_etype) > 0:
                all_eids = g.edges(etype=canonical_etype, form='eid')
                masked_g.remove_edges(all_eids, etype=canonical_etype)

        edge_masks = {}
        
        # Process each edge type from the original graph to determine masks
        for canonical_etype in g.canonical_etypes:
            src_type, rel_type, dst_type = canonical_etype
            
            if g.num_edges(etype=canonical_etype) == 0:
                continue  # Skip edge types with no edges
                
            # Get all edges of this type
            src_nodes, dst_nodes = g.edges(etype=canonical_etype)
            
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
                # Nodes are already correctly managed by the new graph initialization approach.
                # masked_g.add_nodes(g.num_nodes(src_type), ntype=src_type) # Not needed
                # masked_g.add_nodes(g.num_nodes(dst_type), ntype=dst_type) # Not needed
                masked_g.add_edges(
                    src_nodes[keep_mask], 
                    dst_nodes[keep_mask], 
                    etype=canonical_etype # Use the full canonical_etype tuple/string
                )
        
        return masked_g, edge_masks
    
    def compute_feature_loss(self, 
                           pred_feats_dict: Dict[str, torch.Tensor],
                           orig_feats_dict: Dict[str, torch.Tensor],
                           mask_dict: Dict[str, torch.Tensor],
                           target_device: torch.device,
                           return_details_for_eval: bool = False) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Compute feature reconstruction loss.
        
        Args:
            pred_feats_dict: Dictionary of predicted features per type
            orig_feats_dict: Dictionary of original features per type
            mask_dict: Dictionary of feature masks per type
            target_device: Device for intermediate operations
            return_details_for_eval: If True, returns a dict with 'loss' and 'details'
                                     (list of {pred_masked, true_masked}) for evaluation.
            
        Returns:
            Feature reconstruction loss (MSE). If return_details_for_eval is True,
            returns a dictionary: {'loss': loss_tensor, 'details': list_of_pred_true_pairs}
        """
        loss_components = [] # List to store individual loss tensors
        num_types_processed = 0
        
        eval_details_list = [] # For storing (pred_masked, true_masked) during evaluation

        # Move MSELoss to target_device
        self.recon_loss_fn = self.recon_loss_fn.to(target_device) # Ensure loss_fn is on target_device
        
        logger.debug(f"compute_feature_loss: pred_feats_dict keys: {list(pred_feats_dict.keys())}")
        logger.debug(f"compute_feature_loss: orig_feats_dict keys: {list(orig_feats_dict.keys())}")
        logger.debug(f"compute_feature_loss: mask_dict keys: {list(mask_dict.keys())}")

        for node_type, pred in pred_feats_dict.items():
            # Detailed logging for the current node_type
            has_orig = node_type in orig_feats_dict and orig_feats_dict[node_type] is not None
            has_mask = node_type in mask_dict and mask_dict[node_type] is not None
            
            orig_shape_str = "N/A"
            if has_orig:
                orig_tensor = orig_feats_dict[node_type]
                orig_shape_str = str(orig_tensor.shape) if isinstance(orig_tensor, torch.Tensor) else "Not Tensor"
            
            mask_shape_str = "N/A"
            if has_mask:
                mask_tensor = mask_dict[node_type]
                mask_shape_str = str(mask_tensor.shape) if isinstance(mask_tensor, torch.Tensor) else "Not Tensor"

            logger.debug(f"compute_feature_loss for node_type='{node_type}': "
                         f"pred.shape={pred.shape if isinstance(pred, torch.Tensor) else 'Not Tensor'}, "
                         f"has_orig={has_orig} (shape: {orig_shape_str}), "
                         f"has_mask={has_mask} (shape: {mask_shape_str})")

            if not has_orig or not has_mask:
                logger.warning(f"Skipping feature loss for '{node_type}': missing original features (exists: {has_orig}) or mask (exists: {has_mask}).")
                continue
                
            pred = pred.to(target_device) 
            mask = mask_dict[node_type].to(target_device)
            true = orig_feats_dict[node_type].to(target_device)
            
            if pred.shape[0] != mask.shape[0]: 
                logger.warning(
                    f"Shape mismatch for node_type '{node_type}' in feature reconstruction loss: "
                    f"pred has {pred.shape[0]} nodes, while mask/true (from original feats) have {mask.shape[0]} nodes. "
                    "This may indicate an inconsistency where the graph processed by the GNN has a different "
                    "number of nodes for this type than the input feature matrix."
                )
                if pred.shape[0] < mask.shape[0]:
                    logger.info(
                        f"  Adjusting by slicing mask and true tensors for '{node_type}' from {mask.shape[0]} to {pred.shape[0]} rows "
                        "to match pred. This assumes pred corresponds to the initial subset of original nodes."
                    )
                    mask = mask[:pred.shape[0]]
                    true = true[:pred.shape[0]]
                else: 
                    logger.error(
                        f"  Pred has MORE nodes ({pred.shape[0]}) than mask/true ({mask.shape[0]}) for '{node_type}'. "
                        "This is unexpected and loss cannot be reliably computed. Skipping loss for this type."
                    )
                    continue
            
            if pred.shape[1] != mask.shape[1] or pred.shape[1] != true.shape[1]: 
                logger.error(
                    f"Feature dimension mismatch for node_type '{node_type}': "
                    f"pred: {pred.shape[1]}, mask: {mask.shape[1]}, true: {true.shape[1]}. Skipping loss for this type."
                )
                continue

            if not torch.any(mask): 
                logger.info(f"No features were masked for node type '{node_type}'. Skipping feature loss calculation.")
                continue

            pred_masked_elements = pred[mask]
            true_masked_elements = true[mask]

            if pred_masked_elements.numel() == 0: 
                logger.info(f"No elements selected by mask for node type '{node_type}' after potential slicing. Skipping loss.")
                continue

            loss_val_tensor = self.recon_loss_fn(pred_masked_elements, true_masked_elements)
            loss_components.append(loss_val_tensor) # Keep as tensor
            
            if return_details_for_eval:
                eval_details_list.append({
                    'pred_masked': pred_masked_elements.detach(), # Detach for eval
                    'true_masked': true_masked_elements.detach()
                })
            
            num_types_processed += 1
            logger.debug(f"  Recon loss for {node_type}: {loss_val_tensor.item():.4f} (masked elements: {pred_masked_elements.numel()})")
        
        if loss_components: # If any losses were computed
            # Stack and mean to get average loss, maintaining gradient history
            final_loss_tensor = torch.stack(loss_components).mean()
        else: # No losses computed, return a zero tensor that requires grad if in training
            # The requires_grad should be true if this loss contributes to a parameter update.
            # It will be part of the larger computation graph, so its requires_grad status
            # will be determined by subsequent operations if it's used in training.
            # For safety, explicitly set based on whether overall model is training, though weights later control contribution.
            # The main training loop in pretrain_hgmae checks if self.training is true before calling backward.
            final_loss_tensor = torch.tensor(0.0, device=target_device, requires_grad=self.training) 

        if return_details_for_eval:
            return {
                'loss': final_loss_tensor,
                'details': eval_details_list
            }
        else:
            return final_loss_tensor
    
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
    
    def forward(self, g: dgl.DGLGraph, feats_dict: Dict[str, torch.Tensor], return_val_details: bool = False) -> Dict[str, Any]:
        """
        Forward pass with masking and reconstruction.
        
        Args:
            g: Heterogeneous graph (already on target device from training loop)
            feats_dict: Dictionary of node features per type (already on target device)
            return_val_details: If True and model is in eval mode, return detailed metrics.
            
        Returns:
            Dictionary with losses and predictions. 
            If model.training is False and return_val_details is True, also includes evaluation details.
        """
        # Determine the device from model parameters; assumes model is already on the target device.
        # This is crucial for ensuring all ops inside forward happen on the correct device.
        target_device = next(self.parameters()).device

        # Graph g and feats_dict are assumed to be on target_device already by the calling training loop (pretrain_hgmae).
        # No explicit g.to(target_device) or feats_dict[...].to(target_device) needed here if called correctly.

        # 1. Apply feature and structure masking
        # apply_feature_masks uses feats.device, which should be target_device.
        # apply_structure_masks needs to ensure masked_g is on target_device.
        masked_feats_dict, feat_masks = self.apply_feature_masks(feats_dict) # feats_dict is on target_device
        masked_g, edge_masks = self.apply_structure_masks(g) # g is on target_device
        
        # Ensure masked_g is on the target_device (apply_structure_masks should handle this, but verify)
        if masked_g.device != target_device:
            masked_g = masked_g.to(target_device)
            logger.debug(f"Moved masked_g to {target_device} within forward pass.")
        
        # 2. Encode masked graph
        # self.encoder (HeterogeneousGNN) should operate on target_device based on its parameters.
        # Inputs masked_g and masked_feats_dict must be on target_device.
        node_embeddings = self.encode(masked_g, masked_feats_dict)
        
        # 3. Reconstruct features
        # self.feature_decoders operate on target_device.
        # node_embeddings should be on target_device.
        reconstructed_feats = self.decode_features(node_embeddings)
        
        # 4. Compute feature reconstruction loss
        # self.compute_feature_loss needs to handle devices internally or ensure inputs are correct.
        # Pass target_device to compute_feature_loss for explicit internal device handling.
        feat_recon_loss_outputs = self.compute_feature_loss(
            reconstructed_feats,
            feats_dict, # Original features, already on target_device
            feat_masks, # feat_masks created on device of feats_dict, so on target_device
            target_device=target_device, # Pass target_device explicitly
            return_details_for_eval=(not self.training and return_val_details) # Pass evaluation flag
        )

        eval_feature_details = None # Initialize
        if isinstance(feat_recon_loss_outputs, dict): # In eval mode with details
            feat_recon_loss = feat_recon_loss_outputs['loss']
            eval_feature_details = feat_recon_loss_outputs['details']
        else: # Training mode or no details requested
            feat_recon_loss = feat_recon_loss_outputs
        
        # 5. Reconstruct edges (prepare inputs)
        # Initialize loss tensor on the target_device.
        edge_recon_loss_sum = torch.tensor(0.0, device=target_device, requires_grad=True if self.training and self.pretrain_edge_loss_weight > 0 else False)
        num_canonical_edge_types_for_loss = 0
        eval_edge_metrics_accumulated = {'total_correct': 0, 'total_processed': 0, 'accuracy': 0.0} # For eval

        for canonical_etype, mask_tensor in edge_masks.items():
            if not mask_tensor.any():
                continue
                
            src_type_str, rel_name_str, dst_type_str = canonical_etype
            original_src_nodes, original_dst_nodes = g.edges(etype=canonical_etype) # g is on target_device
            
            masked_src_nodes_for_loss = original_src_nodes[mask_tensor]
            masked_dst_nodes_for_loss = original_dst_nodes[mask_tensor]
            
            if masked_src_nodes_for_loss.numel() == 0: 
                continue

            if src_type_str not in node_embeddings or dst_type_str not in node_embeddings or \
               node_embeddings[src_type_str].nelement() == 0 or node_embeddings[dst_type_str].nelement() == 0:
                logger.warning(f"Embeddings missing or empty for {src_type_str} or {dst_type_str} in edge loss for {canonical_etype}. Skipping.")
                continue
            
            # Ensure indices are valid before gathering
            if masked_src_nodes_for_loss.max() >= node_embeddings[src_type_str].shape[0] or \
               masked_dst_nodes_for_loss.max() >= node_embeddings[dst_type_str].shape[0]:
                logger.warning(f"Index out of bounds for embeddings in edge loss for {canonical_etype}. Max src_idx: {masked_src_nodes_for_loss.max()}, Emb shape: {node_embeddings[src_type_str].shape[0]}. Max dst_idx: {masked_dst_nodes_for_loss.max()}, Emb shape: {node_embeddings[dst_type_str].shape[0]}. Skipping.")
                continue
                
            src_embeds = node_embeddings[src_type_str][masked_src_nodes_for_loss]
            dst_embeds = node_embeddings[dst_type_str][masked_dst_nodes_for_loss]
            
            try:
                rel_idx = self.rel_names.index(rel_name_str)
            except ValueError:
                logger.error(f"Relation '{rel_name_str}' ({canonical_etype}) not in self.rel_names. Skipping edge loss.")
                continue
            
            true_rels_for_loss = torch.full((masked_src_nodes_for_loss.size(0),), rel_idx, dtype=torch.long, device=target_device)
            edge_logits = self.decode_edges(src_embeds, dst_embeds) # Should be on target_device if inputs are
            
            if self.training or (not self.training and return_val_details):
                current_edge_loss = self.compute_edge_loss(edge_logits, true_rels_for_loss) # Pass target_device if needed by compute_edge_loss
                if self.training:
                    edge_recon_loss_sum = edge_recon_loss_sum + current_edge_loss 
                num_canonical_edge_types_for_loss += 1

                if not self.training and return_val_details:
                    preds = torch.argmax(edge_logits, dim=1)
                    correct_preds = (preds == true_rels_for_loss).sum().item()
                    eval_edge_metrics_accumulated['total_correct'] += correct_preds
                    eval_edge_metrics_accumulated['total_processed'] += true_rels_for_loss.size(0)
        
        edge_recon_loss = torch.tensor(0.0, device=target_device) # Default for non-training or if no edges processed
        if self.training and num_canonical_edge_types_for_loss > 0:
            edge_recon_loss = edge_recon_loss_sum / num_canonical_edge_types_for_loss
        elif self.training: # No edges processed for loss, but ensure it has grad if weight > 0
            edge_recon_loss = torch.tensor(0.0, device=target_device, requires_grad=True if self.pretrain_edge_loss_weight > 0 else False)
        
        # 6. Compute total loss (weighted sum)
        total_loss = torch.tensor(0.0, device=target_device)
        if self.pretrain_feat_loss_weight > 0 and feat_recon_loss is not None:
            total_loss = total_loss + self.pretrain_feat_loss_weight * feat_recon_loss
        if self.pretrain_edge_loss_weight > 0 and edge_recon_loss is not None: # Check if edge_recon_loss requires grad if it's training
            if self.training and not edge_recon_loss.requires_grad and self.pretrain_edge_loss_weight > 0 and num_canonical_edge_types_for_loss > 0:
                 # This case should ideally not happen if edge_recon_loss_sum had requires_grad=True
                 # and was part of the computation. Re-evaluate if this warning appears.
                 logger.warning("Edge reconstruction loss does not require grad during training, but weight > 0.")
            total_loss = total_loss + self.pretrain_edge_loss_weight * edge_recon_loss
        
        output_dict = {
            'feature_loss': feat_recon_loss, # For training loop, direct access
            'edge_loss': edge_recon_loss,     # For training loop, direct access
        }
        if self.training:
            output_dict['total_loss'] = total_loss # Only really needed for training backward pass
        
        # For validation/evaluation detailed output when requested
        if not self.training and return_val_details:
            output_dict['total_loss_val'] = total_loss # Store the calculated total loss for validation
            if eval_feature_details:
                output_dict['feature_loss_details'] = eval_feature_details
            if eval_edge_metrics_accumulated['total_processed'] > 0:
                eval_edge_metrics_accumulated['accuracy'] = eval_edge_metrics_accumulated['total_correct'] / eval_edge_metrics_accumulated['total_processed']
            output_dict['edge_eval_metrics'] = eval_edge_metrics_accumulated

        return output_dict


def pretrain_hgmae(model, g, feats_dict, optimizer, epochs=100, device='cuda', patience=10, scheduler=None, gradient_clip_norm=1.0, early_stopping_metric="total_loss"):
    """
    Pre-train the HGMAE model using node feature and structure masking.
    
    Args:
        model: The HGMAE model instance.
        g: The DGL graph.
        feats_dict: Dictionary of input features.
        optimizer: The PyTorch optimizer.
        epochs: Number of training epochs.
        device: Device to train on ('cuda' or 'cpu').
        patience: Number of epochs to wait for improvement before early stopping.
        scheduler: Learning rate scheduler.
        gradient_clip_norm: Max norm for gradient clipping.
        early_stopping_metric: Metric to use for early stopping ('total_loss' or 'feature_mse').
    """
    logger.info(f"Starting HGMAE pre-training on device: {device}")
    logger.info(f"Training for {epochs} epochs with patience {patience}.")
    logger.info(f"Early stopping metric: {early_stopping_metric}") # Log the chosen metric
    if scheduler:
        logger.info(f"Using learning rate scheduler: {type(scheduler).__name__}")
    if gradient_clip_norm:
        logger.info(f"Using gradient clipping with norm: {gradient_clip_norm}")

    # Initialize GradScaler if on CUDA and device is indeed a CUDA device string
    # Convert device string to torch.device to check its type
    torch_device_obj = torch.device(device)
    scaler = GradScaler() if torch_device_obj.type == 'cuda' else None
    if scaler:
        logger.info("CUDA device detected. Using GradScaler for Automatic Mixed Precision.")
    else:
        logger.info(f"Device is {torch_device_obj.type}. GradScaler (AMP) will not be used.")

    # Ensure model and initial data are on the correct device
    try:
        model.to(torch_device_obj) # Use the torch.device object
        g = g.to(torch_device_obj)
        feats_dict = {k: v.to(torch_device_obj) for k, v in feats_dict.items()}
        logger.info(f"Moved model, graph, and features to {torch_device_obj}.")
    except Exception as e:
        logger.error(f"Error moving model/data to device {torch_device_obj}: {e}", exc_info=True)
        raise

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    actual_epochs_run = 0

    # History tracking
    history_total_loss = []
    history_feature_loss = []
    history_edge_loss = []
    history_lr = []
    history_val_total_loss = []
    history_val_feature_mse = []
    history_val_feature_cosine_similarity = []
    history_val_edge_accuracy = []

    epoch_times = []
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        optimizer.zero_grad()

        # Forward pass with autocast if scaler is enabled
        if scaler:
            with autocast(): # Applies to CUDA operations within this block
                outputs = model(g, feats_dict) 
                loss_feat = outputs.get('feature_loss')
                loss_edge = outputs.get('edge_loss')
                
                # Consolidate loss calculation
                if loss_feat is None and loss_edge is None:
                    logger.error("Both feature_loss and edge_loss are None from model output. Cannot compute total loss.")
                    total_loss = torch.tensor(0.0, device=torch_device_obj, requires_grad=True)
                elif loss_feat is None:
                    total_loss = model.pretrain_edge_loss_weight * loss_edge
                    loss_feat = torch.tensor(0.0, device=torch_device_obj) 
                elif loss_edge is None:
                    total_loss = model.pretrain_feat_loss_weight * loss_feat
                    loss_edge = torch.tensor(0.0, device=torch_device_obj)
                else:
                    total_loss = model.pretrain_feat_loss_weight * loss_feat + \
                                 model.pretrain_edge_loss_weight * loss_edge
        else: # No scaler (e.g., CPU training or other non-CUDA device)
            outputs = model(g, feats_dict)
            loss_feat = outputs.get('feature_loss')
            loss_edge = outputs.get('edge_loss')

            if loss_feat is None and loss_edge is None:
                logger.error("Both feature_loss and edge_loss are None from model output (non-AMP path).")
                total_loss = torch.tensor(0.0, device=torch_device_obj, requires_grad=True)
            elif loss_feat is None:
                total_loss = model.pretrain_edge_loss_weight * loss_edge
                loss_feat = torch.tensor(0.0, device=torch_device_obj)
            elif loss_edge is None:
                total_loss = model.pretrain_feat_loss_weight * loss_feat
                loss_edge = torch.tensor(0.0, device=torch_device_obj)
            else:
                total_loss = model.pretrain_feat_loss_weight * loss_feat + \
                             model.pretrain_edge_loss_weight * loss_edge
        
        # Backward pass and optimizer step
        if scaler: # AMP is active
            scaler.scale(total_loss).backward()
            if gradient_clip_norm:
                scaler.unscale_(optimizer) # Unscale gradients before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else: # Standard CPU/non-AMP path
            total_loss.backward()
            if gradient_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        if scheduler:
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step() # Step scheduler after optimizer.step()
        else:
            current_lr = optimizer.param_groups[0]['lr']

        # Record training losses
        history_total_loss.append(total_loss.item())
        history_feature_loss.append(loss_feat.item() if torch.is_tensor(loss_feat) else loss_feat) # Handle if it's already a float (e.g. from dummy tensor)
        history_edge_loss.append(loss_edge.item() if torch.is_tensor(loss_edge) else loss_edge)
        history_lr.append(current_lr)
        
        # --- Validation Phase ---
        model.eval()
        val_total_loss_epoch = 0
        val_feat_mse_epoch = 0
        val_feat_cos_sim_epoch = 0
        val_edge_acc_epoch = 0
        num_val_batches = 0 # Assuming validation might be batched in a more complex setup; here, one pass.

        with torch.no_grad():
            # Use autocast for validation forward pass if scaler is active (i.e., on CUDA)
            if scaler: # AMP is active
                with autocast():
                    val_outputs = model(g, feats_dict, return_val_details=True) 
            else: # No scaler
                val_outputs = model(g, feats_dict, return_val_details=True)

            # val_loss_feat_details is a list of dicts: [{'pred_masked': t, 'true_masked': t}, ...]
            # val_outputs['feature_loss'] is the already aggregated validation feature loss scalar tensor.
            val_feat_loss_scalar = val_outputs.get('feature_loss', torch.tensor(0.0, device=torch_device_obj)).item()
            
            list_of_feat_details = val_outputs.get('feature_loss_details', []) # This is eval_details_list
            val_edge_eval_metrics = val_outputs.get('edge_eval_metrics', {})
            
            current_val_total_loss = val_outputs.get('total_loss_val', 0.0) 
            if not torch.is_tensor(current_val_total_loss): 
                current_val_total_loss = torch.tensor(current_val_total_loss, device=torch_device_obj)

            val_total_loss_epoch += current_val_total_loss.item()
            
            # Calculate MSE and Cosine Similarity from details
            all_pred_masked = []
            all_true_masked = []
            per_type_mse_list = []
            per_type_cosine_sim_list = []

            if isinstance(list_of_feat_details, list):
                for detail_item in list_of_feat_details:
                    if isinstance(detail_item, dict) and 'pred_masked' in detail_item and 'true_masked' in detail_item:
                        pred_m = detail_item['pred_masked']
                        true_m = detail_item['true_masked']
                        
                        if pred_m.numel() > 0 and true_m.numel() > 0:
                            # Ensure they are float32 for metric calculation if not already
                            pred_m_float = pred_m.float()
                            true_m_float = true_m.float()

                            mse = F.mse_loss(pred_m_float, true_m_float, reduction='mean').item()
                            per_type_mse_list.append(mse)
                            
                            # Flatten for cosine similarity
                            cos_sim = F.cosine_similarity(pred_m_float.flatten(), true_m_float.flatten(), dim=0).item()
                            per_type_cosine_sim_list.append(cos_sim)
                            
                            all_pred_masked.append(pred_m_float.flatten())
                            all_true_masked.append(true_m_float.flatten())
            
            if per_type_mse_list:
                val_feat_mse_epoch = sum(per_type_mse_list) / len(per_type_mse_list)
            else:
                val_feat_mse_epoch = 0.0 # Default if no details or no elements
            
            if per_type_cosine_sim_list:
                val_feat_cos_sim_epoch = sum(per_type_cosine_sim_list) / len(per_type_cosine_sim_list)
            else:
                val_feat_cos_sim_epoch = 0.0 # Default

            # Alternative: Global MSE/Cosine Sim if preferred (can be noisy if element counts vary wildly)
            # if all_pred_masked and all_true_masked:
            #     global_pred_masked = torch.cat(all_pred_masked)
            #     global_true_masked = torch.cat(all_true_masked)
            #     if global_pred_masked.numel() > 0:
            #         val_feat_mse_epoch = F.mse_loss(global_pred_masked, global_true_masked, reduction='mean').item()
            #         val_feat_cos_sim_epoch = F.cosine_similarity(global_pred_masked, global_true_masked, dim=0).item()

            val_edge_acc_epoch += val_edge_eval_metrics.get('accuracy', 0.0) # Assuming edge_eval_metrics contains 'accuracy'
            num_val_batches = 1 # Since we do one pass

        avg_val_total_loss = val_total_loss_epoch / num_val_batches if num_val_batches > 0 else 0
        avg_val_feat_mse = val_feat_mse_epoch / num_val_batches if num_val_batches > 0 else 0
        avg_val_feat_cos_sim = val_feat_cos_sim_epoch / num_val_batches if num_val_batches > 0 else 0
        avg_val_edge_acc = val_edge_acc_epoch / num_val_batches if num_val_batches > 0 else 0

        history_val_total_loss.append(avg_val_total_loss)
        history_val_feature_mse.append(avg_val_feat_mse)
        history_val_feature_cosine_similarity.append(avg_val_feat_cos_sim)
        history_val_edge_accuracy.append(avg_val_edge_acc)
        
        epoch_duration = time.time() - epoch_start_time
        epoch_times.append(epoch_duration)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = epochs - (epoch + 1)
        est_remaining_time = remaining_epochs * avg_epoch_time
        
        # Progress Bar update (Simplified for direct logging)
        progress_message = (
            f"Epoch [{epoch+1}/{epochs}] - Train Total Loss: {total_loss.item():.4f} "
            f"(Feat: {history_feature_loss[-1]:.4f}, Edge: {history_edge_loss[-1]:.4f}) | "
            f"Valid Total Loss: {avg_val_total_loss:.4f} "
            f"(Feat MSE: {avg_val_feat_mse:.4f}, Feat CosSim: {avg_val_feat_cos_sim:.4f}, Edge Acc: {avg_val_edge_acc:.4f}) | "
            f"LR: {current_lr:.2e} | Time/Epoch: {epoch_duration:.2f}s | ETA: {est_remaining_time/60:.1f}min"
        )
        logger.info(progress_message)
        sys.stdout.flush() # Ensure logs are flushed immediately

        actual_epochs_run = epoch + 1

        # Determine the metric to use for early stopping
        current_metric_for_stopping = 0.0
        if early_stopping_metric == "feature_mse":
            current_metric_for_stopping = avg_val_feat_mse
            metric_name_log = "validation feature MSE"
        elif early_stopping_metric == "total_loss":
            current_metric_for_stopping = avg_val_total_loss
            metric_name_log = "validation total loss"
        else: # Default to total_loss if an invalid string is provided
            logger.warning(f"Invalid early_stopping_metric '{early_stopping_metric}'. Defaulting to 'total_loss'.")
            current_metric_for_stopping = avg_val_total_loss
            metric_name_log = "validation total loss (defaulted)"

        if current_metric_for_stopping < best_val_loss: # best_val_loss now tracks the chosen metric
            best_val_loss = current_metric_for_stopping
            try:
                best_model_state = deepcopy(model.state_dict())
                logger.info(f"Epoch {epoch+1}: New best {metric_name_log}: {best_val_loss:.4f}. Model state saved.")
            except Exception as e_deepcopy:
                logger.error(f"Could not deepcopy model state: {e_deepcopy}. Best model will not be saved correctly.")
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"Epoch {epoch+1}: {metric_name_log} did not improve ({current_metric_for_stopping:.4f} vs best {best_val_loss:.4f}). Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch+1} due to no improvement in {metric_name_log} for {patience} epochs.")
                break
    
    # After the loop, if a best model state was saved, load it back into the model
    if best_model_state:
        logger.info(f"Loading best model state from epoch with validation loss: {best_val_loss:.4f}")
        model.load_state_dict(best_model_state)
    else:
        logger.warning("No best model state was recorded (e.g., if training stopped on first epoch or due to error). Returning current model state.")

    logger.info(f"Pre-training finished after {actual_epochs_run} epochs. Best validation total loss: {best_val_loss:.4f}")
    
    return (model, actual_epochs_run, 
            history_total_loss, history_feature_loss, history_edge_loss, history_lr,
            history_val_total_loss, history_val_feature_mse, 
            history_val_feature_cosine_similarity, history_val_edge_accuracy)


# Helper function to generate a simple text-based progress bar
# This might not be needed if tqdm is used or direct logging is preferred.
# For now, keeping it as it was in the summary.
def get_bar(loss_val, max_loss_for_scale=2.0):
    scaled_loss = min(float(loss_val), max_loss_for_scale) / max_loss_for_scale # Ensure float
    bar_fill = int(scaled_loss * 50)
    return '█' * bar_fill + '·' * (50 - bar_fill) # Use block and dot


def main():
    # Example usage
    model = HGMAE(in_dim=128, hidden_dim=64, out_dim=64, rel_names=['rel1', 'rel2'], original_feat_dims={'node_type1': 128, 'node_type2': 128})
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 4]))
    feats_dict = {'node_type1': torch.randn(4, 128), 'node_type2': torch.randn(4, 128)}
    model, epochs_run, history_total_loss, history_feature_loss, history_edge_loss, history_lr, history_val_total_loss, history_val_feature_mse, history_val_feature_cosine_similarity, history_val_edge_accuracy = pretrain_hgmae(model, g, feats_dict, optimizer)
    print(f"Training completed in {epochs_run} epochs. Best validation total loss: {history_val_total_loss[-1]:.4f}")


if __name__ == "__main__":
    main()
 