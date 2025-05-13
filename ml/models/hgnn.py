"""
Heterogeneous Graph Neural Network architecture for lexical knowledge graphs.

This module implements a heterogeneous GNN with:
1. R-GCN-style local message passing
2. Exphormer-style global attention
3. Multi-task output heads for link prediction and node classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import dgl.function as fn
from typing import Dict, List, Tuple, Optional, Union, Any
import math
import logging

logger = logging.getLogger(__name__)

class RelationalGraphConv(nn.Module):
    """
    Relational Graph Convolutional layer for heterogeneous graphs.
    Based on R-GCN but with additional features for heterogeneous graphs.
    """
    
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int, 
                 rel_names: List[str],
                 num_bases: int = 8,
                 activation: Optional[nn.Module] = None,
                 self_loop: bool = True,
                 dropout: float = 0.0,
                 layer_norm: bool = True):
        """
        Initialize the RelationalGraphConv layer.
        
        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension
            rel_names: List of relation types in the graph
            num_bases: Number of bases for weight decomposition
            activation: Activation function to use
            self_loop: Whether to include self-loops
            dropout: Dropout probability
            layer_norm: Whether to apply layer normalization
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rel_names = rel_names
        self.num_bases = min(num_bases, len(rel_names))
        self.activation = activation
        self.self_loop = self_loop
        
        # Create relation-specific weights using basis decomposition
        self.weight_bases = nn.Parameter(torch.Tensor(self.num_bases, in_dim, out_dim))
        self.weight_coefficients = nn.Parameter(torch.Tensor(len(rel_names), self.num_bases))
        
        # Self-loop weight (if needed)
        if self.self_loop:
            self.self_weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        
        # Additional components
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim) if layer_norm else None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize learnable parameters."""
        nn.init.xavier_uniform_(self.weight_bases)
        nn.init.xavier_uniform_(self.weight_coefficients)
        if self.self_loop:
            nn.init.xavier_uniform_(self.self_weight)
    
    def forward(self, g, feat: torch.Tensor) -> torch.Tensor:
        """
        Forward computation on the heterogeneous graph.
        
        Args:
            g: DGL heterogeneous graph
            feat: Node feature tensor
            
        Returns:
            Updated node features
        """
        # Get device for computation
        device = feat.device
        
        # Initialize output features
        out_feat = torch.zeros(feat.shape[0], self.out_dim, device=device)
        
        # Compute relation-specific weights from bases
        rel_weights = torch.einsum('rb, bio -> rio', self.weight_coefficients, self.weight_bases)
        
        # Process each relation type
        for rel_idx, rel in enumerate(self.rel_names):
            # Get relation-specific weight
            rel_weight = rel_weights[rel_idx]
            
            # Message passing for this relation
            if rel in g.etypes:
                # Transform node features
                rel_feat = torch.matmul(feat, rel_weight)
                
                # Create a homogeneous subgraph for this relation
                rel_graph = dgl.edge_type_subgraph(g, [rel])
                
                # Aggregate messages from neighbors
                rel_graph.ndata['h'] = rel_feat
                rel_graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_agg'))
                
                # Add to output features
                out_feat = out_feat + rel_graph.ndata['h_agg']
        
        # Apply self-loop
        if self.self_loop:
            out_feat = out_feat + torch.matmul(feat, self.self_weight)
        
        # Apply layer normalization if enabled
        if self.layer_norm is not None:
            out_feat = self.layer_norm(out_feat)
        
        # Apply dropout and activation
        out_feat = self.dropout(out_feat)
        if self.activation is not None:
            out_feat = self.activation(out_feat)
        
        return out_feat

class SparseMultiHeadAttention(nn.Module):
    """
    Sparse Multi-Head Attention layer inspired by Exphormer.
    Implements efficient attention using sparse patterns.
    """
    
    def __init__(self, 
                 hidden_dim: int, 
                 num_heads: int = 8, 
                 dropout: float = 0.1,
                 sparsity: float = 0.9):
        """
        Initialize the Sparse Multi-Head Attention layer.
        
        Args:
            hidden_dim: Hidden feature dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            sparsity: Sparsity factor (0.0 = dense, 1.0 = no connections)
        """
        super().__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.sparsity = sparsity
        
        # Projection layers
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Create a virtual node for global information exchange
        self.virtual_token = nn.Parameter(torch.Tensor(1, hidden_dim))
        nn.init.normal_(self.virtual_token, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass computing sparse multi-head attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Append virtual token to the sequence
        virtual_tokens = self.virtual_token.expand(batch_size, 1, self.hidden_dim)
        x_with_virtual = torch.cat([x, virtual_tokens], dim=1)
        
        # Project queries, keys, and values
        q = self.q_proj(x_with_virtual).view(batch_size, seq_len + 1, self.num_heads, self.head_dim)
        k = self.k_proj(x_with_virtual).view(batch_size, seq_len + 1, self.num_heads, self.head_dim)
        v = self.v_proj(x_with_virtual).view(batch_size, seq_len + 1, self.num_heads, self.head_dim)
        
        # Reshape for multi-head attention
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len+1, head_dim]
        k = k.transpose(1, 2)  # [batch_size, num_heads, seq_len+1, head_dim]
        v = v.transpose(1, 2)  # [batch_size, num_heads, seq_len+1, head_dim]
        
        # Generate sparse attention pattern
        if self.sparsity > 0:
            # Every node attends to the virtual token
            virtual_mask = torch.zeros(batch_size, self.num_heads, seq_len + 1, 1, device=device)
            
            # Virtual token attends to all nodes
            global_mask = torch.zeros(batch_size, self.num_heads, 1, seq_len + 1, device=device)
            
            # Generate random sparse connections for regular nodes
            # This is a simplified approximation of expander graph-based sparsity
            sparse_mask = torch.rand(batch_size, self.num_heads, seq_len, seq_len, device=device)
            sparse_mask = sparse_mask > self.sparsity  # Keep only (1-sparsity) fraction of connections
            
            # Ensure each node has at least some connections (prevent disconnected nodes)
            min_connections = max(2, int((1 - self.sparsity) * seq_len * 0.5))
            for i in range(seq_len):
                if torch.sum(sparse_mask[:, :, i, :]) < min_connections:
                    # Select random connections
                    rand_indices = torch.randperm(seq_len)[:min_connections]
                    sparse_mask[:, :, i, rand_indices] = True
            
            # Combine masks
            row_indices = torch.arange(seq_len, device=device).view(-1, 1).expand(-1, 1)
            col_indices = torch.arange(1, device=device).view(1, -1).expand(seq_len, -1) + seq_len
            
            # Full attention mask
            full_mask = torch.zeros(batch_size, self.num_heads, seq_len + 1, seq_len + 1, device=device)
            full_mask[:, :, :seq_len, :seq_len] = sparse_mask.float()
            full_mask[:, :, :seq_len, seq_len:] = virtual_mask[:, :, :seq_len, :].float()
            full_mask[:, :, seq_len:, :] = global_mask[:, :, :, :].float()
            
            # Apply the attention mask
            attention_mask = full_mask
        else:
            # Dense attention (no sparsity)
            attention_mask = None
        
        # Compute scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(2, 3)) / scale
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        if mask is not None:
            # Expand mask for broadcasting
            expanded_mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(expanded_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights
        context = torch.matmul(attn_weights, v)
        
        # Reshape and combine heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len + 1, self.hidden_dim)
        
        # Extract only the original sequence (without virtual token)
        context = context[:, :seq_len, :]
        
        # Final projection
        output = self.o_proj(context)
        
        return output

class GraphGPSLayer(nn.Module):
    """
    Graph GPS-style layer combining local message passing with global attention.
    """
    
    def __init__(self, 
                 hidden_dim: int,
                 rel_names: List[str],
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 num_bases: int = 8,
                 sparsity: float = 0.9,
                 layer_norm: bool = True):
        """
        Initialize the GraphGPS layer.
        
        Args:
            hidden_dim: Hidden feature dimension
            rel_names: List of relation types in the graph
            num_heads: Number of attention heads
            dropout: Dropout probability
            num_bases: Number of bases for weight decomposition
            sparsity: Sparsity factor for attention
            layer_norm: Whether to apply layer normalization
        """
        super().__init__()
        
        # Local message passing
        self.local_mp = RelationalGraphConv(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            rel_names=rel_names,
            num_bases=num_bases,
            activation=None,
            self_loop=True,
            dropout=dropout,
            layer_norm=False
        )
        
        # Global attention
        self.global_attn = SparseMultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            sparsity=sparsity
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        if layer_norm:
            self.layer_norm1 = nn.LayerNorm(hidden_dim)
            self.layer_norm2 = nn.LayerNorm(hidden_dim)
            self.layer_norm3 = nn.LayerNorm(hidden_dim)
        else:
            self.layer_norm1 = None
            self.layer_norm2 = None
            self.layer_norm3 = None
        
        # Gating mechanism to combine local and global features
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, g, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the GraphGPS layer.
        
        Args:
            g: DGL heterogeneous graph
            x: Node feature tensor
            mask: Optional attention mask
            
        Returns:
            Updated node features
        """
        # Residual connection for local message passing
        local_out = self.local_mp(g, x)
        if self.layer_norm1 is not None:
            local_out = self.layer_norm1(local_out)
        
        # Residual connection for global attention
        global_out = self.global_attn(x.unsqueeze(0), mask)
        global_out = global_out.squeeze(0)  # Remove batch dimension
        if self.layer_norm2 is not None:
            global_out = self.layer_norm2(global_out)
        
        # Combine local and global features using a gating mechanism
        gate_input = torch.cat([local_out, global_out], dim=-1)
        gate_weight = torch.sigmoid(self.gate(gate_input))
        combined = gate_weight * local_out + (1 - gate_weight) * global_out
        
        # Feed-forward network with residual connection
        ffn_out = self.ffn(combined) + combined
        if self.layer_norm3 is not None:
            ffn_out = self.layer_norm3(ffn_out)
        
        return ffn_out

class HeterogeneousGNN(nn.Module):
    """
    Heterogeneous Graph Neural Network architecture for lexical knowledge graphs.
    
    Combines R-GCN message passing with Exphormer-style global attention
    using GraphGPS principles.
    """
    
    def __init__(self, 
                 in_dim: int, 
                 hidden_dim: int,
                 out_dim: int,
                 rel_names: List[str],
                 num_layers: int = 3,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 residual: bool = True,
                 layer_norm: bool = True,
                 num_bases: int = 8,
                 sparsity: float = 0.9):
        """
        Initialize the HeterogeneousGNN model.
        
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden feature dimension
            out_dim: Output feature dimension
            rel_names: List of relation types in the graph
            num_layers: Number of GraphGPS layers
            num_heads: Number of attention heads per layer
            dropout: Dropout probability
            residual: Whether to use residual connections
            layer_norm: Whether to apply layer normalization
            num_bases: Number of bases for weight decomposition
            sparsity: Sparsity factor for global attention
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.residual = residual
        
        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # GraphGPS layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GraphGPSLayer(
                    hidden_dim=hidden_dim,
                    rel_names=rel_names,
                    num_heads=num_heads,
                    dropout=dropout,
                    num_bases=num_bases,
                    sparsity=sparsity,
                    layer_norm=layer_norm
                )
            )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, g, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the HeterogeneousGNN model.
        
        Args:
            g: DGL heterogeneous graph
            feats: Dictionary of node features
            
        Returns:
            Node embeddings
        """
        # Concatenate and project input features
        if isinstance(feats, dict):
            x = torch.cat([feats[k] for k in sorted(feats.keys())], dim=1)
        else:
            x = feats
        
        h = self.input_proj(x)
        
        # Apply GraphGPS layers
        for i, layer in enumerate(self.layers):
            h_new = layer(g, h)
            if self.residual and i > 0:  # No residual for the first layer
                h = h + h_new
            else:
                h = h_new
        
        # Final output projection
        out = self.dropout(self.output_proj(h))
        
        return out
    
    def encode(self, g, feats: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the graph into node embeddings.
        
        Args:
            g: DGL heterogeneous graph
            feats: Dictionary of node features
            
        Returns:
            Tuple of hidden representations and output embeddings
        """
        # Same as forward but return both hidden and output representations
        if isinstance(feats, dict):
            x = torch.cat([feats[k] for k in sorted(feats.keys())], dim=1)
        else:
            x = feats
        
        h = self.input_proj(x)
        
        # Apply GraphGPS layers
        for i, layer in enumerate(self.layers):
            h_new = layer(g, h)
            if self.residual and i > 0:  # No residual for the first layer
                h = h + h_new
            else:
                h = h_new
        
        # Final output projection
        out = self.dropout(self.output_proj(h))
        
        return h, out 