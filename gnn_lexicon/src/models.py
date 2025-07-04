"""
GNN model zoo for Philippine Lexicon: R-GCN, GraphSAGE, GATv2.
Supports edge-type and node-type heterogeneity.
"""

from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, SAGEConv, GATv2Conv, to_hetero, HeteroConv, Linear

class HeteroGNN(nn.Module):
    """Base class for heterogeneous GNN models."""
    
    def __init__(self, metadata: Tuple[List[str], List[Tuple[str, str, str]]], 
                 hidden_channels: int, out_channels: int, num_layers: int = 2):
        super().__init__()
        self.metadata = metadata
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

class HeteroRGCN(HeteroGNN):
    """Heterogeneous Relational GCN."""
    
    def __init__(self, metadata: Tuple[List[str], List[Tuple[str, str, str]]], 
                 in_channels_dict: Dict[str, int], hidden_channels: int, 
                 out_channels: int, num_layers: int = 2):
        super().__init__(metadata, hidden_channels, out_channels, num_layers)
        
        self.convs = nn.ModuleList()
        
        # First layer
        conv_dict = {}
        for edge_type in metadata[1]:
            num_relations = 1  # Each edge type is its own relation
            conv_dict[edge_type] = RGCNConv(
                in_channels_dict[edge_type[0]], 
                hidden_channels, 
                num_relations
            )
        self.convs.append(HeteroConv(conv_dict))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            conv_dict = {}
            for edge_type in metadata[1]:
                conv_dict[edge_type] = RGCNConv(
                    hidden_channels, 
                    hidden_channels, 
                    1
                )
            self.convs.append(HeteroConv(conv_dict))
        
        # Output layer
        conv_dict = {}
        for edge_type in metadata[1]:
            conv_dict[edge_type] = RGCNConv(
                hidden_channels, 
                out_channels, 
                1
            )
        self.convs.append(HeteroConv(conv_dict))
        
        # Node type specific output projections
        self.lins = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lins[node_type] = Linear(out_channels, out_channels)
    
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        for i, conv in enumerate(self.convs[:-1]):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=0.1, training=self.training) for key, x in x_dict.items()}
        
        x_dict = self.convs[-1](x_dict, edge_index_dict)
        
        # Apply node-type specific output transformation
        out_dict = {}
        for node_type, x in x_dict.items():
            if node_type in self.lins:
                out_dict[node_type] = self.lins[node_type](x)
            else:
                out_dict[node_type] = x
        
        return out_dict

class HeteroGraphSAGE(HeteroGNN):
    """Heterogeneous GraphSAGE."""
    
    def __init__(self, metadata: Tuple[List[str], List[Tuple[str, str, str]]], 
                 in_channels_dict: Dict[str, int], hidden_channels: int, 
                 out_channels: int, num_layers: int = 2):
        super().__init__(metadata, hidden_channels, out_channels, num_layers)
        
        self.convs = nn.ModuleList()
        
        # First layer
        conv_dict = {}
        for edge_type in metadata[1]:
            conv_dict[edge_type] = SAGEConv(
                (in_channels_dict[edge_type[0]], in_channels_dict[edge_type[2]]),
                hidden_channels
            )
        self.convs.append(HeteroConv(conv_dict))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            conv_dict = {}
            for edge_type in metadata[1]:
                conv_dict[edge_type] = SAGEConv(hidden_channels, hidden_channels)
            self.convs.append(HeteroConv(conv_dict))
        
        # Output layer
        conv_dict = {}
        for edge_type in metadata[1]:
            conv_dict[edge_type] = SAGEConv(hidden_channels, out_channels)
        self.convs.append(HeteroConv(conv_dict))
    
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        for i, conv in enumerate(self.convs[:-1]):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=0.1, training=self.training) for key, x in x_dict.items()}
        
        x_dict = self.convs[-1](x_dict, edge_index_dict)
        
        return x_dict

class HeteroGATv2(HeteroGNN):
    """Heterogeneous GATv2 with attention export."""
    
    def __init__(self, metadata: Tuple[List[str], List[Tuple[str, str, str]]], 
                 in_channels_dict: Dict[str, int], hidden_channels: int, 
                 out_channels: int, heads: int = 4, num_layers: int = 2):
        super().__init__(metadata, hidden_channels, out_channels, num_layers)
        
        self.heads = heads
        self.convs = nn.ModuleList()
        self.attn_weights = {}
        
        # First layer
        conv_dict = {}
        for edge_type in metadata[1]:
            conv_dict[edge_type] = GATv2Conv(
                (in_channels_dict[edge_type[0]], in_channels_dict[edge_type[2]]),
                hidden_channels,
                heads=heads,
                concat=True,
                add_self_loops=False
            )
        self.convs.append(HeteroConv(conv_dict))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            conv_dict = {}
            for edge_type in metadata[1]:
                conv_dict[edge_type] = GATv2Conv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads=heads,
                    concat=True,
                    add_self_loops=False
                )
            self.convs.append(HeteroConv(conv_dict))
        
        # Output layer - use 64 for final output to match checkpoint
        conv_dict = {}
        for edge_type in metadata[1]:
            conv_dict[edge_type] = GATv2Conv(
                hidden_channels * heads,
                64,  # Fixed to 64 to match checkpoint
                heads=1,
                concat=False,
                add_self_loops=False
            )
        self.convs.append(HeteroConv(conv_dict))
    
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        self.attn_weights = {}
        
        for i, conv in enumerate(self.convs[:-1]):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=0.1, training=self.training) for key, x in x_dict.items()}
            
            # Store attention weights
            for edge_type, module in conv.convs.items():
                if hasattr(module, 'alpha') and module.alpha is not None:
                    self.attn_weights[f"layer_{i}_{edge_type}"] = module.alpha.detach().cpu()
        
        x_dict = self.convs[-1](x_dict, edge_index_dict)
        
        # Store final layer attention
        for edge_type, module in self.convs[-1].convs.items():
            if hasattr(module, 'alpha') and module.alpha is not None:
                self.attn_weights[f"layer_{len(self.convs)-1}_{edge_type}"] = module.alpha.detach().cpu()
        
        return x_dict

class LinkPredictor(nn.Module):
    """Link prediction decoder."""
    
    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__()
        self.lin1 = nn.Linear(2 * in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)
    
    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """
        Predict link probability between node pairs.
        Args:
            x_i: Source node embeddings [batch_size, in_channels]
            x_j: Target node embeddings [batch_size, in_channels]
        Returns:
            Link probabilities [batch_size]
        """
        x = torch.cat([x_i, x_j], dim=-1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin2(x).squeeze(-1)
        return x

class RelationClassifier(nn.Module):
    """Relation type classifier."""
    
    def __init__(self, in_channels: int, num_relations: int, hidden_channels: int = 64):
        super().__init__()
        self.lin1 = nn.Linear(2 * in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, num_relations)
    
    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """
        Classify relation type between node pairs.
        Args:
            x_i: Source node embeddings [batch_size, in_channels]
            x_j: Target node embeddings [batch_size, in_channels]
        Returns:
            Relation type logits [batch_size, num_relations]
        """
        x = torch.cat([x_i, x_j], dim=-1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin2(x)
        return x

def create_model(model_type: str, metadata: Tuple[List[str], List[Tuple[str, str, str]]], 
                 in_channels_dict: Dict[str, int], config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create models.
    Args:
        model_type: One of 'rgcn', 'sage', 'gatv2'
        metadata: Graph metadata (node types, edge types)
        in_channels_dict: Input channels per node type
        config: Model configuration
    """
    node_types, edge_types = metadata
    hidden_dim = config.get("hidden_dim", 128)
    out_dim = config.get("out_dim", 64)
    num_layers = config.get("num_layers", 2)
    heads = config.get("heads", 4)
    
    # Filter edge types to only include those where both source and target nodes exist
    available_node_types = set(in_channels_dict.keys())
    filtered_edge_types = [
        edge_type for edge_type in edge_types 
        if edge_type[0] in available_node_types and edge_type[2] in available_node_types
    ]
    
    # Create filtered metadata
    filtered_metadata = (node_types, filtered_edge_types)
    
    if model_type == "rgcn":
        return HeteroRGCN(filtered_metadata, in_channels_dict, hidden_dim, out_dim, num_layers)
    elif model_type == "sage":
        return HeteroGraphSAGE(filtered_metadata, in_channels_dict, hidden_dim, out_dim, num_layers)
    elif model_type == "gatv2":
        return HeteroGATv2(filtered_metadata, in_channels_dict, hidden_dim, out_dim, heads, num_layers)
    else:
        raise ValueError(f"Unknown model type: {model_type}") 