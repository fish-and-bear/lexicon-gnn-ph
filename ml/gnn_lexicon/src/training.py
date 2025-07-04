"""
Training loop for Philippine Lexicon GNN.
Supports neighbor sampling, mixed precision, early stopping, and ablation.
"""

from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import HeteroData
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from .evaluation import evaluate_link_prediction, evaluate_hits_at_k
from .models import LinkPredictor, RelationClassifier

def negative_sampling(edge_index: torch.Tensor, num_nodes: int, num_neg_samples: int = 1) -> torch.Tensor:
    """
    Generate negative edges for link prediction.
    Args:
        edge_index: Positive edges [2, num_edges]
        num_nodes: Total number of nodes
        num_neg_samples: Number of negative samples per positive edge
    Returns:
        Negative edge indices [2, num_edges * num_neg_samples]
    """
    num_pos_edges = edge_index.size(1)
    num_neg_edges = num_pos_edges * num_neg_samples
    
    # Create set of existing edges for fast lookup
    edge_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    
    neg_edges = []
    while len(neg_edges) < num_neg_edges:
        src = torch.randint(0, num_nodes, (num_neg_edges - len(neg_edges),))
        dst = torch.randint(0, num_nodes, (num_neg_edges - len(neg_edges),))
        
        for s, d in zip(src.tolist(), dst.tolist()):
            if s != d and (s, d) not in edge_set:
                neg_edges.append([s, d])
                if len(neg_edges) >= num_neg_edges:
                    break
    
    return torch.tensor(neg_edges[:num_neg_edges], dtype=torch.long).t()

def train_epoch(model: nn.Module, 
                data: HeteroData,
                optimizer: torch.optim.Optimizer,
                link_predictor: LinkPredictor,
                device: torch.device,
                config: Dict[str, Any],
                scaler: Optional[GradScaler] = None,
                edge_type_to_train: str = "RELATED") -> float:
    """
    Train one epoch.
    Args:
        model: GNN model
        data: Training data
        optimizer: Optimizer
        link_predictor: Link prediction head
        device: Device
        config: Training configuration
        scaler: GradScaler for mixed precision
        edge_type_to_train: Which edge type to train on
    Returns:
        Average loss
    """
    model.train()
    link_predictor.train()
    
    total_loss = 0.0
    num_batches = 0
    
    # Create neighbor loader
    loader = NeighborLoader(
        data,
        num_neighbors=[10] * config["num_layers"],
        batch_size=config["batch_size"],
        input_nodes=("Word", None),
        shuffle=True
    )
    
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        with autocast(enabled=scaler is not None):
            # Forward pass through GNN
            out_dict = model(batch.x_dict, batch.edge_index_dict)
            
            # Get positive edges for the target edge type
            edge_key = ("Word", edge_type_to_train, "Word")
            if edge_key in batch.edge_index_dict:
                pos_edge_index = batch.edge_index_dict[edge_key]
                
                if pos_edge_index.size(1) > 0:
                    # Generate negative edges
                    num_word_nodes = out_dict["Word"].size(0)
                    neg_edge_index = negative_sampling(pos_edge_index, num_word_nodes, num_neg_samples=1)
                    
                    # Get embeddings for positive edges
                    pos_src = out_dict["Word"][pos_edge_index[0]]
                    pos_dst = out_dict["Word"][pos_edge_index[1]]
                    pos_pred = link_predictor(pos_src, pos_dst)
                    
                    # Get embeddings for negative edges
                    neg_src = out_dict["Word"][neg_edge_index[0]]
                    neg_dst = out_dict["Word"][neg_edge_index[1]]
                    neg_pred = link_predictor(neg_src, neg_dst)
                    
                    # Compute loss
                    pos_loss = F.binary_cross_entropy_with_logits(
                        pos_pred, torch.ones_like(pos_pred)
                    )
                    neg_loss = F.binary_cross_entropy_with_logits(
                        neg_pred, torch.zeros_like(neg_pred)
                    )
                    loss = pos_loss + neg_loss
                else:
                    continue
            else:
                continue
        
        if scaler is not None:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip", 1.0))
            torch.nn.utils.clip_grad_norm_(link_predictor.parameters(), config.get("grad_clip", 1.0))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip", 1.0))
            torch.nn.utils.clip_grad_norm_(link_predictor.parameters(), config.get("grad_clip", 1.0))
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)

def train_gnn(
    model: nn.Module,
    train_data: HeteroData,
    val_data: HeteroData,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: Dict[str, Any],
    amp: bool = False,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Full training loop with validation and early stopping.
    Args:
        model: GNN model
        train_data: Training data
        val_data: Validation data
        optimizer: Optimizer
        device: Device
        config: Training configuration
        amp: Use automatic mixed precision
        save_path: Path to save best model
    Returns:
        Training history and best model state
    """
    scaler = GradScaler() if amp else None
    link_predictor = LinkPredictor(config["out_dim"]).to(device)
    
    # Add link predictor parameters to optimizer
    optimizer.add_param_group({'params': link_predictor.parameters()})
    
    best_val_auc = 0.0
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_auc": [], "val_hits": []}
    
    for epoch in range(config["epochs"]):
        # Training
        train_loss = train_epoch(
            model, train_data, optimizer, link_predictor, 
            device, config, scaler
        )
        
        # Validation
        val_auc = evaluate_link_prediction(
            model, link_predictor, val_data, device
        )
        val_hits = evaluate_hits_at_k(
            model, link_predictor, val_data, device, k=10
        )
        
        history["train_loss"].append(train_loss)
        history["val_auc"].append(val_auc)
        history["val_hits"].append(val_hits)
        
        print(f"Epoch {epoch+1}/{config['epochs']}: "
              f"Loss={train_loss:.4f}, Val AUC={val_auc:.4f}, Val Hits@10={val_hits:.4f}")
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {
                'model': model.state_dict(),
                'link_predictor': link_predictor.state_dict(),
                'epoch': epoch,
                'val_auc': val_auc
            }
            patience_counter = 0
            
            if save_path:
                torch.save(best_state, save_path)
                print(f"Model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.get("early_stopping_patience", 10):
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return {
        "history": history,
        "best_state": best_state,
        "best_val_auc": best_val_auc
    }

def run_ablation(
    model_class: type,
    base_data: HeteroData,
    config: Dict[str, Any],
    device: torch.device,
    edge_types_to_ablate: List[Tuple[str, str, str]]
) -> Dict[str, Dict[str, float]]:
    """
    Run ablation study by removing different edge types.
    Args:
        model_class: Model class to use
        base_data: Full graph data
        config: Training configuration
        device: Device
        edge_types_to_ablate: List of edge types to ablate
    Returns:
        Results for each ablation
    """
    from .graph_builder import split_edges
    
    results = {}
    
    # Baseline with all edges
    print("Training baseline model with all edges...")
    train_data, val_data, test_data = split_edges(base_data)
    
    # Get metadata and input channels
    metadata = (base_data.node_types, base_data.edge_types)
    in_channels_dict = {
        node_type: base_data[node_type].x.size(1) 
        for node_type in base_data.node_types
    }
    
    # Train baseline
    model = model_class(metadata, in_channels_dict, config["hidden_dim"], 
                       config["out_dim"], config["num_layers"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    
    train_results = train_gnn(
        model, train_data, val_data, optimizer, device, 
        {**config, "epochs": 20}  # Fewer epochs for ablation
    )
    
    results["baseline"] = {
        "val_auc": train_results["best_val_auc"],
        "history": train_results["history"]
    }
    
    # Ablate each edge type
    for edge_type in edge_types_to_ablate:
        print(f"\nAblating edge type: {edge_type}")
        
        # Create data without this edge type
        ablated_data = HeteroData()
        
        # Copy nodes
        for node_type in base_data.node_types:
            for key, value in base_data[node_type].items():
                ablated_data[node_type][key] = value
        
        # Copy edges except ablated type
        for et in base_data.edge_types:
            if et != edge_type:
                for key, value in base_data[et].items():
                    ablated_data[et][key] = value
        
        # Split and train
        train_data, val_data, test_data = split_edges(ablated_data)
        
        model = model_class(metadata, in_channels_dict, config["hidden_dim"], 
                           config["out_dim"], config["num_layers"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
        
        train_results = train_gnn(
            model, train_data, val_data, optimizer, device, 
            {**config, "epochs": 20}
        )
        
        results[f"without_{edge_type}"] = {
            "val_auc": train_results["best_val_auc"],
            "history": train_results["history"]
        }
    
    return results 