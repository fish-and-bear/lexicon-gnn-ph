"""
Evaluation metrics for Philippine Lexicon GNN.
Includes ROC-AUC, Hits@k, and ablation toggles.
"""

from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from torch_geometric.data import HeteroData
from tqdm import tqdm

def evaluate_link_prediction(
    model: nn.Module,
    link_predictor: nn.Module,
    data: HeteroData,
    device: torch.device,
    edge_type: str = "shares_phon"
) -> float:
    """
    Evaluates link prediction ROC-AUC on the validation/test set.
    Args:
        model: GNN model
        link_predictor: Link prediction head
        data: Evaluation data
        device: Device
        edge_type: Which edge type to evaluate
    Returns:
        ROC-AUC score
    """
    model.eval()
    link_predictor.eval()
    
    with torch.no_grad():
        # Move data to device
        data = data.to(device)
        
        # Forward pass through GNN
        out_dict = model(data.x_dict, data.edge_index_dict)
        
        # Try different edge types that might exist
        edge_types_to_try = [
            ("Word", "SHARES_PHONOLOGY", "Word"),
            ("Word", "DERIVED_FROM", "Word"),
            ("Word", "HAS_FORM", "Form"),
            ("Word", "HAS_SENSE", "Sense")
        ]
        
        for edge_key in edge_types_to_try:
            if edge_key in data.edge_index_dict:
                pos_edge_index = data.edge_index_dict[edge_key]
                
                if pos_edge_index.size(1) > 0:
                    # Get the number of Word nodes in this batch
                    num_word_nodes = out_dict["Word"].size(0)
                    
                    # Filter edges to only include those where both nodes are in the batch
                    valid_mask = (pos_edge_index[0] < num_word_nodes) & (pos_edge_index[1] < num_word_nodes)
                    valid_edges = pos_edge_index[:, valid_mask]
                    
                    if valid_edges.size(1) == 0:
                        continue
                    
                    # Generate negative edges
                    num_pos_edges = valid_edges.size(1)
                    
                    # Simple negative sampling
                    neg_edges = []
                    edge_set = set(zip(valid_edges[0].tolist(), valid_edges[1].tolist()))
                    
                    attempts = 0
                    while len(neg_edges) < num_pos_edges and attempts < num_pos_edges * 10:
                        src = torch.randint(0, num_word_nodes, (num_pos_edges,))
                        dst = torch.randint(0, num_word_nodes, (num_pos_edges,))
                        
                        for s, d in zip(src.tolist(), dst.tolist()):
                            if s != d and (s, d) not in edge_set:
                                neg_edges.append([s, d])
                                if len(neg_edges) >= num_pos_edges:
                                    break
                        attempts += num_pos_edges
                    
                    if len(neg_edges) < num_pos_edges:
                        print(f"[evaluation] Could only generate {len(neg_edges)} negative edges")
                        neg_edges.extend([[0, 1]] * (num_pos_edges - len(neg_edges)))
                    
                    neg_edge_index = torch.tensor(neg_edges[:num_pos_edges], dtype=torch.long).t().to(device)
                    
                    # Predict scores
                    pos_src = out_dict["Word"][valid_edges[0]]
                    pos_dst = out_dict["Word"][valid_edges[1]]
                    pos_scores = torch.sigmoid(link_predictor(pos_src, pos_dst)).cpu().numpy()
                    
                    neg_src = out_dict["Word"][neg_edge_index[0]]
                    neg_dst = out_dict["Word"][neg_edge_index[1]]
                    neg_scores = torch.sigmoid(link_predictor(neg_src, neg_dst)).cpu().numpy()
                    
                    # Compute AUC
                    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
                    y_scores = np.concatenate([pos_scores, neg_scores])
                    
                    try:
                        auc = roc_auc_score(y_true, y_scores)
                        print(f"[evaluation] Using edge type {edge_key} with {valid_edges.size(1)} valid edges, AUC: {auc:.4f}")
                        return auc
                    except ValueError:
                        continue
        
        print(f"[evaluation] No valid edge types found for evaluation")
        return 0.5

def evaluate_hits_at_k(
    model: nn.Module,
    link_predictor: nn.Module,
    data: HeteroData,
    device: torch.device,
    k: int = 10,
    edge_type: str = "shares_phon"
) -> float:
    """
    Computes Hits@k for link prediction.
    Args:
        model: GNN model
        link_predictor: Link prediction head
        data: Evaluation data
        device: Device
        k: Top-k value
        edge_type: Which edge type to evaluate
    Returns:
        Hits@k score
    """
    model.eval()
    link_predictor.eval()
    
    with torch.no_grad():
        data = data.to(device)
        out_dict = model(data.x_dict, data.edge_index_dict)
        
        # Try different edge types that might exist
        edge_types_to_try = [
            ("Word", "shares_phon", "Word"),
            ("Word", "shares_etym", "Word"),
            ("Word", "HAS_FORM", "Form"),
            ("Word", "HAS_SENSE", "Sense")
        ]
        
        for edge_key in edge_types_to_try:
            if edge_key in data.edge_index_dict:
                pos_edge_index = data.edge_index_dict[edge_key]
                if pos_edge_index.size(1) > 0:
                    word_embeddings = out_dict["Word"]
                    num_nodes = word_embeddings.size(0)
                    
                    hits = 0
                    total = 0
                    
                    # Evaluate in batches to avoid memory issues
                    batch_size = min(100, pos_edge_index.size(1))
                    
                    for i in range(0, pos_edge_index.size(1), batch_size):
                        batch_edges = pos_edge_index[:, i:i+batch_size]
                        batch_size_actual = batch_edges.size(1)
                        
                        for j in range(batch_size_actual):
                            src = batch_edges[0, j]
                            true_dst = batch_edges[1, j]
                            
                            # Compute scores for all possible destinations
                            src_emb = word_embeddings[src].unsqueeze(0).expand(num_nodes, -1)
                            all_dst_emb = word_embeddings
                            
                            scores = link_predictor(src_emb, all_dst_emb)
                            scores[src] = float('-inf')  # Exclude self-loops
                            
                            # Get top-k predictions
                            _, topk_indices = torch.topk(scores, k)
                            
                            if true_dst in topk_indices:
                                hits += 1
                            total += 1
                    
                    print(f"[evaluation] Using edge type {edge_key} with {pos_edge_index.size(1)} edges, Hits@{k}: {hits/max(total, 1):.4f}")
                    return hits / max(total, 1)
        
        return 0.0

def evaluate_relation_classification(
    model: nn.Module,
    relation_classifier: nn.Module,
    data: HeteroData,
    device: torch.device,
    edge_type: str = "RELATED"
) -> Dict[str, float]:
    """
    Evaluates relation type classification.
    Args:
        model: GNN model
        relation_classifier: Relation classification head
        data: Evaluation data
        device: Device
        edge_type: Which edge type to evaluate
    Returns:
        Dictionary with accuracy and F1 scores
    """
    model.eval()
    relation_classifier.eval()
    
    with torch.no_grad():
        data = data.to(device)
        out_dict = model(data.x_dict, data.edge_index_dict)
        
        edge_key = ("Word", edge_type, "Word")
        if edge_key not in data.edge_index_dict:
            return {"accuracy": 0.0, "f1": 0.0}
        
        edge_index = data.edge_index_dict[edge_key]
        
        if not hasattr(data[edge_key], 'edge_type') or edge_index.size(1) == 0:
            return {"accuracy": 0.0, "f1": 0.0}
        
        edge_types = data[edge_key].edge_type
        
        # Get embeddings
        src_emb = out_dict["Word"][edge_index[0]]
        dst_emb = out_dict["Word"][edge_index[1]]
        
        # Predict relation types
        logits = relation_classifier(src_emb, dst_emb)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        true_labels = edge_types.cpu().numpy()
        
        # Compute metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
        
    return {"accuracy": accuracy, "f1": f1}

def evaluate_node_similarity(
    model: nn.Module,
    data: HeteroData,
    device: torch.device,
    word_pairs: List[Tuple[str, str]],
    word_to_idx: Dict[str, int]
) -> Dict[str, float]:
    """
    Evaluates semantic similarity between word pairs.
    Args:
        model: GNN model
        data: Graph data
        device: Device
        word_pairs: List of (word1, word2) tuples
        word_to_idx: Mapping from word to node index
    Returns:
        Dictionary mapping pair to similarity score
    """
    model.eval()
    
    with torch.no_grad():
        data = data.to(device)
        out_dict = model(data.x_dict, data.edge_index_dict)
        word_embeddings = out_dict["Word"]
        
        similarities = {}
        
        for word1, word2 in word_pairs:
            if word1 in word_to_idx and word2 in word_to_idx:
                idx1 = word_to_idx[word1]
                idx2 = word_to_idx[word2]
                
                emb1 = word_embeddings[idx1]
                emb2 = word_embeddings[idx2]
                
                # Cosine similarity
                sim = torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
                similarities[f"{word1}-{word2}"] = sim
            else:
                similarities[f"{word1}-{word2}"] = 0.0
    
    return similarities

def print_evaluation_summary(results: Dict[str, Any]) -> None:
    """
    Pretty print evaluation results.
    Args:
        results: Dictionary of evaluation metrics
    """
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric:.<30} {value:.4f}")
        elif isinstance(value, dict):
            print(f"\n{metric}:")
            for sub_metric, sub_value in value.items():
                print(f"  {sub_metric:.<28} {sub_value:.4f}")
    
    print("="*50) 