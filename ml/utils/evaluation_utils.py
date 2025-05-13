"""
Evaluation utilities for graph neural network models.

This module provides functions to evaluate link prediction and 
node classification performance of graph neural networks.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score
)
import logging

logger = logging.getLogger(__name__)

def evaluate_link_prediction(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    threshold: float = 0.5,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Evaluate link prediction performance.
    
    Args:
        pos_scores: Scores for positive edges
        neg_scores: Scores for negative edges
        threshold: Score threshold for classification
        k_values: Values of k for Hits@k metric
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Ensure inputs are numpy arrays
    pos_scores = np.array(pos_scores).flatten()
    neg_scores = np.array(neg_scores).flatten()
    
    # Apply sigmoid to convert logits to probabilities if needed
    if np.min(pos_scores) < 0 or np.max(pos_scores) > 1:
        pos_probs = 1 / (1 + np.exp(-pos_scores))
        neg_probs = 1 / (1 + np.exp(-neg_scores))
    else:
        pos_probs = pos_scores
        neg_probs = neg_scores
    
    # Binary classification metrics
    y_true = np.concatenate([np.ones_like(pos_probs), np.zeros_like(neg_probs)])
    y_scores = np.concatenate([pos_probs, neg_probs])
    y_pred = (y_scores >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    # AUC and AP
    try:
        auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
    except ValueError as e:
        logger.warning(f"Error calculating AUC/AP: {e}")
        auc = 0.5  # Random baseline
        ap = sum(y_true) / len(y_true)  # Proportion of positives
    
    # Ranking metrics (Hits@k and MRR)
    hits_at_k = {}
    all_scores = np.concatenate([pos_probs.reshape(-1, 1), neg_probs.reshape(-1, 1)], axis=1)
    
    # MRR calculation
    mrr_sum = 0.0
    num_queries = len(pos_probs)
    
    for i in range(num_queries):
        # For each query (positive edge), rank against all negatives
        query_pos_score = pos_probs[i]
        query_neg_scores = neg_probs
        
        # Combine scores and get ranks
        all_scores = np.concatenate([[query_pos_score], query_neg_scores])
        ranks = np.argsort(np.argsort(-all_scores))  # Higher scores get lower ranks
        
        # Rank of the positive example (0 is the index of the positive score)
        pos_rank = ranks[0] + 1  # Make rank 1-based
        
        # Update MRR
        mrr_sum += 1.0 / pos_rank
        
        # Update Hits@k
        for k in k_values:
            hit_key = f'hits@{k}'
            if hit_key not in hits_at_k:
                hits_at_k[hit_key] = 0
            
            if pos_rank <= k:
                hits_at_k[hit_key] += 1
    
    # Normalize Hits@k
    for k in k_values:
        hits_at_k[f'hits@{k}'] = hits_at_k[f'hits@{k}'] / num_queries if num_queries > 0 else 0.0
    
    # Calculate MRR
    mrr = mrr_sum / num_queries if num_queries > 0 else 0.0
    
    # Combine all metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'ap': ap,
        'mrr': mrr,
        **hits_at_k
    }
    
    return metrics

def evaluate_node_classification(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Evaluate node classification performance.
    
    Args:
        y_pred: Predicted node labels or probabilities
        y_true: Ground truth node labels
        mask: Optional mask for nodes to evaluate
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Apply mask if provided
    if mask is not None:
        y_pred = y_pred[mask]
        y_true = y_true[mask]
    
    # Convert probabilities to class predictions if needed
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        y_pred_classes = (y_pred > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_classes)
    
    # Handle multi-class vs binary classification
    if len(np.unique(y_true)) > 2:
        # Multi-class
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_classes, average='macro'
        )
        
        # Class-wise metrics
        class_metrics = {}
        labels = np.unique(np.concatenate([y_true, y_pred_classes]))
        for label in labels:
            label_precision, label_recall, label_f1, _ = precision_recall_fscore_support(
                y_true == label, y_pred_classes == label, average='binary'
            )
            class_metrics[f'class_{label}_precision'] = label_precision
            class_metrics[f'class_{label}_recall'] = label_recall
            class_metrics[f'class_{label}_f1'] = label_f1
    else:
        # Binary classification
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_classes, average='binary', zero_division=0
        )
        class_metrics = {}
    
    # Combine all metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        **class_metrics
    }
    
    return metrics

def evaluate_entity_alignment(
    src_embeddings: np.ndarray,
    tgt_embeddings: np.ndarray,
    ground_truth: List[Tuple[int, int]],
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Evaluate entity alignment performance.
    
    Args:
        src_embeddings: Source entity embeddings
        tgt_embeddings: Target entity embeddings
        ground_truth: List of (source_index, target_index) pairs
        k_values: Values of k for Hits@k metric
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Extract source and target indices from ground truth
    src_indices = [x[0] for x in ground_truth]
    tgt_indices = [x[1] for x in ground_truth]
    
    # Get embeddings for the aligned entities
    src_embs = src_embeddings[src_indices]
    tgt_embs = tgt_embeddings[tgt_indices]
    
    # Normalize embeddings
    src_embs = src_embs / np.linalg.norm(src_embs, axis=1, keepdims=True)
    tgt_embs = tgt_embs / np.linalg.norm(tgt_embs, axis=1, keepdims=True)
    
    # Compute similarity scores (cosine similarity)
    sim_scores = np.dot(src_embs, tgt_embs.T)
    
    # Calculate Hits@k and MRR
    hits_at_k = {f'hits@{k}': 0.0 for k in k_values}
    mrr_sum = 0.0
    
    for i in range(len(src_indices)):
        # Get similarity scores for the current source entity
        scores = sim_scores[i]
        
        # Get ranks (argsort returns indices that would sort the array)
        ranks = np.argsort(-scores)  # Descending order
        
        # Find the rank of the correct target entity (i)
        rank = np.where(ranks == i)[0][0] + 1  # +1 for 1-indexed rank
        
        # Update MRR
        mrr_sum += 1.0 / rank
        
        # Update Hits@k
        for k in k_values:
            if rank <= k:
                hits_at_k[f'hits@{k}'] += 1.0
    
    # Normalize metrics
    num_entities = len(src_indices)
    mrr = mrr_sum / num_entities
    
    for k in k_values:
        hits_at_k[f'hits@{k}'] /= num_entities
    
    # Combine metrics
    metrics = {
        'mrr': mrr,
        **hits_at_k
    }
    
    return metrics

def evaluate_knowledge_graph_completion(
    model,
    graph,
    features,
    test_triples: List[Tuple[int, str, int]],
    all_entities: np.ndarray,
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate knowledge graph completion performance.
    
    Args:
        model: Trained GNN model
        graph: DGL graph
        features: Node features
        test_triples: List of (head, relation, tail) test triples
        all_entities: Array of all entity IDs
        k_values: Values of k for Hits@k metric
        
    Returns:
        Dictionary of relation-specific metrics
    """
    import torch
    
    # Move to the same device as the model
    device = next(model.parameters()).device
    
    # Set model to evaluation mode
    model.eval()
    
    # Group test triples by relation type
    relation_triples = {}
    for h, r, t in test_triples:
        if r not in relation_triples:
            relation_triples[r] = []
        relation_triples[r].append((h, t))
    
    # Evaluate each relation type
    all_metrics = {}
    
    for relation, triples in relation_triples.items():
        # Extract heads and tails
        heads = np.array([h for h, _ in triples])
        tails = np.array([t for _, t in triples])
        
        # Convert to tensors and move to device
        heads_tensor = torch.tensor(heads, device=device)
        tails_tensor = torch.tensor(tails, device=device)
        
        # Prepare metrics
        hits_at_k = {f'hits@{k}': 0.0 for k in k_values}
        mrr_sum = 0.0
        
        # Evaluate head prediction
        with torch.no_grad():
            for i, (h, t) in enumerate(zip(heads, tails)):
                # Create candidate head entities (all entities except the true one)
                candidate_heads = np.setdiff1d(all_entities, [h])
                candidate_heads_tensor = torch.tensor(candidate_heads, device=device)
                
                # Create edges for scoring (candidate_heads -> t)
                candidate_tail_tensor = torch.full_like(candidate_heads_tensor, t)
                
                # Score candidate edges
                candidate_edge_index = torch.stack([candidate_heads_tensor, candidate_tail_tensor], dim=0)
                candidate_scores = model.score_edges(
                    features, candidate_edge_index, edge_type=('word', relation, 'word')
                )
                
                # Score true edge (h -> t)
                true_edge_index = torch.tensor([[h], [t]], device=device)
                true_score = model.score_edges(
                    features, true_edge_index, edge_type=('word', relation, 'word')
                )
                
                # Combine scores and get rank of true head
                all_scores = torch.cat([true_score, candidate_scores])
                ranks = torch.argsort(torch.argsort(all_scores, descending=True))
                rank = ranks[0].item() + 1  # +1 for 1-indexed rank
                
                # Update metrics
                mrr_sum += 1.0 / rank
                for k in k_values:
                    if rank <= k:
                        hits_at_k[f'hits@{k}'] += 1.0
        
        # Evaluate tail prediction
        with torch.no_grad():
            for i, (h, t) in enumerate(zip(heads, tails)):
                # Create candidate tail entities (all entities except the true one)
                candidate_tails = np.setdiff1d(all_entities, [t])
                candidate_tails_tensor = torch.tensor(candidate_tails, device=device)
                
                # Create edges for scoring (h -> candidate_tails)
                candidate_head_tensor = torch.full_like(candidate_tails_tensor, h)
                
                # Score candidate edges
                candidate_edge_index = torch.stack([candidate_head_tensor, candidate_tails_tensor], dim=0)
                candidate_scores = model.score_edges(
                    features, candidate_edge_index, edge_type=('word', relation, 'word')
                )
                
                # Score true edge (h -> t)
                true_edge_index = torch.tensor([[h], [t]], device=device)
                true_score = model.score_edges(
                    features, true_edge_index, edge_type=('word', relation, 'word')
                )
                
                # Combine scores and get rank of true tail
                all_scores = torch.cat([true_score, candidate_scores])
                ranks = torch.argsort(torch.argsort(all_scores, descending=True))
                rank = ranks[0].item() + 1  # +1 for 1-indexed rank
                
                # Update metrics
                mrr_sum += 1.0 / rank
                for k in k_values:
                    if rank <= k:
                        hits_at_k[f'hits@{k}'] += 1.0
        
        # Normalize metrics (divide by 2*|triples| as we evaluate both head and tail predictions)
        num_predictions = 2 * len(triples)
        relation_metrics = {
            'mrr': mrr_sum / num_predictions
        }
        
        for k in k_values:
            relation_metrics[f'hits@{k}'] = hits_at_k[f'hits@{k}'] / num_predictions
        
        all_metrics[relation] = relation_metrics
    
    # Calculate average metrics across relations
    avg_metrics = {}
    for metric in ['mrr'] + [f'hits@{k}' for k in k_values]:
        values = [rel_metrics[metric] for rel_metrics in all_metrics.values()]
        avg_metrics[metric] = sum(values) / len(values) if values else 0.0
    
    all_metrics['average'] = avg_metrics
    
    return all_metrics

def compute_confidence_scores(
    model,
    graph,
    features,
    candidate_triples: List[Tuple[int, str, int]],
    k_neighbors: int = 5
) -> np.ndarray:
    """
    Compute confidence scores for candidate triples.
    
    Args:
        model: Trained GNN model
        graph: DGL graph
        features: Node features
        candidate_triples: List of (head, relation, tail) candidate triples
        k_neighbors: Number of neighbors to consider for neighborhood similarity
        
    Returns:
        Array of confidence scores for each candidate triple
    """
    import torch
    
    # Move to the same device as the model
    device = next(model.parameters()).device
    
    # Set model to evaluation mode
    model.eval()
    
    # Extract node embeddings
    with torch.no_grad():
        node_embeddings = model.get_node_embeddings(features)['word'].cpu().numpy()
    
    # Compute confidence scores for each triple
    confidence_scores = []
    
    for h, r, t in candidate_triples:
        # 1. Direct edge score
        head_tensor = torch.tensor([h], device=device)
        tail_tensor = torch.tensor([t], device=device)
        edge_index = torch.stack([head_tensor, tail_tensor], dim=0)
        
        with torch.no_grad():
            edge_score = model.score_edges(
                features, edge_index, edge_type=('word', r, 'word')
            ).item()
        
        # 2. Embedding similarity (cosine)
        head_emb = node_embeddings[h]
        tail_emb = node_embeddings[t]
        emb_similarity = np.dot(head_emb, tail_emb) / (np.linalg.norm(head_emb) * np.linalg.norm(tail_emb))
        
        # 3. Neighborhood overlap
        # Get k-nearest neighbors for head and tail
        head_neighbors = get_k_nearest_neighbors(node_embeddings, h, k=k_neighbors)
        tail_neighbors = get_k_nearest_neighbors(node_embeddings, t, k=k_neighbors)
        
        # Compute Jaccard similarity between neighbor sets
        neighborhood_similarity = len(set(head_neighbors) & set(tail_neighbors)) / len(set(head_neighbors) | set(tail_neighbors))
        
        # Combine scores (simple weighted average for now)
        combined_score = 0.6 * edge_score + 0.3 * emb_similarity + 0.1 * neighborhood_similarity
        confidence_scores.append(combined_score)
    
    return np.array(confidence_scores)

def get_k_nearest_neighbors(embeddings: np.ndarray, node_idx: int, k: int = 5) -> List[int]:
    """
    Get k-nearest neighbors for a node based on embedding similarity.
    
    Args:
        embeddings: Node embedding matrix
        node_idx: Index of the query node
        k: Number of neighbors to return
        
    Returns:
        List of k nearest neighbor indices
    """
    # Get query node embedding
    query_emb = embeddings[node_idx]
    
    # Compute similarity to all other nodes
    sim_scores = np.dot(embeddings, query_emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
    )
    
    # Sort nodes by similarity (excluding the query node itself)
    sim_scores[node_idx] = -np.inf  # Exclude self
    top_indices = np.argsort(-sim_scores)[:k]  # Descending order
    
    return top_indices.tolist() 