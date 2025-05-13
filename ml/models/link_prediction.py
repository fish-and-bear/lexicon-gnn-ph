"""
Link prediction head for predicting relations between lemmas.

This module implements the link prediction component for the heterogeneous GNN,
allowing the model to predict missing relationships between lemmas.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)

class LinkPredictionHead(nn.Module):
    """Link prediction head for predicting relations between nodes."""
    
    def __init__(self, 
                 embedding_dim: int,
                 relation_types: List[str],
                 dropout: float = 0.1,
                 score_type: str = "distmult",
                 margin: float = 1.0):
        """
        Initialize the link prediction head.
        
        Args:
            embedding_dim: Dimension of node embeddings
            relation_types: List of relation types to predict
            dropout: Dropout probability
            score_type: Scoring function type ('distmult', 'complex', 'mlp')
            margin: Margin for ranking loss
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.relation_types = relation_types
        self.num_relations = len(relation_types)
        self.score_type = score_type
        self.margin = margin
        
        # Create relation embeddings/weights based on scoring function type
        if score_type == 'distmult':
            # DistMult: bilinear diagonal form
            self.relation_weights = nn.Parameter(torch.Tensor(self.num_relations, embedding_dim))
        
        elif score_type == 'complex':
            # ComplEx: complex-valued embeddings (stored as pairs of real values)
            # Real part
            self.relation_weights_real = nn.Parameter(torch.Tensor(self.num_relations, embedding_dim))
            # Imaginary part
            self.relation_weights_imag = nn.Parameter(torch.Tensor(self.num_relations, embedding_dim))
        
        elif score_type == 'mlp':
            # MLP-based scoring function
            hidden_dim = 2 * embedding_dim
            self.mlp = nn.ModuleDict()
            for rel in relation_types:
                self.mlp[rel] = nn.Sequential(
                    nn.Linear(2 * embedding_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1)
                )
        else:
            raise ValueError(f"Unknown score type: {score_type}")
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize model parameters."""
        if self.score_type == 'distmult':
            nn.init.xavier_uniform_(self.relation_weights)
        elif self.score_type == 'complex':
            nn.init.xavier_uniform_(self.relation_weights_real)
            nn.init.xavier_uniform_(self.relation_weights_imag)
    
    def score_edges(self, 
                    embeddings: torch.Tensor, 
                    src_indices: torch.Tensor, 
                    dst_indices: torch.Tensor, 
                    relation_idx: int) -> torch.Tensor:
        """
        Score edges using the selected scoring function.
        
        Args:
            embeddings: Node embeddings
            src_indices: Source node indices
            dst_indices: Target node indices
            relation_idx: Index of the relation type
            
        Returns:
            Edge scores
        """
        # Extract embeddings for source and destination nodes
        src_emb = embeddings[src_indices]
        dst_emb = embeddings[dst_indices]
        
        # Apply dropout
        src_emb = self.dropout(src_emb)
        dst_emb = self.dropout(dst_emb)
        
        # Score based on selected function
        if self.score_type == 'distmult':
            # DistMult: <s, r, o> = sum_i (s_i * r_i * o_i)
            rel_emb = self.relation_weights[relation_idx]
            scores = torch.sum(src_emb * rel_emb * dst_emb, dim=1)
        
        elif self.score_type == 'complex':
            # ComplEx: Real(<s, r, conjugate(o)>)
            rel_real = self.relation_weights_real[relation_idx]
            rel_imag = self.relation_weights_imag[relation_idx]
            
            # Real part: s_real * r_real * o_real + s_imag * r_imag * o_real
            real_part = src_emb * rel_real * dst_emb
            
            # Imaginary part: s_real * r_imag * o_imag + s_imag * r_real * o_imag
            imag_part = src_emb * rel_imag * dst_emb
            
            scores = torch.sum(real_part - imag_part, dim=1)
        
        elif self.score_type == 'mlp':
            # MLP: concatenate source and destination embeddings
            pairs = torch.cat([src_emb, dst_emb], dim=1)
            rel_name = self.relation_types[relation_idx]
            scores = self.mlp[rel_name](pairs).squeeze()
        
        return scores
    
    def forward(self, 
                embeddings: torch.Tensor, 
                pos_edges: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                neg_edges: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for link prediction.
        
        Args:
            embeddings: Node embeddings
            pos_edges: Dictionary of positive edges (src, dst) per relation
            neg_edges: Dictionary of negative edges (src, dst) per relation
            
        Returns:
            Dictionary of scores for positive and negative edges
        """
        scores = {}
        
        # Process each relation type
        for rel_idx, rel_name in enumerate(self.relation_types):
            # Skip if no edges for this relation
            if rel_name not in pos_edges:
                continue
            
            # Get positive edges
            pos_src, pos_dst = pos_edges[rel_name]
            pos_scores = self.score_edges(embeddings, pos_src, pos_dst, rel_idx)
            scores[f"{rel_name}_pos"] = pos_scores
            
            # Get negative edges (if provided)
            if neg_edges is not None and rel_name in neg_edges:
                neg_src, neg_dst = neg_edges[rel_name]
                neg_scores = self.score_edges(embeddings, neg_src, neg_dst, rel_idx)
                scores[f"{rel_name}_neg"] = neg_scores
        
        return scores
    
    def predict(self, 
                embeddings: torch.Tensor, 
                src_indices: torch.Tensor,
                dst_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict relation scores for pairs of nodes.
        
        Args:
            embeddings: Node embeddings
            src_indices: Source node indices
            dst_indices: Target node indices
            
        Returns:
            Dictionary of scores per relation type
        """
        scores = {}
        
        # Score each pair for each relation type
        for rel_idx, rel_name in enumerate(self.relation_types):
            rel_scores = self.score_edges(embeddings, src_indices, dst_indices, rel_idx)
            scores[rel_name] = rel_scores
        
        return scores
    
    def loss(self, scores: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute the ranking loss for link prediction.
        
        Args:
            scores: Dictionary of scores for positive and negative edges
            
        Returns:
            Loss value
        """
        loss = 0.0
        num_rels = 0
        
        # Compute loss for each relation type
        for rel_name in self.relation_types:
            pos_key = f"{rel_name}_pos"
            neg_key = f"{rel_name}_neg"
            
            if pos_key in scores and neg_key in scores:
                pos_scores = scores[pos_key]
                neg_scores = scores[neg_key]
                
                # Max-margin loss: max(0, margin - pos + neg)
                # We want pos_scores to be higher than neg_scores
                rel_loss = F.relu(self.margin - pos_scores + neg_scores).mean()
                loss += rel_loss
                num_rels += 1
        
        # Return average loss across relations
        if num_rels > 0:
            return loss / num_rels
        else:
            # Return zero loss if no relations were computed
            return torch.tensor(0.0, device=loss.device)

    def get_best_candidates(self, 
                           embeddings: torch.Tensor,
                           query_indices: torch.Tensor,
                           rel_type: str,
                           k: int = 10,
                           excluded_indices: Optional[List[int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the top-k prediction candidates for a relation type.
        
        Args:
            embeddings: Node embeddings
            query_indices: Indices of query nodes
            rel_type: Relation type to predict
            k: Number of candidates to return
            excluded_indices: Indices to exclude from candidates
            
        Returns:
            Tuple of (candidate_indices, scores)
        """
        rel_idx = self.relation_types.index(rel_type)
        device = embeddings.device
        
        # For each query node, score against all possible target nodes
        all_scores = []
        all_candidates = []
        
        batch_size = 128  # Process in batches to save memory
        
        for i in range(0, len(query_indices), batch_size):
            batch_queries = query_indices[i:i+batch_size]
            batch_scores = []
            
            # For each query in the batch
            for query_idx in batch_queries:
                # Create pairs with all nodes
                num_nodes = embeddings.shape[0]
                src_indices = torch.full((num_nodes,), query_idx, device=device)
                dst_indices = torch.arange(num_nodes, device=device)
                
                # Score all pairs
                scores = self.score_edges(embeddings, src_indices, dst_indices, rel_idx)
                
                # Exclude specified indices (e.g., existing edges)
                if excluded_indices is not None:
                    scores[excluded_indices] = float('-inf')
                
                # Get top-k candidates
                top_scores, top_indices = torch.topk(scores, k=min(k, num_nodes))
                
                batch_scores.append((top_indices, top_scores))
            
            all_candidates.extend([indices for indices, _ in batch_scores])
            all_scores.extend([scores for _, scores in batch_scores])
        
        return torch.stack(all_candidates), torch.stack(all_scores) 