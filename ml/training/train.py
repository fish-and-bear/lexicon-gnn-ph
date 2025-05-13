"""
Training script for the HGNN model with multi-task learning.

This module provides functionality for training the HGNN model
with joint link prediction and node classification tasks.
"""

import os
import time
import logging
import torch
import torch.nn.functional as F
import dgl
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any
import json
from datetime import datetime

from ml.models import GraphEncoder, LinkPredictionHead, NodeClassificationHead

logger = logging.getLogger(__name__)

class MultiTaskTrainer:
    """Trainer for multi-task learning with HGNNs."""
    
    def __init__(self, 
                 encoder: GraphEncoder,
                 link_prediction_head: LinkPredictionHead,
                 node_classification_head: NodeClassificationHead,
                 device: str = 'cuda',
                 save_dir: str = 'saved_models',
                 experiment_name: Optional[str] = None):
        """
        Initialize the trainer.
        
        Args:
            encoder: Graph encoder model
            link_prediction_head: Link prediction head
            node_classification_head: Node classification head
            device: Device to train on ('cuda' or 'cpu')
            save_dir: Directory to save models
            experiment_name: Name of the experiment (if None, will use timestamp)
        """
        self.encoder = encoder
        self.link_prediction_head = link_prediction_head
        self.node_classification_head = node_classification_head
        self.device = device
        
        # Create save directory
        self.save_dir = save_dir
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(save_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Move models to device
        self.encoder.to(device)
        self.link_prediction_head.to(device)
        self.node_classification_head.to(device)
        
        # Initialize history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'link_pred_loss': [],
            'node_class_loss': [],
            'val_link_pred_loss': [],
            'val_node_class_loss': [],
            'val_link_pred_mrr': [],
            'val_node_class_acc': [],
        }
    
    def train_step(self, 
                   g: dgl.DGLGraph, 
                   features: Dict[str, torch.Tensor],
                   pos_edges: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                   neg_edges: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                   node_labels: torch.Tensor,
                   node_mask: torch.Tensor,
                   link_weight: float = 1.0,
                   node_weight: float = 1.0) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            g: DGL heterogeneous graph
            features: Dictionary of node features
            pos_edges: Dictionary of positive edges (src, dst) per relation
            neg_edges: Dictionary of negative edges (src, dst) per relation
            node_labels: Node labels for classification
            node_mask: Mask for valid node labels (1 for valid, 0 for invalid)
            link_weight: Weight for link prediction loss
            node_weight: Weight for node classification loss
            
        Returns:
            Dictionary with loss values
        """
        # Move data to device
        if isinstance(features, dict):
            features = {k: v.to(self.device) for k, v in features.items()}
        else:
            features = features.to(self.device)
        
        pos_edges = {k: (src.to(self.device), dst.to(self.device)) 
                   for k, (src, dst) in pos_edges.items()}
        neg_edges = {k: (src.to(self.device), dst.to(self.device)) 
                   for k, (src, dst) in neg_edges.items()}
        node_labels = node_labels.to(self.device)
        node_mask = node_mask.to(self.device)
        
        # Forward pass
        # 1. Get node embeddings
        node_embeddings = self.encoder(g, features)
        
        # 2. Link prediction
        link_scores = self.link_prediction_head(node_embeddings, pos_edges, neg_edges)
        link_loss = self.link_prediction_head.loss(link_scores)
        
        # 3. Node classification
        node_loss = self.node_classification_head.loss(node_embeddings, node_labels, mask=node_mask)
        
        # 4. Combine losses with weights
        total_loss = link_weight * link_loss + node_weight * node_loss
        
        return {
            'total_loss': total_loss,
            'link_loss': link_loss,
            'node_loss': node_loss
        }
    
    def validate(self, 
                g: dgl.DGLGraph, 
                features: Dict[str, torch.Tensor],
                val_pos_edges: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                val_neg_edges: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                val_node_labels: torch.Tensor,
                val_node_mask: torch.Tensor,
                link_weight: float = 1.0,
                node_weight: float = 1.0) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            g: DGL heterogeneous graph
            features: Dictionary of node features
            val_pos_edges: Dictionary of positive validation edges
            val_neg_edges: Dictionary of negative validation edges
            val_node_labels: Validation node labels
            val_node_mask: Validation mask for node labels
            link_weight: Weight for link prediction loss
            node_weight: Weight for node classification loss
            
        Returns:
            Dictionary with validation metrics
        """
        # Set models to evaluation mode
        self.encoder.eval()
        self.link_prediction_head.eval()
        self.node_classification_head.eval()
        
        # Move data to device
        if isinstance(features, dict):
            features = {k: v.to(self.device) for k, v in features.items()}
        else:
            features = features.to(self.device)
        
        val_pos_edges = {k: (src.to(self.device), dst.to(self.device)) 
                       for k, (src, dst) in val_pos_edges.items()}
        val_neg_edges = {k: (src.to(self.device), dst.to(self.device)) 
                       for k, (src, dst) in val_neg_edges.items()}
        val_node_labels = val_node_labels.to(self.device)
        val_node_mask = val_node_mask.to(self.device)
        
        with torch.no_grad():
            # Get node embeddings
            node_embeddings = self.encoder(g, features)
            
            # Link prediction
            link_scores = self.link_prediction_head(node_embeddings, val_pos_edges, val_neg_edges)
            link_loss = self.link_prediction_head.loss(link_scores)
            
            # Node classification
            node_loss = self.node_classification_head.loss(node_embeddings, val_node_labels, mask=val_node_mask)
            node_metrics = self.node_classification_head.evaluate(node_embeddings, val_node_labels, mask=val_node_mask)
            
            # Calculate MRR for link prediction
            mrr_values = []
            for rel_type in val_pos_edges.keys():
                pos_key = f"{rel_type}_pos"
                neg_key = f"{rel_type}_neg"
                
                if pos_key in link_scores and neg_key in link_scores:
                    pos_scores = link_scores[pos_key]
                    neg_scores = link_scores[neg_key].view(-1, pos_scores.shape[0])
                    
                    # Calculate ranks
                    all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores.t()], dim=1)
                    ranks = (all_scores >= all_scores[:, 0].unsqueeze(1)).float().sum(dim=1)
                    
                    # MRR
                    mrr = (1.0 / ranks).mean().item()
                    mrr_values.append(mrr)
            
            mrr = np.mean(mrr_values) if mrr_values else 0.0
            
            # Combine losses
            total_loss = link_weight * link_loss + node_weight * node_loss
            
            metrics = {
                'val_total_loss': total_loss.item(),
                'val_link_loss': link_loss.item(),
                'val_node_loss': node_loss.item(),
                'val_mrr': mrr
            }
            
            # Add node classification metrics
            metrics.update(node_metrics)
        
        # Set models back to training mode
        self.encoder.train()
        self.link_prediction_head.train()
        self.node_classification_head.train()
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        Save a model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'link_prediction_head_state_dict': self.link_prediction_head.state_dict(),
            'node_classification_head_state_dict': self.node_classification_head.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.experiment_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, latest_path)
        
        # Save epoch checkpoint
        epoch_path = os.path.join(self.experiment_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, epoch_path)
        
        # Save best model if needed
        if is_best:
            best_path = os.path.join(self.experiment_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.link_prediction_head.load_state_dict(checkpoint['link_prediction_head_state_dict'])
        self.node_classification_head.load_state_dict(checkpoint['node_classification_head_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        
        return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})
    
    def train(self, 
             g: dgl.DGLGraph, 
             features: Dict[str, torch.Tensor],
             train_pos_edges: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
             train_neg_edges: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
             train_node_labels: torch.Tensor,
             train_node_mask: torch.Tensor,
             val_pos_edges: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
             val_neg_edges: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
             val_node_labels: torch.Tensor,
             val_node_mask: torch.Tensor,
             epochs: int = 100,
             lr: float = 0.001,
             link_weight: float = 1.0,
             node_weight: float = 1.0,
             patience: int = 10,
             checkpoint_interval: int = 10,
             monitor: str = 'val_total_loss',
             monitor_mode: str = 'min'):
        """
        Train the model.
        
        Args:
            g: DGL heterogeneous graph
            features: Dictionary of node features
            train_pos_edges: Dictionary of positive training edges
            train_neg_edges: Dictionary of negative training edges
            train_node_labels: Training node labels
            train_node_mask: Training mask for node labels
            val_pos_edges: Dictionary of positive validation edges
            val_neg_edges: Dictionary of negative validation edges
            val_node_labels: Validation node labels
            val_node_mask: Validation mask for node labels
            epochs: Number of epochs
            lr: Learning rate
            link_weight: Weight for link prediction loss
            node_weight: Weight for node classification loss
            patience: Early stopping patience
            checkpoint_interval: Interval for saving checkpoints
            monitor: Metric to monitor for early stopping and best model
            monitor_mode: Mode for monitoring ('min' or 'max')
        """
        # Create optimizer
        params = list(self.encoder.parameters()) + \
                list(self.link_prediction_head.parameters()) + \
                list(self.node_classification_head.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        
        # For early stopping
        best_value = float('inf') if monitor_mode == 'min' else float('-inf')
        no_improvement = 0
        
        # Training loop
        logger.info(f"Starting training for {epochs} epochs")
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training step
            self.encoder.train()
            self.link_prediction_head.train()
            self.node_classification_head.train()
            
            optimizer.zero_grad()
            
            losses = self.train_step(
                g, features, train_pos_edges, train_neg_edges, 
                train_node_labels, train_node_mask, link_weight, node_weight
            )
            
            losses['total_loss'].backward()
            optimizer.step()
            
            # Validation
            val_metrics = self.validate(
                g, features, val_pos_edges, val_neg_edges,
                val_node_labels, val_node_mask, link_weight, node_weight
            )
            
            # Update history
            self.history['train_loss'].append(losses['total_loss'].item())
            self.history['link_pred_loss'].append(losses['link_loss'].item())
            self.history['node_class_loss'].append(losses['node_loss'].item())
            self.history['val_loss'].append(val_metrics['val_total_loss'])
            self.history['val_link_pred_loss'].append(val_metrics['val_link_loss'])
            self.history['val_node_class_loss'].append(val_metrics['val_node_loss'])
            self.history['val_link_pred_mrr'].append(val_metrics['val_mrr'])
            self.history['val_node_class_acc'].append(val_metrics.get('pos_tags_accuracy', 0.0))
            
            # Log progress
            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {losses['total_loss']:.4f} | "
                    f"Val Loss: {val_metrics['val_total_loss']:.4f} | "
                    f"Val MRR: {val_metrics['val_mrr']:.4f} | "
                    f"Val Acc: {val_metrics.get('pos_tags_accuracy', 0.0):.4f}"
                )
            
            # Save checkpoint
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch + 1, val_metrics)
            
            # Check for improvement
            current_value = val_metrics[monitor]
            is_improvement = (monitor_mode == 'min' and current_value < best_value) or \
                           (monitor_mode == 'max' and current_value > best_value)
            
            if is_improvement:
                best_value = current_value
                self.save_checkpoint(epoch + 1, val_metrics, is_best=True)
                no_improvement = 0
                logger.info(f"New best model saved! {monitor}: {best_value:.4f}")
            else:
                no_improvement += 1
                if patience > 0 and no_improvement >= patience:
                    logger.info(f"Early stopping after {epoch+1} epochs")
                    break
        
        # Final evaluation
        final_metrics = self.validate(
            g, features, val_pos_edges, val_neg_edges, 
            val_node_labels, val_node_mask, link_weight, node_weight
        )
        
        # Log training completion and final metrics
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        logger.info(f"Final metrics: {final_metrics}")
        
        # Save final model
        self.save_checkpoint(epochs, final_metrics)
        
        # Save history
        history_path = os.path.join(self.experiment_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f)
        
        return self.history

def train_model(config, data):
    """
    Train a model with the given configuration and data.
    
    Args:
        config: Dictionary with training configuration
        data: Dictionary with preprocessed data
        
    Returns:
        Trained trainer object
    """
    # Extract data
    g = data['graph']
    features = data['features']
    edge_splits = data['edge_splits']
    negative_samples = data.get('negative_samples', {})
    node_labels = data['node_labels']['pos_tags']
    
    # Get device
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare node labels and mask
    # Create mask for valid labels (not -1)
    node_mask = (node_labels >= 0)
    num_classes = int(node_labels[node_mask].max().item()) + 1
    
    # Create train/val masks for nodes
    node_indices = torch.where(node_mask)[0]
    num_nodes = len(node_indices)
    
    # Use 80% for training, 20% for validation
    perm = torch.randperm(num_nodes)
    train_size = int(num_nodes * 0.8)
    
    train_indices = node_indices[perm[:train_size]]
    val_indices = node_indices[perm[train_size:]]
    
    train_node_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
    train_node_mask[train_indices] = True
    
    val_node_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
    val_node_mask[val_indices] = True
    
    # Prepare edge splits
    train_pos_edges = {}
    val_pos_edges = {}
    test_pos_edges = {}
    
    # Collect relation types to predict
    relation_types = []
    
    for rel in g.etypes:
        if "rev" not in rel:  # Skip reverse relations
            relation_types.append(rel)
            
            if rel in edge_splits:
                # Get source and destination nodes
                src, dst = g.edges(etype=rel)
                
                # Use edge masks to get train/val/test edges
                train_mask = edge_splits[rel]['train']
                val_mask = edge_splits[rel]['val']
                test_mask = edge_splits[rel]['test']
                
                # Extract edges
                train_pos_edges[rel] = (src[train_mask], dst[train_mask])
                val_pos_edges[rel] = (src[val_mask], dst[val_mask])
                test_pos_edges[rel] = (src[test_mask], dst[test_mask])
    
    # Prepare negative edges
    train_neg_edges = {}
    val_neg_edges = {}
    
    for rel in relation_types:
        if rel in negative_samples:
            neg_edges = negative_samples[rel]
            num_neg = len(neg_edges)
            
            # Split negative edges for train/val
            train_size = int(0.8 * num_neg)
            
            train_neg = neg_edges[:train_size]
            val_neg = neg_edges[train_size:]
            
            train_neg_edges[rel] = (train_neg[:, 0], train_neg[:, 1])
            val_neg_edges[rel] = (val_neg[:, 0], val_neg[:, 1])
    
    # Get feature dimension
    in_dim = sum(feat.shape[1] for feat in features.values())
    
    # Create models
    encoder = GraphEncoder(
        in_dim=in_dim,
        hidden_dim=config.get('hidden_dim', 256),
        out_dim=config.get('embedding_dim', 128),
        rel_names=relation_types,
        num_layers=config.get('num_layers', 3),
        dropout=config.get('dropout', 0.1),
    )
    
    link_prediction_head = LinkPredictionHead(
        embedding_dim=config.get('embedding_dim', 128),
        relation_types=relation_types,
        dropout=config.get('dropout', 0.1),
        score_type=config.get('score_type', 'distmult'),
    )
    
    node_classification_head = NodeClassificationHead(
        embedding_dim=config.get('embedding_dim', 128),
        num_classes=num_classes,
        hidden_dim=config.get('hidden_dim', 256),
        dropout=config.get('dropout', 0.1),
    )
    
    # Create trainer
    trainer = MultiTaskTrainer(
        encoder=encoder,
        link_prediction_head=link_prediction_head,
        node_classification_head=node_classification_head,
        device=device,
        save_dir=config.get('save_dir', 'saved_models'),
        experiment_name=config.get('experiment_name'),
    )
    
    # Check if pretraining is enabled
    if config.get('pretrain', False):
        logger.info("Starting pretraining...")
        encoder.pretrain(
            g=g,
            features=features,
            num_epochs=config.get('pretrain_epochs', 50),
            lr=config.get('pretrain_lr', 0.001),
            device=device
        )
        logger.info("Pretraining completed")
    
    # Train the model
    trainer.train(
        g=g,
        features=features,
        train_pos_edges=train_pos_edges,
        train_neg_edges=train_neg_edges,
        train_node_labels=node_labels,
        train_node_mask=train_node_mask,
        val_pos_edges=val_pos_edges,
        val_neg_edges=val_neg_edges,
        val_node_labels=node_labels,
        val_node_mask=val_node_mask,
        epochs=config.get('epochs', 100),
        lr=config.get('lr', 0.001),
        link_weight=config.get('link_weight', 1.0),
        node_weight=config.get('node_weight', 1.0),
        patience=config.get('patience', 10),
    )
    
    return trainer


if __name__ == "__main__":
    import argparse
    from ml.data.preprocess import run_preprocessing
    
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--pretrain', action='store_true', help='Enable pretraining')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--save_dir', type=str, default='saved_models', help='Save directory')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {
        'pretrain': args.pretrain,
        'epochs': args.epochs,
        'lr': args.lr,
        'hidden_dim': args.hidden_dim, 
        'embedding_dim': args.embedding_dim,
        'save_dir': args.save_dir
    }
    
    if args.config:
        with open(args.config, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    # Run preprocessing to get data
    data = run_preprocessing()
    
    # Train model
    trainer = train_model(config, data)