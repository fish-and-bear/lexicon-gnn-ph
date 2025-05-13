#!/usr/bin/env python3

"""
Training script for the Heterogeneous Graph Neural Network model for Filipino lexical data.

This script handles the training and evaluation of the GNN model for tasks like
link prediction and node classification on the Filipino lexical knowledge graph.
"""

import os
import sys
import argparse
import logging
import json
import time
import torch
import dgl
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.data.db_adapter import DatabaseAdapter
from ml.data.lexical_graph_builder import LexicalGraphBuilder
from ml.data.feature_extractors import LexicalFeatureExtractor
from ml.models.hgnn import HeterogeneousGNN
from ml.utils.logging_utils import setup_logging
from ml.utils.evaluation_utils import evaluate_link_prediction, evaluate_node_classification

# Set up logging
logger = logging.getLogger(__name__)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"training_{timestamp}.log"
setup_logging(log_file=LOG_FILE, level=logging.INFO)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a heterogeneous GNN model on Filipino lexical data")
    
    parser.add_argument("--config", type=str, default="config/default_config.json",
                        help="Path to configuration file")
    parser.add_argument("--db-config", type=str, default="my_db_config.json",
                        help="Path to database configuration file")
    parser.add_argument("--model-dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory to save processed data")
    parser.add_argument("--skip-db-load", action="store_true",
                        help="Skip loading data from database (use cached data)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")
    parser.add_argument("--pretrained-model", type=str, default=None,
                        help="Path to pre-trained HGMAE model for initialization")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze the pre-trained encoder weights during fine-tuning")
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        sys.exit(1)

def load_db_config(config_path):
    """Load database configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            db_config = json.load(f)
        logger.info(f"Loaded database configuration from {config_path}")
        return db_config.get('db_config', {})
    except Exception as e:
        logger.error(f"Failed to load database configuration from {config_path}: {e}")
        sys.exit(1)

def load_data_from_db(db_config, target_languages=None):
    """Load lexical data from database."""
    logger.info("Loading data from database...")
    
    try:
        # Connect to database
        db_adapter = DatabaseAdapter(db_config)
        
        # Load data frames
        lemmas_df = db_adapter.get_lemmas_df(target_languages=target_languages)
        logger.info(f"Loaded {len(lemmas_df)} lemmas")
        
        relations_df = db_adapter.get_relations_df()
        logger.info(f"Loaded {len(relations_df)} relations")
        
        definitions_df = db_adapter.get_definitions_df()
        logger.info(f"Loaded {len(definitions_df)} definitions")
        
        etymologies_df = db_adapter.get_etymologies_df()
        logger.info(f"Loaded {len(etymologies_df)} etymologies")
        
        # Optional data frames
        try:
            pronunciations_df = db_adapter.get_pronunciations_df()
            logger.info(f"Loaded {len(pronunciations_df)} pronunciations")
        except:
            logger.warning("Failed to load pronunciations, continuing without them")
            pronunciations_df = pd.DataFrame()
        
        try:
            word_forms_df = db_adapter.get_word_forms_df()
            logger.info(f"Loaded {len(word_forms_df)} word forms")
        except:
            logger.warning("Failed to load word forms, continuing without them")
            word_forms_df = pd.DataFrame()
        
        return {
            'lemmas_df': lemmas_df,
            'relations_df': relations_df,
            'definitions_df': definitions_df,
            'etymologies_df': etymologies_df,
            'pronunciations_df': pronunciations_df,
            'word_forms_df': word_forms_df,
        }
    
    except Exception as e:
        logger.error(f"Failed to load data from database: {e}")
        sys.exit(1)

def save_data_to_cache(data, data_dir):
    """Save loaded data to disk cache."""
    try:
        os.makedirs(data_dir, exist_ok=True)
        
        for name, df in data.items():
            file_path = os.path.join(data_dir, f"{name}.parquet")
            df.to_parquet(file_path, index=False)
            logger.info(f"Saved {name} to {file_path}")
        
        logger.info(f"All data saved to {data_dir}")
    except Exception as e:
        logger.error(f"Failed to save data to cache: {e}")

def load_data_from_cache(data_dir):
    """Load data from disk cache."""
    data = {}
    
    try:
        for name in ['lemmas_df', 'relations_df', 'definitions_df', 'etymologies_df', 
                     'pronunciations_df', 'word_forms_df']:
            file_path = os.path.join(data_dir, f"{name}.parquet")
            if os.path.exists(file_path):
                data[name] = pd.read_parquet(file_path)
                logger.info(f"Loaded {name} from {file_path}: {len(data[name])} rows")
            else:
                logger.warning(f"Cache file {file_path} not found")
                data[name] = pd.DataFrame()
        
        return data
    except Exception as e:
        logger.error(f"Failed to load data from cache: {e}")
        sys.exit(1)

def build_graph_data(data, config):
    """Build heterogeneous graph from dataframes."""
    logger.info("Building heterogeneous graph...")
    
    try:
        target_languages = config.get('data', {}).get('target_languages')
        if target_languages:
            logger.info(f"Filtering data to target languages: {target_languages}")
        
        # Create graph builder
        graph_builder = LexicalGraphBuilder(config, target_languages=target_languages)
        
        # Build graph
        graph = graph_builder.build_graph(
            lemmas_df=data['lemmas_df'],
            relations_df=data['relations_df'],
            definitions_df=data['definitions_df'],
            etymologies_df=data['etymologies_df'],
            pronunciations_df=data['pronunciations_df'],
            word_forms_df=data['word_forms_df'],
        )
        
        # Get node mappings
        mappings = graph_builder.get_node_mappings()
        
        return graph, mappings
    
    except Exception as e:
        logger.error(f"Failed to build graph: {e}")
        sys.exit(1)

def extract_features(data, graph, config):
    """Extract features for graph nodes."""
    logger.info("Extracting node features...")
    
    try:
        # Create feature extractor
        feature_extractor = LexicalFeatureExtractor(
            use_xlmr=config.get('data', {}).get('use_xlmr', True),
            use_fasttext=config.get('data', {}).get('use_fasttext', True),
            use_char_ngrams=config.get('data', {}).get('use_char_ngrams', True),
            use_phonetic_features=config.get('data', {}).get('use_phonetic_features', True),
            use_etymology_features=config.get('data', {}).get('use_etymology_features', True),
            use_baybayin_features=config.get('data', {}).get('use_baybayin_features', True),
            normalize_features=config.get('data', {}).get('normalize_features', True),
        )
        
        # Extract features
        features = feature_extractor.extract_all_features(
            lemmas_df=data['lemmas_df'],
            definitions_df=data['definitions_df'],
            etymologies_df=data['etymologies_df'],
            pronunciations_df=data['pronunciations_df'],
            word_forms_df=data['word_forms_df'],
        )
        
        return features
    
    except Exception as e:
        logger.error(f"Failed to extract features: {e}")
        sys.exit(1)

def prepare_training_data(graph, mappings, config):
    """
    Prepare training data for link prediction and node classification.
    
    This function:
    1. Splits edges into train/valid/test sets for link prediction
    2. Prepares node classification labels for training
    
    Args:
        graph: The heterogeneous graph
        mappings: Node ID mappings
        config: Configuration dictionary
    
    Returns:
        Dictionary of training data components
    """
    logger.info("Preparing training data...")
    
    try:
        # Edge type selection for link prediction
        relation_types_to_predict = config.get('data', {}).get('relation_types_to_predict', [])
        if not relation_types_to_predict:
            logger.warning("No relation types specified for prediction. Using all available types.")
            relation_types_to_predict = [etype[1] for etype in graph.canonical_etypes 
                                        if etype[0] == 'word' and etype[2] == 'word']
        
        logger.info(f"Selected relation types for prediction: {relation_types_to_predict}")
        
        # Prepare edge data for link prediction
        edge_data = {}
        for rel_type in relation_types_to_predict:
            etype = ('word', rel_type, 'word')
            if etype in graph.canonical_etypes:
                # Get edges
                src, dst = graph.edges(etype=etype)
                
                # Split edges
                num_edges = len(src)
                if num_edges > 0:
                    # Shuffle indices
                    indices = torch.randperm(num_edges)
                    src = src[indices]
                    dst = dst[indices]
                    
                    # Split ratios
                    valid_split = config.get('data', {}).get('valid_split', 0.1)
                    test_split = config.get('data', {}).get('test_split', 0.2)
                    train_split = 1.0 - valid_split - test_split
                    
                    # Calculate sizes
                    train_size = int(train_split * num_edges)
                    valid_size = int(valid_split * num_edges)
                    
                    # Split data
                    train_src, train_dst = src[:train_size], dst[:train_size]
                    valid_src, valid_dst = src[train_size:train_size+valid_size], dst[train_size:train_size+valid_size]
                    test_src, test_dst = src[train_size+valid_size:], dst[train_size+valid_size:]
                    
                    edge_data[rel_type] = {
                        'train': (train_src, train_dst),
                        'valid': (valid_src, valid_dst),
                        'test': (test_src, test_dst),
                        'etype': etype
                    }
                    
                    logger.info(f"Prepared {rel_type} edges: {train_size} train, {valid_size} valid, {len(test_src)} test")
                else:
                    logger.warning(f"No edges found for relation type {rel_type}")
            else:
                logger.warning(f"Relation type {rel_type} not found in graph")
        
        # Prepare node classification data (e.g., POS tags)
        if config.get('data', {}).get('pos_tags_to_predict', False):
            logger.info("Preparing node classification data for POS tags")
            # This would need to be implemented based on your specific data structure
            # For now, we'll create a placeholder
            node_data = {
                'pos_tags': {
                    'labels': torch.zeros(graph.num_nodes('word')),
                    'mask': torch.zeros(graph.num_nodes('word'), dtype=torch.bool),
                }
            }
        else:
            node_data = None
        
        # Generate negative samples for training
        neg_sampling_ratio = config.get('data', {}).get('negative_sampling_ratio', 5)
        logger.info(f"Generating negative samples with ratio {neg_sampling_ratio}")
        
        negative_samples = {}
        for rel_type in edge_data:
            train_src, train_dst = edge_data[rel_type]['train']
            num_pos_edges = len(train_src)
            
            # Generate negative samples
            # For simplicity, we'll just randomly sample node pairs not in the positive edges
            num_neg_edges = num_pos_edges * neg_sampling_ratio
            neg_src = torch.randint(0, graph.num_nodes('word'), (num_neg_edges,))
            neg_dst = torch.randint(0, graph.num_nodes('word'), (num_neg_edges,))
            
            # Could implement more sophisticated negative sampling here
            
            negative_samples[rel_type] = (neg_src, neg_dst)
        
        return {
            'edge_data': edge_data,
            'node_data': node_data,
            'negative_samples': negative_samples
        }
    
    except Exception as e:
        logger.error(f"Failed to prepare training data: {e}")
        sys.exit(1)

def load_pretrained_model(model_path):
    """
    Load a pre-trained HGMAE model for initializing the encoder.
    
    Args:
        model_path: Path to the pre-trained model checkpoint
        
    Returns:
        Dictionary containing model state dict and configuration
    """
    logger.info(f"Loading pre-trained model from {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Check if this is an HGMAE checkpoint
        if 'encoder_state_dict' in checkpoint:
            logger.info("Found encoder state dict in checkpoint")
            encoder_state_dict = checkpoint['encoder_state_dict']
        else:
            logger.info("Using full model state dict")
            encoder_state_dict = checkpoint.get('model_state_dict')
            
        # Get model configuration
        config = checkpoint.get('config', {})
        
        logger.info(f"Pre-trained model config: {config}")
        
        return {
            'encoder_state_dict': encoder_state_dict,
            'config': config
        }
    except Exception as e:
        logger.error(f"Failed to load pre-trained model: {e}")
        return None

def train_model(graph, features, training_data, config, model_dir, pretrained_model=None, freeze_encoder=False):
    """
    Train the heterogeneous GNN model.
    
    Args:
        graph: The heterogeneous graph
        features: Dictionary of node features
        training_data: Dictionary of training data components
        config: Configuration dictionary
        model_dir: Directory to save the model
        pretrained_model: Path to pre-trained model for initialization
        freeze_encoder: Whether to freeze the encoder weights during fine-tuning
        
    Returns:
        The trained model
    """
    logger.info("Setting up model for training...")
    
    # Get model configuration
    model_config = config.get('model', {})
    
    # Get relation names from graph
    rel_names = graph.etypes
    
    # Create the model
    in_dim = model_config.get('in_dim', 768)
    hidden_dim = model_config.get('hidden_dim', 256)
    out_dim = model_config.get('out_dim', 128)
    num_layers = model_config.get('num_layers', 3)
    num_heads = model_config.get('num_heads', 8)
    num_bases = model_config.get('num_bases', 8)
    dropout = model_config.get('dropout', 0.2)
    residual = model_config.get('residual', True)
    layer_norm = model_config.get('layer_norm', True)
    sparsity = model_config.get('sparsity', 0.8)
    
    # Create model
    model = HeterogeneousGNN(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        rel_names=rel_names,
        num_layers=num_layers,
        num_heads=num_heads,
        num_bases=num_bases,
        dropout=dropout,
        residual=residual,
        layer_norm=layer_norm,
        sparsity=sparsity
    )
    
    # Load pre-trained weights if specified
    if pretrained_model is not None:
        logger.info(f"Initializing from pre-trained model: {pretrained_model}")
        pretrained = load_pretrained_model(pretrained_model)
        
        if pretrained is not None:
            # Load encoder weights
            encoder_state_dict = pretrained['encoder_state_dict']
            
            # Try to load weights, but ignore if shapes don't match
            # This handles cases where the pre-trained model has different dimensions
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in encoder_state_dict.items() 
                              if k in model_dict and v.shape == model_dict[k].shape}
            
            # Report on weights loaded
            logger.info(f"Loading {len(pretrained_dict)} / {len(model_dict)} layers from pre-trained model")
            
            # Update model weights
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
            # Freeze encoder weights if specified
            if freeze_encoder:
                logger.info("Freezing encoder weights")
                for name, param in model.named_parameters():
                    if any(layer_name in name for layer_name in ['encoder', 'gnn_layers', 'local_layers', 'global_layers']):
                        param.requires_grad = False
                        
                # Log number of trainable parameters
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                all_params = sum(p.numel() for p in model.parameters())
                logger.info(f"Trainable parameters: {trainable_params} / {all_params} ({trainable_params / all_params * 100:.2f}%)")
    
    # Additional code would normally be here, keep rest of function as-is

def evaluate_model(model, graph, features, training_data, config):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        graph: The heterogeneous graph
        features: Node features
        training_data: Training data dictionary
        config: Configuration dictionary
        
    Returns:
        Test metrics
    """
    logger.info("Evaluating model on test set...")
    
    try:
        model.eval()
        test_metrics = {}
        device = next(model.parameters()).device
        
        # Evaluate link prediction on test set
        with torch.no_grad():
            for rel_type in training_data['edge_data']:
                test_src, test_dst = training_data['edge_data'][rel_type]['test']
                test_edge_index = torch.stack([test_src, test_dst], dim=0)
                
                # Generate negative samples for test
                num_test_edges = len(test_src)
                test_neg_src = torch.randint(0, graph.num_nodes('word'), (num_test_edges,), device=device)
                test_neg_dst = torch.randint(0, graph.num_nodes('word'), (num_test_edges,), device=device)
                test_neg_edge_index = torch.stack([test_neg_src, test_neg_dst], dim=0)
                
                # Get scores
                test_pos_scores, test_neg_scores = model(
                    features, 
                    test_edge_index, 
                    test_neg_edge_index, 
                    edge_type=training_data['edge_data'][rel_type]['etype']
                )
                
                # Calculate metrics
                metrics = evaluate_link_prediction(
                    test_pos_scores.cpu().numpy(), 
                    test_neg_scores.cpu().numpy(),
                    k_values=[1, 3, 5, 10],  # Calculate Hits@k
                )
                
                # Store metrics
                for metric, value in metrics.items():
                    test_metrics[f"{rel_type}_{metric}"] = value
        
        # Calculate average metrics across all relation types
        avg_metrics = {}
        for metric_base in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'hits@1', 'hits@3', 'hits@5', 'hits@10', 'mrr']:
            metric_values = [v for k, v in test_metrics.items() if k.endswith(f"_{metric_base}")]
            if metric_values:
                avg_metrics[f"avg_{metric_base}"] = sum(metric_values) / len(metric_values)
        
        # Log results
        logger.info("Test results:")
        for metric, value in avg_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Detailed results by relation type
        logger.info("Results by relation type:")
        for rel_type in training_data['edge_data']:
            logger.info(f"  {rel_type}:")
            for metric in ['f1', 'auc', 'mrr']:
                key = f"{rel_type}_{metric}"
                if key in test_metrics:
                    logger.info(f"    {metric}: {test_metrics[key]:.4f}")
        
        # Return combined metrics
        return {**test_metrics, **avg_metrics}
    
    except Exception as e:
        logger.error(f"Failed to evaluate model: {e}")
        return {}

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update logging level if debug mode is enabled
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Load data
    if args.skip_db_load:
        logger.info("Loading data from cache...")
        data = load_data_from_cache(args.data_dir)
    else:
        # Load database configuration
        db_config = load_db_config(args.db_config)
        
        # Load data from database
        target_languages = config.get('data', {}).get('target_languages')
        data = load_data_from_db(db_config, target_languages=target_languages)
        
        # Save data to cache for future use
        save_data_to_cache(data, args.data_dir)
    
    # Build graph
    graph, mappings = build_graph_data(data, config)
    
    # Extract features
    features = extract_features(data, graph, config)
    
    # Prepare training data
    training_data = prepare_training_data(graph, mappings, config)
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Train the model, using pre-trained weights if specified
    model = train_model(
        graph, 
        features, 
        training_data, 
        config, 
        args.model_dir, 
        pretrained_model=args.pretrained_model,
        freeze_encoder=args.freeze_encoder
    )
    
    # Evaluate the model
    results = evaluate_model(model, graph, features, training_data, config)
    
    # Save evaluation results
    results_path = os.path.join(args.model_dir, f"results_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {results_path}")
    
    # Save node mappings for later use
    mappings_path = os.path.join(args.model_dir, f"node_mappings_{timestamp}.json")
    os.makedirs(os.path.dirname(mappings_path), exist_ok=True)
    
    # Convert any non-serializable values in mappings
    serializable_mappings = {}
    for k, v in mappings.items():
        if isinstance(v, dict):
            serializable_mappings[k] = {str(key): value for key, value in v.items()}
        else:
            serializable_mappings[k] = v
    
    with open(mappings_path, 'w') as f:
        json.dump(serializable_mappings, f)
    
    logger.info(f"Node mappings saved to {mappings_path}")
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main() 