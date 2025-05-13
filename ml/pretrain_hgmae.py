#!/usr/bin/env python3

"""
Pre-training script for Heterogeneous Graph Masked Autoencoder (HGMAE).

This script implements self-supervised pre-training of the HGMAE model on 
lexical knowledge graphs derived from Filipino dictionaries, as described in
"Multi-Relational Graph Neural Networks for Automated Knowledge Graph Enhancement
in Low-Resource Philippine Languages".
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

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.data.db_adapter import DatabaseAdapter
from ml.data.lexical_graph_builder import LexicalGraphBuilder
from ml.data.feature_extractors import LexicalFeatureExtractor
from ml.models.hgmae import HGMAE, pretrain_hgmae
from ml.utils.logging_utils import setup_logging

# Set up logging
logger = logging.getLogger(__name__)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"pretraining_{timestamp}.log"
setup_logging(log_file=LOG_FILE, level=logging.INFO)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Pre-train a HGMAE model on Filipino lexical data")
    
    parser.add_argument("--config", type=str, default="config/default_config.json",
                        help="Path to configuration file")
    parser.add_argument("--db-config", type=str, default="my_db_config.json",
                        help="Path to database configuration file")
    parser.add_argument("--model-dir", type=str, default="checkpoints/pretraining",
                        help="Directory to save pre-trained model")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Directory to save processed data")
    parser.add_argument("--skip-db-load", action="store_true",
                        help="Skip loading data from database (use cached data)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of pre-training epochs")
    parser.add_argument("--feature-mask-rate", type=float, default=0.3,
                        help="Ratio of node features to mask")
    parser.add_argument("--edge-mask-rate", type=float, default=0.3,
                        help="Ratio of edges to mask")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--no-metapath-mask", action="store_false", dest="metapath_mask",
                        help="Disable metapath-aware masking")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")
    
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

def pretrain(graph, features, config, args):
    """Pre-train the HGMAE model."""
    logger.info("Setting up HGMAE model for pre-training...")
    
    # Get relation names from graph
    rel_names = graph.etypes
    
    # Get node types from graph
    node_types = graph.ntypes
    
    # Feature dimension
    in_dim = config.get('model', {}).get('in_dim', 768)
    hidden_dim = config.get('model', {}).get('hidden_dim', 256)
    out_dim = config.get('model', {}).get('out_dim', 128)
    
    # HGMAE configuration
    num_layers = config.get('model', {}).get('num_layers', 3)
    num_heads = config.get('model', {}).get('num_heads', 8)
    num_bases = config.get('model', {}).get('num_bases', 8)
    dropout = config.get('model', {}).get('dropout', 0.2)
    residual = config.get('model', {}).get('residual', True)
    layer_norm = config.get('model', {}).get('layer_norm', True)
    sparsity = config.get('model', {}).get('sparsity', 0.8)
    
    # Create the HGMAE model
    model = HGMAE(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        rel_names=rel_names,
        node_types=node_types,
        mask_rate=args.feature_mask_rate,  # Use command-line args for masking
        feature_mask_rate=args.feature_mask_rate,
        edge_mask_rate=args.edge_mask_rate,
        metapath_mask=args.metapath_mask,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        residual=residual,
        layer_norm=layer_norm,
        num_bases=num_bases,
        sparsity=sparsity
    )
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Set up optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Pre-train the model
    logger.info(f"Starting pre-training for {args.epochs} epochs...")
    start_time = time.time()
    
    # Call the pre-training function from the HGMAE module
    model = pretrain_hgmae(
        model=model,
        g=graph,
        feats_dict=features,
        optimizer=optimizer,
        epochs=args.epochs,
        device=device
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Pre-training completed in {elapsed_time:.2f} seconds")
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Save the pre-trained model
    model_path = os.path.join(args.model_dir, f"pretrained_hgmae_{timestamp}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'encoder_state_dict': model.encoder.state_dict(),  # Save just the encoder part for fine-tuning
        'config': {
            'in_dim': in_dim,
            'hidden_dim': hidden_dim,
            'out_dim': out_dim,
            'rel_names': rel_names,
            'node_types': node_types,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'num_bases': num_bases,
            'dropout': dropout,
            'residual': residual,
            'layer_norm': layer_norm,
            'sparsity': sparsity
        },
        'timestamp': timestamp,
        'pre_train_args': {
            'epochs': args.epochs,
            'feature_mask_rate': args.feature_mask_rate,
            'edge_mask_rate': args.edge_mask_rate,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay
        }
    }, model_path)
    
    logger.info(f"Pre-trained model saved to {model_path}")
    
    return model, model_path

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
    
    # Pre-train the model
    model, model_path = pretrain(graph, features, config, args)
    
    # Display success message
    logger.info("Pre-training completed successfully!")
    logger.info(f"Pre-trained model saved to {model_path}")
    
    # Save mappings for later use
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
    
    # Return paths for other scripts to use
    return {
        'model_path': model_path,
        'mappings_path': mappings_path
    }

if __name__ == "__main__":
    main() 