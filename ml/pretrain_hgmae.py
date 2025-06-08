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
from typing import Optional, Set, Dict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.data.db_adapter import DatabaseAdapter
from ml.data.lexical_graph_builder import LexicalGraphBuilder
from ml.data.feature_extraction import LexicalFeatureExtractor
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
    parser.add_argument("--device", type=str, default="cpu", 
                        help="Device to use for training (e.g., 'cuda', 'cpu'). Defaults to CPU for stability.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")
    
    # Added arguments
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs. Overrides config if provided.")
    parser.add_argument("--feature-mask-rate", type=float, default=None,
                        help="Rate for masking features. Overrides config if provided.")
    parser.add_argument("--edge-mask-rate", type=float, default=None,
                        help="Rate for masking edges. Overrides config if provided.")
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from JSON file."""
    logger.info(f"Attempting to load configuration from: {config_path}")
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    if not os.path.isfile(config_path):
        logger.error(f"Configuration path is not a file: {config_path}")
        sys.exit(1)
        
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        if not isinstance(config, dict):
            logger.error(f"Configuration file {config_path} did not load as a dictionary. Loaded type: {type(config)}")
            sys.exit(1)
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at path: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from configuration file {config_path}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading configuration from {config_path}: {e}")
        sys.exit(1)

def load_db_config(config_path):
    """Load database configuration from JSON file."""
    logger.info(f"Attempting to load database configuration from: {config_path}")
    if not os.path.exists(config_path):
        logger.error(f"Database configuration file not found: {config_path}")
        sys.exit(1)
    if not os.path.isfile(config_path):
        logger.error(f"Database configuration path is not a file: {config_path}")
        sys.exit(1)

    try:
        with open(config_path, 'r') as f:
            db_config_full = json.load(f)
        
        if not isinstance(db_config_full, dict):
            logger.error(f"Database configuration file {config_path} did not load as a dictionary. Loaded type: {type(db_config_full)}")
            sys.exit(1)
            
        db_params = db_config_full.get('db_config')
        if db_params is None:
            logger.error(f"Key 'db_config' not found in database configuration file: {config_path}")
            sys.exit(1)
        if not isinstance(db_params, dict):
            logger.error(f"Value of 'db_config' in {config_path} is not a dictionary. Found type: {type(db_params)}")
            sys.exit(1)
            
        logger.info(f"Successfully loaded database parameters from {config_path}")
        return db_params
    except FileNotFoundError:
        logger.error(f"Database configuration file not found at path: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from database configuration file {config_path}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading database configuration from {config_path}: {e}")
        sys.exit(1)

def load_data_from_db(db_config, target_languages=None):
    """Load lexical data from database."""
    logger.info("Loading data from database...")
    
    # Initialize data dictionary with empty DataFrames for all expected keys
    data_frames = {
        'lemmas_df': pd.DataFrame(),
        'relations_df': pd.DataFrame(),
        'definitions_df': pd.DataFrame(),
        'etymologies_df': pd.DataFrame(),
        'pronunciations_df': pd.DataFrame(),
        'word_forms_df': pd.DataFrame(),
        'pos_df': pd.DataFrame() # Added for parts_of_speech data
    }
    
    try:
        # Connect to database
        db_adapter = DatabaseAdapter(db_config)
        
        # Load mandatory data frames
        data_frames['lemmas_df'] = db_adapter.get_lemmas_df(target_languages=target_languages)
        logger.info(f"Loaded {len(data_frames['lemmas_df'])} lemmas")
        if data_frames['lemmas_df'].empty:
            logger.error("CRITICAL: No lemmas loaded from the database. Cannot proceed.")
            sys.exit(1)
        
        data_frames['relations_df'] = db_adapter.get_relations_df()
        logger.info(f"Loaded {len(data_frames['relations_df'])} relations")
        if data_frames['relations_df'].empty:
            logger.warning("No relations loaded from the database.")
        
        data_frames['definitions_df'] = db_adapter.get_definitions_df()
        logger.info(f"Loaded {len(data_frames['definitions_df'])} definitions")
        if data_frames['definitions_df'].empty:
            logger.warning("No definitions loaded from the database.")
            
        data_frames['etymologies_df'] = db_adapter.get_etymologies_df()
        logger.info(f"Loaded {len(data_frames['etymologies_df'])} etymologies")
        # Not critical if empty, can be a warning
        if data_frames['etymologies_df'].empty:
            logger.info("No etymologies loaded (this might be expected).")

        # Load parts_of_speech data
        try:
            data_frames['pos_df'] = db_adapter.get_pos_df()
            logger.info(f"Loaded {len(data_frames['pos_df'])} parts of speech entries")
            if data_frames['pos_df'].empty:
                logger.warning("No parts of speech data loaded. This might affect POS tagging features.")
        except Exception as e_pos:
            logger.warning(f"Failed to load parts of speech data, continuing without it. Error: {e_pos}")
            # data_frames['pos_df'] remains an empty DataFrame as initialized

        # Optional data frames - attempt to load, but don't fail if they are missing
        try:
            data_frames['pronunciations_df'] = db_adapter.get_pronunciations_df()
            logger.info(f"Loaded {len(data_frames['pronunciations_df'])} pronunciations")
        except Exception as e_pron:
            logger.warning(f"Failed to load pronunciations, continuing without them. Error: {e_pron}")
            # data_frames['pronunciations_df'] remains an empty DataFrame as initialized
        
        try:
            data_frames['word_forms_df'] = db_adapter.get_word_forms_df()
            logger.info(f"Loaded {len(data_frames['word_forms_df'])} word forms")
        except Exception as e_wf:
            logger.warning(f"Failed to load word forms, continuing without them. Error: {e_wf}")
            # data_frames['word_forms_df'] remains an empty DataFrame as initialized
        
        return data_frames
    
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading data from database: {e}")
        # Ensure a consistent return type (dict of empty DFs) even on catastrophic failure before exiting
        # This helps if any part of the calling code tries to iterate over keys before checking sys.exit status.
        # However, the sys.exit(1) below will terminate the script.
        logger.info("Exiting due to database loading failure.")
        sys.exit(1)

def save_data_to_cache(data, data_dir):
    """Save loaded data to disk cache."""
    logger.info(f"Attempting to save data to cache directory: {data_dir}")
    try:
        if not os.path.exists(data_dir):
            logger.info(f"Cache directory {data_dir} does not exist. Creating it.")
        os.makedirs(data_dir, exist_ok=True)
        elif not os.path.isdir(data_dir):
            logger.error(f"Cache path {data_dir} exists but is not a directory. Cannot save cache.")
            return # Do not exit, allow script to continue if caching is optional
        
        for name, df in data.items():
            if not isinstance(df, pd.DataFrame):
                logger.warning(f"Item '{name}' in data is not a DataFrame (type: {type(df)}). Skipping caching for this item.")
                continue
            file_path = os.path.join(data_dir, f"{name}.parquet")
            try:
            df.to_parquet(file_path, index=False)
                logger.info(f"Saved {name} ({len(df)} rows) to {file_path}")
            except Exception as e_df_save:
                logger.error(f"Failed to save DataFrame '{name}' to {file_path}: {e_df_save}")
        
        logger.info(f"Data caching process to {data_dir} finished.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during save_data_to_cache: {e}")
        # Optionally, allow script to continue if caching fails

def load_data_from_cache(data_dir, critical_dataframes=None):
    """Load data from disk cache."""
    logger.info(f"Attempting to load data from cache directory: {data_dir}")
    if critical_dataframes is None:
        critical_dataframes = ['lemmas_df'] # Define which dataframes are critical for script continuation
        
    data = {}
    expected_data_names = [
        'lemmas_df', 'relations_df', 'definitions_df', 'etymologies_df', 
        'pronunciations_df', 'word_forms_df', 'pos_df' # Added pos_df
    ]

    if not os.path.exists(data_dir):
        logger.error(f"Cache directory {data_dir} not found. Cannot load from cache.")
        # If critical dataframes were expected from cache, this is a fatal error.
        if any(crit_df in expected_data_names for crit_df in critical_dataframes):
             logger.error("Critical data was expected from cache, but cache directory does not exist. Exiting.")
             sys.exit(1)
        # Initialize all expected dataframes as empty if cache dir doesn't exist and no critical data was strictly required from it
        for name in expected_data_names:
            data[name] = pd.DataFrame()
        return data
        
    if not os.path.isdir(data_dir):
        logger.error(f"Cache path {data_dir} is not a directory. Cannot load from cache.")
        if any(crit_df in expected_data_names for crit_df in critical_dataframes):
             logger.error("Critical data was expected from cache, but cache path is not a directory. Exiting.")
             sys.exit(1)
        for name in expected_data_names:
            data[name] = pd.DataFrame()
        return data

    try:
        for name in expected_data_names:
            file_path = os.path.join(data_dir, f"{name}.parquet")
            if os.path.exists(file_path):
                if not os.path.isfile(file_path):
                    logger.warning(f"Cache path {file_path} for '{name}' exists but is not a file. Skipping.")
                    data[name] = pd.DataFrame()
                    if name in critical_dataframes:
                        logger.error(f"Critical cache file {file_path} is not a file. Exiting.")
                        sys.exit(1)
                    continue
                try:
                data[name] = pd.read_parquet(file_path)
                logger.info(f"Loaded {name} from {file_path}: {len(data[name])} rows")
                except Exception as e_df_load:
                    logger.error(f"Failed to load DataFrame '{name}' from {file_path}: {e_df_load}")
                    data[name] = pd.DataFrame() # Ensure key exists with empty DF
                    if name in critical_dataframes:
                        logger.error(f"Failed to load critical cache file {file_path}. Exiting.")
                        sys.exit(1)
            else:
                logger.warning(f"Cache file {file_path} for '{name}' not found.")
                data[name] = pd.DataFrame() # Ensure key exists with empty DF
                if name in critical_dataframes:
                    logger.error(f"Critical cache file {file_path} for '{name}' not found. Exiting.")
                    sys.exit(1)
        
        logger.info(f"Data loading from cache directory {data_dir} finished.")
        return data
    except Exception as e:
        logger.error(f"An unexpected error occurred during load_data_from_cache: {e}")
        # Fallback: return dict of empty DFs, but exit if critical data was truly expected
        for name in expected_data_names:
            if name not in data: # Ensure all keys exist
                data[name] = pd.DataFrame()
        if any(crit_df in expected_data_names and data.get(crit_df, pd.DataFrame()).empty for crit_df in critical_dataframes):
            logger.error("Exiting due to failure in loading critical data from cache.")
        sys.exit(1)
        return data

def build_graph_data(data, config):
    """Build heterogeneous graph from dataframes."""
    logger.info("Attempting to build heterogeneous graph...")
    
    if not isinstance(data, dict):
        logger.error(f"Input 'data' to build_graph_data is not a dictionary (type: {type(data)}). Cannot proceed.")
        sys.exit(1)
        
    critical_dfs = ['lemmas_df', 'relations_df', 'definitions_df'] # Etymologies can be optional for graph structure
    for df_name in critical_dfs:
        if df_name not in data:
            logger.error(f"Missing critical DataFrame '{df_name}' in input data for graph building. Cannot proceed.")
            sys.exit(1)
        if not isinstance(data[df_name], pd.DataFrame):
            logger.error(f"Input '{df_name}' is not a DataFrame (type: {type(data[df_name])}). Cannot proceed.")
            sys.exit(1)
            
    if data['lemmas_df'].empty:
        logger.error("CRITICAL: lemmas_df is empty. Cannot build graph without lemmas. Exiting.")
        sys.exit(1)
    
    try:
        target_languages = config.get('data', {}).get('target_languages')
        if target_languages:
            logger.info(f"Filtering data to target languages for graph building: {target_languages}")
        
        # Create graph builder
        graph_builder = LexicalGraphBuilder(config, target_languages=target_languages)
        
        # Build graph
        # Ensure all expected DFs are passed, even if empty (GraphBuilder should handle empty optional DFs)
        graph = graph_builder.build_graph(
            lemmas_df=data.get('lemmas_df', pd.DataFrame()),
            relations_df=data.get('relations_df', pd.DataFrame()),
            definitions_df=data.get('definitions_df', pd.DataFrame()),
            etymologies_df=data.get('etymologies_df', pd.DataFrame()),
            pronunciations_df=data.get('pronunciations_df', pd.DataFrame()),
            word_forms_df=data.get('word_forms_df', pd.DataFrame()),
        )
        
        if graph is None:
            logger.error("Graph building failed and returned None. Exiting.")
            sys.exit(1)
            
        if not isinstance(graph, dgl.DGLGraph):
            logger.error(f"Graph building did not return a DGLGraph object (returned type: {type(graph)}). Exiting.")
            sys.exit(1)
            
        logger.info(f"Graph built: {graph}")
        if graph.num_nodes() == 0:
            logger.error("Graph built successfully but contains no nodes. This usually indicates issues with input data or filtering. Exiting.")
            sys.exit(1)
        
        # Get node mappings
        mappings = graph_builder.get_node_mappings()
        if not isinstance(mappings, dict):
            logger.warning(f"Node mappings from graph_builder is not a dictionary (type: {type(mappings)}). This might be an issue.")
        
        logger.info("Successfully built heterogeneous graph and retrieved mappings.")
        return graph, mappings
    
    except Exception as e:
        logger.error(f"An unexpected error occurred during build_graph_data: {e}", exc_info=True)
        sys.exit(1)

def extract_features(data, graph, config, 
                     node_to_original_id_maps_for_lfe: Optional[Dict[str, Dict[int, int]]] = None,
                     relevant_word_ids=None, 
                     relevant_definition_ids=None, 
                     relevant_etymology_ids=None):
    """Extract features for graph nodes."""
    logger.info("Extracting features...")
    data_config = config.get('data', {})
    # Get the feature_extraction specific config
    feature_extraction_config = config.get('feature_extraction', {}) 
    sentence_transformer_name = feature_extraction_config.get('sentence_transformer_model_name')

    # Initialize feature extractor with settings from the 'data' section of the config
    # Also pass model_original_feat_dims if available in config for placeholder creation
    model_original_feat_dims = config.get('model', {}).get('original_feat_dims', {})
    logger.info(f"Instantiating LexicalFeatureExtractor with model_original_feat_dims: {model_original_feat_dims}")

        feature_extractor = LexicalFeatureExtractor(
        use_xlmr=data_config.get('use_xlmr', True),
        use_transformer_embeddings=data_config.get('use_transformer_embeddings', True),
        sentence_transformer_model_name=sentence_transformer_name,  # Pass the model name
        use_char_ngrams=data_config.get('use_char_ngrams', True),
        use_phonetic_features=data_config.get('use_phonetic_features', True),
        use_etymology_features=data_config.get('use_etymology_features', True),
        use_baybayin_features=data_config.get('use_baybayin_features', True),
        normalize_features=data_config.get('normalize_features', True),
        model_original_feat_dims=model_original_feat_dims # Pass this to LFE
    )
    
    graph_num_nodes_per_type = {ntype: graph.num_nodes(ntype) for ntype in graph.ntypes}
    logger.info(f"Graph node counts for feature extraction: {graph_num_nodes_per_type}")

    # Debug: Log information about inputs to LFE
    logger.info(f"DEBUG_LFE_INPUT: relevant_word_ids count: {len(relevant_word_ids) if relevant_word_ids else 'None'}")
    logger.info(f"DEBUG_LFE_INPUT: relevant_definition_ids count: {len(relevant_definition_ids) if relevant_definition_ids else 'None'}")
    logger.info(f"DEBUG_LFE_INPUT: relevant_etymology_ids count: {len(relevant_etymology_ids) if relevant_etymology_ids else 'None'}")

    if node_to_original_id_maps_for_lfe:
        for ntype, id_map in node_to_original_id_maps_for_lfe.items():
            logger.info(f"DEBUG_LFE_INPUT: node_to_original_id_maps_for_lfe for '{ntype}' type: {'Present' if id_map is not None else 'None or Empty'}")
            if id_map and isinstance(id_map, dict) and len(id_map) > 0:
                 # Log a small sample of the map {graph_idx: db_id}
                 sample_items = list(id_map.items())[:5] # Get first 5 items
                 logger.info(f"DEBUG_LFE_INPUT: Sample node_to_original_id_maps_for_lfe['{ntype}'] (local graph idx -> orig DB id): {dict(sample_items)}")
    else:
        logger.warning("DEBUG_LFE_INPUT: node_to_original_id_maps_for_lfe is None or empty.")


    features_dict, ordered_ids_map = feature_extractor.extract_all_features(
        lemmas_df=data.get('lemmas_df', pd.DataFrame()),
        graph_num_nodes_per_type=graph_num_nodes_per_type,
        node_to_original_id_maps=node_to_original_id_maps_for_lfe, # Pass the mapping here
        definitions_df=data.get('definitions_df', pd.DataFrame()),
        etymologies_df=data.get('etymologies_df', pd.DataFrame()), # Pass etymologies_df if available
        pronunciations_df=data.get('pronunciations_df', pd.DataFrame()),
        word_forms_df=data.get('word_forms_df', pd.DataFrame()),
        relevant_word_ids=relevant_word_ids,
        relevant_definition_ids=relevant_definition_ids,
        relevant_etymology_ids=relevant_etymology_ids # Pass relevant_etymology_ids for 'etymology' node type
    )
    
    if not isinstance(features_dict, dict):
        logger.error(f"Feature extraction did not return a dictionary for features_dict (type: {type(features_dict)}). Exiting.")
        sys.exit(1)
    # ordered_ids_map is also returned, can be used for deeper validation if needed but not directly by HGMAE model loop here.
    logger.info(f"LFE returned ordered_ids_map with keys: {list(ordered_ids_map.keys()) if isinstance(ordered_ids_map, dict) else 'Invalid type'}")

    # Critical validation of features_dict against graph structure
    for ntype in graph.ntypes:
        num_nodes_of_type = graph.num_nodes(ntype)
        if num_nodes_of_type == 0:
            if ntype in features_dict and features_dict[ntype].nelement() > 0:
                logger.warning(f"Node type '{ntype}' has 0 nodes in graph, but features_dict[{ntype}] is not empty (shape: {features_dict[ntype].shape}). This is unexpected.")
            continue

        if ntype not in features_dict:
            logger.error(f"CRITICAL: features_dict is missing node type '{ntype}', but graph has {num_nodes_of_type} nodes of this type. Placeholder creation in LFE likely failed. Exiting.")
            sys.exit(1)
        
        feature_tensor = features_dict[ntype]
        if not isinstance(feature_tensor, torch.Tensor):
            logger.error(f"CRITICAL: features_dict['{ntype}'] is not a torch.Tensor (type: {type(feature_tensor)}). Exiting.")
            sys.exit(1)
        
        if feature_tensor.shape[0] != num_nodes_of_type:
            logger.error(f"CRITICAL ROW MISMATCH for '{ntype}': features_dict has {feature_tensor.shape[0]} rows, graph expects {num_nodes_of_type} nodes. Exiting.")
        sys.exit(1)
        
        logger.info(f"Validated features for '{ntype}': shape {feature_tensor.shape}, matches graph node count {num_nodes_of_type}.")

    logger.info("Successfully validated features_dict against graph structure.")
    return features_dict # Main training loop expects only the features dictionary

def pretrain(graph, features, config, args):
    """Instantiate and train the HGMAE model, then save it."""
    logger.info("Starting pretrain function (script level)...")
    try:
        if not isinstance(graph, dgl.DGLGraph): # dgl.heterograph.DGLHeteroGraph ?
            logger.error(f"Invalid graph object passed to pretrain (type: {type(graph)}). Exiting.")
            sys.exit(1)
        if graph.num_nodes() == 0:
            logger.error("Graph passed to pretrain has no nodes. Exiting.")
            sys.exit(1)
        if not isinstance(features, dict):
            logger.error(f"Invalid features object passed to pretrain (type: {type(features)}). Exiting.")
            sys.exit(1)
        if not features and graph.ntypes: # if there are node types, we expect some features
            logger.warning("Features dictionary passed to pretrain is empty, but graph has node types.")

        default_epochs = 15 
        
        if 'model' not in config or not isinstance(config['model'], dict):
            logger.error("'model' configuration is missing or not a dictionary. Exiting.")
            sys.exit(1)
        if 'training' not in config or not isinstance(config['training'], dict):
            logger.error("'training' configuration is missing or not a dictionary. Exiting.")
            sys.exit(1)
            
        model_params = config['model']
        train_params = config['training']
        
        hidden_dim_original = model_params.get('hidden_dim', 256)
        if not isinstance(hidden_dim_original, int) or hidden_dim_original <= 0:
            logger.error(f"Invalid 'hidden_dim' ({hidden_dim_original}). Must be a positive integer. Exiting.")
            sys.exit(1)
        hidden_dim_reduced = min(128, hidden_dim_original)
        
        out_dim_original = model_params.get('out_dim', 768) # This seems high for an out_dim if hidden is 128/256
        if not isinstance(out_dim_original, int) or out_dim_original <= 0:
            logger.error(f"Invalid 'out_dim' ({out_dim_original}). Must be a positive integer. Exiting.")
            sys.exit(1)
        # The 'out_dim' for HGMAE typically matches hidden_dim or is related to decoder output, not necessarily large like 768
        # If HGMAE's internal GNN output dim is hidden_dim_reduced, then 'out_dim' for HGMAE might also be hidden_dim_reduced.
        # Let's assume out_dim in config refers to the GNN's internal feature dimension before reconstruction.
        out_dim_reduced = hidden_dim_reduced # For consistency, GNN output before reconstruction will be hidden_dim_reduced

        num_layers_original = model_params.get('num_layers', 3)
        if not isinstance(num_layers_original, int) or num_layers_original <= 0:
            logger.error(f"Invalid 'num_layers' ({num_layers_original}). Must be a positive integer. Exiting.")
            sys.exit(1)
        num_layers_reduced = min(2, num_layers_original)
        
        reduced_parameters_log = {
            'hidden_dim_gnn': f"{hidden_dim_reduced} (original config: {hidden_dim_original})",
            'out_dim_gnn_final_projection': f"{out_dim_reduced} (tied to hidden_dim_gnn for this run)",
            'num_layers_gnn': f"{num_layers_reduced} (original config: {num_layers_original})"
        }
        logger.info(f"Using model parameters for pretraining (GNN part of HGMAE): {reduced_parameters_log}")
        
        # Prioritize command-line args for epochs, feature_mask_rate, edge_mask_rate
        epochs_from_config = train_params.get('epochs', default_epochs)
        epochs = args.epochs if args.epochs is not None else epochs_from_config
        if args.epochs is not None:
            logger.info(f"Using command-line --epochs {args.epochs} (overriding config value {epochs_from_config if epochs_from_config != default_epochs else 'default '+str(default_epochs)})")
        if not isinstance(epochs, int) or epochs <= 0:
            logger.warning(f"Invalid 'epochs' value ({epochs}). Must be a positive integer. Using default from config or hardcoded: {epochs_from_config if epochs_from_config != default_epochs else default_epochs}")
            epochs = epochs_from_config if epochs_from_config != default_epochs else default_epochs
            
        lr = train_params.get('learning_rate', 0.0001) # Adjusted default as per summary (1e-4)
        if not isinstance(lr, float) or lr <= 0:
            logger.warning(f"Invalid 'learning_rate' ({lr}). Must be a positive float. Using default: 0.0001")
            lr = 0.0001
            
        weight_decay = train_params.get('weight_decay', 1e-5)
        if not isinstance(weight_decay, float) or weight_decay < 0:
            logger.warning(f"Invalid 'weight_decay' ({weight_decay}). Must be a non-negative float. Using default: 1e-5")
            weight_decay = 1e-5
        
        patience_val = train_params.get('patience', 5)
        if not isinstance(patience_val, int) or patience_val <= 0:
            logger.warning(f"Invalid 'patience' ({patience_val}). Must be positive. Using default: 5")
            patience_val = 5

        # New: Configure early stopping metric
        early_stopping_metric_config = train_params.get('early_stopping_metric', 'total_loss')
        if early_stopping_metric_config not in ['total_loss', 'feature_mse']:
            logger.warning(f"Invalid 'early_stopping_metric' ({early_stopping_metric_config}). Must be 'total_loss' or 'feature_mse'. Defaulting to 'total_loss'.")
            early_stopping_metric_config = 'total_loss'
        logger.info(f"Configured early stopping metric: {early_stopping_metric_config}")

        gradient_clip_norm_val = train_params.get('gradient_clip_norm', 1.0)
        if not isinstance(gradient_clip_norm_val, (float, int)) or gradient_clip_norm_val <= 0:
            logger.warning(f"Invalid 'gradient_clip_norm' ({gradient_clip_norm_val}). Must be a positive float/int. Using default: 1.0")
            gradient_clip_norm_val = 1.0

        use_warmup = train_params.get('use_warmup', True)
        warmup_epochs = train_params.get('warmup_epochs', max(1, epochs // 10)) # e.g. 10% of epochs, min 1
        if not isinstance(warmup_epochs, int) or warmup_epochs < 0:
            logger.warning(f"Invalid 'warmup_epochs' ({warmup_epochs}). Using default: {max(1, epochs // 10)} if use_warmup is True.")
            warmup_epochs = max(1, epochs // 10)
        if warmup_epochs >= epochs:
            logger.warning(f"Warmup epochs ({warmup_epochs}) >= total epochs ({epochs}). Disabling warmup.")
            use_warmup = False
            warmup_epochs = 0
        
        rel_names = [etype[1] for etype in graph.canonical_etypes] # etype is (src_type, rel_name, dst_type)
        if not rel_names:
            logger.warning("No relation names (canonical etypes) found in the graph. R-GCN components might not work as expected if they rely on distinct relation names.")
        
    node_types = graph.ntypes
        if not node_types:
            logger.error("Graph has no node types (graph.ntypes is empty). Exiting.")
            sys.exit(1)
        
        feat_dims = {}
        for ntype in node_types:
            if ntype in features:
                if not isinstance(features[ntype], torch.Tensor):
                    logger.error(f"Features for node type '{ntype}' are not a torch.Tensor (type: {type(features[ntype])}). Exiting.")
                    sys.exit(1)
                feat_dims[ntype] = features[ntype].shape[1]
                if features[ntype].shape[0] != graph.num_nodes(ntype):
                     logger.error(f"Mismatch for {ntype}: features rows {features[ntype].shape[0]} vs graph nodes {graph.num_nodes(ntype)}. Exiting.")
                     sys.exit(1)
            else:
                # If a node type from graph.ntypes has no features, it means LexicalFeatureExtractor didn't produce any for it.
                # HGMAE's original_feat_dims needs an entry for all node types it will encounter in the graph.
                # If a node type has no actual features, its feature dimension is 0.
                # The decoders in HGMAE will attempt to reconstruct to this original_feat_dims.
                logger.warning(f"Node type '{ntype}' is in graph.ntypes but has no entry in the features dictionary. Assuming 0-dim features for this type.")
                feat_dims[ntype] = 0 

        if not feat_dims and node_types: # If there are node types, we must have feat_dims for them
             logger.error("Feature dimensions (feat_dims) could not be determined for any node type, yet graph has node types. Exiting.")
             sys.exit(1)
        elif not node_types and feat_dims: # Should not happen if graph has no node types
             logger.warning("feat_dims were derived but graph has no node_types. This is unusual.")


        logger.info(f"Determined feature dimensions for HGMAE instantiation: {feat_dims}")

        model = None
        try:
            # Parameters from args override config if they exist (e.g. feature_mask_rate)
            # For this script, we use model_params from config directly.
            # The args like feature_mask_rate, edge_mask_rate were removed from parse_args earlier
            # as they are better suited to be in the config['model'] or config['training'] sections.
            # Let's assume they are in model_params:
            feature_m_rate_config = model_params.get('feature_mask_rate', 0.3) # Default from summary
            edge_m_rate_config = model_params.get('edge_mask_rate', 0.3) # Default from summary

            feature_m_rate = args.feature_mask_rate if args.feature_mask_rate is not None else feature_m_rate_config
            edge_m_rate = args.edge_mask_rate if args.edge_mask_rate is not None else edge_m_rate_config

            if args.feature_mask_rate is not None:
                logger.info(f"Using command-line --feature-mask-rate {args.feature_mask_rate} (overriding config value {feature_m_rate_config})")
            if args.edge_mask_rate is not None:
                logger.info(f"Using command-line --edge-mask-rate {args.edge_mask_rate} (overriding config value {edge_m_rate_config})")

            # Get pretrain loss weights from model config
            pretrain_feat_loss_weight = model_params.get('pretrain_feat_loss_weight', 0.7) # Default to current behavior
            pretrain_edge_loss_weight = model_params.get('pretrain_edge_loss_weight', 0.3) # Default to current behavior
            logger.info(f"Using pretrain loss weights: Feature={pretrain_feat_loss_weight}, Edge={pretrain_edge_loss_weight}")

            # Calculate num_bases_to_use
            num_bases_config = model_params.get('num_bases', 2) 
            if rel_names:
                num_bases_to_use = min(num_bases_config, len(rel_names))
                if num_bases_to_use <= 0: # Ensure it's positive if rel_names exist but min resulted in <=0
                    logger.warning(f"num_bases calculation resulted in {num_bases_to_use} with {len(rel_names)} relations. Setting to 1.")
                    num_bases_to_use = 1 
            else:
                logger.warning("No relation names (rel_names) for HGMAE. Using num_bases = num_bases_config (default or from config).")
                num_bases_to_use = num_bases_config
                if num_bases_to_use <= 0: # Ensure it's positive
                    logger.warning(f"num_bases_config is {num_bases_to_use}. Setting to 2 as a safe default for no relations case.")
                    num_bases_to_use = 2

            logger.info(f"Sanity check before HGMAE instantiation: hidden_dim_reduced type: {type(hidden_dim_reduced)}, value: {hidden_dim_reduced}")
            logger.info(f"Sanity check before HGMAE instantiation: out_dim_reduced type: {type(out_dim_reduced)}, value: {out_dim_reduced}")
            logger.info(f"Sanity check before HGMAE instantiation: feat_dims type: {type(feat_dims)}, value: {feat_dims}")

    model = HGMAE(
                in_dim=feat_dims,
                hidden_dim=hidden_dim_reduced, # Explicitly use the integer hidden_dim_reduced
                out_dim=out_dim_reduced,       # Explicitly use the integer out_dim_reduced
        rel_names=rel_names,
                original_feat_dims=feat_dims, 
        node_types=node_types,
                # mask_rate is a general rate, feature_mask_rate and edge_mask_rate are specific
                mask_rate=model_params.get('mask_rate', 0.3), # General, might not be used if specific ones are
                feature_mask_rate=feature_m_rate,
                edge_mask_rate=edge_m_rate,
                num_layers=num_layers_reduced, 
                num_heads=model_params.get('num_heads', 4), # Reduced default from 8 for CPU
                dropout=model_params.get('dropout', 0.1),
                residual=model_params.get('residual', True),
                layer_norm=model_params.get('layer_norm', True),
                num_bases=num_bases_to_use, # Use calculated num_bases_to_use
                pretrain_feat_loss_weight=pretrain_feat_loss_weight,
                pretrain_edge_loss_weight=pretrain_edge_loss_weight
            )
            logger.info("HGMAE model instantiated successfully.")
        except Exception as e_model_init:
            logger.error(f"Failed to instantiate HGMAE model: {e_model_init}", exc_info=True)
            sys.exit(1)
        
        requested_device_str = args.device
        if not isinstance(requested_device_str, str):
            logger.warning(f"args.device is not a string ({type(requested_device_str)}). Defaulting to 'cpu'.")
            requested_device_str = "cpu"
            
        actual_training_device_str = args.device # Use the device specified in arguments
        logger.info(f"Requested device: {requested_device_str}. Using this device for pretrain_hgmae call: {actual_training_device_str}.")
        
        # Move model to device BEFORE optimizer creation and passing to training loop
        try:
            target_device_torch = torch.device(actual_training_device_str)
            model = model.to(target_device_torch)
            logger.info(f"Moved HGMAE model to device: {target_device_torch}")
        except Exception as e_model_to_device:
            logger.error(f"Failed to move HGMAE model to device {actual_training_device_str}: {e_model_to_device}", exc_info=True)
            sys.exit(1)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Initialize schedulers: one for warmup, one for cosine annealing
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs - warmup_epochs if use_warmup and epochs > warmup_epochs else epochs, # Adjust T_max if warmup is used
            eta_min=lr * 0.01 
        )
        
        if use_warmup and warmup_epochs > 0 and epochs > warmup_epochs:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs]
            )
        else:
            scheduler = main_scheduler # Use only cosine annealing if no warmup or warmup covers all epochs
            if use_warmup and warmup_epochs > 0 : # Log if warmup was skipped due to epoch counts
                 logger.info("Warmup configured but warmup_epochs >= epochs or warmup_epochs is 0. Using only CosineAnnealingLR.")


        model_dir_path = Path(args.model_dir)
        try:
            model_dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured model save directory exists: {model_dir_path}")
        except Exception as e_mkdir:
            logger.error(f"Could not create model directory {args.model_dir}: {e_mkdir}. Exiting")
            sys.exit(1)
        
        # Using the main script's timestamp for the log file for this run.
        # For model, use a new timestamp to avoid collision if script is run multiple times quickly.
        model_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"hgmae_pretrain_{model_timestamp}.pt"
        model_save_path = model_dir_path / model_filename
        
        trained_model = None
        actual_epochs_run = epochs # Placeholder, ideally updated by pretrain_hgmae if it returns it
        train_start_time = time.time()
        try:
            logger.info(f"Starting core training loop (ml.models.hgmae.pretrain_hgmae) for up to {epochs} epochs on device '{actual_training_device_str}'. Patience: {patience_val}")
            
            # Call the pretrain_hgmae from ml.models.hgmae
            # This function should handle the actual training loop, early stopping, and model.to(device)
            # It now returns additional validation metric histories
            (trained_model, actual_epochs_run, 
             history_total, history_feat, history_edge, history_lr,
             history_val_total, history_val_feat_mse, 
             history_val_feat_cos_sim, history_val_edge_acc) = pretrain_hgmae(
        model=model,
        g=graph,
        feats_dict=features,
        optimizer=optimizer,
                epochs=epochs,
                device=actual_training_device_str, 
                patience=patience_val,
                scheduler=scheduler,
                gradient_clip_norm=gradient_clip_norm_val,
                early_stopping_metric=early_stopping_metric_config
            )
            logger.info(f"Core training loop (ml.models.hgmae.pretrain_hgmae) completed after {actual_epochs_run} epochs.")
            training_histories = {
                "total_loss": history_total,
                "feature_loss": history_feat,
                "edge_loss": history_edge,
                "learning_rate": history_lr, # Added learning rate history
                "validation_total_loss": history_val_total,
                "validation_feature_mse": history_val_feat_mse,
                "validation_feature_cosine_similarity": history_val_feat_cos_sim,
                "validation_edge_accuracy": history_val_edge_acc
            }
        except Exception as e_train_loop:
            logger.error(f"Error during the core pretrain_hgmae training loop: {e_train_loop}", exc_info=True)
            sys.exit(1) # Critical failure in training
            
        train_duration = time.time() - train_start_time
        logger.info(f"Total core training time: {train_duration:.2f} seconds ({train_duration/60:.2f} minutes).")
        
        if trained_model is None:
            logger.error("Training loop finished but returned no model. Exiting.")
            sys.exit(1)

        logger.info(f"Preparing to save final model to: {model_save_path}")
        try:
            # Consolidate model construction parameters actually used
            hgmae_constructor_params = {
                'in_dim': feat_dims,
                'hidden_dim': hidden_dim_reduced,
                'out_dim': out_dim_reduced, 
                'num_layers': num_layers_reduced,
                'original_feat_dims': feat_dims, # Crucial
                'node_types': node_types,       # Crucial
                'rel_names': rel_names,         # Crucial
                'mask_rate': model_params.get('mask_rate', 0.3),
                'feature_mask_rate': feature_m_rate,
                'edge_mask_rate': edge_m_rate,
                'num_heads': model.gnn.gnn_layers[0].att.num_heads if hasattr(model, 'gnn') and model.gnn.gnn_layers and hasattr(model.gnn.gnn_layers[0], 'att') else model_params.get('num_heads', 4), # Get actual if possible
                'dropout': model_params.get('dropout', 0.1),
                'residual': model_params.get('residual', True),
                'layer_norm': model_params.get('layer_norm', True),
                'num_bases': model.gnn.input_projs['word'].conv.num_bases if hasattr(model, 'gnn') and 'word' in model.gnn.input_projs and hasattr(model.gnn.input_projs['word'], 'conv') else model_params.get('num_bases', 8), # Get actual if possible
            }

    torch.save({
                'model_state_dict': trained_model.state_dict(),
                'config_full_original': config, # Full original config used for the run
                'args_original': vars(args),   # Command line args used for the run
                'model_constructor_params_used': hgmae_constructor_params,
                'training_params_used': {
                    'epochs_requested': epochs,
                    'epochs_run': actual_epochs_run, # Actual epochs run
                    'learning_rate_initial': lr,
                    'weight_decay': weight_decay,
                    'patience_setting': patience_val,
                    'warmup_epochs_setting': warmup_epochs if use_warmup else 0,
                    'final_learning_rate': optimizer.param_groups[0]['lr'] # LR at the end of training
                },
                'training_histories': training_histories, # Add full histories here
                'notes': 'Pretrained HGMAE model. Parameters might have been reduced from config for this CPU pretraining run.',
                'parameter_reduction_details': reduced_parameters_log,
                'saved_timestamp': model_timestamp
            }, model_save_path)
            logger.info(f"Model and metadata saved successfully to {model_save_path}")
        except Exception as e_save:
            logger.error(f"Failed to save model to {model_save_path}: {e_save}", exc_info=True)
            # Do not exit, training completed, but log error.
            
        return trained_model, model_save_path, actual_epochs_run, training_histories

    except Exception as e_pretrain_func:
        logger.error(f"An unexpected error occurred within the script-level pretrain function: {e_pretrain_func}", exc_info=True)
        sys.exit(1) # Exit if the setup for pretrain_hgmae fails


def main():
    """Main function to orchestrate the pre-training pipeline."""
    args = parse_args()
    
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_level = logging.DEBUG if args.debug else logging.INFO
    
    # Ensure model_dir (used for log saving here) exists
    log_save_dir = Path(args.model_dir) # Using model_dir for logs for this script
    try:
        log_save_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e_mkdir_log:
        # If log dir can't be made, we can't save file log, but can continue with console logging.
        print(f"Warning: Could not create log directory {log_save_dir}: {e_mkdir_log}. Log file will not be saved.")
        log_file_path = None # Fallback to no file logging
    else:
        log_file_path = log_save_dir / f"pretrain_script_{run_timestamp}.log"

    # Setup logging once here
    setup_logging(log_file=str(log_file_path) if log_file_path else None, level=log_level)
    if log_file_path:
        logger.info(f"Logging to file: {log_file_path}")
    else:
        logger.info("File logging is disabled as log directory creation failed.")

    logger.info(f"Starting HGMAE pre-training script with args: {args}")
    
    config = load_config(args.config)
    # Corrected: Use load_db_config for database parameters
    db_params = load_db_config(args.db_config) 

    data_cache_dir = Path(args.data_dir)
    lemmas_cache_file = data_cache_dir / 'lemmas_df.parquet'

    if args.skip_db_load and lemmas_cache_file.exists():
        logger.info(f"Skipping database load. Attempting to load all data from cache: {data_cache_dir}")
        # Define critical dataframes for cache loading phase
        data = load_data_from_cache(str(data_cache_dir), critical_dataframes=['lemmas_df', 'relations_df', 'definitions_df'])
    else:
        if args.skip_db_load and not lemmas_cache_file.exists():
            logger.warning(f"--skip-db-load was specified, but cache file {lemmas_cache_file} not found. Will load from DB.")
        logger.info("Loading data from database...")
        target_languages = config.get('data', {}).get('target_languages')
        data = load_data_from_db(db_params, target_languages=target_languages)
        
        logger.info(f"Saving loaded data to cache: {data_cache_dir}")
        save_data_to_cache(data, str(data_cache_dir))

    if data.get('lemmas_df', pd.DataFrame()).empty:
        logger.error("Lemmas data (lemmas_df) is empty after data loading phase. Cannot proceed.")
        sys.exit(1)

    # --- Merge POS information --- 
    definitions_df = data.get('definitions_df')
    pos_df = data.get('pos_df')
    lemmas_df = data.get('lemmas_df')

    if definitions_df is not None and not definitions_df.empty and \
       pos_df is not None and not pos_df.empty:
        
        logger.info("Attempting to merge POS information into definitions and lemmas dataframes...")
        # Ensure 'id' in pos_df and 'standardized_pos_id' in definitions_df are of compatible types
        if 'standardized_pos_id' in definitions_df.columns:
            definitions_df['standardized_pos_id'] = pd.to_numeric(definitions_df['standardized_pos_id'], errors='coerce')
        if 'id' in pos_df.columns: # pos_df['id'] is the PK for parts_of_speech
            pos_df['id'] = pd.to_numeric(pos_df['id'], errors='coerce')

        # Merge definitions with POS codes
        merged_definitions_df = pd.merge(
            definitions_df, 
            pos_df[['id', 'code']], 
            left_on='standardized_pos_id', 
            right_on='id', 
            how='left',
            suffixes=('', '_pos_table')
        ).rename(columns={'code': 'pos_code', 'id_pos_table': 'pos_id_from_pos_table'})
        data['definitions_df'] = merged_definitions_df # Update in the main data dictionary
        logger.info(f"Merged 'definitions_df' with 'pos_df'. 'definitions_df' now has {len(data['definitions_df'].columns)} columns.")
        if 'pos_code' not in data['definitions_df'].columns:
            logger.warning("'pos_code' column not found in 'definitions_df' after merge. POS information might be missing.")
        else:
            logger.info(f"Successfully added 'pos_code' to 'definitions_df'. Example POS codes: {data['definitions_df']['pos_code'].dropna().unique()[:5]}")

        # Derive primary POS for lemmas_df
        if lemmas_df is not None and not lemmas_df.empty and 'pos_code' in merged_definitions_df.columns:
            # Get the first non-null POS tag for each word_id from definitions
            # Sort by definition ID (if available) or an arbitrary stable sort to make 'first()' deterministic
            if 'id' in merged_definitions_df.columns: # Assuming 'id' is definition_id
                sorted_defs = merged_definitions_df.sort_values(by=['word_id', 'id'])
            else:
                sorted_defs = merged_definitions_df # No definition id to sort by, rely on original order
            
            word_primary_pos = sorted_defs.dropna(subset=['pos_code']).groupby('word_id', as_index=False)['pos_code'].first()
            
            data['lemmas_df'] = pd.merge(lemmas_df, 
                                         word_primary_pos, 
                                         left_on='id',         # 'id' is the word identifier in lemmas_df
                                         right_on='word_id',   # 'word_id' is the identifier in word_primary_pos
                                         how='left')
            # After the merge, lemmas_df will have 'id' (original from words table) 
            # and 'pos_code'. It will also gain a 'word_id' column from word_primary_pos.
            # This 'word_id' column from the merge is redundant if 'id' is the canonical word ID.
            # We can choose to drop it if it's not needed and to avoid confusion, or rename 'id' to 'word_id' earlier.
            # For now, let's keep it simple and allow the merge to add it.
            # If 'word_id' (the one from word_primary_pos) is not needed later, it can be dropped:
            # data['lemmas_df'] = data['lemmas_df'].drop(columns=['word_id_from_merge']) # Example of dropping

            logger.info(f"Merged primary POS codes into 'lemmas_df'. 'lemmas_df' now has {len(data['lemmas_df'].columns)} columns.")
            if 'pos_code' in data['lemmas_df'].columns:
                logger.info(f"Successfully added 'pos_code' to 'lemmas_df'. Missing POS codes after merge: {data['lemmas_df']['pos_code'].isnull().sum()}/{len(data['lemmas_df'])}")
            else:
                logger.warning("'pos_code' column not found in 'lemmas_df' after merge.")
        elif lemmas_df is None or lemmas_df.empty:
            logger.warning("Cannot derive primary POS for words because 'lemmas_df' is empty.")
        else: # 'pos_code' not in merged_definitions_df.columns
            logger.warning("Cannot derive primary POS for words because 'pos_code' was not successfully added to definitions.")
    else:
        logger.warning("Could not merge POS information as 'definitions_df' or 'pos_df' (or both) are missing or empty.")


    # --- Build Graph ---
    logger.info("Building graph data...")
    graph, mappings = build_graph_data(data, config)
    logger.info(f"Graph built: {graph}")
    logger.info(f"Node mappings generated for types: {list(mappings.keys()) if isinstance(mappings, dict) else 'N/A'}")

    if graph.num_nodes() == 0: # build_graph_data should exit if graph is None or no nodes, but double check.
        logger.error("Graph is empty after build_graph_data. Check data and graph building process.")
        sys.exit(1)

    # Extract relevant IDs from graph mappings to ensure feature alignment
    relevant_word_ids: Optional[Set[int]] = None
    relevant_definition_ids: Optional[Set[int]] = None
    relevant_etymology_ids: Optional[Set[int]] = None
    
    # Prepare node_to_original_id_maps for LexicalFeatureExtractor
    node_to_original_id_maps_for_lfe: Dict[str, Dict[int, int]] = {}

    if isinstance(mappings, dict):
        # For relevant_X_ids (sets of original DB IDs used by LFE for filtering DataFrames)
        if 'word_id_to_node' in mappings and isinstance(mappings['word_id_to_node'], dict):
            relevant_word_ids = set(mappings['word_id_to_node'].keys())
            logger.info(f"Extracted {len(relevant_word_ids)} relevant word_ids from graph mappings.")
        else:
            logger.warning("Could not extract relevant_word_ids from mappings. 'word_id_to_node' key missing or not a dict. Word features might not align perfectly if LFE relies on this for primary data filtering.")

        if 'def_id_to_node' in mappings and isinstance(mappings['def_id_to_node'], dict):
            relevant_definition_ids = set(mappings['def_id_to_node'].keys())
            logger.info(f"Extracted {len(relevant_definition_ids)} relevant definition_ids from graph mappings.")
        else:
            logger.warning("Could not extract relevant_definition_ids from mappings. 'def_id_to_node' key missing or not a dict. Definition features might not align optimally.")

        if 'etym_id_to_node' in mappings and isinstance(mappings['etym_id_to_node'], dict):
            relevant_etymology_ids = set(mappings['etym_id_to_node'].keys())
            logger.info(f"Extracted {len(relevant_etymology_ids)} relevant etymology_ids from graph mappings for 'etymology' nodes (if treated as separate nodes).")
        else:
            logger.warning("Could not extract relevant_etymology_ids from mappings. 'etym_id_to_node' key missing or not a dict. Etymology features (for 'etymology' nodes) might not align.")

        # For node_to_original_id_maps_for_lfe (maps graph index to original DB ID, per node type)
        # These are crucial for LFE's internal alignment, especially _determine_ordered_ids_for_ntype
        if 'node_to_word_id' in mappings and isinstance(mappings['node_to_word_id'], dict):
            node_to_original_id_maps_for_lfe['word'] = mappings['node_to_word_id']
            logger.info(f"Prepared 'node_to_word_id' for LFE (size: {len(mappings['node_to_word_id'])}).")
        else:
            logger.warning("Mapping 'node_to_word_id' not found in graph_builder mappings. LFE might not correctly map features if placeholders are extensively used for 'word'.")

        if 'node_to_def_id' in mappings and isinstance(mappings['node_to_def_id'], dict):
            node_to_original_id_maps_for_lfe['definition'] = mappings['node_to_def_id']
            logger.info(f"Prepared 'node_to_def_id' for LFE (size: {len(mappings['node_to_def_id'])}).")
        else:
            logger.warning("Mapping 'node_to_definition_id' not found in graph_builder mappings. LFE might not correctly map features if placeholders are extensively used for 'definition'.") # Log message updated for clarity

        if 'node_to_etym_id' in mappings and isinstance(mappings['node_to_etym_id'], dict):
            node_to_original_id_maps_for_lfe['etymology'] = mappings['node_to_etym_id']
            logger.info(f"Prepared 'node_to_etym_id' for LFE (size: {len(mappings['node_to_etym_id'])}).")
        else:
            logger.warning("Mapping 'node_to_etymology_id' not found in graph_builder mappings. LFE might not correctly map features if placeholders are extensively used for 'etymology'.") # Log message updated for clarity
            
    else:
        logger.warning("Graph mappings are not in the expected dictionary format. Cannot extract relevant_ids or node_to_original_id_maps.")
        
    features = extract_features(
        data, 
        graph, 
        config, 
        node_to_original_id_maps_for_lfe=node_to_original_id_maps_for_lfe, # Pass the prepared maps
        relevant_word_ids=relevant_word_ids, 
        relevant_definition_ids=relevant_definition_ids,
        relevant_etymology_ids=relevant_etymology_ids
    )
    logger.info(f"Features extracted for node types: {list(features.keys()) if features else 'None'}")
    # extract_features should sys.exit on critical errors like feature-node count mismatch.

    # The pretrain function in this script now handles model instantiation, training, and saving.
    # It returns the trained model object, the path where it was saved, and actual epochs run.
    trained_model, final_model_save_path, epochs_actually_run, training_histories = pretrain(graph, features, config, args)

    if trained_model and final_model_save_path:
        logger.info(f"Pre-training finished. Model was saved to {final_model_save_path} after {epochs_actually_run} epochs.")
    else:
        logger.error("Pre-training script finished, but model or save path was not returned properly.")
        sys.exit(1) # If pretrain itself didn't exit, but returned invalid results.

    logger.info("Pre-training script completed successfully.")

    # Output loss histories as JSON for capturing by the calling script
    if training_histories:
        try:
            # Convert all list values in training_histories to basic Python lists if they are numpy arrays or tensors
            serializable_histories = {}
            for key, value_list in training_histories.items():
                if isinstance(value_list, list) and value_list:
                    if isinstance(value_list[0], (torch.Tensor, np.ndarray)):
                        serializable_histories[key] = [v.item() if hasattr(v, 'item') else v for v in value_list]
                    elif isinstance(value_list[0], (int, float, str, bool)) or value_list[0] is None:
                        serializable_histories[key] = value_list
                    else:
                        logger.warning(f"History list '{key}' contains unsupported types for JSON: {type(value_list[0])}. Attempting direct serialization.")
                        serializable_histories[key] = value_list # Try direct, might fail
                else:
                     serializable_histories[key] = value_list # Empty list or already serializable

            training_histories_json = json.dumps(serializable_histories)
            print(f"FINAL_TRAINING_HISTORY_JSON:{training_histories_json}")
            logger.info("Successfully printed training histories as JSON.")
        except Exception as e_json:
            logger.error(f"Failed to serialize training histories to JSON: {e_json}")

if __name__ == "__main__":
    main() 