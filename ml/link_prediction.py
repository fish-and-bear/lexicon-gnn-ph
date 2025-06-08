#!/usr/bin/env python3

"""
Link prediction and knowledge graph enhancement using heterogeneous graph neural networks.

This script uses a trained GNN model to predict missing links in the Filipino lexical 
knowledge graph and enhance it with high-confidence predictions.
"""

import os
import sys
import argparse
import logging
import json
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
from ml.models.encoder import GraphEncoder
from ml.utils.logging_utils import setup_logging
from ml.utils.evaluation_utils import evaluate_link_prediction, compute_confidence_scores

# Set up logging
logger = logging.getLogger(__name__)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"link_prediction_{timestamp}.log"
setup_logging(log_file=LOG_FILE, level=logging.INFO)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Predict missing links in Filipino lexical knowledge graph")
    
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained GraphEncoder model checkpoint")
    parser.add_argument("--config", type=str, default="config/default_config.json",
                        help="Path to configuration file")
    parser.add_argument("--db-config", type=str, default="my_db_config.json",
                        help="Path to database configuration file")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory with cached data")
    parser.add_argument("--output-dir", type=str, default="predictions",
                        help="Directory to save predictions")
    parser.add_argument("--relation-types", type=str, nargs="+",
                        help="Relation types to predict (default: all)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top predictions to output per relation type")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold for predictions (0.0 to 1.0)")
    parser.add_argument("--export-visualizations", action="store_true",
                        help="Export graph visualizations")
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

def load_model_from_checkpoint(model_path: str, graph: dgl.DGLGraph, config: dict) -> GraphEncoder:
    """
    Load a trained GraphEncoder model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        graph: The graph used to infer node_types if not in checkpoint
        config: Overall configuration dict (used for model HPs if not in checkpoint)
    
    Returns:
        Loaded GraphEncoder model
    """
    try:
        logger.info(f"Loading model from {model_path}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract model configuration from checkpoint, fallback to main config
        # The training script should save these under 'model_hyperparameters' or similar key
        model_hp = checkpoint.get('model_hyperparameters', {})
        
        # Essential parameters for GraphEncoder from its __init__ signature
        # Try to get from checkpoint, then from general config, then provide defaults or raise error
        original_feat_dims = model_hp.get('original_feat_dims', checkpoint.get('original_feat_dims')) # Legacy support
        if original_feat_dims is None:
            # This is critical and ideally should be saved. 
            # For now, if link prediction has its own feature extraction, it could be inferred.
            # However, the trained model depends on specific dimensions it was trained with.
            raise ValueError("original_feat_dims not found in model checkpoint or config. This is required.")

        node_types = model_hp.get('node_types', graph.ntypes)
        rel_names = model_hp.get('rel_names', graph.etypes) # DGL graph.etypes gives relation names

        # Use get with fallback to general config model section, then to a default
        # Ensure these names match GraphEncoder __init__ params
        graph_encoder_params = {
            'original_feat_dims': original_feat_dims,
            'hidden_dim': model_hp.get('hidden_dim', config.get('model', {}).get('hidden_dim', 256)),
            'out_dim': model_hp.get('out_dim', config.get('model', {}).get('out_dim', 128)),
            'node_types': node_types,
            'rel_names': rel_names,
            'num_encoder_layers': model_hp.get('num_encoder_layers', config.get('model', {}).get('num_encoder_layers', 3)),
            'num_decoder_layers': model_hp.get('num_decoder_layers', config.get('model', {}).get('num_decoder_layers', 1)),
            'dropout': model_hp.get('dropout', config.get('model', {}).get('dropout', 0.1)),
            'feature_mask_rate': model_hp.get('feature_mask_rate', config.get('model', {}).get('feature_mask_rate', 0.3)),
            'edge_mask_rate': model_hp.get('edge_mask_rate', config.get('model', {}).get('edge_mask_rate', 0.3)),
            'num_heads': model_hp.get('num_heads', config.get('model', {}).get('num_heads', 8)),
            'residual': model_hp.get('residual', config.get('model', {}).get('residual', True)),
            'layer_norm': model_hp.get('layer_norm', config.get('model', {}).get('layer_norm', True)),
            'num_bases': model_hp.get('num_bases', config.get('model', {}).get('num_bases', 8)),
            'hgnn_sparsity': model_hp.get('hgnn_sparsity', config.get('model', {}).get('hgnn_sparsity', 0.9)),
        }

        model = GraphEncoder(**graph_encoder_params)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        logger.info(f"GraphEncoder model loaded successfully from epoch {checkpoint.get('epoch', 'N/A')}")
        logger.info(f"Model Hyperparameters used: {graph_encoder_params}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}", exc_info=True)
        sys.exit(1)

def generate_candidate_edges(graph, lemmas_df, mappings, relation_types):
    """
    Generate candidate edges for prediction.
    
    Args:
        graph: The heterogeneous graph
        lemmas_df: DataFrame with word lemmas
        mappings: Node mappings dictionary
        relation_types: List of relation types to predict
    
    Returns:
        Dictionary mapping relation types to candidate edge indices
    """
    logger.info("Generating candidate edges for prediction")
    
    # Get word_id to node_id mapping
    word_id_to_node = mappings['word_id_to_node']
    node_to_word_id = mappings['node_to_word_id']
    
    # Get existing edges by relation type
    existing_edges = {}
    for rel_type in relation_types:
        etype = ('word', rel_type, 'word')
        if etype in graph.canonical_etypes:
            src, dst = graph.edges(etype=etype)
            edge_indices = set(zip(src.tolist(), dst.tolist()))
            existing_edges[rel_type] = edge_indices
            logger.info(f"Found {len(edge_indices)} existing edges for relation type {rel_type}")
        else:
            existing_edges[rel_type] = set()
            logger.warning(f"Relation type {rel_type} not found in graph")
    
    # Initialize candidate edges
    candidates = {rel_type: [] for rel_type in relation_types}
    
    # Group lemmas by language for better candidate selection
    language_groups = {}
    for _, row in lemmas_df.iterrows():
        lang = row['language_code'] or 'unknown'
        word_id = row['id']
        if lang not in language_groups:
            language_groups[lang] = []
        
        if word_id in word_id_to_node:  # Only include words in the graph
            language_groups[lang].append(word_id)
    
    logger.info(f"Grouped words by language: {', '.join([f'{k}: {len(v)}' for k, v in language_groups.items()])}")
    
    # Generate candidates with a sampling strategy to avoid combinatorial explosion
    for rel_type in relation_types:
        logger.info(f"Generating candidates for relation type {rel_type}")
        
        # Different strategies based on relation type
        if rel_type in ['synonym', 'antonym', 'related', 'see_also', 'variant']:
            # These relations typically connect words in the same language
            for lang, word_ids in language_groups.items():
                if len(word_ids) <= 1:
                    continue
                
                # Sample pairs within the same language
                num_samples = min(1000, len(word_ids) * 10)  # Limit samples to avoid huge number
                
                for _ in range(num_samples):
                    # Randomly sample two words
                    w1, w2 = np.random.choice(word_ids, 2, replace=False)
                    
                    # Convert to node IDs
                    n1 = word_id_to_node[w1]
                    n2 = word_id_to_node[w2]
                    
                    # Check if edge already exists
                    if (n1, n2) not in existing_edges[rel_type]:
                        candidates[rel_type].append((n1, n2))
        
        elif rel_type in ['has_translation', 'translation_of', 'cognate_of']:
            # These relations connect words across languages
            languages = list(language_groups.keys())
            for i in range(len(languages)):
                for j in range(i+1, len(languages)):
                    lang1 = languages[i]
                    lang2 = languages[j]
                    
                    word_ids1 = language_groups[lang1]
                    word_ids2 = language_groups[lang2]
                    
                    if not word_ids1 or not word_ids2:
                        continue
                    
                    # Sample pairs across different languages
                    num_samples = min(1000, len(word_ids1) * len(word_ids2) // 100)
                    
                    for _ in range(num_samples):
                        # Randomly sample from each language
                        w1 = np.random.choice(word_ids1)
                        w2 = np.random.choice(word_ids2)
                        
                        # Convert to node IDs
                        n1 = word_id_to_node[w1]
                        n2 = word_id_to_node[w2]
                        
                        # Check if edge already exists
                        if (n1, n2) not in existing_edges[rel_type]:
                            candidates[rel_type].append((n1, n2))
        
        else:
            # For other relation types, sample random pairs
            all_nodes = list(range(graph.num_nodes('word')))
            num_samples = min(5000, len(all_nodes) * 10)  # Limit samples
            
            for _ in range(num_samples):
                n1, n2 = np.random.choice(all_nodes, 2, replace=False)
                
                if (n1, n2) not in existing_edges[rel_type]:
                    candidates[rel_type].append((n1, n2))
        
        logger.info(f"Generated {len(candidates[rel_type])} candidate edges for {rel_type}")
    
    return candidates

def predict_links(model, graph, features, candidates, threshold, top_k):
    """
    Predict links for candidate edges.
    
    Args:
        model: Trained GNN model
        graph: The heterogeneous graph
        features: Node features
        candidates: Dictionary mapping relation types to candidate edge indices
        threshold: Confidence threshold for predictions
        top_k: Number of top predictions to return
        
    Returns:
        Dictionary mapping relation types to predicted edges with scores
    """
    logger.info("Predicting links for candidate edges")
    
    # Move model to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Move features to device
    for ntype in features:
        features[ntype] = features[ntype].to(device)
    
    # Initialize predictions
    predictions = {}
    
    # Batch size for processing candidates
    batch_size = 1000
    
    # Process each relation type
    for rel_type, candidate_edges in candidates.items():
        logger.info(f"Processing {len(candidate_edges)} candidate edges for {rel_type}")
        
        edge_scores = []
        
        # Process in batches
        for i in range(0, len(candidate_edges), batch_size):
            batch_edges = candidate_edges[i:i+batch_size]
            
            # Prepare edge indices
            src = [e[0] for e in batch_edges]
            dst = [e[1] for e in batch_edges]
            
            edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
            
            # Get edge scores
            with torch.no_grad():
                scores = model.score_edges(
                    features, edge_index, edge_type=('word', rel_type, 'word')
                )
            
            # Convert to probabilities if needed
            if scores.min() < 0 or scores.max() > 1:
                probs = torch.sigmoid(scores)
            else:
                probs = scores
            
            # Convert to numpy
            probs_np = probs.cpu().numpy()
            
            # Store edges with scores
            for j, (s, d) in enumerate(batch_edges):
                edge_scores.append((s, d, probs_np[j]))
        
        # Filter by threshold and get top-k
        filtered_scores = [(s, d, score) for s, d, score in edge_scores if score >= threshold]
        filtered_scores.sort(key=lambda x: x[2], reverse=True)
        top_predictions = filtered_scores[:top_k]
        
        predictions[rel_type] = top_predictions
        
        logger.info(f"Found {len(filtered_scores)} predictions above threshold for {rel_type}")
        logger.info(f"Top prediction scores: {[round(score, 3) for _, _, score in top_predictions[:5]]}")
    
    return predictions

def format_predictions(predictions, node_to_word_id, lemmas_df):
    """
    Format predictions for output.
    
    Args:
        predictions: Dictionary mapping relation types to predicted edges with scores
        node_to_word_id: Mapping from node IDs to word IDs
        lemmas_df: DataFrame with word lemmas
        
    Returns:
        Formatted predictions for output
    """
    logger.info("Formatting predictions")
    
    # Create word ID to lemma mapping for faster lookup
    word_id_to_lemma = {}
    word_id_to_lang = {}
    
    for _, row in lemmas_df.iterrows():
        word_id = row['id']
        word_id_to_lemma[word_id] = row['lemma']
        word_id_to_lang[word_id] = row['language_code']
    
    # Format predictions by relation type
    formatted_predictions = {}
    
    for rel_type, edges in predictions.items():
        rel_predictions = []
        
        for src_node, dst_node, score in edges:
            # Convert node IDs to word IDs
            src_word_id = node_to_word_id.get(str(src_node), node_to_word_id.get(src_node))
            dst_word_id = node_to_word_id.get(str(dst_node), node_to_word_id.get(dst_node))
            
            if src_word_id is None or dst_word_id is None:
                logger.warning(f"Failed to map node IDs {src_node}, {dst_node} to word IDs")
                continue
            
            # Get lemmas and languages
            src_lemma = word_id_to_lemma.get(src_word_id, f"Unknown-{src_word_id}")
            dst_lemma = word_id_to_lemma.get(dst_word_id, f"Unknown-{dst_word_id}")
            src_lang = word_id_to_lang.get(src_word_id)
            dst_lang = word_id_to_lang.get(dst_word_id)
            
            # Format prediction
            rel_predictions.append({
                'from_word_id': src_word_id,
                'from_lemma': src_lemma,
                'from_language': src_lang,
                'to_word_id': dst_word_id,
                'to_lemma': dst_lemma,
                'to_language': dst_lang,
                'relation_type': rel_type,
                'confidence_score': float(score),
                'is_automatic': True,
            })
        
        formatted_predictions[rel_type] = rel_predictions
    
    return formatted_predictions

def save_predictions(predictions, output_dir):
    """
    Save predictions to output files.
    
    Args:
        predictions: Formatted predictions
        output_dir: Output directory
        
    Returns:
        Path to the saved predictions file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all predictions to a single JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"predicted_relations_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    logger.info(f"Saved predictions to {output_file}")
    
    # Save relation-specific files
    total_predictions = 0
    for rel_type, rel_predictions in predictions.items():
        rel_file = os.path.join(output_dir, f"{rel_type}_{timestamp}.json")
        
        with open(rel_file, 'w') as f:
            json.dump(rel_predictions, f, indent=2)
        
        logger.info(f"Saved {len(rel_predictions)} {rel_type} predictions to {rel_file}")
        total_predictions += len(rel_predictions)
    
    logger.info(f"Saved {total_predictions} total predictions")
    return output_file

def export_csv_for_review(predictions, output_dir):
    """
    Export predictions to CSV files for manual review.
    
    Args:
        predictions: Formatted predictions
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine all predictions
    all_rows = []
    for rel_type, rel_predictions in predictions.items():
        for pred in rel_predictions:
            row = {
                'from_word_id': pred['from_word_id'],
                'from_lemma': pred['from_lemma'],
                'from_language': pred['from_language'],
                'to_word_id': pred['to_word_id'],
                'to_lemma': pred['to_lemma'],
                'to_language': pred['to_language'],
                'relation_type': pred['relation_type'],
                'confidence_score': pred['confidence_score'],
                'is_valid': '',  # To be filled by reviewer
                'notes': '',     # To be filled by reviewer
            }
            all_rows.append(row)
    
    # Create DataFrame and sort by confidence score
    df = pd.DataFrame(all_rows)
    df = df.sort_values(by=['relation_type', 'confidence_score'], ascending=[True, False])
    
    # Export to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(output_dir, f"predictions_for_review_{timestamp}.csv")
    df.to_csv(csv_file, index=False)
    
    logger.info(f"Exported {len(df)} predictions to CSV for review: {csv_file}")

def export_visualizations(graph, predictions, node_to_word_id, lemmas_df, output_dir):
    """
    Export graph visualizations of predictions.
    
    Args:
        graph: The heterogeneous graph
        predictions: Formatted predictions
        node_to_word_id: Mapping from node IDs to word IDs
        lemmas_df: DataFrame with word lemmas
        output_dir: Output directory
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        from matplotlib import cm
        
        logger.info("Exporting graph visualizations")
        
        # Create visualization directory
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Create word ID to lemma mapping for faster lookup
        word_id_to_lemma = {}
        for _, row in lemmas_df.iterrows():
            word_id = row['id']
            word_id_to_lemma[word_id] = row['lemma']
        
        # Process each relation type
        for rel_type, rel_predictions in predictions.items():
            if not rel_predictions:
                continue
                
            logger.info(f"Creating visualization for {rel_type}")
            
            # Create a NetworkX graph for visualization
            G = nx.DiGraph()
            
            # Add nodes and edges from predictions
            for pred in rel_predictions:
                from_word_id = pred['from_word_id']
                to_word_id = pred['to_word_id']
                from_lemma = pred['from_lemma']
                to_lemma = pred['to_lemma']
                confidence = pred['confidence_score']
                
                # Add nodes if not already present
                if from_word_id not in G.nodes:
                    G.add_node(from_word_id, label=from_lemma)
                if to_word_id not in G.nodes:
                    G.add_node(to_word_id, label=to_lemma)
                
                # Add edge with confidence score
                G.add_edge(from_word_id, to_word_id, weight=confidence, confidence=confidence)
            
            # Limit graph size for visualization
            if len(G) > 50:
                # Keep only high-confidence predictions or a subset
                subgraph_nodes = set()
                for pred in sorted(rel_predictions, key=lambda x: x['confidence_score'], reverse=True)[:25]:
                    subgraph_nodes.add(pred['from_word_id'])
                    subgraph_nodes.add(pred['to_word_id'])
                
                G = G.subgraph(subgraph_nodes)
            
            # Create figure
            plt.figure(figsize=(12, 10))
            
            # Set up layout
            pos = nx.spring_layout(G, seed=42)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', alpha=0.8)
            
            # Draw edges with color based on confidence
            edges = G.edges(data=True)
            edge_colors = [e[2]['confidence'] for e in edges]
            nx.draw_networkx_edges(
                G, pos, width=2, alpha=0.7,
                edge_color=edge_colors, edge_cmap=cm.viridis,
                edge_vmin=0.5, edge_vmax=1.0,
                arrows=True, arrowstyle='-|>', arrowsize=15
            )
            
            # Draw labels
            labels = {node: G.nodes[node].get('label', str(node)) for node in G.nodes}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=0.5, vmax=1.0))
            plt.colorbar(sm, label='Confidence Score')
            
            # Set title and layout
            plt.title(f'Predicted {rel_type} Relations')
            plt.axis('off')
            
            # Save figure
            fig_path = os.path.join(vis_dir, f"{rel_type}_graph.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved visualization to {fig_path}")
    
    except ImportError as e:
        logger.warning(f"Could not export visualizations: {e}")
        logger.warning("Install networkx and matplotlib for visualization support")

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set debug level if needed
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")
    
    # Load configuration
    config = load_config(args.config)
    db_config = load_db_config(args.db_config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    logger.info("Loading data from cache")
    data = load_data_from_cache(args.data_dir)
    
    # Build graph
    logger.info("Building graph from data")
    target_languages = config.get('data', {}).get('target_languages')
    graph_builder = LexicalGraphBuilder(config, target_languages=target_languages)
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
    
    # Extract features
    logger.info("Extracting node features")
    feature_extractor = LexicalFeatureExtractor(
        use_xlmr=config.get('data', {}).get('use_xlmr', True),
        use_fasttext=config.get('data', {}).get('use_fasttext', True),
        use_char_ngrams=config.get('data', {}).get('use_char_ngrams', True),
        use_phonetic_features=config.get('data', {}).get('use_phonetic_features', True),
        use_etymology_features=config.get('data', {}).get('use_etymology_features', True),
        use_baybayin_features=config.get('data', {}).get('use_baybayin_features', True),
        normalize_features=config.get('data', {}).get('normalize_features', True),
    )
    
    features = feature_extractor.extract_all_features(
        lemmas_df=data['lemmas_df'],
        definitions_df=data['definitions_df'],
        etymologies_df=data['etymologies_df'],
        pronunciations_df=data['pronunciations_df'],
        word_forms_df=data['word_forms_df'],
    )
    
    # Load model
    model = load_model_from_checkpoint(args.model_path, graph, config)
    
    # Determine relation types to predict
    if args.relation_types:
        relation_types = args.relation_types
    else:
        relation_types = config.get('data', {}).get('relation_types_to_predict', [])
        if not relation_types:
            # Use all word-to-word relations in the graph
            relation_types = [etype[1] for etype in graph.canonical_etypes 
                             if etype[0] == 'word' and etype[2] == 'word']
    
    logger.info(f"Will predict the following relation types: {relation_types}")
    
    # Generate candidate edges
    candidates = generate_candidate_edges(graph, data['lemmas_df'], mappings, relation_types)
    
    # Predict links
    predictions = predict_links(model, graph, features, candidates, args.threshold, args.top_k)
    
    # Format predictions
    formatted_predictions = format_predictions(predictions, mappings['node_to_word_id'], data['lemmas_df'])
    
    # Save predictions
    output_file = save_predictions(formatted_predictions, args.output_dir)
    
    # Export CSV for review
    export_csv_for_review(formatted_predictions, args.output_dir)
    
    # Export visualizations if requested
    if args.export_visualizations:
        export_visualizations(graph, formatted_predictions, mappings['node_to_word_id'], 
                             data['lemmas_df'], args.output_dir)
    
    logger.info(f"All done! Link predictions saved to {output_file}")

if __name__ == "__main__":
    main() 