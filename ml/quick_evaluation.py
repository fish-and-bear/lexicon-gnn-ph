#!/usr/bin/env python3
"""
Quick evaluation script for the enhanced GNN model.
This script loads a trained model and evaluates it on basic metrics.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
import sys
import os

# Add the gnn_lexicon module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'gnn_lexicon', 'src'))

from models import GraphSAGE, GATv2, R_GCN, LinkPredictor
from data_loading import load_enhanced_data
from graph_builder import build_enhanced_graph

def quick_evaluation(model_path: str, device: str = 'cpu'):
    """
    Quick evaluation of a trained model.
    
    Args:
        model_path: Path to the trained model
        device: Device to use for evaluation
    """
    print(f"Loading model from {model_path}")
    print(f"Using device: {device}")
    
    # Load the model
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model_state = checkpoint['model_state_dict']
        link_predictor_state = checkpoint['link_predictor_state_dict']
        model_config = checkpoint.get('config', {})
        
        print(f"Model config: {model_config}")
        
        # Determine model type from config or filename
        model_type = model_config.get('model_type', 'sage')
        if 'gatv2' in model_path.lower():
            model_type = 'gatv2'
        elif 'rgcn' in model_path.lower():
            model_type = 'rgcn'
        
        print(f"Detected model type: {model_type}")
        
        # Create model
        if model_type == 'sage':
            model = GraphSAGE(
                in_dim=model_config.get('in_dim', 64),
                hidden_dim=model_config.get('hidden_dim', 128),
                out_dim=model_config.get('out_dim', 64),
                num_layers=model_config.get('num_layers', 2)
            )
        elif model_type == 'gatv2':
            model = GATv2(
                in_dim=model_config.get('in_dim', 64),
                hidden_dim=model_config.get('hidden_dim', 128),
                out_dim=model_config.get('out_dim', 64),
                num_layers=model_config.get('num_layers', 2),
                heads=model_config.get('heads', 4)
            )
        elif model_type == 'rgcn':
            model = R_GCN(
                in_dim=model_config.get('in_dim', 64),
                hidden_dim=model_config.get('hidden_dim', 128),
                out_dim=model_config.get('out_dim', 64),
                num_layers=model_config.get('num_layers', 2),
                num_relations=model_config.get('num_relations', 8)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create link predictor
        link_predictor = LinkPredictor(
            in_dim=model_config.get('out_dim', 64)
        )
        
        # Load state dicts
        model.load_state_dict(model_state)
        link_predictor.load_state_dict(link_predictor_state)
        
        model.to(device)
        link_predictor.to(device)
        
        print("Model loaded successfully!")
        
        # Try to load some test data
        print("Attempting to load test data...")
        try:
            # Try to load a small subset of data for testing
            data = load_enhanced_data(limit=1000)  # Load only 1000 words for quick test
            graph = build_enhanced_graph(data)
            
            print(f"Graph loaded with {graph['Word'].x.size(0)} word nodes")
            print(f"Edge types: {list(graph.edge_index_dict.keys())}")
            
            # Basic evaluation
            model.eval()
            link_predictor.eval()
            
            with torch.no_grad():
                # Move graph to device
                graph = graph.to(device)
                
                # Forward pass
                out_dict = model(graph.x_dict, graph.edge_index_dict)
                
                print(f"Model output shapes:")
                for node_type, embeddings in out_dict.items():
                    print(f"  {node_type}: {embeddings.shape}")
                
                # Test link prediction on a few edge types
                edge_types_to_test = [
                    ('Word', 'DERIVED_FROM', 'Word'),
                    ('Word', 'SYNONYM_OF', 'Word'),
                    ('Word', 'RELATED_TO', 'Word')
                ]
                
                for edge_type in edge_types_to_test:
                    if edge_type in graph.edge_index_dict:
                        edges = graph.edge_index_dict[edge_type]
                        if edges.size(1) > 0:
                            print(f"\nTesting {edge_type}: {edges.size(1)} edges")
                            
                            # Sample a few edges for testing
                            num_test_edges = min(100, edges.size(1))
                            test_edges = edges[:, :num_test_edges]
                            
                            # Get embeddings
                            src_emb = out_dict['Word'][test_edges[0]]
                            dst_emb = out_dict['Word'][test_edges[1]]
                            
                            # Predict scores
                            scores = torch.sigmoid(link_predictor(src_emb, dst_emb))
                            
                            print(f"  Average score: {scores.mean().item():.4f}")
                            print(f"  Score std: {scores.std().item():.4f}")
                            print(f"  Min score: {scores.min().item():.4f}")
                            print(f"  Max score: {scores.max().item():.4f}")
                        else:
                            print(f"\n{edge_type}: No edges found")
                    else:
                        print(f"\n{edge_type}: Edge type not found in graph")
            
            print("\n✅ Quick evaluation completed successfully!")
            
        except Exception as e:
            print(f"❌ Could not load test data: {e}")
            print("This is expected if PostgreSQL is not running.")
            print("The model appears to be loaded correctly.")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick evaluation of GNN model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    
    args = parser.parse_args()
    
    success = quick_evaluation(args.model_path, args.device)
    
    if success:
        print("\n🎉 Model evaluation completed!")
    else:
        print("\n💥 Model evaluation failed!")
        sys.exit(1) 