#!/usr/bin/env python3
"""
Test script to demonstrate the trained GNN models.
"""

import torch
import psycopg2
import yaml
from src.data_loading import fetch_graph_from_postgres
from src.graph_builder import build_hetero_graph
from src.models import create_model, LinkPredictor

def load_config():
    """Load configuration from yaml file."""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def test_models():
    """Test both trained models with sample word pairs."""
    
    # Load config
    config = load_config()
    device = torch.device("cpu")
    
    # Load data
    print("Loading data from PostgreSQL...")
    conn = psycopg2.connect(dbname='fil_dict_db', user='postgres', password='postgres', host='localhost', port=5432)
    raw_data = fetch_graph_from_postgres(conn)
    conn.close()
    
    # Build graph
    print("Building graph...")
    data = build_hetero_graph(raw_data, device)
    
    # Create word mapping
    word_to_idx = {w['lemma']: i for i, w in enumerate(raw_data['words'])}
    print(f"Loaded {len(word_to_idx)} words")
    
    # Test both models
    models = [
        ("GraphSAGE", "sage", "sage_model.pt"),
        ("GATv2", "gatv2", "gatv2_model.pt")
    ]
    
    for model_name, model_type, model_path in models:
        print(f"\n{'='*50}")
        print(f"Testing {model_name} Model")
        print(f"{'='*50}")
        
        try:
            # Load model
            metadata = (data.node_types, data.edge_types)
            in_channels_dict = {node_type: data[node_type].x.size(1) for node_type in data.node_types}
            model = create_model(model_type, metadata, in_channels_dict, config).to(device)
            
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            
            link_predictor = LinkPredictor(config["out_dim"]).to(device)
            link_predictor.load_state_dict(checkpoint["link_predictor"])
            
            model.eval()
            link_predictor.eval()
            
            # Get embeddings
            with torch.no_grad():
                out_dict = model(data.x_dict, data.edge_index_dict)
                word_embeddings = out_dict["Word"]
                
                print(f"Word embeddings shape: {word_embeddings.shape}")
                print(f"Model validation AUC: {checkpoint['val_auc']:.4f}")
                
                # Test some word pairs
                sample_words = list(word_to_idx.keys())[:10]
                print(f"\nSample word pairs from dataset:")
                
                for i in range(0, len(sample_words), 2):
                    if i + 1 < len(sample_words):
                        word1 = sample_words[i]
                        word2 = sample_words[i + 1]
                        
                        if word1 in word_to_idx and word2 in word_to_idx:
                            idx1 = word_to_idx[word1]
                            idx2 = word_to_idx[word2]
                            
                            emb1 = word_embeddings[idx1].unsqueeze(0)
                            emb2 = word_embeddings[idx2].unsqueeze(0)
                            
                            # Link prediction score
                            link_score = torch.sigmoid(link_predictor(emb1, emb2)).item()
                            
                            # Cosine similarity
                            cos_sim = torch.cosine_similarity(emb1, emb2).item()
                            
                            print(f"  {word1} <-> {word2}:")
                            print(f"    Link probability: {link_score:.4f}")
                            print(f"    Cosine similarity: {cos_sim:.4f}")
                
        except Exception as e:
            print(f"Error testing {model_name}: {e}")

if __name__ == "__main__":
    test_models() 