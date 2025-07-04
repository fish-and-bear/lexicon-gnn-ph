#!/usr/bin/env python3
"""
Script to use existing trained models and export 100 predictions for manual judgement.
"""

import torch
import pandas as pd
import numpy as np
import time
import psutil

from src.data_loading import fetch_graph_from_postgres, load_pg_connection
from src.graph_builder import build_hetero_graph
from src.models import create_model, LinkPredictor

def load_existing_model(model_path, model_type, config, metadata, in_channels_dict, device):
    """Load an existing trained model."""
    print(f"Loading model from {model_path}...")
    
    # Create model with the same architecture
    model = create_model(model_type, metadata, in_channels_dict, config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    
    # Create and load link predictor
    link_predictor = LinkPredictor(config["out_dim"]).to(device)
    link_predictor.load_state_dict(checkpoint["link_predictor"])
    
    print(f"Model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, link_predictor

def export_predictions(model, link_predictor, data, raw_data, output_path):
    """Export top 100 predicted links for manual judgement."""
    
    model.eval()
    link_predictor.eval()
    
    # Get all word node indices
    word_nodes = list(range(data["Word"].x.size(0)))
    
    # Build set of gold edges (existing edges)
    gold_edges = set()
    for edge_type in data.edge_types:
        if edge_type[0] == "Word" and edge_type[2] == "Word":
            edge_index = data[edge_type].edge_index
            for i in range(edge_index.size(1)):
                src = edge_index[0, i].item()
                dst = edge_index[1, i].item()
                gold_edges.add((src, dst))
    
    print(f"Found {len(gold_edges)} existing Word-Word edges")
    
    # Score all possible pairs not in gold_edges
    scores = []
    with torch.no_grad():
        out_dict = model(data.x_dict, data.edge_index_dict)
        word_emb = out_dict["Word"]
        
        # Sample pairs for efficiency (don't check all possible pairs)
        num_samples = 20000  # Increased for better coverage
        print(f"Sampling {num_samples} word pairs for prediction...")
        
        for _ in range(num_samples):
            i = np.random.randint(0, len(word_nodes))
            j = np.random.randint(0, len(word_nodes))
            if i == j or (i, j) in gold_edges:
                continue
            emb1 = word_emb[i].unsqueeze(0)
            emb2 = word_emb[j].unsqueeze(0)
            score = torch.sigmoid(link_predictor(emb1, emb2)).item()
            scores.append((i, j, score))
    
    # Get top 100 predictions
    scores.sort(key=lambda x: -x[2])
    top100 = scores[:100]
    
    # Map indices to lemmas
    idx2lemma = {idx: w["lemma"] for idx, w in enumerate(raw_data["words"])}
    
    # Export to CSV
    results = []
    for i, j, score in top100:
        word1 = idx2lemma.get(i, f"word_{i}")
        word2 = idx2lemma.get(j, f"word_{j}")
        results.append({
            "word1": word1,
            "word2": word2,
            "score": score,
            "predicted_relation": "SYNONYM_OF" if score > 0.5 else "ROOT_OF"
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Top 100 predictions exported to {output_path}")
    
    # Print some examples
    print("\nTop 10 predictions:")
    for i, row in df.head(10).iterrows():
        print(f"{row['word1']} <-> {row['word2']}: {row['score']:.4f} ({row['predicted_relation']})")

def main():
    print("Loading data from PostgreSQL...")
    start_time = time.time()
    
    # Create database connection
    db_config = {
        "host": "localhost",
        "port": 5432,
        "database": "fil_dict_db",
        "user": "postgres",
        "password: "***"
    }
    
    # Load data
    conn = load_pg_connection(db_config)
    if conn is None:
        print("Failed to connect to database")
        return
    
    raw_data = fetch_graph_from_postgres(conn)
    data = build_hetero_graph(raw_data, device="cpu")
    
    print(f"Graph loaded in {time.time() - start_time:.2f}s")
    print(f"Word nodes: {data['Word'].x.size(0)}")
    
    # Try different models with their configurations
    models_to_try = [
        {
            "name": "GATv2 Basic",
            "path": "gatv2_model.pt",
            "type": "gatv2",
            "config": {
                "hidden_dim": 512,
                "out_dim": 64,
                "num_layers": 2,
                "heads": 4,
                "num_relations": 8
            }
        },
        {
            "name": "GraphSAGE Basic", 
            "path": "sage_model.pt",
            "type": "sage",
            "config": {
                "hidden_dim": 256,
                "out_dim": 64,
                "num_layers": 2,
                "num_relations": 8
            }
        }
    ]
    
    metadata = (list(data.x_dict.keys()), list(data.edge_types))
    in_channels_dict = {node_type: data[node_type].x.size(-1) for node_type in data.x_dict.keys()}
    
    for model_info in models_to_try:
        try:
            print(f"\n{'='*50}")
            print(f"Trying {model_info['name']}")
            print(f"{'='*50}")
            
            model, link_predictor = load_existing_model(
                model_info["path"], 
                model_info["type"], 
                model_info["config"], 
                metadata, 
                in_channels_dict, 
                "cpu"
            )
            
            # Export predictions
            output_path = f"manual_judgement_{model_info['name'].lower().replace(' ', '_')}.csv"
            export_predictions(model, link_predictor, data, raw_data, output_path)
            
            print(f"Successfully used {model_info['name']}!")
            break  # Use the first successful model
            
        except Exception as e:
            print(f"Failed to load {model_info['name']}: {e}")
            continue
    
    # Runtime/memory reporting
    elapsed = time.time() - start_time
    if psutil:
        process = psutil.Process()
        mem = process.memory_info()
        peak = getattr(mem, 'peak_wset', mem.rss) / (1024 * 1024)
        print(f"\n[Resource] Total time: {elapsed:.2f} seconds | Peak memory: {peak:.2f} MB")
    else:
        print(f"\n[Resource] Total time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main() 