#!/usr/bin/env python3
"""
Simple training script for GNN models on the Philippine lexicon data.
"""

import torch
import json
import yaml
from src.cli import load_pg_connection, fetch_graph_from_postgres, create_toy_graph
from src.graph_builder import build_hetero_graph
from src.models import create_model
from src.training import train_gnn, split_edges
from src.link_prediction import LinkPredictor

def main():
    print("Training new GNN model on Philippine lexicon data...")
    
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load data from PostgreSQL
    print("Loading data from postgres...")
    conn = load_pg_connection(config["postgres"])
    if conn is None:
        print("Failed to connect to PostgreSQL. Using toy graph instead.")
        raw_data = create_toy_graph()
    else:
        raw_data = fetch_graph_from_postgres(conn)
        conn.close()
    
    # Build graph
    print("Building heterogeneous graph...")
    data = build_hetero_graph(raw_data, device)
    
    print(f"Graph statistics:")
    print(f"  Node types: {data.node_types}")
    print(f"  Edge types: {data.edge_types}")
    for node_type in data.node_types:
        if node_type in data.node_types:
            print(f"  {node_type} nodes: {data[node_type].x.size(0)}")
    for edge_type in data.edge_types:
        if edge_type in data.edge_types:
            print(f"  {edge_type} edges: {data[edge_type].edge_index.size(1)}")
    
    # Get metadata and input channels
    metadata = (data.node_types, data.edge_types)
    in_channels_dict = {
        node_type: data[node_type].x.size(1) 
        for node_type in data.node_types
    }
    
    # Train GATv2 model
    print("\nTraining GATv2 model...")
    
    # Split data
    train_data, val_data, test_data = split_edges(data)
    
    # Create model
    model = create_model("gatv2", metadata, in_channels_dict, config).to(device)
    print(f"Model: GATv2")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    
    # Train
    results = train_gnn(
        model, train_data, val_data, optimizer, device, 
        config, amp=False, save_path="gatv2_fresh.pt"
    )
    
    print(f"\nBest validation AUC: {results['best_val_auc']:.4f}")
    
    # Save training history
    history_path = "gatv2_fresh_history.json"
    with open(history_path, "w") as f:
        json.dump(results["history"], f, indent=2)
    print(f"Training history saved to {history_path}")
    
    # Train SAGE model
    print("\nTraining SAGE model...")
    
    # Create model
    model = create_model("sage", metadata, in_channels_dict, config).to(device)
    print(f"Model: SAGE")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    
    # Train
    results = train_gnn(
        model, train_data, val_data, optimizer, device, 
        config, amp=False, save_path="sage_fresh.pt"
    )
    
    print(f"\nBest validation AUC: {results['best_val_auc']:.4f}")
    
    # Save training history
    history_path = "sage_fresh_history.json"
    with open(history_path, "w") as f:
        json.dump(results["history"], f, indent=2)
    print(f"Training history saved to {history_path}")
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main() 