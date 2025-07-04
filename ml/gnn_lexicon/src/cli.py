"""
Command-line interface for Philippine Lexicon GNN.
Supports training, evaluation, inference, and ablation studies.
"""

import argparse
import os
import json
import torch
from typing import Dict, Any

from .data_loading import load_pg_connection, fetch_graph_from_postgres
from .graph_builder import build_hetero_graph, split_edges
from .models import create_model, LinkPredictor, RelationClassifier
from .training import train_gnn, run_ablation
from .evaluation import evaluate_link_prediction, evaluate_hits_at_k, print_evaluation_summary
from .utils import load_config, get_default_config, save_model, load_model

def main():
    parser = argparse.ArgumentParser(description="Philippine Lexicon GNN Toolkit")
    
    parser.add_argument("command", choices=["train", "evaluate", "infer", "ablate"],
                        help="Command to run")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--model-type", type=str, default="rgcn",
                        choices=["rgcn", "sage", "gatv2"],
                        help="Model architecture")
    parser.add_argument("--model-path", type=str, default="model.pt",
                        help="Path to save/load model")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cuda/cpu/auto)")
    parser.add_argument("--amp", action="store_true",
                        help="Use automatic mixed precision")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--query", type=str, nargs=2, action="append",
                        help="Word pairs for inference (can be used multiple times)")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line args
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.lr is not None:
        config["lr"] = args.lr
    
    # Set device
    if args.device == "auto":
        device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load data (always from PostgreSQL)
    print(f"Loading data from PostgreSQL...")
    conn = load_pg_connection(config["postgres"])
    if conn is None:
        print("Failed to connect to PostgreSQL. Exiting.")
        return
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
    
    # Execute command
    if args.command == "train":
        print("\nTraining model...")
        
        # Split data
        train_data, val_data, test_data = split_edges(data)
        
        # Create model
        model = create_model(args.model_type, metadata, in_channels_dict, config).to(device)
        print(f"Model: {args.model_type}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
        
        # Train
        results = train_gnn(
            model, train_data, val_data, optimizer, device, 
            config, amp=args.amp, save_path=args.model_path
        )
        
        print(f"\nBest validation AUC: {results['best_val_auc']:.4f}")
        
        # Save training history
        history_path = args.model_path.replace(".pt", "_history.json")
        with open(history_path, "w") as f:
            json.dump(results["history"], f, indent=2)
        print(f"Training history saved to {history_path}")
        
    elif args.command == "evaluate":
        print("\nEvaluating model...")
        
        # Load model
        model = create_model(args.model_type, metadata, in_channels_dict, config).to(device)
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        
        link_predictor = LinkPredictor(config["out_dim"]).to(device)
        link_predictor.load_state_dict(checkpoint["link_predictor"])
        
        # Split data
        _, _, test_data = split_edges(data)
        
        # Evaluate
        results = {}
        results["link_auc"] = evaluate_link_prediction(model, link_predictor, test_data, device)
        results["hits@10"] = evaluate_hits_at_k(model, link_predictor, test_data, device, k=10)
        results["hits@5"] = evaluate_hits_at_k(model, link_predictor, test_data, device, k=5)
        results["hits@1"] = evaluate_hits_at_k(model, link_predictor, test_data, device, k=1)
        
        print_evaluation_summary(results)
        
    elif args.command == "infer":
        print("\nRunning inference...")
        
        if not args.query:
            print("Please provide word pairs using --query word1 word2")
            return
        
        # Load model
        model = create_model(args.model_type, metadata, in_channels_dict, config).to(device)
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        
        link_predictor = LinkPredictor(config["out_dim"]).to(device)
        link_predictor.load_state_dict(checkpoint["link_predictor"])
        
        model.eval()
        link_predictor.eval()
        
        # Create word to index mapping
        word_to_idx = {w["lemma"]: i for i, w in enumerate(raw_data["words"])}
        
        with torch.no_grad():
            out_dict = model(data.x_dict, data.edge_index_dict)
            word_embeddings = out_dict["Word"]
            
            print("\nWord pair predictions:")
            print("-" * 50)
            
            for word1, word2 in args.query:
                if word1 in word_to_idx and word2 in word_to_idx:
                    idx1 = word_to_idx[word1]
                    idx2 = word_to_idx[word2]
                    
                    emb1 = word_embeddings[idx1].unsqueeze(0)
                    emb2 = word_embeddings[idx2].unsqueeze(0)
                    
                    # Link prediction score
                    link_score = torch.sigmoid(link_predictor(emb1, emb2)).item()
                    
                    # Cosine similarity
                    cos_sim = torch.cosine_similarity(emb1, emb2).item()
                    
                    print(f"{word1} <-> {word2}:")
                    print(f"  Link probability: {link_score:.4f}")
                    print(f"  Cosine similarity: {cos_sim:.4f}")
                else:
                    missing = []
                    if word1 not in word_to_idx:
                        missing.append(word1)
                    if word2 not in word_to_idx:
                        missing.append(word2)
                    print(f"{word1} <-> {word2}: Missing words: {', '.join(missing)}")
                print()
        
    elif args.command == "ablate":
        print("\nRunning ablation study...")
        
        # Define edge types to ablate
        edge_types_to_ablate = [
            ("Word", "HAS_FORM", "Form"),
            ("Word", "HAS_SENSE", "Sense"),
            ("Word", "HAS_AFFIX", "Morpheme"),
            ("Word", "SHARES_PHONOLOGY", "Word"),
            ("Word", "SHARES_ETYMOLOGY", "Word")
        ]
        
        # Run ablation
        model_class = {
            "rgcn": lambda m, i, h, o, l: create_model("rgcn", m, i, {"hidden_dim": h, "out_dim": o, "num_layers": l}),
            "sage": lambda m, i, h, o, l: create_model("sage", m, i, {"hidden_dim": h, "out_dim": o, "num_layers": l}),
            "gatv2": lambda m, i, h, o, l: create_model("gatv2", m, i, {"hidden_dim": h, "out_dim": o, "num_layers": l, "heads": 4})
        }[args.model_type]
        
        results = run_ablation(
            model_class, data, config, device, edge_types_to_ablate
        )
        
        # Print results
        print("\nAblation Results:")
        print("-" * 50)
        for condition, metrics in results.items():
            print(f"{condition}: AUC = {metrics['val_auc']:.4f}")
        
        # Save results
        ablation_path = args.model_path.replace(".pt", "_ablation.json")
        with open(ablation_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nAblation results saved to {ablation_path}")

if __name__ == "__main__":
    main() 