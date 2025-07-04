"""
Example usage of Philippine Lexicon GNN Toolkit.
"""

import torch
from gnn_lexicon.src import (
    create_toy_graph,
    build_hetero_graph,
    split_edges,
    create_model,
    train_gnn,
    evaluate_link_prediction,
    evaluate_hits_at_k,
    LinkPredictor
)

def main():
    # Configuration
    config = {
        "hidden_dim": 64,
        "out_dim": 32,
        "num_layers": 2,
        "heads": 4,
        "lr": 0.001,
        "batch_size": 32,
        "epochs": 20,
        "grad_clip": 1.0,
        "early_stopping_patience": 5
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Step 1: Load data
    print("\n1. Loading toy graph data...")
    raw_data = create_toy_graph()
    print(f"   Words: {len(raw_data['words'])}")
    print(f"   Relations: {len(raw_data['relations'])}")
    
    # Step 2: Build graph
    print("\n2. Building heterogeneous graph...")
    data = build_hetero_graph(raw_data, device)
    print(f"   Node types: {data.node_types}")
    print(f"   Edge types: {[str(et) for et in data.edge_types]}")
    
    # Step 3: Split data
    print("\n3. Splitting edges for train/val/test...")
    train_data, val_data, test_data = split_edges(data, train_ratio=0.6, val_ratio=0.2)
    
    # Step 4: Create model
    print("\n4. Creating GATv2 model...")
    metadata = (data.node_types, data.edge_types)
    in_channels_dict = {
        node_type: data[node_type].x.size(1) 
        for node_type in data.node_types
    }
    
    model = create_model("gatv2", metadata, in_channels_dict, config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {num_params:,}")
    
    # Step 5: Train model
    print("\n5. Training model...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    
    results = train_gnn(
        model, train_data, val_data, optimizer, device, config,
        amp=False, save_path="example_model.pt"
    )
    
    print(f"\n   Best validation AUC: {results['best_val_auc']:.4f}")
    print(f"   Training stopped at epoch: {len(results['history']['train_loss'])}")
    
    # Step 6: Evaluate on test set
    print("\n6. Evaluating on test set...")
    
    # Load best model
    checkpoint = torch.load("example_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    
    link_predictor = LinkPredictor(config["out_dim"]).to(device)
    link_predictor.load_state_dict(checkpoint["link_predictor"])
    
    test_auc = evaluate_link_prediction(model, link_predictor, test_data, device)
    test_hits = evaluate_hits_at_k(model, link_predictor, test_data, device, k=3)
    
    print(f"   Test AUC: {test_auc:.4f}")
    print(f"   Test Hits@3: {test_hits:.4f}")
    
    # Step 7: Query specific word pairs
    print("\n7. Querying word relationships...")
    
    word_to_idx = {w["lemma"]: i for i, w in enumerate(raw_data["words"])}
    
    model.eval()
    link_predictor.eval()
    
    with torch.no_grad():
        out_dict = model(data.x_dict, data.edge_index_dict)
        word_embeddings = out_dict["Word"]
        
        # Query pairs
        pairs = [("takbo", "tumakbo"), ("kain", "kumain"), ("takbo", "lakad")]
        
        print("\n   Word pair predictions:")
        for word1, word2 in pairs:
            if word1 in word_to_idx and word2 in word_to_idx:
                idx1 = word_to_idx[word1]
                idx2 = word_to_idx[word2]
                
                emb1 = word_embeddings[idx1].unsqueeze(0)
                emb2 = word_embeddings[idx2].unsqueeze(0)
                
                # Link prediction
                link_prob = torch.sigmoid(link_predictor(emb1, emb2)).item()
                
                # Cosine similarity
                cos_sim = torch.cosine_similarity(emb1, emb2).item()
                
                print(f"   {word1} <-> {word2}:")
                print(f"      Link probability: {link_prob:.4f}")
                print(f"      Cosine similarity: {cos_sim:.4f}")
    
    # Step 8: Attention analysis (for GATv2)
    print("\n8. Analyzing attention weights...")
    if hasattr(model, 'attn_weights') and model.attn_weights:
        print("   Attention weights captured for edge types:")
        for key in sorted(model.attn_weights.keys()):
            attn = model.attn_weights[key]
            print(f"   - {key}: shape {attn.shape}")
    
    print("\nExample completed!")

if __name__ == "__main__":
    main() 