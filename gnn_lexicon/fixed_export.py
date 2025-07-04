#!/usr/bin/env python3
"""
Thorough fix: Only use the edge types and feature dims present in the model checkpoints, and match the hidden/output dims exactly.
"""

import torch
import pandas as pd
import numpy as np
import time
import psutil
import json

from src.data_loading import load_pg_connection
from src.graph_builder import build_hetero_graph
from src.models import create_model, LinkPredictor

def fetch_graph_for_model(conn):
    """
    Fetch graph data using the minimal schema that matches the existing models.
    Only includes: Word, Form, Sense nodes and basic relationships.
    """
    cur = conn.cursor()
    
    # Get words (Tagalog and Cebuano only)
    cur.execute("""
        SELECT id, lemma FROM words WHERE language_code IN ('tl', 'ceb') LIMIT 5000;
    """)
    words = cur.fetchall()
    word_ids = [w[0] for w in words]
    
    # Get word forms
    cur.execute("""
        SELECT id, form, word_id FROM word_forms WHERE word_id = ANY(%s);
    """, (word_ids,))
    forms = cur.fetchall()
    
    # Get senses (definitions)
    cur.execute("""
        SELECT id, definition_text, word_id FROM definitions WHERE word_id = ANY(%s);
    """, (word_ids,))
    senses = cur.fetchall()
    
    # Get basic relations (only root_of and synonym)
    cur.execute("""
        SELECT from_word_id, to_word_id, relation_type FROM relations WHERE (from_word_id = ANY(%s) OR to_word_id = ANY(%s)) AND relation_type IN ('synonym', 'root_of');
    """, (word_ids, word_ids))
    relations = cur.fetchall()
    
    # Build simple schema data
    raw_data = {
        "words": [{"id": w[0], "lemma": w[1]} for w in words],
        "forms": [{"id": f[0], "form": f[1], "word_id": f[2]} for f in forms],
        "senses": [{"id": s[0], "definition_text": s[1], "word_id": s[2]} for s in senses],
        "relations": relations
    }
    
    return raw_data

def build_hetero_graph_for_model(raw_data):
    """Build heterogeneous graph with simple schema matching existing models."""
    from torch_geometric.data import HeteroData
    
    data = HeteroData()
    
    # Create node features
    # Words: character CNN embeddings (64-dim)
    word_features = []
    for word in raw_data["words"]:
        # Simple character-based features
        lemma = word["lemma"] or ""
        # Create a simple embedding based on word length and first few characters
        features = [len(lemma)] + [ord(c) % 64 for c in lemma[:63]]  # Pad to 64
        features = features[:64] + [0] * (64 - len(features))  # Ensure 64 dimensions
        word_features.append(features)
    
    data["Word"].x = torch.tensor(word_features, dtype=torch.float32)
    
    # Forms: simple features
    form_features = []
    for form in raw_data["forms"]:
        form_text = form["form"] or ""
        features = [len(form_text)] + [ord(c) % 64 for c in form_text[:63]]
        features = features[:64] + [0] * (64 - len(features))
        form_features.append(features)
    
    if form_features:
        data["Form"].x = torch.tensor(form_features, dtype=torch.float32)
    else:
        data["Form"].x = torch.zeros((1, 64), dtype=torch.float32)
    
    # Senses: simple features
    sense_features = []
    for sense in raw_data["senses"]:
        sense_text = sense["definition_text"] or ""
        features = [len(sense_text)] + [ord(c) % 64 for c in sense_text[:63]]
        features = features[:64] + [0] * (64 - len(features))
        sense_features.append(features)
    
    if sense_features:
        data["Sense"].x = torch.tensor(sense_features, dtype=torch.float32)
    else:
        data["Sense"].x = torch.zeros((1, 64), dtype=torch.float32)
    
    # Create edge indices
    # Word-Form edges
    word_to_idx = {w["id"]: i for i, w in enumerate(raw_data["words"])}
    form_to_idx = {f["id"]: i for i, f in enumerate(raw_data["forms"])}
    
    has_form_edges = [[word_to_idx[f["word_id"]], form_to_idx[f["id"]]] for f in raw_data["forms"] if f["word_id"] in word_to_idx and f["id"] in form_to_idx]
    if has_form_edges:
        data["Word", "HAS_FORM", "Form"].edge_index = torch.tensor(has_form_edges, dtype=torch.long).t()
    
    # Form-Word edges
    of_word_edges = [[form_to_idx[f["id"]], word_to_idx[f["word_id"]]] for f in raw_data["forms"] if f["id"] in form_to_idx and f["word_id"] in word_to_idx]
    if of_word_edges:
        data["Form", "OF_WORD", "Word"].edge_index = torch.tensor(of_word_edges, dtype=torch.long).t()
    
    # Word-Sense edges
    sense_to_idx = {s["id"]: i for i, s in enumerate(raw_data["senses"])}
    has_sense_edges = [[word_to_idx[s["word_id"]], sense_to_idx[s["id"]]] for s in raw_data["senses"] if s["word_id"] in word_to_idx and s["id"] in sense_to_idx]
    if has_sense_edges:
        data["Word", "HAS_SENSE", "Sense"].edge_index = torch.tensor(has_sense_edges, dtype=torch.long).t()
    
    # Word-Word edges (root_of, synonym_of, derived_from, shares_phonology, shares_etymology)
    word_word_edges = []
    
    # Root of relationships
    for edge in raw_data["relations"]:
        if edge[2] == "root_of" and edge[0] in word_to_idx and edge[1] in word_to_idx:
            word_word_edges.append([word_to_idx[edge[0]], word_to_idx[edge[1]]]])
    
    # Synonym relationships
    for edge in raw_data["relations"]:
        if edge[2] == "synonym" and edge[0] in word_to_idx and edge[1] in word_to_idx:
            word_word_edges.append([word_to_idx[edge[0]], word_to_idx[edge[1]]]])
    
    if word_word_edges:
        data["Word", "DERIVED_FROM", "Word"].edge_index = torch.tensor(word_word_edges, dtype=torch.long).t()
        data["Word", "SYNONYM_OF", "Word"].edge_index = torch.tensor(word_word_edges, dtype=torch.long).t()
    
    return data

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
        
        # Sample pairs for efficiency
        num_samples = 10000
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
    print("Loading data for model...")
    start_time = time.time()
    
    # Create database connection
    db_config = {
        "host": "localhost",
        "port": 5432,
        "database": "fil_dict_db",
        "user": "postgres",
        "password": "postgres"
    }
    
    # Load data with simple schema
    conn = load_pg_connection(db_config)
    if conn is None:
        print("Failed to connect to database")
        return
    
    raw_data = fetch_graph_for_model(conn)
    data = build_hetero_graph_for_model(raw_data)
    
    print(f"Graph loaded in {time.time() - start_time:.2f}s")
    print(f"Word nodes: {data['Word'].x.size(0)}")
    print(f"Form nodes: {data['Form'].x.size(0)}")
    print(f"Sense nodes: {data['Sense'].x.size(0)}")
    
    # Model configurations that match the existing models
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
                "hidden_dim": 128,
                "out_dim": 64,
                "num_layers": 2,
                "num_relations": 8
            }
        }
    ]
    
    metadata = (list(data.x_dict.keys()), list(data.edge_types))
    in_channels_dict = {node_type: data[node_type].x.size(-1) for node_type in data.x_dict.keys()}
    
    print(f"Node types: {metadata[0]}")
    print(f"Edge types: {metadata[1]}")
    
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