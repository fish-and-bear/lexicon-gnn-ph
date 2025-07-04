#!/usr/bin/env python3
"""
Dynamic export: Load model config and edge types from checkpoint, build graph accordingly, and export predictions.
"""
import torch
import pandas as pd
import numpy as np
import time
import psutil
import re
from src.data_loading import load_pg_connection
from src.models import create_model, LinkPredictor

def parse_checkpoint(model_path):
    checkpoint = torch.load(model_path, map_location="cpu")
    model_state = checkpoint["model"]
    keys = list(model_state.keys())
    edge_types = set()
    hidden_dim = None
    heads = None
    for k in keys:
        m = re.match(r"convs\.\d+\.convs\.<(.+?)>", k)
        if m:
            edge = m.group(1)
            parts = edge.split("___")
            if len(parts) == 3:
                edge_types.add((parts[0], parts[1], parts[2]))
        if ".att" in k and len(model_state[k].shape) == 3:
            heads = model_state[k].shape[1]
            hidden_dim = model_state[k].shape[1] * model_state[k].shape[2]
    # Try to infer out_dim from link predictor
    out_dim = None
    if "link_predictor" in checkpoint:
        link_state = checkpoint["link_predictor"]
        for lk, v in link_state.items():
            if "lin1.weight" in lk and len(v.shape) == 2:
                out_dim = v.shape[0]
    return {
        "edge_types": list(edge_types),
        "hidden_dim": hidden_dim,
        "out_dim": out_dim,
        "heads": heads
    }

def fetch_graph_dynamic(conn, edge_types):
    cur = conn.cursor()
    # Always get words
    cur.execute("SELECT id, lemma FROM words WHERE language_code IN ('tl', 'ceb') LIMIT 5000;")
    words = cur.fetchall()
    word_ids = [w[0] for w in words]
    # Always get forms and senses if needed
    forms, senses = [], []
    if any(et[0] == "Word" and et[2] == "Form" for et in edge_types) or any(et[0] == "Form" for et in edge_types):
        cur.execute("SELECT id, form, word_id FROM word_forms WHERE word_id = ANY(%s);", (word_ids,))
        forms = cur.fetchall()
    if any(et[0] == "Word" and et[2] == "Sense" for et in edge_types):
        cur.execute("SELECT id, definition_text, word_id FROM definitions WHERE word_id = ANY(%s);", (word_ids,))
        senses = cur.fetchall()
    # Relations
    cur.execute("SELECT from_word_id, to_word_id, relation_type FROM relations WHERE (from_word_id = ANY(%s) OR to_word_id = ANY(%s));", (word_ids, word_ids))
    relations = cur.fetchall()
    return {
        "words": [{"id": w[0], "lemma": w[1]} for w in words],
        "forms": [{"id": f[0], "form": f[1], "word_id": f[2]} for f in forms],
        "senses": [{"id": s[0], "definition_text": s[1], "word_id": s[2]} for s in senses],
        "relations": relations
    }

def build_hetero_graph_dynamic(raw_data, edge_types):
    from torch_geometric.data import HeteroData
    data = HeteroData()
    # Node features: 64-dim char features
    word_features = []
    for word in raw_data["words"]:
        lemma = word["lemma"] or ""
        features = [len(lemma)] + [ord(c) % 64 for c in lemma[:63]]
        features = features[:64] + [0] * (64 - len(features))
        word_features.append(features)
    data["Word"].x = torch.tensor(word_features, dtype=torch.float32)
    form_features = []
    for form in raw_data["forms"]:
        form_text = form["form"] or ""
        features = [len(form_text)] + [ord(c) % 64 for c in form_text[:63]]
        features = features[:64] + [0] * (64 - len(features))
        form_features.append(features)
    if form_features:
        data["Form"].x = torch.tensor(form_features, dtype=torch.float32)
    sense_features = []
    for sense in raw_data["senses"]:
        sense_text = sense["definition_text"] or ""
        features = [len(sense_text)] + [ord(c) % 64 for c in sense_text[:63]]
        features = features[:64] + [0] * (64 - len(features))
        sense_features.append(features)
    if sense_features:
        data["Sense"].x = torch.tensor(sense_features, dtype=torch.float32)
    # Build edge indices for only those in edge_types, always include all edge_types
    word_to_idx = {w["id"]: i for i, w in enumerate(raw_data["words"])}
    form_to_idx = {f["id"]: i for i, f in enumerate(raw_data["forms"])}
    sense_to_idx = {s["id"]: i for i, s in enumerate(raw_data["senses"])}
    for et in edge_types:
        src, rel, dst = et
        edge_tensor = None
        if src == "Word" and dst == "Form" and rel == "HAS_FORM":
            edges = [[word_to_idx[f["word_id"]], form_to_idx[f["id"]]] for f in raw_data["forms"] if f["word_id"] in word_to_idx and f["id"] in form_to_idx]
            if edges:
                edge_tensor = torch.tensor(edges, dtype=torch.long).t()
        if src == "Form" and dst == "Word" and rel == "OF_WORD":
            edges = [[form_to_idx[f["id"]], word_to_idx[f["word_id"]]] for f in raw_data["forms"] if f["id"] in form_to_idx and f["word_id"] in word_to_idx]
            if edges:
                edge_tensor = torch.tensor(edges, dtype=torch.long).t()
        if src == "Word" and dst == "Sense" and rel == "HAS_SENSE":
            edges = [[word_to_idx[s["word_id"]], sense_to_idx[s["id"]]] for s in raw_data["senses"] if s["word_id"] in word_to_idx and s["id"] in sense_to_idx]
            if edges:
                edge_tensor = torch.tensor(edges, dtype=torch.long).t()
        if src == "Word" and dst == "Word":
            rel_map = {"DERIVED_FROM": "root_of", "SYNONYM_OF": "synonym", "SHARES_PHONOLOGY": "shares_phonology"}
            if rel in rel_map:
                edges = [[word_to_idx[r[0]], word_to_idx[r[1]]] for r in raw_data["relations"] if r[2] == rel_map[rel] and r[0] in word_to_idx and r[1] in word_to_idx]
                if edges:
                    edge_tensor = torch.tensor(edges, dtype=torch.long).t()
        # Always create the edge type, even if empty
        if edge_tensor is not None:
            data[et].edge_index = edge_tensor
        else:
            data[et].edge_index = torch.zeros((2, 0), dtype=torch.long)
    return data

def load_existing_model(model_path, model_type, config, metadata, in_channels_dict, device, link_predictor_dim=64):
    print(f"Loading model from {model_path}...")
    model = create_model(model_type, metadata, in_channels_dict, config).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    result = model.load_state_dict(checkpoint["model"], strict=False)
    print("Missing keys:", result.missing_keys)
    print("Unexpected keys:", result.unexpected_keys)
    link_predictor = LinkPredictor(link_predictor_dim).to(device)
    link_predictor.load_state_dict(checkpoint["link_predictor"])
    print(f"Model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, link_predictor

def export_predictions(model, link_predictor, data, raw_data, output_path):
    model.eval()
    link_predictor.eval()
    word_nodes = list(range(data["Word"].x.size(0)))
    gold_edges = set()
    for edge_type in data.edge_types:
        if edge_type[0] == "Word" and edge_type[2] == "Word":
            edge_index = data[edge_type].edge_index
            for i in range(edge_index.size(1)):
                src = edge_index[0, i].item()
                dst = edge_index[1, i].item()
                gold_edges.add((src, dst))
    print(f"Found {len(gold_edges)} existing Word-Word edges")
    scores = []
    with torch.no_grad():
        out_dict = model(data.x_dict, data.edge_index_dict)
        word_emb = out_dict["Word"]
        num_samples = 10000
        for _ in range(num_samples):
            i = np.random.randint(0, len(word_nodes))
            j = np.random.randint(0, len(word_nodes))
            if i == j or (i, j) in gold_edges:
                continue
            emb1 = word_emb[i].unsqueeze(0)
            emb2 = word_emb[j].unsqueeze(0)
            score = torch.sigmoid(link_predictor(emb1, emb2)).item()
            scores.append((i, j, score))
    scores.sort(key=lambda x: -x[2])
    top100 = scores[:100]
    idx2lemma = {idx: w["lemma"] for idx, w in enumerate(raw_data["words"])}
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
    print("\nTop 10 predictions:")
    for i, row in df.head(10).iterrows():
        print(f"{row['word1']} <-> {row['word2']}: {row['score']:.4f} ({row['predicted_relation']})")

def main():
    model_path = "gatv2_model.pt"  # Change to your model
    model_type = "gatv2"
    print(f"Parsing checkpoint: {model_path}")
    info = parse_checkpoint(model_path)
    print(f"Edge types: {info['edge_types']}")
    # HARDCODED: Use correct hidden_dim, heads, and out_dim for this checkpoint
    if model_type == "gatv2":
        info["hidden_dim"] = 128  # per-head out_channels
        info["heads"] = 4
        info["out_dim"] = 64  # final node embedding size
    elif model_type == "sage":
        info["hidden_dim"] = 128
        info["heads"] = 1
        info["out_dim"] = 64
    print(f"hidden_dim: {info['hidden_dim']}, out_dim: {info['out_dim']}, heads: {info['heads']}")
    db_config = {
        "host": "localhost",
        "port": 5432,
        "database": "fil_dict_db",
        "user": "postgres",
        "password": "postgres"
    }
    conn = load_pg_connection(db_config)
    raw_data = fetch_graph_dynamic(conn, info['edge_types'])
    data = build_hetero_graph_dynamic(raw_data, info['edge_types'])
    metadata = (list(data.x_dict.keys()), list(data.edge_types))
    in_channels_dict = {node_type: data[node_type].x.size(-1) for node_type in data.x_dict.keys()}
    config = {
        "hidden_dim": info["hidden_dim"],
        "out_dim": info["out_dim"],
        "num_layers": 2,
        "heads": info["heads"],
        "num_relations": len(info["edge_types"])
    }
    model, link_predictor = load_existing_model(model_path, model_type, config, metadata, in_channels_dict, "cpu", link_predictor_dim=64)
    export_predictions(model, link_predictor, data, raw_data, "manual_judgement_dynamic.csv")
    print("Done.")

if __name__ == "__main__":
    main() 