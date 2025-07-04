#!/usr/bin/env python3
"""
Quick script to train a fresh model and export 100 predictions for manual judgement using GraphSAGE.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
import psutil

from src.data_loading import fetch_graph_from_postgres, load_pg_connection
from src.graph_builder import build_hetero_graph
from src.models import LinkPredictor

def create_simple_sage(metadata, in_channels_dict, config):
    class SimpleHeteroSAGE(torch.nn.Module):
        def __init__(self, metadata, in_channels_dict, config):
            super().__init__()
            self.hidden_dim = config["hidden_dim"]
            self.out_dim = config["out_dim"]
            self.num_layers = config["num_layers"]
            self.node_encoders = torch.nn.ModuleDict()
            for node_type in metadata[0]:
                if node_type in in_channels_dict:
                    self.node_encoders[node_type] = Linear(
                        in_channels_dict[node_type], self.hidden_dim
                    )
                else:
                    self.node_encoders[node_type] = torch.nn.Embedding(1000, self.hidden_dim)
            self.convs = torch.nn.ModuleList()
            for _ in range(self.num_layers):
                conv_dict = {}
                for edge_type in metadata[1]:
                    conv_dict[edge_type] = SAGEConv((self.hidden_dim, self.hidden_dim), self.hidden_dim)
                self.convs.append(HeteroConv(conv_dict, aggr='sum'))
            self.lin = Linear(self.hidden_dim, self.out_dim)
        def forward(self, x_dict, edge_index_dict):
            h_dict = {}
            for node_type, x in x_dict.items():
                if node_type in self.node_encoders:
                    if isinstance(self.node_encoders[node_type], torch.nn.Embedding):
                        h_dict[node_type] = self.node_encoders[node_type](x.long().squeeze(-1))
                    else:
                        h_dict[node_type] = self.node_encoders[node_type](x)
            for conv in self.convs:
                h_dict = conv(h_dict, edge_index_dict)
                h_dict = {k: F.relu(v) for k, v in h_dict.items()}
            out_dict = {k: self.lin(v) for k, v in h_dict.items()}
            return out_dict
    return SimpleHeteroSAGE(metadata, in_channels_dict, config)

def train_simple_model(model, data, device, epochs=10):
    link_predictor = LinkPredictor(model.out_dim).to(device)
    word_edges = []
    for edge_type in data.edge_types:
        if edge_type[0] == "Word" and edge_type[2] == "Word":
            edge_index = data[edge_type].edge_index
            for i in range(edge_index.size(1)):
                src = edge_index[0, i].item()
                dst = edge_index[1, i].item()
                word_edges.append((src, dst, 1))
    num_words = data["Word"].x.size(0)
    num_neg = len(word_edges)
    neg_edges = []
    for _ in range(num_neg):
        src = np.random.randint(0, num_words)
        dst = np.random.randint(0, num_words)
        if src != dst and (src, dst, 1) not in word_edges:
            neg_edges.append((src, dst, 0))
    all_edges = word_edges + neg_edges
    np.random.shuffle(all_edges)
    train_edges, _ = train_test_split(all_edges, test_size=0.2, random_state=42)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(link_predictor.parameters()), lr=0.001)
    model.train()
    link_predictor.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out_dict = model(data.x_dict, data.edge_index_dict)
        word_emb = out_dict["Word"]
        batch_size = min(1024, len(train_edges))
        batch_edges = train_edges[:batch_size]
        loss = 0
        for src, dst, label in batch_edges:
            emb1 = word_emb[src].unsqueeze(0)
            emb2 = word_emb[dst].unsqueeze(0)
            pred = torch.sigmoid(link_predictor(emb1, emb2))
            label_tensor = torch.tensor([label], dtype=torch.float32, device=device)
            loss += F.binary_cross_entropy(pred, label_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 2 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
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
            "score": score
        })
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Top 100 predictions exported to {output_path}")
    print("\nTop 10 predictions:")
    for i, row in df.head(10).iterrows():
        print(f"{row['word1']} <-> {row['word2']}: {row['score']:.4f}")

def main():
    print("Loading data from PostgreSQL...")
    start_time = time.time()
    db_config = {
        "host": "localhost",
        "port": 5432,
        "database": "fil_dict_db",
        "user": "postgres",
        "password": "postgres"
    }
    conn = load_pg_connection(db_config)
    if conn is None:
        print("Failed to connect to database")
        return
    raw_data = fetch_graph_from_postgres(conn)
    data = build_hetero_graph(raw_data, device="cpu")
    print(f"Graph loaded in {time.time() - start_time:.2f}s")
    print(f"Word nodes: {data['Word'].x.size(0)}")
    config = {"hidden_dim": 128, "out_dim": 64, "num_layers": 2}
    metadata = (list(data.x_dict.keys()), list(data.edge_types))
    in_channels_dict = {node_type: data[node_type].x.size(-1) for node_type in data.x_dict.keys()}
    model = create_simple_sage(metadata, in_channels_dict, config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nTraining model...")
    train_start = time.time()
    model, link_predictor = train_simple_model(model, data, "cpu", epochs=6)
    print(f"Training completed in {time.time() - train_start:.2f}s")
    print("\nExporting predictions...")
    export_predictions(model, link_predictor, data, raw_data, "manual_judgement_predictions.csv")
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