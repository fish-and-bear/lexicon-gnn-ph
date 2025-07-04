#!/usr/bin/env python3
"""
Simple Working ML Pipeline for FilRelex
This replaces the complex, broken pipeline with something that actually works.
"""

import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn
import pandas as pd
import numpy as np
import json
from sqlalchemy import create_engine, text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleGNN(nn.Module):
    """Simple Graph Neural Network that actually works."""
    
    def __init__(self, in_dim, hidden_dim=128, out_dim=64):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim, activation=nn.ReLU())
        self.conv2 = dglnn.GraphConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, graph, features):
        h = self.conv1(graph, features)
        h = self.dropout(h)
        h = self.conv2(graph, h)
        return h

def load_data():
    """Load data from database."""
    logger.info("Loading data from database...")
    
    with open("my_db_config.json", "r") as f:
        config = json.load(f)["database"]
    
    engine = create_engine(
        f"postgresql://{config[\"user\"]}:{config[\"password\"]}@"
        f"{config[\"host\"]}:{config[\"port\"]}/{config[\"database\"]}"
        f"?sslmode={config[\"ssl_mode\"]}"
    )
    
    # Load words (limited for testing)
    words_df = pd.read_sql("""
        SELECT w.id, w.lemma, w.language_code
        FROM words w 
        WHERE w.lemma IS NOT NULL 
        LIMIT 1000
    """, engine)
    
    logger.info(f"Loaded {len(words_df)} words")
    
    # Load relations
    word_ids = words_df["id"].tolist()
    if len(word_ids) > 0:
        placeholders = ",".join(map(str, word_ids))
        relations_df = pd.read_sql(f"""
            SELECT from_word_id, to_word_id, relation_type
            FROM relations 
            WHERE from_word_id IN ({placeholders})
            AND to_word_id IN ({placeholders})
        """, engine)
    else:
        relations_df = pd.DataFrame(columns=["from_word_id", "to_word_id", "relation_type"])
    
    logger.info(f"Loaded {len(relations_df)} relations")
    
    return words_df, relations_df

def create_graph(words_df, relations_df):
    """Create DGL graph from data."""
    logger.info("Creating graph...")
    
    # Create node mapping
    word_ids = words_df["id"].values
    id_to_idx = {word_id: idx for idx, word_id in enumerate(word_ids)}
    
    # Create edges
    if len(relations_df) > 0:
        valid_rels = relations_df[
            relations_df["from_word_id"].isin(word_ids) & 
            relations_df["to_word_id"].isin(word_ids)
        ]
        
        if len(valid_rels) > 0:
            src = [id_to_idx[id] for id in valid_rels["from_word_id"]]
            dst = [id_to_idx[id] for id in valid_rels["to_word_id"]]
        else:
            # Create chain if no relations
            src = list(range(len(word_ids) - 1))
            dst = list(range(1, len(word_ids)))
    else:
        # Create chain if no relations
        src = list(range(len(word_ids) - 1))
        dst = list(range(1, len(word_ids)))
    
    # Create DGL graph
    graph = dgl.graph((src, dst), num_nodes=len(word_ids))
    graph = dgl.add_self_loop(graph)
    
    # Create simple features (random for now)
    features = np.random.randn(len(word_ids), 64)
    graph.ndata["feat"] = torch.FloatTensor(features)
    
    logger.info(f"Created graph: {graph}")
    return graph

def train_model(graph, epochs=20):
    """Simple training loop."""
    logger.info("Training model...")
    
    model = SimpleGNN(graph.ndata["feat"].shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        embeddings = model(graph, graph.ndata["feat"])
        
        # Simple reconstruction loss
        loss = criterion(embeddings, graph.ndata["feat"][:, :embeddings.shape[1]])
        
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch:3d}, Loss: {loss.item():.4f}")
    
    logger.info("Training completed!")
    return model

def main():
    """Run the simple pipeline."""
    print("🚀 Running Simple FilRelex ML Pipeline")
    print("=" * 50)
    
    try:
        # Load data
        words_df, relations_df = load_data()
        
        # Create graph
        graph = create_graph(words_df, relations_df)
        
        # Train model
        model = train_model(graph, epochs=20)
        
        # Save model
        torch.save(model.state_dict(), "simple_model.pt")
        logger.info("Model saved to simple_model.pt")
        
        print("\n🎉 Pipeline completed successfully!")
        print("✅ Model trained and saved")
        print("✅ Graph created with", graph.num_nodes(), "nodes")
        print("✅ Features shape:", graph.ndata[\"feat\"].shape)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

