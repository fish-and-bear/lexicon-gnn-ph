#!/usr/bin/env python3
"""
Simple Working ML Pipeline for FilRelex - No DGL Dependencies
This version uses pure PyTorch and NetworkX for graph operations.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import networkx as nx
from sqlalchemy import create_engine, text
import logging
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleGraphNN(nn.Module):
    """Simple Graph Neural Network using adjacency matrix operations."""
    
    def __init__(self, in_dim, hidden_dim=128, out_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, features, adj_matrix):
        # Simple graph convolution: A * X * W
        h = torch.mm(adj_matrix, features)
        h = self.fc1(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return h

def load_data():
    """Load data from database."""
    logger.info("Loading data from database...")
    
    with open("my_db_config.json", "r") as f:
        config = json.load(f)["database"]
    
    conn_str = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}?sslmode={config['ssl_mode']}"
    
    # Load words (limited for testing) - using connection string directly
    words_df = pd.read_sql("""
        SELECT w.id, w.lemma, w.language_code, w.has_baybayin
        FROM words w 
        WHERE w.lemma IS NOT NULL 
        LIMIT 500
    """, conn_str)
    
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
            LIMIT 1000
        """, conn_str)
    else:
        relations_df = pd.DataFrame(columns=["from_word_id", "to_word_id", "relation_type"])
    
    logger.info(f"Loaded {len(relations_df)} relations")
    
    return words_df, relations_df

def create_features(words_df):
    """Create node features from word data."""
    logger.info("Creating features...")
    
    features = []
    
    # Language encoding
    if 'language_code' in words_df.columns:
        lang_encoder = LabelEncoder()
        lang_features = lang_encoder.fit_transform(words_df['language_code'].fillna('unknown'))
        features.append(lang_features.reshape(-1, 1))
    
    # Baybayin flag
    if 'has_baybayin' in words_df.columns:
        baybayin_features = words_df['has_baybayin'].fillna(False).astype(int).values
        features.append(baybayin_features.reshape(-1, 1))
    
    # Lemma length (simple feature)
    if 'lemma' in words_df.columns:
        length_features = words_df['lemma'].str.len().fillna(0).values
        features.append(length_features.reshape(-1, 1))
    
    # Combine features
    if features:
        combined_features = np.concatenate(features, axis=1)
    else:
        combined_features = np.random.randn(len(words_df), 10)
    
    # Pad to minimum size
    min_features = 32
    if combined_features.shape[1] < min_features:
        additional_features = np.random.randn(len(words_df), min_features - combined_features.shape[1])
        combined_features = np.concatenate([combined_features, additional_features], axis=1)
    
    logger.info(f"Created features with shape: {combined_features.shape}")
    return torch.FloatTensor(combined_features)

def create_adjacency_matrix(words_df, relations_df):
    """Create adjacency matrix from relations."""
    logger.info("Creating adjacency matrix...")
    
    word_ids = words_df["id"].values
    id_to_idx = {word_id: idx for idx, word_id in enumerate(word_ids)}
    n_nodes = len(word_ids)
    
    # Initialize adjacency matrix
    adj_matrix = np.eye(n_nodes)  # Self-loops
    
    # Add edges from relations
    if len(relations_df) > 0:
        for _, row in relations_df.iterrows():
            if row['from_word_id'] in id_to_idx and row['to_word_id'] in id_to_idx:
                i = id_to_idx[row['from_word_id']]
                j = id_to_idx[row['to_word_id']]
                adj_matrix[i, j] = 1.0
                adj_matrix[j, i] = 1.0  # Undirected
    
    # Normalize adjacency matrix (simple normalization)
    row_sums = adj_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    adj_matrix = adj_matrix / row_sums
    
    logger.info(f"Created adjacency matrix: {adj_matrix.shape}")
    return torch.FloatTensor(adj_matrix)

def train_model(features, adj_matrix, epochs=30):
    """Train the model."""
    logger.info("Training model...")
    
    model = SimpleGraphNN(features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        embeddings = model(features, adj_matrix)
        
        # Simple reconstruction loss
        target = features[:, :embeddings.shape[1]]
        loss = criterion(embeddings, target)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch:3d}, Loss: {loss.item():.4f}")
    
    logger.info("Training completed!")
    return model

def main():
    """Run the simple pipeline."""
    print("🚀 Running Simple FilRelex ML Pipeline (Pure PyTorch)")
    print("=" * 60)
    
    try:
        # Load data
        words_df, relations_df = load_data()
        
        # Create features and adjacency matrix
        features = create_features(words_df)
        adj_matrix = create_adjacency_matrix(words_df, relations_df)
        
        # Train model
        model = train_model(features, adj_matrix, epochs=30)
        
        # Save model
        torch.save(model.state_dict(), "simple_model_pure_pytorch.pt")
        logger.info("Model saved to simple_model_pure_pytorch.pt")
        
        print("\n🎉 Pipeline completed successfully!")
        print("✅ Model trained and saved")
        print(f"✅ Graph created with {len(words_df)} nodes")
        print(f"✅ Features shape: {features.shape}")
        print(f"✅ Adjacency matrix shape: {adj_matrix.shape}")
        
        # Test the trained model
        model.eval()
        with torch.no_grad():
            embeddings = model(features, adj_matrix)
            print(f"✅ Final embeddings shape: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎊 SUCCESS: ML pipeline is now working!")
    else:
        print("\n💥 FAILED: Check the error messages above.") 