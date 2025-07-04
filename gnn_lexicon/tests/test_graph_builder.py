"""
Tests for graph construction.
"""

import pytest
import torch
from ..src.data_loading import create_toy_graph
from ..src.graph_builder import build_hetero_graph, split_edges

def test_build_hetero_graph():
    """Test building heterogeneous graph from toy data."""
    raw_data = create_toy_graph()
    data = build_hetero_graph(raw_data)
    
    # Check node types
    assert "Word" in data.node_types
    assert "Morpheme" in data.node_types
    assert "Form" in data.node_types
    assert "Sense" in data.node_types
    assert "Language" in data.node_types
    
    # Check edge types
    assert ("Word", "HAS_FORM", "Form") in data.edge_types
    assert ("Form", "OF_WORD", "Word") in data.edge_types
    assert ("Word", "HAS_SENSE", "Sense") in data.edge_types
    
    # Check node features
    assert data["Word"].x.shape[0] == len(raw_data["words"])
    assert data["Word"].x.shape[1] == 64  # Default char-CNN embedding dim
    
    # Check edge indices
    assert data["Word", "HAS_FORM", "Form"].edge_index.shape[0] == 2

def test_split_edges():
    """Test edge splitting for train/val/test."""
    raw_data = create_toy_graph()
    data = build_hetero_graph(raw_data)
    
    train_data, val_data, test_data = split_edges(data, train_ratio=0.8, val_ratio=0.1)
    
    # Check that nodes are preserved
    for node_type in data.node_types:
        assert train_data[node_type].x.shape == data[node_type].x.shape
        assert val_data[node_type].x.shape == data[node_type].x.shape
        assert test_data[node_type].x.shape == data[node_type].x.shape
    
    # Check edge splits
    for edge_type in data.edge_types:
        if edge_type in data.edge_types and data[edge_type].edge_index.size(1) > 0:
            total_edges = data[edge_type].edge_index.size(1)
            train_edges = train_data[edge_type].edge_index.size(1)
            val_edges = val_data[edge_type].edge_index.size(1)
            test_edges = test_data[edge_type].edge_index.size(1)
            
            # Allow for rounding differences
            assert abs(train_edges + val_edges + test_edges - total_edges) <= 1

def test_empty_edge_types():
    """Test handling of empty edge types."""
    raw_data = create_toy_graph()
    # Remove all edges of a certain type
    raw_data["shares_etym"] = []
    
    data = build_hetero_graph(raw_data)
    
    # Should still create the edge type but with empty edge_index
    assert ("Word", "SHARES_ETYMOLOGY", "Word") in data.edge_types
    assert data["Word", "SHARES_ETYMOLOGY", "Word"].edge_index.shape[1] == 0

def test_char_cnn_embedding():
    """Test character CNN embedding."""
    from ..src.utils import char_cnn_embed
    
    texts = ["hello", "world", ""]
    embeddings = char_cnn_embed(texts)
    
    assert embeddings.shape == (3, 64)  # 3 texts, 64-dim embeddings
    assert torch.isfinite(embeddings).all()

if __name__ == "__main__":
    pytest.main([__file__]) 