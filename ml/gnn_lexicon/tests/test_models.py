"""
Tests for GNN models.
"""

import pytest
import torch
from ..src.data_loading import create_toy_graph
from ..src.graph_builder import build_hetero_graph
from ..src.models import create_model, LinkPredictor, RelationClassifier

@pytest.fixture
def sample_data():
    """Create sample graph data."""
    raw_data = create_toy_graph()
    data = build_hetero_graph(raw_data)
    return data

@pytest.fixture
def model_config():
    """Model configuration."""
    return {
        "hidden_dim": 32,
        "out_dim": 16,
        "num_layers": 2,
        "heads": 2
    }

def test_rgcn_forward(sample_data, model_config):
    """Test R-GCN forward pass."""
    metadata = (sample_data.node_types, sample_data.edge_types)
    in_channels_dict = {
        node_type: sample_data[node_type].x.size(1) 
        for node_type in sample_data.node_types
    }
    
    model = create_model("rgcn", metadata, in_channels_dict, model_config)
    
    # Forward pass
    out_dict = model(sample_data.x_dict, sample_data.edge_index_dict)
    
    # Check output shapes
    for node_type in sample_data.node_types:
        assert node_type in out_dict
        assert out_dict[node_type].shape[0] == sample_data[node_type].x.shape[0]
        assert out_dict[node_type].shape[1] == model_config["out_dim"]

def test_graphsage_forward(sample_data, model_config):
    """Test GraphSAGE forward pass."""
    metadata = (sample_data.node_types, sample_data.edge_types)
    in_channels_dict = {
        node_type: sample_data[node_type].x.size(1) 
        for node_type in sample_data.node_types
    }
    
    model = create_model("sage", metadata, in_channels_dict, model_config)
    
    # Forward pass
    out_dict = model(sample_data.x_dict, sample_data.edge_index_dict)
    
    # Check output shapes
    for node_type in sample_data.node_types:
        assert node_type in out_dict
        assert out_dict[node_type].shape[0] == sample_data[node_type].x.shape[0]
        assert out_dict[node_type].shape[1] == model_config["out_dim"]

def test_gatv2_forward(sample_data, model_config):
    """Test GATv2 forward pass."""
    metadata = (sample_data.node_types, sample_data.edge_types)
    in_channels_dict = {
        node_type: sample_data[node_type].x.size(1) 
        for node_type in sample_data.node_types
    }
    
    model = create_model("gatv2", metadata, in_channels_dict, model_config)
    
    # Forward pass
    out_dict = model(sample_data.x_dict, sample_data.edge_index_dict)
    
    # Check output shapes
    for node_type in sample_data.node_types:
        assert node_type in out_dict
        assert out_dict[node_type].shape[0] == sample_data[node_type].x.shape[0]
        assert out_dict[node_type].shape[1] == model_config["out_dim"]
    
    # Check attention weights were stored
    assert hasattr(model, 'attn_weights')

def test_link_predictor():
    """Test link prediction decoder."""
    in_channels = 16
    batch_size = 10
    
    predictor = LinkPredictor(in_channels)
    
    # Create dummy embeddings
    x_i = torch.randn(batch_size, in_channels)
    x_j = torch.randn(batch_size, in_channels)
    
    # Forward pass
    scores = predictor(x_i, x_j)
    
    assert scores.shape == (batch_size,)
    assert torch.isfinite(scores).all()

def test_relation_classifier():
    """Test relation classification decoder."""
    in_channels = 16
    num_relations = 5
    batch_size = 10
    
    classifier = RelationClassifier(in_channels, num_relations)
    
    # Create dummy embeddings
    x_i = torch.randn(batch_size, in_channels)
    x_j = torch.randn(batch_size, in_channels)
    
    # Forward pass
    logits = classifier(x_i, x_j)
    
    assert logits.shape == (batch_size, num_relations)
    assert torch.isfinite(logits).all()

def test_model_device_transfer(sample_data, model_config):
    """Test moving model to different devices."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    metadata = (sample_data.node_types, sample_data.edge_types)
    in_channels_dict = {
        node_type: sample_data[node_type].x.size(1) 
        for node_type in sample_data.node_types
    }
    
    model = create_model("rgcn", metadata, in_channels_dict, model_config)
    
    # Move to CUDA
    model = model.cuda()
    sample_data = sample_data.cuda()
    
    # Forward pass on CUDA
    out_dict = model(sample_data.x_dict, sample_data.edge_index_dict)
    
    # Check outputs are on CUDA
    for node_type, embeddings in out_dict.items():
        assert embeddings.is_cuda

if __name__ == "__main__":
    pytest.main([__file__]) 