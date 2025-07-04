"""
Tests for training functionality.
"""

import pytest
import torch
import torch.nn as nn
from ..src.data_loading import create_toy_graph
from ..src.graph_builder import build_hetero_graph, split_edges
from ..src.models import create_model, LinkPredictor
from ..src.training import negative_sampling, train_epoch, train_gnn

@pytest.fixture
def setup_training():
    """Setup training components."""
    # Create data
    raw_data = create_toy_graph()
    data = build_hetero_graph(raw_data)
    train_data, val_data, test_data = split_edges(data, train_ratio=0.6, val_ratio=0.2)
    
    # Create model
    metadata = (data.node_types, data.edge_types)
    in_channels_dict = {
        node_type: data[node_type].x.size(1) 
        for node_type in data.node_types
    }
    config = {
        "hidden_dim": 32,
        "out_dim": 16,
        "num_layers": 2,
        "lr": 0.01,
        "batch_size": 2,
        "epochs": 3,
        "grad_clip": 1.0,
        "early_stopping_patience": 2
    }
    
    device = torch.device("cpu")
    model = create_model("rgcn", metadata, in_channels_dict, config).to(device)
    link_predictor = LinkPredictor(config["out_dim"]).to(device)
    
    return {
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "model": model,
        "link_predictor": link_predictor,
        "config": config,
        "device": device
    }

def test_negative_sampling():
    """Test negative edge sampling."""
    # Create simple edge index
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    num_nodes = 4
    
    # Sample negative edges
    neg_edges = negative_sampling(edge_index, num_nodes, num_neg_samples=2)
    
    assert neg_edges.shape == (2, 6)  # 3 positive edges * 2 neg samples
    
    # Check no self-loops
    assert (neg_edges[0] != neg_edges[1]).all()
    
    # Check no overlap with positive edges
    pos_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    neg_set = set(zip(neg_edges[0].tolist(), neg_edges[1].tolist()))
    assert len(pos_set.intersection(neg_set)) == 0

def test_train_epoch(setup_training):
    """Test single training epoch."""
    components = setup_training
    
    optimizer = torch.optim.Adam(
        list(components["model"].parameters()) + 
        list(components["link_predictor"].parameters()),
        lr=components["config"]["lr"]
    )
    
    # Run one epoch
    loss = train_epoch(
        components["model"],
        components["train_data"],
        optimizer,
        components["link_predictor"],
        components["device"],
        components["config"]
    )
    
    assert isinstance(loss, float)
    assert loss > 0

def test_train_gnn_smoke(setup_training):
    """Smoke test for full training loop."""
    components = setup_training
    
    optimizer = torch.optim.Adam(
        components["model"].parameters(),
        lr=components["config"]["lr"]
    )
    
    # Run short training
    results = train_gnn(
        components["model"],
        components["train_data"],
        components["val_data"],
        optimizer,
        components["device"],
        components["config"],
        amp=False
    )
    
    assert "history" in results
    assert "best_state" in results
    assert "best_val_auc" in results
    
    # Check history
    assert len(results["history"]["train_loss"]) <= components["config"]["epochs"]
    assert len(results["history"]["val_auc"]) == len(results["history"]["train_loss"])
    
    # Check best state
    assert results["best_state"] is not None
    assert "model" in results["best_state"]
    assert "link_predictor" in results["best_state"]

def test_gradient_clipping(setup_training):
    """Test gradient clipping."""
    components = setup_training
    
    # Create optimizer with large learning rate to trigger clipping
    optimizer = torch.optim.SGD(
        list(components["model"].parameters()) + 
        list(components["link_predictor"].parameters()),
        lr=10.0  # Very large LR
    )
    
    # Store initial parameters
    initial_params = [p.clone() for p in components["model"].parameters()]
    
    # Run one epoch with gradient clipping
    loss = train_epoch(
        components["model"],
        components["train_data"],
        optimizer,
        components["link_predictor"],
        components["device"],
        components["config"]
    )
    
    # Check parameters changed but not exploded
    for p_init, p_after in zip(initial_params, components["model"].parameters()):
        assert not torch.equal(p_init, p_after)  # Parameters changed
        assert torch.isfinite(p_after).all()  # No NaN or inf

def test_early_stopping(setup_training):
    """Test early stopping functionality."""
    components = setup_training
    
    # Modify config for quick early stopping
    components["config"]["early_stopping_patience"] = 1
    components["config"]["epochs"] = 10
    
    optimizer = torch.optim.Adam(
        components["model"].parameters(),
        lr=0.0  # Zero LR to ensure no improvement
    )
    
    results = train_gnn(
        components["model"],
        components["train_data"],
        components["val_data"],
        optimizer,
        components["device"],
        components["config"]
    )
    
    # Should stop early
    assert len(results["history"]["train_loss"]) < components["config"]["epochs"]

if __name__ == "__main__":
    pytest.main([__file__]) 