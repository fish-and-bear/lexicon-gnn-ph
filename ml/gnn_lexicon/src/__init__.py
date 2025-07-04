"""
Philippine Lexicon GNN Toolkit.
A graph neural network system for modeling Philippine language lexicons.
"""

from .data_loading import (
    load_pg_connection, 
    fetch_graph_from_postgres
)

from .graph_builder import (
    build_hetero_graph,
    split_edges,
    NodeTypes,
    EdgeTypes
)

from .models import (
    HeteroGNN,
    HeteroRGCN,
    HeteroGraphSAGE,
    HeteroGATv2,
    LinkPredictor,
    RelationClassifier,
    create_model
)

from .training import (
    train_epoch,
    train_gnn,
    run_ablation,
    negative_sampling
)

from .evaluation import (
    evaluate_link_prediction,
    evaluate_hits_at_k,
    evaluate_relation_classification,
    evaluate_node_similarity,
    print_evaluation_summary
)

from .utils import (
    char_cnn_embed,
    normalize_numeric,
    load_config,
    get_default_config,
    save_model,
    load_model
)

__version__ = "0.1.0"
__author__ = "Philippine Lexicon GNN Team"

__all__ = [
    # Data loading
    "load_pg_connection",
    "fetch_graph_from_postgres",
    # Graph building
    "build_hetero_graph",
    "split_edges",
    "NodeTypes",
    "EdgeTypes",
    # Models
    "HeteroGNN",
    "HeteroRGCN",
    "HeteroGraphSAGE",
    "HeteroGATv2",
    "LinkPredictor",
    "RelationClassifier",
    "create_model",
    # Training
    "train_epoch",
    "train_gnn",
    "run_ablation",
    "negative_sampling",
    # Evaluation
    "evaluate_link_prediction",
    "evaluate_hits_at_k",
    "evaluate_relation_classification",
    "evaluate_node_similarity",
    "print_evaluation_summary",
    # Utils
    "char_cnn_embed",
    "normalize_numeric",
    "load_config",
    "get_default_config",
    "save_model",
    "load_model"
] 