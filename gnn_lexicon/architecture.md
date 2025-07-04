# GNN Lexicon Architecture

## System Overview

The Philippine Lexicon GNN toolkit is designed as a modular system for modeling linguistic relationships using graph neural networks.

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Data Source   │────▶│  Graph Builder   │────▶│   GNN Models    │
│  (PostgreSQL/   │     │  (HeteroData)    │     │  (R-GCN/SAGE/   │
│   JSON/Toy)     │     │                  │     │    GATv2)       │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Evaluation    │◀────│    Training      │◀────│   Task Heads    │
│  (ROC-AUC,      │     │  (Neighbor       │     │  (Link Pred/    │
│   Hits@k)       │     │   Sampling)      │     │   Relation)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Component Details

### 1. Data Loading (`data_loading.py`)

Handles multiple data sources:
- **PostgreSQL**: Direct connection to Philippine dictionary database
- **JSON**: Offline data files for reproducibility
- **Toy Graph**: Small synthetic graph for testing

Key functions:
- `load_pg_connection()`: Establish database connection
- `fetch_graph_from_postgres()`: Extract nodes and edges
- `create_toy_graph()`: Generate synthetic test data

### 2. Graph Builder (`graph_builder.py`)

Constructs PyTorch Geometric `HeteroData` objects:

```python
HeteroData(
  Word={ x: [num_words, 64], freq: [num_words, 1] },
  Morpheme={ x: [num_morphemes, 64] },
  Form={ x: [num_forms, 64] },
  Sense={ x: [num_senses, 64] },
  Language={ x: [num_langs, 64] },
  (Word, HAS_FORM, Form)={ edge_index: [2, num_edges] },
  (Word, DERIVED_FROM, Word)={ edge_index: [2, num_edges] },
  ...
)
```

Features:
- Character-level CNN embeddings for text
- Normalized numeric features
- ID mapping for consistent indexing
- Train/val/test edge splitting

### 3. Models (`models.py`)

#### Base Architecture

All models inherit from `HeteroGNN`:
- Handles heterogeneous node and edge types
- Supports multi-layer message passing
- Configurable hidden dimensions

#### R-GCN (Relational GCN)
```python
# Relation-specific transformations
W_r * h_i + b_r  # For each relation type r
```
- Best for: Many relation types
- Memory: O(|R| × d × d) for |R| relations

#### GraphSAGE
```python
# Sampling and aggregation
h_i = σ(W · CONCAT(h_i, AGG({h_j | j ∈ N(i)})))
```
- Best for: Large graphs
- Scalability: Mini-batch training

#### GATv2
```python
# Attention mechanism
α_ij = softmax(LeakyReLU(a^T[W_h h_i || W_h h_j]))
```
- Best for: Interpretability
- Feature: Attention weight export

### 4. Training (`training.py`)

Features:
- **Neighbor Sampling**: Efficient mini-batch training
- **Negative Sampling**: Generate negative edges for link prediction
- **Mixed Precision**: Optional AMP for faster training
- **Early Stopping**: Prevent overfitting
- **Gradient Clipping**: Stable training

Training loop:
1. Sample batch of nodes
2. Extract k-hop neighborhood
3. Forward pass through GNN
4. Compute link prediction loss
5. Backpropagate and update

### 5. Evaluation (`evaluation.py`)

Metrics:
- **ROC-AUC**: Overall link prediction quality
- **Hits@k**: Ranking-based evaluation
- **Accuracy/F1**: For relation classification

Special features:
- Efficient negative sampling
- Batch evaluation for large graphs
- Per-edge-type metrics

### 6. CLI (`cli.py`)

Commands:
- `train`: Train a new model
- `evaluate`: Test model performance
- `infer`: Query word relationships
- `ablate`: Study edge type importance
- `export`: Save graph to JSON

## Design Decisions

### 1. Heterogeneous Graph Structure

Why: Philippine languages have rich morphology and multiple relationship types that are best modeled with different node and edge types.

### 2. Character-level Embeddings

Why: Handle out-of-vocabulary words and capture morphological patterns common in Philippine languages.

### 3. Neighbor Sampling

Why: Scale to large dictionaries (100k+ words) while maintaining reasonable memory usage.

### 4. Link Prediction Focus

Why: Many dictionary relationships are incomplete; predicting missing links is a valuable task.

## Extension Points

### Adding New Node Types

1. Update `NodeTypes` in `graph_builder.py`
2. Add extraction logic in `fetch_graph_from_postgres()`
3. Add embedding logic in `build_hetero_graph()`

### Adding New Edge Types

1. Update `EdgeTypes` in `graph_builder.py`
2. Add edge extraction in data loading
3. Update model metadata handling

### Custom Embeddings

Replace `char_cnn_embed()` with:
- Pre-trained word embeddings
- Subword tokenizers (BPE, WordPiece)
- Phonetic encoders

### New Tasks

Extend beyond link prediction:
- Node classification (POS tagging)
- Graph-level tasks (language identification)
- Sequence generation (morphological inflection)

## Performance Considerations

### Memory Usage

- Node features: O(|V| × d)
- Edge storage: O(|E|)
- Model parameters: O(L × d²) for L layers

### Computation

- Forward pass: O(|E| × d) per layer
- Neighbor sampling: O(k^L) for L layers, k neighbors
- Attention (GATv2): O(|E| × heads)

### Optimization Tips

1. Use sparse operations for large graphs
2. Enable mixed precision training
3. Tune batch size and neighbor sampling
4. Consider edge dropout for regularization
5. Use gradient accumulation for large models 