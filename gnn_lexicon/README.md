# Philippine Lexicon GNN Toolkit

A Graph Neural Network toolkit for modeling Philippine language lexicons, supporting heterogeneous graphs with multiple node and edge types.

## Features

- **Heterogeneous Graph Support**: Models words, morphemes, forms, senses, and languages as different node types
- **Multiple GNN Architectures**: R-GCN, GraphSAGE, and GATv2 implementations
- **Rich Edge Types**: Morphological, semantic, phonological, and etymological relationships
- **PostgreSQL Integration**: Direct loading from Philippine dictionary database
- **Link Prediction**: Predict missing relationships between words
- **Relation Classification**: Classify the type of relationships
- **Ablation Studies**: Evaluate the contribution of different edge types

## Installation

```bash
pip install torch torch-geometric psycopg2-binary scikit-learn pyyaml tqdm
```

## Quick Start

### 1. Training a Model

```bash
# Train on toy data
python -m gnn_lexicon.src.cli train --model-type rgcn --epochs 50

# Train on PostgreSQL data
python -m gnn_lexicon.src.cli train --data-source postgres --model-type gatv2 --amp

# Train with custom config
python -m gnn_lexicon.src.cli train --config my_config.yaml --model-path my_model.pt
```

### 2. Evaluating a Model

```bash
python -m gnn_lexicon.src.cli evaluate --model-path model.pt --model-type rgcn
```

### 3. Running Inference

```bash
# Query word relationships
python -m gnn_lexicon.src.cli infer --model-path model.pt \
  --query takbo tumakbo \
  --query kain kumain
```

### 4. Ablation Study

```bash
python -m gnn_lexicon.src.cli ablate --model-type sage
```

## Configuration

Create a `config.yaml` file:

```yaml
# Model architecture
in_dim: 64
hidden_dim: 128
out_dim: 64
num_layers: 2
heads: 4  # For GATv2

# Training
lr: 0.001
batch_size: 128
epochs: 100
grad_clip: 1.0
early_stopping_patience: 10

# Database
postgres:
  dbname: fil_dict_db
  user: postgres
  password: postgres
  host: localhost
  port: 5432

# Device
device: cuda  # or cpu
```

## Graph Schema

### Node Types
- **Word**: Lexical entries with lemma, language code, frequency
- **Morpheme**: Affixes and morphological units
- **Form**: Word forms and inflections
- **Sense**: Word definitions and meanings
- **Language**: Language codes

### Edge Types
- **HAS_FORM**: Word → Form (inflections)
- **OF_WORD**: Form → Word (reverse)
- **HAS_SENSE**: Word → Sense (definitions)
- **DERIVED_FROM**: Word → Word (derivations)
- **HAS_AFFIX**: Word → Morpheme (morphology)
- **RELATED**: Word → Word (semantic relations)
- **SHARES_PHONOLOGY**: Word ↔ Word (phonological similarity)
- **SHARES_ETYMOLOGY**: Word ↔ Word (etymological connection)

## API Usage

```python
from gnn_lexicon.src import (
    load_pg_connection, 
    fetch_graph_from_postgres,
    build_hetero_graph,
    create_model,
    train_gnn
)

# Load data
conn = load_pg_connection({"dbname": "fil_dict_db", ...})
raw_data = fetch_graph_from_postgres(conn)
data = build_hetero_graph(raw_data)

# Create model
metadata = (data.node_types, data.edge_types)
in_channels = {nt: data[nt].x.size(1) for nt in data.node_types}
model = create_model("gatv2", metadata, in_channels, config)

# Train
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
results = train_gnn(model, train_data, val_data, optimizer, device, config)
```

## Model Architectures

### R-GCN (Relational GCN)
- Handles multiple edge types with relation-specific transformations
- Good for datasets with many relation types
- Memory efficient for sparse graphs

### GraphSAGE
- Samples and aggregates neighbor features
- Scales to large graphs
- Inductive learning capability

### GATv2
- Attention-based aggregation
- Learns importance weights for neighbors
- Exports attention weights for interpretability

## Evaluation Metrics

- **ROC-AUC**: Area under ROC curve for link prediction
- **Hits@k**: Percentage of true links in top-k predictions
- **Accuracy**: For relation type classification
- **F1 Score**: Macro-averaged F1 for multi-class tasks

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{phil_lexicon_gnn,
  title = {Philippine Lexicon GNN Toolkit},
  author = {Philippine Lexicon GNN Team},
  year = {2025},
  url = {https://github.com/yourusername/gnn_lexicon}
}
```

## License

MIT License 