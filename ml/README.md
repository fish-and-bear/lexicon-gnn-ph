# Filipino Lexical Knowledge Graph - Machine Learning Module

This module contains the machine learning components for the Filipino Lexical Knowledge Graph project. It provides functionality for:

1. Building heterogeneous graph representations from lexical data
2. Training graph neural networks for link prediction and node classification
3. Extracting rich features from multilingual lexical data
4. Generating predictions for improving the lexical database

## Overview

The machine learning pipeline extracts data from the Filipino Lexical Database, builds a heterogeneous graph representation, and trains advanced graph neural networks to predict missing relationships between lexical entries and classify nodes according to their grammatical properties.

### Key Features

- **Comprehensive Database Utilization**: Extracts data from all relevant tables in the lexical database including words, definitions, etymologies, pronunciations, word forms, relations, etc.
- **Rich Feature Extraction**: Combines multilingual text embeddings, phonetic features, etymology features, Baybayin script features, and more.
- **Advanced GNN Architecture**: Uses a heterogeneous GNN with relational graph convolution and sparse multi-head attention for improved performance.
- **Multi-task Learning**: Supports both link prediction and node classification in a single model.
- **Prediction Export**: Can export predictions back to the database for semi-automatic knowledge graph enhancement.

## Requirements

- Python 3.8+
- PyTorch 1.9+
- DGL (Deep Graph Library) 0.8+
- PostgreSQL 12+ (for database access)
- Additional dependencies in requirements.txt

## Quick Start

1. Ensure database connectivity by setting environment variables or config file:

```bash
export DB_HOST="localhost"
export DB_PORT=5432
export DB_NAME="filipino_lexicon"
export DB_USER="postgres"
export DB_PASSWORD="your_password"
```

2. Run the training pipeline:

```bash
cd ml
python run_pipeline.py --config config/default_config.json --mode train --task both
```

3. Evaluate a trained model:

```bash
python run_pipeline.py --mode evaluate --model-path results/trained_20230425_123456
```

4. Generate predictions:

```bash
python run_pipeline.py --mode predict --model-path results/trained_20230425_123456
```

## Directory Structure

```
ml/
├── config/                  # Configuration files
│   ├── default_config.json  # Default configuration
│   └── config_utils.py      # Helper functions for loading config
├── data/                    # Data processing
│   ├── db_adapter.py        # Database adapter for graph construction
│   └── preprocess.py        # Feature extraction and preprocessing
├── models/                  # Neural network models
│   ├── hgnn.py              # Heterogeneous Graph Neural Network
│   ├── link_prediction.py   # Link prediction head
│   └── node_classification.py # Node classification head
├── training/                # Training utilities
│   └── train.py             # Training and evaluation functions
├── utils/                   # General utilities
├── requirements.txt         # Python dependencies
├── run_pipeline.py          # Main pipeline script
└── README.md                # This file
```

## Configuration Options

The pipeline can be configured through a JSON configuration file. Key configuration options include:

### Database Configuration
- `host`, `port`, `dbname`, `user`, `password`: Database connection details

### Data Processing
- `test_size`, `val_size`: Test and validation set sizes
- `num_negative_samples`: Number of negative samples to generate
- Feature extraction options: `use_xlmr`, `use_fasttext`, `use_char_ngrams`, etc.

### Model Architecture
- `hidden_dim`: Hidden dimension size
- `n_layers`: Number of GNN layers
- `num_heads`: Number of attention heads
- Other architecture parameters: `dropout`, `residual`, `layer_norm`, etc.

### Training
- `lr`: Learning rate
- `weight_decay`: L2 regularization strength
- `num_epochs`: Maximum training epochs
- `patience`: Early stopping patience

### Prediction
- `export_to_database`: Whether to export predictions to database
- `target_relations`: Types of relations to predict
- `top_k_predictions`: Number of top predictions to return

## Database Integration

The model can export predictions back to the database through staging tables:

- `predicted_relations`: For link prediction results
- `predicted_pos_tags`: For part-of-speech classification results

These can be reviewed by linguists and lexicographers before being incorporated into the main database.

## Citing This Work

If you use this module in your research, please cite:

```
@article{naguio2023filipino,
  title={Multi-Relational Graph Neural Networks for Automated Knowledge Graph Enhancement in Low-Resource Philippine Languages},
  author={Naguio, Angelica Anne A. and Roxas, Rachel Edita O.},
  journal={[Journal Name]},
  year={2023}
}
``` 