# Fil-Relex ML Pipeline Documentation

## Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Components](#key-components)
- [Data Flow](#data-flow)
- [Configuration](#configuration)
- [Models](#models)
- [Running the Pipeline](#running-the-pipeline)
- [Database Integration](#database-integration)
- [Analysis and Evaluation](#analysis-and-evaluation)
- [Troubleshooting](#troubleshooting)

## Overview

The Fil-Relex ML Pipeline is a machine learning system designed to analyze lexical relationships in multilingual dictionary data. It uses graph neural network (GNN) approaches to learn and predict relationships between words across languages, with a focus on Filipino and related languages.

The system ingests lexical data from a database (PostgreSQL or SQLite), builds a heterogeneous graph representation, and trains models to:
1. Learn effective semantic representations of words and their relationships
2. Perform link prediction to suggest potential new lexical connections
3. Analyze and visualize the dictionary network structure

## System Architecture

The system follows a modular design with these main components:

```
ml/
├── config/           # Configuration files
├── data/             # Data loading and preprocessing 
│   ├── db_adapter.py # Database connectivity
│   ├── feature_extraction.py
│   ├── preprocess.py
│   └── lexical_graph_builder.py
├── models/           # Neural network model definitions
│   ├── encoder.py
│   ├── hgmae.py      # Heterogeneous Graph Masked Autoencoder
│   ├── hgnn.py       # Heterogeneous Graph Neural Networks
│   └── link_prediction.py
├── training/         # Training utilities
├── utils/            # Helper functions
├── run_pipeline.py   # Main entry point
├── pretrain_hgmae.py # Pretraining script
├── train_hgnn.py     # GNN training script
└── link_prediction.py # Link prediction execution
```

The pipeline consists of:
1. **Database Connection**: Retrieves lexical data from PostgreSQL or SQLite
2. **Preprocessing**: Cleans data and generates features
3. **Graph Construction**: Builds a heterogeneous graph from lexical relationships
4. **Training**: Trains graph neural network models
5. **Inference**: Performs link prediction or other downstream tasks
6. **Evaluation**: Analyzes model performance
7. **Visualization**: Visualizes the lexical network and results

## Key Components

### Database Adapter (`data/db_adapter.py`)
- Provides database connectivity to either PostgreSQL or SQLite
- Loads lexical data including lemmas, definitions, and relationships
- Abstracts database operations from the ML code

### Feature Extraction (`data/feature_extraction.py`)
- Generates embeddings for words using Sentence Transformers (replacing FastText)
- Processes textual data into numerical features
- Handles multilingual feature generation

### Graph Builder (`data/lexical_graph_builder.py`)
- Constructs heterogeneous graph from lexical database
- Creates nodes for lemmas and edges for relationships
- Incorporates node features from embeddings

### HGMAE Model (`models/hgmae.py`)
- Heterogeneous Graph Masked Autoencoder
- Self-supervised pretraining architecture
- Learns word representations by reconstructing masked features

### HGNN Model (`models/hgnn.py`)
- Heterogeneous Graph Neural Network architecture
- Leverages pretrained embeddings for downstream tasks
- Handles different node and edge types

### Link Prediction (`models/link_prediction.py`, `link_prediction.py`)
- Predicts potential new relationships between lexical items
- Uses trained GNN models to suggest connections
- Evaluates prediction accuracy with various metrics

## Data Flow

The pipeline processes data as follows:

1. **Data Loading**:
   - Connect to database (PostgreSQL/SQLite)
   - Load lemmas, definitions, and relationships
   - Filter by target languages if specified

2. **Feature Generation**:
   - Generate embeddings for lemmas using Sentence Transformers
   - Process textual metadata into features
   - Normalize features according to configuration

3. **Graph Construction**:
   - Build heterogeneous graph with DGL (Deep Graph Library)
   - Create nodes for lemmas with features
   - Create edges for different relationship types
   - Split data for training/validation/testing

4. **Model Training**:
   - Pretrain HGMAE using self-supervised learning
   - Fine-tune HGNN for specific tasks
   - Save model checkpoints and embeddings

5. **Inference & Evaluation**:
   - Perform link prediction to suggest new connections
   - Evaluate model performance with metrics
   - Visualize network structure and predictions

## Configuration

The system uses JSON configuration files:

### Main Config (`config/default_config.json`)
Controls model architecture, training parameters, and data processing:
```json
{
  "db_config": {
    "db_type": "sqlite",
    "db_path": "fil_relex_colab.sqlite"
  },
  "data": {
    "use_transformer_embeddings": true,
    "transformer_model": "meedan/paraphrase-filipino-mpnet-base-v2",
    "include_definitions": true,
    "normalize_features": true,
    "target_languages": ["tgl", "en"]
  },
  "model": {
    "hidden_dim": 256,
    "num_layers": 3,
    "dropout": 0.2
  },
  "training": {
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 100,
    "early_stopping": 10
  }
}
```

### Database Config (`my_db_config.json`)
Contains database connection details:
```json
{
  "db_config": {
    "db_type": "sqlite",
    "db_path": "fil_relex_colab.sqlite"
  }
}
```

## Models

### Heterogeneous Graph Masked Autoencoder (HGMAE)
- **Purpose**: Self-supervised pretraining to learn node representations
- **Architecture**: 
  - Encoder: Graph attention or convolution layers
  - Masking: Random feature masking
  - Decoder: Reconstruction of masked features
- **Training**: Minimizes reconstruction loss of masked features

### Heterogeneous Graph Neural Network (HGNN)
- **Purpose**: Task-specific model for link prediction
- **Architecture**:
  - Heterogeneous message passing layers
  - Attention mechanisms for different relation types
  - Task-specific output layers
- **Training**: Fine-tuned with task-specific loss functions

## Running the Pipeline

### Local Execution

```bash
# Run the full pipeline
python ml/run_pipeline.py --config ml/config/default_config.json --db-config ml/my_db_config.json --output-dir output/experiment

# Run pretraining only
python ml/pretrain_hgmae.py --config ml/config/default_config.json --db-config ml/my_db_config.json

# Run link prediction
python ml/link_prediction.py --config ml/config/default_config.json --db-config ml/my_db_config.json --model-path output/experiment/model.pt
```

### Google Colab Execution

Use the `ml_pipeline_colab_script.py` to run the pipeline in Google Colab:

1. Upload the project to Google Drive (Shared Drive "ML-SP2")
2. Create a new Colab notebook
3. Copy cells from the script into the notebook
4. Run cells sequentially

## Database Integration

The system was originally designed for PostgreSQL but now also supports SQLite databases through a flexible database adapter.

### PostgreSQL Connection
```python
db_config = {
    "host": "localhost",
    "port": 5432,
    "dbname": "fil_dict_db",
    "user": "username",
    "password": "password"
}
```

### SQLite Connection
```python
db_config = {
    "db_type": "sqlite",
    "db_path": "fil_relex_colab.sqlite"
}
```

### Database Schema
The system expects tables for:
- `lemmas`: Word entries
- `definitions`: Word meanings
- `relation_types`: Types of lexical relationships
- `relations`: Connections between words

## Analysis and Evaluation

The system includes tools for analyzing dictionaries and evaluating models:

### Dictionary Analysis
- `analyze_db.py`: Basic statistical analysis of the lexical database
- `analyze_db_extended.py`: In-depth analysis of graph structure and connectivity

### Model Evaluation
- **Metrics**: Accuracy, precision, recall, F1-score, ROC AUC
- **Ablation Studies**: Testing impact of different features and architectures
- **Cross-lingual Evaluation**: Assessing performance across language pairs

## Troubleshooting

### Common Issues

1. **Dependency Conflicts**: DGL and PyTorch versions may conflict
   - Solution: Use the explicit installation commands in `ml_pipeline_colab_script.py`

2. **GPU Memory Issues**: Graph models can require substantial GPU memory
   - Solution: Reduce batch size or model size in configuration

3. **Database Connection Issues**:
   - For PostgreSQL: Check credentials and database availability
   - For SQLite: Ensure the database file exists and has correct permissions

4. **Out-of-Memory During Graph Construction**:
   - Solution: Filter data to specific languages or limit the number of relationships

### Dependencies

Main dependencies include:
- PyTorch
- DGL (Deep Graph Library)
- Sentence Transformers
- NumPy, SciPy
- pandas
- psycopg2 (for PostgreSQL)
- SQLite3 (built into Python)

For a complete list of dependencies, see `ml/requirements.txt`. 