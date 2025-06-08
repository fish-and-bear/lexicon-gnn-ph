# Machine Learning Components

Advanced machine learning pipeline for Filipino language processing, featuring graph neural networks, active learning strategies, and explainability tools for the FilRelex project.

## 🤖 Overview

The ML components provide:
- **Graph Neural Networks** for modeling word relationships
- **Active Learning** for intelligent data annotation
- **Explainability Tools** for model interpretation
- **Pretraining Pipelines** for representation learning
- **Evaluation Frameworks** for performance assessment

## 🏗️ Architecture Overview

```
ml/
├── models/                    # Neural network model definitions
├── training/                  # Training scripts and utilities
├── evaluation/                # Model evaluation and metrics
├── active_learning/           # Active learning strategies
├── explanation/               # Explainability tools
├── data/                      # Data processing utilities
├── utils/                     # Shared utilities
├── config/                    # Configuration files
├── notebook/                  # Jupyter notebooks
└── requirements.txt           # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- CUDA-compatible GPU (recommended)
- At least 8GB RAM
- PostgreSQL access (for database connection)

### Installation

1. **Navigate to ML Directory:**
   ```bash
   cd fil-relex/ml
   ```

2. **Create Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Database Connection:**
   Create `my_db_config.json`:
   ```json
   {
     "host": "fil-dict-db-jessegarfieldscats-becf.h.aivencloud.com",
     "port": 18251,
     "database": "defaultdb",
     "user": "public_user",
     "password": "AVNS_kWlkz-O3MvuC1PQEu3I",
     "ssl_mode": "require"
   }
   ```

   **Note**: This uses the public read-only user. For full database access or write operations, contact aanaguio@up.edu.ph for admin credentials.

5. **Verify Installation:**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## 📊 Core Components

### 1. Graph Neural Networks (`train_hgnn.py`)

Train heterogeneous graph neural networks for modeling Filipino word relationships:

```bash
# Basic training
python train_hgnn.py

# With custom configuration
python train_hgnn.py --config config/hgnn_config.yaml --epochs 100

# GPU training
python train_hgnn.py --device cuda --batch_size 64
```

**Features:**
- Heterogeneous graph modeling
- Multiple relationship types
- Attention mechanisms
- Regularization techniques

### 2. Pretraining Pipeline (`pretrain_hgmae.py`)

Masked autoencoder pretraining for graph representations:

```bash
# Standard pretraining
python pretrain_hgmae.py --epochs 50 --mask_rate 0.3

# Advanced configuration
python pretrain_hgmae.py \
  --config config/pretrain_config.yaml \
  --output_dir outputs/pretrain_$(date +%Y%m%d) \
  --device cuda
```

**Key Features:**
- Masked graph autoencoder architecture
- Self-supervised learning
- Representation quality optimization
- Transfer learning capabilities

### 3. Link Prediction (`link_prediction.py`)

Predict semantic relationships between Filipino words:

```bash
# Train link prediction model
python link_prediction.py --mode train --data_split 0.8

# Evaluate existing model
python link_prediction.py --mode eval --model_path models/link_predictor.pt

# Predict new relationships
python link_prediction.py --mode predict --input_words "anak,magulang"
```

**Capabilities:**
- Binary and multi-class relationship prediction
- Confidence scoring
- Batch prediction support
- Model interpretability

### 4. Active Learning Pipeline (`run_pipeline.py`)

Intelligent sample selection for efficient annotation:

```bash
# Start active learning loop
python run_pipeline.py --strategy uncertainty --budget 1000

# Query-by-committee approach
python run_pipeline.py --strategy committee --n_models 5

# Diversity-based sampling
python run_pipeline.py --strategy diversity --clustering_method kmeans
```

**Strategies:**
- Uncertainty sampling
- Query-by-committee
- Diversity-based selection
- Hybrid approaches

## 📓 Jupyter Notebooks

### Active Learning + Explainability Notebook

**Location**: `Pretraining_+_Active_Learning_+_Explainability.ipynb`

A comprehensive notebook demonstrating:

#### Section 1: Environment Setup
- Google Colab configuration
- Package installation and management
- Database connection setup

#### Section 2: Data Loading and Preprocessing
```python
# Load Filipino word data
words_df = load_word_data(db_config)
relationships_df = load_relationships(db_config)

# Create graph structure
graph = create_heterogeneous_graph(words_df, relationships_df)
```

#### Section 3: Model Training
```python
# Initialize model
model = HGMAEModel(
    hidden_dim=256,
    num_layers=3,
    num_heads=8
)

# Train with active learning
trainer = ActiveLearningTrainer(
    model=model,
    strategy='uncertainty',
    budget=1000
)

trainer.train(graph_data)
```

#### Section 4: Explainability Analysis
```python
# SHAP analysis
explainer = GraphSHAP(model)
shap_values = explainer.explain(sample_nodes)

# Attention visualization
attention_weights = model.get_attention_weights(graph)
visualize_attention(attention_weights)
```

#### Section 5: Evaluation and Metrics
```python
# Comprehensive evaluation
evaluator = ModelEvaluator(model)
metrics = evaluator.evaluate_all()

# Performance visualization
plot_training_curves(trainer.history)
plot_performance_metrics(metrics)
```

### Running the Notebook

#### Google Colab (Recommended)
1. Upload notebook to Google Colab
2. Mount Google Drive
3. Upload FilRelex project to Drive
4. Update `project_path` variable
5. Run cells sequentially

#### Local Jupyter
```bash
# Install Jupyter
pip install jupyter ipykernel

# Start Jupyter server
jupyter notebook

# Open notebook
# Navigate to Pretraining_+_Active_Learning_+_Explainability.ipynb
```

## ⚙️ Configuration

### Model Configuration (`config/model_config.yaml`)

```yaml
# Model Architecture
model:
  type: "HGMAE"
  hidden_dim: 256
  num_layers: 3
  num_heads: 8
  dropout: 0.1
  activation: "relu"

# Training Parameters
training:
  learning_rate: 0.001
  weight_decay: 0.0001
  batch_size: 32
  epochs: 100
  early_stopping: true
  patience: 10

# Active Learning
active_learning:
  strategy: "uncertainty"
  initial_budget: 100
  query_budget: 50
  max_iterations: 20

# Data Configuration
data:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  neg_sampling_ratio: 1.0
```

### Database Configuration

```json
{
  "host": "fil-dict-db-jessegarfieldscats-becf.h.aivencloud.com",
  "port": 18251,
  "database": "defaultdb",
  "user": "public_user",
  "password": "AVNS_kWlkz-O3MvuC1PQEu3I",
  "ssl_mode": "require"
}
```

## 🔬 Evaluation and Analysis

### Performance Metrics

```bash
# Run comprehensive evaluation
python evaluation/evaluate_model.py --model_path models/trained_model.pt

# Generate performance report
python evaluation/generate_report.py --results_dir outputs/eval_results
```

**Available Metrics:**
- Link prediction accuracy
- Relationship classification F1-score
- Embedding quality (clustering metrics)
- Active learning efficiency
- Model interpretability scores

### Visualization Tools

```python
# Training progress visualization
from utils.visualization import plot_training_curves
plot_training_curves(training_history)

# Graph structure analysis
from utils.graph_analysis import analyze_graph_structure
stats = analyze_graph_structure(graph_data)

# Embedding visualization
from utils.embedding_viz import plot_embeddings_tsne
plot_embeddings_tsne(node_embeddings, labels)
```

## 🚀 Deployment

### Model Serving

#### Local Deployment
```bash
# Start model server
python serve_model.py --model_path models/best_model.pt --port 8080

# Test predictions
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"words": ["anak", "magulang"], "relationship_type": "family"}'
```

#### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8080

CMD ["python", "serve_model.py", "--port", "8080"]
```

### Batch Processing

```bash
# Process large datasets
python batch_process.py \
  --input_file data/word_pairs.csv \
  --output_file predictions/relationships.csv \
  --model_path models/link_predictor.pt \
  --batch_size 1000
```

## 🔍 Monitoring and Logging

### Training Monitoring

```python
# MLflow integration
import mlflow

with mlflow.start_run():
    mlflow.log_params(config)
    mlflow.log_metrics(metrics)
    mlflow.pytorch.log_model(model, "model")
```

### Performance Monitoring

```bash
# Monitor GPU usage
nvidia-smi --loop=1

# Monitor training progress
tail -f logs/training.log

# Watch metrics
tensorboard --logdir=outputs/tensorboard
```

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
python -m pytest tests/

# Test specific module
python -m pytest tests/test_models.py

# Test with coverage
python -m pytest --cov=ml tests/
```

### Integration Tests

```bash
# Test full pipeline
python tests/test_pipeline_integration.py

# Test data loading
python tests/test_data_loading.py

# Test model training
python tests/test_training.py
```

## 🔧 Development

### Adding New Models

1. **Create Model Class:**
   ```python
   # models/new_model.py
   class NewModel(torch.nn.Module):
       def __init__(self, config):
           super().__init__()
           # Model implementation
   ```

2. **Add Training Script:**
   ```python
   # training/train_new_model.py
   def train_new_model(config):
       # Training implementation
   ```

3. **Update Configuration:**
   ```yaml
   # config/new_model_config.yaml
   model:
     type: "NewModel"
     # Configuration parameters
   ```

### Code Style

```bash
# Format code
black ml/
isort ml/

# Lint code
flake8 ml/

# Type checking
mypy ml/
```

## 📈 Performance Optimization

### GPU Optimization

```python
# Mixed precision training
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()
with autocast():
    output = model(input_data)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Memory Optimization

```python
# Gradient checkpointing
model = torch.utils.checkpoint.checkpoint_sequential(
    model, segments=4, input=input_data
)

# Data loading optimization
dataloader = DataLoader(
    dataset, 
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)
```

## 📞 Support

### Common Issues

**CUDA Memory Errors:**
- Reduce batch size
- Use gradient accumulation
- Enable gradient checkpointing

**Slow Training:**
- Check data loading bottlenecks
- Optimize model architecture
- Use mixed precision training

**Database Connection Issues:**
- Verify connection string
- Check firewall settings
- Test connection manually

### Getting Help

- **Technical Issues**: Create GitHub issue with error logs
- **Model Performance**: Contact aanaguio@up.edu.ph
- **Data Access**: Email aanaguio@up.edu.ph for dataset access

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](../LICENSE.md) file for details.

---

For more information, see the main project [README.md](../README.md) 