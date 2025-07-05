# Training Guide

This guide explains how to train and evaluate models in the Philippine Lexicon GNN Toolkit.

## Prerequisites

- Python 3.8+
- PyTorch, PyTorch Geometric, DGL, scikit-learn, etc.
- PostgreSQL database with lexicon data

## Configuration

- Edit config files in `ml/gnn_lexicon/` (YAML/JSON)
- Set database connection in environment variables or config
- Key hyperparameters:
  - `num_layers`: Number of GNN layers (default: 2)
  - `hidden_dim`: Hidden units per layer (default: 128)
  - `num_heads`: Attention heads (default: 8)
  - `dropout`: Dropout rate (default: 0.2)
  - `learning_rate`: (default: 0.001)
  - `epochs`: (default: 100)

## Steps

1. **Prepare Data**
   - Ensure the database is populated and accessible.
   - Update config files as needed.

2. **Train the Model**
   ```bash
   cd ml
   python gnn_lexicon/comprehensive_training.py
   ```
   - Logs and checkpoints are saved in `ml/models/`

3. **Evaluate the Model**
   ```bash
   python quick_evaluation.py
   ```
   - Outputs metrics and saves results to `outputs/` or `ml/`

4. **Inference**
   - Use `use_existing_model.py` for predictions

## Troubleshooting

- Check database connectivity and credentials
- Ensure all dependencies are installed (see `ml/requirements.txt`)
- For CUDA errors, set `CUDA_VISIBLE_DEVICES` or use CPU mode
- For memory issues, reduce batch size or model size

## Scaling & Customization

- For large datasets, use distributed training (see PyTorch docs)
- Modify model architecture in `ml/gnn_lexicon/`
- Add new features or relationship types by updating data loaders and model inputs

For troubleshooting, see the Issues page or contact the maintainers. 