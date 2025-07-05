# Training Guide

This guide explains how to train and evaluate models in the Philippine Lexicon GNN Toolkit.

## Prerequisites

- Python 3.8+
- PyTorch, PyTorch Geometric, DGL, scikit-learn, etc.
- PostgreSQL database with lexicon data

## Steps

1. **Prepare Data**
   - Ensure the database is populated and accessible.
   - Update config files as needed.

2. **Train the Model**
   ```bash
   cd ml
   python gnn_lexicon/comprehensive_training.py
   ```

3. **Evaluate the Model**
   ```bash
   python quick_evaluation.py
   ```

4. **Save and Export**
   - Trained weights are saved in `ml/models/`
   - Use `use_existing_model.py` for inference

## Tips

- Use Docker for reproducible environments.
- For custom experiments, modify the config YAMLs in `ml/gnn_lexicon/`.

For troubleshooting, see the Issues page or contact the maintainers. 