# Evaluation Framework

This document describes how to evaluate the Philippine Lexicon GNN Toolkit models.

## Automatic Metrics

- **ROC-AUC**: Measures link prediction quality
- **Hits@10**: Fraction of correct links in top 10 predictions
- **MRR**: Mean reciprocal rank

## Expert Validation

- Human experts review top predictions for correctness and novelty

## Usage

```bash
python quick_evaluation.py
```

- Outputs metrics and saves results to `outputs/` or `ml/`

## Custom Evaluation

- Modify `quick_evaluation.py` or use your own scripts for custom metrics

For questions, open an issue or see the main README. 