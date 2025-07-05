# Evaluation Framework

This document describes how to evaluate the Philippine Lexicon GNN Toolkit models.

## Automatic Metrics

- **ROC-AUC**: Measures link prediction quality (higher is better, max 1.0)
- **Hits@10**: Fraction of correct links in top 10 predictions (higher is better)
- **MRR (Mean Reciprocal Rank)**: Average inverse rank of correct predictions
- **Precision/Recall/F1**: For binary link prediction (optional)

## Expert Validation

- Human experts review top predictions for correctness and novelty
- Results are compared to gold-standard or expert-annotated links

## Evaluation Methodology

- Split data into train/validation/test sets
- Evaluate on held-out links (unseen during training)
- Use negative sampling for robust metrics
- Aggregate results over multiple runs for stability

## Usage

```bash
python quick_evaluation.py
```
- Outputs metrics and saves results to `outputs/` or `ml/`
- Example output:
  - `ROC-AUC: 0.994`
  - `Hits@10: 0.92`
  - `MRR: 0.88`

## Interpreting Results

- **High ROC-AUC**: Model distinguishes true links from random well
- **High Hits@10/MRR**: Model ranks correct links highly
- Compare to baseline (random or majority class)

## Custom Evaluation

- Modify `quick_evaluation.py` or add new scripts in `ml/`
- Add new metrics by extending the evaluation function
- For large-scale or custom tasks, see PyTorch Geometric evaluation utilities

For questions, open an issue or see the main README. 