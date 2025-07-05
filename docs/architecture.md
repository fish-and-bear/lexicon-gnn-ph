# Model Architecture

This document describes the architecture of the Philippine Lexicon GNN Toolkit.

## System Overview

- **Frontend:** React + TypeScript, D3.js for graph visualization
- **Backend:** Flask API, PostgreSQL database
- **ML Pipeline:** PyTorch Geometric, GATv2-based heterogeneous GNN

## Knowledge Graph

- Nodes: Lemmas (words) with features (character CNN, language, morphology)
- Edges: 12+ relationship types (synonym, root_of, derived_from, etc.)

## GNN Model

- 2-layer GATv2, 128 hidden units, 8 heads
- Node features: concatenated character CNN, language one-hot, morphology
- Trained with negative sampling, binary cross-entropy loss

## Serving

- Model weights loaded into Flask API (Gunicorn)
- Real-time link prediction and explanation endpoints

## Deployment

- Docker Compose for full stack
- Redis for caching, PostgreSQL for storage

For more details, see the code in `ml/` and `backend/`. 