# Model Architecture

This document describes the architecture of the Philippine Lexicon GNN Toolkit.

## System Overview

- **Frontend:** React + TypeScript, D3.js for graph visualization
- **Backend:** Flask API, PostgreSQL database
- **ML Pipeline:** PyTorch Geometric, GATv2-based heterogeneous GNN

## System Architecture Diagram

```mermaid
graph TD
  A[User Browser] -->|HTTP/REST| B(Frontend: React)
  B -->|API Calls| C(Backend: Flask API)
  C -->|SQL| D[(PostgreSQL DB)]
  C -->|Model Inference| E[ML Pipeline (PyTorch Geometric)]
  E -->|Model Files| F[Model Storage]
  C -->|Cache| G[Redis]
```

## Data Flow

1. User interacts with the web UI (search, explore, visualize)
2. Frontend sends REST API requests to backend
3. Backend queries database and/or invokes ML pipeline for predictions
4. Results are returned to frontend for visualization

## Component Interactions

- **Frontend**: Handles user input, visualization, and API communication
- **Backend**: Orchestrates data retrieval, ML inference, and business logic
- **Database**: Stores lexicon entries, relationships, metadata
- **ML Pipeline**: Loads GNN model, performs link prediction, returns scores
- **Redis**: Caches frequent queries and model results for performance

## Knowledge Graph

- **Nodes**: Lemmas (words) with features (character CNN, language, morphology)
- **Edges**: 12+ relationship types (synonym, root_of, derived_from, etc.)

## GNN Model

- 2-layer GATv2, 128 hidden units, 8 heads
- Node features: concatenated character CNN, language one-hot, morphology
- Trained with negative sampling, binary cross-entropy loss
- Model files stored in `ml/models/`

## ML Pipeline Details

- **Training**: See `ml/gnn_lexicon/comprehensive_training.py`
- **Inference**: Model loaded in backend for real-time prediction
- **Evaluation**: See `ml/quick_evaluation.py`

## Deployment

- Docker Compose for full stack
- Redis for caching, PostgreSQL for storage
- Nginx for load balancing (see `deploy/nginx/`)

For more details, see the code in `ml/` and `backend/`. 