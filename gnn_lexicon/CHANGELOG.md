# Changelog

All notable changes to the Philippine Lexicon GNN Toolkit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-03

### Added
- Initial release of Philippine Lexicon GNN Toolkit
- Support for heterogeneous graphs with 5 node types and 8 edge types
- Three GNN architectures: R-GCN, GraphSAGE, and GATv2
- PostgreSQL integration for Philippine dictionary database
- Link prediction and relation classification tasks
- Neighbor sampling for scalable training
- Mixed precision training support
- Command-line interface with train/evaluate/infer/ablate commands
- Comprehensive test suite
- Character-level CNN embeddings for text features
- Early stopping and gradient clipping
- Ablation study functionality
- Export/import graph data as JSON

### Graph Schema
- Node types: Word, Morpheme, Form, Sense, Language
- Edge types: HAS_FORM, OF_WORD, HAS_SENSE, DERIVED_FROM, HAS_AFFIX, RELATED, SHARES_PHONOLOGY, SHARES_ETYMOLOGY

### Known Issues
- Phonological and etymological relations are currently simulated
- Limited to 100 words for demo purposes in some relations
- Requires manual PostgreSQL setup

## [Unreleased]

### Planned Features
- Support for additional Philippine languages beyond Tagalog
- Integration with pre-trained language models
- Web API for model serving
- Visualization tools for graph exploration
- Support for dynamic graphs (temporal evolution)
- Multi-task learning across different linguistic tasks
- Distributed training support
- ONNX export for deployment 