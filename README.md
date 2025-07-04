# Philippine Lexicon GNN Toolkit

An open-source framework for enhancing lexical knowledge graphs of low-resource Philippine languages using heterogeneous graph neural networks (GNNs).

## 🚀 Live Demo

Visit our live demonstration at: https://explorer.hapinas.net/

## 📖 Overview

The Philippine Lexicon GNN Toolkit addresses the critical gap in NLP resources for underrepresented languages by automatically predicting lexical relationships using trained GATv2 models. The system achieves strong empirical performance (AUC: 0.994) on Tagalog data and includes comprehensive evaluation by Philippine linguistics experts.

## ✨ Key Features

- **Heterogeneous GNN Architecture**: Specialized graph neural networks for multilingual lexical data
- **Real-time Link Prediction**: Live demonstration of GNN predictions with confidence scoring
- **Interactive Visualization**: D3.js-powered graph exploration with zoom, pan, and node selection
- **Cultural Integration**: Baybayin script support and cultural context preservation
- **Expert Validation**: Comprehensive evaluation framework with linguistic expert panels
- **Open Source**: MIT-licensed with complete codebase and API access

## 🏗️ System Architecture

The system employs a production-ready microservices architecture:

- **Frontend**: React + TypeScript with Material-UI and D3.js visualization
- **Backend**: Flask API gateway with SQLAlchemy and PostgreSQL
- **ML Pipeline**: PyTorch Geometric GNN implementation
- **Deployment**: Docker containerization with Nginx load balancing

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Node.js 16+
- PostgreSQL 13+
- Docker (optional)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fil-relex.git
   cd fil-relex
   ```

2. **Install dependencies**
   ```bash
   # Backend dependencies
   pip install -r requirements.txt
   
   # Frontend dependencies
   npm install
   ```

3. **Set up database**
   ```bash
   # Create PostgreSQL database
   createdb fil_dict_db
   
   # Run migrations
   cd backend
   python migrate.py
   ```

4. **Start the application**
   ```bash
   # Start backend
   cd backend
   python app.py
   
   # Start frontend (in another terminal)
   npm run dev
   ```

### Docker Deployment

```bash
# Using Docker Compose
docker-compose -f docker-compose.local.yml up -d
```

## 🚀 Usage

### Web Interface

1. **Access the Application**
   - Open your browser and navigate to `http://localhost:5173`
   - The main interface provides search, visualization, and exploration tools

2. **Search Functionality**
   - Use the search bar to find words in Tagalog, Cebuano, and other Philippine languages
   - Results include definitions, etymologies, and related words
   - Filter by language, part of speech, or other criteria

3. **Graph Visualization**
   - Explore lexical relationships through interactive D3.js visualizations
   - Click on nodes to see detailed word information
   - Zoom, pan, and filter the graph to focus on specific relationships

4. **Baybayin Support**
   - Toggle between Latin and Baybayin script displays
   - View historical and cultural context for Philippine writing systems

### API Usage

The backend provides a comprehensive REST API:

```bash
# Search for words
curl "http://localhost:5000/api/v2/search?q=kain&language=tl"

# Get word details
curl "http://localhost:5000/api/v2/words/123"

# Get semantic network
curl "http://localhost:5000/api/v2/semantic_network?word=kain&depth=2"

# Get etymology tree
curl "http://localhost:5000/api/v2/etymology/tree?word=kain"
```

### Machine Learning Pipeline

1. **Training Models**
   ```bash
   cd ml
   python gnn_lexicon/comprehensive_training.py
   ```

2. **Making Predictions**
   ```bash
   cd ml
   python gnn_lexicon/use_existing_model.py
   ```

3. **Evaluation**
   ```bash
   cd ml
   python quick_evaluation.py
   ```

### Configuration

The system uses environment variables for configuration:

```bash
# Database configuration
DATABASE_URL=postgresql://username:password@localhost:5432/fil_dict_db
DB_HOST=localhost
DB_PORT=5432
DB_NAME=fil_dict_db
DB_USER=your_username
DB_PASSWORD=your_password

# ML configuration
ML_MODEL_PATH=./models/
ML_LOG_LEVEL=INFO
```

## 📊 Model Performance

Our GATv2 model achieves excellent performance on Philippine lexicon data:

- **Link Prediction AUC**: 0.994
- **Response Time**: < 500ms average
- **Scalability**: Support for 100+ concurrent users
- **Memory Efficiency**: Optimized for standard hardware

## 🔬 Research Impact

This work contributes to:

- **Linguistic Diversity Preservation**: Practical tools for preserving linguistic heritage
- **Digital Inclusion**: Addressing the digital language divide for underrepresented languages
- **AI Ethics**: Human-centric AI development through expert validation
- **Methodological Advancement**: Application of heterogeneous GNNs to real-world linguistic challenges

## 📚 Documentation

- [API Documentation](docs/api.md)
- [Model Architecture](docs/architecture.md)
- [Training Guide](docs/training.md)
- [Evaluation Framework](docs/evaluation.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Special thanks to the Philippine linguistics experts who participated in our evaluation panel, and to the open-source community for their invaluable contributions.

## 📞 Contact

- **Maintainer**: Angelica Anne A. Naguio (aanaguio@up.edu.ph)
- **Advisor**: Dr. Rachel Edita O. Roxas (reroxas@up.edu.ph)
- **Institution**: University of the Philippines Los Baños

## 📖 Citation

If you use this toolkit in your research, please cite:

```bibtex
@inproceedings{naguio2025philippine,
  title={Philippine Lexicon GNN Toolkit: Live Demonstration of Heterogeneous Graph Neural Networks for Low-Resource Language Lexicon Enhancement},
  author={Naguio, Angelica Anne A. and Roxas, Rachel Edita O.},
  booktitle={Proceedings of EMNLP 2025 System Demonstrations},
  year={2025}
}
```
