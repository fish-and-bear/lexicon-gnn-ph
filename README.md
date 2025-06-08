# FilRelex: Filipino Lexical Resource

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)
[![Production Ready](https://img.shields.io/badge/Status-Production%20Ready-green.svg)]()

## Overview

FilRelex is a comprehensive Filipino dictionary and linguistic resource application with robust Baybayin script support. It provides powerful search capabilities, word relationship visualization, etymology tracking, and advanced linguistic analysis tools for Filipino languages.

This is a production-ready open source release with all debug code removed and optimized for deployment.

## 🏗️ Architecture Overview

FilRelex consists of three main components:

- **Frontend** (`/src`): React TypeScript application with D3.js visualizations
- **Backend** (`/backend`): Python Flask API with PostgreSQL database
- **ML Components** (`/ml`): Machine learning models for active learning and explainability

## 🚀 Quick Deployment Guide

📋 **For detailed setup instructions, see [SETUP.md](SETUP.md)**

### Prerequisites

Ensure you have the following installed:
- Node.js (v16+ recommended)
- Python (v3.9+ recommended)
- PostgreSQL client tools
- Git

### 🗄️ Database Access

The FilRelex database is hosted on Aiven cloud. There are two levels of access available:

#### Public Read-Only Access
For general API access and development:

**Connection Details:**
```
Host: fil-dict-db-jessegarfieldscats-becf.h.aivencloud.com
Port: 18251
Database: defaultdb
User: public_user
Password: AVNS_kWlkz-O3MvuC1PQEu3I
SSL Mode: require
```

**Connection String:**
```
postgres://public_user:AVNS_kWlkz-O3MvuC1PQEu3I@fil-dict-db-jessegarfieldscats-becf.h.aivencloud.com:18251/defaultdb?sslmode=require
```

#### Full Database Access
For **complete data access, database dumps, and administrative operations**, please contact:

**📧 Email**: aanaguio@up.edu.ph

**Admin Connection** (for reference):
```
Host: fil-dict-db-jessegarfieldscats-becf.h.aivencloud.com
Port: 18251
Database: defaultdb
User: avnadmin
Password: AVNS_RjMphxAprfpCEUs1DJA
SSL Mode: require
```

#### Security Note
The `public_user` account has read-only permissions configured through Aiven's role-based access control system, ensuring safe public access to the API endpoints while protecting the database integrity.

---

## 🎨 Frontend Deployment

### Local Development

1. **Install Dependencies:**
   ```bash
   npm install
   ```

2. **Environment Configuration:**
   Create `.env` file in the root directory:
   ```bash
   VITE_API_BASE_URL=http://localhost:5000/api/v2
   VITE_VERSION=1.0.0
   ```

3. **Start Development Server:**
   ```bash
   npm run dev
   ```
   The frontend will be available at `http://localhost:5173`.

### Production Build

1. **Build for Production:**
   ```bash
   npm run build
   ```

2. **Serve Static Files:**
   ```bash
   # Using a simple HTTP server
   npx serve -s dist -l 3000
   
   # Or using nginx (recommended for production)
   # Copy dist/ contents to your nginx web root
   ```

### Frontend Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `VITE_API_BASE_URL` | Backend API base URL | `https://your-api.com/api/v2` |
| `VITE_VERSION` | Application version | `1.0.0` |

### Deployment Platforms

- **Vercel**: Use `vercel.json` configuration
- **Netlify**: Deploy from `/dist` folder
- **GitHub Pages**: Use GitHub Actions workflow
- **AWS S3 + CloudFront**: Upload `/dist` to S3 bucket

---

## 🔧 Backend Deployment

### Local Development

1. **Navigate to Backend Directory:**
   ```bash
   cd backend
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

4. **Environment Configuration:**
   Create `.env` file in `/backend` directory:
   ```bash
   # Database Configuration
   DATABASE_URL=postgres://avnadmin:AVNS_RjMphxAprfpCEUs1DJA@fil-dict-db-jessegarfieldscats-becf.h.aivencloud.com:18251/defaultdb?sslmode=require
   
   # Flask Configuration
   SECRET_KEY=your-secret-key-here
   FLASK_ENV=production
   FLASK_APP=app.py
   
   # API Configuration
   ALLOWED_ORIGINS=http://localhost:5173,https://yourdomain.com
   ```

5. **Initialize Database (if needed):**
   ```bash
   flask db upgrade
   ```

6. **Start Development Server:**
   ```bash
   python app.py
   ```
   The backend API will be available at `http://localhost:5000`.

### Production Deployment

#### Using Gunicorn (Recommended)

1. **Install Gunicorn:**
   ```bash
   pip install gunicorn
   ```

2. **Start Production Server:**
   ```bash
   gunicorn -w 4 -b 0.0.0.0:8000 'app:create_app()'
   ```

#### Using Docker

1. **Create Dockerfile** (create in `/backend`):
   ```dockerfile
   FROM python:3.11-slim

   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .
   EXPOSE 8000

   CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:create_app()"]
   ```

2. **Build and Run:**
   ```bash
   docker build -t filrelex-backend .
   docker run -p 8000:8000 --env-file .env filrelex-backend
   ```

### Backend Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL connection string | Yes |
| `SECRET_KEY` | Flask secret key for sessions | Yes |
| `FLASK_ENV` | Flask environment (development/production) | Yes |
| `ALLOWED_ORIGINS` | CORS allowed origins | Yes |

### API Endpoints

The backend provides a RESTful API:

- **Base URL**: `/api/v2`
- **Documentation**: Available at `/api/v2/docs` when running
- **Health Check**: `/api/v2/test`

Key endpoints:
- `GET /words/{word}` - Get word details
- `GET /words/{word}/network` - Get word relationship network
- `GET /search` - Search words with filters
- `GET /statistics` - Get database statistics

---

## 🤖 ML Components Deployment

### Prerequisites

The ML components require additional dependencies for machine learning and graph processing:

```bash
cd ml
pip install -r requirements.txt
```

### Core ML Scripts

#### 1. Graph Neural Network Training (`train_hgnn.py`)

Train heterogeneous graph neural networks for word relationship modeling:

```bash
cd ml
python train_hgnn.py --config config/default.yaml
```

#### 2. Link Prediction (`link_prediction.py`)

Predict relationships between words using trained models:

```bash
python link_prediction.py --model_path models/hgnn_model.pt
```

#### 3. Pretraining Pipeline (`pretrain_hgmae.py`)

Pretrain models using masked autoencoders:

```bash
python pretrain_hgmae.py --epochs 100 --batch_size 32
```

### ML Configuration

Create `ml/config/local.yaml`:

```yaml
# Database Configuration
database:
  host: "fil-dict-db-jessegarfieldscats-becf.h.aivencloud.com"
  port: 18251
  database: "defaultdb"
  user: "avnadmin"
  password: "AVNS_RjMphxAprfpCEUs1DJA"
  ssl_mode: "require"

# Model Configuration
model:
  hidden_dim: 256
  num_layers: 3
  dropout: 0.1

# Training Configuration
training:
  learning_rate: 0.001
  batch_size: 32
  epochs: 100
  device: "cuda"  # or "cpu"
```

### Active Learning & Explainability Notebook

The project includes a comprehensive Jupyter notebook for active learning and explainability analysis:

**Location**: `Pretraining_+_Active_Learning_+_Explainability.ipynb`

#### Running the Notebook

1. **Install Jupyter Dependencies:**
   ```bash
   pip install jupyter ipykernel notebook
   ```

2. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

3. **Open the Notebook:**
   Navigate to `Pretraining_+_Active_Learning_+_Explainability.ipynb`

#### Google Colab Usage

The notebook is optimized for Google Colab:

1. Upload the notebook to Google Colab
2. Mount Google Drive and upload the FilRelex project
3. Follow the notebook's step-by-step instructions
4. Update the `project_path` variable to match your Drive structure

#### Notebook Features

- **Active Learning Strategies**: Implement uncertainty sampling and query-by-committee
- **Explainability Analysis**: SHAP values, attention visualization, feature attribution
- **Model Evaluation**: Comprehensive metrics and visualizations
- **Filipino Language Analysis**: Language-specific insights and patterns

---

## 🌐 Full Stack Deployment

### Development Environment

1. **Start Backend:**
   ```bash
   cd backend
   source venv/bin/activate
   python app.py
   ```

2. **Start Frontend:**
   ```bash
   # In a new terminal
   npm run dev
   ```

3. **Access Application:**
   - Frontend: `http://localhost:5173`
   - Backend API: `http://localhost:5000`

### Production Environment

#### Option 1: Separate Deployment

1. **Deploy Backend** to platforms like:
   - Heroku: Use `Procfile`
   - Railway: Use `railway.json`
   - AWS Elastic Beanstalk
   - DigitalOcean App Platform

2. **Deploy Frontend** to:
   - Vercel (recommended)
   - Netlify
   - AWS S3 + CloudFront

#### Option 2: Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - SECRET_KEY=${SECRET_KEY}
    
  frontend:
    build: .
    ports:
      - "3000:3000"
    environment:
      - VITE_API_BASE_URL=http://localhost:8000/api/v2
```

Run with:
```bash
docker-compose up -d
```

---

## 📊 Database Schema & Data

### Database Structure

The FilRelex database contains:

- **Words**: Filipino words with definitions, etymology, and metadata
- **Relationships**: Semantic relationships between words
- **Baybayin**: Traditional Filipino script representations
- **Analytics**: Usage statistics and language metrics

### Data Access Levels

1. **Public API**: Read-only access via the provided connection string
2. **Full Database Access**: Contact aanaguio@up.edu.ph for:
   - Complete database dumps
   - Write access permissions
   - Raw linguistic data
   - Research collaboration opportunities

---

## 🛠️ Development Guidelines

### Code Structure

```
fil-relex/
├── src/                    # Frontend React application
│   ├── components/         # UI components
│   ├── api/               # API integration
│   └── utils/             # Utility functions
├── backend/               # Python Flask backend
│   ├── models/            # Database models
│   ├── routes.py          # API routes
│   └── utils/             # Backend utilities
├── ml/                    # Machine learning components
│   ├── models/            # ML model definitions
│   ├── training/          # Training scripts
│   └── evaluation/        # Evaluation tools
└── deploy/                # Deployment configurations
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow the existing code style
4. Remove all debug code before submitting
5. Add tests for new features
6. Submit a Pull Request

---

## 📞 Support & Contact

### Technical Issues
- **GitHub Issues**: [Create an issue](https://github.com/your-username/fil-relex/issues)
- **Email**: aanaguio@up.edu.ph

### Data Access & Research Collaboration
- **Email**: aanaguio@up.edu.ph
- **Subject**: "FilRelex Data Access Request"
- **Include**: Your research affiliation and intended use case

### Community
- Join our discussions on GitHub
- Follow the project for updates
- Star the repository if you find it useful

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 🙏 Acknowledgments

- Filipino linguistic experts and community members
- University of the Philippines research team
- Open-source community contributors
- Aiven for database hosting

---

**Note**: This is a production-ready release optimized for deployment. For the latest updates and research data, please contact the development team.