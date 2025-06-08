# FilRelex Setup Guide

This document provides step-by-step instructions for setting up the FilRelex project for development and production.

## 🚀 Quick Start

### Prerequisites

Ensure you have the following installed:
- **Node.js** (v16 or higher)
- **Python** (v3.9 or higher)
- **Git**
- **PostgreSQL client tools** (optional, for database operations)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd fil-relex
```

### 2. Frontend Setup

```bash
# Install dependencies
npm install

# Create environment file
cp .env.example .env

# Update .env with your configuration
# VITE_API_BASE_URL=http://localhost:5000/api/v2
# VITE_VERSION=1.0.0

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:5173`.

### 3. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
copy .env.example .env  # Windows
cp .env.example .env    # macOS/Linux

# Update .env with database configuration
# DATABASE_URL=postgres://public_user:AVNS_kWlkz-O3MvuC1PQEu3I@fil-dict-db-jessegarfieldscats-becf.h.aivencloud.com:18251/defaultdb?sslmode=require
# SECRET_KEY=your-secret-key-here
# FLASK_ENV=development

# Start development server
python app.py
```

The backend API will be available at `http://localhost:5000`.

### 4. ML Components Setup (Optional)

```bash
# Navigate to ML directory
cd ml

# Create virtual environment (if not using backend's)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install ML dependencies
pip install -r requirements.txt

# Create configuration file
cp config/example.yaml config/local.yaml

# Update config/local.yaml with your settings
# Run a simple test
python train_hgnn.py --config config/local.yaml --epochs 1
```

## 🗄️ Database Configuration

FilRelex uses a PostgreSQL database hosted on Aiven. There are two access levels:

### Public Read-Only Access (Recommended for Development)

```env
DATABASE_URL=postgres://public_user:AVNS_kWlkz-O3MvuC1PQEu3I@fil-dict-db-jessegarfieldscats-becf.h.aivencloud.com:18251/defaultdb?sslmode=require
```

This provides read-only access to the database, sufficient for most development and testing purposes.

### Full Database Access

For complete data access, database dumps, and administrative operations, contact:
**Email**: aanaguio@up.edu.ph

### Local Database Setup (Alternative)

If you prefer to run a local database:

```bash
# Create local database
createdb filrelex_dev

# Update DATABASE_URL in .env
DATABASE_URL=postgresql://username:password@localhost:5432/filrelex_dev

# Run migrations (if available)
cd backend
flask db upgrade
```

## 🔧 Environment Configuration

### Frontend Environment Variables

Create `.env` in the root directory:

```env
# API Configuration
VITE_API_BASE_URL=http://localhost:5000/api/v2

# Application Configuration
VITE_VERSION=1.0.0

# Analytics (optional)
VITE_ANALYTICS_ID=your-analytics-id
```

### Backend Environment Variables

Create `backend/.env`:

```env
# Database Configuration
DATABASE_URL=postgres://public_user:AVNS_kWlkz-O3MvuC1PQEu3I@fil-dict-db-jessegarfieldscats-becf.h.aivencloud.com:18251/defaultdb?sslmode=require

# Flask Configuration
SECRET_KEY=your-secret-key-here-change-in-production
FLASK_ENV=development
FLASK_APP=app.py

# API Configuration
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000

# Optional: Monitoring
ENABLE_METRICS=false
METRICS_PORT=9090
LOG_LEVEL=INFO
```

### ML Environment Variables

Create `ml/.env`:

```env
# Database Configuration (same as backend)
DATABASE_URL=postgres://public_user:AVNS_kWlkz-O3MvuC1PQEu3I@fil-dict-db-jessegarfieldscats-becf.h.aivencloud.com:18251/defaultdb?sslmode=require

# ML Configuration
DEVICE=cuda  # or cpu
MODEL_CACHE_DIR=./models
DATA_CACHE_DIR=./cache
```

## 🧪 Testing Setup

### Frontend Tests

```bash
# Install test dependencies (if not already installed)
npm install

# Run unit tests
npm test

# Run integration tests
npm run test:integration

# Run end-to-end tests
npm run test:e2e
```

### Backend Tests

```bash
cd backend
source venv/bin/activate

# Run unit tests
python -m pytest

# Run with coverage
python -m pytest --cov=app --cov-report=html

# Run specific test file
python -m pytest tests/test_routes.py
```

### ML Tests

```bash
cd ml
source venv/bin/activate

# Run ML pipeline tests
python -m pytest tests/

# Test specific module
python -m pytest tests/test_models.py
```

## 🐳 Docker Setup (Alternative)

### Full Stack Docker

```bash
# Build and run all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Individual Service Docker

**Frontend:**
```bash
# Build frontend image
docker build -t filrelex-frontend .

# Run frontend container
docker run -p 3000:3000 --env-file .env filrelex-frontend
```

**Backend:**
```bash
cd backend

# Build backend image
docker build -t filrelex-backend .

# Run backend container
docker run -p 8000:8000 --env-file .env filrelex-backend
```

## 🛠️ Development Workflow

### Code Style and Linting

**Frontend:**
```bash
# Format code
npm run format

# Lint code
npm run lint

# Fix linting issues
npm run lint:fix
```

**Backend:**
```bash
cd backend

# Format code with Black
black app.py routes.py

# Sort imports
isort app.py routes.py

# Lint with flake8
flake8 app.py routes.py

# Type checking
mypy app.py routes.py
```

### Database Migrations

```bash
cd backend
source venv/bin/activate

# Create new migration
flask db migrate -m "Description of changes"

# Apply migrations
flask db upgrade

# Rollback migration
flask db downgrade
```

### Running the Full Stack

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
python app.py
```

**Terminal 2 - Frontend:**
```bash
npm run dev
```

**Terminal 3 - ML (Optional):**
```bash
cd ml
source venv/bin/activate
python train_hgnn.py --config config/local.yaml
```

## 🚀 Production Deployment

### Frontend Production Build

```bash
# Create production build
npm run build

# Preview production build
npm run preview

# Deploy to Vercel (recommended)
npm install -g vercel
vercel --prod
```

### Backend Production Deployment

```bash
cd backend

# Install production dependencies
pip install -r requirements.txt

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 'app:create_app()'

# Or run with provided script
python serve.py
```

### Environment-Specific Configuration

**Development:**
- Use local database or public read-only access
- Enable debug logging
- Hot reloading enabled

**Staging:**
- Use staging database
- Reduced logging
- Production-like environment

**Production:**
- Use production database
- Minimal logging
- Security headers enabled
- HTTPS enforced

## 🔍 Troubleshooting

### Common Issues

**Port Already in Use:**
```bash
# Kill process using port 5000
lsof -ti:5000 | xargs kill -9  # macOS/Linux
netstat -ano | findstr :5000   # Windows

# Or change port in configuration
```

**Database Connection Errors:**
- Verify DATABASE_URL is correct
- Check network connectivity
- Ensure PostgreSQL client libraries are installed

**Import/Module Errors:**
- Activate virtual environment
- Reinstall requirements: `pip install -r requirements.txt`
- Check Python path

**Permission Errors:**
- Ensure proper file permissions
- Run with appropriate user privileges
- Check virtual environment activation

### Getting Help

- **Technical Issues**: Create GitHub issue
- **Database Access**: Contact aanaguio@up.edu.ph
- **Documentation**: Check README.md and component-specific docs

## 📚 Additional Resources

- [Frontend Documentation](src/README.md)
- [Backend API Documentation](backend/README.md)
- [ML Pipeline Documentation](ml/README.md)
- [Production Checklist](PRODUCTION_CHECKLIST.md)

---

**Note**: This setup guide is for the production-ready version of FilRelex. For research and development purposes, contact the development team for additional resources and data access. 