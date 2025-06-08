# Backend API

A powerful Flask-based REST API for the Filipino Lexical Resource project, providing comprehensive access to Filipino dictionary data, word relationships, and linguistic analysis tools.

## 🏗️ Architecture Overview

The backend is built with:
- **Flask** - Lightweight web framework
- **SQLAlchemy** - Database ORM
- **PostgreSQL** - Primary database
- **Alembic** - Database migrations
- **Prometheus** - Metrics and monitoring

## 📁 Project Structure

```
backend/
├── app.py                    # Main Flask application
├── routes.py                 # API route definitions
├── models/                   # SQLAlchemy database models
├── schemas.py                # API request/response schemas
├── utils/                    # Utility functions and helpers
├── processors/               # Data processing modules
├── database/                 # Database configuration and migrations
├── requirements.txt          # Python dependencies
├── alembic.ini              # Database migration config
├── serve.py                 # Production server setup
└── deploy/                  # Deployment configurations
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- PostgreSQL client libraries
- Virtual environment (recommended)

### Installation

1. **Clone and Navigate:**
   ```bash
   git clone <repository-url>
   cd fil-relex/backend
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
   Create `.env` file:
   ```bash
   # Database Configuration
   DATABASE_URL=postgres://public_user:AVNS_kWlkz-O3MvuC1PQEu3I@fil-dict-db-jessegarfieldscats-becf.h.aivencloud.com:18251/defaultdb?sslmode=require
   
   # Flask Configuration
   SECRET_KEY=your-secret-key-here
   FLASK_ENV=development
   FLASK_APP=app.py
   
   # API Configuration
   ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
   
   # Optional: Monitoring
   ENABLE_METRICS=true
   METRICS_PORT=9090
   ```

5. **Initialize Database:**
   ```bash
   flask db upgrade
   ```

6. **Start Development Server:**
   ```bash
   python app.py
   ```

The API will be available at `http://localhost:5000`

## 🔌 API Endpoints

### Base URL: `/api/v2`

### Health & Status
- `GET /test` - Health check
- `GET /health` - Detailed health status
- `GET /metrics` - Prometheus metrics (if enabled)

### Word Operations
- `GET /words/{word}` - Get detailed word information
- `GET /words/{word}/network` - Get word relationship network
- `GET /words/{word}/etymology` - Get etymology tree
- `GET /words/random` - Get a random word

### Search & Discovery
- `GET /search` - Search words with filters
- `GET /suggestions` - Get word suggestions for autocomplete
- `GET /words` - List all words (paginated)

### Dictionary Data
- `GET /statistics` - Get database statistics
- `GET /pos` - Get parts of speech
- `GET /baybayin` - Get Baybayin script data
- `GET /affixes` - Get word affixes

### API Documentation

When running in development mode, visit:
- Swagger UI: `http://localhost:5000/api/v2/docs`
- OpenAPI Spec: `http://localhost:5000/api/v2/openapi.json`

## 📊 Database Schema

### Core Tables

#### `words`
- Primary word entries with definitions
- Etymology and pronunciation data
- Baybayin script representations
- Language classification

#### `relationships`
- Semantic relationships between words
- Relationship types (synonym, antonym, etc.)
- Confidence scores and metadata

#### `definitions`
- Multiple definitions per word
- Part of speech tagging
- Usage examples and context

#### `etymology`
- Word origins and evolution
- Language family connections
- Historical development tracking

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATABASE_URL` | PostgreSQL connection string | - | Yes |
| `SECRET_KEY` | Flask secret key | - | Yes |
| `FLASK_ENV` | Environment (development/production) | development | No |
| `ALLOWED_ORIGINS` | CORS allowed origins | localhost:5173 | No |
| `ENABLE_METRICS` | Enable Prometheus metrics | false | No |
| `METRICS_PORT` | Metrics server port | 9090 | No |
| `LOG_LEVEL` | Logging level | INFO | No |

### Database Configuration

The application supports multiple database configurations:

1. **Public Read-Only Access** (Recommended for development):
   ```bash
   DATABASE_URL=postgres://public_user:AVNS_kWlkz-O3MvuC1PQEu3I@fil-dict-db-jessegarfieldscats-becf.h.aivencloud.com:18251/defaultdb?sslmode=require
   ```

2. **Admin Access** (Full permissions, contact aanaguio@up.edu.ph for access):
   ```bash
   DATABASE_URL=postgres://avnadmin:AVNS_RjMphxAprfpCEUs1DJA@fil-dict-db-jessegarfieldscats-becf.h.aivencloud.com:18251/defaultdb?sslmode=require
   ```

3. **Local Development**:
   ```bash
   DATABASE_URL=postgresql://username:password@localhost:5432/filrelex_dev
   ```

## 🚀 Deployment

### Local Development

```bash
cd backend
source venv/bin/activate
python app.py
```

### Production Deployment

#### Option 1: Gunicorn (Recommended)

```bash
# Install Gunicorn
pip install gunicorn

# Start with multiple workers
gunicorn -w 4 -b 0.0.0.0:8000 'app:create_app()'

# With detailed configuration
gunicorn -w 4 -b 0.0.0.0:8000 \
  --timeout 120 \
  --keep-alive 5 \
  --max-requests 1000 \
  --preload \
  'app:create_app()'
```

#### Option 2: Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

EXPOSE 8000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:create_app()"]
```

Build and run:
```bash
docker build -t filrelex-backend .
docker run -p 8000:8000 --env-file .env filrelex-backend
```

#### Option 3: Platform Deployment

**Heroku:**
```bash
# Use provided Procfile
git push heroku main
```

**Railway:**
```bash
# Use railway.json configuration
railway up
```

**DigitalOcean App Platform:**
```bash
# Use deploy/digitalocean.yaml
doctl apps create --spec deploy/digitalocean.yaml
```

## 🔍 Monitoring & Logging

### Prometheus Metrics

Enable metrics collection:
```bash
export ENABLE_METRICS=true
export METRICS_PORT=9090
```

Available metrics:
- Request count and duration
- Database query performance
- Error rates and types
- Cache hit/miss ratios

### Logging Configuration

```python
# Set log level via environment
export LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR

# Logs include:
# - API request/response details
# - Database query performance
# - Error tracking and stack traces
# - Performance metrics
```

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=app --cov-report=html

# Run specific test file
python -m pytest tests/test_routes.py
```

### API Testing

```bash
# Health check
curl http://localhost:5000/api/v2/test

# Get word details
curl http://localhost:5000/api/v2/words/anak

# Search words
curl "http://localhost:5000/api/v2/search?q=love&limit=5"

# Get statistics
curl http://localhost:5000/api/v2/statistics
```

### Load Testing

```bash
# Install artillery
npm install -g artillery

# Run load test
artillery run tests/load-test.yml
```

## 🔐 Security

### Authentication & Authorization

The API currently uses:
- CORS protection for cross-origin requests
- Rate limiting on search endpoints
- Input validation and sanitization
- SQL injection prevention via SQLAlchemy

### Security Headers

```python
# Implemented security headers:
# - X-Content-Type-Options: nosniff
# - X-Frame-Options: DENY
# - X-XSS-Protection: 1; mode=block
# - Strict-Transport-Security (HTTPS only)
```

## 📈 Performance Optimization

### Database Optimization

- Connection pooling with SQLAlchemy
- Query result caching
- Database indexing on frequently queried fields
- Optimized pagination for large datasets

### Caching Strategy

```python
# Redis caching (optional)
REDIS_URL=redis://localhost:6379/0

# Cache configuration
CACHE_TYPE=redis
CACHE_DEFAULT_TIMEOUT=300
```

### API Response Optimization

- Gzip compression for responses
- JSON response optimization
- Pagination for large datasets
- Selective field inclusion

## 🛠️ Development

### Code Style

```bash
# Format code with Black
black app.py routes.py

# Sort imports
isort app.py routes.py

# Lint with flake8
flake8 app.py routes.py

# Type checking with mypy
mypy app.py routes.py
```

### Database Migrations

```bash
# Create new migration
flask db migrate -m "Add new feature"

# Apply migrations
flask db upgrade

# Rollback migration
flask db downgrade
```

### Adding New Endpoints

1. **Define Route** in `routes.py`:
   ```python
   @bp.route('/new-endpoint', methods=['GET'])
   def new_endpoint():
       return jsonify({"message": "Hello World"})
   ```

2. **Add Schema** in `schemas.py`:
   ```python
   class NewEndpointSchema(Schema):
       message = fields.Str(required=True)
   ```

3. **Add Tests** in `tests/`:
   ```python
   def test_new_endpoint(client):
       response = client.get('/api/v2/new-endpoint')
       assert response.status_code == 200
   ```

## 📞 Support

### Common Issues

**Database Connection Errors:**
- Verify DATABASE_URL is correct
- Check network connectivity
- Ensure PostgreSQL client libraries are installed

**Import Errors:**
- Activate virtual environment
- Reinstall requirements: `pip install -r requirements.txt`

**Port Already in Use:**
- Change port in app.py or use environment variable
- Kill existing process: `pkill -f flask`

### Getting Help

- **Technical Issues**: Create GitHub issue
- **Database Access**: Contact aanaguio@up.edu.ph
- **Performance Issues**: Enable debug logging

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](../LICENSE.md) file for details.

---

For more information, see the main project [README.md](../README.md) 