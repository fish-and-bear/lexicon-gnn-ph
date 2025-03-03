# Filipino Dictionary Backend

This is the backend API for the Filipino Dictionary application, providing comprehensive RESTful endpoints for accessing dictionary data with enhanced support for searching, detailed word information, Baybayin script, etymology relationships, and statistical analysis.

## Setup

### Prerequisites

- Python 3.9+
- PostgreSQL 13+
- Redis (optional, for enhanced caching)

### Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file based on the example:
   ```
   DB_NAME=fil_dict_db
   DB_USER=postgres
   DB_PASSWORD=your_password
   DB_HOST=localhost
   DB_PORT=5432
   REDIS_ENABLED=false
   FLASK_ENV=development
   ```

### Database Setup

1. Create a PostgreSQL database:
   ```
   createdb fil_dict_db
   ```
2. Run migrations:
   ```
   python migrate.py
   ```

## Running the Application

### Development (Windows)

The application uses Waitress for development on Windows:

```
python serve.py
```

This will start the server on http://127.0.0.1:10000 by default.

### Production (Linux)

For production deployment, we use Gunicorn with Supervisor:

1. Using the deployment script (recommended):
   ```
   python deploy.py
   ```

2. Using Supervisor directly:
   ```
   supervisord -c supervisor.conf
   supervisorctl -c supervisor.conf status
   ```

3. Using Gunicorn directly:
   ```
   gunicorn --config gunicorn_config.py app:app
   ```

## API Documentation

The API is available at the following endpoints:

- `/api/v2/words` - List all words
- `/api/v2/words/<word>` - Get details for a specific word
- `/api/v2/search` - Search for words
- `/api/v2/words/<word>/etymology-tree` - Get etymology tree for a word
- `/api/v2/words/<word>/related` - Get related words
- `/api/v2/baybayin` - Get words with Baybayin script
- `/api/v2/random` - Get a random word
- `/api/v2/statistics` - Get dictionary statistics
- `/api/v2/parts-of-speech` - Get parts of speech
- `/api/v2/affixes` - Get affixes
- `/api/v2/relations` - Get relation types

For detailed API documentation, see the [API Reference](CURSOR_RULES.md).

## Development Guidelines

See [CURSOR_RULES.md](CURSOR_RULES.md) for comprehensive development guidelines and patterns.

## Deployment Configuration

### Environment Variables

- `FLASK_ENV`: Set to `development` or `production`
- `HOST`: Server host (default: 127.0.0.1 for development, 0.0.0.0 for production)
- `PORT`: Server port (default: 10000)
- `WAITRESS_THREADS`: Number of Waitress threads (default: 4)
- `USE_SUPERVISOR`: Whether to use Supervisor in production (default: true)
- `GUNICORN_WORKERS`: Number of Gunicorn workers (default: CPU count * 2 + 1)
- `GUNICORN_WORKER_CLASS`: Gunicorn worker class (default: sync)
- `ENABLE_SSL`: Enable SSL (default: false)

### Server Selection

The application automatically selects the appropriate server based on:
- Platform (Windows vs Linux)
- Environment (development vs production)

Windows or development environments use Waitress, while Linux production environments use Gunicorn with Supervisor.

## Troubleshooting

### Common Issues

1. **Database Connection Errors**:
   - Check your PostgreSQL service is running
   - Verify credentials in `.env` file

2. **Import Errors**:
   - Ensure you're running from the correct directory
   - Check your virtual environment is activated

3. **Server Won't Start**:
   - Check port availability
   - Look for error logs in the `logs/` directory

### Logs

Logs are stored in the `logs/` directory:
- `overall.log`: General application logs
- `error.log`: Error logs
- `gunicorn_access.log` and `gunicorn_error.log`: Gunicorn logs (production)
- `supervisor.err.log` and `supervisor.out.log`: Supervisor logs (production) 