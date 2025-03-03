# Backend Codebase Reference Guide

## Project Structure

- **Core Application Files**:
  - `app.py`: Flask application setup and configuration
  - `serve.py`: Entry point for running the server
  - `models.py`: SQLAlchemy database models
  - `routes.py`: API endpoints and route handlers
  - `caching.py`: Multi-level caching system with Redis

- **Database Management**:
  - `database.py`: Database connection and management utilities
  - `migrate.py`: Database migration tools
  - `setup_db.sql`: SQL scripts for initial database setup

- **Dictionary Management**:
  - `dictionary_manager.py`: Core dictionary management functionality
  - `source_standardization.py`: Standardization of dictionary sources
  - `default_mappings.py`: Default mappings for dictionary data

- **Utilities**:
  - `language_utils.py`: Language-specific utilities
  - `security.py`: Security-related functionality
  - `monitoring.py`: System monitoring tools
  - `word_network.py`: Word relationship network utilities

## API Design Patterns

1. **Route Structure**:
   - All API routes are prefixed with `/api/v2/`
   - Resource-based endpoints (e.g., `/api/v2/words`, `/api/v2/search`)
   - Nested resources for related data (e.g., `/api/v2/words/<word>/etymology-tree`)

2. **Request Validation**:
   - Use schema classes in routes.py (e.g., `WordQuerySchema`, `SearchQuerySchema`)
   - Apply validation with `@validate_query_params` decorator
   - Extract parameters with `**params` pattern

3. **Response Format**:
   - Success responses: `success_response(data, meta={})`
   - Error responses: `error_response(message, status_code, errors, error_code)`

4. **Caching Strategy**:
   - Route caching: `@cached(prefix="cache_key", ttl=seconds)`
   - Multi-level caching with memory and Redis fallback

## Data Models

1. **Word**: Core entity representing dictionary entries
   - Properties: lemma, normalized_lemma, language_code, baybayin_form, etc.
   - Relationships: definitions, etymologies, relations, affixations

2. **Definition**: Word definitions
   - Properties: definition_text, part of speech, examples, usage notes
   - Relationships: word, standardized_pos

3. **Etymology**: Word origins
   - Properties: etymology_text, language_codes, components
   - Relationships: word

4. **Relation**: Word relationships
   - Properties: relation_type, sources
   - Relationships: from_word, to_word

5. **Affixation**: Word affixation patterns
   - Properties: affix_type, sources
   - Relationships: root_word, affixed_word

## Etymology Component Extraction

1. **Component Extraction Functions**:
   - `parse_components_field()`: Parse from structured data
   - `extract_components_from_text()`: Extract from etymology text

2. **Component Patterns**:
   - "From X + Y" pattern: Split by '+' and clean
   - "Borrowed from X" pattern: Extract language and word
   - Proto-language patterns: Handle asterisks and language prefixes

3. **Component Organization**:
   - String components stored in `components` array
   - Component word objects stored in `component_words` array
   - Each component word includes reference to original component string

## Deployment Strategies

1. **Development Environment (Windows)**:
   - **Server**: Waitress WSGI server
   - **Command**: `py serve.py` or `python serve.py`
   - **Configuration**: Environment variables in `.env` file
   - **Logs**: Stored in `logs/` directory

2. **Production Environment (Linux)**:
   - **Server**: Gunicorn with multiple workers
   - **Process Manager**: Supervisor for monitoring and auto-restart
   - **Configuration**:
     - `gunicorn_config.py`: Gunicorn-specific settings
     - `supervisor.conf`: Process management settings
   - **Deployment**: Use `deploy.py` script

3. **Deployment Script**:
   - `deploy.py` handles both development and production deployments
   - Automatically selects appropriate server based on platform and environment
   - Creates necessary directories and runs database migrations
   - Can start the application directly or via supervisor

4. **Environment Detection**:
   - Platform detection: `platform.system() == 'Windows'`
   - Environment detection: `os.getenv('FLASK_ENV', 'development') == 'production'`
   - Server selection based on both platform and environment

5. **Running the Application**:
   - Development: `python deploy.py` or `python serve.py`
   - Production with Supervisor: `supervisord -c supervisor.conf`
   - Production with Gunicorn directly: `gunicorn --config gunicorn_config.py app:app`

## Caching System

1. **Cache Configuration**:
   - `CacheConfig` class in caching.py defines TTLs and settings
   - Memory cache types: default, short, long, permanent, hot
   - Redis connection with circuit breaker pattern

2. **Cache Decorators**:
   - `@cached(prefix, ttl)`: Route-level caching
   - `@multi_level_cache(prefix, ttl)`: Function-level caching
   - `@cache_response(expiry)`: Response caching

3. **Cache Key Generation**:
   - `generate_cache_key(prefix, *args, **kwargs)`: Create consistent keys
   - Include request-specific components when available

## Error Handling and Logging

1. **Logging**:
   - Use `logger` from structlog for structured logging
   - Log request/response details with `log_request_info()`
   - Include context in error logs

2. **Error Responses**:
   - Use `error_response(message, status_code, errors, error_code)`
   - Rate limit errors: `rate_limit_exceeded_response(window)`
   - Global exception handler: `handle_exception(e)`

## Development Guidelines

1. **Adding New API Endpoints**:
   - Create schema class for validation
   - Use `@validate_query_params` decorator
   - Apply `@cached` decorator if appropriate
   - Follow established response format

2. **Extending Models**:
   - Add validation with `@validates` decorator
   - Include `to_dict()` method for serialization
   - Set up relationships with appropriate cascade options
   - Add indexes for frequently queried fields

3. **Improving Etymology Extraction**:
   - Add new patterns to `extract_components_from_text()`
   - Clean and normalize extracted components
   - Handle language-specific extraction cases

4. **Optimizing Performance**:
   - Use eager loading with `joinedload()` to avoid N+1 queries
   - Apply pagination for large result sets
   - Cache expensive operations
   - Use database indexes for frequent queries

5. **Ensuring Security**:
   - Validate and sanitize all user inputs
   - Apply rate limiting for public endpoints
   - Log security-related events
   - Use parameterized queries to prevent injection

## Common Patterns

1. **Query Building**:
```python
query = Word.query.options(
    joinedload(Word.definitions),
    joinedload(Word.etymologies)
).filter(Word.normalized_lemma == normalize_word(word))
```

2. **Response Construction**:
```python
return success_response(
    data,
    meta={
        "total": total,
        "page": page,
        "per_page": per_page
    }
)
```

3. **Error Handling**:
```python
try:
    # Operation
except ValidationError as err:
    return error_response("Invalid parameters", 400, err.messages)
except Exception as e:
    logger.error(f"Error: {str(e)}", exc_info=True)
    return error_response("Operation failed")
```

4. **Component Extraction**:
```python
components = parse_components_field(etym.normalized_components)
if not components:
    components = extract_components_from_text(etym.etymology_text)
```

## Etymology Tree Structure

```json
{
  "id": 123,
  "word": "example",
  "normalized_lemma": "example",
  "language": "tl",
  "has_baybayin": true,
  "baybayin_form": "ᜁᜄ᜔ᜐᜋ᜔ᜉᜎ᜔",
  "romanized_form": "igsampol",
  "etymologies": [
    {
      "id": 456,
      "text": "From X + Y",
      "languages": ["en", "es"],
      "sources": ["source1", "source2"]
    }
  ],
  "components": ["X", "Y"],
  "component_words": [
    {
      "component": "X",
      "word": {
        "id": 789,
        "word": "x-word",
        // ... nested word object
      }
    }
  ]
}
``` 