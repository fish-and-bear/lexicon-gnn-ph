"""
GraphQL view handlers for the Filipino Dictionary API.
"""

from flask import request, jsonify, make_response
from graphql import graphql
from .schema import schema
import structlog
import json
from functools import wraps
from datetime import datetime, timedelta
import redis
import os

# Set up logging
logger = structlog.get_logger(__name__)

# Configure Redis for rate limiting
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
redis_client = redis.from_url(REDIS_URL)

# Rate limiting configuration
RATE_LIMIT = 100  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds

def rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get client IP
        client_ip = request.remote_addr
        
        # Create a key for this IP
        key = f'rate_limit:{client_ip}'
        
        # Get the current request count
        request_count = redis_client.get(key)
        
        if request_count is None:
            # First request, set initial count
            redis_client.setex(key, RATE_LIMIT_WINDOW, 1)
            request_count = 1
        else:
            request_count = int(request_count)
            if request_count >= RATE_LIMIT:
                # Rate limit exceeded
                response = make_response(jsonify({
                    'errors': [{
                        'message': 'Rate limit exceeded',
                        'extensions': {
                            'code': 'RATE_LIMIT_EXCEEDED'
                        }
                    }]
                }), 429)
                response.headers['X-RateLimit-Limit'] = str(RATE_LIMIT)
                response.headers['X-RateLimit-Remaining'] = '0'
                response.headers['X-RateLimit-Reset'] = str(int(redis_client.ttl(key)))
                return response
            
            # Increment the counter
            redis_client.incr(key)
            request_count += 1
        
        # Add rate limit headers to the response
        response = make_response(f(*args, **kwargs))
        response.headers['X-RateLimit-Limit'] = str(RATE_LIMIT)
        response.headers['X-RateLimit-Remaining'] = str(RATE_LIMIT - request_count)
        response.headers['X-RateLimit-Reset'] = str(int(redis_client.ttl(key)))
        return response
        
    return decorated_function

@rate_limit
def graphql_view():
    """Handle GraphQL requests."""
    try:
        if request.method == 'GET':
            # Handle introspection queries
            introspection_query = '''
                query IntrospectionQuery {
                    __schema {
                        types {
                            name
                            description
                            fields {
                                name
                                description
                                type {
                                    name
                                }
                            }
                        }
                    }
                }
            '''
            result = graphql(schema, introspection_query)
            return jsonify({
                'data': result.data,
                'errors': [str(error) for error in result.errors] if result.errors else None
            })

        # Handle POST requests with queries
        data = request.get_json()
        if not data:
            return jsonify({
                'errors': [{'message': 'No query provided'}]
            }), 400

        query = data.get('query')
        variables = data.get('variables')

        if not query:
            return jsonify({
                'errors': [{'message': 'Query is required'}]
            }), 400

        logger.info("Processing GraphQL query", query=query, variables=variables)
        
        result = graphql(
            schema,
            query,
            variable_values=variables,
            context_value={'request': request}
        )

        response_data = {
            'data': result.data,
            'errors': [{'message': str(error), 'path': error.path} for error in result.errors] if result.errors else None
        }

        return jsonify(response_data), 200

    except Exception as e:
        logger.error("Error processing GraphQL request", error=str(e))
        return jsonify({
            'errors': [{'message': 'Internal server error', 'details': str(e)}]
        }), 500 