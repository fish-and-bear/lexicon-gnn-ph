"""
Flask application initialization for the Filipino Dictionary API.
"""

from flask import Flask
from flask_cors import CORS
from backend.models import db
from backend import routes
from backend import security
import redis
import os
import structlog
from backend.gql.views import graphql_view

# Set up logging
logger = structlog.get_logger(__name__)

def create_app(test_config=None):
    """Create and configure the Flask application."""
    # Initialize Flask app
    app = Flask(__name__)
    
    if test_config is None:
        # Configure database
        app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
            'DATABASE_URL',
            'postgresql://postgres:postgres@localhost:5432/fil_dict_db'
        )
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        
        # Initialize Redis for rate limiting
        redis_client = redis.from_url(
            os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        )
    else:
        # Use test configuration
        app.config.update(test_config)
        redis_client = redis.from_url('redis://localhost:6379/1')  # Use different DB for tests
    
    # Initialize CORS
    CORS(app)
    
    # Initialize database
    db.init_app(app)
    
    # Initialize security
    security.init_security(app, redis_client)
    
    # Register blueprints
    app.register_blueprint(routes.bp)
    
    # Register GraphQL endpoint
    app.add_url_rule('/graphql', view_func=graphql_view, methods=['GET', 'POST'])
    
    # Error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return {"error": "Not found"}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return {"error": "Internal server error"}, 500
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        return {"status": "healthy"}, 200
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run()