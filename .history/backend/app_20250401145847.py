"""
Flask application entry point.
"""

import os
import sys
from flask import Flask, redirect, jsonify
from flask_cors import CORS
from flask_migrate import Migrate
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Configure database from .env
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/fil_dict_db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_size': int(os.getenv('DB_MIN_CONNECTIONS', 5)),
        'max_overflow': int(os.getenv('DB_MAX_CONNECTIONS', 20)),
        'pool_timeout': 30,
        'pool_recycle': 1800
    }
    
    # Import database after app configuration
    from database import db, teardown_db
    
    # Initialize extensions
    CORS(app, resources={r"/api/*": {"origins": os.getenv("ALLOWED_ORIGINS", "*").split(",")}})
    db.init_app(app)
    migrate = Migrate(app, db)
    
    # Import and register blueprints
    from routes import bp
    app.register_blueprint(bp, url_prefix='/api/v2')
    
    # Initialize and register GraphQL view
    with app.app_context():
        from gql import init_graphql
        from flask_graphql import GraphQLView
        schema, _ = init_graphql()
        
        # Register GraphQL endpoint
        app.add_url_rule(
            '/api/v2/graphql',
            view_func=GraphQLView.as_view(
                'graphql',
                schema=schema,
                graphiql=True
            )
        )
    
    # Root route redirect to API docs or test endpoint
    @app.route('/')
    def index():
        return redirect('/api/v2/test')
    
    # Register teardown
    @app.teardown_appcontext
    def shutdown_session(exception=None):
        """Clean up database resources on app shutdown."""
        teardown_db()
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Not found',
            'message': 'The requested resource was not found on this server.',
            'status_code': 404
        }), 404

    @app.errorhandler(500)
    def server_error(error):
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred.',
            'status_code': 500
        }), 500
    
    return app

app = create_app()

if __name__ == '__main__':
    # Run the Flask application
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"Starting Flask application on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)