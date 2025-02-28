"""
Main application file for the Filipino Dictionary application.
This module initializes the Flask application and sets up the database connection.
"""

from flask import Flask
from models import db
from flask_migrate import Migrate
import os

def create_app(config=None):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Default configuration
    app.config.update(
        SQLALCHEMY_DATABASE_URI=os.environ.get('DATABASE_URL', 'sqlite:///filipino_dictionary.db'),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev-key-change-in-production'),
        REDIS_URL=os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
        DEBUG=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    )
    
    # Override config if provided
    if config:
        app.config.update(config)
    
    # Initialize extensions
    db.init_app(app)
    migrate = Migrate(app, db)
    
    # Register blueprints
    from routes import bp as api_bp
    app.register_blueprint(api_bp)
    
    # Initialize rate limiter
    with app.app_context():
        from routes import init_rate_limiter
        init_rate_limiter(app)
    
    return app

# Create the application instance
app = create_app()

if __name__ == "__main__":
    app.run(debug=True)