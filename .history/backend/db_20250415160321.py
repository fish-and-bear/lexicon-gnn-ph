"""
Database connection module for the application.
This module provides a central place to access the database connection.
"""

from flask_sqlalchemy import SQLAlchemy

# Create a SQLAlchemy instance to be used throughout the application
db = SQLAlchemy()

def init_db(app):
    """Initialize the database with the given Flask app."""
    db.init_app(app)
    return db 