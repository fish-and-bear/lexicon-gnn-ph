from flask import Flask
from flask_migrate import Migrate, init, migrate, upgrade
from app import create_app
from models import db

def run_migrations():
    """Run database migrations."""
    # Create Flask app with test config
    app = create_app({
        'SQLALCHEMY_TRACK_MODIFICATIONS': False,
        'TESTING': True
    })
    
    # Initialize migration environment
    migrate = Migrate(app, db)
    
    with app.app_context():
        # Create the migration
        init()
        migrate()
        # Apply the migration
        upgrade()

if __name__ == '__main__':
    run_migrations() 