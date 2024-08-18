from flask import Flask
from flask_cors import CORS
from routes import bp
from database import db_session, init_db
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize the database
logger.info("Initializing database...")
try:
    init_db()
    logger.info("Database initialized successfully")
except Exception as e:
    logger.error(f"Error initializing database: {str(e)}")
@app.route('/')
def index():
    return "Welcome to the Word Relationship API!"

@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content response

app.register_blueprint(bp)
app.register_error_handler(404, bp.error_handlers[404])
app.register_error_handler(500, bp.error_handlers[500])

# Teardown the database session after each request
@app.teardown_appcontext
def shutdown_session(exception=None):
    db_session.remove()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)