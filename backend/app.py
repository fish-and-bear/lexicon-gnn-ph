from flask import Flask
from flask_cors import CORS
from routes import bp  # Import the blueprint from your routes.py

app = Flask(__name__)
CORS(app)  # Enable CORS if your frontend and backend are running on different domains or ports

# Register the blueprint
app.register_blueprint(bp)

@app.route('/')
def index():
    return "Welcome to the Word Relationship API!"

# Handle 404 errors
@app.errorhandler(404)
def not_found(error):
    return {"error": "Resource not found"}, 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

