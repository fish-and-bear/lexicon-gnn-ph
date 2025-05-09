# FilRelex: Filipino Lexical Resource

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)

## Overview

FilRelex is a comprehensive Filipino dictionary and linguistic resource application with robust Baybayin script support. It provides powerful search capabilities, word relationship visualization, etymology tracking, and advanced linguistic analysis tools for Filipino languages.

![FilRelex Screenshot](docs/screenshots/main.png)

## Features

- **Multiple Language Support**: Primary focus on Filipino (Tagalog) with support for other Philippine languages
- **Comprehensive Word Information**:
  - Definitions with examples
  - Etymology information
  - Pronunciation guides
  - Part of speech
  - Word relationships
  - Inflections and forms
- **Baybayin Script Integration**:
  - Baybayin form for words
  - Baybayin search functionality
  - Romanized text to Baybayin conversion
  - Baybayin statistics and analytics
- **Advanced Search Capabilities**:
  - Multi-criteria filtering
  - Semantic search
  - Phonetic matching
  - Affixation patterns
- **Word Relationship Visualization**:
  - Interactive network graph
  - Semantic relationship explorer
  - Etymology trees
- **Statistical Analysis**:
  - Dictionary completeness metrics
  - Language coverage statistics
  - Baybayin usage analytics
  - Word frequency data
- **Import/Export Functionality**:
  - Customizable export formats
  - Bulk data operations

## Tech Stack

### Frontend
- React with TypeScript
- Interactive visualizations using D3.js
- CSS for styling
- Web Workers for performance optimization

### Backend
- Python with Flask
- PostgreSQL database
- SQLAlchemy ORM
- GraphQL API (complementary to REST)
- Prometheus for monitoring

## Project Structure

A brief overview of the main directories:

- `/.git/`: Git version control files.
- `/backend/`: Contains the Python Flask backend application.
  - `backend/app.py`: Main Flask application entry point.
  - `backend/routes.py`: API endpoint definitions.
  - `backend/models/`: Database models (SQLAlchemy).
  - `backend/schemas.py`: Data serialization and validation schemas (Marshmallow).
  - `backend/dictionary_manager/`: Core logic for dictionary operations.
  - `backend/migrations/`: Database migration scripts (Alembic).
- `/src/`: Contains the React frontend application.
  - `src/components/`: Reusable UI components.
  - `src/pages/`: Top-level page components.
  - `src/App.tsx`: Main frontend application component.
- `/docs/`: Project documentation, including API details, design documents, etc.
- `/scripts/`: Utility scripts for development, data processing, etc.
- `/tests/`: Automated tests for backend and frontend.
- `requirements.txt`: Python backend dependencies.
- `package.json`: Frontend dependencies and scripts.

## Getting Started

Follow these instructions to get Fil-Relex up and running on your local machine for development and testing purposes.

### Prerequisites

Ensure you have the following installed:
- Node.js (v16+ recommended)
- Python (v3.9+ recommended)
- PostgreSQL (v12+ recommended)
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/fish-and-bear/fil-relex.git # Replace with your actual repo URL
   cd fil-relex
   ```

2. **Backend Setup:**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   # Copy and configure environment variables
   cp .env.example .env # Create .env if it doesn't exist from an example
   # Edit .env with your PostgreSQL database URI (DATABASE_URL) and other settings like SECRET_KEY
   # Example for DATABASE_URL: postgresql://user:password@host:port/database_name
   # Initialize/Upgrade the database
   flask db upgrade # Assuming Flask-Migrate is set up
   cd ..
   ```

3. **Frontend Setup:**
   ```bash
   cd src # Or your frontend root directory if different
   npm install
   cd ..
   ```

## Running the Application

### Development Mode

1. **Start the Backend Server:**
   Navigate to the `backend` directory and ensure your virtual environment is activated.
   ```bash
   cd backend
   source venv/bin/activate # Or venv\Scripts\activate
   flask run --debug # Or your specific run command, e.g., from run.py
   ```
   The backend API will typically be available at `http://127.0.0.1:5000`.

2. **Start the Frontend Development Server:**
   Navigate to your frontend directory (`src` or project root if configured).
   ```bash
   cd src # Or your frontend root
   npm run dev
   ```
   The frontend will typically be available at `http://localhost:3000` or `http://localhost:5173` (for Vite).

### Production Mode
(Instructions for running in a production-like environment. This section can be expanded with details about Docker, Gunicorn, Nginx, etc., as your project evolves.)

- **Backend:** Typically involves running the Flask app with a production-grade WSGI server like Gunicorn.
  ```bash
  # Example (from backend directory):
  # gunicorn -w 4 -b 0.0.0.0:8000 'app:create_app()' # Adjust app:create_app() to your app factory
  ```
- **Frontend:** Build the static assets and serve them using a web server like Nginx or via a hosting platform.
  ```bash
  # Example (from frontend directory):
  # npm run build
  # (Serve the 'dist' or 'build' folder)
  ```

## Running Tests
(This section assumes you have tests. If not, it's a good reminder to add them!)

1. **Backend Tests:**
   Navigate to the `backend` directory.
   ```bash
   cd backend
   # Ensure virtual environment is active
   # pytest # Or your specific test command (e.g., python -m unittest discover)
   cd ..
   ```

2. **Frontend Tests:**
   Navigate to the frontend directory.
   ```bash
   cd src # Or your frontend root
   # npm test # Or your specific test command (e.g., npm run test:unit, npm run test:e2e)
   cd ..
   ```

## Deployment
(Details on how to deploy the application. This can include notes on platforms like Vercel, Netlify, Railway, Heroku, AWS, Docker Hub, etc.)

- Refer to `render.yaml`, `railway.json`, `vercel.json`, `Procfile`, `nixpacks.toml` for examples of deployment configurations used.
- Dockerization is recommended for consistent deployments. See `docker-compose.local.yml` for local setup, and consider creating a production `Dockerfile`.

## Usage

### Web Interface
Once the application is running, access the web interface via your browser (typically `http://localhost:3000` or `http://localhost:5173` in development).
The primary sections include:
- **Word Explorer**: Search, browse, and filter words.
- **Word Details**: View comprehensive information for a selected word, including definitions, etymology, relations, and Baybayin forms.
- **Word Graph**: Visualize semantic networks and etymological connections.
- **Statistics**: Explore dictionary statistics and language analytics.

### API
FilRelex offers a RESTful API for programmatic access.

**Base URL (example for local development):** `http://127.0.0.1:5000/api/v1` (Adjust version and port as needed)

**Authentication:** (Specify if API keys or other authentication methods are required)

**Key Endpoint Categories:**
- `/words/`: Retrieve and manage word entries.
- `/search/`: Perform simple and advanced searches.
- `/baybayin/`: Utilities for Baybayin script (conversion, analysis).
- `/statistics/`: Access dictionary and language statistics.
- `/export/`, `/import/`: For data management (if exposed via API).

**Example API Calls (Python):**
```python
import requests

# Adjust API_BASE_URL based on your deployment (local/production)
API_BASE_URL = "http://127.0.0.1:5000/api/v1" # Example

# Search for a word
try:
    response = requests.get(f"{API_BASE_URL}/search?q=bahay&mode=exact")
    response.raise_for_status() # Raises an exception for HTTP errors
    print("Search results:", response.json())
except requests.exceptions.RequestException as e:
    print(f"Error searching: {e}")

# Get word details
try:
    response = requests.get(f"{API_BASE_URL}/words/bahay") # Assuming 'bahay' is a valid identifier
    response.raise_for_status()
    print("Word details:", response.json())
except requests.exceptions.RequestException as e:
    print(f"Error fetching word details: {e}")

# Convert text to Baybayin
try:
    payload = {"text": "Magandang umaga", "language_code": "tl"} # Use 'tl' for Tagalog consistently
    response = requests.post(f"{API_BASE_URL}/baybayin/convert", json=payload)
    response.raise_for_status()
    print("Baybayin conversion:", response.json())
except requests.exceptions.RequestException as e:
    print(f"Error converting to Baybayin: {e}")
```
For complete API documentation, including all endpoints, parameters, request/response formats, and authentication details, please refer to:
- `docs/API.md` (if you have a detailed one)
- Or, explore the API using tools like Postman or Insomnia with the provided base URL.
- Swagger/OpenAPI documentation may be available at `/api/v1/docs` (if integrated).

## Data Model

FilRelex utilizes a relational database (PostgreSQL) with a schema designed to capture rich linguistic information:
- **Words**: Core entries storing lemma, language, Baybayin forms, and associated metadata.
- **Definitions**: Meanings of words, including part of speech, usage examples, and notes.
- **Relations**: Semantic and etymological links between words (e.g., synonyms, antonyms, cognates, derivations).
- **Etymologies**: Detailed word origin and historical development.
- **Pronunciations**: Phonetic transcriptions and audio references.
- **Forms**: Inflected forms, variants, and other morphological variations.
- **Sources**: Tracking of source information for data entries.

(Consider adding a link to a visual schema diagram or more detailed data model documentation in `/docs` if available.)

## Contributing

We welcome contributions from the community! Whether it's reporting a bug, proposing a new feature, improving documentation, or writing code, your help is appreciated.

Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on how to contribute, including our development workflow and coding standards.

All participants are expected to adhere to our [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Our sincere gratitude to the Filipino linguistic experts and community members who have contributed, or whose works have inspired, the dictionary content.
- This project stands on the shoulders of giants – the many open-source libraries and tools that make its development possible.
- To everyone who has provided feedback, suggestions, and support.

## Contact

For questions, support, or to report an issue, please use the following channels:
- **GitHub Issues:** [https://github.com/fish-and-bear/fil-relex/issues](https://github.com/fish-and-bear/fil-relex/issues) (Replace with actual repo URL)
- **Project Email:** `aanaguio@up.edu.ph` (Replace with a valid contact email)

---

*This README is a living document and will be updated as the project evolves.*