#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "Starting local deployment..."

# Check prerequisites
echo "Checking prerequisites..."

if ! command_exists psql; then
    echo "PostgreSQL is not installed. Please install PostgreSQL first."
    exit 1
fi

if ! command_exists python; then
    echo "Python is not installed. Please install Python first."
    exit 1
fi

if ! command_exists npm; then
    echo "Node.js/npm is not installed. Please install Node.js first."
    exit 1
fi

# Setup database
echo "Setting up database..."
export PGPASSWORD="ta3m1n.!"  # Set PostgreSQL password
psql -U postgres -f backend/setup_db.sql || echo "Database setup completed with warnings (this is normal if the database already exists)"

# Setup backend
echo "Setting up backend..."
cd backend
if [ ! -d "venv" ]; then
    python -m venv venv
fi
source venv/bin/activate

# Install requirements
echo "Installing Python dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

# Set PYTHONPATH
export PYTHONPATH="$PWD:$PYTHONPATH"

# Run tests
echo "Running tests..."
python -m pytest tests/

# Start backend server
echo "Starting backend server..."
python app.py &
BACKEND_PID=$!
cd ..

# Setup frontend
echo "Setting up frontend..."

# Create package.json if it doesn't exist
if [ ! -f "package.json" ]; then
    cat > package.json << EOF
{
    "name": "word-relationship-webapp",
    "version": "0.1.0",
    "private": true,
    "scripts": {
        "start": "react-scripts start",
        "build": "react-scripts build",
        "test": "react-scripts test",
        "eject": "react-scripts eject"
    }
}
EOF
fi

# Install dependencies
npm install --legacy-peer-deps
npm install --save-dev @babel/plugin-proposal-private-property-in-object --legacy-peer-deps

# Update browserslist
echo "Updating browserslist..."
npx update-browserslist-db@latest

# Build and start frontend
echo "Building frontend..."
npm run build

echo "Starting frontend server..."
npm start &
FRONTEND_PID=$!

echo "Deployment complete!"
echo "Backend running on http://localhost:10000"
echo "Frontend running on http://localhost:3000"
echo "Press Enter to stop the servers..."

# Wait for user input
read -p "Press Enter to stop the servers..."

# Cleanup
if [ -n "$BACKEND_PID" ]; then
    kill $BACKEND_PID 2>/dev/null || echo "Backend process already stopped"
fi

if [ -n "$FRONTEND_PID" ]; then
    kill $FRONTEND_PID 2>/dev/null || echo "Frontend process already stopped"
    pkill -f "node.*react-scripts start" 2>/dev/null
fi

deactivate

echo "Servers stopped." 