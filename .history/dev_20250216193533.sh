#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

# Check requirements
check_requirements() {
    log "Checking requirements..."
    command -v docker >/dev/null 2>&1 || { error "Docker is required but not installed."; exit 1; }
    command -v docker-compose >/dev/null 2>&1 || { error "Docker Compose is required but not installed."; exit 1; }
    command -v python3 >/dev/null 2>&1 || { error "Python 3 is required but not installed."; exit 1; }
    command -v node >/dev/null 2>&1 || { error "Node.js is required but not installed."; exit 1; }
}

# Setup environment
setup_env() {
    log "Setting up environment..."
    
    # Create local env file if it doesn't exist
    if [ ! -f ".env.local" ]; then
        log "Creating .env.local..."
        cp .env.example .env.local || warn "No .env.example found, skipping..."
    fi
    
    # Load environment variables
    set -a
    source .env.local
    set +a
    
    # Create necessary directories
    mkdir -p logs
    mkdir -p data
}

# Setup database
setup_db() {
    log "Setting up database..."
    
    # Wait for database to be ready
    until docker-compose -f docker-compose.local.yml exec -T db pg_isready -U "${DB_USER}" -d "${DB_NAME}"; do
        warn "Waiting for database to be ready..."
        sleep 2
    done
    
    # Run migrations
    log "Running database migrations..."
    docker-compose -f docker-compose.local.yml exec -T backend alembic upgrade head
    
    # Load initial data if needed
    if [ -f "backend/setup_db.sql" ]; then
        log "Loading initial data..."
        docker-compose -f docker-compose.local.yml exec -T db psql -U "${DB_USER}" -d "${DB_NAME}" -f /docker-entrypoint-initdb.d/setup_db.sql
    fi
}

# Start development environment
start_dev() {
    log "Starting development environment..."
    
    # Build and start containers
    docker-compose -f docker-compose.local.yml up -d --build
    
    # Setup database
    setup_db
    
    # Show service status
    docker-compose -f docker-compose.local.yml ps
    
    # Show access URLs
    log "Development environment is ready!"
    echo -e "${GREEN}Frontend: ${NC}http://localhost:3000"
    echo -e "${GREEN}Backend API: ${NC}http://localhost:10000"
    echo -e "${GREEN}Grafana: ${NC}http://localhost:3001"
    echo -e "${GREEN}Prometheus: ${NC}http://localhost:9090"
    echo -e "${GREEN}Jaeger UI: ${NC}http://localhost:16686"
}

# Stop development environment
stop_dev() {
    log "Stopping development environment..."
    docker-compose -f docker-compose.local.yml down
}

# Clean development environment
clean_dev() {
    log "Cleaning development environment..."
    docker-compose -f docker-compose.local.yml down -v
    rm -rf logs/*
    rm -rf data/*
}

# Watch logs
watch_logs() {
    log "Watching logs..."
    docker-compose -f docker-compose.local.yml logs -f
}

# Show help
show_help() {
    echo "Usage: $0 [command]"
    echo
    echo "Commands:"
    echo "  start    Start development environment"
    echo "  stop     Stop development environment"
    echo "  clean    Clean development environment (removes volumes and logs)"
    echo "  logs     Watch logs from all services"
    echo "  help     Show this help message"
}

# Main execution
main() {
    # Check command
    case "$1" in
        start)
            check_requirements
            setup_env
            start_dev
            ;;
        stop)
            stop_dev
            ;;
        clean)
            clean_dev
            ;;
        logs)
            watch_logs
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@" 