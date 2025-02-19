#!/bin/bash
set -e

# Configuration
DEPLOY_ENV=${1:-production}
VERSION=$(git describe --tags --always)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/${TIMESTAMP}"

# Load environment variables
if [ -f ".env.${DEPLOY_ENV}" ]; then
    source ".env.${DEPLOY_ENV}"
else
    echo "Error: Environment file .env.${DEPLOY_ENV} not found"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

# Check required tools
check_requirements() {
    log "Checking requirements..."
    command -v docker >/dev/null 2>&1 || { error "Docker is required but not installed."; exit 1; }
    command -v docker-compose >/dev/null 2>&1 || { error "Docker Compose is required but not installed."; exit 1; }
    command -v curl >/dev/null 2>&1 || { error "curl is required but not installed."; exit 1; }
}

# Create backup
create_backup() {
    log "Creating backup..."
    mkdir -p "${BACKUP_DIR}"
    
    # Backup database
    log "Backing up database..."
    docker-compose exec -T db pg_dump -U "${DB_USER}" "${DB_NAME}" > "${BACKUP_DIR}/database.sql"
    
    # Backup Redis data
    log "Backing up Redis data..."
    docker-compose exec -T redis redis-cli SAVE
    docker cp $(docker-compose ps -q redis):/data/dump.rdb "${BACKUP_DIR}/redis.rdb"
    
    # Backup configuration
    log "Backing up configuration..."
    cp .env* "${BACKUP_DIR}/"
    cp docker-compose.yml "${BACKUP_DIR}/"
}

# Build images
build_images() {
    log "Building Docker images..."
    export VERSION
    docker-compose build --pull --no-cache
}

# Run database migrations
run_migrations() {
    log "Running database migrations..."
    docker-compose run --rm backend alembic upgrade head
}

# Health check
health_check() {
    local service=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    log "Checking health of ${service}..."
    while [ $attempt -le $max_attempts ]; do
        if curl -f "${url}" >/dev/null 2>&1; then
            log "${service} is healthy"
            return 0
        fi
        warn "${service} is not ready (attempt ${attempt}/${max_attempts})"
        sleep 5
        attempt=$((attempt + 1))
    done
    
    error "${service} failed health check"
    return 1
}

# Deploy
deploy() {
    log "Starting deployment to ${DEPLOY_ENV}..."
    
    # Check requirements
    check_requirements
    
    # Create backup
    create_backup
    
    # Pull latest changes
    log "Pulling latest changes..."
    git pull origin main
    
    # Build images
    build_images
    
    # Stop current services
    log "Stopping current services..."
    docker-compose down --remove-orphans
    
    # Start new services
    log "Starting new services..."
    docker-compose up -d
    
    # Run migrations
    run_migrations
    
    # Health checks
    health_check "Backend" "http://localhost:10000/health"
    health_check "Frontend" "http://localhost:3000/health"
    
    # Verify deployment
    verify_deployment
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    # Check all services are running
    local services_status=$(docker-compose ps --services --filter "status=running")
    local expected_services="frontend backend db redis nginx prometheus grafana"
    
    for service in $expected_services; do
        if ! echo "$services_status" | grep -q "^$service$"; then
            error "Service $service is not running"
            rollback
            exit 1
        fi
    done
    
    # Check application version
    local api_version=$(curl -s http://localhost:10000/api/v2/ | jq -r '.version')
    if [ "$api_version" != "$VERSION" ]; then
        error "API version mismatch: expected $VERSION, got $api_version"
        rollback
        exit 1
    fi
    
    log "Deployment verified successfully"
}

# Rollback
rollback() {
    error "Deployment failed, rolling back..."
    
    # Restore from backup
    if [ -d "${BACKUP_DIR}" ]; then
        log "Restoring from backup..."
        
        # Restore database
        if [ -f "${BACKUP_DIR}/database.sql" ]; then
            docker-compose exec -T db psql -U "${DB_USER}" "${DB_NAME}" < "${BACKUP_DIR}/database.sql"
        fi
        
        # Restore Redis data
        if [ -f "${BACKUP_DIR}/redis.rdb" ]; then
            docker cp "${BACKUP_DIR}/redis.rdb" $(docker-compose ps -q redis):/data/dump.rdb
            docker-compose restart redis
        fi
        
        # Restore configuration
        cp "${BACKUP_DIR}"/.env* .
        cp "${BACKUP_DIR}"/docker-compose.yml .
        
        # Restart services
        docker-compose down
        docker-compose up -d
        
        log "Rollback completed"
    else
        error "No backup found for rollback"
    fi
}

# Cleanup
cleanup() {
    log "Cleaning up..."
    
    # Remove old backups (keep last 5)
    cd backups
    ls -t | tail -n +6 | xargs -r rm -r
    cd ..
    
    # Remove unused Docker resources
    docker system prune -f
    
    # Remove old images
    docker images | grep '<none>' | awk '{print $3}' | xargs -r docker rmi
}

# Main execution
main() {
    # Trap errors
    trap 'error "An error occurred. Exiting..."; rollback; exit 1' ERR
    
    # Start deployment
    deploy
    
    # Cleanup
    cleanup
    
    log "Deployment completed successfully"
}

# Run main function
main 