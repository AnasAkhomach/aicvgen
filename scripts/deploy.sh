#!/bin/bash

# AI CV Generator Deployment Script
# This script handles deployment of the AI CV Generator application
# Supports development, staging, and production environments

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEFAULT_ENV="development"
DEFAULT_PORT="8501"
DEFAULT_PROFILE="basic"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
AI CV Generator Deployment Script

Usage: $0 [OPTIONS] COMMAND

Commands:
  build         Build Docker images
  deploy        Deploy the application
  start         Start the application
  stop          Stop the application
  restart       Restart the application
  logs          Show application logs
  status        Show application status
  cleanup       Clean up Docker resources
  backup        Backup application data
  restore       Restore application data
  health        Check application health

Options:
  -e, --env ENV         Environment (development|staging|production) [default: $DEFAULT_ENV]
  -p, --port PORT       Port to expose [default: $DEFAULT_PORT]
  -f, --profile PROFILE Docker compose profile (basic|production|monitoring|caching) [default: $DEFAULT_PROFILE]
  -d, --detach          Run in detached mode
  -v, --verbose         Verbose output
  -h, --help            Show this help message

Examples:
  $0 deploy                           # Deploy in development mode
  $0 -e production -f production deploy  # Deploy in production with nginx
  $0 -e staging -p 8502 start         # Start staging environment on port 8502
  $0 logs                             # Show application logs
  $0 cleanup                          # Clean up Docker resources

EOF
}

# Parse command line arguments
ENVIRONMENT="$DEFAULT_ENV"
PORT="$DEFAULT_PORT"
PROFILE="$DEFAULT_PROFILE"
DETACH=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -f|--profile)
            PROFILE="$2"
            shift 2
            ;;
        -d|--detach)
            DETACH=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        build|deploy|start|stop|restart|logs|status|cleanup|backup|restore|health)
            COMMAND="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate command
if [[ -z "$COMMAND" ]]; then
    log_error "No command specified"
    show_help
    exit 1
fi

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
    log_error "Invalid environment: $ENVIRONMENT"
    exit 1
fi

# Set verbose mode
if [[ "$VERBOSE" == "true" ]]; then
    set -x
fi

# Change to project root
cd "$PROJECT_ROOT"

# Environment-specific configurations
setup_environment() {
    log_info "Setting up $ENVIRONMENT environment"
    
    # Create .env file if it doesn't exist
    if [[ ! -f ".env" ]]; then
        log_info "Creating .env file from template"
        cp ".env.example" ".env" 2>/dev/null || {
            log_warning ".env.example not found, creating basic .env"
            cat > .env << EOF
# AI CV Generator Environment Configuration
LOG_LEVEL=INFO
ENABLE_DEBUG_MODE=false
SESSION_TIMEOUT=3600
MAX_CONCURRENT_REQUESTS=10
GRAFANA_PASSWORD=admin
EOF
        }
    fi
    
    # Environment-specific overrides
    case "$ENVIRONMENT" in
        development)
            export LOG_LEVEL=DEBUG
            export ENABLE_DEBUG_MODE=true
            ;;
        staging)
            export LOG_LEVEL=INFO
            export ENABLE_DEBUG_MODE=false
            ;;
        production)
            export LOG_LEVEL=WARNING
            export ENABLE_DEBUG_MODE=false
            export PROFILE=production
            ;;
    esac
    
    # Create necessary directories
    mkdir -p data/sessions data/output logs config
    
    log_success "Environment setup complete"
}

# Pre-deployment checks
check_prerequisites() {
    log_info "Checking prerequisites"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check required files
    local required_files=("Dockerfile" "docker-compose.yml" "requirements.txt")
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "Required file not found: $file"
            exit 1
        fi
    done
    
    log_success "Prerequisites check passed"
}

# Build Docker images
build_images() {
    log_info "Building Docker images"
    
    local build_args=()
    if [[ "$ENVIRONMENT" == "production" ]]; then
        build_args+=("--target" "production")
    fi
    
    docker-compose build "${build_args[@]}" aicvgen
    
    log_success "Docker images built successfully"
}

# Deploy application
deploy_application() {
    log_info "Deploying AI CV Generator ($ENVIRONMENT environment)"
    
    setup_environment
    check_prerequisites
    build_images
    
    # Set compose options
    local compose_opts=()
    if [[ "$DETACH" == "true" ]]; then
        compose_opts+=("--detach")
    fi
    
    # Set profile
    export COMPOSE_PROFILES="$PROFILE"
    
    # Deploy with docker-compose
    docker-compose up "${compose_opts[@]}"
    
    if [[ "$DETACH" == "true" ]]; then
        log_success "Application deployed successfully in detached mode"
        log_info "Access the application at: http://localhost:$PORT"
    fi
}

# Start application
start_application() {
    log_info "Starting AI CV Generator"
    
    export COMPOSE_PROFILES="$PROFILE"
    
    local compose_opts=()
    if [[ "$DETACH" == "true" ]]; then
        compose_opts+=("--detach")
    fi
    
    docker-compose up "${compose_opts[@]}"
    
    log_success "Application started successfully"
}

# Stop application
stop_application() {
    log_info "Stopping AI CV Generator"
    
    docker-compose down
    
    log_success "Application stopped successfully"
}

# Restart application
restart_application() {
    log_info "Restarting AI CV Generator"
    
    stop_application
    start_application
    
    log_success "Application restarted successfully"
}

# Show logs
show_logs() {
    log_info "Showing application logs"
    
    docker-compose logs -f aicvgen
}

# Show status
show_status() {
    log_info "Application status:"
    
    docker-compose ps
    
    echo
    log_info "Container health:"
    docker-compose exec aicvgen curl -f http://localhost:8501/_stcore/health 2>/dev/null && \
        log_success "Application is healthy" || \
        log_warning "Application health check failed"
}

# Cleanup Docker resources
cleanup_resources() {
    log_info "Cleaning up Docker resources"
    
    # Stop and remove containers
    docker-compose down --remove-orphans
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes (with confirmation)
    read -p "Remove unused volumes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker volume prune -f
    fi
    
    log_success "Cleanup completed"
}

# Backup application data
backup_data() {
    log_info "Backing up application data"
    
    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup data directory
    if [[ -d "data" ]]; then
        cp -r data "$backup_dir/"
        log_success "Data directory backed up"
    fi
    
    # Backup logs
    if [[ -d "logs" ]]; then
        cp -r logs "$backup_dir/"
        log_success "Logs backed up"
    fi
    
    # Backup configuration
    if [[ -f ".env" ]]; then
        cp .env "$backup_dir/"
        log_success "Configuration backed up"
    fi
    
    log_success "Backup completed: $backup_dir"
}

# Restore application data
restore_data() {
    log_info "Available backups:"
    
    if [[ ! -d "backups" ]]; then
        log_error "No backups directory found"
        exit 1
    fi
    
    local backups=(backups/*/)
    if [[ ${#backups[@]} -eq 0 ]]; then
        log_error "No backups found"
        exit 1
    fi
    
    for i in "${!backups[@]}"; do
        echo "$((i+1)). $(basename "${backups[$i]}")"
    done
    
    read -p "Select backup to restore (1-${#backups[@]}): " -r backup_choice
    
    if [[ ! "$backup_choice" =~ ^[0-9]+$ ]] || [[ "$backup_choice" -lt 1 ]] || [[ "$backup_choice" -gt ${#backups[@]} ]]; then
        log_error "Invalid selection"
        exit 1
    fi
    
    local selected_backup="${backups[$((backup_choice-1))]}"
    
    log_warning "This will overwrite current data. Continue? (y/N)"
    read -p "" -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Restore cancelled"
        exit 0
    fi
    
    # Restore data
    if [[ -d "${selected_backup}data" ]]; then
        rm -rf data
        cp -r "${selected_backup}data" .
        log_success "Data restored"
    fi
    
    # Restore configuration
    if [[ -f "${selected_backup}.env" ]]; then
        cp "${selected_backup}.env" .
        log_success "Configuration restored"
    fi
    
    log_success "Restore completed from: $(basename "$selected_backup")"
}

# Health check
health_check() {
    log_info "Performing health check"
    
    # Check if containers are running
    if ! docker-compose ps | grep -q "Up"; then
        log_error "No containers are running"
        exit 1
    fi
    
    # Check application health endpoint
    if docker-compose exec aicvgen curl -f http://localhost:8501/_stcore/health &>/dev/null; then
        log_success "Application is healthy"
    else
        log_error "Application health check failed"
        exit 1
    fi
    
    # Check disk space
    local disk_usage=$(df . | tail -1 | awk '{print $5}' | sed 's/%//')
    if [[ "$disk_usage" -gt 90 ]]; then
        log_warning "Disk usage is high: ${disk_usage}%"
    else
        log_success "Disk usage is normal: ${disk_usage}%"
    fi
    
    # Check memory usage
    local memory_usage=$(docker stats --no-stream --format "table {{.Container}}\t{{.MemPerc}}" | grep aicvgen | awk '{print $2}' | sed 's/%//')
    if [[ -n "$memory_usage" ]] && [[ "${memory_usage%.*}" -gt 80 ]]; then
        log_warning "Memory usage is high: ${memory_usage}%"
    else
        log_success "Memory usage is normal: ${memory_usage:-N/A}%"
    fi
    
    log_success "Health check completed"
}

# Execute command
case "$COMMAND" in
    build)
        setup_environment
        check_prerequisites
        build_images
        ;;
    deploy)
        deploy_application
        ;;
    start)
        start_application
        ;;
    stop)
        stop_application
        ;;
    restart)
        restart_application
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    cleanup)
        cleanup_resources
        ;;
    backup)
        backup_data
        ;;
    restore)
        restore_data
        ;;
    health)
        health_check
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac

log_success "Operation completed successfully"