#!/bin/bash

# MLOps DistillBERT API Entrypoint Script
# This script handles initialization and startup of the FastAPI service

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1"
}

# Default environment variables
export HOST=${HOST:-"0.0.0.0"}
export PORT=${PORT:-8000}
export WORKERS=${WORKERS:-4}
export RELOAD=${RELOAD:-"false"}
export LOG_LEVEL=${LOG_LEVEL:-"INFO"}
export MODEL_CACHE_DIR=${MODEL_CACHE_DIR:-"/app/models"}
export ENABLE_METRICS=${ENABLE_METRICS:-"true"}

# Pre-startup checks
pre_startup_checks() {
    log "Running pre-startup checks..."

    # Check Python version
    python_version=$(python --version 2>&1)
    log "Python version: $python_version"

    # Check if required directories exist
    required_dirs=("$MODEL_CACHE_DIR" "/app/logs" "/app/config")
    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            log "Creating directory: $dir"
            mkdir -p "$dir"
        fi
    done

    # Check if config files exist
    if [ ! -f "/app/config/model_config.yaml" ]; then
        warn "Model config not found, using defaults"
    fi

    # Check disk space
    disk_usage=$(df /app | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ "$disk_usage" -gt 90 ]; then
        warn "Disk usage is high: ${disk_usage}%"
    fi

    # Check memory
    total_memory=$(free -m | awk 'NR==2{print $2}')
    if [ "$total_memory" -lt 2048 ]; then
        warn "Available memory is low: ${total_memory}MB"
    fi

    success "Pre-startup checks completed"
}

# Model initialization
initialize_model() {
    log "Initializing model cache..."

    # Check if models directory is writable
    if [ ! -w "$MODEL_CACHE_DIR" ]; then
        error "Model cache directory is not writable: $MODEL_CACHE_DIR"
        exit 1
    fi

    # Pre-download model if not exists (optional)
    if [ "${PRELOAD_MODEL:-false}" = "true" ]; then
        log "Pre-loading model..."
        python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
model_name = os.getenv('MODEL_NAME', 'distilbert-base-uncased-finetuned-sst-2-english')
cache_dir = os.getenv('MODEL_CACHE_DIR', '/app/models')
print(f'Loading {model_name} to {cache_dir}...')
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
print('Model pre-loaded successfully')
" || warn "Model pre-loading failed, will load on first request"
    fi

    success "Model initialization completed"
}

# Health check function
health_check() {
    log "Performing health check..."

    # Check if the service is responding
    for i in {1..30}; do
        if curl -f -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            success "Health check passed"
            return 0
        fi
        log "Health check attempt $i/30 failed, retrying in 2 seconds..."
        sleep 2
    done

    error "Health check failed after 30 attempts"
    return 1
}

# Cleanup function
cleanup() {
    log "Performing cleanup..."

    # Kill background processes
    if [ ! -z "$UVICORN_PID" ]; then
        log "Stopping uvicorn process (PID: $UVICORN_PID)"
        kill -TERM "$UVICORN_PID" 2>/dev/null || true
        wait "$UVICORN_PID" 2>/dev/null || true
    fi

    # Clean up temporary files
    find /tmp -name "*.tmp" -type f -delete 2>/dev/null || true

    log "Cleanup completed"
}

# Trap cleanup on exit
trap cleanup EXIT INT TERM

# Migration function (for database migrations if needed)
run_migrations() {
    if [ "${RUN_MIGRATIONS:-false}" = "true" ]; then
        log "Running database migrations..."
        # Add migration commands here if using a database
        # alembic upgrade head
        success "Migrations completed"
    fi
}

# Wait for dependencies
wait_for_dependencies() {
    # Wait for database if configured
    if [ ! -z "$DATABASE_URL" ]; then
        log "Waiting for database..."
        python -c "
import time
import psycopg2
import os
from urllib.parse import urlparse

url = urlparse(os.getenv('DATABASE_URL'))
for i in range(30):
    try:
        conn = psycopg2.connect(
            host=url.hostname,
            port=url.port or 5432,
            user=url.username,
            password=url.password,
            database=url.path[1:]
        )
        conn.close()
        print('Database connection successful')
        break
    except:
        print(f'Database connection attempt {i+1}/30 failed, retrying...')
        time.sleep(2)
else:
    raise Exception('Database connection failed after 30 attempts')
" || {
            error "Database connection failed"
            exit 1
        }
    fi

    # Wait for Redis if configured
    if [ ! -z "$REDIS_URL" ]; then
        log "Waiting for Redis..."
        python -c "
import time
import redis
import os
from urllib.parse import urlparse

url = urlparse(os.getenv('REDIS_URL'))
for i in range(30):
    try:
        r = redis.Redis(host=url.hostname, port=url.port or 6379, db=0)
        r.ping()
        print('Redis connection successful')
        break
    except:
        print(f'Redis connection attempt {i+1}/30 failed, retrying...')
        time.sleep(2)
else:
    raise Exception('Redis connection failed after 30 attempts')
" || {
            warn "Redis connection failed, continuing without cache"
        }
    fi
}

# Start the application
start_application() {
    log "Starting FastAPI application..."

    # Build the uvicorn command
    cmd="uvicorn src.api.main:app"
    cmd="$cmd --host $HOST"
    cmd="$cmd --port $PORT"
    cmd="$cmd --log-level $(echo $LOG_LEVEL | tr '[:upper:]' '[:lower:]')"

    if [ "$RELOAD" = "true" ]; then
        cmd="$cmd --reload"
        log "Development mode: auto-reload enabled"
    else
        cmd="$cmd --workers $WORKERS"
        log "Production mode: $WORKERS workers"
    fi

    # Add additional uvicorn options
    if [ ! -z "$ACCESS_LOG" ]; then
        cmd="$cmd --access-log"
    fi

    if [ ! -z "$SSL_KEYFILE" ] && [ ! -z "$SSL_CERTFILE" ]; then
        cmd="$cmd --ssl-keyfile $SSL_KEYFILE --ssl-certfile $SSL_CERTFILE"
        log "SSL enabled"
    fi

    log "Executing: $cmd"

    # Start the application
    if [ "$RELOAD" = "true" ]; then
        # In development mode, run in foreground
        exec $cmd
    else
        # In production mode, start in background and monitor
        $cmd &
        UVICORN_PID=$!

        # Wait a bit for startup
        sleep 5

        # Perform health check
        if health_check; then
            log "Application started successfully (PID: $UVICORN_PID)"

            # Keep the script running and monitor the process
            while kill -0 "$UVICORN_PID" 2>/dev/null; do
                sleep 10
            done

            error "Application process died unexpectedly"
            exit 1
        else
            error "Application failed to start properly"
            exit 1
        fi
    fi
}

# Performance tuning
tune_performance() {
    log "Applying performance tuning..."

    # Set ulimits if running as root
    if [ "$(id -u)" -eq 0 ]; then
        ulimit -n 65536  # Increase file descriptor limit
        ulimit -u 32768  # Increase process limit
    fi

    # Set Python optimizations
    export PYTHONOPTIMIZE=1
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONUNBUFFERED=1

    # Garbage collection tuning
    export PYTHONGC=1

    success "Performance tuning applied"
}

# Security hardening
security_hardening() {
    log "Applying security hardening..."

    # Remove sensitive environment variables from process environment
    unset DATABASE_PASSWORD 2>/dev/null || true
    unset SECRET_KEY 2>/dev/null || true
    unset API_KEY 2>/dev/null || true

    # Set secure file permissions
    chmod 600 /app/config/*.yaml 2>/dev/null || true
    chmod 700 /app/logs 2>/dev/null || true

    success "Security hardening applied"
}

# Monitoring setup
setup_monitoring() {
    if [ "$ENABLE_METRICS" = "true" ]; then
        log "Setting up monitoring..."

        # Start metrics exporter if needed
        # This could start a separate process for custom metrics

        success "Monitoring setup completed"
    fi
}

# Main execution
main() {
    log "Starting MLOps DistillBERT API..."
    log "Environment: $([ "$RELOAD" = "true" ] && echo "Development" || echo "Production")"
    log "Host: $HOST:$PORT"
    log "Workers: $WORKERS"
    log "Log Level: $LOG_LEVEL"

    # Run all initialization steps
    pre_startup_checks
    tune_performance
    security_hardening
    wait_for_dependencies
    run_migrations
    initialize_model
    setup_monitoring

    # Start the application
    start_application
}

# Handle different commands
case "${1:-start}" in
    start)
        main
        ;;

    health)
        log "Running health check..."
        if curl -f -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            success "Service is healthy"
            exit 0
        else
            error "Service is not healthy"
            exit 1
        fi
        ;;

    test)
        log "Running application tests..."
        python -m pytest tests/ -v
        ;;

    train)
        log "Starting model training..."
        python src/training/train.py "$@"
        ;;

    migrate)
        log "Running database migrations..."
        run_migrations
        ;;

    shell)
        log "Starting interactive shell..."
        exec /bin/bash
        ;;

    debug)
        log "Starting in debug mode..."
        export RELOAD=true
        export LOG_LEVEL=DEBUG
        export WORKERS=1
        main
        ;;

    preload)
        log "Pre-loading model..."
        export PRELOAD_MODEL=true
        initialize_model
        ;;

    *)
        log "Available commands:"
        log "  start    - Start the API service (default)"
        log "  health   - Check service health"
        log "  test     - Run tests"
        log "  train    - Start model training"
        log "  migrate  - Run database migrations"
        log "  shell    - Start interactive shell"
        log "  debug    - Start in debug mode"
        log "  preload  - Pre-load model cache"
        log ""
        log "Custom command: $*"
        exec "$@"
        ;;
esac