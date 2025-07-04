#!/bin/bash

# Setup Monitoring for MLOps Pipeline
# Path: scripts/setup_monitoring.sh

set -e

echo "🚀 Setting up MLOps Monitoring Stack..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    # Check if Docker is running
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi

    print_success "Prerequisites check passed"
}

# Create directory structure
create_directories() {
    print_status "Creating directory structure..."

    # Create monitoring directories
    mkdir -p monitoring/grafana/dashboards
    mkdir -p monitoring/grafana/datasources
    mkdir -p data
    mkdir -p models
    mkdir -p logs
    mkdir -p config

    print_success "Directory structure created"
}

# Create Grafana datasource configuration
create_grafana_datasource() {
    print_status "Creating Grafana datasource configuration..."

    cat > monitoring/grafana/datasources/datasources.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
    jsonData:
      httpMethod: POST
      queryTimeout: 60s
      timeInterval: 5s
EOF

    print_success "Grafana datasource configuration created"
}

# Create Grafana dashboard provisioning
create_grafana_provisioning() {
    print_status "Creating Grafana dashboard provisioning..."

    cat > monitoring/grafana/dashboards/dashboards.yml << 'EOF'
apiVersion: 1

providers:
  - name: 'MLOps Dashboards'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 30
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF

    print_success "Grafana dashboard provisioning created"
}

# Create simple Grafana dashboard
create_simple_dashboard() {
    print_status "Creating simple Grafana dashboard..."

    cat > monitoring/grafana/dashboards/mlops-simple-dashboard.json << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "MLOps API Monitoring",
    "tags": ["mlops", "api"],
    "style": "dark",
    "timezone": "browser",
    "refresh": "30s",
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "panels": [
      {
        "id": 1,
        "title": "API Health",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"mlops-api\"}",
            "refId": "A",
            "legendFormat": "API Status"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "mappings": [
              {
                "options": {
                  "0": {
                    "text": "DOWN",
                    "color": "red"
                  },
                  "1": {
                    "text": "UP",
                    "color": "green"
                  }
                },
                "type": "value"
              }
            ],
            "thresholds": {
              "steps": [
                {
                  "color": "red",
                  "value": null
                },
                {
                  "color": "green",
                  "value": 1
                }
              ]
            }
          }
        },
        "gridPos": {
          "h": 8,
          "w": 6,
          "x": 0,
          "y": 0
        }
      },
      {
        "id": 2,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(mlops_prediction_requests_total[5m])",
            "refId": "A",
            "legendFormat": "Requests/sec"
          }
        ],
        "yAxes": [
          {
            "label": "Requests/sec",
            "min": 0,
            "show": true
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 6,
          "y": 0
        }
      },
      {
        "id": 3,
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(mlops_prediction_latency_seconds_bucket[5m]))",
            "refId": "A",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(mlops_prediction_latency_seconds_bucket[5m]))",
            "refId": "B",
            "legendFormat": "50th percentile"
          }
        ],
        "yAxes": [
          {
            "label": "Seconds",
            "min": 0,
            "show": true
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 8
        }
      },
      {
        "id": 4,
        "title": "Error Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(mlops_prediction_requests_total{status=\"error\"}[5m]) / rate(mlops_prediction_requests_total[5m]) * 100",
            "refId": "A",
            "legendFormat": "Error Rate %"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "thresholds": {
              "steps": [
                {
                  "color": "green",
                  "value": null
                },
                {
                  "color": "yellow",
                  "value": 1
                },
                {
                  "color": "red",
                  "value": 5
                }
              ]
            }
          }
        },
        "gridPos": {
          "h": 8,
          "w": 6,
          "x": 12,
          "y": 8
        }
      }
    ]
  }
}
EOF

    print_success "Simple Grafana dashboard created"
}

# Start monitoring stack
start_monitoring() {
    print_status "Starting monitoring stack..."

    # Stop any existing containers
    docker-compose down 2>/dev/null || true

    # Start the services
    docker-compose up -d prometheus grafana node-exporter redis

    # Wait for services to start
    print_status "Waiting for services to start..."
    sleep 30

    # Check if services are running
    if docker-compose ps | grep -q "Up"; then
        print_success "Monitoring stack started successfully"
    else
        print_error "Failed to start monitoring stack"
        docker-compose logs
        exit 1
    fi
}

# Test endpoints
test_endpoints() {
    print_status "Testing endpoints..."

    # Test Prometheus
    if curl -s http://localhost:9090/-/healthy > /dev/null; then
        print_success "✅ Prometheus is healthy: http://localhost:9090"
    else
        print_warning "⚠️ Prometheus is not responding on http://localhost:9090"
    fi

    # Test Grafana
    if curl -s http://localhost:3000/api/health > /dev/null; then
        print_success "✅ Grafana is healthy: http://localhost:3000 (admin/admin123)"
    else
        print_warning "⚠️ Grafana is not responding on http://localhost:3000"
    fi

    # Test Node Exporter
    if curl -s http://localhost:9100/metrics > /dev/null; then
        print_success "✅ Node Exporter is healthy: http://localhost:9100/metrics"
    else
        print_warning "⚠️ Node Exporter is not responding on http://localhost:9100"
    fi
}

# Generate sample data
generate_sample_data() {
    print_status "Generating sample data..."

    if [ -f "generate_sample_data.py" ]; then
        python generate_sample_data.py
        print_success "Sample data generated"
    else
        print_warning "generate_sample_data.py not found, skipping data generation"
    fi
}

# Start API service
start_api() {
    print_status "Starting API service..."

    # Start the API service
    docker-compose up -d mlops-api

    # Wait for API to start
    print_status "Waiting for API to start..."
    sleep 45

    # Test API health
    for i in {1..10}; do
        if curl -s http://localhost:8000/health > /dev/null; then
            print_success "✅ API is healthy: http://localhost:8000"
            print_success "✅ API Metrics: http://localhost:8000/metrics"
            print_success "✅ API Docs: http://localhost:8000/docs"
            return 0
        fi
        print_status "Waiting for API... ($i/10)"
        sleep 10
    done

    print_warning "⚠️ API is not responding, checking logs..."
    docker-compose logs mlops-api
}

# Make test requests to generate metrics
generate_test_metrics() {
    print_status "Generating test metrics..."

    # Make some test requests
    for i in {1..10}; do
        curl -s -X POST "http://localhost:8000/predict" \
            -H "Content-Type: application/json" \
            -d "{\"text\": \"This is test message $i\"}" > /dev/null || true
        sleep 1
    done

    print_success "Test metrics generated"
}

# Main execution
main() {
    echo "============================================"
    echo "🚀 MLOps Monitoring Setup Script"
    echo "============================================"

    check_prerequisites
    create_directories
    create_grafana_datasource
    create_grafana_provisioning
    create_simple_dashboard
    generate_sample_data
    start_monitoring
    start_api
    test_endpoints
    generate_test_metrics

    echo ""
    echo "============================================"
    echo "🎉 Setup Complete!"
    echo "============================================"
    echo ""
    echo "📊 Access your monitoring stack:"
    echo "• Prometheus: http://localhost:9090"
    echo "• Grafana: http://localhost:3000 (admin/admin123)"
    echo "• API Health: http://localhost:8000/health"
    echo "• API Metrics: http://localhost:8000/metrics"
    echo "• API Docs: http://localhost:8000/docs"
    echo ""
    echo "🧪 Test the API:"
    echo "curl -X POST \"http://localhost:8000/predict\" \\"
    echo "  -H \"Content-Type: application/json\" \\"
    echo "  -d '{\"text\": \"I love this product!\"}'"
    echo ""
    echo "📈 View metrics in Grafana:"
    echo "1. Go to http://localhost:3000"
    echo "2. Login with admin/admin123"
    echo "3. Navigate to Dashboards > MLOps API Monitoring"
    echo ""
    echo "🐛 Troubleshooting:"
    echo "• Check logs: docker-compose logs"
    echo "• Restart services: docker-compose restart"
    echo "• Stop all: docker-compose down"
    echo ""
}

# Parse command line arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "MLOps Monitoring Setup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h     Show this help message"
    echo "  --skip-api     Skip API service startup"
    echo "  --no-test      Skip endpoint testing"
    echo ""
    exit 0
fi

# Check for skip options
SKIP_API=false
NO_TEST=false

for arg in "$@"; do
    case $arg in
        --skip-api)
            SKIP_API=true
            shift
            ;;
        --no-test)
            NO_TEST=true
            shift
            ;;
    esac
done

# Run main function with options
if [ "$SKIP_API" = true ]; then
    echo "Skipping API service startup..."
    check_prerequisites
    create_directories
    create_grafana_datasource
    create_grafana_provisioning
    create_simple_dashboard
    start_monitoring
    if [ "$NO_TEST" = false ]; then
        test_endpoints
    fi
else
    main
fi