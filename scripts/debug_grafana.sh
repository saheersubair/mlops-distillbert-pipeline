#!/bin/bash

# Debug Grafana Connection Issues
# Path: scripts/debug_grafana.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
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

echo "🔍 Debugging Grafana Connection Issues..."
echo "================================================"

# Step 1: Check if Docker is running
print_status "Step 1: Checking Docker status..."
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running or not accessible"
    print_status "Try: sudo systemctl start docker"
    exit 1
else
    print_success "Docker is running"
fi

# Step 2: Check for port conflicts
print_status "Step 2: Checking for port conflicts..."
if netstat -tuln 2>/dev/null | grep -q ":3000 "; then
    print_warning "Port 3000 is already in use:"
    netstat -tuln | grep ":3000 " || lsof -i :3000 2>/dev/null || true
    print_status "You may need to stop the conflicting service or change Grafana port"
else
    print_success "Port 3000 is available"
fi

# Step 3: Stop all services and clean up
print_status "Step 3: Cleaning up existing containers..."
docker-compose down -v 2>/dev/null || true
docker container stop mlops-grafana 2>/dev/null || true
docker container rm mlops-grafana 2>/dev/null || true

# Step 4: Start only Grafana first
print_status "Step 4: Starting Grafana container alone..."

# Create Grafana container manually for debugging
docker run -d \
  --name mlops-grafana-debug \
  -p 3000:3000 \
  -e "GF_SECURITY_ADMIN_USER=admin" \
  -e "GF_SECURITY_ADMIN_PASSWORD=admin123" \
  -e "GF_USERS_ALLOW_SIGN_UP=false" \
  -e "GF_LOG_LEVEL=debug" \
  grafana/grafana:latest

print_status "Waiting for Grafana to start (30 seconds)..."
sleep 30

# Step 5: Check container status
print_status "Step 5: Checking Grafana container status..."
if docker ps | grep -q "mlops-grafana-debug"; then
    print_success "Grafana container is running"
    docker ps | grep mlops-grafana-debug
else
    print_error "Grafana container is not running"
    print_status "Container logs:"
    docker logs mlops-grafana-debug 2>/dev/null || echo "No logs available"
    exit 1
fi

# Step 6: Check container logs
print_status "Step 6: Checking Grafana logs..."
echo "--- Last 20 lines of Grafana logs ---"
docker logs --tail 20 mlops-grafana-debug

# Step 7: Test container network
print_status "Step 7: Testing container networking..."
container_ip=$(docker inspect mlops-grafana-debug | grep '"IPAddress"' | head -1 | cut -d'"' -f4)
print_status "Container IP: $container_ip"

# Test if Grafana is responding inside container
if docker exec mlops-grafana-debug wget -qO- http://localhost:3000/api/health 2>/dev/null; then
    print_success "Grafana is responding inside container"
else
    print_error "Grafana is not responding inside container"
    docker exec mlops-grafana-debug ps aux | grep grafana || true
fi

# Step 8: Test port mapping
print_status "Step 8: Testing port mapping..."
if docker port mlops-grafana-debug | grep -q "3000/tcp"; then
    port_mapping=$(docker port mlops-grafana-debug 3000/tcp)
    print_success "Port mapping exists: $port_mapping"
else
    print_error "No port mapping found for port 3000"
fi

# Step 9: Test external access
print_status "Step 9: Testing external access..."

# Test with curl
if command -v curl >/dev/null 2>&1; then
    print_status "Testing with curl..."
    if curl -f -s http://localhost:3000/api/health >/dev/null; then
        print_success "✅ Grafana is accessible via curl"
    else
        print_error "❌ Grafana is not accessible via curl"
        print_status "Curl response:"
        curl -v http://localhost:3000/api/health 2>&1 || true
    fi
else
    print_warning "curl not available for testing"
fi

# Test with wget
if command -v wget >/dev/null 2>&1; then
    print_status "Testing with wget..."
    if wget -qO- http://localhost:3000/api/health >/dev/null; then
        print_success "✅ Grafana is accessible via wget"
    else
        print_error "❌ Grafana is not accessible via wget"
    fi
else
    print_warning "wget not available for testing"
fi

# Step 10: Check system firewall
print_status "Step 10: Checking system firewall..."
if command -v ufw >/dev/null 2>&1; then
    if ufw status | grep -q "Status: active"; then
        print_warning "UFW firewall is active"
        ufw status | grep 3000 || print_status "Port 3000 not explicitly allowed in UFW"
    else
        print_success "UFW firewall is inactive"
    fi
fi

if command -v firewall-cmd >/dev/null 2>&1; then
    if firewall-cmd --state 2>/dev/null | grep -q "running"; then
        print_warning "Firewalld is active"
        if firewall-cmd --list-ports | grep -q "3000"; then
            print_success "Port 3000 is allowed in firewalld"
        else
            print_warning "Port 3000 may not be allowed in firewalld"
        fi
    else
        print_success "Firewalld is not running"
    fi
fi

# Step 11: Alternative access methods
print_status "Step 11: Alternative access methods..."
echo ""
echo "🌐 Try these alternative ways to access Grafana:"
echo ""
echo "1. Direct container IP (if Docker is on same machine):"
echo "   http://$container_ip:3000"
echo ""
echo "2. If using Docker Desktop on Windows/Mac:"
echo "   http://127.0.0.1:3000"
echo ""
echo "3. If running on a remote server:"
echo "   http://YOUR_SERVER_IP:3000"
echo ""
echo "4. Check if you're using WSL2 (Windows):"
echo "   Try: http://$(hostname -I | awk '{print $1}'):3000"
echo ""

# Step 12: Provide manual commands
print_status "Step 12: Manual testing commands..."
echo ""
echo "🔧 Manual testing commands:"
echo ""
echo "# Check if container is running:"
echo "docker ps | grep grafana"
echo ""
echo "# Check container logs:"
echo "docker logs mlops-grafana-debug"
echo ""
echo "# Test inside container:"
echo "docker exec mlops-grafana-debug curl http://localhost:3000/api/health"
echo ""
echo "# Check port mapping:"
echo "docker port mlops-grafana-debug"
echo ""
echo "# Access container shell:"
echo "docker exec -it mlops-grafana-debug /bin/bash"
echo ""

# Step 13: Clean up and restart with docker-compose
print_status "Step 13: Starting with docker-compose..."

# Stop debug container
docker stop mlops-grafana-debug 2>/dev/null || true
docker rm mlops-grafana-debug 2>/dev/null || true

# Create minimal docker-compose for testing
cat > docker-compose-grafana-test.yml << 'EOF'
version: '3.8'
services:
  grafana:
    image: grafana/grafana:latest
    container_name: grafana-test
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin123
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_LOG_LEVEL=info
    restart: unless-stopped
EOF

print_status "Starting Grafana with docker-compose..."
docker-compose -f docker-compose-grafana-test.yml up -d

print_status "Waiting for Grafana to start..."
sleep 30

# Final test
print_status "Final test..."
if curl -f -s http://localhost:3000/api/health >/dev/null 2>&1; then
    print_success "🎉 SUCCESS! Grafana is now accessible at http://localhost:3000"
    print_success "Login: admin / admin123"
    echo ""
    echo "✅ You can now access Grafana dashboard!"
    echo "✅ Use the login credentials: admin / admin123"
else
    print_error "❌ Still cannot access Grafana"
    echo ""
    echo "🆘 TROUBLESHOOTING STEPS:"
    echo "1. Check Docker logs: docker-compose -f docker-compose-grafana-test.yml logs"
    echo "2. Try different ports: Change 3000:3000 to 3001:3000 in docker-compose"
    echo "3. Check your browser: Try incognito mode or different browser"
    echo "4. Check network: If on remote server, ensure port 3000 is open"
    echo "5. Try container IP directly: http://$container_ip:3000"
fi

echo ""
echo "================================================"
echo "🔍 Debug complete!"
echo "================================================"