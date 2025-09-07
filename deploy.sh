#!/bin/bash

# QENEX Unified Deployment Script
# Production deployment for all QENEX components

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
QENEX_VERSION="10.0.0"
QENEX_HOME="/opt/qenex"
QENEX_DATA="/var/lib/qenex"
QENEX_LOG="/var/log/qenex"
QENEX_CONFIG="/etc/qenex"

echo -e "${GREEN}QENEX Unified Financial Operating System - Deployment${NC}"
echo "Version: $QENEX_VERSION"
echo "==========================================="

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check for Python 3.8+
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Python 3.8+ is required${NC}"
        exit 1
    fi
    
    # Check for Node.js
    if ! command -v node &> /dev/null; then
        echo -e "${YELLOW}Installing Node.js...${NC}"
        curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
        sudo apt-get install -y nodejs
    fi
    
    # Check for Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${YELLOW}Installing Docker...${NC}"
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        rm get-docker.sh
    fi
    
    echo -e "${GREEN}Prerequisites check completed${NC}"
}

# Create directory structure
create_directories() {
    echo -e "${YELLOW}Creating directory structure...${NC}"
    
    sudo mkdir -p "$QENEX_HOME"/{core,defi,ai,platform,contracts,services}
    sudo mkdir -p "$QENEX_DATA"/{blockchain,database,cache,temp}
    sudo mkdir -p "$QENEX_LOG"/{system,audit,performance}
    sudo mkdir -p "$QENEX_CONFIG"
    
    echo -e "${GREEN}Directory structure created${NC}"
}

# Install Python dependencies
install_python_deps() {
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    
    # Create virtual environment
    python3 -m venv "$QENEX_HOME/venv"
    source "$QENEX_HOME/venv/bin/activate"
    
    # Install core packages
    pip install --upgrade pip
    pip install \
        asyncio \
        aiohttp \
        web3 \
        numpy \
        pandas \
        scikit-learn \
        tensorflow \
        torch \
        cryptography \
        psutil \
        redis \
        celery \
        fastapi \
        uvicorn \
        sqlalchemy \
        alembic \
        pydantic \
        prometheus-client \
        pytest \
        black \
        pylint
    
    echo -e "${GREEN}Python dependencies installed${NC}"
}

# Deploy smart contracts
deploy_contracts() {
    echo -e "${YELLOW}Deploying smart contracts...${NC}"
    
    cd "$QENEX_HOME/contracts"
    
    # Install Hardhat
    npm init -y
    npm install --save-dev hardhat @nomiclabs/hardhat-ethers ethers
    npm install --save-dev @openzeppelin/contracts
    
    # Compile contracts
    npx hardhat compile
    
    # Deploy to network
    if [ "$1" == "mainnet" ]; then
        npx hardhat run scripts/deploy.js --network mainnet
    elif [ "$1" == "testnet" ]; then
        npx hardhat run scripts/deploy.js --network testnet
    else
        npx hardhat run scripts/deploy.js --network localhost
    fi
    
    echo -e "${GREEN}Smart contracts deployed${NC}"
}

# Configure system
configure_system() {
    echo -e "${YELLOW}Configuring system...${NC}"
    
    # Generate configuration file
    cat > "$QENEX_CONFIG/config.yaml" << EOF
qenex:
  version: $QENEX_VERSION
  mode: production
  
  network:
    host: 0.0.0.0
    port: 8080
    ssl:
      enabled: true
      cert: /etc/qenex/ssl/cert.pem
      key: /etc/qenex/ssl/key.pem
    max_connections: 10000
    timeout: 30000
  
  blockchain:
    network: mainnet
    consensus: proof_of_stake
    block_time: 1000
    max_block_size: 10485760
    gas_limit: 30000000
  
  ai:
    enabled: true
    optimization_interval: 3600
    learning_rate: 0.001
    exploration_rate: 0.1
    model_path: /var/lib/qenex/models
  
  defi:
    enabled: true
    liquidity_pools: true
    staking: true
    lending: true
    yield_farming: true
  
  security:
    encryption: aes-256-gcm
    key_rotation: 86400
    audit_logging: true
    rate_limiting: true
    ddos_protection: true
    quantum_resistant: true
  
  database:
    type: postgresql
    host: localhost
    port: 5432
    name: qenex
    user: qenex
    pool_size: 100
  
  redis:
    host: localhost
    port: 6379
    db: 0
    pool_size: 50
  
  monitoring:
    prometheus:
      enabled: true
      port: 9090
    grafana:
      enabled: true
      port: 3000
    logging:
      level: INFO
      file: /var/log/qenex/system/qenex.log
EOF
    
    echo -e "${GREEN}System configured${NC}"
}

# Setup database
setup_database() {
    echo -e "${YELLOW}Setting up database...${NC}"
    
    # Install PostgreSQL if not present
    if ! command -v psql &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y postgresql postgresql-contrib
    fi
    
    # Create database and user
    sudo -u postgres psql << EOF
CREATE USER qenex WITH PASSWORD 'qenex_secure_password';
CREATE DATABASE qenex OWNER qenex;
GRANT ALL PRIVILEGES ON DATABASE qenex TO qenex;
EOF
    
    # Run migrations
    source "$QENEX_HOME/venv/bin/activate"
    cd "$QENEX_HOME/core"
    alembic upgrade head
    
    echo -e "${GREEN}Database setup completed${NC}"
}

# Setup Redis
setup_redis() {
    echo -e "${YELLOW}Setting up Redis...${NC}"
    
    # Install Redis if not present
    if ! command -v redis-server &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y redis-server
    fi
    
    # Configure Redis
    sudo sed -i 's/^# requirepass/requirepass qenex_redis_password/' /etc/redis/redis.conf
    sudo systemctl restart redis-server
    
    echo -e "${GREEN}Redis setup completed${NC}"
}

# Deploy services
deploy_services() {
    echo -e "${YELLOW}Deploying services...${NC}"
    
    # Copy service files
    cp core/*.py "$QENEX_HOME/core/"
    cp defi/*.py "$QENEX_HOME/defi/"
    cp ai/*.py "$QENEX_HOME/ai/"
    cp platform/*.py "$QENEX_HOME/platform/"
    
    # Create systemd services
    sudo cat > /etc/systemd/system/qenex-core.service << EOF
[Unit]
Description=QENEX Core Financial Engine
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=qenex
Group=qenex
WorkingDirectory=$QENEX_HOME/core
Environment="PATH=$QENEX_HOME/venv/bin"
ExecStart=$QENEX_HOME/venv/bin/python unified_financial_core.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    sudo cat > /etc/systemd/system/qenex-ai.service << EOF
[Unit]
Description=QENEX AI Self-Improvement System
After=network.target qenex-core.service

[Service]
Type=simple
User=qenex
Group=qenex
WorkingDirectory=$QENEX_HOME/ai
Environment="PATH=$QENEX_HOME/venv/bin"
ExecStart=$QENEX_HOME/venv/bin/python self_improving_system.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    sudo cat > /etc/systemd/system/qenex-defi.service << EOF
[Unit]
Description=QENEX DeFi Engine
After=network.target qenex-core.service

[Service]
Type=simple
User=qenex
Group=qenex
WorkingDirectory=$QENEX_HOME/defi
Environment="PATH=$QENEX_HOME/venv/bin"
ExecStart=$QENEX_HOME/venv/bin/python advanced_defi_engine.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    # Create qenex user if not exists
    if ! id -u qenex > /dev/null 2>&1; then
        sudo useradd -r -s /bin/false qenex
    fi
    
    # Set permissions
    sudo chown -R qenex:qenex "$QENEX_HOME"
    sudo chown -R qenex:qenex "$QENEX_DATA"
    sudo chown -R qenex:qenex "$QENEX_LOG"
    sudo chown -R qenex:qenex "$QENEX_CONFIG"
    
    # Reload systemd and start services
    sudo systemctl daemon-reload
    sudo systemctl enable qenex-core qenex-ai qenex-defi
    sudo systemctl start qenex-core qenex-ai qenex-defi
    
    echo -e "${GREEN}Services deployed${NC}"
}

# Setup monitoring
setup_monitoring() {
    echo -e "${YELLOW}Setting up monitoring...${NC}"
    
    # Deploy Prometheus
    docker run -d \
        --name prometheus \
        -p 9090:9090 \
        -v "$QENEX_CONFIG/prometheus.yml:/etc/prometheus/prometheus.yml" \
        prom/prometheus
    
    # Deploy Grafana
    docker run -d \
        --name grafana \
        -p 3000:3000 \
        -e "GF_SECURITY_ADMIN_PASSWORD=qenex_admin" \
        grafana/grafana
    
    echo -e "${GREEN}Monitoring setup completed${NC}"
}

# Setup SSL certificates
setup_ssl() {
    echo -e "${YELLOW}Setting up SSL certificates...${NC}"
    
    # Install certbot if not present
    if ! command -v certbot &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y certbot
    fi
    
    # Generate self-signed certificate for development
    if [ "$1" != "production" ]; then
        sudo mkdir -p "$QENEX_CONFIG/ssl"
        sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout "$QENEX_CONFIG/ssl/key.pem" \
            -out "$QENEX_CONFIG/ssl/cert.pem" \
            -subj "/C=US/ST=State/L=City/O=QENEX/CN=localhost"
    else
        # For production, use Let's Encrypt
        sudo certbot certonly --standalone -d "$2" --non-interactive --agree-tos -m admin@qenex.ai
        sudo ln -s "/etc/letsencrypt/live/$2/fullchain.pem" "$QENEX_CONFIG/ssl/cert.pem"
        sudo ln -s "/etc/letsencrypt/live/$2/privkey.pem" "$QENEX_CONFIG/ssl/key.pem"
    fi
    
    echo -e "${GREEN}SSL certificates configured${NC}"
}

# Setup firewall
setup_firewall() {
    echo -e "${YELLOW}Configuring firewall...${NC}"
    
    # Install ufw if not present
    if ! command -v ufw &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y ufw
    fi
    
    # Configure firewall rules
    sudo ufw default deny incoming
    sudo ufw default allow outgoing
    sudo ufw allow 22/tcp    # SSH
    sudo ufw allow 80/tcp    # HTTP
    sudo ufw allow 443/tcp   # HTTPS
    sudo ufw allow 8080/tcp  # QENEX API
    sudo ufw allow 9090/tcp  # Prometheus
    sudo ufw allow 3000/tcp  # Grafana
    sudo ufw allow 30303/tcp # Blockchain P2P
    
    # Enable firewall
    sudo ufw --force enable
    
    echo -e "${GREEN}Firewall configured${NC}"
}

# Health check
health_check() {
    echo -e "${YELLOW}Performing health check...${NC}"
    
    # Check services
    services=("qenex-core" "qenex-ai" "qenex-defi")
    for service in "${services[@]}"; do
        if systemctl is-active --quiet "$service"; then
            echo -e "${GREEN}✓ $service is running${NC}"
        else
            echo -e "${RED}✗ $service is not running${NC}"
        fi
    done
    
    # Check API endpoint
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health | grep -q "200"; then
        echo -e "${GREEN}✓ API is responding${NC}"
    else
        echo -e "${RED}✗ API is not responding${NC}"
    fi
    
    # Check database connection
    if PGPASSWORD=qenex_secure_password psql -h localhost -U qenex -d qenex -c "SELECT 1" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Database connection successful${NC}"
    else
        echo -e "${RED}✗ Database connection failed${NC}"
    fi
    
    # Check Redis connection
    if redis-cli -a qenex_redis_password ping | grep -q "PONG"; then
        echo -e "${GREEN}✓ Redis connection successful${NC}"
    else
        echo -e "${RED}✗ Redis connection failed${NC}"
    fi
    
    echo -e "${GREEN}Health check completed${NC}"
}

# Main deployment function
main() {
    echo -e "${GREEN}Starting QENEX deployment...${NC}"
    
    # Parse arguments
    ENVIRONMENT="${1:-development}"
    DOMAIN="${2:-localhost}"
    
    echo "Environment: $ENVIRONMENT"
    echo "Domain: $DOMAIN"
    
    # Run deployment steps
    check_prerequisites
    create_directories
    install_python_deps
    deploy_contracts "$ENVIRONMENT"
    configure_system
    setup_database
    setup_redis
    deploy_services
    setup_monitoring
    setup_ssl "$ENVIRONMENT" "$DOMAIN"
    setup_firewall
    health_check
    
    echo -e "${GREEN}==========================================="
    echo "QENEX deployment completed successfully!"
    echo "==========================================="
    echo ""
    echo "Access points:"
    echo "  API: https://$DOMAIN:8080"
    echo "  Prometheus: http://$DOMAIN:9090"
    echo "  Grafana: http://$DOMAIN:3000 (admin/qenex_admin)"
    echo ""
    echo "To check system status:"
    echo "  systemctl status qenex-core"
    echo "  systemctl status qenex-ai"
    echo "  systemctl status qenex-defi"
    echo ""
    echo "Logs are available at: $QENEX_LOG"
    echo -e "${NC}"
}

# Run main function
main "$@"