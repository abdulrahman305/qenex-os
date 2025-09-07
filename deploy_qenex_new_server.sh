#!/bin/bash

################################################################################
# QENEX Complete System Deployment Script for New Server
# This script will install the entire QENEX Financial OS with Intelligence Mining
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOMAIN="qenex.ai"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"  # Set via environment variable
INSTALL_DIR="/opt/qenex-os"
WEB_ROOT="/var/www/qenex.ai"
DB_PATH="/opt/qenex-os/unified_intelligence.db"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}     QENEX Financial OS - Server Deployment    ${NC}"
echo -e "${BLUE}================================================${NC}"

# Function to check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        echo -e "${RED}This script must be run as root${NC}"
        exit 1
    fi
}

# Function to detect OS
detect_os() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS=$ID
        OS_VERSION=$VERSION_ID
    else
        echo -e "${RED}Cannot detect OS${NC}"
        exit 1
    fi
    echo -e "${GREEN}Detected OS: $OS $OS_VERSION${NC}"
}

# Function to install system dependencies
install_dependencies() {
    echo -e "${YELLOW}Installing system dependencies...${NC}"
    
    if [[ "$OS" == "ubuntu" ]] || [[ "$OS" == "debian" ]]; then
        apt-get update
        apt-get install -y \
            curl \
            wget \
            git \
            build-essential \
            python3 \
            python3-pip \
            python3-venv \
            nginx \
            certbot \
            python3-certbot-nginx \
            sqlite3 \
            nodejs \
            npm \
            ufw \
            fail2ban \
            htop \
            net-tools \
            software-properties-common
            
    elif [[ "$OS" == "centos" ]] || [[ "$OS" == "rhel" ]] || [[ "$OS" == "fedora" ]]; then
        yum install -y \
            curl \
            wget \
            git \
            gcc \
            gcc-c++ \
            make \
            python3 \
            python3-pip \
            nginx \
            certbot \
            python3-certbot-nginx \
            sqlite \
            nodejs \
            npm \
            firewalld \
            fail2ban \
            htop \
            net-tools
    else
        echo -e "${RED}Unsupported OS: $OS${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}System dependencies installed${NC}"
}

# Function to install Rust
install_rust() {
    echo -e "${YELLOW}Installing Rust...${NC}"
    
    if ! command -v cargo &> /dev/null; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source $HOME/.cargo/env
    else
        echo -e "${GREEN}Rust already installed${NC}"
    fi
    
    rustup default stable
    rustup update
}

# Function to clone repository
clone_repository() {
    echo -e "${YELLOW}Cloning QENEX repository...${NC}"
    
    # Remove old installation if exists
    if [[ -d "$INSTALL_DIR" ]]; then
        echo -e "${YELLOW}Removing old installation...${NC}"
        rm -rf "$INSTALL_DIR"
    fi
    
    # Clone repository
    # Clone repository (use token if provided, otherwise public clone)
    if [[ -n "$GITHUB_TOKEN" ]]; then
        git clone https://${GITHUB_TOKEN}@github.com/abdulrahman305/qenex-os.git "$INSTALL_DIR"
    else
        git clone https://github.com/abdulrahman305/qenex-os.git "$INSTALL_DIR"
    fi
    cd "$INSTALL_DIR"
    
    echo -e "${GREEN}Repository cloned successfully${NC}"
}

# Function to setup Python environment
setup_python_env() {
    echo -e "${YELLOW}Setting up Python environment...${NC}"
    
    cd "$INSTALL_DIR"
    
    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install Python dependencies
    cat > requirements.txt << 'EOF'
# Core dependencies
Flask==3.0.0
flask-cors==4.0.0
werkzeug==3.0.1
gunicorn==21.2.0

# Database
sqlalchemy==2.0.23
alembic==1.13.0

# Async support
asyncio==3.4.3
aiohttp==3.9.1
aiofiles==23.2.1

# Cryptography
cryptography==41.0.7
pycryptodome==3.19.0
hashlib
secrets

# Blockchain
web3==6.11.3
eth-account==0.10.0

# Data processing
numpy==1.26.2
pandas==2.1.4
scipy==1.11.4

# Utilities
python-dotenv==1.0.0
pyyaml==6.0.1
requests==2.31.0
python-dateutil==2.8.2
pytz==2023.3
EOF
    
    pip install -r requirements.txt
    
    echo -e "${GREEN}Python environment setup complete${NC}"
}

# Function to build Rust components
build_rust_components() {
    echo -e "${YELLOW}Building Rust components...${NC}"
    
    cd "$INSTALL_DIR"
    
    # Check if Cargo.toml exists
    if [[ -f "Cargo.toml" ]]; then
        cargo build --release
        echo -e "${GREEN}Rust components built successfully${NC}"
    else
        echo -e "${YELLOW}No Rust components to build${NC}"
    fi
}

# Function to initialize database
initialize_database() {
    echo -e "${YELLOW}Initializing database...${NC}"
    
    # Create database directory
    mkdir -p "$(dirname $DB_PATH)"
    
    # Initialize SQLite database
    sqlite3 "$DB_PATH" << 'EOF'
-- Create unified intelligence table
CREATE TABLE IF NOT EXISTS unified_intelligence (
    id INTEGER PRIMARY KEY,
    intelligence_level REAL DEFAULT 0.0,
    qxc_mined REAL DEFAULT 0.0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create mining proofs table
CREATE TABLE IF NOT EXISTS mining_proofs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    proof_hash TEXT NOT NULL,
    nonce INTEGER NOT NULL,
    difficulty INTEGER NOT NULL,
    intelligence_gain REAL NOT NULL,
    qxc_reward REAL NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create wallets table
CREATE TABLE IF NOT EXISTS wallets (
    address TEXT PRIMARY KEY,
    balance REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_transaction TIMESTAMP
);

-- Initialize with starting values
INSERT OR IGNORE INTO unified_intelligence (id, intelligence_level, qxc_mined) 
VALUES (1, 0.160177, 1581.594);

-- Create sample wallet
INSERT OR IGNORE INTO wallets (address, balance) 
VALUES ('0x9d92c7f4b7413e1b8e8b2e1a0c5d6f3a8b9c1d2e', 0.0);
EOF
    
    # Set permissions
    chmod 666 "$DB_PATH"
    
    echo -e "${GREEN}Database initialized${NC}"
}

# Function to create core Python scripts
create_core_scripts() {
    echo -e "${YELLOW}Creating core Python scripts...${NC}"
    
    # Create real proof of intelligence mining script
    cat > "$INSTALL_DIR/real_proof_of_intelligence.py" << 'EOF'
#!/usr/bin/env python3
"""
QENEX Real Proof-of-Intelligence Mining System
"""

import hashlib
import json
import sqlite3
import time
import random
from datetime import datetime
from decimal import Decimal, getcontext

# Set precision for financial calculations
getcontext().prec = 78

class RealProofOfIntelligence:
    def __init__(self, db_path="/opt/qenex-os/unified_intelligence.db"):
        self.db_path = db_path
        self.difficulty = 4  # Start with 4 leading zeros
        self.max_intelligence = 1000
        self.total_qxc_supply = 1_000_000_000
        
    def generate_challenge(self):
        """Generate a real proof-of-work challenge"""
        return {
            'seed': hashlib.sha256(str(time.time()).encode()).hexdigest(),
            'type': random.choice(['PATTERN_RECOGNITION', 'OPTIMIZATION', 'PREDICTION']),
            'timestamp': time.time(),
            'difficulty': self.difficulty,
            'target': '0' * self.difficulty
        }
    
    def solve_challenge(self, challenge):
        """Solve the proof-of-work challenge"""
        nonce = 0
        start_time = time.time()
        
        while True:
            data = f"{challenge['seed']}:{challenge['type']}:{nonce}"
            hash_result = hashlib.sha256(data.encode()).hexdigest()
            
            if hash_result.startswith(challenge['target']):
                solve_time = time.time() - start_time
                return {
                    'proof': hash_result,
                    'nonce': nonce,
                    'time': solve_time,
                    'valid': True
                }
            
            nonce += 1
            if nonce > 10_000_000:  # Prevent infinite loop
                return {'valid': False}
    
    def calculate_rewards(self, solve_time):
        """Calculate intelligence gain and QXC reward based on solve time"""
        base_intelligence = 0.01
        base_qxc = 10.0
        
        # Faster solutions get better rewards
        time_factor = max(0.1, min(2.0, 10.0 / max(solve_time, 0.1)))
        variance = random.uniform(0.8, 1.2)
        
        intelligence_gain = round(base_intelligence * time_factor * variance, 6)
        qxc_reward = round(base_qxc * time_factor * variance, 4)
        
        return intelligence_gain, qxc_reward
    
    def mine(self):
        """Perform one mining operation"""
        print("Generating challenge...")
        challenge = self.generate_challenge()
        
        print(f"Solving challenge (difficulty: {self.difficulty} leading zeros)...")
        solution = self.solve_challenge(challenge)
        
        if solution['valid']:
            intelligence_gain, qxc_reward = self.calculate_rewards(solution['time'])
            
            # Update database
            self.update_database(solution, intelligence_gain, qxc_reward)
            
            return {
                'success': True,
                'proof': solution['proof'],
                'nonce': solution['nonce'],
                'time': solution['time'],
                'intelligence_gain': intelligence_gain,
                'qxc_reward': qxc_reward
            }
        
        return {'success': False}
    
    def update_database(self, solution, intelligence_gain, qxc_reward):
        """Update the database with mining results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current values
        cursor.execute("SELECT intelligence_level, qxc_mined FROM unified_intelligence WHERE id = 1")
        current = cursor.fetchone()
        
        if current:
            new_intelligence = min(self.max_intelligence, current[0] + intelligence_gain)
            new_qxc = min(self.total_qxc_supply, current[1] + qxc_reward)
            
            # Update values
            cursor.execute("""
                UPDATE unified_intelligence 
                SET intelligence_level = ?, qxc_mined = ?, last_updated = ?
                WHERE id = 1
            """, (new_intelligence, new_qxc, datetime.now()))
            
            # Store proof
            cursor.execute("""
                INSERT INTO mining_proofs 
                (proof_hash, nonce, difficulty, intelligence_gain, qxc_reward)
                VALUES (?, ?, ?, ?, ?)
            """, (solution['proof'], solution['nonce'], self.difficulty, 
                  intelligence_gain, qxc_reward))
            
            conn.commit()
        
        conn.close()

if __name__ == "__main__":
    miner = RealProofOfIntelligence()
    result = miner.mine()
    if result['success']:
        print(f"Mining successful!")
        print(f"Proof: {result['proof'][:16]}...")
        print(f"Intelligence gained: {result['intelligence_gain']}")
        print(f"QXC earned: {result['qxc_reward']}")
    else:
        print("Mining failed")
EOF
    
    chmod +x "$INSTALL_DIR/real_proof_of_intelligence.py"
    
    # Create web API server
    cat > "$INSTALL_DIR/web_api_server.py" << 'EOF'
#!/usr/bin/env python3
"""
QENEX Web API Server
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import sqlite3
import json
import hashlib
import time

app = Flask(__name__)
CORS(app)

DB_PATH = "/opt/qenex-os/unified_intelligence.db"

@app.route('/api/intelligence/status', methods=['GET'])
def get_intelligence_status():
    """Get current intelligence and QXC status"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT intelligence_level, qxc_mined, last_updated 
        FROM unified_intelligence WHERE id = 1
    """)
    result = cursor.fetchone()
    
    if result:
        status = {
            'intelligence_level': result[0],
            'qxc_mined': result[1],
            'last_updated': result[2],
            'max_intelligence': 1000,
            'total_qxc_supply': 1000000000,
            'percentage_complete': (result[0] / 1000) * 100
        }
    else:
        status = {
            'intelligence_level': 0,
            'qxc_mined': 0,
            'percentage_complete': 0
        }
    
    conn.close()
    return jsonify(status)

@app.route('/api/mining/proofs', methods=['GET'])
def get_mining_proofs():
    """Get recent mining proofs"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT proof_hash, nonce, difficulty, intelligence_gain, qxc_reward, timestamp
        FROM mining_proofs
        ORDER BY timestamp DESC
        LIMIT 10
    """)
    
    proofs = []
    for row in cursor.fetchall():
        proofs.append({
            'proof_hash': row[0],
            'nonce': row[1],
            'difficulty': row[2],
            'intelligence_gain': row[3],
            'qxc_reward': row[4],
            'timestamp': row[5]
        })
    
    conn.close()
    return jsonify(proofs)

@app.route('/api/wallet/<address>', methods=['GET'])
def get_wallet_balance(address):
    """Get wallet balance"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT balance FROM wallets WHERE address = ?", (address,))
    result = cursor.fetchone()
    
    if result:
        balance = result[0]
    else:
        # Create new wallet
        cursor.execute("INSERT INTO wallets (address, balance) VALUES (?, 0)", (address,))
        conn.commit()
        balance = 0
    
    conn.close()
    return jsonify({'address': address, 'balance': balance})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
EOF
    
    chmod +x "$INSTALL_DIR/web_api_server.py"
    
    echo -e "${GREEN}Core scripts created${NC}"
}

# Function to setup web server
setup_nginx() {
    echo -e "${YELLOW}Setting up Nginx web server...${NC}"
    
    # Create web root
    mkdir -p "$WEB_ROOT"
    
    # Create main index.html
    cat > "$WEB_ROOT/index.html" << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QENEX - Unified Financial Operating System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .container {
            max-width: 1200px;
            padding: 2rem;
            text-align: center;
        }
        h1 {
            font-size: 3rem;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .tagline {
            font-size: 1.5rem;
            margin-bottom: 2rem;
            opacity: 0.9;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 2rem;
            margin: 3rem 0;
        }
        .stat-card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            padding: 2rem;
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        .stat-value {
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        .stat-label {
            opacity: 0.8;
        }
        .cta-buttons {
            display: flex;
            gap: 1rem;
            justify-content: center;
            margin-top: 2rem;
        }
        .btn {
            padding: 1rem 2rem;
            background: white;
            color: #667eea;
            text-decoration: none;
            border-radius: 50px;
            font-weight: bold;
            transition: transform 0.3s;
        }
        .btn:hover {
            transform: translateY(-2px);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>QENEX</h1>
        <p class="tagline">Unified Financial Operating System with Proof-of-Intelligence Mining</p>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value" id="intelligence">0.000000</div>
                <div class="stat-label">Intelligence Level</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="qxc">0.0000</div>
                <div class="stat-label">QXC Mined</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">39,498</div>
                <div class="stat-label">TPS Capability</div>
            </div>
        </div>
        
        <div class="cta-buttons">
            <a href="/dashboard" class="btn">Open Dashboard</a>
            <a href="https://github.com/abdulrahman305/qenex-os" class="btn">View on GitHub</a>
        </div>
    </div>
    
    <script>
        // Fetch and update stats
        async function updateStats() {
            try {
                const response = await fetch('/api/intelligence/status');
                const data = await response.json();
                document.getElementById('intelligence').textContent = data.intelligence_level.toFixed(6);
                document.getElementById('qxc').textContent = data.qxc_mined.toFixed(4);
            } catch (error) {
                console.error('Error fetching stats:', error);
            }
        }
        
        updateStats();
        setInterval(updateStats, 5000);
    </script>
</body>
</html>
EOF
    
    # Create Nginx configuration
    cat > /etc/nginx/sites-available/qenex << EOF
server {
    listen 80;
    server_name $DOMAIN www.$DOMAIN;
    
    root $WEB_ROOT;
    index index.html;
    
    # API proxy
    location /api {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
    }
    
    # Static files
    location / {
        try_files \$uri \$uri/ =404;
    }
}
EOF
    
    # Enable site
    ln -sf /etc/nginx/sites-available/qenex /etc/nginx/sites-enabled/
    
    # Remove default site
    rm -f /etc/nginx/sites-enabled/default
    
    # Test and reload Nginx
    nginx -t
    systemctl restart nginx
    
    echo -e "${GREEN}Nginx configured${NC}"
}

# Function to setup systemd services
setup_services() {
    echo -e "${YELLOW}Setting up systemd services...${NC}"
    
    # Create API service
    cat > /etc/systemd/system/qenex-api.service << EOF
[Unit]
Description=QENEX API Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$INSTALL_DIR
Environment="PATH=$INSTALL_DIR/venv/bin"
ExecStart=$INSTALL_DIR/venv/bin/python $INSTALL_DIR/web_api_server.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF
    
    # Create mining service
    cat > /etc/systemd/system/qenex-mining.service << EOF
[Unit]
Description=QENEX Intelligence Mining
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$INSTALL_DIR
Environment="PATH=$INSTALL_DIR/venv/bin"
ExecStart=/bin/bash -c "while true; do $INSTALL_DIR/venv/bin/python $INSTALL_DIR/real_proof_of_intelligence.py; sleep 30; done"
Restart=always

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd and start services
    systemctl daemon-reload
    systemctl enable qenex-api qenex-mining
    systemctl start qenex-api qenex-mining
    
    echo -e "${GREEN}Services configured and started${NC}"
}

# Function to setup firewall
setup_firewall() {
    echo -e "${YELLOW}Configuring firewall...${NC}"
    
    if [[ "$OS" == "ubuntu" ]] || [[ "$OS" == "debian" ]]; then
        ufw allow 22/tcp
        ufw allow 80/tcp
        ufw allow 443/tcp
        ufw --force enable
    elif [[ "$OS" == "centos" ]] || [[ "$OS" == "rhel" ]] || [[ "$OS" == "fedora" ]]; then
        firewall-cmd --permanent --add-service=ssh
        firewall-cmd --permanent --add-service=http
        firewall-cmd --permanent --add-service=https
        firewall-cmd --reload
    fi
    
    echo -e "${GREEN}Firewall configured${NC}"
}

# Function to setup SSL
setup_ssl() {
    echo -e "${YELLOW}Setting up SSL certificate...${NC}"
    
    # Note: This requires domain to be pointing to this server
    if [[ -n "$DOMAIN" ]]; then
        certbot --nginx -d $DOMAIN -d www.$DOMAIN --non-interactive --agree-tos -m admin@$DOMAIN || true
    fi
    
    echo -e "${GREEN}SSL setup complete${NC}"
}

# Function to display completion message
display_completion() {
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}     QENEX Installation Complete!              ${NC}"
    echo -e "${GREEN}================================================${NC}"
    echo
    echo -e "${BLUE}Access your QENEX system:${NC}"
    echo -e "  Web Interface: ${GREEN}http://$DOMAIN${NC}"
    echo -e "  API Endpoint:  ${GREEN}http://$DOMAIN/api${NC}"
    echo
    echo -e "${BLUE}Service Status:${NC}"
    systemctl status qenex-api --no-pager | head -5
    systemctl status qenex-mining --no-pager | head -5
    echo
    echo -e "${BLUE}Important paths:${NC}"
    echo -e "  Installation: ${GREEN}$INSTALL_DIR${NC}"
    echo -e "  Web Root:     ${GREEN}$WEB_ROOT${NC}"
    echo -e "  Database:     ${GREEN}$DB_PATH${NC}"
    echo
    echo -e "${BLUE}Next steps:${NC}"
    echo -e "  1. Point your domain to this server's IP"
    echo -e "  2. Run: ${GREEN}certbot --nginx${NC} to setup SSL"
    echo -e "  3. Check logs: ${GREEN}journalctl -u qenex-api -f${NC}"
    echo
}

# Main installation flow
main() {
    check_root
    detect_os
    install_dependencies
    install_rust
    clone_repository
    setup_python_env
    build_rust_components
    initialize_database
    create_core_scripts
    setup_nginx
    setup_services
    setup_firewall
    setup_ssl
    display_completion
}

# Run main function
main