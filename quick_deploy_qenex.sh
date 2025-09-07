#!/bin/bash

################################################################################
# QENEX Quick Deployment Script - Simplified Version
# For testing on any Linux server
################################################################################

set -e

echo "======================================"
echo "QENEX Quick Deploy - Starting"
echo "======================================"

# Quick check for root
if [[ $EUID -ne 0 ]]; then
   echo "Please run as root: sudo bash $0"
   exit 1
fi

# Install minimal dependencies
echo "Installing dependencies..."
apt-get update || yum update -y
apt-get install -y python3 python3-pip git nginx sqlite3 || \
yum install -y python3 python3-pip git nginx sqlite

# Clone repository
echo "Cloning QENEX repository..."
cd /opt
rm -rf qenex-os
git clone https://github.com/abdulrahman305/qenex-os.git

# Setup Python environment
cd /opt/qenex-os
pip3 install Flask flask-cors gunicorn

# Initialize database
echo "Setting up database..."
sqlite3 /opt/qenex-os/unified_intelligence.db << 'SQL'
CREATE TABLE IF NOT EXISTS unified_intelligence (
    id INTEGER PRIMARY KEY,
    intelligence_level REAL DEFAULT 0.160177,
    qxc_mined REAL DEFAULT 1581.594,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT OR IGNORE INTO unified_intelligence (id) VALUES (1);

CREATE TABLE IF NOT EXISTS mining_proofs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    proof_hash TEXT,
    nonce INTEGER,
    difficulty INTEGER DEFAULT 4,
    intelligence_gain REAL,
    qxc_reward REAL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS wallets (
    address TEXT PRIMARY KEY,
    balance REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
SQL

# Create simple API server
cat > /opt/qenex-os/api_server.py << 'PYTHON'
from flask import Flask, jsonify
from flask_cors import CORS
import sqlite3

app = Flask(__name__)
CORS(app)

@app.route('/api/intelligence/status')
def status():
    conn = sqlite3.connect('/opt/qenex-os/unified_intelligence.db')
    cursor = conn.cursor()
    cursor.execute("SELECT intelligence_level, qxc_mined FROM unified_intelligence WHERE id=1")
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return jsonify({
            'intelligence_level': result[0],
            'qxc_mined': result[1],
            'max_intelligence': 1000,
            'total_supply': 1000000000
        })
    return jsonify({'error': 'No data'})

@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy', 'service': 'QENEX OS'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
PYTHON

# Create web interface
mkdir -p /var/www/html
cat > /var/www/html/index.html << 'HTML'
<!DOCTYPE html>
<html>
<head>
    <title>QENEX OS</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
        }
        .container {
            text-align: center;
            padding: 2rem;
            background: rgba(255,255,255,0.1);
            border-radius: 20px;
            backdrop-filter: blur(10px);
        }
        h1 { font-size: 3rem; margin-bottom: 1rem; }
        .stats { font-size: 1.5rem; margin: 1rem 0; }
        .value { font-weight: bold; color: #ffd700; }
    </style>
</head>
<body>
    <div class="container">
        <h1>QENEX Financial OS</h1>
        <p>Proof-of-Intelligence Mining System</p>
        <div class="stats">
            <p>Intelligence: <span class="value" id="intelligence">Loading...</span> / 1000</p>
            <p>QXC Mined: <span class="value" id="qxc">Loading...</span></p>
        </div>
    </div>
    <script>
        async function updateStats() {
            try {
                const res = await fetch('/api/intelligence/status');
                const data = await res.json();
                document.getElementById('intelligence').textContent = data.intelligence_level.toFixed(6);
                document.getElementById('qxc').textContent = data.qxc_mined.toFixed(4);
            } catch(e) {
                console.error(e);
            }
        }
        updateStats();
        setInterval(updateStats, 5000);
    </script>
</body>
</html>
HTML

# Configure Nginx
cat > /etc/nginx/sites-available/qenex << 'NGINX'
server {
    listen 80;
    server_name _;
    
    root /var/www/html;
    index index.html;
    
    location /api {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location / {
        try_files $uri $uri/ =404;
    }
}
NGINX

ln -sf /etc/nginx/sites-available/qenex /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl restart nginx

# Create systemd service
cat > /etc/systemd/system/qenex-api.service << 'SERVICE'
[Unit]
Description=QENEX API Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/qenex-os
ExecStart=/usr/bin/python3 /opt/qenex-os/api_server.py
Restart=always

[Install]
WantedBy=multi-user.target
SERVICE

# Start services
systemctl daemon-reload
systemctl enable qenex-api
systemctl start qenex-api

# Get server IP
SERVER_IP=$(hostname -I | awk '{print $1}')

echo "======================================"
echo "QENEX Deployment Complete!"
echo "======================================"
echo ""
echo "Access your QENEX system at:"
echo "  http://$SERVER_IP"
echo ""
echo "API endpoints:"
echo "  http://$SERVER_IP/api/intelligence/status"
echo "  http://$SERVER_IP/api/health"
echo ""
echo "Check service status:"
echo "  systemctl status qenex-api"
echo "  systemctl status nginx"
echo ""
echo "View logs:"
echo "  journalctl -u qenex-api -f"
echo ""
echo "======================================"