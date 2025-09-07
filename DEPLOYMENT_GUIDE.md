# QENEX System Deployment Guide

## Quick Start (5 minutes)

For a quick test deployment on any Linux server:

```bash
# Download and run quick deploy script
wget https://raw.githubusercontent.com/abdulrahman305/qenex-os/main/quick_deploy_qenex.sh
sudo bash quick_deploy_qenex.sh
```

This will set up a basic QENEX system with:
- Web interface on port 80
- API server on port 5000
- SQLite database with initial data
- Nginx reverse proxy

## Full Production Deployment (30 minutes)

For a complete production deployment with all features:

```bash
# Download full deployment script
wget https://raw.githubusercontent.com/abdulrahman305/qenex-os/main/deploy_qenex_new_server.sh

# Option 1: Run with public repository (recommended)
sudo bash deploy_qenex_new_server.sh

# Option 2: If you have a GitHub token for private access
sudo GITHUB_TOKEN=your_token_here bash deploy_qenex_new_server.sh
```

This includes:
- Complete Python and Rust environment
- All dependencies and libraries
- Systemd services for auto-start
- SSL certificate setup
- Firewall configuration
- Full mining system
- Production-grade security

## System Requirements

### Minimum (Testing)
- 1 CPU core
- 2 GB RAM
- 10 GB disk space
- Ubuntu 20.04+ or CentOS 8+

### Recommended (Production)
- 4+ CPU cores
- 8+ GB RAM
- 50+ GB SSD storage
- Ubuntu 22.04 LTS
- Domain name pointed to server

## Manual Installation Steps

### 1. Install Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3 python3-pip python3-venv nginx sqlite3 git build-essential

# CentOS/RHEL
sudo yum update -y
sudo yum install -y python3 python3-pip nginx sqlite git gcc gcc-c++ make
```

### 2. Clone Repository

```bash
cd /opt
sudo git clone https://github.com/abdulrahman305/qenex-os.git
cd qenex-os
```

### 3. Setup Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Initialize Database

```bash
sqlite3 unified_intelligence.db < schema.sql
```

### 5. Configure Nginx

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        root /var/www/qenex;
        index index.html;
    }
    
    location /api {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
    }
}
```

### 6. Start Services

```bash
# Start API server
python3 web_api_server.py &

# Start mining service
python3 real_proof_of_intelligence.py &

# Or use systemd (recommended)
sudo systemctl start qenex-api
sudo systemctl start qenex-mining
```

## Verification

After deployment, verify the system is running:

### Check Services
```bash
systemctl status qenex-api
systemctl status nginx
```

### Test API
```bash
curl http://localhost/api/intelligence/status
curl http://localhost/api/health
```

### Check Logs
```bash
journalctl -u qenex-api -f
tail -f /var/log/nginx/access.log
```

## Security Considerations

1. **Change default credentials** - Update any default passwords
2. **Configure firewall** - Only allow necessary ports (80, 443, 22)
3. **Setup SSL** - Use Let's Encrypt for HTTPS
4. **Regular updates** - Keep system and dependencies updated
5. **Backup database** - Regular backups of SQLite database

## Troubleshooting

### Port Already in Use
```bash
# Find process using port
sudo lsof -i :5000
# Kill process if needed
sudo kill -9 <PID>
```

### Permission Denied
```bash
# Fix permissions
sudo chown -R $USER:$USER /opt/qenex-os
sudo chmod -R 755 /opt/qenex-os
```

### Service Won't Start
```bash
# Check logs
journalctl -u qenex-api -n 50
# Check Python path
which python3
# Update service file with correct path
```

## Advanced Configuration

### Custom Domain
1. Point your domain to server IP
2. Update Nginx server_name
3. Install SSL certificate:
```bash
sudo certbot --nginx -d your-domain.com
```

### Performance Tuning
```bash
# Increase file limits
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# Optimize Nginx
worker_processes auto;
worker_connections 2048;
```

### Clustering
For high availability, deploy multiple nodes:
1. Setup load balancer (HAProxy/Nginx)
2. Configure shared database (PostgreSQL)
3. Synchronize mining state

## Monitoring

### System Metrics
```bash
# CPU and Memory
htop
# Disk usage
df -h
# Network
netstat -tunlp
```

### Application Metrics
- Intelligence Level: `/api/intelligence/status`
- Mining Proofs: `/api/mining/proofs`
- Wallet Balances: `/api/wallet/<address>`

## Support

- GitHub Issues: https://github.com/abdulrahman305/qenex-os/issues
- Documentation: https://qenex.ai/docs
- Email: support@qenex.ai

## License

MIT License - See LICENSE file for details