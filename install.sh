#!/bin/bash
#
# QENEX OS Installation Script
# Installs QENEX OS and all dependencies
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print colored output
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

# Check system requirements
check_requirements() {
    print_info "Checking system requirements..."
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    required_version="3.8"
    
    if [[ $(echo "$python_version < $required_version" | bc) -eq 1 ]]; then
        print_error "Python $required_version or higher is required (found $python_version)"
        exit 1
    fi
    
    print_success "Python $python_version found"
    
    # Check available disk space (need at least 1GB)
    available_space=$(df / | awk 'NR==2 {print $4}')
    if [[ $available_space -lt 1048576 ]]; then
        print_error "Insufficient disk space (need at least 1GB)"
        exit 1
    fi
    
    print_success "Sufficient disk space available"
    
    # Check memory (need at least 2GB)
    total_memory=$(free -m | awk 'NR==2 {print $2}')
    if [[ $total_memory -lt 2048 ]]; then
        print_error "Insufficient memory (need at least 2GB)"
        exit 1
    fi
    
    print_success "Sufficient memory available"
}

# Install Python dependencies
install_dependencies() {
    print_info "Installing Python dependencies..."
    
    # Upgrade pip
    python3 -m pip install --upgrade pip
    
    # Install required packages
    python3 -m pip install \
        aiohttp \
        asyncio \
        cryptography \
        numpy \
        psutil \
        pyyaml \
        requests \
        web3 \
        --quiet
    
    print_success "Dependencies installed"
}

# Create system directories
create_directories() {
    print_info "Creating system directories..."
    
    directories=(
        "/opt/qenex-os"
        "/opt/qenex-os/core"
        "/opt/qenex-os/data"
        "/opt/qenex-os/logs"
        "/opt/qenex-os/config"
        "/var/qenex"
        "/var/qenex/cache"
        "/var/qenex/tmp"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        chmod 755 "$dir"
    done
    
    print_success "Directories created"
}

# Copy files to installation directory
install_files() {
    print_info "Installing QENEX OS files..."
    
    # Get the directory of this script
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    
    # Copy core files
    cp -r "$SCRIPT_DIR/core" "/opt/qenex-os/"
    
    # Copy main executable
    cp "$SCRIPT_DIR/qenex-os" "/opt/qenex-os/"
    chmod +x "/opt/qenex-os/qenex-os"
    
    # Create symlink in /usr/local/bin
    ln -sf "/opt/qenex-os/qenex-os" "/usr/local/bin/qenex-os"
    
    print_success "Files installed"
}

# Create default configuration
create_config() {
    print_info "Creating default configuration..."
    
    cat > "/opt/qenex-os/config/system.json" << EOF
{
    "version": "1.0.0",
    "ai": {
        "enabled": true,
        "auto_optimize": true,
        "learning_rate": 0.01
    },
    "security": {
        "level": "maximum",
        "firewall": true,
        "intrusion_detection": true
    },
    "network": {
        "blockchain_sync": true,
        "defi_integration": true,
        "p2p_enabled": true
    },
    "performance": {
        "cpu_governor": "balanced",
        "memory_optimization": "aggressive"
    }
}
EOF
    
    print_success "Configuration created"
}

# Create systemd service
create_service() {
    print_info "Creating systemd service..."
    
    cat > "/etc/systemd/system/qenex-os.service" << EOF
[Unit]
Description=QENEX OS - Unified AI Operating System
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/qenex-os
ExecStart=/usr/local/bin/qenex-os start
ExecStop=/usr/local/bin/qenex-os stop
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd
    systemctl daemon-reload
    
    print_success "Service created"
}

# Main installation
main() {
    echo "======================================"
    echo "   QENEX OS Installation Script"
    echo "======================================"
    echo ""
    
    check_root
    check_requirements
    install_dependencies
    create_directories
    install_files
    create_config
    create_service
    
    echo ""
    echo "======================================"
    print_success "QENEX OS installed successfully!"
    echo "======================================"
    echo ""
    echo "To get started:"
    echo "  1. Initialize the system: qenex-os init"
    echo "  2. Start QENEX OS: qenex-os start"
    echo "  3. Check status: qenex-os status"
    echo ""
    echo "Or enable auto-start on boot:"
    echo "  sudo systemctl enable qenex-os"
    echo "  sudo systemctl start qenex-os"
    echo ""
}

# Run main installation
main "$@"