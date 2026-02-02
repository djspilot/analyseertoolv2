#!/bin/bash
# =============================================================================
# Raspberry Pi Deployment Script for Analyseertool
# =============================================================================
# Usage: ./deploy_pi.sh [install|start|stop|status|logs]

set -e

APP_NAME="analyseertool"
APP_DIR="/home/pi/analyseertool"
VENV_DIR="$APP_DIR/venv"
SERVICE_FILE="/etc/systemd/system/$APP_NAME.service"
PORT=3000

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# INSTALLATION
# =============================================================================

install_dependencies() {
    log_info "Installing system dependencies..."
    sudo apt-get update
    sudo apt-get install -y \
        python3-pip \
        python3-venv \
        python3-dev \
        libatlas-base-dev \
        libopenblas-dev \
        libjpeg-dev \
        zlib1g-dev \
        libffi-dev \
        libssl-dev \
        nodejs \
        npm
    
    # Install Node.js 18+ for Reflex
    if ! command -v node &> /dev/null || [[ $(node -v | cut -d'v' -f2 | cut -d'.' -f1) -lt 18 ]]; then
        log_info "Installing Node.js 18..."
        curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
        sudo apt-get install -y nodejs
    fi
}

setup_venv() {
    log_info "Setting up Python virtual environment..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip wheel setuptools
}

install_python_deps() {
    log_info "Installing Python dependencies (this may take a while on Pi)..."
    source "$VENV_DIR/bin/activate"
    
    # Install numpy first (needed by pandas)
    pip install numpy --prefer-binary
    
    # Install remaining dependencies
    pip install -r "$APP_DIR/requirements.txt" --prefer-binary
    
    # Optional: Install psutil for memory monitoring
    pip install psutil
}

create_service() {
    log_info "Creating systemd service..."
    
    sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=Analyseertool Time Tracking Dashboard
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=$APP_DIR
Environment="PATH=$VENV_DIR/bin"
Environment="ENVIRONMENT=production"
Environment="RASPBERRY_PI_MODE=true"
ExecStart=$VENV_DIR/bin/reflex run --env prod
Restart=always
RestartSec=10

# Resource limits for Raspberry Pi
MemoryMax=512M
CPUQuota=80%

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$APP_NAME

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable "$APP_NAME"
    log_info "Service created and enabled"
}

install() {
    log_info "Starting full installation..."
    install_dependencies
    setup_venv
    install_python_deps
    create_service
    log_info "Installation complete! Run './deploy_pi.sh start' to start the app"
}

# =============================================================================
# SERVICE MANAGEMENT
# =============================================================================

start() {
    log_info "Starting $APP_NAME..."
    sudo systemctl start "$APP_NAME"
    sleep 3
    if systemctl is-active --quiet "$APP_NAME"; then
        log_info "Service started successfully"
        log_info "Access the dashboard at http://$(hostname -I | awk '{print $1}'):$PORT"
    else
        log_error "Failed to start service"
        sudo journalctl -u "$APP_NAME" -n 20 --no-pager
        exit 1
    fi
}

stop() {
    log_info "Stopping $APP_NAME..."
    sudo systemctl stop "$APP_NAME"
    log_info "Service stopped"
}

restart() {
    log_info "Restarting $APP_NAME..."
    sudo systemctl restart "$APP_NAME"
    log_info "Service restarted"
}

status() {
    echo "=== Service Status ==="
    sudo systemctl status "$APP_NAME" --no-pager || true
    echo ""
    echo "=== Memory Usage ==="
    free -h
    echo ""
    echo "=== CPU Temperature ==="
    vcgencmd measure_temp 2>/dev/null || echo "N/A"
}

logs() {
    sudo journalctl -u "$APP_NAME" -f
}

# =============================================================================
# MAIN
# =============================================================================

case "$1" in
    install)
        install
        ;;
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        status
        ;;
    logs)
        logs
        ;;
    *)
        echo "Usage: $0 {install|start|stop|restart|status|logs}"
        exit 1
        ;;
esac
