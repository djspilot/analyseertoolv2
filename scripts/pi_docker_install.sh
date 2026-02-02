#!/bin/bash
# =============================================================================
# Raspberry Pi Docker Install Script for Analyseertool
# =============================================================================
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/djspilot/analyseertoolv2/master/scripts/pi_docker_install.sh | bash
# Or clone the repo and run:
#   ./scripts/pi_docker_install.sh
#
# Environment variables (optional):
#   APP_DIR=/opt/analyseertool
#   REPO_URL=https://github.com/<user>/<repo>.git
#   USE_PROXY=1   # Enable homepage.dev reverse proxy via Caddy

set -euo pipefail

APP_DIR="${APP_DIR:-/opt/analyseertool}"
REPO_URL="${REPO_URL:-}"
USE_PROXY="${USE_PROXY:-0}"

log() { echo "[INFO] $1"; }
warn() { echo "[WARN] $1"; }

install_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    log "Installing Docker..."
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker "$USER"
    warn "Docker installed. You may need to log out and back in for docker group to apply."
  else
    log "Docker already installed."
  fi

  if ! docker compose version >/dev/null 2>&1; then
    log "Installing docker compose plugin..."
    sudo apt-get update
    sudo apt-get install -y docker-compose-plugin
  fi
}

clone_or_update_repo() {
  if [[ -z "$REPO_URL" ]]; then
    if [[ ! -d "$APP_DIR/.git" ]]; then
      warn "REPO_URL not set. Skipping clone."
      warn "Set REPO_URL or run this script inside an existing repo."
      return 1
    fi
    return 0
  fi

  if [[ -d "$APP_DIR/.git" ]]; then
    log "Updating existing repo in $APP_DIR..."
    git -C "$APP_DIR" pull
  else
    log "Cloning repo to $APP_DIR..."
    sudo mkdir -p "$APP_DIR"
    sudo chown "$USER":"$USER" "$APP_DIR"
    git clone "$REPO_URL" "$APP_DIR"
  fi
}

ensure_env() {
  if [[ ! -f "$APP_DIR/.env" ]]; then
    log "Creating .env from .env.example..."
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    warn "Update $APP_DIR/.env with your secrets before first run."
  fi
}

start_stack() {
  cd "$APP_DIR"
  if [[ "$USE_PROXY" == "1" ]]; then
    log "Starting Docker stack with Caddy reverse proxy (homepage.dev)..."
    docker compose -f docker-compose.pi.proxy.yml up -d --build
  else
    log "Starting Docker stack..."
    docker compose -f docker-compose.pi.yml up -d --build
  fi
}

main() {
  install_docker
  clone_or_update_repo || true
  ensure_env
  start_stack
  log "Done."
  log "App: http://<pi-ip>:3000"
  if [[ "$USE_PROXY" == "1" ]]; then
    warn "homepage.dev requires local DNS/hosts mapping to your Pi IP and trusting Caddy's local CA."
  fi
}

main
