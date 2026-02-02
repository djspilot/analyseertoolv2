# Raspberry Pi Deployment

This project supports two deployment paths on Raspberry Pi:

- **Docker (recommended)**: fast, clean, and easy to update.
- **Native (systemd)**: uses the existing `deploy_pi.sh` script.

## Option A — Docker (recommended)

### 1) Install via script

From your Pi, run:

```bash
curl -fsSL https://raw.githubusercontent.com/<user>/<repo>/main/scripts/pi_docker_install.sh | bash
```

Or clone and run locally:

```bash
git clone https://github.com/<user>/<repo>.git
cd analyseertoolv2
./scripts/pi_docker_install.sh
```

The script will:
- install Docker and the Compose plugin
- clone/update the repo (if `REPO_URL` is set)
- create `.env` from `.env.example`
- build and start the container

App URL (default):

```
http://<pi-ip>:3000
```

### 2) Use homepage.dev (optional)

To expose the app on https://homepage.dev using a local reverse proxy:

```bash
USE_PROXY=1 ./scripts/pi_docker_install.sh
```

This starts Caddy with an internal TLS certificate.

**You must map homepage.dev to your Pi’s IP** using your router DNS or local hosts file.

**On each client device**, trust Caddy’s local CA:

1. Copy the root cert from the Pi:
   - Container stores it in the caddy data volume
   - Path: `/data/caddy/pki/authorities/local/root.crt`
2. Install it as a trusted certificate on your device/browser.

> Note: `.dev` is HSTS preloaded, so HTTPS is required. Caddy’s `tls internal` handles this.

## Option B — Native install (systemd)

Use the existing script:

```bash
./deploy_pi.sh install
./deploy_pi.sh start
```

App URL:

```
http://<pi-ip>:3000
```
