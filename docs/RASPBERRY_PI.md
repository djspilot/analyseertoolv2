# =============================================================================
# Raspberry Pi Deployment Guide - Analyseertool
# =============================================================================

## Systeemvereisten

- **Raspberry Pi 4** (2GB RAM minimum, 4GB aanbevolen)
- **Raspberry Pi OS** (64-bit Bookworm aanbevolen)
- **32GB+ SD-kaart** (Class 10 of sneller)
- **Stabiele internetverbinding** (voor Z.ai API)

## Snelle Start

### Optie 1: Native Installatie (Aanbevolen voor Pi 4 2GB)

```bash
# 1. Clone of kopieer de bestanden naar je Pi
scp -r ./analyseertoolv2 pi@raspberrypi:~/analyseertool

# 2. SSH naar je Pi
ssh pi@raspberrypi

# 3. Ga naar de app directory
cd ~/analyseertool

# 4. Maak deploy script uitvoerbaar
chmod +x deploy_pi.sh

# 5. Installeer alles
./deploy_pi.sh install

# 6. Configureer API key
echo 'export Z_AI_API_KEY="jouw-api-key"' >> ~/.bashrc
source ~/.bashrc

# 7. Start de app
./deploy_pi.sh start

# 8. Bekijk logs
./deploy_pi.sh logs
```

### Optie 2: Docker (Aanbevolen voor Pi 4 4GB)

```bash
# 1. Installeer Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker pi
# Log uit en weer in

# 2. Maak .env bestand
echo 'Z_AI_API_KEY=jouw-api-key' > .env

# 3. Start met docker-compose
docker-compose -f docker-compose.pi.yml up -d

# 4. Bekijk logs
docker-compose -f docker-compose.pi.yml logs -f
```

## Configuratie

### Environment Variables

| Variable | Beschrijving | Default |
|----------|--------------|---------|
| `Z_AI_API_KEY` | API key voor Z.ai LLM | *verplicht* |
| `RASPBERRY_PI_MODE` | Activeert Pi optimalisaties | auto-detect |
| `ENVIRONMENT` | `production` of `development` | `development` |

### Geheugen Optimalisatie

De app detecteert automatisch of het op een Raspberry Pi draait en past aan:
- Minder datapunten per grafiek (500 max)
- Animaties uitgeschakeld
- Agressieve garbage collection
- WebGL rendering waar mogelijk

### Port Wijzigen

Standaard draait de app op poort 3000. Om dit te wijzigen:

1. **Native**: Edit `rxconfig.py`
2. **Docker**: Wijzig port mapping in `docker-compose.pi.yml`

## Service Beheer

```bash
# Status bekijken
./deploy_pi.sh status

# Herstarten
./deploy_pi.sh restart

# Stoppen
./deploy_pi.sh stop

# Logs volgen
./deploy_pi.sh logs
```

## Netwerk Toegang

Na starten is de app beschikbaar op:
```
http://<pi-ip-adres>:3000
```

Vind je Pi's IP met:
```bash
hostname -I
```

### Nginx Reverse Proxy (Optioneel)

Voor toegang via poort 80:

```bash
sudo apt install nginx

sudo tee /etc/nginx/sites-available/analyseertool <<EOF
server {
    listen 80;
    server_name _;
    
    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_cache_bypass \$http_upgrade;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/analyseertool /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx
```

## Troubleshooting

### App start niet

1. Check logs: `./deploy_pi.sh logs`
2. Check geheugen: `free -h`
3. Check temperatuur: `vcgencmd measure_temp`

### Langzame performance

1. Verhoog swap space:
   ```bash
   sudo dphys-swapfile swapoff
   sudo nano /etc/dphys-swapfile  # CONF_SWAPSIZE=1024
   sudo dphys-swapfile setup
   sudo dphys-swapfile swapon
   ```

2. Overclock (op eigen risico):
   ```bash
   sudo nano /boot/config.txt
   # Voeg toe:
   # over_voltage=4
   # arm_freq=1800
   ```

### Memory errors

Als je "Out of Memory" errors ziet:
1. Sluit andere processen
2. Verhoog swap (zie boven)
3. Gebruik Docker met memory limits

### Grafieken laden niet

De Pi modus beperkt automatisch het aantal datapunten. Als dit nog steeds traag is:
1. Upload kleinere CSV bestanden
2. Filter op kortere periodes
3. Check browser console voor errors

## Backup

```bash
# Backup data
tar -czf analyseertool-backup.tar.gz data/ uploaded_files/

# Restore
tar -xzf analyseertool-backup.tar.gz
```

## Updates

```bash
# Stop de service
./deploy_pi.sh stop

# Pull/kopieer nieuwe code
git pull  # of scp

# Herstart
./deploy_pi.sh start
```
