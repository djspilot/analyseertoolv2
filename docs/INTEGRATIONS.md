# 🔄 Real-Time Integraties

Deze guide beschrijft hoe je de analyseertool kunt verbinden met externe time tracking apps voor real-time synchronisatie.

## ⚡ Automatische Sync (NIEUW)

De app synchroniseert **automatisch** elke 5 minuten met geconfigureerde integraties. Je hoeft niks te doen!

### Hoe werkt het?

1. **Configureer je API tokens** (zie hieronder)
2. **Start de app** - sync begint automatisch
3. **Check de cloud-indicator** in de header:
   - ☁️ Groen = Verbonden & synchroniserend
   - ☁️ Grijs = Geen integratie geconfigureerd
   - 🔄 Draaiend = Bezig met synchroniseren

### Sync Status

De sync status is zichtbaar in de header van de app:
- Klik op het cloud-icoon om auto-sync aan/uit te zetten
- Hover over het icoon voor status details

### Configuratie

Standaard sync interval is 5 minuten. Dit is geoptimaliseerd voor:
- Minimale API calls (Toggl rate limits)
- Snelle updates na activiteit

---

## Ondersteunde Integraties

| App | Type | Real-time | Apple Watch | Gratis |
|-----|------|-----------|-------------|--------|
| **Toggl Track** | API + Webhooks | ✅ | ✅ | Freemium |
| **Clockify** | API (polling) | ⏱️ | ✅ | ✅ |
| **Apple Health** | Export/Webhook | ⏱️/✅ | ✅ | ✅ |

---

## 🟠 Toggl Track (Aanbevolen)

De beste optie voor real-time tracking met Apple Watch.

### Setup

1. **Maak Toggl account**: https://track.toggl.com

2. **Haal API token op**:
   - Ga naar https://track.toggl.com/profile
   - Scroll naar "API Token"
   - Kopieer de token

3. **Configureer environment variable**:
   ```bash
   export TOGGL_API_TOKEN="your_token_here"
   ```

4. **Optioneel - Webhooks voor real-time**:
   
   Toggl Business/Enterprise heeft webhooks. Voor gratis tier, gebruik polling.
   
   ```bash
   # Webhook secret (voor verificatie)
   export TOGGL_WEBHOOK_SECRET="your_secret"
   ```

### Gebruik

```python
from src.integrations import toggl_sync, toggl_start, toggl_stop, toggl_current

# Sync laatste 7 dagen
entries = await toggl_sync(days=7)

# Start timer
entry_id = await toggl_start("Deep work sessie", "Coding")

# Check huidige timer
current = await toggl_current()
if current:
    print(f"Running: {current.description}")

# Stop timer
completed = await toggl_stop()
```

### Category Mapping

Toggl projects/tags worden automatisch gemapt:

| Toggl Project/Tag | → Categorie |
|-------------------|-------------|
| work, meeting | Work |
| coding, development | Coding |
| sport, gym, fitness | Sport |
| reading, book | Read |
| netflix, youtube | Entertainment |
| yoga, meditation | Yoga |

---

## 🔵 Clockify

Gratis alternatief met goede Apple Watch app.

### Setup

1. **Maak Clockify account**: https://clockify.me

2. **Haal API key op**:
   - Ga naar https://clockify.me/user/settings
   - Genereer API key

3. **Configureer**:
   ```bash
   export CLOCKIFY_API_KEY="your_key_here"
   ```

### Gebruik

```python
from src.integrations import clockify_sync

# Sync laatste 7 dagen
entries = await clockify_sync(days=7)
```

---

## 🍎 Apple Health

Importeer workouts en mindfulness sessies van Apple Watch.

### Optie 1: Handmatige Export

1. Open **Gezondheid** app op iPhone
2. Tik op je profielfoto → **Exporteer alle gezondheidsgegevens**
3. Deel/kopieer `export.zip` naar je computer
4. Importeer:

```python
from src.integrations import import_health_export

entries = import_health_export("path/to/export.zip")
# Of: import_health_export("path/to/export_folder/")
```

### Optie 2: Health Auto Export (Real-time)

De iOS app **Health Auto Export** kan automatisch data naar een webhook sturen.

1. **Installeer** Health Auto Export uit App Store (~€5)

2. **Configureer webhook**:
   - URL: `https://jouw-server.com/api/webhooks/health`
   - Selecteer: Workouts
   - Interval: Realtime of elk uur

3. **Start webhook server**:
   ```bash
   python -m src.integrations.webhook_server
   # Of integreer in Reflex app (zie onder)
   ```

### Workout Type Mapping

| Apple Workout Type | → Categorie |
|-------------------|-------------|
| Running, Cycling, Swimming | Sport |
| Walking, Hiking | Walking |
| Yoga, Pilates | Yoga |
| Strength Training | Sport |
| Mindful Session | Yoga |

---

## 🌐 Webhook Server

Voor real-time sync kun je de webhook server draaien.

### Standalone Server

```bash
# Installeer dependencies
pip install fastapi uvicorn

# Start server
python -m src.integrations.webhook_server
# Draait op http://0.0.0.0:8001
```

Endpoints:
- `POST /webhooks/toggl` - Toggl webhooks
- `POST /webhooks/health` - Health Auto Export webhooks
- `GET /health` - Health check

### Integratie met Reflex

Voeg toe aan `rxconfig.py`:

```python
import reflex as rx
from src.integrations.webhook_server import get_reflex_api_routes

config = rx.Config(
    app_name="app",
    # ... andere config
)
```

---

## 🔧 Automatische Sync

### Cron Job (Polling)

Voor apps zonder webhooks, gebruik een cron job:

```bash
# Elke 15 minuten sync
*/15 * * * * cd /path/to/analyseertoolv2 && python -c "
import asyncio
from src.integrations import sync_all
asyncio.run(sync_all())
"
```

### Systemd Timer

```ini
# /etc/systemd/system/analyseertool-sync.service
[Unit]
Description=Sync time tracking data

[Service]
Type=oneshot
WorkingDirectory=/home/pi/analyseertool
ExecStart=/home/pi/analyseertool/venv/bin/python -c "import asyncio; from src.integrations import sync_all; asyncio.run(sync_all())"

# /etc/systemd/system/analyseertool-sync.timer
[Unit]
Description=Run sync every 15 minutes

[Timer]
OnBootSec=5min
OnUnitActiveSec=15min

[Install]
WantedBy=timers.target
```

```bash
sudo systemctl enable analyseertool-sync.timer
sudo systemctl start analyseertool-sync.timer
```

---

## 🔒 Beveiliging

### Webhook Verificatie

Toggl webhooks hebben een signature header. Stel secret in:

```bash
export TOGGL_WEBHOOK_SECRET="random_secret_string"
```

### HTTPS

Voor productie, gebruik altijd HTTPS. Met nginx:

```nginx
server {
    listen 443 ssl;
    server_name api.jouwdomein.nl;
    
    ssl_certificate /etc/letsencrypt/live/api.jouwdomein.nl/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.jouwdomein.nl/privkey.pem;
    
    location /webhooks/ {
        proxy_pass http://127.0.0.1:8001/webhooks/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 📱 Aanbevolen Apple Watch Workflow

1. **Toggl Track** app op Apple Watch
2. Start/stop timers direct vanaf je pols
3. **Health Auto Export** voor workout data
4. Alles synct automatisch naar de analyseertool

### Toggl Watch Complicaties

- Voeg Toggl complicatie toe aan je watch face
- One-tap start van favoriete timers
- Zie actuele timer direct op je pols
