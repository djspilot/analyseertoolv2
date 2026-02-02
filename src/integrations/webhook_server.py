"""
Webhook Server for Real-Time Integrations
Receives webhooks from Toggl, Health Auto Export, etc.

Run separately: python -m src.integrations.webhook_server
Or integrate into your Reflex app.
"""

import os
import json
import hmac
import hashlib
from datetime import datetime
from typing import Optional

from ..logger import setup_logger
from ..api.database import insert_activities
from .toggl import parse_webhook_payload as parse_toggl
from .apple_health import parse_health_auto_export_payload

logger = setup_logger(__name__)

# Webhook secrets for verification
TOGGL_WEBHOOK_SECRET = os.environ.get("TOGGL_WEBHOOK_SECRET", "")
HEALTH_WEBHOOK_SECRET = os.environ.get("HEALTH_WEBHOOK_SECRET", "")


def verify_toggl_signature(payload: bytes, signature: str) -> bool:
    """Verify Toggl webhook signature."""
    if not TOGGL_WEBHOOK_SECRET:
        logger.warning("TOGGL_WEBHOOK_SECRET not set, skipping verification")
        return True
    
    expected = hmac.new(
        TOGGL_WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(signature, expected)


async def handle_toggl_webhook(payload: dict, signature: Optional[str] = None) -> dict:
    """
    Handle incoming Toggl webhook.
    
    Args:
        payload: Parsed JSON payload
        signature: X-Webhook-Signature header value
        
    Returns:
        Response dict
    """
    logger.info(f"Received Toggl webhook: {payload.get('event_type')}")
    
    result = parse_toggl(payload)
    
    if result:
        if result["action"] == "upsert":
            # Insert or update entry
            entry = result["entry"]
            try:
                # Convert to DataFrame-compatible format
                import pandas as pd
                df = pd.DataFrame([entry])
                df['datetime_from'] = pd.to_datetime(df['datetime_from']).dt.strftime('%Y-%m-%d %H:%M:%S')
                df['datetime_to'] = pd.to_datetime(df['datetime_to']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Add required columns
                df['is_deep_work'] = df['activity_type'].isin(['Work', 'Coding']).astype(int)
                df['fragmentation_risk'] = 0
                
                count = insert_activities(df)
                return {"status": "ok", "inserted": count}
            except Exception as e:
                logger.error(f"Failed to insert Toggl entry: {e}")
                return {"status": "error", "message": str(e)}
        
        elif result["action"] == "delete":
            # Could implement deletion if needed
            return {"status": "ok", "action": "delete_ignored"}
    
    return {"status": "ignored"}


async def handle_health_webhook(payload: dict) -> dict:
    """
    Handle incoming Health Auto Export webhook.
    
    Args:
        payload: Parsed JSON payload from Health Auto Export app
        
    Returns:
        Response dict
    """
    logger.info("Received Health Auto Export webhook")
    
    entries = parse_health_auto_export_payload(payload)
    
    if entries:
        try:
            import pandas as pd
            df = pd.DataFrame(entries)
            df['datetime_from'] = pd.to_datetime(df['datetime_from']).dt.strftime('%Y-%m-%d %H:%M:%S')
            df['datetime_to'] = pd.to_datetime(df['datetime_to']).dt.strftime('%Y-%m-%d %H:%M:%S')
            df['is_deep_work'] = 0
            df['fragmentation_risk'] = 0
            
            count = insert_activities(df)
            return {"status": "ok", "inserted": count}
        except Exception as e:
            logger.error(f"Failed to insert Health entries: {e}")
            return {"status": "error", "message": str(e)}
    
    return {"status": "no_entries"}


# ============================================================================
# FastAPI server (optional standalone deployment)
# ============================================================================

def create_webhook_app():
    """Create FastAPI app for webhook handling."""
    try:
        from fastapi import FastAPI, Request, HTTPException
        from fastapi.responses import JSONResponse
    except ImportError:
        logger.error("FastAPI not installed. Run: pip install fastapi uvicorn")
        return None
    
    app = FastAPI(title="Analyseertool Webhooks")
    
    @app.post("/webhooks/toggl")
    async def toggl_webhook(request: Request):
        signature = request.headers.get("X-Webhook-Signature", "")
        body = await request.body()
        
        if not verify_toggl_signature(body, signature):
            raise HTTPException(status_code=401, detail="Invalid signature")
        
        payload = json.loads(body)
        result = await handle_toggl_webhook(payload, signature)
        return JSONResponse(result)
    
    @app.post("/webhooks/health")
    async def health_webhook(request: Request):
        payload = await request.json()
        result = await handle_health_webhook(payload)
        return JSONResponse(result)
    
    @app.get("/health")
    async def health_check():
        return {"status": "ok", "timestamp": datetime.now().isoformat()}
    
    return app


# ============================================================================
# Reflex API route integration
# ============================================================================

def get_reflex_api_routes():
    """
    Get API route handlers for Reflex.
    
    Usage in rxconfig.py:
        from src.integrations.webhook_server import get_reflex_api_routes
        
        config = rx.Config(
            app_name="app",
            api_routes=get_reflex_api_routes(),
        )
    """
    import reflex as rx
    
    async def toggl_route(request):
        body = await request.body()
        payload = json.loads(body)
        signature = request.headers.get("X-Webhook-Signature", "")
        result = await handle_toggl_webhook(payload, signature)
        return rx.Response(json.dumps(result), media_type="application/json")
    
    async def health_route(request):
        payload = await request.json()
        result = await handle_health_webhook(payload)
        return rx.Response(json.dumps(result), media_type="application/json")
    
    return [
        ("/api/webhooks/toggl", toggl_route, ["POST"]),
        ("/api/webhooks/health", health_route, ["POST"]),
    ]


if __name__ == "__main__":
    import uvicorn
    
    app = create_webhook_app()
    if app:
        print("Starting webhook server on http://0.0.0.0:8001")
        print("Endpoints:")
        print("  POST /webhooks/toggl  - Toggl webhook")
        print("  POST /webhooks/health - Health Auto Export webhook")
        uvicorn.run(app, host="0.0.0.0", port=8001)
