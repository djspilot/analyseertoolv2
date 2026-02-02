"""
REST API for external integrations.
Allows access to time tracking data via HTTP endpoints.
Compatible with iOS Shortcuts, Zapier, n8n, etc.
"""

from fastapi import FastAPI, HTTPException, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, date, timedelta
import os
import hashlib
import secrets

from ..api.database import get_all_activities, add_activity, init_db
from ..processor import calculate_metrics
from ..metrics.basic import calculate_basic_metrics
from ..metrics.advanced import calculate_advanced_metrics

# ============================================================================
# API Configuration
# ============================================================================

API_VERSION = "v1"
API_KEY = os.environ.get("TIME_API_KEY", None)  # Set this for security!

# Generate a random API key if not set (for development)
if not API_KEY:
    API_KEY = secrets.token_urlsafe(32)
    print(f"⚠️  No TIME_API_KEY set. Generated temporary key: {API_KEY}")


# ============================================================================
# Models
# ============================================================================

class ActivityCreate(BaseModel):
    """Create a new activity entry."""
    activity: str = Field(..., description="Activity name", example="Deep Work")
    category: str = Field(default="Werk", description="Category", example="Werk")
    start_time: datetime = Field(..., description="Start time (ISO format)")
    end_time: Optional[datetime] = Field(None, description="End time (ISO format, optional for running timer)")
    notes: Optional[str] = Field(None, description="Optional notes")

class ActivityResponse(BaseModel):
    """Activity response."""
    id: int
    activity: str
    category: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_hours: float
    is_deep_work: bool

class MetricsResponse(BaseModel):
    """Daily/weekly metrics summary."""
    period: str
    total_hours: float
    deep_work_hours: float
    deep_work_ratio: float
    flow_index: float
    productivity_pulse: float
    context_switches: int
    top_activities: List[dict]

class TimerRequest(BaseModel):
    """Start/stop timer request."""
    activity: str
    category: str = "Werk"

class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    version: str
    timestamp: datetime


# ============================================================================
# API Security
# ============================================================================

async def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
    """Verify the API key from header."""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    
    # Constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    return x_api_key


# ============================================================================
# FastAPI App
# ============================================================================

api = FastAPI(
    title="Time Tracking API",
    description="REST API for the Time Analysis Dashboard",
    version=API_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS for external access
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Endpoints
# ============================================================================

@api.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check if API is running."""
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        timestamp=datetime.now()
    )


@api.get("/api/v1/activities", tags=["Activities"])
async def get_activities(
    start_date: Optional[date] = Query(None, description="Filter from date"),
    end_date: Optional[date] = Query(None, description="Filter to date"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(100, le=1000, description="Max results"),
    _: str = Depends(verify_api_key),
):
    """
    Get activity entries with optional filters.
    
    Use this endpoint to retrieve your time tracking data for external tools.
    """
    init_db()
    df = get_all_activities()
    
    if df.empty:
        return {"activities": [], "count": 0}
    
    # Apply filters
    if start_date:
        df = df[df["datetime_from"].dt.date >= start_date]
    if end_date:
        df = df[df["datetime_from"].dt.date <= end_date]
    if category:
        df = df[df["category"].str.lower() == category.lower()]
    
    # Limit results
    df = df.head(limit)
    
    activities = []
    for _, row in df.iterrows():
        activities.append({
            "activity": row["activity"],
            "category": row.get("category", ""),
            "start_time": row["datetime_from"].isoformat(),
            "end_time": row["datetime_to"].isoformat() if "datetime_to" in row else None,
            "duration_hours": round(row.get("duration_hours", 0), 2),
            "is_deep_work": bool(row.get("is_deep_work", False)),
        })
    
    return {"activities": activities, "count": len(activities)}


@api.post("/api/v1/activities", tags=["Activities"])
async def create_activity(
    activity: ActivityCreate,
    _: str = Depends(verify_api_key),
):
    """
    Create a new activity entry.
    
    Use this from iOS Shortcuts, Zapier, or other automation tools.
    """
    init_db()
    
    try:
        # Calculate duration if end time provided
        if activity.end_time:
            duration = (activity.end_time - activity.start_time).total_seconds() / 3600
        else:
            duration = 0
        
        # Add to database
        add_activity(
            activity=activity.activity,
            category=activity.category,
            datetime_from=activity.start_time,
            datetime_to=activity.end_time or activity.start_time,
            duration_hours=duration,
        )
        
        return {
            "success": True,
            "message": f"Activity '{activity.activity}' created",
            "duration_hours": round(duration, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/api/v1/metrics/today", response_model=MetricsResponse, tags=["Metrics"])
async def get_today_metrics(_: str = Depends(verify_api_key)):
    """Get metrics for today."""
    return await _get_metrics_for_period("today")


@api.get("/api/v1/metrics/week", response_model=MetricsResponse, tags=["Metrics"])
async def get_week_metrics(_: str = Depends(verify_api_key)):
    """Get metrics for this week."""
    return await _get_metrics_for_period("week")


@api.get("/api/v1/metrics/month", response_model=MetricsResponse, tags=["Metrics"])
async def get_month_metrics(_: str = Depends(verify_api_key)):
    """Get metrics for this month."""
    return await _get_metrics_for_period("month")


async def _get_metrics_for_period(period: str) -> MetricsResponse:
    """Calculate metrics for a time period."""
    init_db()
    df = get_all_activities()
    
    if df.empty:
        return MetricsResponse(
            period=period,
            total_hours=0,
            deep_work_hours=0,
            deep_work_ratio=0,
            flow_index=0,
            productivity_pulse=0,
            context_switches=0,
            top_activities=[]
        )
    
    df = calculate_metrics(df)
    
    # Filter by period
    today = datetime.now().date()
    if period == "today":
        df = df[df["datetime_from"].dt.date == today]
    elif period == "week":
        week_start = today - timedelta(days=today.weekday())
        df = df[df["datetime_from"].dt.date >= week_start]
    elif period == "month":
        month_start = today.replace(day=1)
        df = df[df["datetime_from"].dt.date >= month_start]
    
    if df.empty:
        return MetricsResponse(
            period=period,
            total_hours=0,
            deep_work_hours=0,
            deep_work_ratio=0,
            flow_index=0,
            productivity_pulse=0,
            context_switches=0,
            top_activities=[]
        )
    
    # Calculate metrics
    basic = calculate_basic_metrics(df)
    advanced = calculate_advanced_metrics(df)
    
    # Top activities
    top = df.groupby("activity")["duration_hours"].sum().nlargest(5)
    top_activities = [{"activity": k, "hours": round(v, 2)} for k, v in top.items()]
    
    return MetricsResponse(
        period=period,
        total_hours=round(df["duration_hours"].sum(), 2),
        deep_work_hours=round(df[df["is_deep_work"] == 1]["duration_hours"].sum(), 2),
        deep_work_ratio=advanced.get("deep_work_ratio", 0),
        flow_index=advanced.get("flow_index", 0),
        productivity_pulse=basic.get("productivity_pulse", 0),
        context_switches=int(basic.get("context_switches", 0)),
        top_activities=top_activities
    )


# ============================================================================
# Timer Endpoints (for quick start/stop)
# ============================================================================

# In-memory running timer (for simplicity - could use Redis for production)
_running_timer: Optional[dict] = None


@api.post("/api/v1/timer/start", tags=["Timer"])
async def start_timer(
    request: TimerRequest,
    _: str = Depends(verify_api_key),
):
    """
    Start a timer for an activity.
    
    Perfect for iOS Shortcuts or automation triggers.
    """
    global _running_timer
    
    if _running_timer:
        return {
            "success": False,
            "message": f"Timer already running: {_running_timer['activity']}",
            "running_since": _running_timer["start_time"].isoformat()
        }
    
    _running_timer = {
        "activity": request.activity,
        "category": request.category,
        "start_time": datetime.now()
    }
    
    return {
        "success": True,
        "message": f"Timer started: {request.activity}",
        "start_time": _running_timer["start_time"].isoformat()
    }


@api.post("/api/v1/timer/stop", tags=["Timer"])
async def stop_timer(_: str = Depends(verify_api_key)):
    """
    Stop the running timer and save the activity.
    """
    global _running_timer
    
    if not _running_timer:
        return {"success": False, "message": "No timer running"}
    
    end_time = datetime.now()
    duration = (end_time - _running_timer["start_time"]).total_seconds() / 3600
    
    # Save to database
    init_db()
    add_activity(
        activity=_running_timer["activity"],
        category=_running_timer["category"],
        datetime_from=_running_timer["start_time"],
        datetime_to=end_time,
        duration_hours=duration,
    )
    
    result = {
        "success": True,
        "message": f"Timer stopped: {_running_timer['activity']}",
        "duration_hours": round(duration, 2),
        "duration_minutes": round(duration * 60, 1)
    }
    
    _running_timer = None
    return result


@api.get("/api/v1/timer/status", tags=["Timer"])
async def timer_status(_: str = Depends(verify_api_key)):
    """Get current timer status."""
    global _running_timer
    
    if not _running_timer:
        return {"running": False}
    
    elapsed = (datetime.now() - _running_timer["start_time"]).total_seconds()
    
    return {
        "running": True,
        "activity": _running_timer["activity"],
        "category": _running_timer["category"],
        "start_time": _running_timer["start_time"].isoformat(),
        "elapsed_minutes": round(elapsed / 60, 1),
        "elapsed_hours": round(elapsed / 3600, 2)
    }


# ============================================================================
# Shortcuts/Automation Helpers
# ============================================================================

@api.get("/api/v1/categories", tags=["Helpers"])
async def get_categories(_: str = Depends(verify_api_key)):
    """Get all available categories."""
    init_db()
    df = get_all_activities()
    
    if df.empty:
        return {"categories": ["Werk", "Sport", "Leren", "Entertainment", "Slaap"]}
    
    categories = df["category"].dropna().unique().tolist()
    return {"categories": sorted(categories)}


@api.get("/api/v1/recent-activities", tags=["Helpers"])
async def get_recent_activities(
    limit: int = Query(10, le=50),
    _: str = Depends(verify_api_key)
):
    """Get recently used activity names (for autocomplete)."""
    init_db()
    df = get_all_activities()
    
    if df.empty:
        return {"activities": []}
    
    # Get most recent unique activities
    recent = df.sort_values("datetime_from", ascending=False)
    unique_activities = recent["activity"].drop_duplicates().head(limit).tolist()
    
    return {"activities": unique_activities}


# ============================================================================
# Mount to main app or run standalone
# ============================================================================

def get_api_app() -> FastAPI:
    """Get the FastAPI app for mounting."""
    return api


def print_api_info():
    """Print API access information."""
    print("\n" + "="*60)
    print("🔌 REST API ENABLED")
    print("="*60)
    print(f"📍 Docs: http://localhost:3000/api/docs")
    print(f"🔑 API Key: {API_KEY}")
    print("\n📱 iOS Shortcut Example:")
    print(f'   curl -X POST "http://YOUR_PI:3000/api/v1/timer/start" \\')
    print(f'        -H "X-API-Key: {API_KEY}" \\')
    print(f'        -H "Content-Type: application/json" \\')
    print(f'        -d \'{{"activity": "Deep Work", "category": "Werk"}}\'')
    print("="*60 + "\n")
