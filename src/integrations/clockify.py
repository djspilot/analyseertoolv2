"""
Clockify Integration Module
Sync with Clockify time tracking (free alternative to Toggl).

Setup:
1. Get API key from https://clockify.me/user/settings
2. Set environment variable: CLOCKIFY_API_KEY=your_key
"""

import os
import httpx
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass

from ..logger import setup_logger

logger = setup_logger(__name__)

# Configuration
CLOCKIFY_API_KEY = os.environ.get("CLOCKIFY_API_KEY", "")
CLOCKIFY_API_BASE = "https://api.clockify.me/api/v1"

# Category mapping
DEFAULT_CATEGORY_MAP = {
    "work": "Work",
    "coding": "Coding",
    "development": "Coding",
    "sport": "Sport",
    "exercise": "Sport",
    "reading": "Read",
    "entertainment": "Entertainment",
    "housework": "Housework",
    "yoga": "Yoga",
    "walking": "Walking",
    "music": "Music",
}


@dataclass
class ClockifyEntry:
    """A Clockify time entry."""
    id: str
    description: str
    start: datetime
    end: Optional[datetime]
    duration: Optional[str]  # ISO 8601 duration
    project_name: Optional[str]
    tags: list[str]
    workspace_id: str


def _get_headers() -> dict:
    """Get headers for Clockify API."""
    if not CLOCKIFY_API_KEY:
        raise ValueError("CLOCKIFY_API_KEY environment variable not set")
    return {"X-Api-Key": CLOCKIFY_API_KEY}


def _parse_duration(duration_str: Optional[str]) -> float:
    """Parse ISO 8601 duration to hours."""
    if not duration_str:
        return 0.0
    
    # Format: PT1H30M15S
    hours = 0.0
    duration_str = duration_str.replace("PT", "")
    
    if "H" in duration_str:
        h, duration_str = duration_str.split("H")
        hours += float(h)
    if "M" in duration_str:
        m, duration_str = duration_str.split("M")
        hours += float(m) / 60
    if "S" in duration_str:
        s = duration_str.replace("S", "")
        hours += float(s) / 3600
    
    return round(hours, 2)


def _map_to_category(entry: ClockifyEntry) -> str:
    """Map Clockify entry to internal category."""
    # Check project name
    if entry.project_name:
        project_lower = entry.project_name.lower()
        for key, category in DEFAULT_CATEGORY_MAP.items():
            if key in project_lower:
                return category
    
    # Check tags
    for tag in entry.tags:
        tag_lower = tag.lower()
        for key, category in DEFAULT_CATEGORY_MAP.items():
            if key in tag_lower:
                return category
    
    # Check description
    if entry.description:
        desc_lower = entry.description.lower()
        for key, category in DEFAULT_CATEGORY_MAP.items():
            if key in desc_lower:
                return category
    
    return "Other"


async def get_user_info() -> dict:
    """Get current user info including default workspace."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{CLOCKIFY_API_BASE}/user",
            headers=_get_headers(),
        )
        if response.status_code == 200:
            return response.json()
        return {}


async def get_entries(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 500,
) -> list[ClockifyEntry]:
    """
    Get time entries from Clockify.
    
    Args:
        start_date: Start of date range (default: 30 days ago)
        end_date: End of date range (default: now)
        limit: Maximum entries to fetch
        
    Returns:
        List of ClockifyEntry objects
    """
    if not start_date:
        start_date = datetime.now() - timedelta(days=30)
    if not end_date:
        end_date = datetime.now()
    
    entries = []
    
    async with httpx.AsyncClient() as client:
        # Get user and workspace
        user = await get_user_info()
        workspace_id = user.get("defaultWorkspace")
        user_id = user.get("id")
        
        if not workspace_id or not user_id:
            logger.error("Could not get workspace/user ID")
            return entries
        
        # Get projects for name lookup
        projects_response = await client.get(
            f"{CLOCKIFY_API_BASE}/workspaces/{workspace_id}/projects",
            headers=_get_headers(),
        )
        project_map = {}
        if projects_response.status_code == 200:
            project_map = {p["id"]: p["name"] for p in projects_response.json()}
        
        # Get time entries
        response = await client.get(
            f"{CLOCKIFY_API_BASE}/workspaces/{workspace_id}/user/{user_id}/time-entries",
            headers=_get_headers(),
            params={
                "start": start_date.isoformat() + "Z",
                "end": end_date.isoformat() + "Z",
                "page-size": limit,
            },
        )
        
        if response.status_code == 200:
            data = response.json()
            
            for item in data:
                interval = item.get("timeInterval", {})
                start_str = interval.get("start")
                end_str = interval.get("end")
                
                if not start_str or not end_str:
                    continue  # Skip running entries
                
                # Get tag names
                tag_ids = item.get("tagIds", [])
                tags = []  # Would need separate API call to get tag names
                
                entries.append(ClockifyEntry(
                    id=item["id"],
                    description=item.get("description", ""),
                    start=datetime.fromisoformat(start_str.replace("Z", "+00:00")),
                    end=datetime.fromisoformat(end_str.replace("Z", "+00:00")),
                    duration=interval.get("duration"),
                    project_name=project_map.get(item.get("projectId")),
                    tags=tags,
                    workspace_id=workspace_id,
                ))
        else:
            logger.error(f"Clockify API error: {response.status_code}")
    
    return entries


def convert_to_dataframe_row(entry: ClockifyEntry) -> dict:
    """Convert Clockify entry to DataFrame row format."""
    duration_hours = _parse_duration(entry.duration)
    
    return {
        "activity_type": _map_to_category(entry),
        "duration_hours": duration_hours,
        "datetime_from": entry.start.replace(tzinfo=None),
        "datetime_to": entry.end.replace(tzinfo=None) if entry.end else datetime.now(),
        "comment": entry.description,
        "source": "clockify",
        "clockify_id": entry.id,
    }


async def sync_entries(days: int = 7) -> list[dict]:
    """Sync recent entries from Clockify."""
    start_date = datetime.now() - timedelta(days=days)
    entries = await get_entries(start_date=start_date)
    
    rows = []
    for entry in entries:
        if entry.end:  # Only completed entries
            rows.append(convert_to_dataframe_row(entry))
    
    logger.info(f"Synced {len(rows)} entries from Clockify")
    return rows


def is_configured() -> bool:
    """Check if Clockify integration is configured."""
    return bool(CLOCKIFY_API_KEY)


async def test_connection() -> tuple[bool, str]:
    """Test the Clockify API connection."""
    if not CLOCKIFY_API_KEY:
        return False, "CLOCKIFY_API_KEY not set"
    
    try:
        user = await get_user_info()
        if user:
            return True, f"Connected as {user.get('name', user.get('email'))}"
        return False, "Could not get user info"
    except Exception as e:
        return False, f"Connection error: {e}"
