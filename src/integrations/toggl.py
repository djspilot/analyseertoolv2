"""
Toggl Track Integration Module
Real-time sync with Toggl Track time tracking app.
Supports webhooks for instant updates and REST API for batch sync.

Setup:
1. Get API token from https://track.toggl.com/profile
2. Set environment variable: TOGGL_API_TOKEN=your_token
3. Optionally set TOGGL_WORKSPACE_ID for specific workspace
"""

import os
import json
import httpx
import base64
from datetime import datetime, timedelta
from typing import Optional, AsyncGenerator
from dataclasses import dataclass

from ..logger import setup_logger

logger = setup_logger(__name__)

# Configuration
TOGGL_API_TOKEN = os.environ.get("TOGGL_API_TOKEN", "")
TOGGL_WORKSPACE_ID = os.environ.get("TOGGL_WORKSPACE_ID", "")
TOGGL_API_BASE = "https://api.track.toggl.com/api/v9"

# Category mapping: Toggl project/tag -> Your category
DEFAULT_CATEGORY_MAP = {
    # Projects (case-insensitive)
    "work": "Work",
    "coding": "Coding", 
    "development": "Coding",
    "programming": "Coding",
    "sport": "Sport",
    "exercise": "Sport",
    "gym": "Sport",
    "fitness": "Sport",
    "reading": "Read",
    "entertainment": "Entertainment",
    "netflix": "Entertainment",
    "youtube": "Entertainment",
    "housework": "Housework",
    "cleaning": "Housework",
    "cooking": "Housework",
    "yoga": "Yoga",
    "meditation": "Yoga",
    "walking": "Walking",
    "music": "Music",
    "practice": "Music",
}


@dataclass
class TogglEntry:
    """A Toggl time entry."""
    id: int
    description: str
    start: datetime
    stop: Optional[datetime]
    duration: int  # seconds, negative if running
    project_name: Optional[str]
    tags: list[str]
    workspace_id: int


def _get_auth_header() -> dict:
    """Get authentication header for Toggl API."""
    if not TOGGL_API_TOKEN:
        raise ValueError("TOGGL_API_TOKEN environment variable not set")
    
    # Toggl uses Basic Auth with token as username and "api_token" as password
    credentials = base64.b64encode(f"{TOGGL_API_TOKEN}:api_token".encode()).decode()
    return {"Authorization": f"Basic {credentials}"}


def _map_to_category(entry: TogglEntry) -> str:
    """Map Toggl entry to internal category."""
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


async def get_current_entry() -> Optional[TogglEntry]:
    """
    Get the currently running time entry.
    
    Returns:
        TogglEntry if timer is running, None otherwise
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{TOGGL_API_BASE}/me/time_entries/current",
            headers=_get_auth_header(),
        )
        
        if response.status_code == 200:
            data = response.json()
            if data:
                return TogglEntry(
                    id=data["id"],
                    description=data.get("description", ""),
                    start=datetime.fromisoformat(data["start"].replace("Z", "+00:00")),
                    stop=None,
                    duration=data["duration"],
                    project_name=data.get("project_name"),
                    tags=data.get("tags", []),
                    workspace_id=data["workspace_id"],
                )
    return None


async def get_entries(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 1000,
) -> list[TogglEntry]:
    """
    Get time entries from Toggl.
    
    Args:
        start_date: Start of date range (default: 30 days ago)
        end_date: End of date range (default: now)
        limit: Maximum entries to fetch
        
    Returns:
        List of TogglEntry objects
    """
    if not start_date:
        start_date = datetime.now() - timedelta(days=30)
    if not end_date:
        end_date = datetime.now()
    
    entries = []
    
    async with httpx.AsyncClient() as client:
        # Get entries
        response = await client.get(
            f"{TOGGL_API_BASE}/me/time_entries",
            headers=_get_auth_header(),
            params={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Get project names (need separate call)
            projects = await _get_projects(client)
            project_map = {p["id"]: p["name"] for p in projects}
            
            for item in data[:limit]:
                if item.get("stop"):  # Only completed entries
                    stop_time = datetime.fromisoformat(item["stop"].replace("Z", "+00:00"))
                else:
                    stop_time = None
                
                entries.append(TogglEntry(
                    id=item["id"],
                    description=item.get("description", ""),
                    start=datetime.fromisoformat(item["start"].replace("Z", "+00:00")),
                    stop=stop_time,
                    duration=item["duration"],
                    project_name=project_map.get(item.get("project_id")),
                    tags=item.get("tags", []),
                    workspace_id=item["workspace_id"],
                ))
        else:
            logger.error(f"Toggl API error: {response.status_code} - {response.text}")
    
    return entries


async def _get_projects(client: httpx.AsyncClient) -> list[dict]:
    """Get all projects from Toggl."""
    response = await client.get(
        f"{TOGGL_API_BASE}/me/projects",
        headers=_get_auth_header(),
    )
    
    if response.status_code == 200:
        return response.json()
    return []


def convert_to_dataframe_row(entry: TogglEntry) -> dict:
    """
    Convert Toggl entry to DataFrame row format.
    
    Returns:
        Dict compatible with your activity DataFrame
    """
    if entry.stop is None:
        # Running entry - use current time
        stop = datetime.now(entry.start.tzinfo)
    else:
        stop = entry.stop
    
    duration_hours = abs(entry.duration) / 3600
    if entry.duration < 0:
        # Running timer - calculate from start
        duration_hours = (datetime.now(entry.start.tzinfo) - entry.start).total_seconds() / 3600
    
    return {
        "activity_type": _map_to_category(entry),
        "duration_hours": round(duration_hours, 2),
        "datetime_from": entry.start.replace(tzinfo=None),
        "datetime_to": stop.replace(tzinfo=None),
        "comment": entry.description,
        "source": "toggl",
        "toggl_id": entry.id,
    }


async def sync_entries(days: int = 7) -> list[dict]:
    """
    Sync recent entries from Toggl.
    
    Args:
        days: Number of days to sync
        
    Returns:
        List of dicts ready for DataFrame/database
    """
    start_date = datetime.now() - timedelta(days=days)
    entries = await get_entries(start_date=start_date)
    
    rows = []
    for entry in entries:
        if entry.stop:  # Only completed entries
            rows.append(convert_to_dataframe_row(entry))
    
    logger.info(f"Synced {len(rows)} entries from Toggl")
    return rows


async def start_timer(description: str, category: str) -> Optional[int]:
    """
    Start a new timer in Toggl.
    
    Args:
        description: Timer description
        category: Category name (will be mapped to project/tag)
        
    Returns:
        Entry ID if successful, None otherwise
    """
    # Find or create project for category
    async with httpx.AsyncClient() as client:
        projects = await _get_projects(client)
        
        # Find matching project
        project_id = None
        for project in projects:
            if project["name"].lower() == category.lower():
                project_id = project["id"]
                break
        
        # Start timer
        now = datetime.utcnow().isoformat() + "Z"
        
        payload = {
            "description": description,
            "start": now,
            "duration": -1,  # Running timer
            "created_with": "analyseertool",
        }
        
        if project_id:
            payload["project_id"] = project_id
        else:
            payload["tags"] = [category]
        
        # Get workspace ID
        me_response = await client.get(
            f"{TOGGL_API_BASE}/me",
            headers=_get_auth_header(),
        )
        if me_response.status_code == 200:
            workspace_id = me_response.json()["default_workspace_id"]
            payload["workspace_id"] = workspace_id
        
        response = await client.post(
            f"{TOGGL_API_BASE}/workspaces/{workspace_id}/time_entries",
            headers={**_get_auth_header(), "Content-Type": "application/json"},
            json=payload,
        )
        
        if response.status_code in [200, 201]:
            return response.json()["id"]
        else:
            logger.error(f"Failed to start timer: {response.status_code} - {response.text}")
    
    return None


async def stop_timer() -> Optional[dict]:
    """
    Stop the currently running timer.
    
    Returns:
        Completed entry as dict, or None if no timer running
    """
    current = await get_current_entry()
    if not current:
        return None
    
    async with httpx.AsyncClient() as client:
        response = await client.patch(
            f"{TOGGL_API_BASE}/workspaces/{current.workspace_id}/time_entries/{current.id}/stop",
            headers=_get_auth_header(),
        )
        
        if response.status_code == 200:
            # Refetch to get final duration
            stopped = response.json()
            return convert_to_dataframe_row(TogglEntry(
                id=stopped["id"],
                description=stopped.get("description", ""),
                start=datetime.fromisoformat(stopped["start"].replace("Z", "+00:00")),
                stop=datetime.fromisoformat(stopped["stop"].replace("Z", "+00:00")),
                duration=stopped["duration"],
                project_name=current.project_name,
                tags=stopped.get("tags", []),
                workspace_id=stopped["workspace_id"],
            ))
    
    return None


# ============================================================================
# Webhook Handler (for real-time sync)
# ============================================================================

def parse_webhook_payload(payload: dict) -> Optional[dict]:
    """
    Parse incoming Toggl webhook payload.
    
    Toggl webhooks send events for:
    - time_entry.created
    - time_entry.updated
    - time_entry.deleted
    
    Args:
        payload: Raw webhook JSON payload
        
    Returns:
        Dict with event_type and entry data, or None if invalid
    """
    event_type = payload.get("event_type")
    
    if event_type in ["time_entry.created", "time_entry.updated"]:
        entry_data = payload.get("payload", {})
        
        if entry_data.get("stop"):  # Completed entry
            entry = TogglEntry(
                id=entry_data["id"],
                description=entry_data.get("description", ""),
                start=datetime.fromisoformat(entry_data["start"].replace("Z", "+00:00")),
                stop=datetime.fromisoformat(entry_data["stop"].replace("Z", "+00:00")),
                duration=entry_data["duration"],
                project_name=entry_data.get("project_name"),
                tags=entry_data.get("tags", []),
                workspace_id=entry_data["workspace_id"],
            )
            
            return {
                "event": event_type,
                "action": "upsert",
                "entry": convert_to_dataframe_row(entry),
            }
    
    elif event_type == "time_entry.deleted":
        return {
            "event": event_type,
            "action": "delete",
            "toggl_id": payload.get("payload", {}).get("id"),
        }
    
    return None


# ============================================================================
# Configuration helpers
# ============================================================================

def get_category_mapping() -> dict:
    """Get the current category mapping."""
    return DEFAULT_CATEGORY_MAP.copy()


def is_configured() -> bool:
    """Check if Toggl integration is configured."""
    return bool(TOGGL_API_TOKEN)


async def test_connection() -> tuple[bool, str]:
    """
    Test the Toggl API connection.
    
    Returns:
        (success, message)
    """
    if not TOGGL_API_TOKEN:
        return False, "TOGGL_API_TOKEN not set"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{TOGGL_API_BASE}/me",
                headers=_get_auth_header(),
            )
            
            if response.status_code == 200:
                user = response.json()
                return True, f"Connected as {user.get('fullname', user.get('email'))}"
            else:
                return False, f"API error: {response.status_code}"
    except Exception as e:
        return False, f"Connection error: {e}"
