"""
aTimeLogger Integration
Automatically sync time tracking data from aTimeLogger app.
"""

import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import requests

from ..logger import setup_logger
from ..api.database import add_activity, get_all_activities, init_db

logger = setup_logger(__name__)


# ============================================================================
# Configuration
# ============================================================================

ATIMELOGGER_USERNAME = os.environ.get("ATIMELOGGER_USERNAME", "")
ATIMELOGGER_PASSWORD = os.environ.get("ATIMELOGGER_PASSWORD", "")
ATIMELOGGER_ENDPOINT = "https://app.atimelogger.com/api/v2"


def is_configured() -> bool:
    """Check if aTimeLogger credentials are configured."""
    return bool(ATIMELOGGER_USERNAME and ATIMELOGGER_PASSWORD)


# ============================================================================
# API Client
# ============================================================================

class aTimeLoggerClient:
    """
    aTimeLogger API client.
    Based on the REST API: http://blog.timetrack.io/rest-api
    """
    
    LIMIT_MAX = 0x7FFF_FFFF
    
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.session = requests.Session()
        self.session.auth = (username, password)
        self._types_cache: Dict[str, dict] = {}
    
    def __enter__(self):
        return self
    
    def __exit__(self, *exc):
        self.close()
    
    def close(self):
        self.session.close()
    
    def _request(
        self,
        method: str,
        model: str,
        guid: str = "",
        params: Optional[dict] = None,
        json: Optional[dict] = None,
    ) -> requests.Response:
        """Make API request."""
        url = f"{ATIMELOGGER_ENDPOINT}/{model}/{guid}"
        response = self.session.request(method, url, params=params, json=json)
        response.raise_for_status()
        return response
    
    def get_types(self) -> List[dict]:
        """Get all activity types."""
        response = self._request("get", "types")
        data = response.json()
        types = data.get("types", [])
        
        # Cache for lookup
        self._types_cache = {t["guid"]: t for t in types}
        
        return types
    
    def get_type_name(self, guid: str) -> str:
        """Get type name by GUID."""
        if not self._types_cache:
            self.get_types()
        
        type_info = self._types_cache.get(guid, {})
        return type_info.get("name", "Unknown")
    
    def get_type_group(self, guid: str) -> Optional[str]:
        """Get type group (category) by GUID."""
        if not self._types_cache:
            self.get_types()
        
        type_info = self._types_cache.get(guid, {})
        return type_info.get("group")
    
    def get_intervals(
        self,
        days: int = 7,
        limit: int = 10000,
        offset: int = 0,
    ) -> List[dict]:
        """
        Get intervals from the last N days.
        
        Args:
            days: Number of days to fetch
            limit: Max number of intervals
            offset: Pagination offset
            
        Returns:
            List of interval records
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        params = {
            "limit": limit,
            "offset": offset,
            "from": int(start_date.timestamp()),
            "to": int(end_date.timestamp()),
            "order": "desc",
        }
        
        response = self._request("get", "intervals", params=params)
        data = response.json()
        
        intervals = data.get("intervals", [])
        meta = data.get("meta", {})
        
        logger.info(f"aTimeLogger: fetched {len(intervals)} intervals (total: {meta.get('total', '?')})")
        
        return intervals
    
    def get_running_activity(self) -> Optional[dict]:
        """Get currently running activity if any."""
        params = {"state": "running", "limit": 1}
        response = self._request("get", "activities", params=params)
        data = response.json()
        activities = data.get("activities", [])
        
        if activities:
            return activities[0]
        return None


# ============================================================================
# Data Conversion
# ============================================================================

@dataclass
class ATLEntry:
    """Converted aTimeLogger entry."""
    activity: str
    category: str
    datetime_from: datetime
    datetime_to: datetime
    duration_hours: float
    comment: Optional[str] = None
    guid: str = ""


def convert_interval(interval: dict, client: aTimeLoggerClient) -> ATLEntry:
    """Convert aTimeLogger interval to our format."""
    type_guid = interval.get("typeGuid") or interval.get("type", {}).get("guid", "")
    
    # Get timestamps
    from_ts = interval.get("from")
    to_ts = interval.get("to")
    
    # Handle datetime objects or timestamps
    if isinstance(from_ts, datetime):
        dt_from = from_ts
    elif isinstance(from_ts, (int, float)):
        dt_from = datetime.fromtimestamp(from_ts)
    else:
        dt_from = datetime.now()
    
    if isinstance(to_ts, datetime):
        dt_to = to_ts
    elif isinstance(to_ts, (int, float)):
        dt_to = datetime.fromtimestamp(to_ts)
    else:
        dt_to = dt_from
    
    # Calculate duration
    duration = (dt_to - dt_from).total_seconds() / 3600
    
    # Get activity name and category
    activity = client.get_type_name(type_guid)
    category = client.get_type_group(type_guid) or infer_category(activity)
    
    return ATLEntry(
        activity=activity,
        category=category,
        datetime_from=dt_from,
        datetime_to=dt_to,
        duration_hours=duration,
        comment=interval.get("comment"),
        guid=interval.get("guid", ""),
    )


def infer_category(activity: str) -> str:
    """Infer category from activity name."""
    activity_lower = activity.lower()
    
    # Category mappings
    mappings = {
        "Werk": ["werk", "work", "coding", "programmeren", "meeting", "vergadering", "email", "focus"],
        "Sport": ["sport", "gym", "fitness", "hardlopen", "run", "yoga", "zwemmen", "fietsen", "wandelen"],
        "Slaap": ["slaap", "sleep", "slapen", "nap", "dutje"],
        "Leren": ["leren", "study", "studeren", "lezen", "reading", "course", "cursus"],
        "Entertainment": ["entertainment", "tv", "netflix", "youtube", "gaming", "game", "film", "serie"],
        "Persoonlijk": ["persoonlijk", "personal", "douche", "eten", "ontbijt", "lunch", "diner"],
        "Reizen": ["reizen", "travel", "commute", "trein", "auto", "fiets"],
    }
    
    for category, keywords in mappings.items():
        if any(kw in activity_lower for kw in keywords):
            return category
    
    return "Overig"


# ============================================================================
# Sync Functions
# ============================================================================

async def sync_entries(days: int = 7) -> List[dict]:
    """
    Sync entries from aTimeLogger.
    
    Args:
        days: Number of days to sync
        
    Returns:
        List of synced entries as dicts
    """
    if not is_configured():
        logger.warning("aTimeLogger not configured. Set ATIMELOGGER_USERNAME and ATIMELOGGER_PASSWORD")
        return []
    
    init_db()
    
    try:
        with aTimeLoggerClient(ATIMELOGGER_USERNAME, ATIMELOGGER_PASSWORD) as client:
            # Get existing entries to avoid duplicates
            existing_df = get_all_activities()
            existing_times = set()
            if not existing_df.empty:
                existing_times = set(existing_df["datetime_from"].astype(str))
            
            # Fetch intervals
            intervals = client.get_intervals(days=days)
            
            # Convert and add new entries
            new_entries = []
            for interval in intervals:
                entry = convert_interval(interval, client)
                
                # Skip if already exists (check by start time)
                if str(entry.datetime_from) in existing_times:
                    continue
                
                # Add to database
                add_activity(
                    activity=entry.activity,
                    category=entry.category,
                    datetime_from=entry.datetime_from,
                    datetime_to=entry.datetime_to,
                    duration_hours=entry.duration_hours,
                )
                
                new_entries.append({
                    "activity": entry.activity,
                    "category": entry.category,
                    "start": entry.datetime_from.isoformat(),
                    "end": entry.datetime_to.isoformat(),
                    "duration_hours": round(entry.duration_hours, 2),
                })
            
            logger.info(f"aTimeLogger: synced {len(new_entries)} new entries")
            return new_entries
    
    except requests.exceptions.HTTPError as e:
        logger.error(f"aTimeLogger API error: {e}")
        return []
    except Exception as e:
        logger.error(f"aTimeLogger sync failed: {e}")
        return []


async def get_current_activity() -> Optional[dict]:
    """Get currently running activity from aTimeLogger."""
    if not is_configured():
        return None
    
    try:
        with aTimeLoggerClient(ATIMELOGGER_USERNAME, ATIMELOGGER_PASSWORD) as client:
            activity = client.get_running_activity()
            if activity:
                type_guid = activity.get("type", {}).get("guid", "")
                return {
                    "activity": client.get_type_name(type_guid),
                    "started_at": activity.get("from"),
                }
        return None
    except Exception as e:
        logger.error(f"Failed to get current activity: {e}")
        return None


async def test_connection() -> tuple[bool, str]:
    """Test connection to aTimeLogger API."""
    if not is_configured():
        return False, "Not configured - set ATIMELOGGER_USERNAME and ATIMELOGGER_PASSWORD"
    
    try:
        with aTimeLoggerClient(ATIMELOGGER_USERNAME, ATIMELOGGER_PASSWORD) as client:
            types = client.get_types()
            return True, f"Connected! Found {len(types)} activity types"
    except requests.exceptions.HTTPError as e:
        return False, f"Auth failed: {e}"
    except Exception as e:
        return False, f"Connection failed: {e}"


async def full_sync(days: int = 365) -> dict:
    """
    Full sync - fetches all data from the last N days.
    Use this for initial import instead of CSV upload.
    
    Args:
        days: Number of days to sync (default: 365 for full year)
        
    Returns:
        Sync statistics
    """
    if not is_configured():
        return {"success": False, "error": "Not configured"}
    
    init_db()
    
    try:
        with aTimeLoggerClient(ATIMELOGGER_USERNAME, ATIMELOGGER_PASSWORD) as client:
            # Get all types first
            types = client.get_types()
            logger.info(f"aTimeLogger: found {len(types)} activity types")
            
            # Fetch all intervals
            intervals = client.get_intervals(days=days, limit=50000)
            
            # Get existing entries
            existing_df = get_all_activities()
            existing_times = set()
            if not existing_df.empty:
                existing_times = set(existing_df["datetime_from"].astype(str))
            
            # Convert and add
            added = 0
            skipped = 0
            
            for interval in intervals:
                entry = convert_interval(interval, client)
                
                if str(entry.datetime_from) in existing_times:
                    skipped += 1
                    continue
                
                add_activity(
                    activity=entry.activity,
                    category=entry.category,
                    datetime_from=entry.datetime_from,
                    datetime_to=entry.datetime_to,
                    duration_hours=entry.duration_hours,
                )
                added += 1
            
            return {
                "success": True,
                "types": len(types),
                "intervals_found": len(intervals),
                "added": added,
                "skipped": skipped,
            }
    
    except Exception as e:
        logger.error(f"Full sync failed: {e}")
        return {"success": False, "error": str(e)}
