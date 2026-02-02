"""
Apple Health Integration Module
Imports workout and mindfulness data from Apple Health via exported XML.

Note: Direct Apple HealthKit access requires a native iOS app.
This module works with exported Health data (Settings > Health > Export All Health Data).

For real-time sync, consider using:
- Shortcuts app automation to export data periodically
- Third-party apps like Health Auto Export (iOS) that can push to webhooks
"""

import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional, Generator
from dataclasses import dataclass
import zipfile

from ..logger import setup_logger

logger = setup_logger(__name__)


# Workout type to category mapping
WORKOUT_TYPE_MAP = {
    "HKWorkoutActivityTypeRunning": "Sport",
    "HKWorkoutActivityTypeCycling": "Sport",
    "HKWorkoutActivityTypeSwimming": "Sport",
    "HKWorkoutActivityTypeWalking": "Walking",
    "HKWorkoutActivityTypeHiking": "Walking",
    "HKWorkoutActivityTypeYoga": "Yoga",
    "HKWorkoutActivityTypePilates": "Yoga",
    "HKWorkoutActivityTypeFunctionalStrengthTraining": "Sport",
    "HKWorkoutActivityTypeTraditionalStrengthTraining": "Sport",
    "HKWorkoutActivityTypeHighIntensityIntervalTraining": "Sport",
    "HKWorkoutActivityTypeDance": "Sport",
    "HKWorkoutActivityTypeMindAndBody": "Yoga",
    "HKWorkoutActivityTypeCoreTraining": "Sport",
    "HKWorkoutActivityTypeFlexibility": "Yoga",
    "HKWorkoutActivityTypeMixedCardio": "Sport",
}


@dataclass
class HealthWorkout:
    """An Apple Health workout record."""
    workout_type: str
    start_date: datetime
    end_date: datetime
    duration_minutes: float
    total_energy_burned: Optional[float]  # kcal
    total_distance: Optional[float]  # meters
    source_name: str


@dataclass
class MindfulSession:
    """An Apple Health mindful session."""
    start_date: datetime
    end_date: datetime
    duration_minutes: float
    source_name: str


def _parse_apple_date(date_str: str) -> datetime:
    """Parse Apple Health date format."""
    # Format: 2024-01-15 08:30:00 +0100
    try:
        return datetime.strptime(date_str[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.fromisoformat(date_str)


def parse_health_export(
    export_path: str | Path,
    start_date: Optional[datetime] = None,
) -> Generator[dict, None, None]:
    """
    Parse Apple Health export zip file.
    
    Args:
        export_path: Path to export.zip or extracted folder
        start_date: Only include records after this date
        
    Yields:
        Dict records ready for DataFrame
    """
    export_path = Path(export_path)
    
    # Handle zip file
    if export_path.suffix == ".zip":
        with zipfile.ZipFile(export_path, 'r') as zf:
            # Extract export.xml
            with zf.open("apple_health_export/export.xml") as f:
                yield from _parse_health_xml(f, start_date)
    elif export_path.is_dir():
        xml_path = export_path / "apple_health_export" / "export.xml"
        if xml_path.exists():
            with open(xml_path, 'rb') as f:
                yield from _parse_health_xml(f, start_date)
    elif export_path.suffix == ".xml":
        with open(export_path, 'rb') as f:
            yield from _parse_health_xml(f, start_date)
    else:
        raise ValueError(f"Unsupported file format: {export_path}")


def _parse_health_xml(file_obj, start_date: Optional[datetime]) -> Generator[dict, None, None]:
    """Parse the Health export XML file using iterparse for memory efficiency."""
    
    # Use iterparse for large files
    context = ET.iterparse(file_obj, events=("end",))
    
    for event, elem in context:
        if elem.tag == "Workout":
            workout = _parse_workout_element(elem)
            if workout and (not start_date or workout.start_date >= start_date):
                yield _workout_to_row(workout)
            elem.clear()
        
        elif elem.tag == "Record":
            record_type = elem.get("type")
            if record_type == "HKCategoryTypeIdentifierMindfulSession":
                session = _parse_mindful_element(elem)
                if session and (not start_date or session.start_date >= start_date):
                    yield _mindful_to_row(session)
            elem.clear()


def _parse_workout_element(elem: ET.Element) -> Optional[HealthWorkout]:
    """Parse a Workout XML element."""
    try:
        workout_type = elem.get("workoutActivityType", "")
        start_date = _parse_apple_date(elem.get("startDate", ""))
        end_date = _parse_apple_date(elem.get("endDate", ""))
        duration = float(elem.get("duration", 0))
        
        # Get optional values
        energy = None
        distance = None
        
        for child in elem:
            if child.tag == "WorkoutStatistics":
                stat_type = child.get("type", "")
                if stat_type == "HKQuantityTypeIdentifierActiveEnergyBurned":
                    energy = float(child.get("sum", 0))
                elif stat_type == "HKQuantityTypeIdentifierDistanceWalkingRunning":
                    distance = float(child.get("sum", 0))
        
        return HealthWorkout(
            workout_type=workout_type,
            start_date=start_date,
            end_date=end_date,
            duration_minutes=duration,
            total_energy_burned=energy,
            total_distance=distance,
            source_name=elem.get("sourceName", "Apple Health"),
        )
    except Exception as e:
        logger.warning(f"Failed to parse workout: {e}")
        return None


def _parse_mindful_element(elem: ET.Element) -> Optional[MindfulSession]:
    """Parse a mindful session Record element."""
    try:
        start_date = _parse_apple_date(elem.get("startDate", ""))
        end_date = _parse_apple_date(elem.get("endDate", ""))
        duration = (end_date - start_date).total_seconds() / 60
        
        return MindfulSession(
            start_date=start_date,
            end_date=end_date,
            duration_minutes=duration,
            source_name=elem.get("sourceName", "Apple Health"),
        )
    except Exception as e:
        logger.warning(f"Failed to parse mindful session: {e}")
        return None


def _workout_to_row(workout: HealthWorkout) -> dict:
    """Convert workout to DataFrame row."""
    category = WORKOUT_TYPE_MAP.get(workout.workout_type, "Sport")
    
    # Build comment with details
    comment_parts = [workout.source_name]
    if workout.total_energy_burned:
        comment_parts.append(f"{workout.total_energy_burned:.0f} kcal")
    if workout.total_distance:
        comment_parts.append(f"{workout.total_distance/1000:.2f} km")
    
    return {
        "activity_type": category,
        "duration_hours": round(workout.duration_minutes / 60, 2),
        "datetime_from": workout.start_date,
        "datetime_to": workout.end_date,
        "comment": " | ".join(comment_parts),
        "source": "apple_health",
        "workout_type": workout.workout_type,
    }


def _mindful_to_row(session: MindfulSession) -> dict:
    """Convert mindful session to DataFrame row."""
    return {
        "activity_type": "Yoga",  # Map mindfulness to Yoga
        "duration_hours": round(session.duration_minutes / 60, 2),
        "datetime_from": session.start_date,
        "datetime_to": session.end_date,
        "comment": f"Mindfulness | {session.source_name}",
        "source": "apple_health",
    }


def import_health_export(
    export_path: str | Path,
    start_date: Optional[datetime] = None,
) -> list[dict]:
    """
    Import all relevant data from Apple Health export.
    
    Args:
        export_path: Path to export.zip or folder
        start_date: Only include records after this date
        
    Returns:
        List of dicts ready for DataFrame/database
    """
    rows = list(parse_health_export(export_path, start_date))
    logger.info(f"Imported {len(rows)} records from Apple Health")
    return rows


# ============================================================================
# Health Auto Export Integration
# (For real-time sync via iOS app webhook)
# ============================================================================

def parse_health_auto_export_payload(payload: dict) -> list[dict]:
    """
    Parse payload from Health Auto Export iOS app.
    
    The app can be configured to POST workout data to a webhook URL.
    
    Args:
        payload: JSON payload from Health Auto Export
        
    Returns:
        List of dicts ready for DataFrame
    """
    rows = []
    
    # Health Auto Export sends metrics in various formats
    # This handles the workout export format
    metrics = payload.get("data", {}).get("metrics", [])
    
    for metric in metrics:
        name = metric.get("name", "")
        
        if "workout" in name.lower():
            for data_point in metric.get("data", []):
                workout_type = data_point.get("workoutType", "")
                
                rows.append({
                    "activity_type": WORKOUT_TYPE_MAP.get(workout_type, "Sport"),
                    "duration_hours": data_point.get("duration", 0) / 60,
                    "datetime_from": datetime.fromisoformat(data_point.get("date")),
                    "datetime_to": datetime.fromisoformat(data_point.get("dateEnd", data_point.get("date"))),
                    "comment": f"{data_point.get('sourceName', '')} | {data_point.get('totalEnergyBurned', 0):.0f} kcal",
                    "source": "health_auto_export",
                })
    
    return rows
