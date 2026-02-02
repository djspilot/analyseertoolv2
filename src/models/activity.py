"""
Activity data models and type definitions.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum


class ActivityType(Enum):
    """Activity type categories."""
    WORK = "Work"
    CODING = "Coding"
    LEARNING = "Learning"
    MEETING = "Meeting"
    BREAK = "Break"
    EXERCISE = "Exercise"
    MEAL = "Meal"
    SLEEP = "Sleep"
    COMMUTE = "Commute"
    PERSONAL = "Personal"
    SOCIAL = "Social"
    OTHER = "Other"


@dataclass
class Activity:
    """
    Represents a single time-tracked activity.
    
    Attributes:
        id: Database record ID
        activity_type: Type/category of activity
        duration_hours: Duration in hours
        datetime_from: Start datetime
        datetime_to: End datetime
        comment: Optional comment/note
        is_deep_work: Whether this is deep work (1) or not (0)
        fragmentation_risk: Risk score for fragmentation
    """
    id: Optional[int]
    activity_type: str
    duration_hours: float
    datetime_from: datetime
    datetime_to: datetime
    comment: Optional[str] = None
    is_deep_work: int = 0
    fragmentation_risk: int = 0
    
    def __post_init__(self):
        """Validate activity data."""
        if self.duration_hours <= 0:
            raise ValueError("Duration must be positive")
        if self.datetime_to <= self.datetime_from:
            raise ValueError("End time must be after start time")
        if not self.activity_type:
            raise ValueError("Activity type cannot be empty")
    
    @property
    def duration_minutes(self) -> int:
        """Duration in minutes."""
        return int(self.duration_hours * 60)
    
    @property
    def is_flow(self) -> bool:
        """Whether this session qualifies as flow (>= 90 minutes of deep work)."""
        return self.is_deep_work == 1 and self.duration_hours >= 1.5
    
    def to_dict(self) -> dict:
        """Convert activity to dictionary."""
        return {
            'id': self.id,
            'activity_type': self.activity_type,
            'duration_hours': self.duration_hours,
            'datetime_from': self.datetime_from.strftime('%Y-%m-%d %H:%M:%S'),
            'datetime_to': self.datetime_to.strftime('%Y-%m-%d %H:%M:%S'),
            'comment': self.comment,
            'is_deep_work': self.is_deep_work,
            'fragmentation_risk': self.fragmentation_risk,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Activity':
        """Create Activity from dictionary."""
        return cls(
            id=data.get('id'),
            activity_type=data['activity_type'],
            duration_hours=data['duration_hours'],
            datetime_from=datetime.fromisoformat(data['datetime_from']),
            datetime_to=datetime.fromisoformat(data['datetime_to']),
            comment=data.get('comment'),
            is_deep_work=data.get('is_deep_work', 0),
            fragmentation_risk=data.get('fragmentation_risk', 0),
        )