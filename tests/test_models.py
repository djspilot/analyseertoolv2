"""
Unit tests for data models
"""
import pytest
from datetime import datetime, timedelta
from pydantic import ValidationError

from src.models.activity import Activity, ActivityType


class TestActivityModel:
    """Test Activity data model."""

    def test_valid_activity_creation(self):
        """Test creating a valid activity."""
        activity = Activity(
            id=1,
            activity_type=ActivityType.WORK,
            duration_hours=2.5,
            datetime_from=datetime(2026, 1, 1, 9, 0),
            datetime_to=datetime(2026, 1, 1, 11, 30),
            comment="Test activity",
        )
        
        assert activity.id == 1
        assert activity.activity_type == ActivityType.WORK
        assert activity.duration_hours == 2.5
        assert activity.comment == "Test activity"
        assert activity.is_deep_work == 1  # Work is deep work

    def test_activity_with_shallow_work(self):
        """Test activity with shallow work category."""
        activity = Activity(
            id=1,
            activity_type=ActivityType.BREAK,
            duration_hours=0.5,
            datetime_from=datetime(2026, 1, 1, 12, 0),
            datetime_to=datetime(2026, 1, 1, 12, 30),
            comment="Lunch break",
        )
        
        assert activity.is_deep_work == 0  # Break is not deep work

    def test_invalid_duration_negative(self):
        """Test validation fails with negative duration."""
        with pytest.raises(ValidationError) as exc_info:
            Activity(
                id=1,
                activity_type=ActivityType.WORK,
                duration_hours=-1.0,
                datetime_from=datetime(2026, 1, 1, 9, 0),
                datetime_to=datetime(2026, 1, 1, 11, 30),
            )
        
        assert "duration_hours" in str(exc_info.value)

    def test_invalid_duration_zero(self):
        """Test validation fails with zero duration."""
        with pytest.raises(ValidationError) as exc_info:
            Activity(
                id=1,
                activity_type=ActivityType.WORK,
                duration_hours=0,
                datetime_from=datetime(2026, 1, 1, 9, 0),
                datetime_to=datetime(2026, 1, 1, 11, 30),
            )
        
        assert "duration_hours" in str(exc_info.value)

    def test_invalid_datetime_order(self):
        """Test validation fails when datetime_from > datetime_to."""
        with pytest.raises(ValidationError) as exc_info:
            Activity(
                id=1,
                activity_type=ActivityType.WORK,
                duration_hours=2.5,
                datetime_from=datetime(2026, 1, 1, 11, 30),
                datetime_to=datetime(2026, 1, 1, 9, 0),  # End before start!
            )
        
        assert "datetime" in str(exc_info.value).lower()

    def test_get_date_method(self):
        """Test get_date helper method."""
        activity = Activity(
            id=1,
            activity_type=ActivityType.WORK,
            duration_hours=2.5,
            datetime_from=datetime(2026, 1, 1, 9, 0),
            datetime_to=datetime(2026, 1, 1, 11, 30),
        )
        
        date = activity.get_date()
        assert date.year == 2026
        assert date.month == 1
        assert date.day == 1

    def test_get_hour_method(self):
        """Test get_hour helper method."""
        activity = Activity(
            id=1,
            activity_type=ActivityType.WORK,
            duration_hours=2.5,
            datetime_from=datetime(2026, 1, 1, 9, 30),
            datetime_to=datetime(2026, 1, 1, 12, 0),
        )
        
        hour = activity.get_hour()
        assert hour == 9

    def test_is_flow_session(self):
        """Test is_flow_session method."""
        # Deep work >= 90min
        flow_activity = Activity(
            id=1,
            activity_type=ActivityType.WORK,
            duration_hours=1.5,
            datetime_from=datetime(2026, 1, 1, 9, 0),
            datetime_to=datetime(2026, 1, 1, 10, 30),
        )
        assert flow_activity.is_flow_session() is True
        
        # Deep work < 90min
        shallow_activity = Activity(
            id=2,
            activity_type=ActivityType.WORK,
            duration_hours=1.0,
            datetime_from=datetime(2026, 1, 1, 9, 0),
            datetime_to=datetime(2026, 1, 1, 10, 0),
        )
        assert shallow_activity.is_flow_session() is False

    def test_activity_type_enum(self):
        """Test ActivityType enum values."""
        assert ActivityType.WORK == "Work"
        assert ActivityType.CODING == "Coding"
        assert ActivityType.BREAK == "Break"
        assert ActivityType.LEARNING == "Learning"
        assert ActivityType.EXERCISE == "Exercise"
        assert ActivityType.OTHER == "Other"


class TestActivityType:
    """Test ActivityType enum."""

    def test_deep_work_categories(self):
        """Test which categories are considered deep work."""
        assert ActivityType.WORK.is_deep_work() is True
        assert ActivityType.CODING.is_deep_work() is True
        assert ActivityType.LEARNING.is_deep_work() is True
        assert ActivityType.BREAK.is_deep_work() is False
        assert ActivityType.EXERCISE.is_deep_work() is False
        assert ActivityType.OTHER.is_deep_work() is False