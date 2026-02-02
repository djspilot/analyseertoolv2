"""
Pytest configuration and fixtures
"""
import os
import sys
from pathlib import Path
import tempfile
from unittest.mock import Mock

import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def test_data():
    """Create sample test data for testing."""
    dates = pd.date_range("2026-01-01", periods=7, freq="D")
    data = {
        "activity_type": ["Work", "Coding", "Break", "Learning", "Work", "Coding", "Break"],
        "duration_hours": [2.5, 1.5, 0.5, 2.0, 3.0, 2.0, 0.5],
        "datetime_from": dates,
        "datetime_to": dates + pd.Timedelta(hours=1),
        "comment": [""] * 7,
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def mock_db():
    """Mock database connection."""
    conn = Mock()
    yield conn


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def sample_activity():
    """Create a sample activity for testing."""
    from datetime import datetime, timedelta
    from src.models.activity import Activity, ActivityType
    
    return Activity(
        id=1,
        activity_type=ActivityType.WORK,
        duration_hours=2.5,
        datetime_from=datetime(2026, 1, 1, 9, 0),
        datetime_to=datetime(2026, 1, 1, 11, 30),
        comment="Test activity",
        is_deep_work=1,
        fragmentation_risk=0,
    )