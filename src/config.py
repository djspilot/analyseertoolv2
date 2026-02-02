"""
Configuration constants for the analyseertool.
Centralized configuration for thresholds, categories, and behavior.
"""

from pathlib import Path
from typing import List

# Database
DB_PATH = Path(__file__).parent.parent / "data" / "lifestyle.db"
BACKUP_PATH = Path(__file__).parent.parent / "data" / "lifestyle.db.backup"

# Deep Work Thresholds
DEEP_WORK_MIN_DURATION_HOURS = 1.0  # 1 hour minimum for deep work
FLOW_MIN_DURATION_HOURS = 1.5  # 90 minutes minimum for flow state
FRAGMENTATION_RISK_THRESHOLD_HOURS = 0.25  # 15 minutes (short sessions)

# Deep Work Categories
DEEP_WORK_CATEGORIES: List[str] = ['Work', 'Coding']

# Activity Category Colors
CATEGORY_COLORS = {
    'Work': '#4CAF50',
    'Coding': '#2196F3',
    'Sport': '#FF9800',
    'Entertainment': '#9C27B0',
    'Housework': '#795548',
    'Read': '#00BCD4',
    'Yoga': '#E91E63',
    'Walking': '#8BC34A',
    'Music': '#673AB7',
    'Internet': '#607D8B',
    'Other': '#9E9E9E',
}

# Chart Defaults
DEFAULT_CHART_HEIGHT = 350
GANNT_CHART_HEIGHT = 400
HEATMAP_CHART_HEIGHT = 350

# Time Range Slider
SLIDER_MIN = 0
SLIDER_MAX = 100
SLIDER_STEP = 1

# Date Formats
CSV_DATE_FORMAT = "%d %b %H:%M"
CSV_ALT_DATE_FORMAT = "%d %B %H:%M"
DB_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# Validation Rules
MAX_DURATION_HOURS = 24  # Maximum realistic session duration
MIN_DURATION_HOURS = 0.01  # Minimum realistic session duration (~30 seconds)
FUTURE_DATE_TOLERANCE_HOURS = 24  # Allow sessions up to 24h in future (timezone issues)

# UI Settings
SIDEBAR_WIDTH = 280
MAX_CATEGORY_FILTER_HEIGHT = 200
MAX_CHART_TOGGLES_HEIGHT = 300

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"