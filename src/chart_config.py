"""
Chart configuration for dynamic visualization toggles.
Allows users to show/hide specific charts and save preferences.
"""

from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path
import json


@dataclass
class ChartConfig:
    """Configuration for a single chart."""
    id: str
    name: str
    category: str  # "flow", "patterns", "circadian", "comparison"
    default_visible: bool = True
    description: str = ""


# All available charts in the dashboard
AVAILABLE_CHARTS: Dict[str, ChartConfig] = {
    # Flow Analysis
    "flow_shallow": ChartConfig(
        id="flow_shallow",
        name="Flow vs Shallow Time", 
        category="flow",
        default_visible=True,
        description="Stacked bar showing deep work vs shallow work per day"
    ),
    "session_dist": ChartConfig(
        id="session_dist",
        name="Session Length Distribution",
        category="flow", 
        default_visible=True,
        description="Histogram of session durations with flow threshold"
    ),
    "flow_prob": ChartConfig(
        id="flow_prob",
        name="Flow Probability by Hour",
        category="flow",
        default_visible=True,
        description="When are you most likely to achieve flow state"
    ),
    "flow_calendar": ChartConfig(
        id="flow_calendar",
        name="Flow Calendar",
        category="flow",
        default_visible=True,
        description="Heatmap calendar of flow hours"
    ),
    
    # Time Patterns
    "breakdown": ChartConfig(
        id="breakdown",
        name="Daily Breakdown",
        category="patterns",
        default_visible=True,
        description="Stacked bar chart of categories per day"
    ),
    "trend": ChartConfig(
        id="trend",
        name="Deep Work Trend",
        category="patterns",
        default_visible=True,
        description="Deep work hours with rolling average"
    ),
    
    # Circadian & Sleep
    "heatmap": ChartConfig(
        id="heatmap",
        name="Activity Heatmap",
        category="circadian",
        default_visible=True,
        description="Hour x Day heatmap of activity intensity"
    ),
    "hourly": ChartConfig(
        id="hourly",
        name="Circadian Profile",
        category="circadian",
        default_visible=True,
        description="Stacked area showing average activity by hour"
    ),
    "frag": ChartConfig(
        id="frag",
        name="Fragmentation Index",
        category="circadian",
        default_visible=False,
        description="How fragmented is each category"
    ),
    "sleep": ChartConfig(
        id="sleep",
        name="Sleep Pattern",
        category="circadian",
        default_visible=False,
        description="Wake and sleep times over time"
    ),
    
    # Timeline
    "gantt": ChartConfig(
        id="gantt",
        name="Daily Timeline (Gantt)",
        category="timeline",
        default_visible=True,
        description="Timeline view for a specific day"
    ),
    
    # Comparison charts (new)
    "comparison_delta": ChartConfig(
        id="comparison_delta",
        name="Period Comparison",
        category="comparison",
        default_visible=True,
        description="Side-by-side metrics for Period A vs B"
    ),
    "comparison_categories": ChartConfig(
        id="comparison_categories",
        name="Category Comparison",
        category="comparison",
        default_visible=True,
        description="Per-category changes between periods"
    ),
}


def get_charts_by_category(category: str) -> List[ChartConfig]:
    """Get all charts in a specific category."""
    return [c for c in AVAILABLE_CHARTS.values() if c.category == category]


def get_default_visibility() -> Dict[str, bool]:
    """Get default visibility settings for all charts."""
    return {chart_id: config.default_visible for chart_id, config in AVAILABLE_CHARTS.items()}


class PreferencesManager:
    """Manages user preferences for chart visibility and other settings."""
    
    PREFS_FILE = Path(__file__).parent.parent / "data" / "preferences.json"
    
    @classmethod
    def load(cls) -> Dict:
        """Load preferences from file."""
        if cls.PREFS_FILE.exists():
            try:
                return json.loads(cls.PREFS_FILE.read_text())
            except (json.JSONDecodeError, IOError):
                pass
        return cls._get_defaults()
    
    @classmethod
    def save(cls, prefs: Dict) -> None:
        """Save preferences to file."""
        cls.PREFS_FILE.parent.mkdir(parents=True, exist_ok=True)
        cls.PREFS_FILE.write_text(json.dumps(prefs, indent=2))
    
    @classmethod
    def _get_defaults(cls) -> Dict:
        """Get default preferences."""
        return {
            "visible_charts": get_default_visibility(),
            "comparison_enabled": False,
            "dark_mode": False,
            "default_period_preset": "Laatste 7 dagen vs 7 dagen daarvoor",
        }
    
    @classmethod
    def get_visible_charts(cls) -> Dict[str, bool]:
        """Get chart visibility settings."""
        prefs = cls.load()
        return prefs.get("visible_charts", get_default_visibility())
    
    @classmethod
    def set_chart_visible(cls, chart_id: str, visible: bool) -> None:
        """Set visibility for a specific chart."""
        prefs = cls.load()
        if "visible_charts" not in prefs:
            prefs["visible_charts"] = get_default_visibility()
        prefs["visible_charts"][chart_id] = visible
        cls.save(prefs)
    
    @classmethod
    def set_comparison_enabled(cls, enabled: bool) -> None:
        """Enable or disable comparison mode."""
        prefs = cls.load()
        prefs["comparison_enabled"] = enabled
        cls.save(prefs)
    
    @classmethod
    def is_comparison_enabled(cls) -> bool:
        """Check if comparison mode is enabled."""
        return cls.load().get("comparison_enabled", False)
