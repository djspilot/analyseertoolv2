"""
Goals management for the analyseertool.
Allows users to set weekly targets and track progress.
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
import pandas as pd
from datetime import datetime, timedelta

from .config import DEEP_WORK_CATEGORIES
from .utils import matches_categories

# Goals file path
GOALS_PATH = Path(__file__).parent.parent / "data" / "goals.json"

# Default goals
DEFAULT_GOALS = {
    "deep_work_hours_weekly": 20.0,
    "sport_hours_weekly": 5.0,
    "sleep_hours_daily": 8.0,
    "max_fragmentation": 2.0,
    "min_flow_sessions_weekly": 5,
    "max_entertainment_hours_weekly": 10.0,
}


@dataclass
class WeeklyGoals:
    """Weekly goals configuration."""
    deep_work_hours_weekly: float = 20.0
    sport_hours_weekly: float = 5.0
    sleep_hours_daily: float = 8.0
    max_fragmentation: float = 2.0
    min_flow_sessions_weekly: int = 5
    max_entertainment_hours_weekly: float = 10.0


@dataclass
class GoalProgress:
    """Progress towards a single goal."""
    name: str
    label: str
    target: float
    current: float
    unit: str
    is_max: bool = False  # True if lower is better (e.g., fragmentation)
    
    @property
    def percentage(self) -> float:
        if self.target == 0:
            return 0
        if self.is_max:
            # For "max" goals, being under target is good
            if self.current <= self.target:
                return 100.0
            return max(0, 100 - ((self.current - self.target) / self.target * 100))
        return min(100, (self.current / self.target) * 100)
    
    @property
    def status(self) -> str:
        """Return 'success', 'warning', or 'danger' based on progress."""
        pct = self.percentage
        if self.is_max:
            if self.current <= self.target:
                return "success"
            elif self.current <= self.target * 1.2:
                return "warning"
            return "danger"
        else:
            if pct >= 80:
                return "success"
            elif pct >= 50:
                return "warning"
            return "danger"
    
    @property
    def icon(self) -> str:
        """Return an appropriate emoji."""
        if self.status == "success":
            return "✅"
        elif self.status == "warning":
            return "⚠️"
        return "❌"


def load_goals() -> WeeklyGoals:
    """Load goals from file or return defaults."""
    if GOALS_PATH.exists():
        try:
            with open(GOALS_PATH, 'r') as f:
                data = json.load(f)
                return WeeklyGoals(**data)
        except (json.JSONDecodeError, TypeError):
            pass
    return WeeklyGoals()


def save_goals(goals: WeeklyGoals) -> None:
    """Save goals to file."""
    GOALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GOALS_PATH, 'w') as f:
        json.dump(asdict(goals), f, indent=2)


def calculate_goal_progress(df: pd.DataFrame, goals: Optional[WeeklyGoals] = None) -> list[GoalProgress]:
    """
    Calculate progress towards weekly goals.
    
    Args:
        df: Full DataFrame with activity data
        goals: Goals configuration (loads from file if not provided)
    
    Returns:
        List of GoalProgress objects
    """
    if goals is None:
        goals = load_goals()
    
    progress = []
    
    if df.empty:
        return progress
    
    # Get data from the last 7 days
    max_date = df['datetime_from'].max()
    week_start = max_date - timedelta(days=7)
    week_df = df[df['datetime_from'] >= week_start]
    
    # 1. Deep Work Hours
    deep_work_df = week_df[week_df['activity_type'].apply(lambda x: matches_categories(x, DEEP_WORK_CATEGORIES))]
    deep_work_hours = deep_work_df['duration_hours'].sum()
    progress.append(GoalProgress(
        name="deep_work",
        label="Deep Work",
        target=goals.deep_work_hours_weekly,
        current=round(deep_work_hours, 1),
        unit="uur/week"
    ))
    
    # 2. Sport Hours
    sport_df = week_df[week_df['activity_type'].apply(lambda x: matches_categories(x, ['Sport', 'Yoga', 'Walking']))]
    sport_hours = sport_df['duration_hours'].sum()
    progress.append(GoalProgress(
        name="sport",
        label="Sport & Beweging",
        target=goals.sport_hours_weekly,
        current=round(sport_hours, 1),
        unit="uur/week"
    ))
    
    # 3. Flow Sessions (≥90 min deep work sessions)
    from .config import FLOW_MIN_DURATION_HOURS
    flow_sessions = deep_work_df[deep_work_df['duration_hours'] >= FLOW_MIN_DURATION_HOURS]
    flow_count = len(flow_sessions)
    progress.append(GoalProgress(
        name="flow_sessions",
        label="Flow Sessies",
        target=float(goals.min_flow_sessions_weekly),
        current=float(flow_count),
        unit="sessies/week"
    ))
    
    # 4. Entertainment (max goal - lower is better)
    entertainment_df = week_df[week_df['activity_type'].apply(lambda x: matches_categories(x, ['Entertainment', 'Internet']))]
    entertainment_hours = entertainment_df['duration_hours'].sum()
    progress.append(GoalProgress(
        name="entertainment",
        label="Entertainment",
        target=goals.max_entertainment_hours_weekly,
        current=round(entertainment_hours, 1),
        unit="uur/week",
        is_max=True
    ))
    
    # 5. Fragmentation (max goal - lower is better)
    from .metrics.advanced import calculate_fragmentation_index
    frag_index = calculate_fragmentation_index(week_df)
    avg_frag = sum(frag_index.values()) / len(frag_index) if frag_index else 0
    progress.append(GoalProgress(
        name="fragmentation",
        label="Fragmentatie",
        target=goals.max_fragmentation,
        current=round(avg_frag, 2),
        unit="sessies/uur",
        is_max=True
    ))
    
    # 6. Sleep (daily average)
    from .metrics.advanced import calculate_sleep_regularity_index
    sleep_stats = calculate_sleep_regularity_index(df)  # Use full df for better sleep estimation
    avg_sleep = sleep_stats.get('avg_sleep_hours', 0)
    progress.append(GoalProgress(
        name="sleep",
        label="Slaap",
        target=goals.sleep_hours_daily,
        current=round(avg_sleep, 1),
        unit="uur/nacht"
    ))
    
    return progress


def get_goals_summary(df: pd.DataFrame) -> dict:
    """
    Get a summary of goal progress for display.
    
    Returns:
        Dict with 'goals' list and 'overall_score' percentage
    """
    progress = calculate_goal_progress(df)
    
    if not progress:
        return {"goals": [], "overall_score": 0}
    
    # Calculate overall score (weighted average)
    total_pct = sum(g.percentage for g in progress)
    overall = total_pct / len(progress)
    
    return {
        "goals": [
            {
                "name": g.name,
                "label": g.label,
                "target": g.target,
                "current": g.current,
                "unit": g.unit,
                "percentage": round(g.percentage, 1),
                "status": g.status,
                "icon": g.icon,
                "is_max": g.is_max,
            }
            for g in progress
        ],
        "overall_score": round(overall, 1),
    }
