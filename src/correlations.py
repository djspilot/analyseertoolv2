"""
Correlation analysis for time tracking data.
Discovers relationships between activities and their effects on performance.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Optional
from dataclasses import dataclass

from .config import DEEP_WORK_CATEGORIES, FLOW_MIN_DURATION_HOURS
from .utils import matches_categories


@dataclass
class Correlation:
    """A correlation between two activities/metrics."""
    source: str
    target: str
    correlation: float
    lag_days: int
    interpretation: str
    strength: str  # "strong", "moderate", "weak"
    direction: str  # "positive", "negative"
    
    @property
    def icon(self) -> str:
        if self.strength == "strong":
            return "🔥" if self.direction == "positive" else "⚠️"
        elif self.strength == "moderate":
            return "👍" if self.direction == "positive" else "👎"
        return "➡️"


def calculate_activity_correlations(df: pd.DataFrame, min_days: int = 14) -> list[Correlation]:
    """
    Calculate correlations between activities.
    
    Analyzes:
    - Same-day correlations (e.g., Sport → better Work focus)
    - Next-day correlations (e.g., Good sleep → more deep work)
    - Weekly patterns
    
    Args:
        df: DataFrame with activity data
        min_days: Minimum days of data required
    
    Returns:
        List of significant correlations, sorted by strength
    """
    if df.empty:
        return []
    
    # Check if we have enough data
    unique_days = df['datetime_from'].dt.date.nunique()
    if unique_days < min_days:
        return []
    
    correlations = []
    
    # Create daily aggregates
    daily = _create_daily_aggregates(df)
    
    if len(daily) < min_days:
        return []
    
    # Define activity groups for correlation
    activity_pairs = [
        # (source activity, target metric, lag_days, interpretation template)
        ("Sport", "deep_work_hours", 1, "Sport → {dir} deep work de volgende dag"),
        ("Sport", "flow_sessions", 1, "Sport → {dir} flow sessies de volgende dag"),
        ("Yoga", "deep_work_hours", 0, "Yoga → {dir} deep work dezelfde dag"),
        ("Yoga", "deep_work_hours", 1, "Yoga → {dir} deep work de volgende dag"),
        ("sleep_hours", "deep_work_hours", 0, "Slaap → {dir} deep work"),
        ("sleep_hours", "flow_sessions", 0, "Slaap → {dir} flow sessies"),
        ("Entertainment", "deep_work_hours", 1, "Entertainment → {dir} deep work de volgende dag"),
        ("Internet", "deep_work_hours", 0, "Internet → {dir} deep work dezelfde dag"),
        ("deep_work_hours", "sleep_hours", 0, "Deep work → {dir} slaap dezelfde nacht"),
        ("total_hours", "sleep_hours", 0, "Totale activiteit → {dir} slaap"),
    ]
    
    for source, target, lag, interpretation_template in activity_pairs:
        if source not in daily.columns or target not in daily.columns:
            continue
        
        # Apply lag
        if lag > 0:
            source_series = daily[source].iloc[:-lag]
            target_series = daily[target].iloc[lag:]
        else:
            source_series = daily[source]
            target_series = daily[target]
        
        # Calculate correlation
        if len(source_series) < 10 or source_series.std() == 0 or target_series.std() == 0:
            continue
        
        corr = source_series.corr(target_series)
        
        if pd.isna(corr):
            continue
        
        # Only include meaningful correlations
        abs_corr = abs(corr)
        if abs_corr < 0.2:
            continue
        
        # Determine strength and direction
        if abs_corr >= 0.5:
            strength = "strong"
        elif abs_corr >= 0.3:
            strength = "moderate"
        else:
            strength = "weak"
        
        direction = "positive" if corr > 0 else "negative"
        dir_text = "meer" if corr > 0 else "minder"
        interpretation = interpretation_template.format(dir=dir_text)
        
        correlations.append(Correlation(
            source=source,
            target=target,
            correlation=round(corr, 3),
            lag_days=lag,
            interpretation=interpretation,
            strength=strength,
            direction=direction,
        ))
    
    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x.correlation), reverse=True)
    
    return correlations


def _create_daily_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Create daily aggregates for correlation analysis."""
    df = df.copy()
    df['date'] = df['datetime_from'].dt.date
    
    # Basic aggregates per day
    daily = df.groupby('date').agg({
        'duration_hours': 'sum',
    }).rename(columns={'duration_hours': 'total_hours'})
    
    # Deep work hours per day
    deep_work = df[df['activity_type'].apply(lambda x: matches_categories(x, DEEP_WORK_CATEGORIES))]
    deep_work_daily = deep_work.groupby(deep_work['datetime_from'].dt.date)['duration_hours'].sum()
    daily['deep_work_hours'] = deep_work_daily.reindex(daily.index).fillna(0)
    
    # Flow sessions per day (sessions >= 90 min)
    flow_sessions = deep_work[deep_work['duration_hours'] >= FLOW_MIN_DURATION_HOURS]
    flow_daily = flow_sessions.groupby(flow_sessions['datetime_from'].dt.date).size()
    daily['flow_sessions'] = flow_daily.reindex(daily.index).fillna(0)
    
    # Activity-specific hours
    for activity in ['Sport', 'Yoga', 'Entertainment', 'Internet', 'Work', 'Coding']:
        activity_df = df[df['activity_type'] == activity]
        if not activity_df.empty:
            activity_daily = activity_df.groupby(activity_df['datetime_from'].dt.date)['duration_hours'].sum()
            daily[activity] = activity_daily.reindex(daily.index).fillna(0)
        else:
            daily[activity] = 0
    
    # Estimate sleep hours (from gaps in activity)
    daily['sleep_hours'] = _estimate_daily_sleep(df)
    
    return daily


def _estimate_daily_sleep(df: pd.DataFrame) -> pd.Series:
    """Estimate sleep hours per day from activity gaps."""
    df = df.copy()
    df = df.sort_values('datetime_from')
    df['date'] = df['datetime_from'].dt.date
    
    sleep_hours = {}
    
    for date in df['date'].unique():
        day_df = df[df['date'] == date]
        if day_df.empty:
            continue
        
        # Look for the longest gap (likely sleep)
        day_df = day_df.sort_values('datetime_from')
        
        # Calculate gaps between activities
        gaps = []
        for i in range(1, len(day_df)):
            prev_end = day_df.iloc[i-1]['datetime_to']
            curr_start = day_df.iloc[i]['datetime_from']
            gap = (curr_start - prev_end).total_seconds() / 3600
            if gap > 0:
                gaps.append(gap)
        
        # Also consider gap from last activity to midnight and midnight to first activity
        first_activity = day_df.iloc[0]['datetime_from']
        last_activity = day_df.iloc[-1]['datetime_to']
        
        # Hours from midnight to first activity
        hours_before = first_activity.hour + first_activity.minute / 60
        # Hours from last activity to midnight
        hours_after = 24 - (last_activity.hour + last_activity.minute / 60)
        
        # Estimate sleep as max gap or overnight gap
        overnight = hours_before + hours_after
        max_gap = max(gaps) if gaps else 0
        
        # Sleep is likely the larger of overnight or longest daytime gap
        sleep_hours[date] = max(overnight, max_gap, 6)  # Minimum 6 hours assumed
    
    return pd.Series(sleep_hours)


def get_correlation_insights(df: pd.DataFrame) -> list[dict]:
    """
    Get correlation insights formatted for display.
    
    Returns:
        List of insight dicts ready for UI display
    """
    correlations = calculate_activity_correlations(df)
    
    if not correlations:
        return [{
            "icon": "📊",
            "title": "Onvoldoende data",
            "message": "Er zijn minimaal 14 dagen data nodig voor correlatie-analyse.",
            "strength": "neutral",
            "correlation": 0,
        }]
    
    insights = []
    
    for corr in correlations[:8]:  # Top 8 correlations
        insights.append({
            "icon": corr.icon,
            "title": corr.interpretation,
            "message": f"Correlatie: {corr.correlation:+.2f} ({corr.strength})",
            "strength": corr.strength,
            "direction": corr.direction,
            "correlation": corr.correlation,
            "source": corr.source,
            "target": corr.target,
            "lag_days": corr.lag_days,
        })
    
    return insights


def get_correlation_summary(df: pd.DataFrame) -> dict:
    """
    Get a summary of correlations for the AI to use.
    
    Returns:
        Dict with key findings for LLM context
    """
    correlations = calculate_activity_correlations(df)
    
    if not correlations:
        return {"has_data": False, "correlations": []}
    
    # Group by positive/negative impact on deep work
    positive_for_deep_work = [c for c in correlations 
                              if c.target == "deep_work_hours" and c.direction == "positive"]
    negative_for_deep_work = [c for c in correlations 
                              if c.target == "deep_work_hours" and c.direction == "negative"]
    
    return {
        "has_data": True,
        "total_correlations": len(correlations),
        "positive_factors": [
            {"activity": c.source, "correlation": c.correlation, "lag": c.lag_days}
            for c in positive_for_deep_work[:3]
        ],
        "negative_factors": [
            {"activity": c.source, "correlation": c.correlation, "lag": c.lag_days}
            for c in negative_for_deep_work[:3]
        ],
        "strongest": {
            "interpretation": correlations[0].interpretation,
            "correlation": correlations[0].correlation,
        } if correlations else None,
    }
