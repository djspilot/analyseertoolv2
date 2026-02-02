"""
Comparison module for flexible period analysis.
Enables comparing two time periods (A vs B) with delta metrics and presets.
"""

import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any


@dataclass
class DateRange:
    """Represents a date range with start and end dates."""
    start: datetime
    end: datetime
    label: str = ""
    
    @property
    def days(self) -> int:
        """Number of days in the range."""
        return (self.end - self.start).days + 1
    
    def contains(self, date: datetime) -> bool:
        """Check if a date falls within this range."""
        return self.start <= date <= self.end


@dataclass 
class ComparisonResult:
    """Result of comparing two periods."""
    period_a: DateRange
    period_b: DateRange
    metrics_a: Dict[str, Any]
    metrics_b: Dict[str, Any]
    deltas: Dict[str, float]  # Percentage changes
    

def get_comparison_presets() -> List[Tuple[str, DateRange, DateRange]]:
    """
    Get predefined comparison presets.
    
    Returns:
        List of tuples: (label, period_a, period_b)
    """
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Calculate week boundaries (Monday = 0)
    days_since_monday = today.weekday()
    this_week_start = today - timedelta(days=days_since_monday)
    last_week_start = this_week_start - timedelta(days=7)
    last_week_end = this_week_start - timedelta(days=1)
    
    # Calculate month boundaries
    this_month_start = today.replace(day=1)
    last_month_end = this_month_start - timedelta(days=1)
    last_month_start = last_month_end.replace(day=1)
    
    presets = [
        (
            "Deze week vs Vorige week",
            DateRange(this_week_start, today, "Deze week"),
            DateRange(last_week_start, last_week_end, "Vorige week")
        ),
        (
            "Laatste 7 dagen vs 7 dagen daarvoor",
            DateRange(today - timedelta(days=6), today, "Laatste 7 dagen"),
            DateRange(today - timedelta(days=13), today - timedelta(days=7), "7 dagen daarvoor")
        ),
        (
            "Laatste 14 dagen vs 14 dagen daarvoor",
            DateRange(today - timedelta(days=13), today, "Laatste 14 dagen"),
            DateRange(today - timedelta(days=27), today - timedelta(days=14), "14 dagen daarvoor")
        ),
        (
            "Deze maand vs Vorige maand",
            DateRange(this_month_start, today, "Deze maand"),
            DateRange(last_month_start, last_month_end, "Vorige maand")
        ),
        (
            "Laatste 30 dagen vs 30 dagen daarvoor",
            DateRange(today - timedelta(days=29), today, "Laatste 30 dagen"),
            DateRange(today - timedelta(days=59), today - timedelta(days=30), "30 dagen daarvoor")
        ),
    ]
    
    return presets


def filter_by_period(df: pd.DataFrame, period: DateRange) -> pd.DataFrame:
    """
    Filter DataFrame to only include activities within the given period.
    
    Args:
        df: DataFrame with 'datetime_from' column
        period: DateRange to filter by
        
    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df
    
    mask = (
        (df['datetime_from'].dt.date >= period.start.date()) & 
        (df['datetime_from'].dt.date <= period.end.date())
    )
    return df[mask].copy()


def calculate_period_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for a period.
    
    Args:
        df: DataFrame with activity data (should have metrics calculated)
        
    Returns:
        Dictionary with all relevant metrics
    """
    if df.empty:
        return {
            'total_hours': 0,
            'deep_work_hours': 0,
            'deep_work_pct': 0,
            'avg_daily_hours': 0,
            'total_activities': 0,
            'unique_categories': 0,
            'avg_session_length': 0,
            'context_switches': 0,
            'fragmented_sessions': 0,
            'category_breakdown': {}
        }
    
    total_hours = df['duration_hours'].sum()
    deep_work_df = df[df['is_deep_work'] == 1] if 'is_deep_work' in df.columns else pd.DataFrame()
    deep_work_hours = deep_work_df['duration_hours'].sum() if not deep_work_df.empty else 0
    deep_work_pct = (deep_work_hours / total_hours * 100) if total_hours > 0 else 0
    
    # Days calculation
    if 'date' in df.columns:
        num_days = df['date'].nunique()
    else:
        date_range = (df['datetime_from'].max() - df['datetime_from'].min()).days + 1
        num_days = max(date_range, 1)
    
    avg_daily_hours = total_hours / num_days if num_days > 0 else 0
    
    # Category breakdown
    category_breakdown = df.groupby('activity_type')['duration_hours'].sum().to_dict()
    
    # Context switches
    df_sorted = df.sort_values('datetime_from')
    if 'date' not in df_sorted.columns:
        df_sorted['date'] = df_sorted['datetime_from'].dt.date
    df_sorted['prev_activity'] = df_sorted.groupby('date')['activity_type'].shift(1)
    df_sorted['is_switch'] = (
        (df_sorted['activity_type'] != df_sorted['prev_activity']) & 
        df_sorted['prev_activity'].notna()
    )
    context_switches = df_sorted['is_switch'].sum()
    
    # Fragmented sessions (< 15 min)
    fragmented = (df['duration_hours'] < 0.25).sum() if 'duration_hours' in df.columns else 0
    
    return {
        'total_hours': round(total_hours, 2),
        'deep_work_hours': round(deep_work_hours, 2),
        'deep_work_pct': round(deep_work_pct, 1),
        'avg_daily_hours': round(avg_daily_hours, 2),
        'total_activities': len(df),
        'unique_categories': df['activity_type'].nunique(),
        'avg_session_length': round(total_hours / len(df), 2) if len(df) > 0 else 0,
        'context_switches': int(context_switches),
        'fragmented_sessions': int(fragmented),
        'category_breakdown': category_breakdown
    }


def calculate_delta(value_a: float, value_b: float) -> float:
    """
    Calculate percentage change from B to A.
    Positive means A is higher than B.
    
    Args:
        value_a: Current/new value
        value_b: Previous/old value (baseline)
        
    Returns:
        Percentage change (e.g., 15.5 means +15.5%)
    """
    if value_b == 0:
        return 100.0 if value_a > 0 else 0.0
    return round(((value_a - value_b) / value_b) * 100, 1)


def calculate_delta_metrics(metrics_a: Dict[str, Any], metrics_b: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate percentage deltas between two sets of metrics.
    
    Args:
        metrics_a: Metrics for period A
        metrics_b: Metrics for period B (baseline)
        
    Returns:
        Dictionary with delta percentages for each numeric metric
    """
    numeric_keys = [
        'total_hours', 'deep_work_hours', 'deep_work_pct', 
        'avg_daily_hours', 'total_activities', 'avg_session_length',
        'context_switches', 'fragmented_sessions'
    ]
    
    deltas = {}
    for key in numeric_keys:
        val_a = metrics_a.get(key, 0)
        val_b = metrics_b.get(key, 0)
        deltas[key] = calculate_delta(val_a, val_b)
    
    return deltas


def compare_periods(
    df: pd.DataFrame, 
    period_a: DateRange, 
    period_b: DateRange
) -> ComparisonResult:
    """
    Compare two time periods and return comprehensive comparison result.
    
    Args:
        df: Full DataFrame with all activity data
        period_a: First period (typically current/recent)
        period_b: Second period (typically previous/baseline)
        
    Returns:
        ComparisonResult with metrics and deltas
    """
    df_a = filter_by_period(df, period_a)
    df_b = filter_by_period(df, period_b)
    
    metrics_a = calculate_period_metrics(df_a)
    metrics_b = calculate_period_metrics(df_b)
    
    deltas = calculate_delta_metrics(metrics_a, metrics_b)
    
    return ComparisonResult(
        period_a=period_a,
        period_b=period_b,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        deltas=deltas
    )


def get_category_comparison(
    df: pd.DataFrame, 
    period_a: DateRange, 
    period_b: DateRange
) -> pd.DataFrame:
    """
    Get per-category comparison between two periods.
    
    Returns:
        DataFrame with category, hours_a, hours_b, delta_pct columns
    """
    df_a = filter_by_period(df, period_a)
    df_b = filter_by_period(df, period_b)
    
    cat_a = df_a.groupby('activity_type')['duration_hours'].sum()
    cat_b = df_b.groupby('activity_type')['duration_hours'].sum()
    
    # Combine all categories
    all_categories = set(cat_a.index) | set(cat_b.index)
    
    comparison_data = []
    for cat in all_categories:
        hours_a = cat_a.get(cat, 0)
        hours_b = cat_b.get(cat, 0)
        delta = calculate_delta(hours_a, hours_b)
        
        comparison_data.append({
            'category': cat,
            'hours_a': round(hours_a, 2),
            'hours_b': round(hours_b, 2),
            'delta_pct': delta
        })
    
    result = pd.DataFrame(comparison_data)
    return result.sort_values('hours_a', ascending=False)
