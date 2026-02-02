"""
Basic metrics calculations for activity data.
Includes deep work flags, fragmentation risk, and daily summaries.
"""

import pandas as pd

from ..config import (
    DEEP_WORK_CATEGORIES,
    DEEP_WORK_MIN_DURATION_HOURS,
    FRAGMENTATION_RISK_THRESHOLD_HOURS,
)
from ..logger import setup_logger

logger = setup_logger(__name__)


def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate derived metrics for the activity data.

    Adds:
        - is_deep_work: True if activity is Work/Coding AND duration >= DEEP_WORK_MIN_DURATION_HOURS
        - fragmentation_risk: True if duration < FRAGMENTATION_RISK_THRESHOLD_HOURS
        - date: Just the date part for grouping
        - weekday: Day of week (0=Monday, 6=Sunday)

    Args:
        df: DataFrame with activity data

    Returns:
        DataFrame with added metric columns
    """
    df = df.copy()

    # Deep work: Work or Coding with duration >= threshold
    # Activity types are stored as "Activity (Category)", so check if any deep work category is in the name
    df['is_deep_work'] = (
        df['activity_type'].apply(lambda x: any(cat in x for cat in DEEP_WORK_CATEGORIES)) &
        (df['duration_hours'] >= DEEP_WORK_MIN_DURATION_HOURS)
    ).astype(int)

    # Fragmentation risk: short activities
    df['fragmentation_risk'] = (
        df['duration_hours'] < FRAGMENTATION_RISK_THRESHOLD_HOURS
    ).astype(int)

    # Date columns for grouping
    df['date'] = df['datetime_from'].dt.date
    df['weekday'] = df['datetime_from'].dt.dayofweek
    df['week'] = df['datetime_from'].dt.isocalendar().week

    logger.debug(f"Calculated metrics for {len(df)} activities")

    return df


def calculate_circadian_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily circadian metrics (start time, end time).

    Args:
        df: DataFrame with activity data

    Returns:
        DataFrame with daily start/end times and duration span
    """
    # Group by date
    daily = df.groupby('date').agg({
        'datetime_from': 'min',
        'datetime_to': 'max'
    }).reset_index()

    # Calculate start/end hours as float (e.g., 08:30 -> 8.5)
    daily['start_hour'] = daily['datetime_from'].dt.hour + daily['datetime_from'].dt.minute / 60
    daily['end_hour'] = daily['datetime_to'].dt.hour + daily['datetime_to'].dt.minute / 60
    daily['day_span_hours'] = daily['end_hour'] - daily['start_hour']

    return daily


def calculate_consistency_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate consistency metrics like context switches.

    Args:
        df: DataFrame sorted by datetime_from

    Returns:
        DataFrame with daily switch counts
    """
    df_sorted = df.sort_values('datetime_from')

    # Count switches: where current activity type != previous activity type
    # Group by date to count switches per day
    df_sorted['prev_activity'] = df_sorted.groupby('date')['activity_type'].shift(1)
    df_sorted['is_switch'] = (df_sorted['activity_type'] != df_sorted['prev_activity']) & df_sorted['prev_activity'].notna()

    daily_switches = df_sorted.groupby('date')['is_switch'].sum().reset_index(name='context_switches')

    return daily_switches


def get_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a daily summary of activities.

    Args:
        df: DataFrame with activity data (should have metrics calculated)

    Returns:
        DataFrame with daily aggregations
    """
    daily = df.groupby('date').agg({
        'duration_hours': 'sum',
        'is_deep_work': lambda x: df.loc[x.index, 'duration_hours'][x == 1].sum(),
        'fragmentation_risk': 'sum',
        'activity_type': 'count'
    }).rename(columns={
        'duration_hours': 'total_hours',
        'is_deep_work': 'deep_work_hours',
        'fragmentation_risk': 'fragmented_count',
        'activity_type': 'activity_count'
    })

    daily['deep_work_pct'] = (daily['deep_work_hours'] / daily['total_hours'] * 100).round(1)

    return daily.reset_index()


def get_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary by activity category.

    Args:
        df: DataFrame with activity data

    Returns:
        DataFrame with category aggregations
    """
    summary = df.groupby('activity_type').agg({
        'duration_hours': 'sum',
        'activity_type': 'count'
    }).rename(columns={
        'duration_hours': 'total_hours',
        'activity_type': 'count'
    })

    total_hours = summary['total_hours'].sum()
    summary['percentage'] = (summary['total_hours'] / total_hours * 100).round(1)

    return summary.reset_index().sort_values('total_hours', ascending=False)
