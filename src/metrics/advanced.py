"""
Advanced metrics calculations for activity data.
Includes flow index, sleep regularity, and circadian profiles.
"""

import pandas as pd

from ..config import (
    DEEP_WORK_CATEGORIES,
    FLOW_MIN_DURATION_HOURS,
)
from ..logger import setup_logger
from ..utils import matches_categories

logger = setup_logger(__name__)


def calculate_fragmentation_index(df: pd.DataFrame) -> dict:
    """
    Calculate Fragmentation Index per category.
    FI = session_count / total_hours
    High FI = many short sessions (reactive), Low FI = consolidated blocks.

    Args:
        df: DataFrame with activity data

    Returns:
        Dict with category -> fragmentation index
    """
    fi = {}
    for category in df['activity_type'].unique():
        cat_df = df[df['activity_type'] == category]
        session_count = len(cat_df)
        total_hours = cat_df['duration_hours'].sum()
        fi[category] = round(session_count / total_hours, 2) if total_hours > 0 else 0
    return fi


def calculate_deep_work_ratio(df: pd.DataFrame, threshold_hours: float = None) -> float:
    """
    Calculate Deep Work Ratio (DWR).
    Hours in deep work sessions (Work/Coding >= threshold) / total tracked hours.

    Args:
        df: DataFrame with activity data
        threshold_hours: Minimum session length for deep work (defaults to config)

    Returns:
        Percentage of deep work time
    """
    if threshold_hours is None:
        threshold_hours = FLOW_MIN_DURATION_HOURS

    deep_work_df = df[
        df['activity_type'].apply(lambda x: matches_categories(x, DEEP_WORK_CATEGORIES)) &
        (df['duration_hours'] >= threshold_hours)
    ]
    deep_work_hours = deep_work_df['duration_hours'].sum()
    total_hours = df['duration_hours'].sum()
    return round((deep_work_hours / total_hours) * 100, 1) if total_hours > 0 else 0


def calculate_context_switching_penalty(df: pd.DataFrame) -> dict:
    """
    Calculate enhanced context switching metrics.

    Args:
        df: DataFrame sorted by datetime_from

    Returns:
        Dict with total switches, daily average, and switches per hour
    """
    df_sorted = df.sort_values('datetime_from').copy()
    df_sorted['date'] = df_sorted['datetime_from'].dt.date

    df_sorted['prev_activity'] = df_sorted.groupby('date')['activity_type'].shift(1)
    df_sorted['is_switch'] = (
        (df_sorted['activity_type'] != df_sorted['prev_activity']) &
        df_sorted['prev_activity'].notna()
    )

    total_switches = df_sorted['is_switch'].sum()
    total_days = df_sorted['date'].nunique()
    total_hours = df['duration_hours'].sum()

    return {
        'total_switches': int(total_switches),
        'daily_average': round(total_switches / total_days, 1) if total_days > 0 else 0,
        'switches_per_hour': round(total_switches / total_hours, 2) if total_hours > 0 else 0
    }


def calculate_flow_index(df: pd.DataFrame, flow_threshold_hours: float = None) -> float:
    """
    Calculate Flow Index.
    Percentage of deep work time in sessions >= flow threshold (90 min).

    Args:
        df: DataFrame with activity data
        flow_threshold_hours: Minimum session length for flow state (defaults to config)

    Returns:
        Flow index as percentage
    """
    if flow_threshold_hours is None:
        flow_threshold_hours = FLOW_MIN_DURATION_HOURS

    deep_work_df = df[df['activity_type'].apply(lambda x: matches_categories(x, DEEP_WORK_CATEGORIES))]

    if deep_work_df.empty:
        return 0.0

    total_deep_hours = deep_work_df['duration_hours'].sum()
    flow_sessions = deep_work_df[deep_work_df['duration_hours'] >= flow_threshold_hours]
    flow_hours = flow_sessions['duration_hours'].sum()

    return round((flow_hours / total_deep_hours) * 100, 1) if total_deep_hours > 0 else 0


def infer_sleep_from_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer sleep duration via negative space analysis.
    Find largest gap between last activity of day D and first of day D+1.

    Args:
        df: DataFrame with activity data

    Returns:
        DataFrame with date, sleep_start, wake_time, sleep_hours
    """
    df_sorted = df.sort_values('datetime_from').copy()
    df_sorted['date'] = df_sorted['datetime_from'].dt.date

    daily_bounds = df_sorted.groupby('date').agg({
        'datetime_from': 'min',
        'datetime_to': 'max'
    }).reset_index()
    daily_bounds.columns = ['date', 'first_activity', 'last_activity']
    daily_bounds = daily_bounds.sort_values('date')

    sleep_data = []
    dates = daily_bounds['date'].tolist()

    for i in range(len(dates) - 1):
        current_day = daily_bounds[daily_bounds['date'] == dates[i]].iloc[0]
        next_day = daily_bounds[daily_bounds['date'] == dates[i + 1]].iloc[0]

        sleep_start = current_day['last_activity']
        wake_time = next_day['first_activity']
        gap_hours = (wake_time - sleep_start).total_seconds() / 3600

        sleep_data.append({
            'date': dates[i],
            'sleep_start': sleep_start,
            'wake_time': wake_time,
            'sleep_hours': round(gap_hours, 2)
        })

    return pd.DataFrame(sleep_data)


def calculate_sleep_regularity_index(df: pd.DataFrame) -> dict:
    """
    Calculate Sleep Regularity Index (SRI).
    Based on standard deviation of sleep start times and wake times.
    Lower values = more consistent sleep schedule.

    Args:
        df: DataFrame with activity data

    Returns:
        Dict with sleep_start_std, wake_time_std, and combined SRI
    """
    sleep_df = infer_sleep_from_gaps(df)

    if sleep_df.empty:
        return {'sleep_start_std': 0, 'wake_time_std': 0, 'sri': 0}

    sleep_df['sleep_start_hour'] = (
        sleep_df['sleep_start'].dt.hour +
        sleep_df['sleep_start'].dt.minute / 60
    )
    sleep_df['wake_hour'] = (
        sleep_df['wake_time'].dt.hour +
        sleep_df['wake_time'].dt.minute / 60
    )

    sleep_start_std = sleep_df['sleep_start_hour'].std()
    wake_std = sleep_df['wake_hour'].std()

    return {
        'sleep_start_std': round(sleep_start_std, 2) if pd.notna(sleep_start_std) else 0,
        'wake_time_std': round(wake_std, 2) if pd.notna(wake_std) else 0,
        'sri': round((sleep_start_std + wake_std) / 2, 2) if pd.notna(sleep_start_std) else 0,
        'avg_sleep_hours': round(sleep_df['sleep_hours'].mean(), 2) if not sleep_df.empty else 0
    }


def get_hourly_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate hour-by-hour activity distribution for circadian analysis.

    Args:
        df: DataFrame with activity data

    Returns:
        DataFrame with hour (0-23), activity_type, and total_hours
    """
    df = df.copy()
    df['hour'] = df['datetime_from'].dt.hour

    hourly = df.groupby(['hour', 'activity_type'])['duration_hours'].sum().reset_index()
    hourly.columns = ['hour', 'activity_type', 'total_hours']

    pivot = hourly.pivot(index='hour', columns='activity_type', values='total_hours').fillna(0)
    pivot = pivot.reindex(range(24), fill_value=0)

    return pivot.reset_index()


def get_circadian_profile(df: pd.DataFrame) -> dict:
    """
    Identify peak hours for each activity type.

    Args:
        df: DataFrame with activity data

    Returns:
        Dict with activity_type -> peak_hour
    """
    df = df.copy()
    df['hour'] = df['datetime_from'].dt.hour

    hourly = df.groupby(['hour', 'activity_type'])['duration_hours'].sum().reset_index()

    peak_hours = {}
    for activity in df['activity_type'].unique():
        activity_hourly = hourly[hourly['activity_type'] == activity]
        if not activity_hourly.empty:
            peak_row = activity_hourly.loc[activity_hourly['duration_hours'].idxmax()]
            peak_hours[activity] = {
                'peak_hour': int(peak_row['hour']),
                'hours_at_peak': round(peak_row['duration_hours'], 2)
            }

    return peak_hours


def calculate_advanced_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate all advanced metrics in one call.

    Args:
        df: DataFrame with activity data

    Returns:
        Dict containing all advanced metrics:
        - fragmentation_index: per-category FI
        - deep_work_ratio: percentage of deep work time
        - context_switching: enhanced switch metrics
        - flow_index: percentage of flow-state time
        - sleep_regularity: SRI metrics
        - circadian_profile: peak hours per activity
    """
    return {
        'fragmentation_index': calculate_fragmentation_index(df),
        'deep_work_ratio': calculate_deep_work_ratio(df),
        'context_switching': calculate_context_switching_penalty(df),
        'flow_index': calculate_flow_index(df),
        'sleep_regularity': calculate_sleep_regularity_index(df),
        'circadian_profile': get_circadian_profile(df)
    }
