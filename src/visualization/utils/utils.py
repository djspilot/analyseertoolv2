"""
Utility functions for chart creation.
Common helpers used across all visualization modules.
"""

import pandas as pd
from typing import List, Optional

from ...config import CATEGORY_COLORS


def get_color(category: str) -> str:
    """Get color for a category, with fallback to grey.
    
    Handles both formats:
    - Simple: "Work", "Coding"
    - aTimeLogger: "Coding (Werk)", "Sport (Sport)"
    """
    # Direct match
    if category in CATEGORY_COLORS:
        return CATEGORY_COLORS[category]
    
    # Extract activity name from "Activity (Category)" format
    if " (" in category:
        activity_name = category.split(" (")[0]
        if activity_name in CATEGORY_COLORS:
            return CATEGORY_COLORS[activity_name]
    
    return '#9E9E9E'


def ensure_datetime(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Ensure specified columns are datetime type.

    Args:
        df: DataFrame to process
        columns: List of column names to convert. Defaults to ['datetime_from', 'datetime_to']

    Returns:
        DataFrame with converted columns (copy)
    """
    if columns is None:
        columns = ['datetime_from']

    df = df.copy()
    for col in columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = pd.to_datetime(df[col])

    return df


def build_color_map(activity_types: List[str]) -> dict:
    """Build a color map for activity types, handling 'Activity (Category)' format."""
    return {act: get_color(act) for act in activity_types}


def filter_deep_work(df: pd.DataFrame, categories: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Filter DataFrame to only include deep work categories.

    Args:
        df: DataFrame with activity_type column
        categories: Categories to include. Defaults to ['Work', 'Coding']

    Returns:
        Filtered DataFrame (copy)
    """
    if categories is None:
        categories = ['Work', 'Coding']

    # Handle both "Work" and "Work (Werk)" formats
    mask = df['activity_type'].apply(
        lambda x: any(cat in x for cat in categories)
    )
    return df[mask].copy()
