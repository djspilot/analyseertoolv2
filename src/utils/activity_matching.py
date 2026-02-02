"""
Activity type matching utilities.
Handles both simple format ("Work") and aTimeLogger format ("Work (Werk)").
"""

from typing import List


def extract_activity_name(activity_type: str) -> str:
    """Extract activity name from 'Activity (Category)' format.
    
    Examples:
        'Coding (Werk)' -> 'Coding'
        'Sport' -> 'Sport'
    """
    if " (" in activity_type:
        return activity_type.split(" (")[0]
    return activity_type


def matches_categories(activity_type: str, categories: List[str]) -> bool:
    """Check if activity type matches any of the given categories.
    
    Uses substring matching to handle both formats.
    
    Examples:
        matches_categories('Coding (Werk)', ['Work', 'Coding']) -> True
        matches_categories('Sport (Sport)', ['Work', 'Coding']) -> False
    """
    return any(cat in activity_type for cat in categories)


def map_to_weight(activity_type: str, weights: dict, default: float = 0) -> float:
    """Map activity type to a weight value.
    
    Tries direct match first, then extracts name from 'Activity (Category)' format.
    
    Examples:
        map_to_weight('Coding (Werk)', {'Coding': 100, 'Work': 75}, 10) -> 100
        map_to_weight('Unknown', {'Coding': 100}, 10) -> 10
    """
    if activity_type in weights:
        return weights[activity_type]
    name = extract_activity_name(activity_type)
    return weights.get(name, default)


def filter_by_categories(df, categories: List[str], column: str = 'activity_type'):
    """Filter DataFrame rows where activity matches any category.
    
    Args:
        df: pandas DataFrame
        categories: List of category names to match
        column: Column name containing activity types
    
    Returns:
        Filtered DataFrame (copy)
    """
    mask = df[column].apply(lambda x: matches_categories(x, categories))
    return df[mask].copy()
