"""
Smart Data Aggregation Module
Provides intelligent data loading strategies to optimize performance.
Implements progressive loading and smart caching for large datasets.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Callable, Generator
import pandas as pd
import hashlib

from ..logger import setup_logger
from ..production import RASPBERRY_PI_MODE

logger = setup_logger(__name__)


@dataclass
class DataSlice:
    """A slice of data with metadata."""
    df: pd.DataFrame
    start_date: datetime
    end_date: datetime
    is_complete: bool  # Whether this is the full dataset
    total_records: int
    

@dataclass  
class AggregationLevel:
    """Defines an aggregation level."""
    name: str
    resample_rule: str  # 'D', 'W', 'M'
    max_points: int
    threshold_days: int  # Use this level if data spans > threshold days


# Aggregation levels (from finest to coarsest)
AGGREGATION_LEVELS = [
    AggregationLevel("raw", "H", 10000, 7),      # Raw hourly for < 1 week
    AggregationLevel("daily", "D", 1000, 90),    # Daily for < 3 months
    AggregationLevel("weekly", "W", 500, 365),   # Weekly for < 1 year
    AggregationLevel("monthly", "M", 200, 3650), # Monthly for < 10 years
]


def get_optimal_aggregation(df: pd.DataFrame, target_points: int = 500) -> str:
    """
    Determine optimal aggregation level based on data size.
    
    Args:
        df: Input DataFrame
        target_points: Target number of data points
        
    Returns:
        Resample rule ('D', 'W', 'M', or 'raw')
    """
    if df.empty:
        return 'D'
    
    date_range = (df['datetime_from'].max() - df['datetime_from'].min()).days
    
    for level in AGGREGATION_LEVELS:
        if date_range <= level.threshold_days:
            return level.resample_rule
    
    return 'M'  # Default to monthly for very long ranges


def smart_aggregate(
    df: pd.DataFrame,
    columns: list[str],
    agg_funcs: dict[str, str],
    target_points: int = 500,
) -> pd.DataFrame:
    """
    Intelligently aggregate DataFrame to target size.
    
    Args:
        df: Input DataFrame with datetime_from column
        columns: Columns to include in aggregation
        agg_funcs: Dict of column -> aggregation function ('sum', 'mean', 'count')
        target_points: Target number of output rows
        
    Returns:
        Aggregated DataFrame
    """
    if df.empty:
        return df
    
    current_points = len(df)
    
    if current_points <= target_points:
        return df
    
    # Determine aggregation level
    rule = get_optimal_aggregation(df, target_points)
    
    if rule == 'raw':
        return df
    
    # Perform aggregation
    df_indexed = df.set_index('datetime_from')
    
    agg_dict = {col: agg_funcs.get(col, 'sum') for col in columns if col in df.columns}
    
    aggregated = df_indexed.resample(rule).agg(agg_dict).reset_index()
    
    logger.debug(f"Aggregated {current_points} rows to {len(aggregated)} using rule '{rule}'")
    
    return aggregated


def progressive_load(
    load_func: Callable[[], pd.DataFrame],
    chunk_days: int = 30,
) -> Generator[DataSlice, None, None]:
    """
    Load data progressively in chunks, newest first.
    
    Yields DataSlice objects that can be used to update UI incrementally.
    
    Args:
        load_func: Function that loads all data
        chunk_days: Days per chunk
        
    Yields:
        DataSlice objects from newest to oldest
    """
    # Load all data first (required for proper date handling)
    full_df = load_func()
    
    if full_df.empty:
        yield DataSlice(
            df=full_df,
            start_date=datetime.now(),
            end_date=datetime.now(),
            is_complete=True,
            total_records=0,
        )
        return
    
    total_records = len(full_df)
    min_date = full_df['datetime_from'].min()
    max_date = full_df['datetime_from'].max()
    
    # Generate chunks from newest to oldest
    current_end = max_date
    accumulated_df = pd.DataFrame()
    
    while current_end > min_date:
        current_start = current_end - timedelta(days=chunk_days)
        if current_start < min_date:
            current_start = min_date
        
        chunk = full_df[
            (full_df['datetime_from'] >= current_start) &
            (full_df['datetime_from'] <= current_end)
        ]
        
        if accumulated_df.empty:
            accumulated_df = chunk
        else:
            accumulated_df = pd.concat([chunk, accumulated_df]).sort_values('datetime_from')
        
        is_complete = current_start <= min_date
        
        yield DataSlice(
            df=accumulated_df.copy(),
            start_date=current_start,
            end_date=max_date,
            is_complete=is_complete,
            total_records=total_records,
        )
        
        current_end = current_start - timedelta(days=1)


def get_data_fingerprint(df: pd.DataFrame) -> str:
    """
    Generate a fingerprint for the dataset to detect changes.
    
    Returns:
        MD5 hash of key data characteristics
    """
    if df.empty:
        return "empty"
    
    # Create fingerprint from key characteristics
    parts = [
        str(len(df)),
        str(df['datetime_from'].min()),
        str(df['datetime_from'].max()),
        str(df['duration_hours'].sum()),
        str(df['activity_type'].nunique()),
    ]
    
    return hashlib.md5("_".join(parts).encode()).hexdigest()[:12]


class SmartDataManager:
    """
    Manages data loading and caching with smart strategies.
    """
    
    def __init__(self, max_cache_mb: int = 100):
        self.max_cache_mb = max_cache_mb
        self._cache: dict[str, pd.DataFrame] = {}
        self._fingerprints: dict[str, str] = {}
        self._last_full_load: Optional[datetime] = None
        self._full_df: Optional[pd.DataFrame] = None
    
    def get_filtered(
        self,
        df: pd.DataFrame,
        start_pct: float,
        end_pct: float,
        optimize: bool = True,
    ) -> pd.DataFrame:
        """
        Get filtered data with optional optimization.
        
        Args:
            df: Full DataFrame
            start_pct: Start percentage (0-100)
            end_pct: End percentage (0-100)
            optimize: Whether to apply smart aggregation
            
        Returns:
            Filtered (and optionally aggregated) DataFrame
        """
        if df.empty:
            return df
        
        min_date = df['datetime_from'].min()
        max_date = df['datetime_from'].max()
        total_range = (max_date - min_date).total_seconds()
        
        start_offset = timedelta(seconds=total_range * start_pct / 100)
        end_offset = timedelta(seconds=total_range * end_pct / 100)
        
        filter_start = min_date + start_offset
        filter_end = min_date + end_offset
        
        filtered = df[
            (df['datetime_from'] >= filter_start) &
            (df['datetime_from'] <= filter_end)
        ]
        
        if optimize and RASPBERRY_PI_MODE:
            # Aggressive downsampling for Pi
            max_points = 300
        elif optimize:
            max_points = 500
        else:
            return filtered
        
        if len(filtered) > max_points:
            # Sample evenly
            step = len(filtered) // max_points
            filtered = filtered.iloc[::step]
        
        return filtered
    
    def invalidate_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        self._fingerprints.clear()
        self._full_df = None
        logger.info("Data cache invalidated")


# Singleton instance
_data_manager: Optional[SmartDataManager] = None


def get_data_manager() -> SmartDataManager:
    """Get or create the global data manager."""
    global _data_manager
    if _data_manager is None:
        _data_manager = SmartDataManager()
    return _data_manager
