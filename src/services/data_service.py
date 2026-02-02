"""
Data service - Centralized data operations with caching.
Provides a single source of truth for data access across all frontends.
"""

import pandas as pd
from typing import Optional
from datetime import datetime, timedelta
import time

from ..api.database import get_all_activities, init_db
from ..api.cache import cache_chart
from ..processor import calculate_metrics
from ..logger import setup_logger

logger = setup_logger(__name__)


class DataService:
    """
    Service layer for data operations with built-in caching.
    Eliminates duplicate code across multiple frontends.
    """
    
    def __init__(self):
        self._df_cache: Optional[pd.DataFrame] = None
        self._cache_timestamp: Optional[float] = None
        self._cache_ttl_seconds = 300  # 5 minutes cache
    
    def get_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get all activity data with caching.
        
        Args:
            force_refresh: Bypass cache and reload from database
            
        Returns:
            DataFrame with all activities including calculated metrics
        """
        current_time = time.time()
        
        # Check if cache is valid
        if not force_refresh and self._df_cache is not None:
            if self._cache_timestamp and (current_time - self._cache_timestamp) < self._cache_ttl_seconds:
                logger.debug("Returning cached data")
                return self._df_cache
        
        # Load from database
        logger.info("Loading data from database")
        init_db()
        df = get_all_activities()
        
        if not df.empty:
            df = calculate_metrics(df)
        
        # Update cache
        self._df_cache = df
        self._cache_timestamp = current_time
        
        return df
    
    def get_filtered_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        activity_types: Optional[list[str]] = None,
        deep_work_only: bool = False
    ) -> pd.DataFrame:
        """
        Get filtered data based on criteria.
        
        Args:
            start_date: Start date filter
            end_date: End date filter
            activity_types: List of activity types to include
            deep_work_only: Only return deep work activities
            
        Returns:
            Filtered DataFrame
        """
        df = self.get_data()
        
        if df.empty:
            return df
        
        # Date range filter
        if start_date:
            df = df[df['datetime_from'] >= start_date]
        if end_date:
            df = df[df['datetime_from'] <= end_date]
        
        # Activity type filter
        if activity_types:
            df = df[df['activity_type'].isin(activity_types)]
        
        # Deep work filter
        if deep_work_only:
            df = df[df['is_deep_work'] == 1]
        
        return df
    
    def get_date_range(self) -> tuple[datetime, datetime]:
        """
        Get the date range of available data.
        
        Returns:
            Tuple of (min_date, max_date)
        """
        df = self.get_data()
        
        if df.empty:
            return datetime.now(), datetime.now()
        
        return df['datetime_from'].min(), df['datetime_from'].max()
    
    def get_available_dates(self) -> list[datetime]:
        """
        Get list of all dates with activity data.
        
        Returns:
            Sorted list of unique dates
        """
        df = self.get_data()
        
        if df.empty:
            return []
        
        return sorted(df['date'].unique(), reverse=True)
    
    def get_data_by_date(self, date: datetime) -> pd.DataFrame:
        """
        Get all activities for a specific date.
        
        Args:
            date: Date to filter to
            
        Returns:
            DataFrame with activities for that date
        """
        df = self.get_data()
        
        if df.empty:
            return df
        
        return df[df['date'] == date.date()]
    
    def get_data_by_percentage_range(self, start_pct: float, end_pct: float) -> pd.DataFrame:
        """
        Get data based on percentage of total date range.
        Useful for time range sliders (0-100%).
        
        Args:
            start_pct: Start percentage (0-100)
            end_pct: End percentage (0-100)
            
        Returns:
            Filtered DataFrame
        """
        df = self.get_data()
        
        if df.empty:
            return df
        
        min_date = df['datetime_from'].min()
        max_date = df['datetime_from'].max()
        total_range = (max_date - min_date).total_seconds()
        
        start_offset = timedelta(seconds=total_range * start_pct / 100)
        end_offset = timedelta(seconds=total_range * end_pct / 100)
        
        filter_start = min_date + start_offset
        filter_end = min_date + end_offset
        
        filtered = df[(df['datetime_from'] >= filter_start) & 
                      (df['datetime_from'] <= filter_end)]
        
        # Fallback to all data if filter returns empty
        return filtered if not filtered.empty else df
    
    def get_total_days(self) -> int:
        """
        Get total number of days in dataset.
        
        Returns:
            Number of days
        """
        df = self.get_data()
        
        if df.empty:
            return 0
        
        min_date = df['datetime_from'].min()
        max_date = df['datetime_from'].max()
        
        return (max_date - min_date).days + 1
    
    def invalidate_cache(self):
        """Clear the data cache."""
        logger.info("Invalidating data cache")
        self._df_cache = None
        self._cache_timestamp = None


# Global singleton instance
_data_service = DataService()


def get_data_service() -> DataService:
    """
    Get the global DataService instance.
    
    Returns:
        DataService singleton
    """
    return _data_service