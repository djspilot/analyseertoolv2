"""
Metrics service - Centralized metrics calculations.
Provides consistent metrics across all frontends.
"""

import pandas as pd
from typing import Optional
from datetime import datetime

from ..metrics import (
    calculate_metrics,
    calculate_circadian_metrics,
    calculate_consistency_metrics,
    get_daily_summary,
    get_category_summary,
    calculate_advanced_metrics,
)
from ..logger import setup_logger

logger = setup_logger(__name__)


class MetricsService:
    """
    Service layer for metrics calculations.
    Provides a unified interface for all metric computations.
    """
    
    def __init__(self):
        pass
    
    def calculate_all_metrics(self, df: pd.DataFrame) -> dict:
        """
        Calculate all available metrics for a dataset.
        
        Args:
            df: DataFrame with activity data
            
        Returns:
            Dictionary with all calculated metrics
        """
        if df.empty:
            return self._empty_metrics()
        
        # Basic metrics
        total_hours = round(df['duration_hours'].sum(), 1)
        deep_work_hours = round(df[df['is_deep_work'] == 1]['duration_hours'].sum(), 1)
        
        days = df['date'].nunique()
        daily_avg = round(total_hours / days, 1) if days > 0 else 0.0
        
        # Consistency metrics
        consistency = calculate_consistency_metrics(df)
        total_switches = int(consistency['context_switches'].sum())
        
        # Advanced metrics
        advanced = calculate_advanced_metrics(df)
        
        # Category summary
        categories = get_category_summary(df).to_dict('records')
        
        return {
            'total_hours': total_hours,
            'deep_work_hours': deep_work_hours,
            'daily_avg': daily_avg,
            'total_switches': total_switches,
            'deep_work_ratio': advanced['deep_work_ratio'],
            'flow_index': advanced['flow_index'],
            'avg_sleep': advanced['sleep_regularity']['avg_sleep_hours'],
            'sleep_regularity': advanced['sleep_regularity']['sri'],
            'categories': categories,
            'circadian': calculate_circadian_metrics(df),
            'daily_summary': get_daily_summary(df),
            'advanced': advanced,
            'consistency': consistency
        }
    
    def get_basic_metrics(self, df: pd.DataFrame) -> dict:
        """
        Get basic metrics only (faster calculation).
        
        Args:
            df: DataFrame with activity data
            
        Returns:
            Dictionary with basic metrics
        """
        if df.empty:
            return self._empty_basic_metrics()
        
        total_hours = round(df['duration_hours'].sum(), 1)
        deep_work_hours = round(df[df['is_deep_work'] == 1]['duration_hours'].sum(), 1)
        
        days = df['date'].nunique()
        daily_avg = round(total_hours / days, 1) if days > 0 else 0.0
        
        return {
            'total_hours': total_hours,
            'deep_work_hours': deep_work_hours,
            'daily_avg': daily_avg,
        }
    
    def get_advanced_metrics(self, df: pd.DataFrame) -> dict:
        """
        Get advanced metrics (flow, sleep, fragmentation).
        
        Args:
            df: DataFrame with activity data
            
        Returns:
            Dictionary with advanced metrics
        """
        if df.empty:
            return self._empty_advanced_metrics()
        
        consistency = calculate_consistency_metrics(df)
        advanced = calculate_advanced_metrics(df)
        
        return {
            'total_switches': int(consistency['context_switches'].sum()),
            'deep_work_ratio': advanced['deep_work_ratio'],
            'flow_index': advanced['flow_index'],
            'avg_sleep': advanced['sleep_regularity']['avg_sleep_hours'],
            'sleep_regularity': advanced['sleep_regularity']['sri'],
            'fragmentation_index': advanced.get('fragmentation_index', 0),
        }
    
    def get_category_metrics(self, df: pd.DataFrame) -> list[dict]:
        """
        Get time distribution by category.
        
        Args:
            df: DataFrame with activity data
            
        Returns:
            List of category metrics
        """
        if df.empty:
            return []
        
        cat_df = get_category_summary(df)
        return cat_df.to_dict('records')
    
    def get_circadian_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get circadian rhythm metrics (wake/sleep times).
        
        Args:
            df: DataFrame with activity data
            
        Returns:
            DataFrame with circadian metrics
        """
        if df.empty:
            return pd.DataFrame()
        
        return calculate_circadian_metrics(df)
    
    def get_daily_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get daily summary of activities.
        
        Args:
            df: DataFrame with activity data
            
        Returns:
            DataFrame with daily summaries
        """
        if df.empty:
            return pd.DataFrame()
        
        return get_daily_summary(df)
    
    def compare_periods(
        self,
        df: pd.DataFrame,
        period_a_start: datetime,
        period_a_end: datetime,
        period_b_start: datetime,
        period_b_end: datetime
    ) -> dict:
        """
        Compare metrics between two time periods.
        
        Args:
            df: DataFrame with activity data
            period_a_start: Start of period A
            period_a_end: End of period A
            period_b_start: Start of period B
            period_b_end: End of period B
            
        Returns:
            Dictionary with comparison results
        """
        # Filter data for both periods
        df_a = df[(df['datetime_from'] >= period_a_start) & 
                  (df['datetime_from'] <= period_a_end)]
        df_b = df[(df['datetime_from'] >= period_b_start) & 
                  (df['datetime_from'] <= period_b_end)]
        
        # Calculate metrics for both periods
        metrics_a = self.get_basic_metrics(df_a)
        metrics_b = self.get_basic_metrics(df_b)
        
        # Calculate deltas
        delta_hours = self._calculate_percentage_change(
            metrics_a['total_hours'],
            metrics_b['total_hours']
        )
        delta_deep_work = self._calculate_percentage_change(
            metrics_a['deep_work_hours'],
            metrics_b['deep_work_hours']
        )
        delta_daily_avg = self._calculate_percentage_change(
            metrics_a['daily_avg'],
            metrics_b['daily_avg']
        )
        
        return {
            'metrics_a': metrics_a,
            'metrics_b': metrics_b,
            'deltas': {
                'total_hours': delta_hours,
                'deep_work_hours': delta_deep_work,
                'daily_avg': delta_daily_avg,
            }
        }
    
    def _calculate_percentage_change(self, current: float, previous: float) -> float:
        """
        Calculate percentage change between two values.
        
        Args:
            current: Current value
            previous: Previous value
            
        Returns:
            Percentage change
        """
        if previous == 0:
            return 100.0 if current > 0 else 0.0
        return round(((current - previous) / previous) * 100, 1)
    
    def _empty_metrics(self) -> dict:
        """Return empty metrics dict."""
        return {
            'total_hours': 0.0,
            'deep_work_hours': 0.0,
            'daily_avg': 0.0,
            'total_switches': 0,
            'deep_work_ratio': 0.0,
            'flow_index': 0.0,
            'avg_sleep': 0.0,
            'sleep_regularity': 0.0,
            'categories': [],
            'circadian': pd.DataFrame(),
            'daily_summary': pd.DataFrame(),
            'advanced': {},
            'consistency': pd.DataFrame()
        }
    
    def _empty_basic_metrics(self) -> dict:
        """Return empty basic metrics dict."""
        return {
            'total_hours': 0.0,
            'deep_work_hours': 0.0,
            'daily_avg': 0.0,
        }
    
    def _empty_advanced_metrics(self) -> dict:
        """Return empty advanced metrics dict."""
        return {
            'total_switches': 0,
            'deep_work_ratio': 0.0,
            'flow_index': 0.0,
            'avg_sleep': 0.0,
            'sleep_regularity': 0.0,
            'fragmentation_index': 0.0,
        }


# Global singleton instance
_metrics_service = MetricsService()


def get_metrics_service() -> MetricsService:
    """
    Get the global MetricsService instance.
    
    Returns:
        MetricsService singleton
    """
    return _metrics_service