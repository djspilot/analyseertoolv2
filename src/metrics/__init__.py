"""Metrics package for activity data analysis."""

from .basic import (
    calculate_metrics,
    calculate_circadian_metrics,
    calculate_consistency_metrics,
    get_daily_summary,
    get_category_summary,
)
from .advanced import (
    calculate_advanced_metrics,
    calculate_fragmentation_index,
    calculate_deep_work_ratio,
    calculate_context_switching_penalty,
    calculate_flow_index,
    infer_sleep_from_gaps,
    calculate_sleep_regularity_index,
    get_hourly_distribution,
    get_circadian_profile,
)
