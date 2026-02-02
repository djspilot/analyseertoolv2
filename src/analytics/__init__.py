"""
Analytics subpackage for advanced data analysis.
"""

from .predictions import (
    predict_goal_achievement,
    assess_burnout_risk,
    get_productivity_forecast,
    Prediction,
    BurnoutRisk,
)

from .gamification import (
    get_gamification_profile,
    calculate_streaks,
    check_badges,
    calculate_xp,
)

from .aggregation import (
    smart_aggregate,
    get_optimal_aggregation,
    get_data_manager,
    SmartDataManager,
)

from .categorization import (
    suggest_category,
    learn_from_data,
    get_categorizer,
    ActivityCategorizer,
)

__all__ = [
    # Predictions
    "predict_goal_achievement",
    "assess_burnout_risk", 
    "get_productivity_forecast",
    "Prediction",
    "BurnoutRisk",
    # Gamification
    "get_gamification_profile",
    "calculate_streaks",
    "check_badges",
    "calculate_xp",
    # Aggregation
    "smart_aggregate",
    "get_optimal_aggregation",
    "get_data_manager",
    "SmartDataManager",
    # Categorization
    "suggest_category",
    "learn_from_data",
    "get_categorizer",
    "ActivityCategorizer",
]
