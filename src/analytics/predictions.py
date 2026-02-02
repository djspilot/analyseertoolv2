"""
Predictive Analytics Module
Forecasts goal achievement, burnout risk, and productivity trends.
Uses simple statistical models optimized for small datasets.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np

from ..config import DEEP_WORK_CATEGORIES, FLOW_MIN_DURATION_HOURS
from ..goals import load_goals
from ..logger import setup_logger
from ..utils import matches_categories

logger = setup_logger(__name__)


@dataclass
class Prediction:
    """A prediction with confidence and explanation."""
    metric: str
    current_value: float
    predicted_value: float
    target_value: Optional[float]
    will_achieve: bool
    confidence: float  # 0-1
    trend: str  # "up", "down", "stable"
    message: str
    advice: str


@dataclass
class BurnoutRisk:
    """Burnout risk assessment."""
    risk_level: str  # "low", "medium", "high", "critical"
    risk_score: float  # 0-100
    warning_signs: list[str]
    recommendations: list[str]


def calculate_trend(values: list[float], window: int = 7) -> tuple[str, float]:
    """
    Calculate trend direction and slope from time series.
    
    Returns:
        Tuple of (direction, slope_per_day)
    """
    if len(values) < 3:
        return "stable", 0.0
    
    # Use last N values
    recent = values[-window:] if len(values) >= window else values
    
    # Simple linear regression
    x = np.arange(len(recent))
    y = np.array(recent)
    
    # y = mx + b
    n = len(x)
    slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2 + 1e-10)
    
    # Determine direction based on slope relative to mean
    mean_val = np.mean(y)
    relative_slope = slope / (mean_val + 1e-10)
    
    if relative_slope > 0.05:
        direction = "up"
    elif relative_slope < -0.05:
        direction = "down"
    else:
        direction = "stable"
    
    return direction, slope


def predict_weekly_value(daily_values: list[float], days_remaining: int) -> float:
    """
    Predict end-of-week value based on current trajectory.
    
    Args:
        daily_values: Values so far this week
        days_remaining: Days left in the week
        
    Returns:
        Predicted total for the week
    """
    if not daily_values:
        return 0.0
    
    current_total = sum(daily_values)
    
    if len(daily_values) >= 2:
        # Use trend-adjusted prediction
        trend, slope = calculate_trend(daily_values)
        avg_recent = np.mean(daily_values[-3:]) if len(daily_values) >= 3 else np.mean(daily_values)
        
        # Project remaining days with trend adjustment
        projected_remaining = 0.0
        for i in range(days_remaining):
            day_value = avg_recent + slope * (len(daily_values) + i)
            projected_remaining += max(0, day_value)  # Can't be negative
    else:
        # Not enough data, use simple average
        avg = current_total / len(daily_values)
        projected_remaining = avg * days_remaining
    
    return current_total + projected_remaining


def predict_goal_achievement(df: pd.DataFrame) -> list[Prediction]:
    """
    Predict whether weekly goals will be achieved.
    
    Args:
        df: DataFrame with activity data
        
    Returns:
        List of predictions for each goal
    """
    if df.empty:
        return []
    
    predictions = []
    goals = load_goals()
    
    # Get current week's data
    today = datetime.now().date()
    week_start = today - timedelta(days=today.weekday())  # Monday
    week_end = week_start + timedelta(days=6)
    days_passed = (today - week_start).days + 1
    days_remaining = 7 - days_passed
    
    week_df = df[
        (df['datetime_from'].dt.date >= week_start) &
        (df['datetime_from'].dt.date <= today)
    ]
    
    # 1. Deep Work Prediction
    dw_df = week_df[week_df['activity_type'].apply(lambda x: matches_categories(x, DEEP_WORK_CATEGORIES))]
    daily_dw = dw_df.groupby(dw_df['datetime_from'].dt.date)['duration_hours'].sum().tolist()
    
    # Fill missing days with 0
    all_days = [(week_start + timedelta(days=i)) for i in range(days_passed)]
    daily_dw_filled = []
    for day in all_days:
        day_total = dw_df[dw_df['datetime_from'].dt.date == day]['duration_hours'].sum()
        daily_dw_filled.append(day_total)
    
    current_dw = sum(daily_dw_filled)
    predicted_dw = predict_weekly_value(daily_dw_filled, days_remaining)
    trend, _ = calculate_trend(daily_dw_filled)
    
    will_achieve_dw = predicted_dw >= goals.deep_work_hours_weekly
    confidence_dw = min(0.9, 0.5 + (days_passed / 7) * 0.4)  # More confident as week progresses
    
    if will_achieve_dw:
        message_dw = f"Je ligt op koers! Verwacht: {predicted_dw:.1f}h deep work deze week."
        advice_dw = "Blijf zo doorgaan. Focus op lange sessies voor maximale flow."
    else:
        shortfall = goals.deep_work_hours_weekly - predicted_dw
        daily_needed = shortfall / max(1, days_remaining)
        message_dw = f"⚠️ Je haalt je doel waarschijnlijk niet. Verwacht: {predicted_dw:.1f}h, doel: {goals.deep_work_hours_weekly}h"
        advice_dw = f"Plan {daily_needed:.1f}h extra deep work per dag. Blokkeer ochtenden voor focuswerk."
    
    predictions.append(Prediction(
        metric="Deep Work",
        current_value=current_dw,
        predicted_value=predicted_dw,
        target_value=goals.deep_work_hours_weekly,
        will_achieve=will_achieve_dw,
        confidence=confidence_dw,
        trend=trend,
        message=message_dw,
        advice=advice_dw,
    ))
    
    # 2. Sport Prediction
    sport_df = week_df[week_df['activity_type'] == 'Sport']
    daily_sport = []
    for day in all_days:
        day_total = sport_df[sport_df['datetime_from'].dt.date == day]['duration_hours'].sum()
        daily_sport.append(day_total)
    
    current_sport = sum(daily_sport)
    predicted_sport = predict_weekly_value(daily_sport, days_remaining)
    trend_sport, _ = calculate_trend(daily_sport)
    
    will_achieve_sport = predicted_sport >= goals.sport_hours_weekly
    
    if will_achieve_sport:
        message_sport = f"Goed bezig met sporten! Verwacht: {predicted_sport:.1f}h"
        advice_sport = "Houd dit ritme aan."
    else:
        shortfall = goals.sport_hours_weekly - predicted_sport
        message_sport = f"⚠️ Sport doel in gevaar. Verwacht: {predicted_sport:.1f}h, doel: {goals.sport_hours_weekly}h"
        advice_sport = f"Plan nog {shortfall:.1f}h sport deze week. Korte workouts tellen ook!"
    
    predictions.append(Prediction(
        metric="Sport",
        current_value=current_sport,
        predicted_value=predicted_sport,
        target_value=goals.sport_hours_weekly,
        will_achieve=will_achieve_sport,
        confidence=min(0.85, 0.5 + (days_passed / 7) * 0.35),
        trend=trend_sport,
        message=message_sport,
        advice=advice_sport,
    ))
    
    # 3. Flow Sessions Prediction
    flow_df = week_df[
        (week_df['activity_type'].apply(lambda x: matches_categories(x, DEEP_WORK_CATEGORIES))) &
        (week_df['duration_hours'] >= FLOW_MIN_DURATION_HOURS)
    ]
    current_flow_sessions = len(flow_df)
    
    # Estimate based on rate so far
    if days_passed > 0:
        flow_rate = current_flow_sessions / days_passed
        predicted_flow = current_flow_sessions + (flow_rate * days_remaining)
    else:
        predicted_flow = 0
    
    will_achieve_flow = predicted_flow >= goals.min_flow_sessions_weekly
    
    if will_achieve_flow:
        message_flow = f"Flow sessies op schema. Verwacht: {predicted_flow:.0f} sessies"
        advice_flow = "Blijf lange, ononderbroken blokken plannen."
    else:
        needed = goals.min_flow_sessions_weekly - current_flow_sessions
        message_flow = f"⚠️ Meer flow sessies nodig. Nu: {current_flow_sessions}, doel: {goals.min_flow_sessions_weekly}"
        advice_flow = f"Plan {needed} sessies van 90+ minuten. Schakel notificaties uit."
    
    predictions.append(Prediction(
        metric="Flow Sessies",
        current_value=current_flow_sessions,
        predicted_value=predicted_flow,
        target_value=goals.min_flow_sessions_weekly,
        will_achieve=will_achieve_flow,
        confidence=min(0.8, 0.4 + (days_passed / 7) * 0.4),
        trend="stable",
        message=message_flow,
        advice=advice_flow,
    ))
    
    return predictions


def assess_burnout_risk(df: pd.DataFrame, lookback_days: int = 14) -> BurnoutRisk:
    """
    Assess burnout risk based on work patterns.
    
    Factors:
    - Recovery ratio (recovery/drain activities)
    - Work hours consistency
    - Sleep patterns
    - Weekend work
    - Deep work fragmentation
    """
    if df.empty:
        return BurnoutRisk(
            risk_level="unknown",
            risk_score=0,
            warning_signs=[],
            recommendations=[]
        )
    
    cutoff = datetime.now() - timedelta(days=lookback_days)
    recent = df[df['datetime_from'] >= cutoff]
    
    if recent.empty:
        return BurnoutRisk(
            risk_level="unknown",
            risk_score=0,
            warning_signs=[],
            recommendations=[]
        )
    
    risk_score = 0
    warning_signs = []
    recommendations = []
    
    # 1. Recovery Ratio (target: 0.3-0.5)
    recovery_cats = {'Sport', 'Yoga', 'Walking', 'Entertainment', 'Read', 'Music'}
    drain_cats = {'Work', 'Coding', 'Housework'}
    
    recovery_hours = recent[recent['activity_type'].apply(lambda x: matches_categories(x, recovery_cats))]['duration_hours'].sum()
    drain_hours = recent[recent['activity_type'].apply(lambda x: matches_categories(x, drain_cats))]['duration_hours'].sum()
    
    recovery_ratio = recovery_hours / (drain_hours + 1e-10)
    
    if recovery_ratio < 0.2:
        risk_score += 30
        warning_signs.append(f"Zeer lage recovery ratio: {recovery_ratio:.2f} (doel: 0.3-0.5)")
        recommendations.append("Plan minimaal 30 min ontspanning per dag (sport, wandelen, lezen)")
    elif recovery_ratio < 0.3:
        risk_score += 15
        warning_signs.append(f"Lage recovery ratio: {recovery_ratio:.2f}")
        recommendations.append("Voeg meer herstelmomenten toe aan je dag")
    
    # 2. Daily Work Hours (excessive = > 10h/day average)
    work_df = recent[recent['activity_type'].apply(lambda x: matches_categories(x, drain_cats))]
    if not work_df.empty:
        daily_work = work_df.groupby(work_df['datetime_from'].dt.date)['duration_hours'].sum()
        avg_daily_work = daily_work.mean()
        max_daily_work = daily_work.max()
        
        if avg_daily_work > 10:
            risk_score += 25
            warning_signs.append(f"Gemiddeld {avg_daily_work:.1f}h werk per dag (> 10h)")
            recommendations.append("Beperk werkdagen tot 8-9 uur")
        elif avg_daily_work > 8:
            risk_score += 10
        
        if max_daily_work > 12:
            risk_score += 15
            warning_signs.append(f"Lange werkdag gedetecteerd: {max_daily_work:.1f}h")
    
    # 3. Weekend Work
    recent['weekday'] = recent['datetime_from'].dt.dayofweek
    weekend_work = recent[
        (recent['weekday'].isin([5, 6])) &  # Sat, Sun
        (recent['activity_type'].apply(lambda x: matches_categories(x, drain_cats)))
    ]['duration_hours'].sum()
    
    total_work = recent[recent['activity_type'].apply(lambda x: matches_categories(x, drain_cats))]['duration_hours'].sum()
    weekend_work_ratio = weekend_work / (total_work + 1e-10)
    
    if weekend_work_ratio > 0.3:
        risk_score += 20
        warning_signs.append(f"{weekend_work_ratio*100:.0f}% van je werk is in het weekend")
        recommendations.append("Bescherm je weekenden - plan werk alleen doordeweeks")
    elif weekend_work_ratio > 0.15:
        risk_score += 10
        warning_signs.append("Je werkt regelmatig in het weekend")
    
    # 4. Work Hours Trend (increasing = bad)
    if len(work_df) > 7:
        daily_totals = work_df.groupby(work_df['datetime_from'].dt.date)['duration_hours'].sum().tolist()
        trend, slope = calculate_trend(daily_totals)
        
        if trend == "up" and slope > 0.5:  # Increasing by 0.5h+ per day
            risk_score += 15
            warning_signs.append("Je werkuren zijn stijgend")
            recommendations.append("Bewust grenzen stellen aan werktijd")
    
    # 5. Fragmentation (high = stressed/reactive)
    from ..metrics.advanced import calculate_fragmentation_index
    frag = calculate_fragmentation_index(recent)
    
    work_frag = frag.get('Work', 0) + frag.get('Coding', 0)
    if work_frag > 3:  # More than 3 sessions per hour of work
        risk_score += 15
        warning_signs.append("Hoge fragmentatie in werkblokken (veel onderbrekingen)")
        recommendations.append("Blokkeer notificaties tijdens deep work")
    
    # 6. Sleep (from gaps analysis)
    from ..metrics.advanced import calculate_sleep_regularity_index
    sleep_stats = calculate_sleep_regularity_index(recent)
    
    avg_sleep = sleep_stats.get('avg_sleep_hours', 7)
    if avg_sleep < 6:
        risk_score += 20
        warning_signs.append(f"Te weinig slaap: gemiddeld {avg_sleep:.1f}h")
        recommendations.append("Prioriteer 7-8 uur slaap")
    elif avg_sleep < 7:
        risk_score += 10
        warning_signs.append(f"Suboptimale slaap: {avg_sleep:.1f}h gemiddeld")
    
    # Determine risk level
    if risk_score >= 70:
        risk_level = "critical"
    elif risk_score >= 50:
        risk_level = "high"
    elif risk_score >= 30:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    # Add general recommendations based on level
    if risk_level in ["high", "critical"]:
        recommendations.insert(0, "🚨 Overweeg een rustdag of korte vakantie")
    
    return BurnoutRisk(
        risk_level=risk_level,
        risk_score=min(100, risk_score),
        warning_signs=warning_signs,
        recommendations=recommendations[:5],  # Limit to top 5
    )


def get_productivity_forecast(df: pd.DataFrame) -> dict:
    """
    Generate a comprehensive productivity forecast.
    
    Returns dict with:
    - predictions: list of goal predictions
    - burnout_risk: burnout assessment
    - weekly_outlook: summary message
    """
    predictions = predict_goal_achievement(df)
    burnout = assess_burnout_risk(df)
    
    # Generate weekly outlook
    goals_at_risk = [p for p in predictions if not p.will_achieve]
    goals_on_track = [p for p in predictions if p.will_achieve]
    
    if not predictions:
        outlook = "Niet genoeg data voor voorspellingen."
    elif len(goals_at_risk) == 0:
        outlook = "🎯 Uitstekend! Alle doelen liggen op schema."
    elif len(goals_at_risk) == 1:
        outlook = f"⚠️ Let op: {goals_at_risk[0].metric} doel in gevaar."
    else:
        names = ", ".join([p.metric for p in goals_at_risk])
        outlook = f"⚠️ Meerdere doelen in gevaar: {names}"
    
    if burnout.risk_level in ["high", "critical"]:
        outlook += f" 🔴 Burnout risico: {burnout.risk_level}!"
    
    return {
        "predictions": [
            {
                "metric": p.metric,
                "current": p.current_value,
                "predicted": p.predicted_value,
                "target": p.target_value,
                "will_achieve": p.will_achieve,
                "confidence": p.confidence,
                "trend": p.trend,
                "message": p.message,
                "advice": p.advice,
            }
            for p in predictions
        ],
        "burnout": {
            "level": burnout.risk_level,
            "score": burnout.risk_score,
            "warnings": burnout.warning_signs,
            "recommendations": burnout.recommendations,
        },
        "weekly_outlook": outlook,
    }
