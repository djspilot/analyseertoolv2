"""
AI Insights Generator - Natural language analysis of time tracking data.
Generates actionable insights from activity patterns.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from .config import DEEP_WORK_CATEGORIES, FLOW_MIN_DURATION_HOURS
from .utils import matches_categories
from .metrics.advanced import (
    calculate_fragmentation_index,
    calculate_flow_index,
    calculate_sleep_regularity_index,
    get_circadian_profile,
    infer_sleep_from_gaps,
)


# Energy weights for activities
ENERGY_WEIGHTS = {
    'Sport': 50,
    'Yoga': 60,
    'Walking': 40,
    'Entertainment': 10,
    'Read': 20,
    'Music': 15,
    'Work': -40,
    'Coding': -50,
    'Housework': -20,
    'Internet': -10,
    'Other': 0,
}

RECOVERY_CATEGORIES = {'Sport', 'Yoga', 'Walking', 'Entertainment', 'Read', 'Music'}
DRAIN_CATEGORIES = {'Work', 'Coding', 'Housework'}


def generate_insights(df: pd.DataFrame) -> list[dict]:
    """
    Generate AI insights from activity data.
    
    Args:
        df: DataFrame with activity data including datetime_from, datetime_to,
            activity_type, duration_hours, is_deep_work
    
    Returns:
        List of insight dicts with keys: category, title, message, severity, icon
        Severity: 'positive', 'warning', 'neutral', 'negative'
    """
    if df.empty:
        return [{"category": "info", "title": "No Data", 
                 "message": "Upload a CSV file to get started.", 
                 "severity": "neutral", "icon": "📊"}]
    
    insights = []
    
    # 1. Flow State Analysis
    insights.extend(_analyze_flow_state(df))
    
    # 2. Peak Performance Hours
    insights.extend(_analyze_peak_hours(df))
    
    # 3. Energy Balance
    insights.extend(_analyze_energy_balance(df))
    
    # 4. Fragmentation Analysis
    insights.extend(_analyze_fragmentation(df))
    
    # 5. Sleep Pattern Analysis
    insights.extend(_analyze_sleep_patterns(df))
    
    # 6. Weekly Pattern Analysis
    insights.extend(_analyze_weekly_patterns(df))
    
    # 7. Trend Analysis (recent vs historical)
    insights.extend(_analyze_trends(df))
    
    # 8. Recommendations
    insights.extend(_generate_recommendations(df))
    
    # Sort by severity (warnings first, then positive, then neutral)
    severity_order = {'negative': 0, 'warning': 1, 'positive': 2, 'neutral': 3}
    insights.sort(key=lambda x: severity_order.get(x['severity'], 4))
    
    return insights


def _analyze_flow_state(df: pd.DataFrame) -> list[dict]:
    """Analyze flow state patterns."""
    insights = []
    
    flow_index = calculate_flow_index(df)
    deep_work_df = df[df['activity_type'].apply(lambda x: matches_categories(x, DEEP_WORK_CATEGORIES))]
    flow_sessions = deep_work_df[deep_work_df['duration_hours'] >= FLOW_MIN_DURATION_HOURS]
    
    total_deep_hours = deep_work_df['duration_hours'].sum()
    flow_hours = flow_sessions['duration_hours'].sum()
    
    if flow_index >= 50:
        insights.append({
            "category": "flow",
            "title": "Strong Flow State",
            "message": f"Excellent! {flow_index}% of your deep work time is in flow sessions (≥90min). "
                      f"You've accumulated {flow_hours:.1f}h in deep focus.",
            "severity": "positive",
            "icon": "🎯"
        })
    elif flow_index >= 25:
        insights.append({
            "category": "flow",
            "title": "Moderate Flow",
            "message": f"Your flow index is {flow_index}%. Consider blocking longer uninterrupted periods "
                      f"to increase flow state time. Currently {flow_hours:.1f}h out of {total_deep_hours:.1f}h deep work.",
            "severity": "neutral",
            "icon": "⚡"
        })
    elif total_deep_hours > 0:
        insights.append({
            "category": "flow",
            "title": "Low Flow Index",
            "message": f"Only {flow_index}% of deep work reaches flow state. Your sessions may be too fragmented. "
                      f"Try protecting 90+ minute blocks for complex work.",
            "severity": "warning",
            "icon": "⚠️"
        })
    
    # Flow streak analysis
    df_copy = df.copy()
    df_copy['date'] = df_copy['datetime_from'].dt.date
    daily_flow = flow_sessions.groupby(flow_sessions['datetime_from'].dt.date)['duration_hours'].sum()
    
    if len(daily_flow) > 0:
        streak = 0
        max_streak = 0
        for date in sorted(daily_flow.index):
            if daily_flow[date] >= 2.0:  # 2h flow target
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        
        if max_streak >= 5:
            insights.append({
                "category": "flow",
                "title": f"{max_streak}-Day Flow Streak",
                "message": f"You maintained ≥2h of flow state for {max_streak} consecutive days. "
                          f"This consistency is key for mastery and deep skill development.",
                "severity": "positive",
                "icon": "🔥"
            })
    
    return insights


def _analyze_peak_hours(df: pd.DataFrame) -> list[dict]:
    """Identify peak performance hours."""
    insights = []
    
    profile = get_circadian_profile(df)
    
    # Find peak hours for deep work
    deep_work_peaks = []
    for activity in DEEP_WORK_CATEGORIES:
        if activity in profile:
            deep_work_peaks.append((activity, profile[activity]['peak_hour']))
    
    if deep_work_peaks:
        # Most common peak hour for deep work
        peak_hours = [p[1] for p in deep_work_peaks]
        avg_peak = sum(peak_hours) / len(peak_hours)
        
        if 6 <= avg_peak <= 12:
            time_label = "morning"
            recommendation = "Consider protecting this time from meetings."
        elif 12 < avg_peak <= 17:
            time_label = "afternoon"
            recommendation = "Your post-lunch focus is strong."
        else:
            time_label = "evening"
            recommendation = "Late sessions may affect sleep quality."
        
        insights.append({
            "category": "rhythm",
            "title": f"Peak Focus: {int(avg_peak):02d}:00",
            "message": f"Your deep work peaks in the {time_label} around {int(avg_peak):02d}:00. "
                      f"{recommendation}",
            "severity": "neutral",
            "icon": "⏰"
        })
    
    return insights


def _analyze_energy_balance(df: pd.DataFrame) -> list[dict]:
    """Analyze energy balance between charging and draining activities."""
    insights = []
    
    df_copy = df.copy()
    df_copy['energy'] = df_copy['activity_type'].map(
        lambda x: ENERGY_WEIGHTS.get(x, 0) * df_copy[df_copy['activity_type'] == x]['duration_hours'].mean()
    )
    
    # Calculate recovery vs drain ratio
    recovery_hours = df[df['activity_type'].apply(lambda x: matches_categories(x, RECOVERY_CATEGORIES))]['duration_hours'].sum()
    drain_hours = df[df['activity_type'].apply(lambda x: matches_categories(x, DRAIN_CATEGORIES))]['duration_hours'].sum()
    
    if drain_hours > 0:
        ratio = recovery_hours / drain_hours
        
        if ratio >= 0.4:
            insights.append({
                "category": "energy",
                "title": "Healthy Energy Balance",
                "message": f"Recovery ratio of {ratio:.2f} (target: 0.3-0.5). "
                          f"You're balancing {recovery_hours:.1f}h recovery with {drain_hours:.1f}h productive work.",
                "severity": "positive",
                "icon": "💚"
            })
        elif ratio >= 0.2:
            insights.append({
                "category": "energy",
                "title": "Low Recovery",
                "message": f"Recovery ratio is {ratio:.2f} (below 0.3 target). "
                          f"Consider adding more restorative activities to prevent burnout.",
                "severity": "warning",
                "icon": "🔋"
            })
        else:
            insights.append({
                "category": "energy",
                "title": "Energy Deficit",
                "message": f"Recovery ratio is only {ratio:.2f}. "
                          f"You have {drain_hours:.1f}h of draining activities but only {recovery_hours:.1f}h recovery. "
                          f"Burnout risk is elevated.",
                "severity": "negative",
                "icon": "🚨"
            })
    
    return insights


def _analyze_fragmentation(df: pd.DataFrame) -> list[dict]:
    """Analyze work fragmentation patterns."""
    insights = []
    
    fi = calculate_fragmentation_index(df)
    
    # Calculate overall fragmentation for deep work categories
    deep_work_fi = []
    for cat in DEEP_WORK_CATEGORIES:
        if cat in fi:
            deep_work_fi.append(fi[cat])
    
    if deep_work_fi:
        avg_fi = sum(deep_work_fi) / len(deep_work_fi)
        
        if avg_fi <= 1.0:
            insights.append({
                "category": "patterns",
                "title": "Consolidated Work",
                "message": f"Fragmentation index of {avg_fi:.2f} sessions/hour indicates focused, "
                          f"consolidated work blocks. Keep it up!",
                "severity": "positive",
                "icon": "🧱"
            })
        elif avg_fi <= 2.0:
            insights.append({
                "category": "patterns",
                "title": "Moderate Fragmentation",
                "message": f"Fragmentation index of {avg_fi:.2f} sessions/hour. "
                          f"Some interruptions are present. Consider time-blocking strategies.",
                "severity": "neutral",
                "icon": "📦"
            })
        else:
            insights.append({
                "category": "patterns",
                "title": "High Fragmentation",
                "message": f"Fragmentation index of {avg_fi:.2f} sessions/hour indicates frequent interruptions. "
                          f"This prevents deep work. Try the Pomodoro technique or focus modes.",
                "severity": "warning",
                "icon": "💔"
            })
    
    return insights


def _analyze_sleep_patterns(df: pd.DataFrame) -> list[dict]:
    """Analyze sleep patterns from activity gaps."""
    insights = []
    
    sri = calculate_sleep_regularity_index(df)
    sleep_df = infer_sleep_from_gaps(df)
    
    if not sleep_df.empty:
        avg_sleep = sri.get('avg_sleep_hours', 0)
        regularity = sri.get('sri', 0)
        
        if avg_sleep >= 7:
            insights.append({
                "category": "wellness",
                "title": f"Good Sleep: {avg_sleep:.1f}h avg",
                "message": f"Average sleep of {avg_sleep:.1f}h meets recommended 7-9h range. "
                          f"Sleep regularity index: {regularity:.2f} (lower = more consistent).",
                "severity": "positive",
                "icon": "😴"
            })
        elif avg_sleep >= 6:
            insights.append({
                "category": "wellness",
                "title": f"Borderline Sleep: {avg_sleep:.1f}h",
                "message": f"Average sleep of {avg_sleep:.1f}h is below the recommended 7h. "
                          f"This may impact focus and recovery.",
                "severity": "warning",
                "icon": "😪"
            })
        elif avg_sleep > 0:
            insights.append({
                "category": "wellness",
                "title": f"Sleep Deficit: {avg_sleep:.1f}h",
                "message": f"Average sleep of only {avg_sleep:.1f}h is concerning. "
                          f"Chronic sleep deprivation significantly impairs cognitive performance.",
                "severity": "negative",
                "icon": "🚨"
            })
        
        # Regularity insight
        if regularity > 2.0:
            insights.append({
                "category": "wellness",
                "title": "Irregular Sleep Schedule",
                "message": f"Sleep regularity index of {regularity:.2f} indicates variable sleep times. "
                          f"Consistent sleep/wake times improve circadian rhythm and focus.",
                "severity": "warning",
                "icon": "🌙"
            })
    
    return insights


def _analyze_weekly_patterns(df: pd.DataFrame) -> list[dict]:
    """Analyze patterns by day of week."""
    insights = []
    
    df_copy = df.copy()
    df_copy['weekday'] = df_copy['datetime_from'].dt.day_name()
    
    # Deep work by weekday
    deep_work = df_copy[df_copy['activity_type'].apply(lambda x: matches_categories(x, DEEP_WORK_CATEGORIES))]
    daily_deep = deep_work.groupby('weekday')['duration_hours'].sum()
    
    if not daily_deep.empty:
        best_day = daily_deep.idxmax()
        worst_day = daily_deep.idxmin()
        
        if daily_deep[best_day] > daily_deep[worst_day] * 2:
            insights.append({
                "category": "patterns",
                "title": f"Best Day: {best_day}",
                "message": f"{best_day} has {daily_deep[best_day]:.1f}h deep work vs "
                          f"{daily_deep[worst_day]:.1f}h on {worst_day}. "
                          f"Consider scheduling important tasks on your strongest days.",
                "severity": "neutral",
                "icon": "📅"
            })
    
    return insights


def _analyze_trends(df: pd.DataFrame) -> list[dict]:
    """Analyze recent trends vs historical patterns."""
    insights = []
    
    df_copy = df.copy()
    max_date = df_copy['datetime_from'].max()
    
    # Last 7 days vs previous 7 days
    recent_start = max_date - timedelta(days=7)
    previous_start = recent_start - timedelta(days=7)
    
    recent = df_copy[df_copy['datetime_from'] >= recent_start]
    previous = df_copy[(df_copy['datetime_from'] >= previous_start) & 
                       (df_copy['datetime_from'] < recent_start)]
    
    if not recent.empty and not previous.empty:
        # Compare deep work hours
        recent_dw = recent[recent['activity_type'].apply(lambda x: matches_categories(x, DEEP_WORK_CATEGORIES))]['duration_hours'].sum()
        previous_dw = previous[previous['activity_type'].apply(lambda x: matches_categories(x, DEEP_WORK_CATEGORIES))]['duration_hours'].sum()
        
        if previous_dw > 0:
            change_pct = ((recent_dw - previous_dw) / previous_dw) * 100
            
            if change_pct >= 20:
                insights.append({
                    "category": "trend",
                    "title": f"Deep Work ↑{change_pct:.0f}%",
                    "message": f"Deep work increased from {previous_dw:.1f}h to {recent_dw:.1f}h "
                              f"compared to last week. Great momentum!",
                    "severity": "positive",
                    "icon": "📈"
                })
            elif change_pct <= -20:
                insights.append({
                    "category": "trend",
                    "title": f"Deep Work ↓{abs(change_pct):.0f}%",
                    "message": f"Deep work decreased from {previous_dw:.1f}h to {recent_dw:.1f}h "
                              f"compared to last week. Check for distractions or schedule conflicts.",
                    "severity": "warning",
                    "icon": "📉"
                })
    
    return insights


def _generate_recommendations(df: pd.DataFrame) -> list[dict]:
    """Generate actionable recommendations based on all analysis."""
    insights = []
    
    # Find underutilized recovery time
    recovery_hours = df[df['activity_type'].apply(lambda x: matches_categories(x, RECOVERY_CATEGORIES))]['duration_hours'].sum()
    total_hours = df['duration_hours'].sum()
    
    if total_hours > 0:
        recovery_pct = (recovery_hours / total_hours) * 100
        
        if recovery_pct < 15:
            insights.append({
                "category": "recommendation",
                "title": "Add Recovery Time",
                "message": f"Only {recovery_pct:.0f}% of tracked time is recovery. "
                          f"Consider scheduling Sport, Yoga, or Walking sessions.",
                "severity": "neutral",
                "icon": "💡"
            })
    
    # Check for late-night work
    df_copy = df.copy()
    df_copy['hour'] = df_copy['datetime_from'].dt.hour
    late_work = df_copy[(df_copy['hour'] >= 22) & 
                        (df_copy['activity_type'].apply(lambda x: matches_categories(x, DEEP_WORK_CATEGORIES)))]
    
    if len(late_work) > 5:
        late_hours = late_work['duration_hours'].sum()
        insights.append({
            "category": "recommendation",
            "title": "Evening Work Pattern",
            "message": f"You have {late_hours:.1f}h of deep work after 22:00. "
                      f"Late cognitive work can impact sleep quality and next-day performance.",
            "severity": "warning",
            "icon": "🌃"
        })
    
    return insights


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Get summary statistics for insights panel header."""
    if df.empty:
        return {"total_hours": 0, "deep_work_hours": 0, "flow_index": 0, "days": 0}
    
    total_hours = df['duration_hours'].sum()
    deep_work_df = df[df['activity_type'].apply(lambda x: matches_categories(x, DEEP_WORK_CATEGORIES))]
    deep_work_hours = deep_work_df['duration_hours'].sum()
    flow_index = calculate_flow_index(df)
    days = df['datetime_from'].dt.date.nunique()
    
    return {
        "total_hours": round(total_hours, 1),
        "deep_work_hours": round(deep_work_hours, 1),
        "flow_index": flow_index,
        "days": days
    }
