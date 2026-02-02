"""
Habit Streaks & Gamification Module
Tracks consistency, awards badges, and motivates through game mechanics.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Optional
import json
from pathlib import Path

import pandas as pd

from ..config import DEEP_WORK_CATEGORIES, FLOW_MIN_DURATION_HOURS
from ..logger import setup_logger
from ..utils import matches_categories

logger = setup_logger(__name__)

# Persistence
STREAKS_FILE = Path(__file__).parent.parent.parent / "data" / "streaks.json"


@dataclass
class Badge:
    """An achievement badge."""
    id: str
    name: str
    description: str
    icon: str
    earned_date: Optional[date] = None
    progress: float = 0.0  # 0-1, for badges not yet earned
    

@dataclass
class Streak:
    """A habit streak tracker."""
    habit: str
    current_streak: int = 0
    longest_streak: int = 0
    last_completed: Optional[date] = None
    total_completions: int = 0


@dataclass
class GamificationProfile:
    """Complete gamification profile for a user."""
    level: int = 1
    xp: int = 0
    xp_to_next_level: int = 100
    streaks: dict = field(default_factory=dict)  # habit_name -> Streak
    badges: list = field(default_factory=list)  # List of earned Badge
    weekly_challenge: Optional[str] = None
    weekly_challenge_progress: float = 0.0


# XP rewards
XP_REWARDS = {
    "deep_work_hour": 10,
    "flow_session": 50,
    "sport_session": 25,
    "streak_day": 15,
    "goal_achieved": 100,
    "badge_earned": 200,
}

# Badge definitions
BADGE_DEFINITIONS = [
    # Streak badges
    Badge("streak_3", "Starter", "3 dagen op rij deep work", "🌱"),
    Badge("streak_7", "Week Warrior", "7 dagen op rij deep work", "🔥"),
    Badge("streak_14", "Fortnight Fighter", "14 dagen op rij deep work", "⚡"),
    Badge("streak_30", "Monthly Master", "30 dagen op rij deep work", "🏆"),
    Badge("streak_100", "Century Club", "100 dagen op rij deep work", "💎"),
    
    # Volume badges
    Badge("deep_100h", "Centurion", "100 uur deep work totaal", "💯"),
    Badge("deep_500h", "Half Millennium", "500 uur deep work totaal", "🎖️"),
    Badge("deep_1000h", "Grandmaster", "1000 uur deep work totaal", "👑"),
    
    # Flow badges
    Badge("flow_10", "Flow Finder", "10 flow sessies (90+ min)", "🌊"),
    Badge("flow_50", "Flow Master", "50 flow sessies", "🌀"),
    Badge("flow_100", "Zone Legend", "100 flow sessies", "🔮"),
    
    # Sport badges
    Badge("sport_50h", "Athlete", "50 uur sport totaal", "💪"),
    Badge("sport_100h", "Fitness Fanatic", "100 uur sport totaal", "🏃"),
    
    # Special badges
    Badge("early_bird", "Early Bird", "10 sessies gestart voor 7:00", "🐦"),
    Badge("night_owl", "Night Owl", "10 sessies na 22:00", "🦉"),
    Badge("perfect_week", "Perfect Week", "Alle weekdoelen gehaald", "⭐"),
    Badge("balanced", "Balanced Life", "Recovery ratio > 0.4 voor 7 dagen", "☯️"),
]


def calculate_level(xp: int) -> tuple[int, int, int]:
    """
    Calculate level from XP.
    Uses exponential scaling: each level needs 50% more XP.
    
    Returns:
        (level, xp_in_current_level, xp_needed_for_next)
    """
    level = 1
    xp_for_this_level = 0
    xp_for_next_level = 100
    
    remaining_xp = xp
    
    while remaining_xp >= xp_for_next_level:
        remaining_xp -= xp_for_next_level
        level += 1
        xp_for_this_level = xp_for_next_level
        xp_for_next_level = int(xp_for_next_level * 1.5)
    
    return level, remaining_xp, xp_for_next_level


def calculate_streaks(df: pd.DataFrame) -> dict[str, Streak]:
    """
    Calculate all habit streaks from activity data.
    
    Returns:
        Dict mapping habit name to Streak object
    """
    if df.empty:
        return {}
    
    streaks = {}
    today = datetime.now().date()
    
    # 1. Deep Work Streak (any deep work > 1h counts)
    dw_df = df[df['activity_type'].apply(lambda x: matches_categories(x, DEEP_WORK_CATEGORIES))]
    if not dw_df.empty:
        daily_dw = dw_df.groupby(dw_df['datetime_from'].dt.date)['duration_hours'].sum()
        dw_days = set(daily_dw[daily_dw >= 1.0].index)
        
        streak = calculate_consecutive_days(dw_days, today)
        longest = calculate_longest_streak(dw_days)
        
        streaks["Deep Work"] = Streak(
            habit="Deep Work",
            current_streak=streak,
            longest_streak=longest,
            last_completed=max(dw_days) if dw_days else None,
            total_completions=len(dw_days),
        )
    
    # 2. Flow Sessions Streak (at least 1 flow session per day)
    flow_df = df[
        (df['activity_type'].apply(lambda x: matches_categories(x, DEEP_WORK_CATEGORIES))) &
        (df['duration_hours'] >= FLOW_MIN_DURATION_HOURS)
    ]
    if not flow_df.empty:
        flow_days = set(flow_df['datetime_from'].dt.date.unique())
        
        streaks["Flow"] = Streak(
            habit="Flow",
            current_streak=calculate_consecutive_days(flow_days, today),
            longest_streak=calculate_longest_streak(flow_days),
            last_completed=max(flow_days) if flow_days else None,
            total_completions=len(flow_df),
        )
    
    # 3. Sport Streak
    sport_df = df[df['activity_type'] == 'Sport']
    if not sport_df.empty:
        sport_days = set(sport_df['datetime_from'].dt.date.unique())
        
        streaks["Sport"] = Streak(
            habit="Sport",
            current_streak=calculate_consecutive_days(sport_days, today),
            longest_streak=calculate_longest_streak(sport_days),
            last_completed=max(sport_days) if sport_days else None,
            total_completions=len(sport_days),
        )
    
    # 4. Any Activity Streak (tracked something each day)
    all_days = set(df['datetime_from'].dt.date.unique())
    streaks["Tracking"] = Streak(
        habit="Tracking",
        current_streak=calculate_consecutive_days(all_days, today),
        longest_streak=calculate_longest_streak(all_days),
        last_completed=max(all_days) if all_days else None,
        total_completions=len(all_days),
    )
    
    return streaks


def calculate_consecutive_days(days: set, from_date: date) -> int:
    """Calculate current streak ending at from_date."""
    if not days:
        return 0
    
    streak = 0
    current = from_date
    
    # Allow 1 day grace (if checking today and not logged yet)
    if current not in days:
        current = current - timedelta(days=1)
    
    while current in days:
        streak += 1
        current = current - timedelta(days=1)
    
    return streak


def calculate_longest_streak(days: set) -> int:
    """Calculate the longest consecutive streak in the data."""
    if not days:
        return 0
    
    sorted_days = sorted(days)
    longest = 1
    current = 1
    
    for i in range(1, len(sorted_days)):
        if sorted_days[i] - sorted_days[i-1] == timedelta(days=1):
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    
    return longest


def check_badges(df: pd.DataFrame, streaks: dict[str, Streak]) -> list[Badge]:
    """
    Check which badges have been earned.
    
    Returns:
        List of earned badges with earned_date filled in
    """
    earned = []
    
    if df.empty:
        return earned
    
    # Calculate totals
    dw_df = df[df['activity_type'].apply(lambda x: matches_categories(x, DEEP_WORK_CATEGORIES))]
    total_dw_hours = dw_df['duration_hours'].sum()
    
    flow_df = df[
        (df['activity_type'].apply(lambda x: matches_categories(x, DEEP_WORK_CATEGORIES))) &
        (df['duration_hours'] >= FLOW_MIN_DURATION_HOURS)
    ]
    total_flow_sessions = len(flow_df)
    
    sport_df = df[df['activity_type'] == 'Sport']
    total_sport_hours = sport_df['duration_hours'].sum()
    
    # Early bird / night owl
    early_sessions = len(df[df['datetime_from'].dt.hour < 7])
    late_sessions = len(df[df['datetime_from'].dt.hour >= 22])
    
    dw_streak = streaks.get("Deep Work", Streak("Deep Work"))
    
    # Check each badge
    for badge_def in BADGE_DEFINITIONS:
        badge = Badge(
            id=badge_def.id,
            name=badge_def.name,
            description=badge_def.description,
            icon=badge_def.icon,
        )
        
        # Streak badges
        if badge.id == "streak_3":
            badge.progress = min(1.0, dw_streak.longest_streak / 3)
            if dw_streak.longest_streak >= 3:
                badge.earned_date = datetime.now().date()
        elif badge.id == "streak_7":
            badge.progress = min(1.0, dw_streak.longest_streak / 7)
            if dw_streak.longest_streak >= 7:
                badge.earned_date = datetime.now().date()
        elif badge.id == "streak_14":
            badge.progress = min(1.0, dw_streak.longest_streak / 14)
            if dw_streak.longest_streak >= 14:
                badge.earned_date = datetime.now().date()
        elif badge.id == "streak_30":
            badge.progress = min(1.0, dw_streak.longest_streak / 30)
            if dw_streak.longest_streak >= 30:
                badge.earned_date = datetime.now().date()
        elif badge.id == "streak_100":
            badge.progress = min(1.0, dw_streak.longest_streak / 100)
            if dw_streak.longest_streak >= 100:
                badge.earned_date = datetime.now().date()
        
        # Volume badges
        elif badge.id == "deep_100h":
            badge.progress = min(1.0, total_dw_hours / 100)
            if total_dw_hours >= 100:
                badge.earned_date = datetime.now().date()
        elif badge.id == "deep_500h":
            badge.progress = min(1.0, total_dw_hours / 500)
            if total_dw_hours >= 500:
                badge.earned_date = datetime.now().date()
        elif badge.id == "deep_1000h":
            badge.progress = min(1.0, total_dw_hours / 1000)
            if total_dw_hours >= 1000:
                badge.earned_date = datetime.now().date()
        
        # Flow badges
        elif badge.id == "flow_10":
            badge.progress = min(1.0, total_flow_sessions / 10)
            if total_flow_sessions >= 10:
                badge.earned_date = datetime.now().date()
        elif badge.id == "flow_50":
            badge.progress = min(1.0, total_flow_sessions / 50)
            if total_flow_sessions >= 50:
                badge.earned_date = datetime.now().date()
        elif badge.id == "flow_100":
            badge.progress = min(1.0, total_flow_sessions / 100)
            if total_flow_sessions >= 100:
                badge.earned_date = datetime.now().date()
        
        # Sport badges
        elif badge.id == "sport_50h":
            badge.progress = min(1.0, total_sport_hours / 50)
            if total_sport_hours >= 50:
                badge.earned_date = datetime.now().date()
        elif badge.id == "sport_100h":
            badge.progress = min(1.0, total_sport_hours / 100)
            if total_sport_hours >= 100:
                badge.earned_date = datetime.now().date()
        
        # Special badges
        elif badge.id == "early_bird":
            badge.progress = min(1.0, early_sessions / 10)
            if early_sessions >= 10:
                badge.earned_date = datetime.now().date()
        elif badge.id == "night_owl":
            badge.progress = min(1.0, late_sessions / 10)
            if late_sessions >= 10:
                badge.earned_date = datetime.now().date()
        
        earned.append(badge)
    
    return earned


def calculate_xp(df: pd.DataFrame) -> int:
    """Calculate total XP from activity data."""
    if df.empty:
        return 0
    
    xp = 0
    
    # XP for deep work hours
    dw_df = df[df['activity_type'].apply(lambda x: matches_categories(x, DEEP_WORK_CATEGORIES))]
    dw_hours = dw_df['duration_hours'].sum()
    xp += int(dw_hours * XP_REWARDS["deep_work_hour"])
    
    # XP for flow sessions
    flow_sessions = len(dw_df[dw_df['duration_hours'] >= FLOW_MIN_DURATION_HOURS])
    xp += flow_sessions * XP_REWARDS["flow_session"]
    
    # XP for sport
    sport_sessions = len(df[df['activity_type'] == 'Sport'])
    xp += sport_sessions * XP_REWARDS["sport_session"]
    
    return xp


def get_gamification_profile(df: pd.DataFrame) -> dict:
    """
    Get complete gamification profile.
    
    Returns dict with:
    - level, xp, xp_progress
    - streaks
    - badges (earned and progress)
    - weekly_challenge
    """
    xp = calculate_xp(df)
    level, xp_current, xp_needed = calculate_level(xp)
    
    streaks = calculate_streaks(df)
    badges = check_badges(df, streaks)
    
    earned_badges = [b for b in badges if b.earned_date is not None]
    in_progress = [b for b in badges if b.earned_date is None and b.progress > 0]
    
    # Sort in-progress by closest to completion
    in_progress.sort(key=lambda b: b.progress, reverse=True)
    
    return {
        "level": level,
        "xp": xp,
        "xp_current": xp_current,
        "xp_needed": xp_needed,
        "xp_progress": xp_current / xp_needed if xp_needed > 0 else 0,
        "streaks": {
            name: {
                "current": s.current_streak,
                "longest": s.longest_streak,
                "total": s.total_completions,
            }
            for name, s in streaks.items()
        },
        "badges_earned": [
            {
                "id": b.id,
                "name": b.name,
                "description": b.description,
                "icon": b.icon,
                "earned_date": b.earned_date.isoformat() if b.earned_date else None,
            }
            for b in earned_badges
        ],
        "badges_in_progress": [
            {
                "id": b.id,
                "name": b.name,
                "description": b.description,
                "icon": b.icon,
                "progress": b.progress,
            }
            for b in in_progress[:5]  # Show top 5 closest
        ],
        "total_badges": len(earned_badges),
        "next_badge": in_progress[0] if in_progress else None,
    }
