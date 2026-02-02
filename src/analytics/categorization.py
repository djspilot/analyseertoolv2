"""
Activity Auto-Categorization Module
Learns from historical data to suggest categories for new activities.
Uses simple pattern matching and time-based heuristics.
"""

from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional
from collections import Counter
import json
from pathlib import Path

import pandas as pd

from ..logger import setup_logger

logger = setup_logger(__name__)

# Persistence
PATTERNS_FILE = Path(__file__).parent.parent.parent / "data" / "activity_patterns.json"


@dataclass
class CategorySuggestion:
    """A suggested category with confidence."""
    category: str
    confidence: float  # 0-1
    reason: str


@dataclass
class TimePattern:
    """A time-based activity pattern."""
    hour_start: int
    hour_end: int
    day_of_week: Optional[int]  # 0=Monday, None=any day
    category: str
    frequency: int


def extract_keywords(text: str) -> set[str]:
    """Extract keywords from activity comment."""
    if not text:
        return set()
    
    # Lowercase and split
    words = text.lower().split()
    
    # Remove common words
    stopwords = {'en', 'de', 'het', 'een', 'van', 'in', 'op', 'met', 'voor', 'aan', 'the', 'a', 'and', 'or'}
    
    return {w for w in words if len(w) > 2 and w not in stopwords}


class ActivityCategorizer:
    """
    Learns activity patterns and suggests categories.
    """
    
    # Keyword to category mappings (base knowledge)
    KEYWORD_CATEGORIES = {
        # Work
        'meeting': 'Work', 'vergadering': 'Work', 'call': 'Work', 'email': 'Work',
        'presentatie': 'Work', 'review': 'Work', 'planning': 'Work',
        
        # Coding
        'code': 'Coding', 'coding': 'Coding', 'debug': 'Coding', 'git': 'Coding',
        'deploy': 'Coding', 'test': 'Coding', 'refactor': 'Coding', 'api': 'Coding',
        'python': 'Coding', 'javascript': 'Coding', 'react': 'Coding',
        
        # Sport
        'gym': 'Sport', 'fitness': 'Sport', 'run': 'Sport', 'rennen': 'Sport',
        'zwemmen': 'Sport', 'fietsen': 'Sport', 'voetbal': 'Sport', 'tennis': 'Sport',
        'workout': 'Sport', 'training': 'Sport', 'sport': 'Sport',
        
        # Entertainment
        'netflix': 'Entertainment', 'film': 'Entertainment', 'movie': 'Entertainment',
        'game': 'Entertainment', 'youtube': 'Entertainment', 'tv': 'Entertainment',
        'series': 'Entertainment', 'gaming': 'Entertainment',
        
        # Housework
        'schoonmaken': 'Housework', 'koken': 'Housework', 'boodschappen': 'Housework',
        'was': 'Housework', 'stofzuigen': 'Housework', 'afwas': 'Housework',
        'cleaning': 'Housework', 'cooking': 'Housework',
        
        # Read
        'lezen': 'Read', 'boek': 'Read', 'book': 'Read', 'artikel': 'Read',
        'paper': 'Read', 'studie': 'Read', 'study': 'Read',
        
        # Yoga
        'yoga': 'Yoga', 'meditatie': 'Yoga', 'meditation': 'Yoga',
        'mindfulness': 'Yoga', 'stretching': 'Yoga',
        
        # Walking
        'wandelen': 'Walking', 'walk': 'Walking', 'hond': 'Walking',
        'dog': 'Walking', 'stappen': 'Walking',
        
        # Music
        'muziek': 'Music', 'music': 'Music', 'gitaar': 'Music', 'piano': 'Music',
        'drums': 'Music', 'oefenen': 'Music', 'practice': 'Music',
        
        # Internet
        'browse': 'Internet', 'social': 'Internet', 'twitter': 'Internet',
        'reddit': 'Internet', 'scroll': 'Internet', 'instagram': 'Internet',
    }
    
    # Time-based defaults (hour -> likely category)
    TIME_DEFAULTS = {
        (5, 7): 'Sport',      # Early morning
        (9, 12): 'Work',      # Morning work
        (12, 13): 'Other',    # Lunch
        (13, 18): 'Work',     # Afternoon work
        (18, 19): 'Housework', # Dinner prep
        (19, 21): 'Entertainment', # Evening
        (21, 23): 'Entertainment', # Late evening
    }
    
    def __init__(self):
        self.learned_patterns: dict[str, Counter] = {}  # keyword -> Counter(category)
        self.time_patterns: list[TimePattern] = []
        self._load_patterns()
    
    def _load_patterns(self):
        """Load learned patterns from file."""
        if PATTERNS_FILE.exists():
            try:
                with open(PATTERNS_FILE, 'r') as f:
                    data = json.load(f)
                    self.learned_patterns = {
                        k: Counter(v) for k, v in data.get('keyword_patterns', {}).items()
                    }
                    logger.debug(f"Loaded {len(self.learned_patterns)} learned patterns")
            except Exception as e:
                logger.warning(f"Failed to load patterns: {e}")
    
    def _save_patterns(self):
        """Save learned patterns to file."""
        try:
            PATTERNS_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'keyword_patterns': {k: dict(v) for k, v in self.learned_patterns.items()},
            }
            with open(PATTERNS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save patterns: {e}")
    
    def learn_from_data(self, df: pd.DataFrame):
        """
        Learn patterns from historical activity data.
        
        Args:
            df: DataFrame with activity_type and comment columns
        """
        if df.empty:
            return
        
        # Learn keyword -> category associations
        for _, row in df.iterrows():
            comment = row.get('comment', '')
            category = row.get('activity_type', '')
            
            if not comment or not category:
                continue
            
            keywords = extract_keywords(comment)
            for keyword in keywords:
                if keyword not in self.learned_patterns:
                    self.learned_patterns[keyword] = Counter()
                self.learned_patterns[keyword][category] += 1
        
        self._save_patterns()
        logger.info(f"Learned from {len(df)} activities, {len(self.learned_patterns)} patterns")
    
    def suggest_category(
        self,
        comment: str,
        timestamp: Optional[datetime] = None,
        duration_hours: Optional[float] = None,
    ) -> list[CategorySuggestion]:
        """
        Suggest categories for an activity.
        
        Args:
            comment: Activity description
            timestamp: When the activity occurred
            duration_hours: Duration in hours
            
        Returns:
            List of suggestions sorted by confidence
        """
        suggestions = []
        scores: Counter = Counter()
        reasons: dict[str, list[str]] = {}
        
        # 1. Keyword matching (highest weight)
        keywords = extract_keywords(comment)
        
        for keyword in keywords:
            # Check base knowledge
            if keyword in self.KEYWORD_CATEGORIES:
                cat = self.KEYWORD_CATEGORIES[keyword]
                scores[cat] += 3
                if cat not in reasons:
                    reasons[cat] = []
                reasons[cat].append(f"keyword '{keyword}'")
            
            # Check learned patterns
            if keyword in self.learned_patterns:
                most_common = self.learned_patterns[keyword].most_common(1)
                if most_common:
                    cat, count = most_common[0]
                    # Weight by frequency
                    scores[cat] += min(2, count / 5)
                    if cat not in reasons:
                        reasons[cat] = []
                    reasons[cat].append(f"learned from '{keyword}'")
        
        # 2. Time-based heuristics
        if timestamp:
            hour = timestamp.hour
            for (start, end), cat in self.TIME_DEFAULTS.items():
                if start <= hour < end:
                    scores[cat] += 0.5
                    if cat not in reasons:
                        reasons[cat] = []
                    reasons[cat].append(f"typical for {start}:00-{end}:00")
        
        # 3. Duration heuristics
        if duration_hours:
            if duration_hours >= 1.5:
                # Long sessions are often deep work
                scores['Work'] += 0.3
                scores['Coding'] += 0.3
            elif duration_hours <= 0.25:
                # Short sessions are often admin or breaks
                scores['Other'] += 0.2
        
        # Convert scores to suggestions
        if not scores:
            # No matches - return time-based default
            if timestamp:
                hour = timestamp.hour
                for (start, end), cat in self.TIME_DEFAULTS.items():
                    if start <= hour < end:
                        return [CategorySuggestion(cat, 0.3, "based on time of day")]
            return [CategorySuggestion('Other', 0.2, "no patterns matched")]
        
        total_score = sum(scores.values())
        
        for cat, score in scores.most_common(3):
            confidence = min(0.95, score / total_score) if total_score > 0 else 0.1
            reason = ", ".join(reasons.get(cat, ["pattern match"]))
            suggestions.append(CategorySuggestion(cat, confidence, reason))
        
        return suggestions
    
    def get_all_categories(self) -> list[str]:
        """Get all known categories."""
        categories = set(self.KEYWORD_CATEGORIES.values())
        for counter in self.learned_patterns.values():
            categories.update(counter.keys())
        return sorted(categories)


# Singleton
_categorizer: Optional[ActivityCategorizer] = None


def get_categorizer() -> ActivityCategorizer:
    """Get or create the global categorizer."""
    global _categorizer
    if _categorizer is None:
        _categorizer = ActivityCategorizer()
    return _categorizer


def suggest_category(
    comment: str,
    timestamp: Optional[datetime] = None,
    duration_hours: Optional[float] = None,
) -> list[dict]:
    """
    Convenience function to suggest categories.
    
    Returns:
        List of dicts with category, confidence, reason
    """
    categorizer = get_categorizer()
    suggestions = categorizer.suggest_category(comment, timestamp, duration_hours)
    
    return [
        {
            "category": s.category,
            "confidence": s.confidence,
            "reason": s.reason,
        }
        for s in suggestions
    ]


def learn_from_data(df: pd.DataFrame):
    """Convenience function to learn from data."""
    categorizer = get_categorizer()
    categorizer.learn_from_data(df)
