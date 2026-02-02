"""
Lazy Chart Loading Module
Defers chart rendering until visible using Intersection Observer pattern.
"""

from dataclasses import dataclass
from typing import Optional, Callable, Any
from datetime import datetime

import pandas as pd

from ..logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ChartRequest:
    """A pending chart render request."""
    chart_id: str
    create_func: Callable[[pd.DataFrame], Any]
    priority: int  # Lower = higher priority
    requested_at: datetime
    data_fingerprint: str


class ChartScheduler:
    """
    Schedules chart rendering to avoid blocking the UI.
    Prioritizes visible charts and defers others.
    """
    
    # Priority levels
    PRIORITY_IMMEDIATE = 0  # Currently visible
    PRIORITY_HIGH = 1       # User recently interacted
    PRIORITY_NORMAL = 2     # In viewport soon
    PRIORITY_LOW = 3        # Off-screen
    
    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self.pending: dict[str, ChartRequest] = {}
        self.rendering: set[str] = set()
        self.completed: dict[str, Any] = {}
        self._visible_charts: set[str] = set()
    
    def request_chart(
        self,
        chart_id: str,
        create_func: Callable[[pd.DataFrame], Any],
        data_fingerprint: str,
        priority: int = PRIORITY_NORMAL,
    ):
        """Queue a chart for rendering."""
        # Check if already completed with same data
        if chart_id in self.completed:
            existing = self.pending.get(chart_id)
            if existing and existing.data_fingerprint == data_fingerprint:
                return self.completed[chart_id]
        
        self.pending[chart_id] = ChartRequest(
            chart_id=chart_id,
            create_func=create_func,
            priority=priority,
            requested_at=datetime.now(),
            data_fingerprint=data_fingerprint,
        )
    
    def mark_visible(self, chart_ids: list[str]):
        """Mark charts as currently visible (prioritize them)."""
        self._visible_charts = set(chart_ids)
        
        # Upgrade priority for visible charts
        for chart_id in chart_ids:
            if chart_id in self.pending:
                self.pending[chart_id].priority = self.PRIORITY_IMMEDIATE
    
    def get_next_batch(self) -> list[ChartRequest]:
        """Get next batch of charts to render."""
        if not self.pending:
            return []
        
        # Sort by priority, then by request time
        sorted_requests = sorted(
            self.pending.values(),
            key=lambda r: (r.priority, r.requested_at),
        )
        
        # Return up to max_concurrent
        available_slots = self.max_concurrent - len(self.rendering)
        batch = sorted_requests[:available_slots]
        
        # Mark as rendering
        for req in batch:
            self.rendering.add(req.chart_id)
            del self.pending[req.chart_id]
        
        return batch
    
    def complete_chart(self, chart_id: str, result: Any):
        """Mark a chart as completed."""
        self.rendering.discard(chart_id)
        self.completed[chart_id] = result
    
    def invalidate(self, chart_id: Optional[str] = None):
        """Invalidate cached charts."""
        if chart_id:
            self.completed.pop(chart_id, None)
        else:
            self.completed.clear()
    
    def get_stats(self) -> dict:
        """Get scheduler statistics."""
        return {
            "pending": len(self.pending),
            "rendering": len(self.rendering),
            "completed": len(self.completed),
            "visible": len(self._visible_charts),
        }


# Chart groups for batch loading
CHART_GROUPS = {
    "primary": [
        "flow_vs_shallow",
        "deep_work_trend", 
        "daily_breakdown",
        "session_length",
    ],
    "secondary": [
        "heatmap",
        "circadian",
        "flow_prob",
        "flow_calendar",
    ],
    "tertiary": [
        "fragmentation",
        "sleep_pattern",
        "spiral",
        "chord",
        "violin",
        "streamgraph",
    ],
    "advanced": [
        "rose",
        "barcode",
        "energy",
        "pulse",
        "network",
        "recovery",
        "streak",
        "weekly",
        "burnout",
        "peak",
    ],
}


def get_chart_priority(chart_id: str) -> int:
    """Get default priority for a chart based on its group."""
    for group, charts in CHART_GROUPS.items():
        if chart_id in charts:
            if group == "primary":
                return ChartScheduler.PRIORITY_IMMEDIATE
            elif group == "secondary":
                return ChartScheduler.PRIORITY_HIGH
            elif group == "tertiary":
                return ChartScheduler.PRIORITY_NORMAL
    return ChartScheduler.PRIORITY_LOW


# Singleton scheduler
_scheduler: Optional[ChartScheduler] = None


def get_chart_scheduler() -> ChartScheduler:
    """Get or create the global chart scheduler."""
    global _scheduler
    if _scheduler is None:
        _scheduler = ChartScheduler()
    return _scheduler
