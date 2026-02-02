"""
Pomodoro Timer Module
Built-in focus timer with configurable work/break intervals.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Callable, List
from enum import Enum
import asyncio
import threading

from .logger import setup_logger

logger = setup_logger(__name__)


class PomodoroPhase(Enum):
    """Current phase of the pomodoro timer."""
    IDLE = "idle"
    WORK = "work"
    SHORT_BREAK = "short_break"
    LONG_BREAK = "long_break"
    PAUSED = "paused"


@dataclass
class PomodoroConfig:
    """Pomodoro timer configuration."""
    work_minutes: int = 25
    short_break_minutes: int = 5
    long_break_minutes: int = 15
    long_break_after: int = 4  # Long break after 4 pomodoros
    auto_start_breaks: bool = True
    auto_start_work: bool = False
    sound_enabled: bool = True


@dataclass
class PomodoroStats:
    """Statistics for the current session."""
    pomodoros_completed: int = 0
    total_work_minutes: int = 0
    total_break_minutes: int = 0
    current_streak: int = 0
    best_streak: int = 0
    started_at: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        return {
            "pomodoros_completed": self.pomodoros_completed,
            "total_work_minutes": self.total_work_minutes,
            "total_break_minutes": self.total_break_minutes,
            "current_streak": self.current_streak,
            "best_streak": self.best_streak,
            "started_at": self.started_at.isoformat() if self.started_at else None,
        }


@dataclass
class PomodoroSession:
    """A single pomodoro work session."""
    activity: str
    category: str = "Werk"
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    completed: bool = False
    interrupted: bool = False
    notes: str = ""


class PomodoroTimer:
    """
    Pomodoro timer with automatic phase transitions.
    
    Usage:
        timer = PomodoroTimer()
        timer.start("Deep Work", "Werk")
        
        # Timer runs automatically, call these to check status:
        status = timer.get_status()
        
        # Manual controls:
        timer.pause()
        timer.resume()
        timer.skip()
        timer.stop()
    """
    
    def __init__(
        self,
        config: Optional[PomodoroConfig] = None,
        on_phase_change: Optional[Callable[[PomodoroPhase, int], None]] = None,
        on_tick: Optional[Callable[[int], None]] = None,
        on_complete: Optional[Callable[[PomodoroSession], None]] = None,
    ):
        self.config = config or PomodoroConfig()
        self.on_phase_change = on_phase_change
        self.on_tick = on_tick
        self.on_complete = on_complete
        
        self.phase = PomodoroPhase.IDLE
        self.stats = PomodoroStats()
        self.sessions: List[PomodoroSession] = []
        self.current_session: Optional[PomodoroSession] = None
        
        self._remaining_seconds: int = 0
        self._phase_start: Optional[datetime] = None
        self._paused_at: Optional[datetime] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    def start(self, activity: str, category: str = "Werk"):
        """Start a new pomodoro work session."""
        if self.phase != PomodoroPhase.IDLE and self.phase != PomodoroPhase.PAUSED:
            logger.warning("Timer already running")
            return
        
        self.current_session = PomodoroSession(
            activity=activity,
            category=category,
            started_at=datetime.now()
        )
        self.stats.started_at = self.stats.started_at or datetime.now()
        
        self._start_phase(PomodoroPhase.WORK)
    
    def _start_phase(self, phase: PomodoroPhase):
        """Start a new phase."""
        self.phase = phase
        self._phase_start = datetime.now()
        self._running = True
        
        # Set duration based on phase
        if phase == PomodoroPhase.WORK:
            self._remaining_seconds = self.config.work_minutes * 60
        elif phase == PomodoroPhase.SHORT_BREAK:
            self._remaining_seconds = self.config.short_break_minutes * 60
        elif phase == PomodoroPhase.LONG_BREAK:
            self._remaining_seconds = self.config.long_break_minutes * 60
        
        # Notify phase change
        if self.on_phase_change:
            self.on_phase_change(phase, self._remaining_seconds)
        
        # Start timer thread
        thread = threading.Thread(target=self._run_timer, daemon=True)
        thread.start()
        
        logger.info(f"Pomodoro phase started: {phase.value} ({self._remaining_seconds}s)")
    
    def _run_timer(self):
        """Run the timer countdown."""
        while self._running and self._remaining_seconds > 0:
            if self.phase == PomodoroPhase.PAUSED:
                threading.Event().wait(0.5)
                continue
            
            threading.Event().wait(1)
            
            if self._running and self.phase != PomodoroPhase.PAUSED:
                self._remaining_seconds -= 1
                
                if self.on_tick:
                    self.on_tick(self._remaining_seconds)
        
        if self._running:
            self._phase_completed()
    
    def _phase_completed(self):
        """Handle phase completion."""
        if self.phase == PomodoroPhase.WORK:
            # Work session completed
            self.stats.pomodoros_completed += 1
            self.stats.current_streak += 1
            self.stats.best_streak = max(self.stats.best_streak, self.stats.current_streak)
            self.stats.total_work_minutes += self.config.work_minutes
            
            if self.current_session:
                self.current_session.completed = True
                self.current_session.ended_at = datetime.now()
                self.sessions.append(self.current_session)
                
                if self.on_complete:
                    self.on_complete(self.current_session)
            
            # Determine next break type
            if self.stats.pomodoros_completed % self.config.long_break_after == 0:
                next_phase = PomodoroPhase.LONG_BREAK
            else:
                next_phase = PomodoroPhase.SHORT_BREAK
            
            if self.config.auto_start_breaks:
                self._start_phase(next_phase)
            else:
                self.phase = PomodoroPhase.IDLE
                if self.on_phase_change:
                    self.on_phase_change(PomodoroPhase.IDLE, 0)
        
        elif self.phase in (PomodoroPhase.SHORT_BREAK, PomodoroPhase.LONG_BREAK):
            # Break completed
            if self.phase == PomodoroPhase.SHORT_BREAK:
                self.stats.total_break_minutes += self.config.short_break_minutes
            else:
                self.stats.total_break_minutes += self.config.long_break_minutes
            
            if self.config.auto_start_work and self.current_session:
                # Continue with same activity
                self._start_phase(PomodoroPhase.WORK)
            else:
                self.phase = PomodoroPhase.IDLE
                if self.on_phase_change:
                    self.on_phase_change(PomodoroPhase.IDLE, 0)
    
    def pause(self):
        """Pause the timer."""
        if self.phase in (PomodoroPhase.WORK, PomodoroPhase.SHORT_BREAK, PomodoroPhase.LONG_BREAK):
            self._paused_at = datetime.now()
            self.phase = PomodoroPhase.PAUSED
            logger.info("Pomodoro paused")
    
    def resume(self):
        """Resume from pause."""
        if self.phase == PomodoroPhase.PAUSED:
            # Resume to work phase
            self.phase = PomodoroPhase.WORK
            logger.info("Pomodoro resumed")
    
    def skip(self):
        """Skip to next phase."""
        self._remaining_seconds = 0
    
    def stop(self):
        """Stop the timer completely."""
        self._running = False
        
        if self.current_session and not self.current_session.completed:
            self.current_session.interrupted = True
            self.current_session.ended_at = datetime.now()
            self.sessions.append(self.current_session)
            self.stats.current_streak = 0  # Reset streak on interruption
        
        self.phase = PomodoroPhase.IDLE
        self.current_session = None
        self._remaining_seconds = 0
        
        logger.info("Pomodoro stopped")
    
    def reset_stats(self):
        """Reset session statistics."""
        self.stats = PomodoroStats()
        self.sessions = []
    
    def get_status(self) -> dict:
        """Get current timer status."""
        minutes = self._remaining_seconds // 60
        seconds = self._remaining_seconds % 60
        
        return {
            "phase": self.phase.value,
            "phase_display": self._get_phase_display(),
            "remaining_seconds": self._remaining_seconds,
            "remaining_display": f"{minutes:02d}:{seconds:02d}",
            "progress_percent": self._get_progress_percent(),
            "activity": self.current_session.activity if self.current_session else None,
            "category": self.current_session.category if self.current_session else None,
            "stats": self.stats.to_dict(),
            "is_running": self._running and self.phase != PomodoroPhase.PAUSED,
            "is_paused": self.phase == PomodoroPhase.PAUSED,
        }
    
    def _get_phase_display(self) -> str:
        """Get human-readable phase name."""
        return {
            PomodoroPhase.IDLE: "Klaar om te starten",
            PomodoroPhase.WORK: "🍅 Focus tijd",
            PomodoroPhase.SHORT_BREAK: "☕ Korte pauze",
            PomodoroPhase.LONG_BREAK: "🌴 Lange pauze",
            PomodoroPhase.PAUSED: "⏸️ Gepauzeerd",
        }.get(self.phase, "")
    
    def _get_progress_percent(self) -> float:
        """Get progress percentage for current phase."""
        if self.phase == PomodoroPhase.IDLE:
            return 0
        
        if self.phase == PomodoroPhase.WORK:
            total = self.config.work_minutes * 60
        elif self.phase == PomodoroPhase.SHORT_BREAK:
            total = self.config.short_break_minutes * 60
        elif self.phase == PomodoroPhase.LONG_BREAK:
            total = self.config.long_break_minutes * 60
        else:
            return 0
        
        elapsed = total - self._remaining_seconds
        return min(100, (elapsed / total) * 100)


# ============================================================================
# Global Timer Instance
# ============================================================================

_global_timer: Optional[PomodoroTimer] = None


def get_pomodoro_timer(
    on_phase_change: Optional[Callable] = None,
    on_tick: Optional[Callable] = None,
    on_complete: Optional[Callable] = None,
) -> PomodoroTimer:
    """Get or create the global pomodoro timer."""
    global _global_timer
    
    if _global_timer is None:
        _global_timer = PomodoroTimer(
            on_phase_change=on_phase_change,
            on_tick=on_tick,
            on_complete=on_complete,
        )
    
    return _global_timer


def reset_pomodoro_timer():
    """Reset the global timer."""
    global _global_timer
    if _global_timer:
        _global_timer.stop()
    _global_timer = None
