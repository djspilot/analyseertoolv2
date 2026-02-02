"""
Brutalist/Minimalist Time Tracking Dashboard with Sidebar Navigation
Built with Reflex - Terminal aesthetic
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")

import reflex as rx
from datetime import datetime, timedelta
from typing import Any, Optional
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add project root to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.database import init_db, get_all_activities, get_activity_count
from src.processor import (
    ingest_csv_to_db,
    calculate_metrics,
    get_category_summary,
    calculate_consistency_metrics,
    calculate_advanced_metrics,
)
from src.comparison import (
    DateRange,
    get_comparison_presets,
    compare_periods,
    filter_by_period,
    calculate_period_metrics,
    get_category_comparison,
)
from src.chart_config import (
    AVAILABLE_CHARTS,
    PreferencesManager,
    get_default_visibility,
)
from src.visualization import (
    create_gantt_chart,
    create_daily_breakdown,
    create_deep_work_trend,
    create_activity_heatmap,
    create_fragmentation_chart,
    create_sleep_pattern_chart,
    create_hourly_profile,
    create_flow_sessions_timeline,
    create_session_length_distribution,
    create_flow_probability_by_hour,
    create_flow_streak_calendar,
    create_flow_vs_shallow_chart,
    create_spiral_plot,
    create_chord_diagram,
    create_violin_plot,
    create_streamgraph,
    create_rose_chart,
    create_barcode_plot,
    create_energy_balance_chart,
    create_productivity_pulse_chart,
    create_network_graph,
    calculate_productivity_pulse,
    create_recovery_ratio_chart,
    create_flow_streak_chart,
    create_weekly_pattern_chart,
    create_burnout_risk_chart,
    create_peak_hours_chart,
    optimize_figure,
    downsample_dataframe,
)
from src.api.cache import cache_chart
from src.insights import generate_insights, get_summary_stats
from src.llm import generate_ai_insights, ask_about_data, generate_ai_insights_streaming, ask_about_data_streaming
from src.goals import load_goals, save_goals, WeeklyGoals, get_goals_summary
from src.correlations import get_correlation_insights
from src.export import generate_report_html
from src.production import RASPBERRY_PI_MODE, ProductionConfig, cleanup_if_needed
from src.analytics import get_productivity_forecast, get_gamification_profile, learn_from_data
from src.integrations import (
    start_background_sync,
    stop_background_sync,
    toggl_configured,
    clockify_configured,
    atimelogger_configured,
    atimelogger_full_sync,
    SyncConfig,
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Brutalist style constants
MONO = "JetBrains Mono, monospace"
BLACK = "#000"
WHITE = "#fff"
GREY = "#888"
LIGHT_GREY = "#f5f5f5"
BORDER_COLOR = "#ddd"
HOVER_COLOR = "#e0e0e0"

# Chart tab definitions (module-level constant)
CHART_TABS = {
    "all": [],  # Empty = show all
    "flow": ["flow_vs_shallow", "deep_work_trend", "session_length", "flow_prob", "flow_calendar", "streak", "violin"],
    "time": ["daily_breakdown", "streamgraph", "gantt", "barcode"],
    "rhythm": ["circadian", "heatmap", "spiral", "rose", "sleep_pattern", "peak"],
    "patterns": ["fragmentation", "chord", "network", "weekly"],
    "wellness": ["energy", "recovery", "burnout", "pulse"],
}


def apply_brutalist_style(fig: go.Figure) -> go.Figure:
    """Apply compact brutalist styling - no legend, tight margins, autosize."""
    fig.update_layout(
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=dict(family=MONO, color=BLACK, size=8),
        margin=dict(l=25, r=5, t=20, b=25),
        showlegend=False,  # Hide legends to save space
        title=dict(font=dict(size=9)),
        autosize=True,  # Let Plotly scale to container
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#eee",
        linewidth=1,
        linecolor=BORDER_COLOR,
        zeroline=False,
        tickfont=dict(size=7),
        title=dict(font=dict(size=8)),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#eee",
        linewidth=1,
        linecolor=BORDER_COLOR,
        zeroline=False,
        tickfont=dict(size=7),
        title=dict(font=dict(size=8)),
    )
    return fig


def create_empty_figure() -> go.Figure:
    """Create an empty Plotly figure with proper styling."""
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=dict(family=MONO, color=BLACK, size=11),
        margin=dict(l=40, r=20, t=30, b=40),
        showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
    )
    return fig


def is_empty_figure(fig: go.Figure) -> bool:
    """Check if a Plotly figure is empty (has no data traces)."""
    return len(fig.data) == 0


class State(rx.State):
    # Metrics
    total_hours: float = 0.0
    deep_work_hours: float = 0.0
    daily_avg: float = 0.0
    total_switches: int = 0
    deep_work_ratio: float = 0.0
    flow_index: float = 0.0
    avg_sleep: float = 0.0
    sleep_regularity: float = 0.0
    categories: list[dict] = []
    available_dates: list[str] = []
    selected_date: str = ""
    menu_sidebar_open: bool = False  # For CSV upload menu
    status_msg: str = ""
    
    # Chart navigation
    selected_chart: str = "flow_shallow"
    sidebar_expanded: bool = False  # Sidebar state for mobile
    
    # Active chart tab/group
    active_tab: str = "all"  # "all", "flow", "time", "rhythm", "patterns", "wellness"

    # Chart visibility toggles
    toggle_panel_open: bool = False
    visible_flow_shallow: bool = True
    visible_trend: bool = True
    visible_breakdown: bool = True
    visible_session_dist: bool = True
    visible_heatmap: bool = True
    visible_hourly: bool = True
    visible_flow_prob: bool = True
    visible_flow_calendar: bool = True
    visible_frag: bool = True
    visible_sleep: bool = True
    visible_gantt: bool = True
    visible_spiral: bool = True
    visible_chord: bool = True
    visible_violin: bool = True
    visible_streamgraph: bool = True
    visible_rose: bool = True
    visible_barcode: bool = True
    visible_energy: bool = True
    visible_pulse: bool = True
    visible_network: bool = True
    visible_recovery: bool = True
    visible_streak: bool = True
    visible_weekly: bool = True
    visible_burnout: bool = True
    visible_peak: bool = True
    
    # Fullscreen focus mode
    focused_chart: str = ""  # Empty = no focus, otherwise chart name
    productivity_pulse: float = 0.0
    
    # AI Insights
    insights_panel_open: bool = False
    insights: list[dict] = []
    
    # AI Chat
    ai_analysis: str = ""
    ai_loading: bool = False
    chat_input: str = ""
    chat_history: list[dict] = []  # [{role: "user"|"assistant", content: str}]
    suggested_questions: list[str] = [
        "Wat is mijn meest productieve dag?",
        "Hoe kan ik meer flow sessies krijgen?",
        "Welke activiteit kost me de meeste energie?",
        "Geef me 3 tips voor volgende week",
        "Analyseer mijn slaappatroon",
    ]
    
    # Goals
    goals_panel_open: bool = False
    goals_progress: list[dict] = []
    goals_overall_score: float = 0.0
    # Goal settings (editable)
    goal_deep_work: float = 20.0
    goal_sport: float = 5.0
    goal_sleep: float = 8.0
    goal_max_entertainment: float = 10.0
    goal_min_flow_sessions: int = 5
    goal_max_fragmentation: float = 2.0
    
    # Correlations
    correlations: list[dict] = []
    
    # Dark mode
    dark_mode: bool = False
    
    # Predictions & Forecasting
    predictions_panel_open: bool = False
    predictions: list[dict] = []
    burnout_risk: dict[str, Any] = {"score": 0, "level": ""}
    burnout_warnings: list[str] = []
    weekly_outlook: str = ""
    
    # Activity Log
    activity_log_open: bool = False
    activity_log: list[dict[str, Any]] = []
    activity_log_filter: str = ""  # Filter by activity type
    
    # Gamification
    gamification_panel_open: bool = False
    player_level: int = 1
    player_xp: int = 0
    xp_progress: float = 0.0
    xp_needed: int = 100
    streaks: list[dict[str, Any]] = []
    badges_earned: list[dict] = []
    badges_in_progress: list[dict] = []
    
    # Real-time Sync
    sync_enabled: bool = True
    sync_interval: int = 15 if RASPBERRY_PI_MODE else 5  # minutes (longer on Pi)
    sync_status: str = "Niet verbonden"
    sync_is_syncing: bool = False
    last_sync_time: str = ""
    sync_entries_count: int = 0
    integrations_configured: list[str] = []
    eco_mode: bool = RASPBERRY_PI_MODE  # Low power mode (auto-enabled on Pi)
    
    # Pomodoro Timer
    pomodoro_open: bool = False
    pomodoro_phase: str = "idle"  # idle, work, short_break, long_break, paused
    pomodoro_remaining: str = "25:00"
    pomodoro_progress: float = 0.0
    pomodoro_activity: str = "Deep Work"
    pomodoro_category: str = "Werk"
    pomodoro_completed: int = 0
    pomodoro_streak: int = 0
    pomodoro_is_running: bool = False
    
    # Comparison mode
    comparison_mode: bool = False
    comparison_preset: str = "week"  # "week", "month", "custom"
    
    # Comparison metrics
    comp_total_hours_current: float = 0.0
    comp_total_hours_previous: float = 0.0
    comp_deep_work_current: float = 0.0
    comp_deep_work_previous: float = 0.0
    comp_flow_index_current: float = 0.0
    comp_flow_index_previous: float = 0.0
    comp_pulse_current: float = 0.0
    comp_pulse_previous: float = 0.0

    # Chart display labels
    CHART_LABELS = {
        "flow_shallow": "Flow vs Shallow",
        "session_dist": "Session Distribution",
        "flow_prob": "Flow Probability",
        "flow_calendar": "Flow Calendar",
        "daily_breakdown": "Daily Breakdown",
        "dw_trend": "Deep Work Trend",
        "activity_heatmap": "Activity Heatmap",
        "circadian_profile": "Circadian Profile",
        "fragmentation": "Fragmentation",
        "sleep_pattern": "Sleep Pattern",
        "gantt": "Daily Timeline",
    }
    
    # Time range slider
    range_start: float = 0
    range_end: float = 100
    range_label: str = "ALL TIME"
    total_days: int = 0
    
    # Chart data
    chart_breakdown: go.Figure = create_empty_figure()
    chart_trend: go.Figure = create_empty_figure()
    chart_heatmap: go.Figure = create_empty_figure()
    chart_hourly: go.Figure = create_empty_figure()
    chart_frag: go.Figure = create_empty_figure()
    chart_sleep: go.Figure = create_empty_figure()
    chart_flow_shallow: go.Figure = create_empty_figure()
    chart_session_dist: go.Figure = create_empty_figure()
    chart_flow_prob: go.Figure = create_empty_figure()
    chart_flow_calendar: go.Figure = create_empty_figure()
    chart_gantt: go.Figure = create_empty_figure()
    chart_spiral: go.Figure = create_empty_figure()
    chart_chord: go.Figure = create_empty_figure()
    chart_violin: go.Figure = create_empty_figure()
    chart_streamgraph: go.Figure = create_empty_figure()
    chart_rose: go.Figure = create_empty_figure()
    chart_barcode: go.Figure = create_empty_figure()
    chart_energy: go.Figure = create_empty_figure()
    chart_pulse: go.Figure = create_empty_figure()
    chart_network: go.Figure = create_empty_figure()
    chart_recovery: go.Figure = create_empty_figure()
    chart_streak: go.Figure = create_empty_figure()
    chart_weekly: go.Figure = create_empty_figure()
    chart_burnout: go.Figure = create_empty_figure()
    chart_peak: go.Figure = create_empty_figure()
    
    _df: pd.DataFrame = pd.DataFrame()

    def load_data(self):
        init_db()
        df = get_all_activities()
        if df.empty:
            self._df = df
            # Still start sync even with no data
            self._start_background_sync()
            return

        df = calculate_metrics(df)
        self._df = df
        
        # Learn activity patterns for auto-categorization
        learn_from_data(df)

        # Calculate total days range
        if not df.empty:
            min_date = df["datetime_from"].min()
            max_date = df["datetime_from"].max()
            self.total_days = (max_date - min_date).days + 1

        self.range_start = 0
        self.range_end = 100
        self._load_goals()
        self._update_goals_progress()
        self._update_correlations()
        self._update_predictions()
        self._update_gamification()
        self._update_filtered_view()
        
        # Start automatic background sync
        self._start_background_sync()

    def set_range(self, values: list[float]):
        if len(values) >= 2:
            self.range_start = values[0]
            self.range_end = values[1]
            self._update_filtered_view()



    def _update_filtered_view(self):
        df = self._df
        if df.empty:
            return

        min_date = df["datetime_from"].min()
        max_date = df["datetime_from"].max()
        total_range = (max_date - min_date).total_seconds()

        start_offset = timedelta(seconds=total_range * self.range_start / 100)
        end_offset = timedelta(seconds=total_range * self.range_end / 100)

        filter_start = min_date + start_offset
        filter_end = min_date + end_offset

        filtered = df[(df["datetime_from"] >= filter_start) & (df["datetime_from"] <= filter_end)]

        if filtered.empty:
            filtered = df

        days_selected = (filter_end - filter_start).days + 1
        if self.range_start == 0 and self.range_end == 100:
            self.range_label = "ALL TIME"
        else:
            self.range_label = f"{days_selected} DAYS"

        self._update_metrics(filtered)
        self._update_charts(filtered)
        self._update_insights(filtered)

    def _update_metrics(self, filtered):
        if filtered.empty:
            self.total_hours = 0.0
            self.deep_work_hours = 0.0
            self.daily_avg = 0.0
            self.total_switches = 0
            self.deep_work_ratio = 0.0
            self.flow_index = 0.0
            self.avg_sleep = 0.0
            self.sleep_regularity = 0.0
            self.categories = []
            return

        self.total_hours = round(filtered["duration_hours"].sum(), 1)
        self.deep_work_hours = round(filtered[filtered["is_deep_work"] == 1]["duration_hours"].sum(), 1)

        days = filtered["date"].nunique()
        self.daily_avg = round(self.total_hours / days, 1) if days > 0 else 0.0

        switches = calculate_consistency_metrics(filtered)
        self.total_switches = int(switches["context_switches"].sum())

        advanced = calculate_advanced_metrics(filtered)
        self.deep_work_ratio = advanced['deep_work_ratio']
        self.flow_index = advanced['flow_index']
        self.avg_sleep = advanced['sleep_regularity']['avg_sleep_hours']
        self.sleep_regularity = advanced['sleep_regularity']['sri']

        cat_df = get_category_summary(filtered)
        self.categories = cat_df.to_dict("records")
        
        self.productivity_pulse = calculate_productivity_pulse(filtered)

        dates = sorted(filtered["date"].unique(), reverse=True)
        self.available_dates = [str(d) for d in dates]
        if self.available_dates:
            self.selected_date = self.available_dates[0]

    def _update_charts(self, filtered):
        """Update all charts with filtered data. Applies optimizations for performance."""
        # Downsample data for Raspberry Pi
        if RASPBERRY_PI_MODE:
            filtered = downsample_dataframe(filtered, ProductionConfig.MAX_DATA_POINTS)
        
        # Helper to style and optionally optimize charts
        def style_chart(fig):
            fig = apply_brutalist_style(fig)
            if RASPBERRY_PI_MODE:
                fig = optimize_figure(fig)
            return fig
        
        # Primary charts (always loaded)
        fig = create_daily_breakdown(filtered, last_n_days=None)
        self.chart_breakdown = style_chart(fig)

        fig = create_deep_work_trend(filtered, window=7)
        self.chart_trend = style_chart(fig)

        fig = create_flow_vs_shallow_chart(filtered)
        self.chart_flow_shallow = style_chart(fig)

        fig = create_session_length_distribution(filtered)
        self.chart_session_dist = style_chart(fig)

        # Secondary charts
        fig = create_activity_heatmap(filtered)
        self.chart_heatmap = style_chart(fig)

        fig = create_hourly_profile(filtered)
        self.chart_hourly = style_chart(fig)

        fig = create_fragmentation_chart(filtered)
        self.chart_frag = style_chart(fig)

        fig = create_sleep_pattern_chart(filtered)
        self.chart_sleep = style_chart(fig)

        fig = create_flow_probability_by_hour(filtered)
        self.chart_flow_prob = style_chart(fig)

        fig = create_flow_streak_calendar(filtered)
        self.chart_flow_calendar = style_chart(fig)

        fig = create_spiral_plot(filtered)
        self.chart_spiral = style_chart(fig)

        fig = create_chord_diagram(filtered)
        self.chart_chord = style_chart(fig)

        fig = create_violin_plot(filtered)
        self.chart_violin = style_chart(fig)

        fig = create_streamgraph(filtered)
        self.chart_streamgraph = style_chart(fig)

        fig = create_rose_chart(filtered)
        self.chart_rose = style_chart(fig)

        fig = create_barcode_plot(filtered)
        self.chart_barcode = style_chart(fig)

        fig = create_energy_balance_chart(filtered)
        self.chart_energy = style_chart(fig)

        fig = create_productivity_pulse_chart(filtered)
        self.chart_pulse = style_chart(fig)

        fig = create_network_graph(filtered)
        self.chart_network = style_chart(fig)

        fig = create_recovery_ratio_chart(filtered)
        self.chart_recovery = style_chart(fig)

        fig = create_flow_streak_chart(filtered)
        self.chart_streak = style_chart(fig)

        fig = create_weekly_pattern_chart(filtered)
        self.chart_weekly = style_chart(fig)

        fig = create_burnout_risk_chart(filtered)
        self.chart_burnout = style_chart(fig)

        fig = create_peak_hours_chart(filtered)
        self.chart_peak = style_chart(fig)

        if self.available_dates:
            date_obj = datetime.strptime(self.available_dates[0], "%Y-%m-%d")
            fig = create_gantt_chart(filtered, date_obj)
            self.chart_gantt = style_chart(fig)
        
        # Cleanup memory on Raspberry Pi
        if RASPBERRY_PI_MODE:
            cleanup_if_needed()

    def set_date(self, val: str):
        self.selected_date = val
        if self._df is not None and not self._df.empty and val:
            date_obj = datetime.strptime(val, "%Y-%m-%d")
            fig = create_gantt_chart(self._df, date_obj)
            self.chart_gantt = apply_brutalist_style(fig)

    def set_chart(self, chart_id: str):
        self.selected_chart = chart_id

    def toggle_menu_sidebar(self):
        self.menu_sidebar_open = not self.menu_sidebar_open
    
    def toggle_sidebar(self):
        self.sidebar_expanded = not self.sidebar_expanded

    def toggle_chart_panel(self):
        self.toggle_panel_open = not self.toggle_panel_open

    def toggle_flow_shallow(self):
        self.visible_flow_shallow = not self.visible_flow_shallow

    def toggle_trend(self):
        self.visible_trend = not self.visible_trend

    def toggle_breakdown(self):
        self.visible_breakdown = not self.visible_breakdown

    def toggle_session_dist(self):
        self.visible_session_dist = not self.visible_session_dist

    def toggle_heatmap(self):
        self.visible_heatmap = not self.visible_heatmap

    def toggle_hourly(self):
        self.visible_hourly = not self.visible_hourly

    def toggle_flow_prob(self):
        self.visible_flow_prob = not self.visible_flow_prob

    def toggle_flow_calendar(self):
        self.visible_flow_calendar = not self.visible_flow_calendar

    def toggle_frag(self):
        self.visible_frag = not self.visible_frag

    def toggle_sleep(self):
        self.visible_sleep = not self.visible_sleep

    def toggle_gantt(self):
        self.visible_gantt = not self.visible_gantt

    def toggle_spiral(self):
        self.visible_spiral = not self.visible_spiral

    def toggle_chord(self):
        self.visible_chord = not self.visible_chord

    def toggle_violin(self):
        self.visible_violin = not self.visible_violin

    def toggle_streamgraph(self):
        self.visible_streamgraph = not self.visible_streamgraph

    def toggle_rose(self):
        self.visible_rose = not self.visible_rose

    def toggle_barcode(self):
        self.visible_barcode = not self.visible_barcode

    def toggle_energy(self):
        self.visible_energy = not self.visible_energy

    def toggle_pulse(self):
        self.visible_pulse = not self.visible_pulse

    def toggle_network(self):
        self.visible_network = not self.visible_network

    def toggle_recovery(self):
        self.visible_recovery = not self.visible_recovery

    def toggle_streak(self):
        self.visible_streak = not self.visible_streak

    def toggle_weekly(self):
        self.visible_weekly = not self.visible_weekly

    def toggle_burnout(self):
        self.visible_burnout = not self.visible_burnout

    def toggle_peak(self):
        self.visible_peak = not self.visible_peak

    def toggle_insights_panel(self):
        self.insights_panel_open = not self.insights_panel_open

    def toggle_goals_panel(self):
        self.goals_panel_open = not self.goals_panel_open

    def _load_goals(self):
        """Load goals from file and update state."""
        goals = load_goals()
        self.goal_deep_work = goals.deep_work_hours_weekly
        self.goal_sport = goals.sport_hours_weekly
        self.goal_sleep = goals.sleep_hours_daily
        self.goal_max_entertainment = goals.max_entertainment_hours_weekly
        self.goal_min_flow_sessions = goals.min_flow_sessions_weekly
        self.goal_max_fragmentation = goals.max_fragmentation

    def _update_goals_progress(self):
        """Calculate progress towards goals."""
        if self._df.empty:
            self.goals_progress = []
            self.goals_overall_score = 0.0
            return
        
        summary = get_goals_summary(self._df)
        self.goals_progress = summary["goals"]
        self.goals_overall_score = summary["overall_score"]

    def save_goals(self):
        """Save current goal settings."""
        goals = WeeklyGoals(
            deep_work_hours_weekly=self.goal_deep_work,
            sport_hours_weekly=self.goal_sport,
            sleep_hours_daily=self.goal_sleep,
            max_entertainment_hours_weekly=self.goal_max_entertainment,
            min_flow_sessions_weekly=self.goal_min_flow_sessions,
            max_fragmentation=self.goal_max_fragmentation,
        )
        save_goals(goals)
        self._update_goals_progress()

    def set_goal_deep_work(self, value: str):
        try:
            self.goal_deep_work = float(value)
        except ValueError:
            pass

    def set_goal_sport(self, value: str):
        try:
            self.goal_sport = float(value)
        except ValueError:
            pass

    def set_goal_sleep(self, value: str):
        try:
            self.goal_sleep = float(value)
        except ValueError:
            pass

    def set_goal_max_entertainment(self, value: str):
        try:
            self.goal_max_entertainment = float(value)
        except ValueError:
            pass

    def set_goal_min_flow_sessions(self, value: str):
        try:
            self.goal_min_flow_sessions = int(value)
        except ValueError:
            pass

    def set_goal_max_fragmentation(self, value: str):
        try:
            self.goal_max_fragmentation = float(value)
        except ValueError:
            pass

    def _update_insights(self, filtered):
        """Generate AI insights from the filtered data."""
        self.insights = generate_insights(filtered)

    def _update_correlations(self):
        """Calculate correlations from the full dataset."""
        if self._df.empty:
            self.correlations = []
            return
        self.correlations = get_correlation_insights(self._df)

    def _update_predictions(self):
        """Update predictions and forecasts."""
        if self._df.empty:
            self.predictions = []
            self.burnout_risk = {"score": 0, "level": ""}
            self.burnout_warnings = []
            self.weekly_outlook = ""
            return
        
        forecast = get_productivity_forecast(self._df)
        self.predictions = forecast["predictions"]
        burnout = forecast["burnout"]
        self.burnout_risk = {"score": burnout.get("score", 0), "level": burnout.get("level", "")}
        self.burnout_warnings = burnout.get("warnings", [])
        self.weekly_outlook = forecast["weekly_outlook"]

    def _update_gamification(self):
        """Update gamification stats."""
        if self._df.empty:
            self.player_level = 1
            self.player_xp = 0
            self.xp_progress = 0.0
            self.xp_needed = 100
            self.streaks = []
            self.badges_earned = []
            self.badges_in_progress = []
            return
        
        profile = get_gamification_profile(self._df)
        self.player_level = profile["level"]
        self.player_xp = profile["xp"]
        self.xp_progress = profile["xp_progress"]
        self.xp_needed = profile["xp_needed"]
        raw_streaks = profile["streaks"]
        self.streaks = [{"name": k, "current": v["current"], "longest": v["longest"]} for k, v in raw_streaks.items()]
        self.badges_earned = profile["badges_earned"]
        raw_progress = profile["badges_in_progress"]
        self.badges_in_progress = [
            {**b, "progress_pct": int(b.get("progress", 0) * 100)} for b in raw_progress
        ]

    def _start_background_sync(self):
        """Start automatic background sync with integrations."""
        from datetime import datetime
        
        # Check which integrations are configured
        configured = []
        if atimelogger_configured():
            configured.append("aTimeLogger")
        if toggl_configured():
            configured.append("Toggl")
        if clockify_configured():
            configured.append("Clockify")
        self.integrations_configured = configured
        
        if not configured:
            self.sync_status = "Geen integratie geconfigureerd"
            return
        
        # Use eco mode settings (auto on Pi or manual toggle)
        use_eco = self.eco_mode or RASPBERRY_PI_MODE
        
        if use_eco:
            sync_config = SyncConfig.for_raspberry_pi()
            sync_config.interval_minutes = self.sync_interval
            self.sync_status = f"🍃 Eco ({self.sync_interval}min)"
        else:
            sync_config = SyncConfig(
                interval_minutes=self.sync_interval,
                low_power_mode=False,
            )
            self.sync_status = f"Auto-sync ({', '.join(configured)})"
        
        # Define callbacks
        def on_sync_complete(entries: list[dict]):
            """Called when sync completes with new entries."""
            if entries:
                self.sync_entries_count = len(entries)
                self.last_sync_time = datetime.now().strftime("%H:%M")
                # Run memory cleanup in eco mode
                if use_eco:
                    cleanup_if_needed()
                # Reload data to include new entries
                self.load_data()
        
        def on_status_change(status):
            """Called when sync status changes."""
            if status.is_syncing:
                self.sync_status = "Synchroniseren..."
                self.sync_is_syncing = True
            elif status.last_success:
                mode = "🍃" if use_eco else "☁️"
                self.sync_status = f"{mode} {status.last_sync.strftime('%H:%M')} ({self.sync_interval}m)" if status.last_sync else "Klaar"
                self.last_sync_time = status.last_sync.strftime("%H:%M") if status.last_sync else ""
                self.sync_entries_count = status.entries_synced
                self.sync_is_syncing = False
            else:
                self.sync_status = f"Fout: {status.last_error}"
                self.sync_is_syncing = False
        
        # Start background sync with config
        try:
            from src.integrations.background_sync import BackgroundSyncService
            
            service = BackgroundSyncService(
                config=sync_config,
                on_sync_complete=on_sync_complete,
                on_status_change=on_status_change,
            )
            service.start()
        except Exception as e:
            self.sync_status = f"Kon sync niet starten: {e}"
    
    def toggle_sync(self):
        """Toggle automatic sync on/off."""
        self.sync_enabled = not self.sync_enabled
        if self.sync_enabled:
            self._start_background_sync()
        else:
            stop_background_sync()
            self.sync_status = "Auto-sync uitgeschakeld"
    
    def set_sync_interval(self, interval: int):
        """Set sync interval in minutes."""
        self.sync_interval = max(1, min(60, interval))
        if self.sync_enabled:
            stop_background_sync()
            self._start_background_sync()
    
    def toggle_eco_mode(self):
        """Toggle eco/low-power mode for Pi optimization."""
        self.eco_mode = not self.eco_mode
        if self.eco_mode:
            # Enable low power settings
            self.sync_interval = 15
            self.status_msg = "🍃 Eco-mode ingeschakeld (15min sync, minder CPU)"
        else:
            self.sync_interval = 5
            self.status_msg = "⚡ Normale mode (5min sync)"
        
        # Restart sync with new settings
        if self.sync_enabled:
            stop_background_sync()
            self._start_background_sync()

    def toggle_predictions_panel(self):
        """Toggle predictions panel."""
        self.predictions_panel_open = not self.predictions_panel_open

    def toggle_gamification_panel(self):
        """Toggle gamification panel."""
        self.gamification_panel_open = not self.gamification_panel_open

    def toggle_activity_log(self):
        """Toggle activity log panel and load data."""
        self.activity_log_open = not self.activity_log_open
        if self.activity_log_open:
            self._load_activity_log()

    def _load_activity_log(self):
        """Load activity log from database."""
        df = get_all_activities()
        if df.empty:
            self.activity_log = []
            return
        
        # Sort by most recent first
        df = df.sort_values('datetime_from', ascending=False)
        
        # Convert to list of dicts for display
        self.activity_log = [
            {
                "id": int(row.get("id", i)),
                "activity": str(row["activity_type"]),
                "start": row["datetime_from"].strftime("%d %b %H:%M") if hasattr(row["datetime_from"], "strftime") else str(row["datetime_from"])[:16],
                "end": row["datetime_to"].strftime("%H:%M") if hasattr(row["datetime_to"], "strftime") else str(row["datetime_to"])[11:16],
                "duration": f"{row['duration_hours']:.1f}h",
                "date": row["datetime_from"].strftime("%Y-%m-%d") if hasattr(row["datetime_from"], "strftime") else str(row["datetime_from"])[:10],
            }
            for i, row in df.head(200).iterrows()  # Limit to 200 most recent
        ]

    def set_activity_log_filter(self, value: str):
        """Set filter for activity log."""
        self.activity_log_filter = value

    async def generate_ai_analysis(self):
        """Generate comprehensive AI analysis from the data with streaming."""
        if self._df.empty:
            self.ai_analysis = "No data available. Please upload a CSV file first."
            return
        
        self.ai_loading = True
        self.ai_analysis = ""
        yield
        
        # Stream the response
        async for chunk in generate_ai_insights_streaming(self._df):
            self.ai_analysis += chunk
            yield
        
        self.ai_loading = False

    def set_chat_input(self, value: str):
        self.chat_input = value

    async def send_chat_message(self):
        """Send a question about the data with streaming."""
        if not self.chat_input.strip():
            return
        
        question = self.chat_input
        self.chat_input = ""
        self.chat_history = self.chat_history + [{"role": "user", "content": question}]
        self.ai_loading = True
        yield
        
        # Stream the response
        response = ""
        async for chunk in ask_about_data_streaming(self._df, question):
            response += chunk
            # Update the last message in chat history with accumulated response
            self.chat_history = self.chat_history[:-1] + [{"role": "user", "content": question}] if not response else self.chat_history
            if len(self.chat_history) == 0 or self.chat_history[-1]["role"] == "user":
                self.chat_history = self.chat_history + [{"role": "assistant", "content": response}]
            else:
                self.chat_history = self.chat_history[:-1] + [{"role": "assistant", "content": response}]
            yield
        
        self.ai_loading = False

    def clear_chat(self):
        """Clear chat history."""
        self.chat_history = []
        self.ai_analysis = ""

    def use_suggestion(self, question: str):
        """Use a suggested question."""
        self.chat_input = question

    @rx.var
    def report_download_url(self) -> str:
        """Generate a data URL for the report download."""
        if self._df.empty:
            return ""
        html = generate_report_html(self._df, self.ai_analysis, self.range_label)
        import base64
        b64 = base64.b64encode(html.encode()).decode()
        return f"data:text/html;base64,{b64}"

    @rx.var
    def xp_remaining(self) -> int:
        """XP remaining to next level."""
        return self.xp_needed - int(self.xp_progress * self.xp_needed)

    @rx.var
    def next_level(self) -> int:
        """Next player level."""
        return self.player_level + 1

    def toggle_dark_mode(self):
        """Toggle dark mode."""
        self.dark_mode = not self.dark_mode

    def focus_chart(self, chart_name: str):
        """Toggle fullscreen focus on a chart."""
        if self.focused_chart == chart_name:
            self.focused_chart = ""
        else:
            self.focused_chart = chart_name

    def close_focus(self):
        """Close the focused chart overlay."""
        self.focused_chart = ""

    def handle_key_down(self, key: dict):
        """Handle keyboard events for navigation and shortcuts."""
        key_pressed = key.get("key", "")
        ctrl = key.get("ctrlKey", False)
        meta = key.get("metaKey", False)  # Cmd on Mac
        modifier = ctrl or meta
        
        # ESC - Close panels/overlays
        if key_pressed == "Escape":
            if self.focused_chart:
                self.focused_chart = ""
            elif self.insights_panel_open:
                self.insights_panel_open = False
            elif self.goals_panel_open:
                self.goals_panel_open = False
            elif self.predictions_panel_open:
                self.predictions_panel_open = False
            elif self.gamification_panel_open:
                self.gamification_panel_open = False
            elif self.menu_sidebar_open:
                self.menu_sidebar_open = False
            elif self.toggle_panel_open:
                self.toggle_panel_open = False
        
        # Keyboard shortcuts with modifier (Ctrl/Cmd)
        elif modifier:
            if key_pressed == "g":
                self.goals_panel_open = not self.goals_panel_open
            elif key_pressed == "i":
                self.insights_panel_open = not self.insights_panel_open
            elif key_pressed == "p":
                self.predictions_panel_open = not self.predictions_panel_open
            elif key_pressed == "t":
                self.gamification_panel_open = not self.gamification_panel_open
            elif key_pressed == "d":
                self.dark_mode = not self.dark_mode
            elif key_pressed == "u":
                self.menu_sidebar_open = not self.menu_sidebar_open
        
        # Tab switching with numbers (1-6)
        elif key_pressed in ["1", "2", "3", "4", "5", "6"]:
            tabs = ["all", "flow", "time", "rhythm", "patterns", "wellness"]
            idx = int(key_pressed) - 1
            if idx < len(tabs):
                self.active_tab = tabs[idx]

    def set_tab(self, tab: str):
        """Switch to a chart tab/group."""
        self.active_tab = tab

    def is_chart_in_tab(self, chart_key: str) -> bool:
        """Check if a chart should be visible in the current tab."""
        if self.active_tab == "all":
            return True
        tab_charts = self.CHART_TABS.get(self.active_tab, {}).get("charts", [])
        return chart_key in tab_charts

    def toggle_comparison(self):
        """Toggle comparison mode on/off."""
        self.comparison_mode = not self.comparison_mode
        if self.comparison_mode:
            self._calculate_comparison()

    def set_comparison_preset(self, preset: str):
        """Set comparison preset (week, month)."""
        self.comparison_preset = preset
        self._calculate_comparison()

    def _calculate_comparison(self):
        """Calculate comparison metrics between current and previous period."""
        if self._df.empty:
            return
            
        from datetime import timedelta
        
        df = self._df
        max_date = df["datetime_from"].max()
        
        if self.comparison_preset == "week":
            period_days = 7
        else:  # month
            period_days = 30
            
        # Current period
        current_start = max_date - timedelta(days=period_days)
        current_df = df[df["datetime_from"] >= current_start]
        
        # Previous period
        previous_end = current_start
        previous_start = previous_end - timedelta(days=period_days)
        previous_df = df[(df["datetime_from"] >= previous_start) & (df["datetime_from"] < previous_end)]
        
        # Calculate metrics for current
        self.comp_total_hours_current = round(current_df["duration_hours"].sum(), 1)
        self.comp_deep_work_current = round(
            current_df[current_df["is_deep_work"] == 1]["duration_hours"].sum(), 1
        )
        flow_sessions = current_df[(current_df["is_deep_work"] == 1) & (current_df["duration_hours"] >= 1.5)]
        deep_sessions = current_df[current_df["is_deep_work"] == 1]
        self.comp_flow_index_current = round(
            (len(flow_sessions) / len(deep_sessions) * 100) if len(deep_sessions) > 0 else 0, 1
        )
        self.comp_pulse_current = calculate_productivity_pulse(current_df)
        
        # Calculate metrics for previous
        self.comp_total_hours_previous = round(previous_df["duration_hours"].sum(), 1)
        self.comp_deep_work_previous = round(
            previous_df[previous_df["is_deep_work"] == 1]["duration_hours"].sum(), 1
        )
        flow_sessions_prev = previous_df[(previous_df["is_deep_work"] == 1) & (previous_df["duration_hours"] >= 1.5)]
        deep_sessions_prev = previous_df[previous_df["is_deep_work"] == 1]
        self.comp_flow_index_previous = round(
            (len(flow_sessions_prev) / len(deep_sessions_prev) * 100) if len(deep_sessions_prev) > 0 else 0, 1
        )
        self.comp_pulse_previous = calculate_productivity_pulse(previous_df)

    async def handle_upload(self, files: list[rx.UploadFile]):
        for file in files:
            data = await file.read()
            path = Path(rx.get_upload_dir()) / file.filename
            path.write_bytes(data)
            try:
                count = ingest_csv_to_db(str(path))
                self.status_msg = f"OK: {count} records"
                self.load_data()
            except Exception as e:
                self.status_msg = f"ERR: {e}"


def stat_item(label: str, value, suffix: str = "") -> rx.Component:
    """Ultra compact inline stat."""
    return rx.hstack(
        rx.text(f"{label}:", font_size="9px", color=GREY),
        rx.text(value, font_size="11px", font_weight="700"),
        rx.text(suffix, font_size="9px", color=GREY) if suffix else rx.fragment(),
        spacing="1",
        align_items="baseline",
    )


def is_chart_visible_in_tab(chart_key: str, active_tab: str) -> bool:
    """Check if chart should be visible based on active tab."""
    if active_tab == "all":
        return True
    tab_charts = {
        "flow": ["flow_vs_shallow", "deep_work_trend", "session_length", "flow_prob", "flow_calendar", "streak", "violin"],
        "time": ["daily_breakdown", "streamgraph", "gantt", "barcode"],
        "rhythm": ["circadian", "heatmap", "spiral", "rose", "sleep_pattern", "peak"],
        "patterns": ["fragmentation", "chord", "network", "weekly"],
        "wellness": ["energy", "recovery", "burnout", "pulse"],
    }
    return chart_key in tab_charts.get(active_tab, [])


def toggleable_chart(title: str, chart_data, is_visible, chart_key: str = "") -> rx.Component:
    """Chart wrapper that hides via display:none when not visible or not in active tab. Click to focus."""
    key = chart_key or title.lower().replace(" ", "_")
    
    # Determine which tabs this chart belongs to (at component creation time)
    belongs_to_tabs = ["all"]  # Always show in "all"
    if key in CHART_TABS["flow"]:
        belongs_to_tabs.append("flow")
    if key in CHART_TABS["time"]:
        belongs_to_tabs.append("time")
    if key in CHART_TABS["rhythm"]:
        belongs_to_tabs.append("rhythm")
    if key in CHART_TABS["patterns"]:
        belongs_to_tabs.append("patterns")
    if key in CHART_TABS["wellness"]:
        belongs_to_tabs.append("wellness")
    
    # Build the visibility condition: is_visible AND (active_tab in belongs_to_tabs)
    tab_conditions = [State.active_tab == tab for tab in belongs_to_tabs]
    tab_visible = tab_conditions[0]
    for cond in tab_conditions[1:]:
        tab_visible = tab_visible | cond
    
    return rx.box(
        rx.hstack(
            rx.text(title.upper(), font_size="8px", color=GREY, letter_spacing="0.05em"),
            rx.spacer(),
            rx.text("⤢", font_size="10px", color=GREY, cursor="pointer",
                   on_click=lambda: State.focus_chart(key),
                   title="Click to focus"),
            spacing="1",
            width="100%",
            margin_bottom="2px",
        ),
        rx.cond(
            chart_data != None,
            rx.plotly(data=chart_data, height="100%", style={"min_height": "120px"}),
            rx.text("—", color=GREY, font_size="10px"),
        ),
        background=LIGHT_GREY,
        border=f"1px solid {BORDER_COLOR}",
        padding="4px",
        font_family=MONO,
        overflow="hidden",
        display=rx.cond(
            is_visible & tab_visible,
            "flex",
            "none"
        ),
        flex_direction="column",
    )


def grid_chart(chart_data, is_visible, chart_key: str = "") -> rx.Component:
    """Chart wrapper for main grid view - no title bar, no toolbar, clean display.
    Double-click to open in fullscreen mode. Respects active tab filtering."""
    
    # Determine which tabs this chart belongs to (at component creation time)
    belongs_to_tabs = ["all"]  # Always show in "all"
    if chart_key in CHART_TABS["flow"]:
        belongs_to_tabs.append("flow")
    if chart_key in CHART_TABS["time"]:
        belongs_to_tabs.append("time")
    if chart_key in CHART_TABS["rhythm"]:
        belongs_to_tabs.append("rhythm")
    if chart_key in CHART_TABS["patterns"]:
        belongs_to_tabs.append("patterns")
    if chart_key in CHART_TABS["wellness"]:
        belongs_to_tabs.append("wellness")
    
    # Build the visibility condition: is_visible AND (active_tab in belongs_to_tabs)
    tab_conditions = [State.active_tab == tab for tab in belongs_to_tabs]
    tab_visible = tab_conditions[0]
    for cond in tab_conditions[1:]:
        tab_visible = tab_visible | cond
    
    return rx.box(
        rx.cond(
            chart_data != None,
            rx.plotly(data=chart_data, height="100%", style={"min_height": "120px"},
                      config={"displayModeBar": False}),
            rx.text("—", color=GREY, font_size="10px"),
        ),
        background=LIGHT_GREY,
        border=f"1px solid {BORDER_COLOR}",
        padding="4px",
        font_family=MONO,
        overflow="hidden",
        display=rx.cond(is_visible & tab_visible, "flex", "none"),
        flex_direction="column",
        cursor="pointer",
        on_double_click=lambda: State.focus_chart(chart_key),
    )


def main_content() -> rx.Component:
    """Bento-box grid layout with toggleable charts - no titles, no toolbar.
    Double-click any chart to open in fullscreen mode."""
    return rx.box(
        # Bento-box grid with auto-fit for dynamic sizing
        rx.box(
            # Each chart uses display:none when hidden (removes from grid flow)
            grid_chart(State.chart_flow_shallow, State.visible_flow_shallow, "flow_vs_shallow"),
            grid_chart(State.chart_trend, State.visible_trend, "deep_work_trend"),
            grid_chart(State.chart_breakdown, State.visible_breakdown, "daily_breakdown"),
            grid_chart(State.chart_session_dist, State.visible_session_dist, "session_length"),
            grid_chart(State.chart_heatmap, State.visible_heatmap, "heatmap"),
            grid_chart(State.chart_hourly, State.visible_hourly, "circadian"),
            grid_chart(State.chart_flow_prob, State.visible_flow_prob, "flow_prob"),
            grid_chart(State.chart_flow_calendar, State.visible_flow_calendar, "flow_calendar"),
            grid_chart(State.chart_frag, State.visible_frag, "fragmentation"),
            grid_chart(State.chart_sleep, State.visible_sleep, "sleep_pattern"),
            grid_chart(State.chart_spiral, State.visible_spiral, "spiral"),
            grid_chart(State.chart_chord, State.visible_chord, "chord"),
            grid_chart(State.chart_violin, State.visible_violin, "violin"),
            grid_chart(State.chart_streamgraph, State.visible_streamgraph, "streamgraph"),
            grid_chart(State.chart_rose, State.visible_rose, "rose"),
            grid_chart(State.chart_barcode, State.visible_barcode, "barcode"),
            grid_chart(State.chart_energy, State.visible_energy, "energy"),
            grid_chart(State.chart_pulse, State.visible_pulse, "pulse"),
            grid_chart(State.chart_network, State.visible_network, "network"),
            grid_chart(State.chart_recovery, State.visible_recovery, "recovery"),
            grid_chart(State.chart_streak, State.visible_streak, "streak"),
            grid_chart(State.chart_weekly, State.visible_weekly, "weekly"),
            grid_chart(State.chart_burnout, State.visible_burnout, "burnout"),
            grid_chart(State.chart_peak, State.visible_peak, "peak"),
            # Timeline with date selector
            rx.box(
                rx.hstack(
                    rx.text("TIMELINE", font_size="8px", color=GREY),
                    rx.select(
                        State.available_dates,
                        value=State.selected_date,
                        on_change=State.set_date,
                        size="1",
                        width="100px",
                    ),
                    spacing="2",
                    align_items="center",
                    margin_bottom="2px",
                ),
                rx.cond(
                    State.chart_gantt != None,
                    rx.plotly(data=State.chart_gantt, height="100%", style={"min_height": "100px"},
                              config={"displayModeBar": False}),
                    rx.text("—", color=GREY),
                ),
                background=LIGHT_GREY,
                border=f"1px solid {BORDER_COLOR}",
                padding="4px",
                font_family=MONO,
                overflow="hidden",
                display=rx.cond(State.visible_gantt, "flex", "none"),
                flex_direction="column",
                cursor="pointer",
                on_double_click=lambda: State.focus_chart("gantt"),
            ),

            display="grid",
            # Auto-fit: charts fill available space, min 250px each
            grid_template_columns="repeat(auto-fit, minmax(250px, 1fr))",
            # Auto rows: each row takes equal space
            grid_auto_rows="1fr",
            gap="6px",
            width="100%",
            height="100%",
        ),

        padding="8px",
        flex="1",
        overflow="hidden",
        height="100%",
    )


def chart_toggle_btn(label: str, is_visible, on_click) -> rx.Component:
    """Small toggle button for chart visibility."""
    return rx.box(
        rx.text(label, font_size="9px", color=rx.cond(is_visible, BLACK, GREY)),
        padding="4px 8px",
        background=rx.cond(is_visible, "#e0e0e0", "transparent"),
        border=f"1px solid {BORDER_COLOR}",
        cursor="pointer",
        on_click=on_click,
        _hover={"background": HOVER_COLOR},
    )


def toggle_panel() -> rx.Component:
    """Panel for toggling chart visibility."""
    return rx.cond(
        State.toggle_panel_open,
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.text("CHARTS", font_size="12px", font_weight="700", letter_spacing="0.1em"),
                    rx.spacer(),
                    rx.text("×", font_size="24px", cursor="pointer", on_click=State.toggle_chart_panel),
                    width="100%",
                ),
                rx.box(
                    rx.text("TOGGLE VISIBILITY", font_size="10px", color=GREY, letter_spacing="0.1em", margin_bottom="8px"),
                    rx.box(
                        chart_toggle_btn("Flow/Shallow", State.visible_flow_shallow, State.toggle_flow_shallow),
                        chart_toggle_btn("Trend", State.visible_trend, State.toggle_trend),
                        chart_toggle_btn("Breakdown", State.visible_breakdown, State.toggle_breakdown),
                        chart_toggle_btn("Sessions", State.visible_session_dist, State.toggle_session_dist),
                        chart_toggle_btn("Heatmap", State.visible_heatmap, State.toggle_heatmap),
                        chart_toggle_btn("Circadian", State.visible_hourly, State.toggle_hourly),
                        chart_toggle_btn("Flow Prob", State.visible_flow_prob, State.toggle_flow_prob),
                        chart_toggle_btn("Calendar", State.visible_flow_calendar, State.toggle_flow_calendar),
                        chart_toggle_btn("Fragment", State.visible_frag, State.toggle_frag),
                        chart_toggle_btn("Sleep", State.visible_sleep, State.toggle_sleep),
                        chart_toggle_btn("Timeline", State.visible_gantt, State.toggle_gantt),
                        chart_toggle_btn("Spiral", State.visible_spiral, State.toggle_spiral),
                        chart_toggle_btn("Transitions", State.visible_chord, State.toggle_chord),
                        chart_toggle_btn("Violin", State.visible_violin, State.toggle_violin),
                        chart_toggle_btn("Stream", State.visible_streamgraph, State.toggle_streamgraph),
                        chart_toggle_btn("Rose", State.visible_rose, State.toggle_rose),
                        chart_toggle_btn("Barcode", State.visible_barcode, State.toggle_barcode),
                        chart_toggle_btn("Energy", State.visible_energy, State.toggle_energy),
                        chart_toggle_btn("Pulse", State.visible_pulse, State.toggle_pulse),
                        chart_toggle_btn("Network", State.visible_network, State.toggle_network),
                        chart_toggle_btn("Recovery", State.visible_recovery, State.toggle_recovery),
                        chart_toggle_btn("Streak", State.visible_streak, State.toggle_streak),
                        chart_toggle_btn("Weekly", State.visible_weekly, State.toggle_weekly),
                        chart_toggle_btn("Burnout", State.visible_burnout, State.toggle_burnout),
                        chart_toggle_btn("Peak", State.visible_peak, State.toggle_peak),
                        display="flex",
                        flex_wrap="wrap",
                        gap="4px",
                    ),
                    width="100%",
                    margin_top="16px",
                ),
                spacing="0",
                align_items="stretch",
                width="100%",
            ),
            position="fixed",
            right="0",
            top="0",
            height="100vh",
            width="280px",
            background=LIGHT_GREY,
            padding="24px",
            font_family=MONO,
            z_index="200",
            box_shadow="-4px 0 20px rgba(0,0,0,0.05)",
        ),
        rx.box(),
    )


def insight_card(insight: dict) -> rx.Component:
    """Render a single insight card."""
    severity_colors = {
        "positive": "#4CAF50",
        "warning": "#FF9800",
        "negative": "#F44336",
        "neutral": "#607D8B",
    }
    return rx.box(
        rx.hstack(
            rx.text(insight["icon"], font_size="16px"),
            rx.vstack(
                rx.text(insight["title"], font_size="11px", font_weight="700", color=BLACK),
                rx.text(insight["message"], font_size="10px", color="#555", line_height="1.4"),
                spacing="1",
                align_items="start",
                flex="1",
            ),
            spacing="2",
            align_items="start",
            width="100%",
        ),
        padding="10px",
        background=WHITE,
        border_left=f"3px solid {severity_colors.get(insight['severity'], GREY)}",
        border_radius="0 4px 4px 0",
        margin_bottom="8px",
        box_shadow="0 1px 3px rgba(0,0,0,0.08)",
    )


def correlation_card(corr: dict) -> rx.Component:
    """Render a correlation insight card."""
    strength_colors = {
        "strong": "#4CAF50",
        "moderate": "#FF9800",
        "weak": "#9E9E9E",
    }
    return rx.box(
        rx.hstack(
            rx.text(corr["icon"], font_size="14px"),
            rx.vstack(
                rx.text(corr["title"], font_size="10px", font_weight="600", color=BLACK),
                rx.hstack(
                    rx.text(
                        f"r = {corr['correlation']:+.2f}",
                        font_size="9px",
                        color=rx.cond(
                            corr["direction"] == "positive",
                            "#4CAF50",
                            "#F44336",
                        ),
                        font_weight="600",
                    ),
                    rx.text(
                        f"({corr['strength']})",
                        font_size="9px",
                        color=GREY,
                    ),
                    spacing="1",
                ),
                spacing="0",
                align_items="start",
                flex="1",
            ),
            spacing="2",
            align_items="center",
            width="100%",
        ),
        padding="8px 10px",
        background=WHITE,
        border_left=f"3px solid {strength_colors.get(corr['strength'], GREY)}",
        border_radius="0 4px 4px 0",
        margin_bottom="6px",
        box_shadow="0 1px 2px rgba(0,0,0,0.05)",
    )


def chat_message(msg: dict) -> rx.Component:
    """Render a chat message."""
    is_user = msg["role"] == "user"
    return rx.box(
        rx.text(
            msg["content"],
            font_size="11px",
            color=rx.cond(is_user, BLACK, "#333"),
            line_height="1.5",
            white_space="pre-wrap",
        ),
        padding="10px 12px",
        background=rx.cond(is_user, "#e3f2fd", WHITE),
        border_radius="8px",
        margin_bottom="8px",
        margin_left=rx.cond(is_user, "20px", "0"),
        margin_right=rx.cond(is_user, "0", "20px"),
        box_shadow="0 1px 2px rgba(0,0,0,0.05)",
    )


def insights_panel() -> rx.Component:
    """AI Insights panel with chat interface."""
    return rx.cond(
        State.insights_panel_open,
        rx.box(
            rx.vstack(
                # Header
                rx.hstack(
                    rx.icon("sparkles", size=14, color=BLACK),
                    rx.text("AI INSIGHTS", font_size="12px", font_weight="700", letter_spacing="0.1em"),
                    rx.spacer(),
                    rx.button(
                        rx.icon("trash-2", size=12),
                        on_click=State.clear_chat,
                        background="transparent",
                        border="none",
                        color=GREY,
                        padding="4px",
                        cursor="pointer",
                        title="Clear chat",
                    ),
                    rx.text("×", font_size="24px", cursor="pointer", on_click=State.toggle_insights_panel),
                    width="100%",
                    align_items="center",
                ),
                
                # Generate AI Analysis button
                rx.button(
                    rx.cond(
                        State.ai_loading,
                        rx.hstack(
                            rx.spinner(size="1"),
                            rx.text("Analyzing...", font_size="10px"),
                            spacing="2",
                        ),
                        rx.hstack(
                            rx.icon("brain", size=12),
                            rx.text("Generate AI Analysis", font_size="10px"),
                            spacing="2",
                        ),
                    ),
                    on_click=State.generate_ai_analysis,
                    width="100%",
                    background=BLACK,
                    color=WHITE,
                    border="none",
                    padding="8px 12px",
                    border_radius="4px",
                    cursor="pointer",
                    margin_top="8px",
                    margin_bottom="12px",
                    disabled=State.ai_loading,
                    _hover={"background": "#333"},
                ),
                
                # Content area (scrollable)
                rx.box(
                    # AI Analysis result
                    rx.cond(
                        State.ai_analysis != "",
                        rx.box(
                            rx.hstack(
                                rx.text("AI ANALYSIS", font_size="9px", color=GREY, letter_spacing="0.05em"),
                                rx.spacer(),
                                rx.button(
                                    rx.icon("copy", size=12),
                                    on_click=rx.set_clipboard(State.ai_analysis),
                                    background="transparent",
                                    border="none",
                                    color=GREY,
                                    padding="2px",
                                    cursor="pointer",
                                    title="Copy to clipboard",
                                    _hover={"color": BLACK},
                                ),
                                width="100%",
                                margin_bottom="6px",
                            ),
                            rx.box(
                                rx.markdown(State.ai_analysis),
                                padding="12px",
                                background=WHITE,
                                border_radius="6px",
                                margin_bottom="16px",
                                box_shadow="0 1px 3px rgba(0,0,0,0.08)",
                                font_size="11px",
                                line_height="1.6",
                                class_name="prose prose-sm",
                            ),
                        ),
                        rx.fragment(),
                    ),
                    
                    # Chat history
                    rx.cond(
                        State.chat_history.length() > 0,
                        rx.box(
                            rx.text("CONVERSATION", font_size="9px", color=GREY, letter_spacing="0.05em", margin_bottom="6px"),
                            rx.foreach(State.chat_history, chat_message),
                            margin_bottom="12px",
                        ),
                        rx.fragment(),
                    ),
                    
                    # Correlations section
                    rx.cond(
                        State.correlations.length() > 0,
                        rx.box(
                            rx.text("🔗 CORRELATIES", font_size="9px", color=GREY, letter_spacing="0.05em", margin_bottom="6px"),
                            rx.text("Wat beïnvloedt je prestaties?", font_size="10px", color=GREY, margin_bottom="8px"),
                            rx.foreach(State.correlations, correlation_card),
                            margin_bottom="16px",
                        ),
                        rx.fragment(),
                    ),
                    
                    # Quick insights (rule-based)
                    rx.box(
                        rx.text("QUICK INSIGHTS", font_size="9px", color=GREY, letter_spacing="0.05em", margin_bottom="6px"),
                        rx.foreach(State.insights, insight_card),
                    ),
                    
                    width="100%",
                    flex="1",
                    overflow_y="auto",
                ),
                
                # Chat input
                rx.box(
                    # Suggested questions (only show when no chat history)
                    rx.cond(
                        State.chat_history.length() == 0,
                        rx.box(
                            rx.text("💡 PROBEER", font_size="8px", color=GREY, letter_spacing="0.1em", margin_bottom="6px"),
                            rx.hstack(
                                rx.foreach(
                                    State.suggested_questions,
                                    lambda q: rx.button(
                                        rx.text(q, font_size="9px"),
                                        on_click=lambda: State.use_suggestion(q),
                                        background="transparent",
                                        border=f"1px solid {BORDER_COLOR}",
                                        padding="4px 8px",
                                        border_radius="12px",
                                        cursor="pointer",
                                        _hover={"background": "#f0f0f0"},
                                        white_space="nowrap",
                                    ),
                                ),
                                spacing="1",
                                flex_wrap="wrap",
                                gap="4px",
                            ),
                            margin_bottom="12px",
                        ),
                        rx.fragment(),
                    ),
                    rx.form(
                        rx.hstack(
                            rx.input(
                                placeholder="Ask about your data...",
                                name="chat_input",
                                value=State.chat_input,
                                on_change=State.set_chat_input,
                                width="100%",
                                font_size="11px",
                                padding="8px 12px",
                                border=f"1px solid {BORDER_COLOR}",
                                border_radius="4px",
                                background=WHITE,
                                _focus={"border_color": BLACK, "outline": "none"},
                            ),
                            rx.button(
                                rx.cond(
                                    State.ai_loading,
                                    rx.spinner(size="1"),
                                    rx.icon("send", size=14),
                                ),
                                type="submit",
                                background=BLACK,
                                color=WHITE,
                                border="none",
                                padding="8px",
                                border_radius="4px",
                                cursor="pointer",
                                disabled=State.ai_loading,
                            ),
                            spacing="2",
                            width="100%",
                        ),
                        on_submit=lambda _: State.send_chat_message(),
                        reset_on_submit=False,
                        width="100%",
                    ),
                    padding_top="12px",
                    border_top=f"1px solid {BORDER_COLOR}",
                    margin_top="auto",
                ),
                
                spacing="0",
                align_items="stretch",
                width="100%",
                height="100%",
            ),
            position="fixed",
            right="0",
            top="0",
            height="100vh",
            width="380px",
            background=LIGHT_GREY,
            padding="20px",
            font_family=MONO,
            z_index="200",
            box_shadow="-4px 0 20px rgba(0,0,0,0.1)",
            display="flex",
            flex_direction="column",
        ),
        rx.box(),
    )


def goal_progress_bar(goal: dict) -> rx.Component:
    """Render a single goal progress bar."""
    return rx.box(
        rx.hstack(
            rx.text(goal["icon"], font_size="14px"),
            rx.text(goal["label"], font_size="11px", font_weight="500"),
            rx.spacer(),
            rx.text(
                f"{goal['current']}/{goal['target']} {goal['unit']}",
                font_size="10px",
                color=GREY,
            ),
            width="100%",
            margin_bottom="4px",
        ),
        rx.box(
            rx.box(
                width=f"{goal['percentage']}%",
                height="100%",
                background=rx.cond(
                    goal["status"] == "success",
                    "#4CAF50",
                    rx.cond(
                        goal["status"] == "warning",
                        "#FF9800",
                        "#f44336",
                    ),
                ),
                border_radius="2px",
                transition="width 0.3s ease",
            ),
            width="100%",
            height="6px",
            background="#e0e0e0",
            border_radius="2px",
            overflow="hidden",
        ),
        margin_bottom="12px",
    )


def prediction_card(pred: dict) -> rx.Component:
    """Single prediction card."""
    return rx.box(
        rx.hstack(
            rx.text(
                rx.cond(pred["will_achieve"], "✅", "⚠️"),
                font_size="16px",
            ),
            rx.vstack(
                rx.text(pred["metric"], font_size="11px", font_weight="600"),
                rx.text(pred["message"], font_size="9px", color=GREY, line_height="1.3"),
                spacing="1",
                align_items="start",
            ),
            spacing="3",
            width="100%",
            align_items="start",
        ),
        rx.box(
            rx.hstack(
                rx.text("💡", font_size="10px"),
                rx.text(pred["advice"], font_size="9px", color=GREY, line_height="1.3"),
                spacing="2",
                align_items="start",
            ),
            background="#f8f8f8",
            padding="8px",
            border_radius="4px",
            margin_top="8px",
        ),
        padding="12px",
        background=WHITE,
        border=f"1px solid {BORDER_COLOR}",
        border_radius="6px",
        margin_bottom="10px",
    )


def predictions_panel() -> rx.Component:
    """Predictions and forecasting panel."""
    return rx.cond(
        State.predictions_panel_open,
        rx.box(
            rx.vstack(
                # Header
                rx.hstack(
                    rx.icon("trending-up", size=14, color=BLACK),
                    rx.text("VOORSPELLINGEN", font_size="12px", font_weight="700", letter_spacing="0.1em"),
                    rx.spacer(),
                    rx.text("×", font_size="24px", cursor="pointer", on_click=State.toggle_predictions_panel),
                    width="100%",
                    align_items="center",
                ),
                
                # Weekly outlook
                rx.box(
                    rx.text("WEEK OUTLOOK", font_size="9px", color=GREY, letter_spacing="0.05em", margin_bottom="8px"),
                    rx.text(State.weekly_outlook, font_size="12px", font_weight="500", line_height="1.4"),
                    padding="16px",
                    background=WHITE,
                    border_radius="8px",
                    margin_bottom="16px",
                    box_shadow="0 1px 3px rgba(0,0,0,0.08)",
                ),
                
                # Burnout Risk
                rx.box(
                    rx.hstack(
                        rx.text("BURNOUT RISICO", font_size="9px", color=GREY, letter_spacing="0.05em"),
                        rx.spacer(),
                        rx.text(
                            State.burnout_risk["level"],
                            font_size="10px",
                            font_weight="600",
                            color=rx.cond(
                                State.burnout_risk["level"] == "low",
                                "#4CAF50",
                                rx.cond(
                                    State.burnout_risk["level"] == "medium",
                                    "#FF9800",
                                    "#f44336",
                                ),
                            ),
                            text_transform="uppercase",
                        ),
                        width="100%",
                        margin_bottom="8px",
                    ),
                    rx.box(
                        rx.box(
                            width=f"{State.burnout_risk['score']}%",
                            height="100%",
                            background=rx.cond(
                                State.burnout_risk["score"].to(int) < 30,
                                "#4CAF50",
                                rx.cond(
                                    State.burnout_risk["score"].to(int) < 60,
                                    "#FF9800",
                                    "#f44336",
                                ),
                            ),
                            border_radius="2px",
                        ),
                        width="100%",
                        height="8px",
                        background="#e0e0e0",
                        border_radius="2px",
                        overflow="hidden",
                    ),
                    rx.cond(
                        State.burnout_warnings,
                        rx.box(
                            rx.foreach(
                                State.burnout_warnings,
                                lambda w: rx.text(f"• {w}", font_size="9px", color=GREY, margin_top="4px"),
                            ),
                            margin_top="8px",
                        ),
                        rx.box(),
                    ),
                    padding="16px",
                    background=WHITE,
                    border_radius="8px",
                    margin_bottom="16px",
                    box_shadow="0 1px 3px rgba(0,0,0,0.08)",
                ),
                
                # Predictions
                rx.box(
                    rx.text("DOELVOORSPELLINGEN", font_size="9px", color=GREY, letter_spacing="0.05em", margin_bottom="12px"),
                    rx.foreach(State.predictions, prediction_card),
                    width="100%",
                    flex="1",
                    overflow_y="auto",
                ),
                
                spacing="0",
                align_items="stretch",
                width="100%",
                height="100%",
            ),
            position="fixed",
            right="0",
            top="0",
            height="100vh",
            width="380px",
            background=LIGHT_GREY,
            padding="20px",
            font_family=MONO,
            z_index="200",
            box_shadow="-4px 0 20px rgba(0,0,0,0.1)",
            display="flex",
            flex_direction="column",
        ),
        rx.box(),
    )


def badge_item(badge: dict) -> rx.Component:
    """Single badge display."""
    return rx.box(
        rx.text(badge["icon"], font_size="24px"),
        rx.text(badge["name"], font_size="9px", font_weight="500", text_align="center"),
        display="flex",
        flex_direction="column",
        align_items="center",
        padding="8px",
        background=WHITE,
        border_radius="8px",
        border=f"1px solid {BORDER_COLOR}",
        min_width="70px",
    )


def badge_progress_item(badge: dict[str, Any]) -> rx.Component:
    """Badge in progress with progress bar."""
    return rx.box(
        rx.hstack(
            rx.text(badge["icon"], font_size="16px", opacity="0.5"),
            rx.vstack(
                rx.text(badge["name"], font_size="10px", font_weight="500"),
                rx.box(
                    rx.box(
                        width=badge["progress_pct"].to(str) + "%",
                        height="100%",
                        background=BLACK,
                        border_radius="2px",
                    ),
                    width="100%",
                    height="4px",
                    background="#e0e0e0",
                    border_radius="2px",
                ),
                spacing="1",
                align_items="start",
                flex="1",
            ),
            rx.text(badge["progress_pct"].to(str) + "%", font_size="9px", color=GREY),
            spacing="2",
            width="100%",
            align_items="center",
        ),
        padding="8px",
        background=WHITE,
        border_radius="6px",
        margin_bottom="6px",
    )


def streak_item(streak: dict[str, Any]) -> rx.Component:
    """Single streak display."""
    return rx.hstack(
        rx.cond(streak["current"].to(int) > 0, rx.text("🔥", font_size="14px"), rx.text("❄️", font_size="14px")),
        rx.vstack(
            rx.text(streak["name"], font_size="10px", font_weight="500"),
            rx.text(f"{streak['current']} dagen", font_size="9px", color=GREY),
            spacing="0",
            align_items="start",
        ),
        rx.spacer(),
        rx.vstack(
            rx.text("BEST", font_size="8px", color=GREY),
            rx.text(streak["longest"], font_size="11px", font_weight="600"),
            spacing="0",
            align_items="center",
        ),
        width="100%",
        padding="8px",
        background=WHITE,
        border_radius="6px",
        margin_bottom="6px",
    )


def activity_log_item(item: dict[str, Any]) -> rx.Component:
    """Single activity log entry."""
    return rx.hstack(
        rx.vstack(
            rx.text(item["activity"], font_size="11px", font_weight="500"),
            rx.text(f"{item['start']} - {item['end']}", font_size="9px", color=GREY),
            spacing="0",
            align_items="start",
            flex="1",
        ),
        rx.text(item["duration"], font_size="10px", color=GREY, min_width="40px", text_align="right"),
        width="100%",
        padding="8px 12px",
        background=WHITE,
        border_radius="4px",
        margin_bottom="4px",
        align_items="center",
    )


def activity_log_panel() -> rx.Component:
    """Activity log panel showing all logged activities."""
    return rx.cond(
        State.activity_log_open,
        rx.box(
            rx.vstack(
                # Header
                rx.hstack(
                    rx.icon("list", size=14, color=BLACK),
                    rx.text("ACTIVITEITEN LOG", font_size="12px", font_weight="700", letter_spacing="0.1em"),
                    rx.spacer(),
                    rx.text("×", font_size="24px", cursor="pointer", on_click=State.toggle_activity_log),
                    width="100%",
                    align_items="center",
                ),
                
                # Stats
                rx.hstack(
                    rx.text(f"{State.activity_log.length()} activiteiten", font_size="10px", color=GREY),
                    rx.spacer(),
                    rx.text("Recent eerst", font_size="10px", color=GREY),
                    width="100%",
                    margin_bottom="12px",
                ),
                
                # Activity list
                rx.box(
                    rx.cond(
                        State.activity_log,
                        rx.foreach(State.activity_log, activity_log_item),
                        rx.text("Geen activiteiten gevonden", font_size="11px", color=GREY, text_align="center", padding="20px"),
                    ),
                    width="100%",
                    max_height="calc(100vh - 200px)",
                    overflow_y="auto",
                ),
                
                width="100%",
                spacing="3",
            ),
            position="fixed",
            top="0",
            right="0",
            width="320px",
            height="100vh",
            background="#fafafa",
            padding="20px",
            box_shadow="-2px 0 8px rgba(0,0,0,0.1)",
            z_index="1000",
            overflow_y="auto",
        ),
        rx.box(),
    )


def gamification_panel() -> rx.Component:
    """Gamification panel with levels, streaks, and badges."""
    return rx.cond(
        State.gamification_panel_open,
        rx.box(
            rx.vstack(
                # Header
                rx.hstack(
                    rx.icon("trophy", size=14, color=BLACK),
                    rx.text("ACHIEVEMENTS", font_size="12px", font_weight="700", letter_spacing="0.1em"),
                    rx.spacer(),
                    rx.text("×", font_size="24px", cursor="pointer", on_click=State.toggle_gamification_panel),
                    width="100%",
                    align_items="center",
                ),
                
                # Level & XP
                rx.box(
                    rx.hstack(
                        rx.vstack(
                            rx.text("LEVEL", font_size="9px", color=GREY, letter_spacing="0.05em"),
                            rx.text(State.player_level, font_size="36px", font_weight="700"),
                            spacing="0",
                            align_items="center",
                        ),
                        rx.vstack(
                            rx.text(f"{State.player_xp} XP", font_size="11px", font_weight="500"),
                            rx.box(
                                rx.box(
                                    width=f"{State.xp_progress * 100}%",
                                    height="100%",
                                    background="#4CAF50",
                                    border_radius="2px",
                                ),
                                width="100%",
                                height="8px",
                                background="#e0e0e0",
                                border_radius="2px",
                            ),
                            rx.text(f"{State.xp_remaining} XP tot level {State.next_level}", font_size="9px", color=GREY),
                            spacing="2",
                            align_items="start",
                            flex="1",
                        ),
                        spacing="4",
                        width="100%",
                        align_items="center",
                    ),
                    padding="16px",
                    background=WHITE,
                    border_radius="8px",
                    margin_bottom="16px",
                    box_shadow="0 1px 3px rgba(0,0,0,0.08)",
                ),
                
                # Streaks
                rx.box(
                    rx.text("STREAKS", font_size="9px", color=GREY, letter_spacing="0.05em", margin_bottom="12px"),
                    rx.cond(
                        State.streaks,
                        rx.foreach(State.streaks, streak_item),
                        rx.text("Geen streaks nog", font_size="10px", color=GREY),
                    ),
                    width="100%",
                    margin_bottom="16px",
                ),
                
                # Badges Earned
                rx.box(
                    rx.hstack(
                        rx.text("BADGES", font_size="9px", color=GREY, letter_spacing="0.05em"),
                        rx.spacer(),
                        rx.text(f"{State.badges_earned.length()} verdiend", font_size="9px", color=GREY),
                        width="100%",
                        margin_bottom="12px",
                    ),
                    rx.cond(
                        State.badges_earned,
                        rx.box(
                            rx.foreach(State.badges_earned, badge_item),
                            display="flex",
                            flex_wrap="wrap",
                            gap="8px",
                        ),
                        rx.text("Nog geen badges", font_size="10px", color=GREY),
                    ),
                    width="100%",
                    margin_bottom="16px",
                ),
                
                # Badges In Progress
                rx.box(
                    rx.text("BIJNA DAAR", font_size="9px", color=GREY, letter_spacing="0.05em", margin_bottom="12px"),
                    rx.foreach(State.badges_in_progress, badge_progress_item),
                    width="100%",
                    flex="1",
                    overflow_y="auto",
                ),
                
                spacing="0",
                align_items="stretch",
                width="100%",
                height="100%",
            ),
            position="fixed",
            right="0",
            top="0",
            height="100vh",
            width="350px",
            background=LIGHT_GREY,
            padding="20px",
            font_family=MONO,
            z_index="200",
            box_shadow="-4px 0 20px rgba(0,0,0,0.1)",
            display="flex",
            flex_direction="column",
        ),
        rx.box(),
    )


def goals_panel() -> rx.Component:
    """Goals panel with progress tracking and settings."""
    return rx.cond(
        State.goals_panel_open,
        rx.box(
            rx.vstack(
                # Header
                rx.hstack(
                    rx.icon("target", size=14, color=BLACK),
                    rx.text("DOELEN", font_size="12px", font_weight="700", letter_spacing="0.1em"),
                    rx.spacer(),
                    rx.text("×", font_size="24px", cursor="pointer", on_click=State.toggle_goals_panel),
                    width="100%",
                    align_items="center",
                ),
                
                # Overall score
                rx.box(
                    rx.hstack(
                        rx.text("WEEK SCORE", font_size="9px", color=GREY, letter_spacing="0.05em"),
                        rx.spacer(),
                        rx.text(
                            rx.cond(
                                State.goals_overall_score >= 80,
                                "🔥",
                                rx.cond(
                                    State.goals_overall_score >= 50,
                                    "👍",
                                    "💪",
                                ),
                            ),
                            font_size="16px",
                        ),
                        width="100%",
                    ),
                    rx.text(
                        f"{State.goals_overall_score}%",
                        font_size="32px",
                        font_weight="700",
                        color=rx.cond(
                            State.goals_overall_score >= 80,
                            "#4CAF50",
                            rx.cond(
                                State.goals_overall_score >= 50,
                                "#FF9800",
                                "#f44336",
                            ),
                        ),
                    ),
                    padding="16px",
                    background=WHITE,
                    border_radius="8px",
                    margin_bottom="16px",
                    box_shadow="0 1px 3px rgba(0,0,0,0.08)",
                ),
                
                # Progress bars
                rx.box(
                    rx.text("VOORTGANG DEZE WEEK", font_size="9px", color=GREY, letter_spacing="0.05em", margin_bottom="12px"),
                    rx.foreach(State.goals_progress, goal_progress_bar),
                    width="100%",
                    flex="1",
                    overflow_y="auto",
                ),
                
                # Settings section
                rx.box(
                    rx.text("DOELEN AANPASSEN", font_size="9px", color=GREY, letter_spacing="0.05em", margin_bottom="12px"),
                    rx.vstack(
                        rx.hstack(
                            rx.text("Deep Work", font_size="10px", width="120px"),
                            rx.input(
                                value=State.goal_deep_work,
                                on_change=State.set_goal_deep_work,
                                width="60px",
                                font_size="10px",
                                padding="4px 8px",
                                border=f"1px solid {BORDER_COLOR}",
                                border_radius="4px",
                            ),
                            rx.text("uur/week", font_size="9px", color=GREY),
                            width="100%",
                        ),
                        rx.hstack(
                            rx.text("Sport", font_size="10px", width="120px"),
                            rx.input(
                                value=State.goal_sport,
                                on_change=State.set_goal_sport,
                                width="60px",
                                font_size="10px",
                                padding="4px 8px",
                                border=f"1px solid {BORDER_COLOR}",
                                border_radius="4px",
                            ),
                            rx.text("uur/week", font_size="9px", color=GREY),
                            width="100%",
                        ),
                        rx.hstack(
                            rx.text("Slaap", font_size="10px", width="120px"),
                            rx.input(
                                value=State.goal_sleep,
                                on_change=State.set_goal_sleep,
                                width="60px",
                                font_size="10px",
                                padding="4px 8px",
                                border=f"1px solid {BORDER_COLOR}",
                                border_radius="4px",
                            ),
                            rx.text("uur/nacht", font_size="9px", color=GREY),
                            width="100%",
                        ),
                        rx.hstack(
                            rx.text("Max Entertainment", font_size="10px", width="120px"),
                            rx.input(
                                value=State.goal_max_entertainment,
                                on_change=State.set_goal_max_entertainment,
                                width="60px",
                                font_size="10px",
                                padding="4px 8px",
                                border=f"1px solid {BORDER_COLOR}",
                                border_radius="4px",
                            ),
                            rx.text("uur/week", font_size="9px", color=GREY),
                            width="100%",
                        ),
                        rx.hstack(
                            rx.text("Flow Sessies", font_size="10px", width="120px"),
                            rx.input(
                                value=State.goal_min_flow_sessions,
                                on_change=State.set_goal_min_flow_sessions,
                                width="60px",
                                font_size="10px",
                                padding="4px 8px",
                                border=f"1px solid {BORDER_COLOR}",
                                border_radius="4px",
                            ),
                            rx.text("per week", font_size="9px", color=GREY),
                            width="100%",
                        ),
                        spacing="2",
                        width="100%",
                    ),
                    rx.button(
                        rx.hstack(
                            rx.icon("save", size=12),
                            rx.text("Opslaan", font_size="10px"),
                            spacing="2",
                        ),
                        on_click=State.save_goals,
                        width="100%",
                        background=BLACK,
                        color=WHITE,
                        border="none",
                        padding="8px 12px",
                        border_radius="4px",
                        cursor="pointer",
                        margin_top="12px",
                        _hover={"background": "#333"},
                    ),
                    padding_top="12px",
                    border_top=f"1px solid {BORDER_COLOR}",
                    margin_top="auto",
                ),
                
                spacing="0",
                align_items="stretch",
                width="100%",
                height="100%",
            ),
            position="fixed",
            right="0",
            top="0",
            height="100vh",
            width="350px",
            background=LIGHT_GREY,
            padding="20px",
            font_family=MONO,
            z_index="200",
            box_shadow="-4px 0 20px rgba(0,0,0,0.1)",
            display="flex",
            flex_direction="column",
        ),
        rx.box(),
    )


def keyboard_shortcut_row(keys: str, description: str) -> rx.Component:
    """Single keyboard shortcut row."""
    return rx.hstack(
        rx.box(
            rx.text(keys, font_size="10px", font_weight="500"),
            background="#e0e0e0",
            padding="2px 6px",
            border_radius="3px",
            font_family=MONO,
        ),
        rx.text(description, font_size="10px", color=GREY),
        spacing="3",
        width="100%",
    )


def keyboard_shortcuts_help() -> rx.Component:
    """Keyboard shortcuts help tooltip - shown on hover over ? icon."""
    return rx.box(
        rx.box(
            rx.text("?", font_size="12px", font_weight="700", color=WHITE),
            width="24px",
            height="24px",
            display="flex",
            align_items="center",
            justify_content="center",
            background=BLACK,
            border_radius="50%",
            cursor="pointer",
            _hover={"background": "#333"},
        ),
        rx.box(
            rx.text("KEYBOARD SHORTCUTS", font_size="9px", letter_spacing="0.1em", color=GREY, margin_bottom="12px"),
            rx.vstack(
                keyboard_shortcut_row("ESC", "Sluit panel/overlay"),
                keyboard_shortcut_row("1-6", "Wissel tab (All, Flow, Time, ...)"),
                keyboard_shortcut_row("⌘/Ctrl + G", "Goals panel"),
                keyboard_shortcut_row("⌘/Ctrl + I", "AI Insights panel"),
                keyboard_shortcut_row("⌘/Ctrl + P", "Predictions panel"),
                keyboard_shortcut_row("⌘/Ctrl + T", "Achievements panel"),
                keyboard_shortcut_row("⌘/Ctrl + D", "Dark mode"),
                keyboard_shortcut_row("⌘/Ctrl + U", "Upload menu"),
                spacing="2",
                align_items="start",
                width="100%",
            ),
            position="absolute",
            bottom="36px",
            right="0",
            background=WHITE,
            border=f"1px solid {BORDER_COLOR}",
            border_radius="8px",
            padding="12px",
            width="220px",
            box_shadow="0 4px 12px rgba(0,0,0,0.15)",
            display="none",
            z_index="300",
            class_name="shortcuts-tooltip",
        ),
        position="fixed",
        bottom="16px",
        right="16px",
        z_index="100",
        _hover={
            "& .shortcuts-tooltip": {
                "display": "block",
            },
        },
    )


def menu_sidebar() -> rx.Component:
    return rx.cond(
        State.menu_sidebar_open,
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.text("MENU", font_size="12px", font_weight="700", letter_spacing="0.1em"),
                    rx.spacer(),
                    rx.text("×", font_size="24px", cursor="pointer", on_click=State.toggle_menu_sidebar),
                    width="100%",
                ),
                rx.box(
                    rx.text("UPLOAD CSV", font_size="10px", color=GREY, letter_spacing="0.1em", margin_bottom="8px"),
                    rx.upload(
                        rx.box(
                            rx.text("DROP FILE", font_size="11px", text_align="center", color=GREY),
                            padding="20px",
                            border=f"1px dashed {BORDER_COLOR}",
                            cursor="pointer",
                            _hover={"border_color": BLACK},
                        ),
                        id="csv_upload",
                        accept={".csv": ["text/csv"]},
                        max_files=1,
                        on_drop=State.handle_upload(rx.upload_files(upload_id="csv_upload")),
                    ),
                    rx.text(State.status_msg, font_size="10px", color=GREY, margin_top="4px"),
                    width="100%",
                    margin_top="24px",
                ),
                spacing="0",
                align_items="stretch",
                width="100%",
            ),
            position="fixed",
            right="0",
            top="0",
            height="100vh",
            width="280px",
            background=LIGHT_GREY,
            padding="24px",
            font_family=MONO,
            z_index="200",
            box_shadow="-4px 0 20px rgba(0,0,0,0.05)",
        ),
        rx.box(),
    )


def get_chart_by_key(key: str):
    """Map chart key to chart data."""
    chart_map = {
        "flow_vs_shallow": State.chart_flow_shallow,
        "deep_work_trend": State.chart_trend,
        "daily_breakdown": State.chart_breakdown,
        "session_length": State.chart_session_dist,
        "heatmap": State.chart_heatmap,
        "circadian": State.chart_hourly,
        "flow_prob": State.chart_flow_prob,
        "flow_calendar": State.chart_flow_calendar,
        "fragmentation": State.chart_frag,
        "sleep_pattern": State.chart_sleep,
        "spiral": State.chart_spiral,
        "chord": State.chart_chord,
        "violin": State.chart_violin,
        "streamgraph": State.chart_streamgraph,
        "rose": State.chart_rose,
        "barcode": State.chart_barcode,
        "energy": State.chart_energy,
        "pulse": State.chart_pulse,
        "network": State.chart_network,
        "gantt": State.chart_gantt,
        "recovery": State.chart_recovery,
        "streak": State.chart_streak,
        "weekly": State.chart_weekly,
        "burnout": State.chart_burnout,
        "peak": State.chart_peak,
    }
    return chart_map.get(key, State.chart_flow_shallow)


CHART_EXPLANATIONS = {
    "flow_vs_shallow": {
        "title": "Flow vs Shallow Time",
        "formula": "Flow = sessions ≥ 90min in Work/Coding",
        "description": "Stacked bar chart showing daily hours split between flow state (deep work sessions ≥90 minutes) and shallow work (everything else). Based on Cal Newport's research on ultrarian rhythms.",
        "interpretation": "Higher flow ratio indicates better focus quality. Aim for 2-4h of flow daily."
    },
    "deep_work_trend": {
        "title": "Deep Work Trend", 
        "formula": "Deep Work = Work + Coding sessions ≥ 60min",
        "description": "Line chart with 7-day moving average showing deep work hours over time. Helps identify productivity trends and patterns.",
        "interpretation": "Upward trend = improving focus habits. Look for weekly patterns."
    },
    "daily_breakdown": {
        "title": "Daily Breakdown",
        "formula": "Sum of hours per activity type per day",
        "description": "Stacked bar chart showing how time is distributed across activity categories each day.",
        "interpretation": "Visual balance check. Identify dominant activities and gaps."
    },
    "session_length": {
        "title": "Session Length Distribution",
        "formula": "Histogram of all session durations",
        "description": "Distribution of session lengths with 90-minute flow threshold marked. Shows your typical session patterns.",
        "interpretation": "Peak left of 90min = fragmented work. Peak right = good deep work habits."
    },
    "heatmap": {
        "title": "Activity Heatmap",
        "formula": "Hours per hour-of-day × day-of-week",
        "description": "GitHub-style heatmap showing activity intensity by time of day and day of week.",
        "interpretation": "Darker cells = more active periods. Find your natural productivity windows."
    },
    "circadian": {
        "title": "Circadian Profile",
        "formula": "Avg hours per hour of day (stacked by activity)",
        "description": "Stacked area chart showing your average day profile. Reveals your chronotype and energy patterns.",
        "interpretation": "Peaks show your natural productive hours. Valleys indicate rest periods."
    },
    "flow_prob": {
        "title": "Flow Probability by Hour",
        "formula": "P(flow) = count(sessions ≥90min) / count(all sessions) per hour",
        "description": "Probability of achieving flow state if you start a deep work session at each hour.",
        "interpretation": "Schedule important work during high-probability hours."
    },
    "flow_calendar": {
        "title": "Flow Calendar",
        "formula": "Sum of flow hours (≥90min sessions) per day",
        "description": "Heatmap calendar showing flow hours achieved each day. Visualizes consistency.",
        "interpretation": "Look for streaks and gaps. Consistency matters more than peaks."
    },
    "fragmentation": {
        "title": "Fragmentation Index",
        "formula": "Fragmentation = N_switches / Total_hours",
        "description": "Daily context switches divided by total hours. Measures how interrupted your work is.",
        "interpretation": "Lower is better. High fragmentation = difficulty achieving deep work."
    },
    "sleep_pattern": {
        "title": "Sleep Pattern",
        "formula": "Inferred from activity gaps (longest daily gap)",
        "description": "Sleep timing derived from activity patterns. Shows sleep/wake consistency.",
        "interpretation": "Consistent lines = good sleep hygiene. Erratic = potential issues."
    },
    "spiral": {
        "title": "Spiral Plot (Circadian)",
        "formula": "θ = hour (0-24h → 0-360°), r = date (inner→outer)",
        "description": "Polar visualization showing activities over time. Inner ring = oldest, outer = newest. Angle = time of day.",
        "interpretation": "Straight radial lines = consistent habits. Curves = drifting schedule."
    },
    "chord": {
        "title": "Activity Transitions (Sankey)",
        "formula": "Count of A → B transitions for all activity pairs",
        "description": "Flow diagram showing how often you switch between activities. Thicker links = more frequent transitions.",
        "interpretation": "Identify distraction triggers (e.g., Work → Entertainment patterns)."
    },
    "violin": {
        "title": "Session Duration Violin",
        "formula": "Distribution of durations per activity type",
        "description": "Violin plot showing session length distribution for each activity. Box plot inside shows median/quartiles.",
        "interpretation": "Wide violins = variable sessions. Violins above 90min line = good focus."
    },
    "streamgraph": {
        "title": "Streamgraph (Evolution)",
        "formula": "Cumulative hours per activity over time",
        "description": "Stacked area chart showing how your time allocation evolves over days/weeks.",
        "interpretation": "Watch for category growth/shrinkage. Identify life phase changes."
    },
    "rose": {
        "title": "Rose Chart (Daily Fingerprint)",
        "formula": "Avg hours per hour, colored by dominant activity",
        "description": "24-wedge polar chart showing your 'average day'. Each wedge = 1 hour, colored by most common activity.",
        "interpretation": "Your unique daily rhythm fingerprint. Compare to ideal day."
    },
    "barcode": {
        "title": "Barcode Plot (Session Density)",
        "formula": "Vertical line at each session start time",
        "description": "Rug plot showing when sessions start each day. Dense areas = high activity, gaps = breaks.",
        "interpretation": "Clustered lines = consolidated work. Scattered = fragmented day."
    },
    "energy": {
        "title": "Energy Balance",
        "formula": "Energy = Σ(duration × weight), where Sport/Yoga=+50, Work/Coding=-40",
        "description": "Daily energy balance (charging vs draining) with cumulative trend line.",
        "interpretation": "Negative cumulative = burnout risk. Balance draining with charging activities."
    },
    "pulse": {
        "title": "Productivity Pulse",
        "formula": "Pulse = Σ(duration × weight) / Σ(duration)",
        "description": "RescueTime-style score (0-100). Weights: Coding=100, Work=75, Read=50, Entertainment=0.",
        "interpretation": "50+ = productive day. <30 = recovery/leisure day. Track 7-day average."
    },
    "network": {
        "title": "Activity Network",
        "formula": "Nodes = activities, edges = transition frequency",
        "description": "Network graph showing activity relationships. Node size = total hours, edge thickness = transition count.",
        "interpretation": "Central nodes = hub activities. Thick edges = frequent switches."
    },
    "gantt": {
        "title": "Daily Timeline (Gantt)",
        "formula": "Start/end times visualized as horizontal bars",
        "description": "Timeline view of a single day showing all activities with their exact timing.",
        "interpretation": "See your day structure. Identify gaps and overlaps."
    },
    "recovery": {
        "title": "Recovery Ratio",
        "formula": "Ratio = Recovery Hours / Drain Hours",
        "description": "Daily balance between charging (Sport, Yoga, Walking) and draining (Work, Coding) activities. Target ratio is 0.3-0.5.",
        "interpretation": "Below 0.15 = burnout risk. Green zone (≥0.3) = sustainable pace."
    },
    "streak": {
        "title": "Flow Streak",
        "formula": "Consecutive days with ≥2h flow (sessions ≥90min)",
        "description": "Gamification of flow consistency. Shows current streak and max achieved. Bars show daily flow hours.",
        "interpretation": "Build streaks for momentum. Breaks reset the counter - protect your streak!"
    },
    "weekly": {
        "title": "Weekly Pattern",
        "formula": "Avg deep work & flow hours per day of week",
        "description": "Compares productivity across weekdays to find your optimal days. Based on historical data.",
        "interpretation": "Schedule important work on your statistically best days."
    },
    "burnout": {
        "title": "Burnout Risk Gauge",
        "formula": "Score = f(energy_trend, recovery_ratio, consecutive_negative_days)",
        "description": "Composite risk score (0-100) based on 14-day energy trends, recovery ratios, and consecutive draining days.",
        "interpretation": "Green (0-35) = healthy. Yellow (35-60) = caution. Red (60+) = take action."
    },
    "peak": {
        "title": "Peak Flow Hours",
        "formula": "P(flow) per hour × weekday based on historical sessions",
        "description": "Heatmap showing flow probability for each hour of each weekday. Darker = higher success rate.",
        "interpretation": "Schedule deep work in dark cells. Avoid light cells for important tasks."
    },
}


def chart_explanation(chart_key: str) -> rx.Component:
    """Right panel with chart explanation."""
    return rx.box(
        rx.vstack(
            rx.text("HOW IT WORKS", font_size="11px", font_weight="700", letter_spacing="0.1em", color=BLACK),
            rx.divider(margin_y="8px"),
            # Formula
            rx.text("FORMULA", font_size="9px", color=GREY, letter_spacing="0.05em"),
            rx.code(
                rx.cond(chart_key == "flow_vs_shallow", CHART_EXPLANATIONS["flow_vs_shallow"]["formula"], ""),
                rx.cond(chart_key == "deep_work_trend", CHART_EXPLANATIONS["deep_work_trend"]["formula"], ""),
                rx.cond(chart_key == "daily_breakdown", CHART_EXPLANATIONS["daily_breakdown"]["formula"], ""),
                rx.cond(chart_key == "session_length", CHART_EXPLANATIONS["session_length"]["formula"], ""),
                rx.cond(chart_key == "heatmap", CHART_EXPLANATIONS["heatmap"]["formula"], ""),
                rx.cond(chart_key == "circadian", CHART_EXPLANATIONS["circadian"]["formula"], ""),
                rx.cond(chart_key == "flow_prob", CHART_EXPLANATIONS["flow_prob"]["formula"], ""),
                rx.cond(chart_key == "flow_calendar", CHART_EXPLANATIONS["flow_calendar"]["formula"], ""),
                rx.cond(chart_key == "fragmentation", CHART_EXPLANATIONS["fragmentation"]["formula"], ""),
                rx.cond(chart_key == "sleep_pattern", CHART_EXPLANATIONS["sleep_pattern"]["formula"], ""),
                rx.cond(chart_key == "spiral", CHART_EXPLANATIONS["spiral"]["formula"], ""),
                rx.cond(chart_key == "chord", CHART_EXPLANATIONS["chord"]["formula"], ""),
                rx.cond(chart_key == "violin", CHART_EXPLANATIONS["violin"]["formula"], ""),
                rx.cond(chart_key == "streamgraph", CHART_EXPLANATIONS["streamgraph"]["formula"], ""),
                rx.cond(chart_key == "rose", CHART_EXPLANATIONS["rose"]["formula"], ""),
                rx.cond(chart_key == "barcode", CHART_EXPLANATIONS["barcode"]["formula"], ""),
                rx.cond(chart_key == "energy", CHART_EXPLANATIONS["energy"]["formula"], ""),
                rx.cond(chart_key == "pulse", CHART_EXPLANATIONS["pulse"]["formula"], ""),
                rx.cond(chart_key == "network", CHART_EXPLANATIONS["network"]["formula"], ""),
                rx.cond(chart_key == "gantt", CHART_EXPLANATIONS["gantt"]["formula"], ""),
                rx.cond(chart_key == "recovery", CHART_EXPLANATIONS["recovery"]["formula"], ""),
                rx.cond(chart_key == "streak", CHART_EXPLANATIONS["streak"]["formula"], ""),
                rx.cond(chart_key == "weekly", CHART_EXPLANATIONS["weekly"]["formula"], ""),
                rx.cond(chart_key == "burnout", CHART_EXPLANATIONS["burnout"]["formula"], ""),
                rx.cond(chart_key == "peak", CHART_EXPLANATIONS["peak"]["formula"], ""),
                font_size="10px",
                padding="8px",
                background="#f0f0f0",
                border_radius="4px",
                width="100%",
            ),
            rx.box(height="12px"),
            # Description
            rx.text("DESCRIPTION", font_size="9px", color=GREY, letter_spacing="0.05em"),
            rx.text(
                rx.cond(chart_key == "flow_vs_shallow", CHART_EXPLANATIONS["flow_vs_shallow"]["description"], ""),
                rx.cond(chart_key == "deep_work_trend", CHART_EXPLANATIONS["deep_work_trend"]["description"], ""),
                rx.cond(chart_key == "daily_breakdown", CHART_EXPLANATIONS["daily_breakdown"]["description"], ""),
                rx.cond(chart_key == "session_length", CHART_EXPLANATIONS["session_length"]["description"], ""),
                rx.cond(chart_key == "heatmap", CHART_EXPLANATIONS["heatmap"]["description"], ""),
                rx.cond(chart_key == "circadian", CHART_EXPLANATIONS["circadian"]["description"], ""),
                rx.cond(chart_key == "flow_prob", CHART_EXPLANATIONS["flow_prob"]["description"], ""),
                rx.cond(chart_key == "flow_calendar", CHART_EXPLANATIONS["flow_calendar"]["description"], ""),
                rx.cond(chart_key == "fragmentation", CHART_EXPLANATIONS["fragmentation"]["description"], ""),
                rx.cond(chart_key == "sleep_pattern", CHART_EXPLANATIONS["sleep_pattern"]["description"], ""),
                rx.cond(chart_key == "spiral", CHART_EXPLANATIONS["spiral"]["description"], ""),
                rx.cond(chart_key == "chord", CHART_EXPLANATIONS["chord"]["description"], ""),
                rx.cond(chart_key == "violin", CHART_EXPLANATIONS["violin"]["description"], ""),
                rx.cond(chart_key == "streamgraph", CHART_EXPLANATIONS["streamgraph"]["description"], ""),
                rx.cond(chart_key == "rose", CHART_EXPLANATIONS["rose"]["description"], ""),
                rx.cond(chart_key == "barcode", CHART_EXPLANATIONS["barcode"]["description"], ""),
                rx.cond(chart_key == "energy", CHART_EXPLANATIONS["energy"]["description"], ""),
                rx.cond(chart_key == "pulse", CHART_EXPLANATIONS["pulse"]["description"], ""),
                rx.cond(chart_key == "network", CHART_EXPLANATIONS["network"]["description"], ""),
                rx.cond(chart_key == "gantt", CHART_EXPLANATIONS["gantt"]["description"], ""),
                rx.cond(chart_key == "recovery", CHART_EXPLANATIONS["recovery"]["description"], ""),
                rx.cond(chart_key == "streak", CHART_EXPLANATIONS["streak"]["description"], ""),
                rx.cond(chart_key == "weekly", CHART_EXPLANATIONS["weekly"]["description"], ""),
                rx.cond(chart_key == "burnout", CHART_EXPLANATIONS["burnout"]["description"], ""),
                rx.cond(chart_key == "peak", CHART_EXPLANATIONS["peak"]["description"], ""),
                font_size="11px",
                line_height="1.5",
                color="#333",
            ),
            rx.box(height="12px"),
            # Interpretation
            rx.text("INTERPRETATION", font_size="9px", color=GREY, letter_spacing="0.05em"),
            rx.text(
                rx.cond(chart_key == "flow_vs_shallow", CHART_EXPLANATIONS["flow_vs_shallow"]["interpretation"], ""),
                rx.cond(chart_key == "deep_work_trend", CHART_EXPLANATIONS["deep_work_trend"]["interpretation"], ""),
                rx.cond(chart_key == "daily_breakdown", CHART_EXPLANATIONS["daily_breakdown"]["interpretation"], ""),
                rx.cond(chart_key == "session_length", CHART_EXPLANATIONS["session_length"]["interpretation"], ""),
                rx.cond(chart_key == "heatmap", CHART_EXPLANATIONS["heatmap"]["interpretation"], ""),
                rx.cond(chart_key == "circadian", CHART_EXPLANATIONS["circadian"]["interpretation"], ""),
                rx.cond(chart_key == "flow_prob", CHART_EXPLANATIONS["flow_prob"]["interpretation"], ""),
                rx.cond(chart_key == "flow_calendar", CHART_EXPLANATIONS["flow_calendar"]["interpretation"], ""),
                rx.cond(chart_key == "fragmentation", CHART_EXPLANATIONS["fragmentation"]["interpretation"], ""),
                rx.cond(chart_key == "sleep_pattern", CHART_EXPLANATIONS["sleep_pattern"]["interpretation"], ""),
                rx.cond(chart_key == "spiral", CHART_EXPLANATIONS["spiral"]["interpretation"], ""),
                rx.cond(chart_key == "chord", CHART_EXPLANATIONS["chord"]["interpretation"], ""),
                rx.cond(chart_key == "violin", CHART_EXPLANATIONS["violin"]["interpretation"], ""),
                rx.cond(chart_key == "streamgraph", CHART_EXPLANATIONS["streamgraph"]["interpretation"], ""),
                rx.cond(chart_key == "rose", CHART_EXPLANATIONS["rose"]["interpretation"], ""),
                rx.cond(chart_key == "barcode", CHART_EXPLANATIONS["barcode"]["interpretation"], ""),
                rx.cond(chart_key == "energy", CHART_EXPLANATIONS["energy"]["interpretation"], ""),
                rx.cond(chart_key == "pulse", CHART_EXPLANATIONS["pulse"]["interpretation"], ""),
                rx.cond(chart_key == "network", CHART_EXPLANATIONS["network"]["interpretation"], ""),
                rx.cond(chart_key == "gantt", CHART_EXPLANATIONS["gantt"]["interpretation"], ""),
                rx.cond(chart_key == "recovery", CHART_EXPLANATIONS["recovery"]["interpretation"], ""),
                rx.cond(chart_key == "streak", CHART_EXPLANATIONS["streak"]["interpretation"], ""),
                rx.cond(chart_key == "weekly", CHART_EXPLANATIONS["weekly"]["interpretation"], ""),
                rx.cond(chart_key == "burnout", CHART_EXPLANATIONS["burnout"]["interpretation"], ""),
                rx.cond(chart_key == "peak", CHART_EXPLANATIONS["peak"]["interpretation"], ""),
                font_size="11px",
                line_height="1.5",
                color="#666",
                font_style="italic",
            ),
            spacing="2",
            align_items="start",
            width="100%",
        ),
        width="280px",
        padding="20px",
        background=LIGHT_GREY,
        border_left=f"1px solid {BORDER_COLOR}",
        height="100%",
        overflow_y="auto",
    )


def focus_overlay() -> rx.Component:
    """Fullscreen overlay for focused chart view."""
    return rx.cond(
        State.focused_chart != "",
        rx.box(
            # Hidden input for ESC key capture
            rx.el.input(
                id="focus-trap",
                on_key_down=lambda e: State.handle_key_down(e),
                style={
                    "position": "absolute",
                    "opacity": "0",
                    "width": "1px",
                    "height": "1px",
                    "top": "0",
                    "left": "0",
                    "pointer-events": "none",
                },
                auto_focus=True,
            ),
            # Header bar
            rx.hstack(
                rx.text(State.focused_chart.upper().replace("_", " "), font_size="14px", font_weight="700", letter_spacing="0.1em"),
                rx.spacer(),
                rx.text("Press ESC or click × to close", font_size="10px", color=GREY),
                rx.button(
                    "×",
                    on_click=State.close_focus,
                    font_size="24px",
                    padding="4px 16px",
                    background="transparent",
                    border="none",
                    cursor="pointer",
                    color=BLACK,
                    _hover={"background": "#eee"},
                ),
                width="100%",
                padding="12px 20px",
                align_items="center",
                border_bottom=f"1px solid {BORDER_COLOR}",
                background=WHITE,
            ),
            # Main content: chart + explanation
            rx.hstack(
                # Chart container (centered, takes remaining space)
                rx.box(
                    rx.cond(State.focused_chart == "flow_vs_shallow", rx.plotly(data=State.chart_flow_shallow, height="100%", width="100%"), rx.fragment()),
                    rx.cond(State.focused_chart == "deep_work_trend", rx.plotly(data=State.chart_trend, height="100%", width="100%"), rx.fragment()),
                    rx.cond(State.focused_chart == "daily_breakdown", rx.plotly(data=State.chart_breakdown, height="100%", width="100%"), rx.fragment()),
                    rx.cond(State.focused_chart == "session_length", rx.plotly(data=State.chart_session_dist, height="100%", width="100%"), rx.fragment()),
                    rx.cond(State.focused_chart == "heatmap", rx.plotly(data=State.chart_heatmap, height="100%", width="100%"), rx.fragment()),
                    rx.cond(State.focused_chart == "circadian", rx.plotly(data=State.chart_hourly, height="100%", width="100%"), rx.fragment()),
                    rx.cond(State.focused_chart == "flow_prob", rx.plotly(data=State.chart_flow_prob, height="100%", width="100%"), rx.fragment()),
                    rx.cond(State.focused_chart == "flow_calendar", rx.plotly(data=State.chart_flow_calendar, height="100%", width="100%"), rx.fragment()),
                    rx.cond(State.focused_chart == "fragmentation", rx.plotly(data=State.chart_frag, height="100%", width="100%"), rx.fragment()),
                    rx.cond(State.focused_chart == "sleep_pattern", rx.plotly(data=State.chart_sleep, height="100%", width="100%"), rx.fragment()),
                    rx.cond(State.focused_chart == "spiral", rx.plotly(data=State.chart_spiral, height="100%", width="100%"), rx.fragment()),
                    rx.cond(State.focused_chart == "chord", rx.plotly(data=State.chart_chord, height="100%", width="100%"), rx.fragment()),
                    rx.cond(State.focused_chart == "violin", rx.plotly(data=State.chart_violin, height="100%", width="100%"), rx.fragment()),
                    rx.cond(State.focused_chart == "streamgraph", rx.plotly(data=State.chart_streamgraph, height="100%", width="100%"), rx.fragment()),
                    rx.cond(State.focused_chart == "rose", rx.plotly(data=State.chart_rose, height="100%", width="100%"), rx.fragment()),
                    rx.cond(State.focused_chart == "barcode", rx.plotly(data=State.chart_barcode, height="100%", width="100%"), rx.fragment()),
                    rx.cond(State.focused_chart == "energy", rx.plotly(data=State.chart_energy, height="100%", width="100%"), rx.fragment()),
                    rx.cond(State.focused_chart == "pulse", rx.plotly(data=State.chart_pulse, height="100%", width="100%"), rx.fragment()),
                    rx.cond(State.focused_chart == "network", rx.plotly(data=State.chart_network, height="100%", width="100%"), rx.fragment()),
                    rx.cond(State.focused_chart == "gantt", rx.plotly(data=State.chart_gantt, height="100%", width="100%"), rx.fragment()),
                    rx.cond(State.focused_chart == "recovery", rx.plotly(data=State.chart_recovery, height="100%", width="100%"), rx.fragment()),
                    rx.cond(State.focused_chart == "streak", rx.plotly(data=State.chart_streak, height="100%", width="100%"), rx.fragment()),
                    rx.cond(State.focused_chart == "weekly", rx.plotly(data=State.chart_weekly, height="100%", width="100%"), rx.fragment()),
                    rx.cond(State.focused_chart == "burnout", rx.plotly(data=State.chart_burnout, height="100%", width="100%"), rx.fragment()),
                    rx.cond(State.focused_chart == "peak", rx.plotly(data=State.chart_peak, height="100%", width="100%"), rx.fragment()),
                    flex="1",
                    padding="20px",
                    display="flex",
                    align_items="center",
                    justify_content="center",
                    height="100%",
                ),
                # Explanation panel on the right
                chart_explanation(State.focused_chart),
                spacing="0",
                width="100%",
                height="100%",
                align_items="stretch",
            ),
            position="fixed",
            top="0",
            left="0",
            width="100vw",
            height="100vh",
            background=WHITE,
            z_index="300",
            display="flex",
            flex_direction="column",
            font_family=MONO,
        ),
        rx.fragment(),
    )


def tab_button(tab_key: str, label: str, icon: str) -> rx.Component:
    """Create a tab button for chart groups."""
    return rx.button(
        rx.icon(icon, size=12),
        rx.text(label, font_size="9px", margin_left="4px"),
        on_click=lambda: State.set_tab(tab_key),
        background=rx.cond(State.active_tab == tab_key, "rgba(255,255,255,0.15)", "transparent"),
        border="none",
        color=WHITE,
        padding="4px 8px",
        border_radius="3px",
        cursor="pointer",
        _hover={"background": "rgba(255,255,255,0.1)"},
    )


def comparison_stat(label: str, current, previous, suffix: str = "") -> rx.Component:
    """Stat with comparison indicator."""
    diff = current - previous
    is_positive = diff > 0
    return rx.hstack(
        rx.text(f"{label}:", font_size="9px", color=GREY),
        rx.text(current, font_size="11px", font_weight="700"),
        rx.text(suffix, font_size="9px", color=GREY) if suffix else rx.fragment(),
        rx.cond(
            diff != 0,
            rx.text(
                rx.cond(is_positive, f"+{diff:.1f}", f"{diff:.1f}"),
                font_size="8px",
                color=rx.cond(is_positive, "#4CAF50", "#F44336"),
                margin_left="2px",
            ),
            rx.fragment(),
        ),
        spacing="1",
        align_items="baseline",
    )


def comparison_panel() -> rx.Component:
    """Comparison mode panel below header."""
    return rx.cond(
        State.comparison_mode,
        rx.box(
            rx.hstack(
                rx.text("COMPARING:", font_size="9px", color=GREY, letter_spacing="0.05em"),
                rx.button(
                    "This Week vs Last",
                    on_click=lambda: State.set_comparison_preset("week"),
                    background=rx.cond(State.comparison_preset == "week", "#333", "transparent"),
                    border="1px solid #444",
                    color=WHITE,
                    padding="2px 8px",
                    font_size="9px",
                    border_radius="3px",
                    cursor="pointer",
                ),
                rx.button(
                    "This Month vs Last",
                    on_click=lambda: State.set_comparison_preset("month"),
                    background=rx.cond(State.comparison_preset == "month", "#333", "transparent"),
                    border="1px solid #444",
                    color=WHITE,
                    padding="2px 8px",
                    font_size="9px",
                    border_radius="3px",
                    cursor="pointer",
                ),
                rx.box(width="16px"),
                comparison_stat("Total", State.comp_total_hours_current, State.comp_total_hours_previous, "h"),
                comparison_stat("Deep", State.comp_deep_work_current, State.comp_deep_work_previous, "h"),
                comparison_stat("Flow", State.comp_flow_index_current, State.comp_flow_index_previous, "%"),
                comparison_stat("Pulse", State.comp_pulse_current, State.comp_pulse_previous, ""),
                rx.spacer(),
                rx.button(
                    rx.icon("x", size=12),
                    on_click=State.toggle_comparison,
                    background="transparent",
                    border="none",
                    color=GREY,
                    padding="2px",
                    cursor="pointer",
                ),
                width="100%",
                padding_x="8px",
                align_items="center",
                spacing="3",
            ),
            background="#1a1a1a",
            padding_y="4px",
            font_family=MONO,
            color=WHITE,
        ),
        rx.fragment(),
    )


def index() -> rx.Component:
    return rx.box(
        # Header row 1: Logo, stats, controls
        rx.box(
            rx.hstack(
                rx.text("TIME", font_size="12px", font_weight="700", color=WHITE, letter_spacing="0.1em"),
                rx.box(width="12px"),
                # Inline stats (compact)
                stat_item("Tot", State.total_hours, "h"),
                stat_item("Deep", State.deep_work_hours, "h"),
                stat_item("DWR", State.deep_work_ratio, "%"),
                stat_item("Flow", State.flow_index, "%"),
                stat_item("Pulse", State.productivity_pulse, ""),
                rx.spacer(),
                # Time range slider
                rx.hstack(
                    rx.text(State.range_label, font_size="9px", color="rgba(255,255,255,0.5)"),
                    rx.slider(
                        default_value=[0, 100],
                        min=0,
                        max=100,
                        step=1,
                        on_value_commit=State.set_range,
                        width="120px",
                        size="1",
                    ),
                    spacing="2",
                    align_items="center",
                ),
                rx.button(
                    rx.icon("git-compare", size=14),
                    background=rx.cond(State.comparison_mode, "rgba(255,255,255,0.2)", "transparent"),
                    border="none",
                    color=WHITE,
                    padding="4px",
                    margin_left="4px",
                    on_click=State.toggle_comparison,
                    title="Compare periods",
                ),
                rx.button(
                    rx.icon("target", size=14),
                    background=rx.cond(State.goals_panel_open, "rgba(255,255,255,0.2)", "transparent"),
                    border="none",
                    color=WHITE,
                    padding="4px",
                    on_click=State.toggle_goals_panel,
                    title="Goals",
                ),
                rx.button(
                    rx.icon("trending-up", size=14),
                    background=rx.cond(State.predictions_panel_open, "rgba(255,255,255,0.2)", "transparent"),
                    border="none",
                    color=WHITE,
                    padding="4px",
                    on_click=State.toggle_predictions_panel,
                    title="Voorspellingen",
                ),
                rx.button(
                    rx.icon("trophy", size=14),
                    background=rx.cond(State.gamification_panel_open, "rgba(255,255,255,0.2)", "transparent"),
                    border="none",
                    color=WHITE,
                    padding="4px",
                    on_click=State.toggle_gamification_panel,
                    title="Achievements",
                ),
                rx.button(
                    rx.icon("sparkles", size=14),
                    background=rx.cond(State.insights_panel_open, "rgba(255,255,255,0.2)", "transparent"),
                    border="none",
                    color=WHITE,
                    padding="4px",
                    on_click=State.toggle_insights_panel,
                    title="AI Insights",
                ),
                rx.button(
                    rx.icon("list", size=14),
                    background=rx.cond(State.activity_log_open, "rgba(255,255,255,0.2)", "transparent"),
                    border="none",
                    color=WHITE,
                    padding="4px",
                    on_click=State.toggle_activity_log,
                    title="Activiteiten Log",
                ),
                # Sync status indicator
                rx.tooltip(
                    rx.hstack(
                        rx.cond(
                            State.sync_is_syncing,
                            rx.icon("refresh-cw", size=12, color="#4ade80", class_name="animate-spin"),
                            rx.cond(
                                State.integrations_configured.length() > 0,
                                rx.icon("cloud", size=12, color="#4ade80"),
                                rx.icon("cloud-off", size=12, color="#888"),
                            ),
                        ),
                        spacing="1",
                        align_items="center",
                        padding="4px",
                        cursor="pointer",
                        on_click=State.toggle_sync,
                    ),
                    content=State.sync_status,
                ),
                # Eco mode toggle (for Pi optimization)
                rx.tooltip(
                    rx.button(
                        rx.cond(
                            State.eco_mode,
                            rx.icon("leaf", size=12, color="#4ade80"),
                            rx.icon("zap", size=12, color="#888"),
                        ),
                        background=rx.cond(State.eco_mode, "rgba(74,222,128,0.2)", "transparent"),
                        border="none",
                        padding="4px",
                        cursor="pointer",
                        on_click=State.toggle_eco_mode,
                    ),
                    content=rx.cond(
                        State.eco_mode,
                        "🍃 Eco-mode (15min sync, minder CPU)",
                        "⚡ Normale mode - klik voor eco",
                    ),
                ),
                rx.link(
                    rx.button(
                        rx.icon("download", size=14),
                        background="transparent",
                        border="none",
                        color=WHITE,
                        padding="4px",
                        cursor="pointer",
                        title="Export Report",
                    ),
                    href=State.report_download_url,
                    download="time-report.html",
                    is_external=True,
                ),
                rx.button(
                    rx.cond(
                        State.dark_mode,
                        rx.icon("sun", size=14),
                        rx.icon("moon", size=14),
                    ),
                    background=rx.cond(State.dark_mode, "rgba(255,255,255,0.2)", "transparent"),
                    border="none",
                    color=WHITE,
                    padding="4px",
                    on_click=State.toggle_dark_mode,
                    title="Dark mode",
                ),
                rx.button(
                    rx.icon("settings", size=14),
                    background="transparent",
                    border="none",
                    color=WHITE,
                    padding="4px",
                    on_click=State.toggle_chart_panel,
                    title="Toggle charts",
                ),
                rx.button(
                    rx.icon("upload", size=14),
                    background="transparent",
                    border="none",
                    color=WHITE,
                    padding="4px",
                    on_click=State.toggle_menu_sidebar,
                    title="Upload CSV",
                ),
                width="100%",
                padding_x="8px",
                align_items="center",
                spacing="2",
                color=WHITE,
            ),
            background=BLACK,
            padding_y="4px",
            font_family=MONO,
            flex_shrink="0",
        ),
        
        # Header row 2: Chart tabs
        rx.box(
            rx.hstack(
                tab_button("all", "All", "layout-grid"),
                tab_button("flow", "Flow", "zap"),
                tab_button("time", "Time", "clock"),
                tab_button("rhythm", "Rhythm", "sun"),
                tab_button("patterns", "Patterns", "git-branch"),
                tab_button("wellness", "Wellness", "heart"),
                rx.spacer(),
                rx.text(
                    rx.cond(
                        State.active_tab == "all",
                        "25 charts",
                        rx.cond(State.active_tab == "flow", "7 charts",
                        rx.cond(State.active_tab == "time", "4 charts",
                        rx.cond(State.active_tab == "rhythm", "6 charts",
                        rx.cond(State.active_tab == "patterns", "4 charts", "4 charts"))))
                    ),
                    font_size="9px",
                    color="rgba(255,255,255,0.4)",
                ),
                width="100%",
                padding_x="8px",
                align_items="center",
                spacing="1",
            ),
            background="#222",
            padding_y="4px",
            font_family=MONO,
            flex_shrink="0",
        ),
        
        # Comparison panel (conditional)
        comparison_panel(),

        # Main content fills remaining space
        main_content(),

        menu_sidebar(),
        toggle_panel(),
        insights_panel(),
        goals_panel(),
        predictions_panel(),
        gamification_panel(),
        activity_log_panel(),
        focus_overlay(),
        keyboard_shortcuts_help(),
        background=rx.cond(State.dark_mode, "#1a1a1a", WHITE),
        color=rx.cond(State.dark_mode, "#e0e0e0", BLACK),
        height="100vh",
        overflow="hidden",
        display="flex",
        flex_direction="column",
        on_mount=State.load_data,
        class_name=rx.cond(State.dark_mode, "dark-mode", ""),
    )


# Custom CSS for animations
CUSTOM_CSS = """
@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}
.animate-spin {
    animation: spin 1s linear infinite;
}
"""

app = rx.App(
    style={"font_family": MONO, "background": WHITE},
    stylesheets=[
        "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap",
    ],
    head_components=[
        rx.el.style(CUSTOM_CSS),
    ],
)
app.add_page(index, title="TIME")