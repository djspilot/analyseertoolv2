"""
Visualization module for Plotly charts.
Creates interactive Gantt charts, pie charts, and trend visualizations.
With caching support for performance optimization.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from ..config import (
    CATEGORY_COLORS,
    DEFAULT_CHART_HEIGHT,
    GANNT_CHART_HEIGHT,
    HEATMAP_CHART_HEIGHT,
    FLOW_MIN_DURATION_HOURS,
    DEEP_WORK_CATEGORIES,
)
from ..api.cache import cache_chart, ENABLE_CACHE
from .utils import get_color, ensure_datetime, filter_deep_work, build_color_map


def _extract_activity_name(activity_type: str) -> str:
    """Extract activity name from 'Activity (Category)' format."""
    if " (" in activity_type:
        return activity_type.split(" (")[0]
    return activity_type


def _map_weights(activity_type: str, weights: dict, default: float = 0) -> float:
    """Map activity type to weight, handling 'Activity (Category)' format."""
    if activity_type in weights:
        return weights[activity_type]
    name = _extract_activity_name(activity_type)
    return weights.get(name, default)


def _matches_category(activity_type: str, categories: list) -> bool:
    """Check if activity type matches any category (substring matching)."""
    return any(cat in activity_type for cat in categories)


@cache_chart(ttl_minutes=60, enabled=ENABLE_CACHE)
def create_gantt_chart(df: pd.DataFrame, date_filter: Optional[datetime] = None) -> go.Figure:
    """
    Create an interactive Gantt chart (timeline) for activities.
    CACHED: Results are cached for 60 minutes.
    
    Args:
        df: DataFrame with datetime_from, datetime_to, activity_type columns
        date_filter: Optional specific date to filter to
        
    Returns:
        Plotly Figure with Gantt chart
    """
    plot_df = ensure_datetime(df, ['datetime_from', 'datetime_to'])

    # Filter to specific date if provided
    if date_filter:
        plot_df = plot_df[plot_df['datetime_from'].dt.date == date_filter.date()]
    
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data for selected date",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Create Gantt chart using timeline
    color_map = build_color_map(plot_df['activity_type'].unique().tolist())
    fig = px.timeline(
        plot_df,
        x_start='datetime_from',
        x_end='datetime_to',
        y='activity_type',
        color='activity_type',
        color_discrete_map=color_map,
        hover_data={
            'duration_hours': ':.2f',
            'datetime_from': '|%H:%M',
            'datetime_to': '|%H:%M',
            'activity_type': True
        },
        title='Activity Timeline'
    )
    
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Activity Type',
        showlegend=True,
        legend_title='Category',
        hovermode='closest'
    )
    
    fig.update_yaxes(categoryorder='total ascending')
    
    return fig


def create_category_pie(df: pd.DataFrame) -> go.Figure:
    """
    Create a pie chart showing time distribution by category.
    
    Args:
        df: DataFrame with activity_type and duration_hours columns
        
    Returns:
        Plotly Figure with the pie chart
    """
    summary = df.groupby('activity_type')['duration_hours'].sum().reset_index()
    summary = summary.sort_values('duration_hours', ascending=False)
    
    colors = [get_color(cat) for cat in summary['activity_type']]
    
    fig = go.Figure(data=[go.Pie(
        labels=summary['activity_type'],
        values=summary['duration_hours'],
        marker=dict(colors=colors),
        hole=0.4,
        textinfo='label+percent',
        textposition='outside',
        hovertemplate='<b>%{label}</b><br>%{value:.1f} hours<br>%{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title='Time Distribution by Category',
        showlegend=False
    )
    
    return fig


@cache_chart(ttl_minutes=30, enabled=ENABLE_CACHE)
def create_daily_breakdown(df: pd.DataFrame, last_n_days: Optional[int] = 14, resample_rule: str = 'D') -> go.Figure:
    """
    Create a stacked bar chart showing breakdown by category, with flexible resampling.
    CACHED: Results are cached for 30 minutes.
    
    Args:
        df: DataFrame with datetime_from, activity_type, duration_hours columns
        last_n_days: Number of days to show (None for all)
        resample_rule: 'D' for Daily, 'W' for Weekly, 'M' for Monthly
        
    Returns:
        Plotly Figure with stacked bar chart
    """
    plot_df = ensure_datetime(df)

    # Add date column if not present
    if 'date' not in plot_df.columns:
        plot_df['date'] = plot_df['datetime_from'].dt.date

    # Filter to last N days
    if last_n_days:
        cutoff = datetime.now().date() - timedelta(days=last_n_days)
        plot_df = plot_df[plot_df['date'] >= cutoff]
    
    # Resample logic
    if resample_rule != 'D':
        # Need datetime index for resampling
        plot_df = plot_df.set_index('datetime_from')
        # Group by category and resample
        daily = plot_df.groupby('activity_type')['duration_hours'].resample(resample_rule).sum().reset_index()
        # Rename date column back
        daily = daily.rename(columns={'datetime_from': 'date'})
    else:
        daily = plot_df.groupby(['date', 'activity_type'])['duration_hours'].sum().reset_index()
    
    title_suffix = {
        'D': f'(Last {last_n_days} Days)' if last_n_days else '(Daily)',
        'W': '(Weekly Aggregation)',
        'M': '(Monthly Aggregation)'
    }
    
    # Build dynamic color map for actual activity types
    color_map = build_color_map(daily['activity_type'].unique().tolist())
    
    fig = px.bar(
        daily,
        x='date',
        y='duration_hours',
        color='activity_type',
        color_discrete_map=color_map,
        title=f'Activity Breakdown {title_suffix.get(resample_rule, "")}',
        labels={'duration_hours': 'Hours', 'date': 'Date', 'activity_type': 'Category'}
    )
    
    fig.update_layout(
        barmode='stack',
        xaxis_title='Date',
        yaxis_title='Hours',
        legend_title='Category'
    )

    return fig


def create_consistency_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Create a Github-style heatmap of Deep Work intensity.
    
    Args:
        df: DataFrame with datetime_from and is_deep_work columns
    """
    plot_df = ensure_datetime(df)
    plot_df['date'] = plot_df['datetime_from'].dt.date

    # Calculate daily deep work
    daily_dw = plot_df[plot_df['is_deep_work'] == 1].groupby('date')['duration_hours'].sum().reset_index()
    
    # Create Z data (hours)
    fig = go.Figure(data=go.Heatmap(
        z=daily_dw['duration_hours'],
        x=daily_dw['date'],
        y=['Deep Work'] * len(daily_dw), # Single row
        colorscale='Greens',
        showscale=False
    ))
    
    fig.update_layout(
        title='Deep Work Consistency (Days)',
        yaxis={'visible': False},
        xaxis_title='Date'
    )
    
    return fig


def create_circadian_plot(daily_metrics: pd.DataFrame) -> go.Figure:
    """
    Create a plot showing day start and end times.
    
    Args:
        daily_metrics: DataFrame from calculate_circadian_metrics
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_metrics['date'],
        y=daily_metrics['start_hour'],
        name='Day Start',
        mode='markers',
        marker=dict(color='#FFA726', size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=daily_metrics['date'],
        y=daily_metrics['end_hour'],
        name='Day End',
        mode='markers',
        marker=dict(color='#7E57C2', size=8)
    ))
    
    fig.update_layout(
        title='Circadian Rhythm (Wake/Sleep Cycle)',
        yaxis_title='Hour of Day (24h)',
        xaxis_title='Date',
        yaxis=dict(range=[0, 24])  # Fixed 24h axis
    )
    
    return fig


@cache_chart(ttl_minutes=15, enabled=ENABLE_CACHE)
def create_deep_work_trend(df: pd.DataFrame, window: int = 7) -> go.Figure:
    """
    Create a line chart showing Deep Work hours over time with moving average.
    CACHED: Results are cached for 15 minutes.
    
    Args:
        df: DataFrame with datetime_from, is_deep_work, duration_hours columns
        window: Rolling average window in days
        
    Returns:
        Plotly Figure with trend line
    """
    plot_df = ensure_datetime(df)

    # Add date column if not present
    if 'date' not in plot_df.columns:
        plot_df['date'] = plot_df['datetime_from'].dt.date

    # Calculate daily deep work hours
    daily_deep_work = plot_df[plot_df['is_deep_work'] == 1].groupby('date')['duration_hours'].sum()
    
    # Create a complete date range to fill in zeros
    if not daily_deep_work.empty:
        all_dates = pd.date_range(daily_deep_work.index.min(), daily_deep_work.index.max(), freq='D').date
        daily_deep_work = daily_deep_work.reindex(all_dates, fill_value=0)
    
    # Calculate rolling average
    rolling_avg = daily_deep_work.rolling(window=window, min_periods=1).mean()
    
    # Create figure
    fig = go.Figure()
    
    # Daily values
    fig.add_trace(go.Bar(
        x=list(daily_deep_work.index),
        y=daily_deep_work.values,
        name='Daily Deep Work',
        marker_color='rgba(76, 175, 80, 0.5)'
    ))
    
    # Rolling average line
    fig.add_trace(go.Scatter(
        x=list(rolling_avg.index),
        y=rolling_avg.values,
        name=f'{window}-Day Average',
        line=dict(color='#4CAF50', width=3),
        mode='lines'
    ))
    
    fig.update_layout(
        title='Deep Work Trend',
        xaxis_title='Date',
        yaxis_title='Hours',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig


def _create_weekday_hour_heatmap(
    df: pd.DataFrame,
    colorscale: str = 'Blues',
    title: str = 'Activity Heatmap',
    ensure_all_hours: bool = True
) -> go.Figure:
    """
    Internal function to create weekday x hour heatmaps.

    Args:
        df: DataFrame with datetime_from and duration_hours columns
        colorscale: Plotly colorscale name ('Blues', 'Viridis', etc.)
        title: Chart title
        ensure_all_hours: If True, ensures all 24 hours are represented

    Returns:
        Plotly Figure with the heatmap
    """
    plot_df = ensure_datetime(df)
    plot_df['hour'] = plot_df['datetime_from'].dt.hour
    plot_df['weekday'] = plot_df['datetime_from'].dt.day_name()

    heatmap_data = plot_df.groupby(['weekday', 'hour'])['duration_hours'].sum().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='weekday', columns='hour', values='duration_hours').fillna(0)

    # Ensure all hours 0-23 are present if requested
    if ensure_all_hours:
        for h in range(24):
            if h not in heatmap_pivot.columns:
                heatmap_pivot[h] = 0
        heatmap_pivot = heatmap_pivot.reindex(columns=range(24))

    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_pivot = heatmap_pivot.reindex([d for d in day_order if d in heatmap_pivot.index])

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=list(range(24)) if ensure_all_hours else heatmap_pivot.columns.tolist(),
        y=heatmap_pivot.index.tolist(),
        colorscale=colorscale,
        hovertemplate='%{y} at %{x}:00<br>%{z:.1f} hours<extra></extra>'
    ))

    layout_kwargs = {
        'title': title,
        'xaxis_title': 'Hour of Day',
        'yaxis_title': 'Day of Week'
    }
    if ensure_all_hours:
        layout_kwargs['xaxis'] = dict(tickmode='linear', tick0=0, dtick=2)

    fig.update_layout(**layout_kwargs)
    return fig


def create_weekly_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create a heatmap showing activity intensity by day of week and hour."""
    return _create_weekday_hour_heatmap(
        df,
        colorscale='Viridis',
        title='Activity Intensity by Day and Hour',
        ensure_all_hours=False
    )


def create_activity_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create a heatmap showing activity intensity by hour and day of week."""
    return _create_weekday_hour_heatmap(
        df,
        colorscale='Blues',
        title='Activity Heatmap',
        ensure_all_hours=True
    )


def create_fragmentation_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart showing Fragmentation Index per category.
    Higher FI = more fragmented (many short sessions).
    
    Args:
        df: DataFrame with activity_type and duration_hours columns
        
    Returns:
        Plotly Figure with bar chart sorted by FI value
    """
    fi_data = []
    
    for category in df['activity_type'].unique():
        cat_df = df[df['activity_type'] == category]
        total_hours = cat_df['duration_hours'].sum()
        session_count = len(cat_df)
        
        if total_hours > 0:
            avg_session_length = total_hours / session_count
            fi = session_count / total_hours
            fi_data.append({
                'category': category,
                'fragmentation_index': fi,
                'avg_session_min': avg_session_length * 60,
                'sessions': session_count
            })
    
    fi_df = pd.DataFrame(fi_data).sort_values('fragmentation_index', ascending=True)
    
    colors = [get_color(cat) for cat in fi_df['category']]
    
    fig = go.Figure(data=go.Bar(
        x=fi_df['fragmentation_index'],
        y=fi_df['category'],
        orientation='h',
        marker_color=colors,
        hovertemplate='<b>%{y}</b><br>FI: %{x:.2f}<br>Avg session: %{customdata[0]:.0f} min<br>Sessions: %{customdata[1]}<extra></extra>',
        customdata=fi_df[['avg_session_min', 'sessions']].values
    ))
    
    fig.update_layout(
        title='Fragmentation Index by Category',
        xaxis_title='Fragmentation Index (higher = more fragmented)',
        yaxis_title=''
    )
    
    return fig


def create_sleep_pattern_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a scatter plot showing inferred sleep/wake times.

    Args:
        df: DataFrame with datetime_from and datetime_to columns

    Returns:
        Plotly Figure with sleep start and wake time series
    """
    plot_df = ensure_datetime(df, ['datetime_from', 'datetime_to'])
    plot_df['date'] = plot_df['datetime_from'].dt.date
    
    sleep_data = []
    for date in plot_df['date'].unique():
        day_df = plot_df[plot_df['date'] == date].sort_values('datetime_from')
        if not day_df.empty:
            wake_time = day_df['datetime_from'].iloc[0]
            sleep_start = day_df['datetime_to'].iloc[-1]
            
            sleep_data.append({
                'date': date,
                'wake_hour': wake_time.hour + wake_time.minute / 60,
                'sleep_hour': sleep_start.hour + sleep_start.minute / 60
            })
    
    sleep_df = pd.DataFrame(sleep_data).sort_values('date')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=sleep_df['date'],
        y=sleep_df['wake_hour'],
        name='Wake Time',
        mode='markers+lines',
        marker=dict(color='#FFA726', size=8),
        line=dict(color='#FFA726', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=sleep_df['date'],
        y=sleep_df['sleep_hour'],
        name='Sleep Start',
        mode='markers+lines',
        marker=dict(color='#5C6BC0', size=8),
        line=dict(color='#5C6BC0', width=1)
    ))
    
    fig.update_layout(
        title='Sleep Pattern',
        xaxis_title='Date',
        yaxis_title='Hour of Day',
        yaxis=dict(range=[0, 24], tickmode='linear', tick0=0, dtick=4),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig


def create_hourly_profile(df: pd.DataFrame) -> go.Figure:
    """
    Create a stacked area chart showing average time per activity by hour.

    Args:
        df: DataFrame with datetime_from, activity_type, duration_hours columns

    Returns:
        Plotly Figure showing circadian rhythm of activities
    """
    plot_df = ensure_datetime(df)
    plot_df['hour'] = plot_df['datetime_from'].dt.hour
    plot_df['date'] = plot_df['datetime_from'].dt.date
    
    num_days = plot_df['date'].nunique()
    if num_days == 0:
        num_days = 1
    
    hourly = plot_df.groupby(['hour', 'activity_type'])['duration_hours'].sum().reset_index()
    hourly['avg_hours'] = hourly['duration_hours'] / num_days
    
    hourly_pivot = hourly.pivot(index='hour', columns='activity_type', values='avg_hours').fillna(0)
    
    for h in range(24):
        if h not in hourly_pivot.index:
            hourly_pivot.loc[h] = 0
    hourly_pivot = hourly_pivot.sort_index()
    
    fig = go.Figure()
    
    for activity in hourly_pivot.columns:
        fig.add_trace(go.Scatter(
            x=hourly_pivot.index,
            y=hourly_pivot[activity],
            name=activity,
            mode='lines',
            stackgroup='one',
            line=dict(width=0.5, color=get_color(activity)),
            fillcolor=get_color(activity),
            hovertemplate=f'{activity}<br>Hour: %{{x}}:00<br>Avg: %{{y:.2f}}h<extra></extra>'
        ))
    
    fig.update_layout(
        title='Hourly Activity Profile (Circadian Rhythm)',
        xaxis_title='Hour of Day',
        yaxis_title='Average Hours',
        xaxis=dict(tickmode='linear', tick0=0, dtick=2, range=[0, 23]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    
    return fig


def create_flow_sessions_timeline(df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart showing deep work sessions (Work/Coding).
    Color intensity based on duration, flow sessions (>=90min) highlighted.
    """
    plot_df = ensure_datetime(df, ['datetime_from', 'datetime_to'])
    deep_work = filter_deep_work(plot_df)
    
    if deep_work.empty:
        fig = go.Figure()
        fig.add_annotation(text="No deep work sessions", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    deep_work = deep_work.sort_values('datetime_from').reset_index(drop=True)
    deep_work['session_id'] = range(len(deep_work))
    deep_work['is_flow'] = deep_work['duration_hours'] >= 1.5
    
    fig = go.Figure()
    
    for _, row in deep_work.iterrows():
        is_flow = row['is_flow']
        duration = row['duration_hours']
        color = '#1565C0' if is_flow else '#B0BEC5'
        opacity = min(0.4 + duration * 0.2, 1.0) if not is_flow else 1.0
        
        fig.add_trace(go.Bar(
            x=[duration],
            y=[row['session_id']],
            orientation='h',
            marker=dict(color=color, opacity=opacity),
            hovertemplate=f"{row['activity_type']}<br>{duration:.1f}h<br>{'Flow' if is_flow else 'Shallow'}<extra></extra>",
            showlegend=False
        ))
    
    fig.update_layout(
        title='Deep Work Sessions Timeline',
        xaxis_title='Duration (hours)',
        yaxis_title='Session',
        barmode='overlay',
        yaxis=dict(showticklabels=False)
    )
    
    return fig


def create_session_length_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Histogram of session durations with 90min threshold line.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df['duration_hours'],
        xbins=dict(size=0.25),
        marker_color='#78909C',
        name='Sessions'
    ))
    
    fig.add_vline(x=1.5, line_dash="dash", line_color="#1565C0", line_width=2)
    
    fig.add_annotation(x=0.75, y=0.95, xref="paper", yref="paper",
                      text="Shallow", showarrow=False, font=dict(color='#78909C'))
    fig.add_annotation(x=0.85, y=0.95, xref="paper", yref="paper",
                      text="Flow", showarrow=False, font=dict(color='#1565C0'))
    
    fig.update_layout(
        title='Session Length Distribution',
        xaxis_title='Duration (hours)',
        yaxis_title='Count'
    )
    
    return fig


def create_flow_probability_by_hour(df: pd.DataFrame) -> go.Figure:
    """
    Line chart showing % of deep work sessions at each hour that are >=90min.
    """
    plot_df = ensure_datetime(df)
    deep_work = filter_deep_work(plot_df)
    
    if deep_work.empty:
        fig = go.Figure()
        fig.add_annotation(text="No deep work sessions", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    deep_work['hour'] = deep_work['datetime_from'].dt.hour
    deep_work['is_flow'] = deep_work['duration_hours'] >= 1.5
    
    hourly_stats = deep_work.groupby('hour').agg(
        total=('is_flow', 'count'),
        flow_count=('is_flow', 'sum')
    ).reset_index()
    hourly_stats['flow_pct'] = (hourly_stats['flow_count'] / hourly_stats['total']) * 100
    
    all_hours = pd.DataFrame({'hour': range(24)})
    hourly_stats = all_hours.merge(hourly_stats, on='hour', how='left').fillna(0)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hourly_stats['hour'],
        y=hourly_stats['flow_pct'],
        mode='lines',
        fill='tozeroy',
        line=dict(color='#1565C0', width=2),
        fillcolor='rgba(21, 101, 192, 0.2)',
        hovertemplate='Hour %{x}:00<br>Flow probability: %{y:.0f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='Flow Probability by Hour',
        xaxis_title='Hour of Day',
        yaxis_title='% Sessions ≥90min',
        xaxis=dict(tickmode='linear', tick0=0, dtick=2, range=[0, 23]),
        yaxis=dict(range=[0, 100])
    )
    
    return fig


def create_flow_streak_calendar(df: pd.DataFrame) -> go.Figure:
    """
    Heatmap calendar showing hours of flow (>=90min sessions in Work/Coding) per day.
    """
    plot_df = ensure_datetime(df)
    deep_work = filter_deep_work(plot_df)
    deep_work['date'] = deep_work['datetime_from'].dt.date
    
    flow_sessions = deep_work[deep_work['duration_hours'] >= 1.5]
    daily_flow = flow_sessions.groupby('date')['duration_hours'].sum().reset_index()
    daily_flow.columns = ['date', 'flow_hours']
    
    if daily_flow.empty:
        fig = go.Figure()
        fig.add_annotation(text="No flow sessions found", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    daily_flow['date'] = pd.to_datetime(daily_flow['date'])
    daily_flow['week'] = daily_flow['date'].dt.isocalendar().week
    daily_flow['weekday'] = daily_flow['date'].dt.weekday
    
    pivot = daily_flow.pivot(index='weekday', columns='week', values='flow_hours').fillna(0)
    
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=[day_labels[i] for i in pivot.index],
        colorscale='Greens',
        hovertemplate='Week %{x}<br>%{y}<br>%{z:.1f}h flow<extra></extra>'
    ))
    
    fig.update_layout(
        title='Flow Hours Calendar',
        xaxis_title='Week',
        yaxis_title=''
    )
    
    return fig


def create_flow_vs_shallow_chart(df: pd.DataFrame) -> go.Figure:
    """
    Stacked bar chart per day: Flow (deep work >=90min) vs Shallow (everything else).
    """
    plot_df = ensure_datetime(df)
    plot_df['date'] = plot_df['datetime_from'].dt.date

    deep_work = filter_deep_work(plot_df)
    flow_mask = deep_work['duration_hours'] >= 1.5
    
    flow_hours = deep_work[flow_mask].groupby('date')['duration_hours'].sum()
    shallow_hours = plot_df.groupby('date')['duration_hours'].sum() - flow_hours.reindex(plot_df['date'].unique(), fill_value=0)
    
    daily = pd.DataFrame({
        'date': plot_df['date'].unique(),
    }).sort_values('date')
    daily['Flow'] = daily['date'].map(flow_hours).fillna(0)
    daily['Shallow'] = daily['date'].map(shallow_hours).fillna(0)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=daily['date'],
        y=daily['Flow'],
        name='Flow (≥90min)',
        marker_color='#1565C0'
    ))
    
    fig.add_trace(go.Bar(
        x=daily['date'],
        y=daily['Shallow'],
        name='Shallow',
        marker_color='#B0BEC5'
    ))
    
    fig.update_layout(
        title='Flow vs Shallow Time',
        xaxis_title='Date',
        yaxis_title='Hours',
        barmode='stack',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig


import numpy as np


def create_spiral_plot(df: pd.DataFrame) -> go.Figure:
    """
    Create a spiral plot showing circadian rhythm consistency and drift.
    θ (angle) = hour of day (0-24 mapped to 0-360°)
    r (radius) = date (inner = earliest, outer = latest)
    Color = activity type
    """
    plot_df = ensure_datetime(df)
    
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Calculate polar coordinates
    min_date = plot_df['datetime_from'].min()
    max_date = plot_df['datetime_from'].max()
    date_range = (max_date - min_date).days + 1
    
    # Map hour to angle (0-24h -> 0-360°, but in radians)
    plot_df['hour'] = plot_df['datetime_from'].dt.hour + plot_df['datetime_from'].dt.minute / 60
    plot_df['theta'] = (plot_df['hour'] / 24) * 2 * np.pi  # 0-2π radians
    
    # Map date to radius (earliest = inner, latest = outer)
    plot_df['days_from_start'] = (plot_df['datetime_from'] - min_date).dt.days
    plot_df['r'] = 1 + (plot_df['days_from_start'] / max(date_range, 1)) * 9  # r from 1 to 10
    
    fig = go.Figure()
    
    # Add traces per activity type
    for activity in plot_df['activity_type'].unique():
        activity_df = plot_df[plot_df['activity_type'] == activity]
        
        fig.add_trace(go.Scatterpolar(
            r=activity_df['r'],
            theta=activity_df['theta'] * (180 / np.pi),  # Convert to degrees for plotly
            mode='markers',
            name=activity,
            marker=dict(
                color=get_color(activity),
                size=6 + activity_df['duration_hours'].clip(upper=3) * 2,  # Size by duration
                opacity=0.7
            ),
            hovertemplate=(
                f'{activity}<br>'
                'Date: %{customdata[0]}<br>'
                'Time: %{customdata[1]}<br>'
                'Duration: %{customdata[2]:.1f}h<extra></extra>'
            ),
            customdata=list(zip(
                activity_df['datetime_from'].dt.strftime('%Y-%m-%d'),
                activity_df['datetime_from'].dt.strftime('%H:%M'),
                activity_df['duration_hours']
            ))
        ))
    
    # Add hour labels on the polar axis
    fig.update_layout(
        title='Activity Spiral (Circadian Consistency)',
        polar=dict(
            radialaxis=dict(
                visible=False,
                range=[0, 11]
            ),
            angularaxis=dict(
                tickmode='array',
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00'],
                direction='clockwise',
                rotation=90  # Start at top (midnight)
            )
        ),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5)
    )
    
    return fig


def create_chord_diagram(df: pd.DataFrame) -> go.Figure:
    """
    Create a chord diagram (Sankey) showing activity transitions.
    Shows which activities follow which, revealing context-switching patterns.
    """
    plot_df = ensure_datetime(df).sort_values('datetime_from').reset_index(drop=True)
    
    if len(plot_df) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data for transitions", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Calculate transitions (current activity -> next activity)
    plot_df['next_activity'] = plot_df['activity_type'].shift(-1)
    transitions = plot_df.dropna(subset=['next_activity'])
    
    # Filter out same-activity transitions (continuation)
    transitions = transitions[transitions['activity_type'] != transitions['next_activity']]
    
    if transitions.empty:
        fig = go.Figure()
        fig.add_annotation(text="No transitions found", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Count transitions
    transition_counts = transitions.groupby(['activity_type', 'next_activity']).size().reset_index(name='count')
    
    # Create node list (unique activities)
    activities = list(set(transition_counts['activity_type'].tolist() + 
                         transition_counts['next_activity'].tolist()))
    activity_to_idx = {act: idx for idx, act in enumerate(activities)}
    
    # Create links
    sources = [activity_to_idx[act] for act in transition_counts['activity_type']]
    targets = [activity_to_idx[act] for act in transition_counts['next_activity']]
    values = transition_counts['count'].tolist()
    
    # Node colors
    node_colors = [get_color(act) for act in activities]
    
    # Link colors (semi-transparent source color)
    link_colors = [f"rgba{tuple(list(int(get_color(transition_counts.iloc[i]['activity_type']).lstrip('#')[j:j+2], 16) for j in (0, 2, 4)) + [0.4])}" 
                   for i in range(len(sources))]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=activities,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors
        )
    )])
    
    fig.update_layout(
        title='Activity Transitions (Context Switching)',
        font=dict(size=10)
    )
    
    return fig


def create_violin_plot(df: pd.DataFrame) -> go.Figure:
    """
    Create a violin plot showing session duration distribution per activity type.
    Reveals the depth and variability of focus sessions.
    """
    plot_df = df.copy()
    
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Get top activities by total duration
    top_activities = plot_df.groupby('activity_type')['duration_hours'].sum().nlargest(8).index.tolist()
    plot_df = plot_df[plot_df['activity_type'].isin(top_activities)]
    
    fig = go.Figure()
    
    for activity in top_activities:
        activity_data = plot_df[plot_df['activity_type'] == activity]['duration_hours']
        
        fig.add_trace(go.Violin(
            y=activity_data,
            name=activity,
            box_visible=True,
            meanline_visible=True,
            fillcolor=get_color(activity),
            line_color='black',
            opacity=0.7,
            points='outliers'
        ))
    
    # Add 90min flow threshold line
    fig.add_hline(y=1.5, line_dash="dash", line_color="#1565C0", line_width=1,
                  annotation_text="Flow threshold (90min)", 
                  annotation_position="top right")
    
    fig.update_layout(
        title='Session Duration Distribution by Activity',
        yaxis_title='Duration (hours)',
        xaxis_title='Activity Type',
        showlegend=False,
        violinmode='group'
    )
    
    return fig


def create_streamgraph(df: pd.DataFrame) -> go.Figure:
    """
    Create a streamgraph showing evolution of activity priorities over time.
    Stacked area chart with center baseline for symmetric visualization.
    """
    plot_df = ensure_datetime(df)
    
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    plot_df['date'] = plot_df['datetime_from'].dt.date
    
    # Aggregate by date and activity
    daily = plot_df.groupby(['date', 'activity_type'])['duration_hours'].sum().reset_index()
    
    # Pivot to get activities as columns
    pivot = daily.pivot(index='date', columns='activity_type', values='duration_hours').fillna(0)
    pivot = pivot.sort_index()
    
    fig = go.Figure()
    
    # Add traces for each activity (stacked area)
    for activity in pivot.columns:
        fig.add_trace(go.Scatter(
            x=pivot.index,
            y=pivot[activity],
            name=activity,
            mode='lines',
            stackgroup='one',
            groupnorm='',  # No normalization
            fillcolor=get_color(activity),
            line=dict(width=0.5, color=get_color(activity)),
            hovertemplate=f'{activity}<br>Date: %{{x}}<br>Hours: %{{y:.1f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Activity Evolution (Streamgraph)',
        xaxis_title='Date',
        yaxis_title='Hours',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    
    return fig


def create_rose_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a rose chart (radial bar chart) showing the "fingerprint" of an average day.
    24 wedges representing hours, colored by dominant activity.
    """
    plot_df = ensure_datetime(df)
    
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    plot_df['hour'] = plot_df['datetime_from'].dt.hour
    num_days = plot_df['datetime_from'].dt.date.nunique()
    if num_days == 0:
        num_days = 1
    
    # Aggregate by hour and activity
    hourly = plot_df.groupby(['hour', 'activity_type'])['duration_hours'].sum().reset_index()
    hourly['avg_hours'] = hourly['duration_hours'] / num_days
    
    # Get dominant activity per hour
    hourly_total = hourly.groupby('hour')['avg_hours'].sum().reset_index()
    hourly_total.columns = ['hour', 'total_hours']
    
    # Find dominant activity per hour
    dominant = hourly.loc[hourly.groupby('hour')['avg_hours'].idxmax()][['hour', 'activity_type']]
    hourly_total = hourly_total.merge(dominant, on='hour')
    
    # Ensure all 24 hours present
    all_hours = pd.DataFrame({'hour': range(24)})
    hourly_total = all_hours.merge(hourly_total, on='hour', how='left').fillna({'total_hours': 0, 'activity_type': 'Other'})
    
    # Convert hour to angle (0h = top, clockwise)
    hourly_total['theta'] = hourly_total['hour'] * 15  # 360/24 = 15 degrees per hour
    
    fig = go.Figure()
    
    fig.add_trace(go.Barpolar(
        r=hourly_total['total_hours'],
        theta=hourly_total['theta'],
        width=14,  # Slightly less than 15 for gaps
        marker_color=[get_color(act) for act in hourly_total['activity_type']],
        marker_line_color='white',
        marker_line_width=1,
        hovertemplate='Hour %{customdata[0]}:00<br>%{customdata[1]}<br>%{r:.2f}h avg<extra></extra>',
        customdata=list(zip(hourly_total['hour'], hourly_total['activity_type']))
    ))
    
    fig.update_layout(
        title='Daily Fingerprint (Rose Chart)',
        polar=dict(
            radialaxis=dict(
                visible=True,
                showticklabels=True,
                ticksuffix='h'
            ),
            angularaxis=dict(
                tickmode='array',
                tickvals=list(range(0, 360, 45)),
                ticktext=['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00'],
                direction='clockwise',
                rotation=90
            )
        ),
        showlegend=False
    )
    
    return fig


def create_barcode_plot(df: pd.DataFrame) -> go.Figure:
    """
    Create a barcode/rug plot showing density of session starts.
    Visualizes fragmented vs consolidated work patterns.
    """
    plot_df = ensure_datetime(df)
    
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Get session start times
    plot_df['date'] = plot_df['datetime_from'].dt.date
    plot_df['hour'] = plot_df['datetime_from'].dt.hour + plot_df['datetime_from'].dt.minute / 60
    
    # Last 14 days
    unique_dates = sorted(plot_df['date'].unique())[-14:]
    plot_df = plot_df[plot_df['date'].isin(unique_dates)]
    
    fig = go.Figure()
    
    for i, date in enumerate(unique_dates):
        day_data = plot_df[plot_df['date'] == date]
        
        # Add rug marks for each session start
        for _, row in day_data.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['hour'], row['hour']],
                y=[i - 0.3, i + 0.3],
                mode='lines',
                line=dict(color=get_color(row['activity_type']), width=2),
                showlegend=False,
                hovertemplate=f"{row['activity_type']}<br>{row['datetime_from'].strftime('%H:%M')}<extra></extra>"
            ))
    
    fig.update_layout(
        title='Session Density (Barcode Plot)',
        xaxis=dict(
            title='Hour of Day',
            range=[0, 24],
            tickmode='linear',
            tick0=0,
            dtick=4
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(unique_dates))),
            ticktext=[str(d) for d in unique_dates],
            title=''
        ),
        height=max(300, len(unique_dates) * 25)
    )
    
    return fig


# Energy weights for activities (positive = charging, negative = draining)
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


def get_energy_weight(activity_type: str) -> float:
    """Get energy weight for an activity, handling 'Activity (Category)' format."""
    # Direct match
    if activity_type in ENERGY_WEIGHTS:
        return ENERGY_WEIGHTS[activity_type]
    
    # Extract activity name from "Activity (Category)" format
    if " (" in activity_type:
        activity_name = activity_type.split(" (")[0]
        if activity_name in ENERGY_WEIGHTS:
            return ENERGY_WEIGHTS[activity_name]
    
    return 0


def create_energy_balance_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create an energy balance chart showing charging vs draining activities.
    Cumulative energy balance over time.
    """
    plot_df = ensure_datetime(df)
    
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    plot_df['date'] = plot_df['datetime_from'].dt.date
    
    # Calculate energy contribution per session
    plot_df['energy'] = plot_df.apply(
        lambda r: r['duration_hours'] * get_energy_weight(r['activity_type']), 
        axis=1
    )
    
    # Daily energy balance
    daily = plot_df.groupby('date').agg({
        'energy': 'sum',
        'duration_hours': 'sum'
    }).reset_index()
    daily = daily.sort_values('date')
    
    # Separate positive and negative for stacking
    daily['charging'] = daily['energy'].clip(lower=0)
    daily['draining'] = daily['energy'].clip(upper=0)
    daily['cumulative'] = daily['energy'].cumsum()
    
    fig = go.Figure()
    
    # Bar chart for daily balance
    fig.add_trace(go.Bar(
        x=daily['date'],
        y=daily['charging'],
        name='Charging',
        marker_color='#4CAF50'
    ))
    
    fig.add_trace(go.Bar(
        x=daily['date'],
        y=daily['draining'],
        name='Draining',
        marker_color='#F44336'
    ))
    
    # Cumulative line
    fig.add_trace(go.Scatter(
        x=daily['date'],
        y=daily['cumulative'],
        name='Cumulative',
        mode='lines',
        line=dict(color='#1565C0', width=2),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Energy Balance (Charging vs Draining)',
        xaxis_title='Date',
        yaxis=dict(title='Daily Energy', side='left'),
        yaxis2=dict(title='Cumulative', overlaying='y', side='right'),
        barmode='relative',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    
    return fig


# Productivity weights for Productivity Pulse calculation
PRODUCTIVITY_WEIGHTS = {
    'Coding': 100,
    'Work': 75,
    'Read': 50,
    'Yoga': 40,
    'Sport': 40,
    'Walking': 30,
    'Music': 25,
    'Internet': 25,
    'Housework': 20,
    'Entertainment': 0,
    'Other': 10,
}


def calculate_productivity_pulse(df: pd.DataFrame) -> float:
    """
    Calculate Productivity Pulse (RescueTime style).
    Weighted average: Pulse = Σ(Duration × Weight) / Σ(Duration)
    """
    if df.empty:
        return 0.0
    
    df = df.copy()
    df['weight'] = df['activity_type'].apply(lambda x: _map_weights(x, PRODUCTIVITY_WEIGHTS, 10))
    df['weighted_duration'] = df['duration_hours'] * df['weight']
    
    total_duration = df['duration_hours'].sum()
    if total_duration == 0:
        return 0.0
    
    return round(df['weighted_duration'].sum() / total_duration, 1)


def create_productivity_pulse_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a productivity pulse chart showing daily productivity scores over time.
    """
    plot_df = ensure_datetime(df)
    
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    plot_df['date'] = plot_df['datetime_from'].dt.date
    plot_df['weight'] = plot_df['activity_type'].apply(lambda x: _map_weights(x, PRODUCTIVITY_WEIGHTS, 10))
    plot_df['weighted_duration'] = plot_df['duration_hours'] * plot_df['weight']
    
    # Daily pulse calculation
    daily = plot_df.groupby('date').agg({
        'weighted_duration': 'sum',
        'duration_hours': 'sum'
    }).reset_index()
    daily['pulse'] = daily['weighted_duration'] / daily['duration_hours']
    daily = daily.sort_values('date')
    
    # 7-day moving average
    daily['pulse_ma'] = daily['pulse'].rolling(window=7, min_periods=1).mean()
    
    fig = go.Figure()
    
    # Daily pulse bars
    colors = ['#4CAF50' if p >= 50 else '#FF9800' if p >= 30 else '#F44336' for p in daily['pulse']]
    
    fig.add_trace(go.Bar(
        x=daily['date'],
        y=daily['pulse'],
        name='Daily Pulse',
        marker_color=colors,
        opacity=0.7
    ))
    
    # Moving average line
    fig.add_trace(go.Scatter(
        x=daily['date'],
        y=daily['pulse_ma'],
        name='7-day Avg',
        mode='lines',
        line=dict(color='#1565C0', width=2)
    ))
    
    # Add reference lines
    fig.add_hline(y=50, line_dash="dash", line_color="#4CAF50", line_width=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#FF9800", line_width=1)
    
    fig.update_layout(
        title='Productivity Pulse',
        xaxis_title='Date',
        yaxis=dict(title='Pulse Score', range=[0, 100]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    
    return fig


def create_network_graph(df: pd.DataFrame) -> go.Figure:
    """
    Create a network graph showing activity transition patterns.
    Nodes = activities, edges = transitions with thickness = frequency.
    """
    plot_df = ensure_datetime(df).sort_values('datetime_from').reset_index(drop=True)
    
    if len(plot_df) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Calculate transitions
    plot_df['next_activity'] = plot_df['activity_type'].shift(-1)
    transitions = plot_df.dropna(subset=['next_activity'])
    transitions = transitions[transitions['activity_type'] != transitions['next_activity']]
    
    if transitions.empty:
        fig = go.Figure()
        fig.add_annotation(text="No transitions", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Count transitions
    trans_counts = transitions.groupby(['activity_type', 'next_activity']).size().reset_index(name='count')
    
    # Get unique activities and their total time for node sizing
    activities = list(set(trans_counts['activity_type'].tolist() + trans_counts['next_activity'].tolist()))
    activity_hours = plot_df.groupby('activity_type')['duration_hours'].sum().to_dict()
    
    # Create circular layout
    n = len(activities)
    angles = [2 * np.pi * i / n for i in range(n)]
    node_x = [3 * np.cos(a) for a in angles]
    node_y = [3 * np.sin(a) for a in angles]
    pos = {act: (node_x[i], node_y[i]) for i, act in enumerate(activities)}
    
    fig = go.Figure()
    
    # Add edges (transitions)
    max_count = trans_counts['count'].max()
    for _, row in trans_counts.iterrows():
        x0, y0 = pos[row['activity_type']]
        x1, y1 = pos[row['next_activity']]
        width = 1 + (row['count'] / max_count) * 4
        
        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode='lines',
            line=dict(width=width, color='rgba(150,150,150,0.5)'),
            hoverinfo='text',
            text=f"{row['activity_type']} → {row['next_activity']}: {row['count']}x",
            showlegend=False
        ))
    
    # Add nodes
    node_sizes = [20 + min(activity_hours.get(act, 0), 50) for act in activities]
    node_colors = [get_color(act) for act in activities]
    
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=2, color='white')),
        text=activities,
        textposition='top center',
        hovertemplate='%{text}<br>%{customdata:.1f}h total<extra></extra>',
        customdata=[activity_hours.get(act, 0) for act in activities],
        showlegend=False
    ))
    
    fig.update_layout(
        title='Activity Network (Transitions)',
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='x'),
        hovermode='closest'
    )
    
    return fig


# Recovery categories (charging activities)
RECOVERY_CATEGORIES = ['Sport', 'Yoga', 'Walking', 'Entertainment', 'Read', 'Music']
# Draining categories
DRAIN_CATEGORIES = ['Work', 'Coding', 'Housework']


def create_recovery_ratio_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a recovery ratio chart showing balance between recovery and draining activities.
    Ratio = Recovery Hours / Drain Hours per day.
    Healthy ratio is typically 0.3-0.5 (30-50% recovery per drain hour).
    """
    plot_df = ensure_datetime(df)
    
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    plot_df['date'] = plot_df['datetime_from'].dt.date
    
    # Calculate recovery and drain hours per day (substring matching for "Activity (Category)" format)
    recovery_mask = plot_df['activity_type'].apply(lambda x: any(cat in x for cat in RECOVERY_CATEGORIES))
    drain_mask = plot_df['activity_type'].apply(lambda x: any(cat in x for cat in DRAIN_CATEGORIES))
    recovery = plot_df[recovery_mask].groupby('date')['duration_hours'].sum()
    drain = plot_df[drain_mask].groupby('date')['duration_hours'].sum()
    
    # Create daily dataframe
    all_dates = sorted(plot_df['date'].unique())
    daily = pd.DataFrame({'date': all_dates})
    daily['recovery'] = daily['date'].map(recovery).fillna(0)
    daily['drain'] = daily['date'].map(drain).fillna(0)
    daily['ratio'] = daily.apply(lambda r: r['recovery'] / r['drain'] if r['drain'] > 0 else 0, axis=1)
    daily['ratio_ma'] = daily['ratio'].rolling(window=7, min_periods=1).mean()
    
    fig = go.Figure()
    
    # Ratio bars with color coding
    colors = ['#4CAF50' if r >= 0.3 else '#FF9800' if r >= 0.15 else '#F44336' for r in daily['ratio']]
    
    fig.add_trace(go.Bar(
        x=daily['date'],
        y=daily['ratio'],
        name='Daily Ratio',
        marker_color=colors,
        opacity=0.7,
        hovertemplate='Date: %{x}<br>Ratio: %{y:.2f}<br>(%{customdata[0]:.1f}h recovery / %{customdata[1]:.1f}h drain)<extra></extra>',
        customdata=list(zip(daily['recovery'], daily['drain']))
    ))
    
    # 7-day moving average
    fig.add_trace(go.Scatter(
        x=daily['date'],
        y=daily['ratio_ma'],
        name='7-day Avg',
        mode='lines',
        line=dict(color='#1565C0', width=2)
    ))
    
    # Target zone
    fig.add_hline(y=0.3, line_dash="dash", line_color="#4CAF50", line_width=1,
                  annotation_text="Healthy (0.3)", annotation_position="right")
    fig.add_hline(y=0.15, line_dash="dash", line_color="#FF9800", line_width=1)
    
    fig.update_layout(
        title='Recovery Ratio (Recovery ÷ Drain)',
        xaxis_title='Date',
        yaxis_title='Ratio',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    
    return fig


def create_flow_streak_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a flow streak chart showing consecutive days with ≥2h of flow.
    Gamification element - visualizes momentum and consistency.
    """
    plot_df = ensure_datetime(df)
    
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    plot_df['date'] = plot_df['datetime_from'].dt.date
    
    # Calculate flow hours per day (sessions ≥90min in deep work categories)
    deep_work = filter_deep_work(plot_df)
    flow_sessions = deep_work[deep_work['duration_hours'] >= 1.5]
    daily_flow = flow_sessions.groupby('date')['duration_hours'].sum()
    
    # Create full date range
    all_dates = sorted(plot_df['date'].unique())
    daily = pd.DataFrame({'date': all_dates})
    daily['flow_hours'] = daily['date'].map(daily_flow).fillna(0)
    daily['hit_target'] = daily['flow_hours'] >= 2.0  # Target: 2h flow per day
    
    # Calculate streaks
    streaks = []
    current_streak = 0
    max_streak = 0
    
    for _, row in daily.iterrows():
        if row['hit_target']:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
        streaks.append(current_streak)
    
    daily['streak'] = streaks
    
    fig = go.Figure()
    
    # Flow hours bars
    colors = ['#1565C0' if hit else '#B0BEC5' for hit in daily['hit_target']]
    
    fig.add_trace(go.Bar(
        x=daily['date'],
        y=daily['flow_hours'],
        name='Flow Hours',
        marker_color=colors,
        hovertemplate='Date: %{x}<br>Flow: %{y:.1f}h<br>Streak: %{customdata}<extra></extra>',
        customdata=daily['streak']
    ))
    
    # Streak line on secondary axis
    fig.add_trace(go.Scatter(
        x=daily['date'],
        y=daily['streak'],
        name='Streak Days',
        mode='lines+markers',
        line=dict(color='#FF9800', width=2),
        marker=dict(size=4),
        yaxis='y2'
    ))
    
    # Target line
    fig.add_hline(y=2.0, line_dash="dash", line_color="#1565C0", line_width=1,
                  annotation_text="Target (2h)", annotation_position="right")
    
    fig.update_layout(
        title=f'Flow Streak (Max: {max_streak} days)',
        xaxis_title='Date',
        yaxis=dict(title='Flow Hours', side='left'),
        yaxis2=dict(title='Streak Days', overlaying='y', side='right', showgrid=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    
    return fig


def create_weekly_pattern_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a weekly pattern chart comparing productivity metrics by day of week.
    Shows which days are most productive for deep work and flow.
    """
    plot_df = ensure_datetime(df)
    
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    plot_df['date'] = plot_df['datetime_from'].dt.date
    plot_df['weekday'] = plot_df['datetime_from'].dt.dayofweek
    plot_df['weekday_name'] = plot_df['datetime_from'].dt.day_name()
    
    # Deep work hours
    deep_work = filter_deep_work(plot_df)
    deep_by_weekday = deep_work.groupby('weekday')['duration_hours'].mean()
    
    # Flow hours (≥90min sessions)
    flow_sessions = deep_work[deep_work['duration_hours'] >= 1.5]
    flow_by_weekday = flow_sessions.groupby('weekday')['duration_hours'].mean()
    
    # Count days to normalize
    days_count = plot_df.groupby('weekday')['date'].nunique()
    
    # Create summary
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    summary = pd.DataFrame({
        'weekday': range(7),
        'weekday_name': weekdays,
        'deep_work': [deep_by_weekday.get(i, 0) for i in range(7)],
        'flow': [flow_by_weekday.get(i, 0) for i in range(7)],
        'sample_days': [days_count.get(i, 0) for i in range(7)]
    })
    
    fig = go.Figure()
    
    # Deep work bars
    fig.add_trace(go.Bar(
        x=summary['weekday_name'],
        y=summary['deep_work'],
        name='Avg Deep Work',
        marker_color='#2196F3',
        opacity=0.7
    ))
    
    # Flow bars
    fig.add_trace(go.Bar(
        x=summary['weekday_name'],
        y=summary['flow'],
        name='Avg Flow',
        marker_color='#1565C0'
    ))
    
    # Find best day
    best_day = summary.loc[summary['deep_work'].idxmax(), 'weekday_name'] if not summary.empty else 'N/A'
    
    fig.update_layout(
        title=f'Weekly Pattern (Best: {best_day})',
        xaxis_title='Day of Week',
        yaxis_title='Average Hours',
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    
    return fig


def create_burnout_risk_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a burnout risk gauge/indicator based on:
    - Energy balance trend (negative = risk)
    - Recovery ratio (low = risk)
    - Consecutive high-drain days
    """
    plot_df = ensure_datetime(df)
    
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    plot_df['date'] = plot_df['datetime_from'].dt.date
    
    # Calculate daily energy
    plot_df['energy'] = plot_df.apply(
        lambda r: r['duration_hours'] * get_energy_weight(r['activity_type']), 
        axis=1
    )
    daily_energy = plot_df.groupby('date')['energy'].sum()
    
    # Calculate recovery ratio (use substring matching for categories)
    recovery_mask = plot_df['activity_type'].apply(lambda x: any(cat in x for cat in RECOVERY_CATEGORIES))
    drain_mask = plot_df['activity_type'].apply(lambda x: any(cat in x for cat in DRAIN_CATEGORIES))
    recovery = plot_df[recovery_mask].groupby('date')['duration_hours'].sum()
    drain = plot_df[drain_mask].groupby('date')['duration_hours'].sum()
    
    # Last 14 days analysis
    recent_dates = sorted(plot_df['date'].unique())[-14:]
    
    # Metrics for risk calculation
    recent_energy = [daily_energy.get(d, 0) for d in recent_dates]
    avg_energy = np.mean(recent_energy) if recent_energy else 0
    energy_trend = np.polyfit(range(len(recent_energy)), recent_energy, 1)[0] if len(recent_energy) > 1 else 0
    
    recent_recovery = [recovery.get(d, 0) for d in recent_dates]
    recent_drain = [drain.get(d, 0) for d in recent_dates]
    avg_ratio = np.mean([r/d if d > 0 else 0 for r, d in zip(recent_recovery, recent_drain)])
    
    # Count consecutive negative energy days
    consecutive_negative = 0
    for e in reversed(recent_energy):
        if e < 0:
            consecutive_negative += 1
        else:
            break
    
    # Calculate risk score (0-100)
    risk_score = 0
    
    # Energy trend component (0-35 points)
    if energy_trend < -5:
        risk_score += 35
    elif energy_trend < 0:
        risk_score += 20
    elif energy_trend < 5:
        risk_score += 10
    
    # Average energy component (0-30 points)
    if avg_energy < -20:
        risk_score += 30
    elif avg_energy < 0:
        risk_score += 20
    elif avg_energy < 20:
        risk_score += 10
    
    # Recovery ratio component (0-20 points)
    if avg_ratio < 0.15:
        risk_score += 20
    elif avg_ratio < 0.25:
        risk_score += 12
    elif avg_ratio < 0.35:
        risk_score += 5
    
    # Consecutive negative days (0-15 points)
    risk_score += min(consecutive_negative * 3, 15)
    
    # Determine risk level
    if risk_score >= 60:
        risk_level = "HIGH"
        color = "#F44336"
    elif risk_score >= 35:
        risk_level = "MEDIUM"
        color = "#FF9800"
    else:
        risk_level = "LOW"
        color = "#4CAF50"
    
    # Create gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        title={'text': f"Burnout Risk: {risk_level}", 'font': {'size': 14}},
        delta={'reference': 35, 'increasing': {'color': "#F44336"}, 'decreasing': {'color': "#4CAF50"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#ddd",
            'steps': [
                {'range': [0, 35], 'color': '#E8F5E9'},
                {'range': [35, 60], 'color': '#FFF3E0'},
                {'range': [60, 100], 'color': '#FFEBEE'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 2},
                'thickness': 0.75,
                'value': risk_score
            }
        }
    ))
    
    # Add breakdown text
    fig.add_annotation(
        x=0.5, y=-0.15,
        text=f"Energy Trend: {energy_trend:.1f} | Avg Ratio: {avg_ratio:.2f} | Negative Days: {consecutive_negative}",
        showarrow=False,
        font=dict(size=10, color="#666"),
        xref="paper", yref="paper"
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=40)
    )
    
    return fig


def create_peak_hours_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a peak hours heatmap showing when flow states are most likely.
    Combines hour-of-day with day-of-week for optimal scheduling.
    """
    plot_df = ensure_datetime(df)
    
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Get deep work sessions
    deep_work = filter_deep_work(plot_df)
    deep_work['hour'] = deep_work['datetime_from'].dt.hour
    deep_work['weekday'] = deep_work['datetime_from'].dt.dayofweek
    deep_work['is_flow'] = deep_work['duration_hours'] >= 1.5
    
    # Calculate flow probability per hour × weekday
    flow_matrix = np.zeros((7, 24))
    count_matrix = np.zeros((7, 24))
    
    for _, row in deep_work.iterrows():
        weekday = row['weekday']
        hour = row['hour']
        count_matrix[weekday, hour] += 1
        if row['is_flow']:
            flow_matrix[weekday, hour] += 1
    
    # Calculate probability (avoid division by zero)
    prob_matrix = np.divide(flow_matrix, count_matrix, 
                           out=np.zeros_like(flow_matrix), 
                           where=count_matrix != 0) * 100
    
    weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    hours = [f'{h:02d}:00' for h in range(24)]
    
    fig = go.Figure(data=go.Heatmap(
        z=prob_matrix,
        x=hours,
        y=weekdays,
        colorscale='Blues',
        hovertemplate='%{y} %{x}<br>Flow probability: %{z:.0f}%<br>Sessions: %{customdata}<extra></extra>',
        customdata=count_matrix
    ))
    
    # Find peak hour
    max_idx = np.unravel_index(np.argmax(prob_matrix), prob_matrix.shape)
    peak_day = weekdays[max_idx[0]]
    peak_hour = hours[max_idx[1]]
    
    fig.update_layout(
        title=f'Peak Flow Hours (Best: {peak_day} {peak_hour})',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        xaxis=dict(tickmode='array', tickvals=list(range(0, 24, 2)), ticktext=[f'{h:02d}' for h in range(0, 24, 2)])
    )
    
    return fig
