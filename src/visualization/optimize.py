"""
Plotly chart optimization utilities.
Optimizes charts for faster loading and lower memory usage.
Especially useful for Raspberry Pi and low-resource environments.
"""

import plotly.graph_objects as go
from typing import Optional
import pandas as pd


# Optimization settings
OPTIMIZE_FOR_RASPBERRY_PI = True
MAX_DATA_POINTS = 500  # Limit data points for performance
DISABLE_ANIMATIONS = True
USE_WEBGL = True  # Use WebGL for scatter plots (faster rendering)


def optimize_figure(fig: go.Figure, for_raspberry_pi: bool = OPTIMIZE_FOR_RASPBERRY_PI) -> go.Figure:
    """
    Apply optimizations to a Plotly figure for better performance.
    
    Args:
        fig: Plotly figure to optimize
        for_raspberry_pi: Apply extra optimizations for low-resource environments
    
    Returns:
        Optimized figure
    """
    # Disable animations
    if DISABLE_ANIMATIONS:
        fig.update_layout(
            transition_duration=0,
        )
    
    # Reduce rendering complexity
    fig.update_layout(
        # Disable modebar for cleaner look and less JS
        modebar_remove=['lasso2d', 'select2d', 'autoScale2d', 'pan2d', 'zoom2d'],
        # Simpler hover
        hovermode='closest',
        # Disable drag mode by default
        dragmode=False,
    )
    
    if for_raspberry_pi:
        # Extra optimizations for Raspberry Pi
        fig.update_layout(
            # Reduce margins
            margin=dict(l=20, r=10, t=25, b=20),
            # Smaller fonts
            font=dict(size=9),
            # Disable legend if many items
            showlegend=fig.layout.showlegend if hasattr(fig.layout, 'showlegend') else False,
        )
        
        # Simplify axes
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='#eee',
            zeroline=False,
            showspikes=False,
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='#eee',
            zeroline=False,
            showspikes=False,
        )
    
    return fig


def downsample_dataframe(df: pd.DataFrame, max_points: int = MAX_DATA_POINTS) -> pd.DataFrame:
    """
    Downsample a DataFrame for better chart performance.
    
    Args:
        df: DataFrame to downsample
        max_points: Maximum number of points to keep
    
    Returns:
        Downsampled DataFrame
    """
    if len(df) <= max_points:
        return df
    
    # Sample evenly across the dataset
    step = len(df) // max_points
    return df.iloc[::step].copy()


def simplify_trace(trace: go.Scatter, max_points: int = MAX_DATA_POINTS) -> go.Scatter:
    """
    Simplify a scatter trace by reducing points.
    Uses LTTB (Largest Triangle Three Buckets) algorithm approximation.
    """
    if trace.x is None or len(trace.x) <= max_points:
        return trace
    
    # Simple downsampling (every nth point)
    step = len(trace.x) // max_points
    trace.x = trace.x[::step]
    trace.y = trace.y[::step]
    
    return trace


def get_minimal_config() -> dict:
    """
    Get minimal Plotly config for production/Pi deployment.
    
    Returns:
        Config dict for plotly figures
    """
    return {
        'displayModeBar': False,
        'staticPlot': False,  # Keep some interactivity
        'responsive': True,
        'scrollZoom': False,
        'doubleClick': False,
        'showTips': False,
        'displaylogo': False,
        'modeBarButtonsToRemove': [
            'pan2d', 'lasso2d', 'select2d', 'zoom2d', 'zoomIn2d', 'zoomOut2d',
            'autoScale2d', 'resetScale2d', 'hoverClosestCartesian',
            'hoverCompareCartesian', 'toggleSpikelines'
        ],
    }


def optimize_for_mobile(fig: go.Figure) -> go.Figure:
    """
    Optimize figure for mobile/small screens.
    """
    fig.update_layout(
        margin=dict(l=30, r=10, t=30, b=30),
        font=dict(size=10),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='center',
            x=0.5,
            font=dict(size=8),
        ),
    )
    return fig


def create_loading_placeholder() -> go.Figure:
    """
    Create a lightweight loading placeholder figure.
    """
    fig = go.Figure()
    fig.add_annotation(
        text="Loading...",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color='#888'),
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='#fafafa',
        paper_bgcolor='#fafafa',
        height=200,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig
