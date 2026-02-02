"""
Export functionality for generating PDF reports.
Creates professional reports with metrics, charts, and insights.
"""

import io
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd

from .config import DEEP_WORK_CATEGORIES, FLOW_MIN_DURATION_HOURS
from .utils import matches_categories
from .metrics.advanced import (
    calculate_fragmentation_index,
    calculate_flow_index,
    calculate_sleep_regularity_index,
)
from .goals import get_goals_summary
from .correlations import get_correlation_insights


def generate_report_html(
    df: pd.DataFrame,
    ai_analysis: str = "",
    period_label: str = "All Time",
) -> str:
    """
    Generate an HTML report that can be converted to PDF or displayed.
    
    Args:
        df: Activity DataFrame
        ai_analysis: Optional AI-generated analysis text
        period_label: Label for the time period
    
    Returns:
        HTML string of the report
    """
    if df.empty:
        return "<html><body><h1>No data available</h1></body></html>"
    
    # Calculate metrics
    total_hours = df['duration_hours'].sum()
    total_days = df['datetime_from'].dt.date.nunique()
    daily_avg = total_hours / total_days if total_days > 0 else 0
    
    deep_work_df = df[df['activity_type'].apply(lambda x: matches_categories(x, DEEP_WORK_CATEGORIES))]
    deep_work_hours = deep_work_df['duration_hours'].sum()
    deep_work_ratio = (deep_work_hours / total_hours * 100) if total_hours > 0 else 0
    
    flow_sessions = deep_work_df[deep_work_df['duration_hours'] >= FLOW_MIN_DURATION_HOURS]
    flow_hours = flow_sessions['duration_hours'].sum()
    flow_index = calculate_flow_index(df)
    
    sleep_stats = calculate_sleep_regularity_index(df)
    fragmentation = calculate_fragmentation_index(df)
    
    # Category breakdown
    category_summary = df.groupby('activity_type')['duration_hours'].agg(['sum', 'count']).round(1)
    category_summary.columns = ['Hours', 'Sessions']
    category_summary = category_summary.sort_values('Hours', ascending=False)
    
    # Goals progress
    goals_data = get_goals_summary(df)
    
    # Correlations
    correlations = get_correlation_insights(df)
    
    # Date range
    date_start = df['datetime_from'].min().strftime('%d %b %Y')
    date_end = df['datetime_from'].max().strftime('%d %b %Y')
    
    # Build HTML
    html = f"""
<!DOCTYPE html>
<html lang="nl">
<head>
    <meta charset="UTF-8">
    <title>Time Analysis Report - {period_label}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            line-height: 1.6;
            color: #000;
            background: #fff;
            padding: 40px;
            max-width: 800px;
            margin: 0 auto;
        }}
        
        h1 {{
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }}
        
        h2 {{
            font-size: 14px;
            font-weight: 700;
            margin-top: 32px;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 2px solid #000;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }}
        
        h3 {{
            font-size: 12px;
            font-weight: 600;
            margin-top: 20px;
            margin-bottom: 12px;
        }}
        
        .subtitle {{
            color: #666;
            font-size: 12px;
            margin-bottom: 24px;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }}
        
        .metric-card {{
            padding: 16px;
            background: #f5f5f5;
            border-left: 3px solid #000;
        }}
        
        .metric-value {{
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 4px;
        }}
        
        .metric-label {{
            font-size: 9px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 24px;
        }}
        
        th, td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        th {{
            font-weight: 600;
            text-transform: uppercase;
            font-size: 9px;
            letter-spacing: 0.1em;
            color: #666;
        }}
        
        .goal-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }}
        
        .goal-label {{
            flex: 1;
        }}
        
        .goal-progress {{
            width: 120px;
            height: 8px;
            background: #eee;
            border-radius: 4px;
            overflow: hidden;
            margin: 0 16px;
        }}
        
        .goal-bar {{
            height: 100%;
            background: #000;
            border-radius: 4px;
        }}
        
        .goal-value {{
            width: 80px;
            text-align: right;
            font-weight: 600;
        }}
        
        .correlation-item {{
            display: flex;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }}
        
        .correlation-icon {{
            font-size: 16px;
            margin-right: 12px;
        }}
        
        .correlation-text {{
            flex: 1;
        }}
        
        .correlation-value {{
            font-weight: 600;
        }}
        
        .ai-analysis {{
            background: #f9f9f9;
            padding: 20px;
            border-radius: 4px;
            border-left: 3px solid #000;
            white-space: pre-wrap;
        }}
        
        .ai-analysis h3, .ai-analysis h4 {{
            margin-top: 16px;
            margin-bottom: 8px;
        }}
        
        .ai-analysis ul {{
            margin-left: 20px;
            margin-bottom: 12px;
        }}
        
        .footer {{
            margin-top: 48px;
            padding-top: 16px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 9px;
            text-align: center;
        }}
        
        @media print {{
            body {{
                padding: 20px;
            }}
            .page-break {{
                page-break-before: always;
            }}
        }}
    </style>
</head>
<body>
    <h1>Time Analysis Report</h1>
    <p class="subtitle">{date_start} - {date_end} ({total_days} dagen)</p>
    
    <h2>📊 Key Metrics</h2>
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value">{total_hours:.0f}h</div>
            <div class="metric-label">Totaal Getracked</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{daily_avg:.1f}h</div>
            <div class="metric-label">Dagelijks Gemiddeld</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{deep_work_hours:.0f}h</div>
            <div class="metric-label">Deep Work</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{flow_index:.0f}%</div>
            <div class="metric-label">Flow Index</div>
        </div>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value">{len(flow_sessions)}</div>
            <div class="metric-label">Flow Sessies (≥90min)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{deep_work_ratio:.0f}%</div>
            <div class="metric-label">Deep Work Ratio</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{sleep_stats.get('avg_sleep_hours', 0):.1f}h</div>
            <div class="metric-label">Gem. Slaap</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{sleep_stats.get('sri', 0):.2f}</div>
            <div class="metric-label">Slaap Regulariteit</div>
        </div>
    </div>
    
    <h2>📈 Category Breakdown</h2>
    <table>
        <thead>
            <tr>
                <th>Activiteit</th>
                <th>Uren</th>
                <th>Sessies</th>
                <th>% van Totaal</th>
            </tr>
        </thead>
        <tbody>
            {''.join(f'''
            <tr>
                <td>{cat}</td>
                <td>{row["Hours"]:.1f}h</td>
                <td>{int(row["Sessions"])}</td>
                <td>{row["Hours"]/total_hours*100:.1f}%</td>
            </tr>
            ''' for cat, row in category_summary.iterrows())}
        </tbody>
    </table>
    
    <h2>🎯 Goals Progress</h2>
    <div style="margin-bottom: 16px;">
        <strong>Week Score: {goals_data.get('overall_score', 0):.0f}%</strong>
    </div>
    {''.join(f'''
    <div class="goal-row">
        <span class="goal-label">{g['icon']} {g['label']}</span>
        <div class="goal-progress">
            <div class="goal-bar" style="width: {min(100, g['percentage'])}%; background: {'#4CAF50' if g['status'] == 'success' else '#FF9800' if g['status'] == 'warning' else '#F44336'};"></div>
        </div>
        <span class="goal-value">{g['current']}/{g['target']} {g['unit'].split('/')[0]}</span>
    </div>
    ''' for g in goals_data.get('goals', []))}
    
    <h2>🔗 Correlaties</h2>
    <p style="color: #666; margin-bottom: 16px;">Wat beïnvloedt je prestaties?</p>
    {''.join(f'''
    <div class="correlation-item">
        <span class="correlation-icon">{c['icon']}</span>
        <span class="correlation-text">{c['title']}</span>
        <span class="correlation-value" style="color: {'#4CAF50' if c.get('direction') == 'positive' else '#F44336'};">r = {c['correlation']:+.2f}</span>
    </div>
    ''' for c in correlations[:6]) if correlations else '<p style="color: #666;">Onvoldoende data voor correlatie-analyse</p>'}
    
    {f'''
    <div class="page-break"></div>
    <h2>🤖 AI Analyse</h2>
    <div class="ai-analysis">{_markdown_to_html(ai_analysis)}</div>
    ''' if ai_analysis else ''}
    
    <div class="footer">
        Gegenereerd op {datetime.now().strftime('%d %b %Y om %H:%M')} • Analyseertool v2
    </div>
</body>
</html>
"""
    return html


def _markdown_to_html(text: str) -> str:
    """Simple markdown to HTML conversion."""
    if not text:
        return ""
    
    import re
    
    # Headers
    text = re.sub(r'^### (.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    
    # Lists
    text = re.sub(r'^\* (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'^- (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    
    # Wrap consecutive <li> items in <ul>
    text = re.sub(r'(<li>.*?</li>\n?)+', lambda m: f'<ul>{m.group(0)}</ul>', text)
    
    # Paragraphs (double newlines)
    text = re.sub(r'\n\n', '</p><p>', text)
    text = f'<p>{text}</p>'
    
    return text


def generate_pdf_bytes(df: pd.DataFrame, ai_analysis: str = "") -> bytes:
    """
    Generate a PDF report as bytes.
    
    Note: Requires weasyprint to be installed.
    Falls back to HTML if weasyprint is not available.
    """
    html = generate_report_html(df, ai_analysis)
    
    try:
        from weasyprint import HTML
        pdf_bytes = HTML(string=html).write_pdf()
        return pdf_bytes
    except ImportError:
        # WeasyPrint not installed, return HTML encoded as bytes
        return html.encode('utf-8')
    except Exception as e:
        # WeasyPrint failed (common on some systems)
        return html.encode('utf-8')


def get_report_download_link(df: pd.DataFrame, ai_analysis: str = "") -> str:
    """
    Generate a base64-encoded download link for the report.
    
    Returns:
        Data URI string for download
    """
    html = generate_report_html(df, ai_analysis)
    
    # Try PDF first
    try:
        from weasyprint import HTML
        pdf_bytes = HTML(string=html).write_pdf()
        b64 = base64.b64encode(pdf_bytes).decode()
        return f"data:application/pdf;base64,{b64}"
    except:
        # Fall back to HTML
        b64 = base64.b64encode(html.encode()).decode()
        return f"data:text/html;base64,{b64}"
