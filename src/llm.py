import os
import json
import httpx
import logging
import pandas as pd
import traceback
from typing import Optional
from datetime import datetime, timedelta

# Import your local configuration and metrics
from .config import DEEP_WORK_CATEGORIES, FLOW_MIN_DURATION_HOURS
from .utils import matches_categories
from .metrics.advanced import (
    calculate_fragmentation_index,
    calculate_flow_index,
    calculate_sleep_regularity_index,
    get_circadian_profile,
)
from .correlations import get_correlation_summary

# --- Configuration ---

# Setup Logging (Best practice instead of print statements)
logger = logging.getLogger(__name__)

# API Configuration
API_KEY = os.environ.get("GLM_API_KEY", "")

# Z.ai International Endpoint
API_BASE_URL = "https://api.z.ai/api/paas/v4"

# Model Selection
# Use "glm-4.7" for maximum reasoning (Flagship)
# Use "glm-4.7-flash" for speed and lower cost
DEFAULT_MODEL = "glm-4.7-flash" 

ENERGY_WEIGHTS = {
    'Sport': 50, 'Yoga': 60, 'Walking': 40, 'Entertainment': 10,
    'Read': 20, 'Music': 15, 'Work': -40, 'Coding': -50,
    'Housework': -20, 'Internet': -10, 'Other': 0,
}

RECOVERY_CATEGORIES = {'Sport', 'Yoga', 'Walking', 'Entertainment', 'Read', 'Music'}
DRAIN_CATEGORIES = {'Work', 'Coding', 'Housework'}


def create_data_summary(df: pd.DataFrame) -> str:
    """Create a comprehensive data summary for the LLM context."""
    if df.empty:
        return "No data available."
    
    # Basic stats
    total_hours = df['duration_hours'].sum()
    total_days = df['datetime_from'].dt.date.nunique() or 1
    
    date_range = f"{df['datetime_from'].min().strftime('%Y-%m-%d')} to {df['datetime_from'].max().strftime('%Y-%m-%d')}"
    
    # Category breakdown
    category_summary = df.groupby('activity_type')['duration_hours'].agg(['sum', 'count', 'mean']).round(2)
    category_summary.columns = ['total_hours', 'session_count', 'avg_session_hours']
    category_summary = category_summary.sort_values('total_hours', ascending=False)
    
    # Deep work stats
    deep_work_df = df[df['activity_type'].apply(lambda x: matches_categories(x, DEEP_WORK_CATEGORIES))]
    deep_work_hours = deep_work_df['duration_hours'].sum()
    flow_sessions = deep_work_df[deep_work_df['duration_hours'] >= FLOW_MIN_DURATION_HOURS]
    flow_hours = flow_sessions['duration_hours'].sum()
    
    # Metrics
    flow_index = calculate_flow_index(df)
    fragmentation = calculate_fragmentation_index(df)
    sleep_stats = calculate_sleep_regularity_index(df)
    circadian = get_circadian_profile(df)
    
    # Energy balance
    recovery_hours = df[df['activity_type'].apply(lambda x: matches_categories(x, list(RECOVERY_CATEGORIES)))]['duration_hours'].sum()
    drain_hours = df[df['activity_type'].apply(lambda x: matches_categories(x, list(DRAIN_CATEGORIES)))]['duration_hours'].sum()
    recovery_ratio = recovery_hours / drain_hours if drain_hours > 0 else 0
    
    # Weekly pattern
    weekly_deep = deep_work_df.copy()
    if not weekly_deep.empty:
        weekly_deep['weekday'] = weekly_deep['datetime_from'].dt.day_name()
        weekly_summary = weekly_deep.groupby('weekday')['duration_hours'].sum().to_dict()
    else:
        weekly_summary = {}
    
    # Recent trend (last 7 days vs previous 7 days)
    max_date = df['datetime_from'].max()
    recent_start = max_date - timedelta(days=7)
    previous_start = recent_start - timedelta(days=7)
    
    recent = df[df['datetime_from'] >= recent_start]
    previous = df[(df['datetime_from'] >= previous_start) & (df['datetime_from'] < recent_start)]
    
    recent_dw = recent[recent['activity_type'].apply(lambda x: matches_categories(x, DEEP_WORK_CATEGORIES))]['duration_hours'].sum()
    previous_dw = previous[previous['activity_type'].apply(lambda x: matches_categories(x, DEEP_WORK_CATEGORIES))]['duration_hours'].sum()
    
    change_pct = ((recent_dw - previous_dw) / previous_dw * 100) if previous_dw > 0 else 0
    
    # Build summary
    summary = f"""
# Time Tracking Data Summary

## Overview
- Date Range: {date_range}
- Total Days: {total_days}
- Total Tracked Hours: {total_hours:.1f}h
- Average Daily Tracked: {total_hours/total_days:.1f}h

## Activity Breakdown
{category_summary.to_string()}

## Deep Work & Flow Analysis
- Deep Work Categories: {', '.join(DEEP_WORK_CATEGORIES)}
- Total Deep Work Hours: {deep_work_hours:.1f}h
- Flow Sessions (≥{FLOW_MIN_DURATION_HOURS}h): {len(flow_sessions)} sessions, {flow_hours:.1f}h total
- Flow Index: {flow_index}% (% of deep work in flow state)
- Deep Work Ratio: {(deep_work_hours/total_hours*100) if total_hours > 0 else 0:.1f}% of total time

## Fragmentation Index (sessions per hour, lower = more consolidated)
{json.dumps(fragmentation, indent=2)}

## Sleep Patterns (inferred from activity gaps)
- Average Sleep: {sleep_stats.get('avg_sleep_hours', 0):.1f}h
- Sleep Regularity Index: {sleep_stats.get('sri', 0):.2f} (lower = more consistent)
- Sleep Start Std Dev: {sleep_stats.get('sleep_start_std', 0):.2f}h
- Wake Time Std Dev: {sleep_stats.get('wake_time_std', 0):.2f}h

## Energy Balance
- Recovery Activities (Sport, Yoga, Walking, etc.): {recovery_hours:.1f}h
- Draining Activities (Work, Coding, Housework): {drain_hours:.1f}h
- Recovery Ratio: {recovery_ratio:.2f} (healthy target: 0.3-0.5)

## Peak Hours by Activity
{json.dumps(circadian, indent=2)}

## Weekly Pattern (Deep Work Hours)
{json.dumps(weekly_summary, indent=2)}

## Recent Trend (Last 7 Days vs Previous 7 Days)
- Recent Deep Work: {recent_dw:.1f}h
- Previous Deep Work: {previous_dw:.1f}h
- Change: {change_pct:.1f}%

## Correlation Analysis (What affects your performance?)
{_format_correlations(df)}
"""
    return summary


def _format_correlations(df: pd.DataFrame) -> str:
    """Format correlation data for the LLM."""
    corr_summary = get_correlation_summary(df)
    
    if not corr_summary.get("has_data"):
        return "Onvoldoende data voor correlatie-analyse (minimaal 14 dagen nodig)"
    
    lines = []
    
    if corr_summary.get("positive_factors"):
        lines.append("Positieve invloed op deep work:")
        for factor in corr_summary["positive_factors"]:
            lag_text = f" (volgende dag)" if factor["lag"] > 0 else ""
            lines.append(f"  - {factor['activity']}: r={factor['correlation']:+.2f}{lag_text}")
    
    if corr_summary.get("negative_factors"):
        lines.append("Negatieve invloed op deep work:")
        for factor in corr_summary["negative_factors"]:
            lag_text = f" (volgende dag)" if factor["lag"] > 0 else ""
            lines.append(f"  - {factor['activity']}: r={factor['correlation']:+.2f}{lag_text}")
    
    if corr_summary.get("strongest"):
        lines.append(f"\nSterkste correlatie: {corr_summary['strongest']['interpretation']} (r={corr_summary['strongest']['correlation']:+.2f})")
    
    return "\n".join(lines) if lines else "Geen significante correlaties gevonden"


SYSTEM_PROMPT = """Je bent een ervaren productiviteitscoach en data-analist. Je analyseert tijdtrackingdata om diepgaande, persoonlijke inzichten te geven.

## Jouw Analyse Stijl
- **Data-gedreven**: Citeer altijd concrete cijfers uit de data
- **Eerlijk maar ondersteunend**: Benoem problemen direct, maar met een constructieve toon
- **Actiegericht**: Elk inzicht moet leiden tot een concrete actie
- **Holistisch**: Bekijk de balans tussen werk, herstel en slaap

## Kernconcepten
- **Flow State**: Ononderbroken deep work sessies ≥90 minuten. Dit is waar echte voortgang wordt geboekt.
- **Deep Work Ratio**: % tijd in cognitief veeleisende taken (target: 40-60%)
- **Fragmentation Index**: Sessies per uur. Lager = betere focusblokken. >2.0 = te veel context switching.
- **Recovery Ratio**: Verhouding herstel/belastende activiteiten. Gezond target: 0.3-0.5. >1.0 = mogelijk vermijdingsgedrag.
- **Sleep Regularity Index**: Consistentie in slaaptijden. Lager = consistenter = beter.

## Analyse Framework
Structureer je analyse ALTIJD als volgt:

### 📈 Positieve Trends
Wat gaat goed? Welke gewoontes zijn sterk? Waar is groei zichtbaar?

### ⚠️ Waarschuwingssignalen
Welke patronen zijn zorgelijk? Wees specifiek over risico's (burnout, procrastinatie, slaapgebrek).

### 📊 Categorie Deep-Dive
Analyseer de top 3-5 categorieën qua tijdsbesteding. Wat vertelt elk patroon?

### 🎯 Concrete Acties
Geef 3-5 specifieke, meetbare acties voor de komende week.

## Belangrijk
- Gebruik emoji's voor visuele structuur
- Wees bondig maar volledig - geen vage algemeenheden
- Als je een probleem signaleert, leg uit WAAROM het een probleem is
- Eindig altijd met een positieve, motiverende noot

Antwoord in het Nederlands."""


async def _call_glm_api(messages: list, api_key: str, model: str = DEFAULT_MODEL) -> str:
    """Internal helper to handle the raw API call with retry logic and better logging."""
    if not api_key:
        return "⚠️ No API key configured. Set GLM_API_KEY environment variable."

    url = f"{API_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept-Language": "en-US,en", # Force English response headers
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "stream": False,
        "max_tokens": 20000,
    }

    logger.debug(f"Calling GLM API: {url} | Model: {model}")

    try:
        # GLM-4.7 can take time to "think", so we use a generous timeout
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            
            if response.status_code != 200:
                error_msg = f"API Error {response.status_code}: {response.text}"
                logger.error(error_msg)
                return f"⚠️ {error_msg}"
            
            result = response.json()
            return result["choices"][0]["message"]["content"]

    except httpx.TimeoutException:
        logger.error("GLM API Timeout")
        return "⚠️ Error: Request timed out (Model took too long to respond)."
    except httpx.ConnectError as e:
        logger.error(f"GLM Connection Error: {e}")
        return f"⚠️ Connection Error: Could not reach {API_BASE_URL}"
    except Exception as e:
        logger.exception("Unexpected error in GLM API call")
        return f"⚠️ Error: {str(e)}"


async def _stream_glm_api(messages: list, api_key: str, model: str = DEFAULT_MODEL):
    """Streaming version of the GLM API call. Yields chunks of text as they arrive."""
    if not api_key:
        yield "⚠️ No API key configured. Set GLM_API_KEY environment variable."
        return

    url = f"{API_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept-Language": "en-US,en",
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "stream": True,
        "max_tokens": 20000,
    }

    logger.debug(f"Streaming GLM API: {url} | Model: {model}")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield f"⚠️ API Error {response.status_code}: {error_text.decode()}"
                    return
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix
                        if data == "[DONE]":
                            break
                        try:
                            import json
                            chunk = json.loads(data)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue

    except httpx.TimeoutException:
        logger.error("GLM API Timeout (streaming)")
        yield "⚠️ Error: Request timed out."
    except httpx.ConnectError as e:
        logger.error(f"GLM Connection Error: {e}")
        yield f"⚠️ Connection Error: Could not reach {API_BASE_URL}"
    except Exception as e:
        logger.exception("Unexpected error in GLM streaming")
        yield f"⚠️ Error: {str(e)}"


async def generate_ai_insights_streaming(df: pd.DataFrame, api_key: str = ""):
    """Generate AI-powered insights with streaming output."""
    key = api_key or API_KEY
    data_summary = create_data_summary(df)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""Analyseer mijn tijdtrackingdata en geef een complete analyse.

## Data Overzicht
{data_summary}

## Gevraagde Analyse

Geef een grondige analyse met de volgende structuur:

### 📈 Positieve Trends (Focus & Groei)
Benoem 3-5 sterke punten. Wat gaat goed? Waar is verbetering zichtbaar?

### ⚠️ Waarschuwingssignalen (Balans & Risico's)  
Identificeer 3-5 zorgelijke patronen. Wees specifiek over:
- Burnout risico's (overwerk, slaaptekort, geen herstel)
- Procrastinatie patronen (te veel "herstel", vermijdingsgedrag)
- Fragmentatie problemen (te veel context switching)

### 📊 Categorie Analyse
Analyseer elke significante categorie (>5% van totale tijd):
- Hoeveel tijd, sessies, gemiddelde sessieduur
- Fragmentatie score en wat dit betekent
- Trend (stijgend/dalend vs vorige periode)

### 🎯 Actieplan voor Komende Week
Geef 5 concrete, meetbare acties. Bijvoorbeeld:
- "Blokkeer 2 uur ononderbroken deep work op di/do ochtend"
- "Limiteer Sport sessies tot max 3x per week"

### 💡 Eén Gouden Tip
Eindig met één krachtige insight die het grootste verschil kan maken.

Wees specifiek, citeer cijfers, en vermijd vage algemeenheden."""}
    ]
    
    async for chunk in _stream_glm_api(messages, key):
        yield chunk


async def ask_about_data_streaming(df: pd.DataFrame, question: str, api_key: str = ""):
    """Ask a question about the data with streaming output."""
    key = api_key or API_KEY
    
    if not question.strip():
        yield "Stel een vraag over je data."
        return
    
    data_summary = create_data_summary(df)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""Hier is mijn tijdtrackingdata:

{data_summary}

Mijn vraag: {question}

Beantwoord op basis van de data. Wees specifiek en citeer relevante cijfers."""}
    ]
    
    async for chunk in _stream_glm_api(messages, key):
        yield chunk


async def generate_ai_insights(df: pd.DataFrame, api_key: str = "") -> str:
    """Generate AI-powered insights from the data."""
    key = api_key or API_KEY
    data_summary = create_data_summary(df)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""Analyseer mijn tijdtrackingdata en geef een complete analyse.

## Data Overzicht
{data_summary}

## Gevraagde Analyse

Geef een grondige analyse met de volgende structuur:

### 📈 Positieve Trends (Focus & Groei)
Benoem 3-5 sterke punten. Wat gaat goed? Waar is verbetering zichtbaar?

### ⚠️ Waarschuwingssignalen (Balans & Risico's)  
Identificeer 3-5 zorgelijke patronen. Wees specifiek over:
- Burnout risico's (overwerk, slaaptekort, geen herstel)
- Procrastinatie patronen (te veel "herstel", vermijdingsgedrag)
- Fragmentatie problemen (te veel context switching)

### 📊 Categorie Analyse
Analyseer elke significante categorie (>5% van totale tijd):
- Hoeveel tijd, sessies, gemiddelde sessieduur
- Fragmentatie score en wat dit betekent
- Trend (stijgend/dalend vs vorige periode)

### 🎯 Actieplan voor Komende Week
Geef 5 concrete, meetbare acties. Bijvoorbeeld:
- "Blokkeer 2 uur ononderbroken deep work op di/do ochtend"
- "Limiteer Sport sessies tot max 3x per week"

### 💡 Eén Gouden Tip
Eindig met één krachtige insight die het grootste verschil kan maken.

Wees specifiek, citeer cijfers, en vermijd vage algemeenheden."""}
    ]
    
    return await _call_glm_api(messages, key)


async def ask_about_data(df: pd.DataFrame, question: str, api_key: str = "") -> str:
    """Ask a question about the data."""
    key = api_key or API_KEY
    
    if not question.strip():
        return "Stel een vraag over je data."
    
    data_summary = create_data_summary(df)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""Hier is mijn tijdtrackingdata:

{data_summary}

Mijn vraag: {question}

Beantwoord op basis van de data. Wees specifiek en citeer relevante cijfers."""}
    ]
    
    return await _call_glm_api(messages, key)