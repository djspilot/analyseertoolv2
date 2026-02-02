# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the Reflex dashboard (brutalist UI at localhost:3000)
reflex run

# Install dependencies
pip install -r requirements.txt
```

## Architecture

**Stack**: Python 3.10+ with Reflex frontend, Pandas for data processing, Plotly for visualizations, SQLite for persistence.

### Project Structure

```
analyseertoolv2/
├── app.py                    # Reflex dashboard (main entry point)
├── rxconfig.py               # Reflex configuration
├── requirements.txt
├── src/
│   ├── config.py             # Constants, thresholds, colors
│   ├── database.py           # SQLite CRUD with get_connection()
│   ├── logger.py             # Logging setup
│   ├── processor.py          # Re-exports from parsing/ and metrics/
│   ├── comparison.py         # Period comparison (DateRange, ComparisonResult)
│   ├── chart_config.py       # Chart registry and PreferencesManager
│   ├── cache.py              # @cache_chart decorator for Plotly caching
│   ├── visuals.py            # Plotly chart factories
│   ├── parsing/              # CSV ingestion
│   │   ├── csv_parser.py     # parse_csv(), validate_dataframe()
│   │   └── year_inference.py # Reverse Chronological Year Inference
│   ├── metrics/              # Calculations
│   │   ├── basic.py          # calculate_metrics(), summaries
│   │   └── advanced.py       # DWR, flow_index, SRI, circadian
│   └── charts/               # Chart utilities
│       └── utils.py          # ensure_datetime(), get_color(), filter_deep_work()
├── data/
│   ├── lifestyle.db          # SQLite database
│   └── preferences.json      # Chart visibility settings
└── archive/
    └── app_streamlit.py      # Deprecated Streamlit frontend
```

### Key Modules

- **processor.py** — Thin wrapper re-exporting from `parsing/` and `metrics/`. Use `ingest_csv_to_db()` for full pipeline.

- **parsing/year_inference.py** — **Reverse Chronological Year Inference**: dates without year (e.g., "24 Jan 08:29") get years based on whether month > reference month (previous year if so).

- **visuals.py** — Plotly chart factories using `@cache_chart` decorator. Uses `ensure_datetime()` and `filter_deep_work()` from `charts/utils.py`.

- **comparison.py** — Period comparison with `compare_periods()` returning `ComparisonResult`. Presets for "this week vs last week", etc.

## Advanced Metrics

- **Deep Work Ratio (DWR)**: % of time in Work/Coding sessions ≥1.5h
- **Flow Index**: % of deep work time in sessions ≥90min
- **Fragmentation Index**: sessions/hours per category (lower = more consolidated)
- **Sleep Regularity Index (SRI)**: std dev of sleep/wake times inferred from activity gaps

## Code Style

- Type hints for function arguments
- Docstrings for public functions
- `pd.DataFrame` for all data passing
- Context manager `get_connection()` for database access
- European decimal notation: CSV uses `,` as decimal separator
