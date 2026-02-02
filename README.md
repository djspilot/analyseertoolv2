# Analyseertool - Time Analysis Dashboard

A brutalist-style time tracking analysis dashboard for personal productivity optimization. Analyzes time registration data (CSV exports) and provides insights about deep work, flow states, fragmentation, circadian rhythms, and more.

## Features

- **31 Visualizations**: Comprehensive charts including Spiral Plots, Chord Diagrams, Violin Plots, Streamgraphs, Network Graphs, and more
- **Deep Work Analysis**: Track and visualize deep work sessions (≥60 min) and flow states (≥90 min)
- **Flow Index**: Measure percentage of deep work in sessions ≥90 minutes
- **Fragmentation Index**: Identify fragmented work patterns
- **Sleep Regularity**: Analyze sleep/wake patterns inferred from activity gaps
- **Circadian Rhythms**: Understand your optimal work hours
- **Energy Balance**: Track charging vs draining activities
- **Burnout Risk**: Gauge based on energy trends and recovery ratios
- **Comparison Mode**: Compare current vs previous week/month metrics
- **Fullscreen Focus**: Double-click any chart for detailed view with explanations

## Tech Stack

- **Frontend & Backend**: [Reflex](https://reflex.dev) - Full-stack Python web framework
- **Visualization**: Plotly - Interactive charts
- **Data Processing**: Pandas - DataFrame operations
- **Database**: SQLite - Lightweight persistence
- **Styling**: Brutalist/minimalist aesthetic with JetBrains Mono

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/djspilot/analyseertoolv2.git
cd analyseertoolv2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the dashboard:
```bash
reflex run
```

4. Open http://localhost:3000 in your browser

## GitHub

This repo is ready to publish. See [docs/GITHUB.md](docs/GITHUB.md) for the quick push steps.

## Raspberry Pi Deployment

Use Docker (recommended) or native systemd.

### Docker (recommended)

```bash
./scripts/pi_docker_install.sh
```

Optional: run with a local reverse proxy for https://homepage.dev:

```bash
USE_PROXY=1 ./scripts/pi_docker_install.sh
```

Full instructions: [docs/DEPLOY_PI.md](docs/DEPLOY_PI.md)

### Native (systemd)

```bash
./deploy_pi.sh install
./deploy_pi.sh start
```

## Project Structure

```
analyseertoolv2/
├── app/
│   └── app.py              # Reflex dashboard (main entry point)
├── src/
│   ├── config.py           # Constants, thresholds, colors
│   ├── processor.py        # Re-exports from parsing/ and metrics/
│   ├── comparison.py       # Period comparison (DateRange, ComparisonResult)
│   ├── chart_config.py     # Chart registry and PreferencesManager
│   ├── parsing/            # CSV ingestion
│   │   ├── csv_parser.py   # parse_csv(), validate_dataframe()
│   │   └── year_inference.py # Reverse Chronological Year Inference
│   ├── metrics/            # Calculations
│   │   ├── basic.py        # calculate_metrics(), summaries
│   │   └── advanced.py     # DWR, flow_index, SRI, circadian
│   ├── visualization/      # Plotly chart factories
│   │   ├── charts.py       # All chart functions
│   │   └── utils/          # Helper functions
│   └── api/
│       ├── database.py     # SQLite CRUD with get_connection()
│       └── cache.py        # @cache_chart decorator
├── data/
│   ├── lifestyle.db        # SQLite database
│   └── preferences.json    # Chart visibility settings
├── docs/
│   └── GRAPHS.md           # Documentation of all 31 charts
├── rxconfig.py             # Reflex configuration
├── requirements.txt
└── flow.md                 # Methodology documentation
```

## CSV Format

Import a CSV file with the following columns:

| Column | Description | Format | Example |
|--------|-------------|--------|---------|
| `From` | Start time | `DD MMM HH:MM` | `24 Jan 08:29` |
| `To` | End time | `DD MMM HH:MM` | `24 Jan 11:00` |
| `Activity type` | Category name | Text | `Work`, `Coding`, `Sport` |
| `Duration` | Duration in hours | European decimal | `2,5` = 2.5 hours |
| `Comment` | Optional note | Free text | `Morning coding` |

**Notes:**
- Use `;` (semicolon) as the column separator
- Use `,` (comma) as the decimal separator
- Year is inferred automatically using Reverse Chronological Year Inference

## Chart Categories

### Flow & Deep Work (7 charts)
- Flow vs Shallow, Deep Work Trend, Session Distribution
- Flow Probability by Hour, Flow Calendar, Flow Streak, Violin Plot

### Time-based (4 charts)
- Daily Breakdown, Streamgraph, Gantt Timeline, Barcode Plot

### Rhythm (6 charts)
- Circadian Profile, Activity Heatmap, Spiral Plot
- Rose Chart, Sleep Pattern, Peak Hours

### Patterns (4 charts)
- Fragmentation, Chord Diagram, Network Graph, Weekly Pattern

### Wellness (4 charts)
- Energy Balance, Recovery Ratio, Burnout Risk, Productivity Pulse

See [docs/GRAPHS.md](docs/GRAPHS.md) for detailed documentation of all visualizations.

## Metrics Explained

### Deep Work Ratio (DWR)
Percentage of time spent in deep work categories (Work, Coding).

```
DWR = (Deep Work Hours / Total Hours) × 100%
```

### Flow Index
Percentage of deep work sessions that are ≥90 minutes long.

```
Flow Index = (Sessions ≥90min / Total Deep Work Sessions) × 100%
```

### Fragmentation Index
Number of sessions per hour. Lower = more consolidated work.

```
FI = Total Sessions / Total Hours
```

### Productivity Pulse
RescueTime-style weighted productivity score (0-100).

```
Pulse = Σ(Duration × Weight) / Σ(Duration)
```

### Recovery Ratio
Balance between charging and draining activities.

```
Recovery Ratio = Recovery Hours ÷ Drain Hours
Healthy range: 0.3 - 0.5
```

## Usage Tips

- **Upload CSV**: Click the upload button (top-left) to import your time data
- **Filter by Tab**: Use header tabs (All, Flow, Time, Rhythm, Patterns, Wellness)
- **Toggle Charts**: Click the grid icon to show/hide specific charts
- **Fullscreen**: Double-click any chart for detailed view with explanations
- **Time Range**: Use the slider to filter data by date range
- **Comparison Mode**: Toggle to compare current vs previous period

## Development

### Code Quality

```bash
# Lint
flake8 src/

# Type check (optional)
mypy src/

# Run tests
pytest
```

### Adding New Charts

1. Create chart function in `src/visualization/charts.py`
2. Export in `src/visualization/__init__.py`
3. Add to state in `app/app.py`
4. Add to grid in `main_content()`
5. Document in `docs/GRAPHS.md`

## Configuration

Key settings in `src/config.py`:

```python
DEEP_WORK_MIN_DURATION_HOURS = 1.0   # 1 hour for deep work
FLOW_MIN_DURATION_HOURS = 1.5        # 90 minutes for flow state
DEEP_WORK_CATEGORIES = ['Work', 'Coding']
```

## License

This project is licensed under the MIT License.

---

**Built with**: Reflex, Plotly, Pandas, SQLite  
**Aesthetic**: Brutalist/Minimalist (JetBrains Mono, B&W)
