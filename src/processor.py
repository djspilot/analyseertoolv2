"""
Processor module - orchestrates CSV parsing and metric calculations.
This module provides backwards compatibility by re-exporting from subpackages.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from .logger import setup_logger

# Re-export parsing functions
from .parsing import infer_year, parse_csv, validate_dataframe

# Re-export metric functions
from .metrics import (
    calculate_metrics,
    calculate_circadian_metrics,
    calculate_consistency_metrics,
    get_daily_summary,
    get_category_summary,
    calculate_advanced_metrics,
    calculate_fragmentation_index,
    calculate_deep_work_ratio,
    calculate_context_switching_penalty,
    calculate_flow_index,
    infer_sleep_from_gaps,
    calculate_sleep_regularity_index,
    get_hourly_distribution,
    get_circadian_profile,
)

logger = setup_logger(__name__)


def ingest_csv_to_db(filepath: str | Path, reference_date: Optional[datetime] = None) -> int:
    """
    Complete pipeline: Read CSV, process, and store in database.

    Args:
        filepath: Path to the CSV file
        reference_date: Reference date for year inference

    Returns:
        Number of records inserted
    """
    from .database import init_db, insert_activities

    # Parse CSV
    df = parse_csv(filepath, reference_date)

    # Calculate metrics
    df = calculate_metrics(df)

    # Prepare for database (convert datetime to string)
    db_df = df.copy()
    db_df['datetime_from'] = db_df['datetime_from'].dt.strftime('%Y-%m-%d %H:%M:%S')
    db_df['datetime_to'] = db_df['datetime_to'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Only keep columns that match database schema
    db_df = db_df[['activity_type', 'duration_hours', 'datetime_from', 'datetime_to',
                   'comment', 'is_deep_work', 'fragmentation_risk']]

    # Initialize database and insert
    init_db()
    count = insert_activities(db_df)

    return count
