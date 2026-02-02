"""
CSV parsing module for time tracking data.
Handles file reading, validation, and date parsing.
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from ..config import (
    MAX_DURATION_HOURS,
    MIN_DURATION_HOURS,
    FUTURE_DATE_TOLERANCE_HOURS,
)
from ..logger import setup_logger, log_dataframe_stats
from .year_inference import infer_year

logger = setup_logger(__name__)


def validate_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Validate and clean DataFrame.

    Args:
        df: DataFrame to validate

    Returns:
        Tuple of (cleaned_df, warnings)
    """
    warnings = []

    # Check for missing required columns
    required_columns = ['activity_type', 'duration_hours', 'datetime_from', 'datetime_to', 'comment']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Remove rows with NaN in critical columns
    before_count = len(df)
    df = df.dropna(subset=['activity_type', 'datetime_from', 'datetime_to', 'duration_hours'])
    dropped = before_count - len(df)
    if dropped > 0:
        warnings.append(f"Removed {dropped} rows with missing data")

    # Validate duration range
    invalid_duration = df[
        (df['duration_hours'] < MIN_DURATION_HOURS) |
        (df['duration_hours'] > MAX_DURATION_HOURS)
    ]
    if not invalid_duration.empty:
        warnings.append(
            f"Found {len(invalid_duration)} rows with invalid duration "
            f"(range: {MIN_DURATION_HOURS}-{MAX_DURATION_HOURS}h)"
        )
        df = df[(df['duration_hours'] >= MIN_DURATION_HOURS) &
                (df['duration_hours'] <= MAX_DURATION_HOURS)]

    # Check for future dates (with tolerance)
    now = datetime.now()
    future_dates = df[df['datetime_from'] > now + timedelta(hours=FUTURE_DATE_TOLERANCE_HOURS)]
    if not future_dates.empty:
        warnings.append(
            f"Found {len(future_dates)} rows with future dates "
            f"(tolerance: {FUTURE_DATE_TOLERANCE_HOURS}h)"
        )

    # Check for datetime_from > datetime_to
    invalid_time = df[df['datetime_from'] > df['datetime_to']]
    if not invalid_time.empty:
        warnings.append(
            f"Found {len(invalid_time)} rows where start time > end time"
        )
        df = df[df['datetime_from'] <= df['datetime_to']]

    # Check for duplicates
    duplicates = df.duplicated(subset=['activity_type', 'datetime_from', 'duration_hours'], keep=False)
    if duplicates.any():
        dup_count = duplicates.sum()
        warnings.append(f"Found {dup_count} potential duplicate rows")
        # Keep first occurrence
        df = df[~duplicates]

    return df, warnings


def parse_csv(filepath: str | Path, reference_date: Optional[datetime] = None) -> pd.DataFrame:
    """
    Parse a time tracking CSV file and infer years for dates.

    Args:
        filepath: Path to the CSV file
        reference_date: Reference date for year inference (defaults to now)

    Returns:
        DataFrame with parsed and complete datetime columns

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV format is invalid
    """
    if reference_date is None:
        reference_date = datetime.now()

    filepath = Path(filepath)

    # Check file exists
    if not filepath.exists():
        logger.error(f"CSV file not found: {filepath}")
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    logger.info(f"Parsing CSV file: {filepath.name}")

    try:
        # Read CSV with semicolon separator
        df = pd.read_csv(
            filepath,
            sep=';',
            encoding='utf-8',
            decimal=','
        )
        logger.debug(f"CSV loaded: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        raise ValueError(f"Invalid CSV file: {e}") from e

    # Check required columns
    required_cols = ['From', 'To', 'Activity type', 'Duration', 'Comment']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Skip summary rows at the end
    summary_start = df[df['Activity type'] == 'Activity type'].index
    if len(summary_start) > 0:
        df = df.iloc[:summary_start[0]]
        logger.debug(f"Removed {len(df) - summary_start[0]} summary rows")

    # Filter out rows where From doesn't contain a date pattern
    before_count = len(df)
    df = df[df['From'].str.contains(r'\d{2}\s+\w{3}\s+\d{2}:\d{2}', na=False, regex=True)]
    filtered = before_count - len(df)
    if filtered > 0:
        logger.debug(f"Filtered out {filtered} rows with invalid date format")

    # Parse datetime columns with year inference
    try:
        df['datetime_from'] = df['From'].apply(lambda x: infer_year(x, reference_date))
        df['datetime_to'] = df['To'].apply(lambda x: infer_year(x, reference_date))
        logger.info("Date parsing and year inference completed")
    except Exception as e:
        logger.error(f"Failed to parse dates: {e}")
        raise

    # Rename and clean columns
    df = df.rename(columns={
        'Activity type': 'activity_type',
        'Duration': 'duration_hours',
        'Comment': 'comment'
    })

    # Select and reorder columns
    df = df[['activity_type', 'duration_hours', 'datetime_from', 'datetime_to', 'comment']]

    # Convert duration to float
    if df['duration_hours'].dtype == 'object':
        df['duration_hours'] = df['duration_hours'].str.replace(',', '.').astype(float)

    # Fill NaN comments with empty string
    df['comment'] = df['comment'].fillna('')

    # Validate and clean data
    df, warnings = validate_dataframe(df)
    for warning in warnings:
        logger.warning(warning)

    log_dataframe_stats(df, logger, "Parsed CSV")

    return df
