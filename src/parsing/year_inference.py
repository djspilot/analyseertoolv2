"""
Year inference algorithm for dates without year information.
Uses Reverse Chronological Year Inference.
"""

from datetime import datetime

from ..config import CSV_DATE_FORMAT, CSV_ALT_DATE_FORMAT
from ..logger import setup_logger

logger = setup_logger(__name__)


def infer_year(date_str: str, reference_date: datetime) -> datetime:
    """
    Infer the year for a date string without year information.

    Uses Reverse Chronological Year Inference:
    - If the month in date_str is greater than the reference month,
      it must be from the previous year.

    Args:
        date_str: Date string like "24 Jan 08:29" or "31 Dec 15:00"
        reference_date: The reference date (typically today or export date)

    Returns:
        Complete datetime with inferred year

    Raises:
        ValueError: If date string cannot be parsed
    """
    date_str = date_str.strip()

    # Parse the date without year
    try:
        parsed = datetime.strptime(date_str, CSV_DATE_FORMAT)
    except ValueError:
        try:
            parsed = datetime.strptime(date_str, CSV_ALT_DATE_FORMAT)
        except ValueError as e:
            logger.error(f"Failed to parse date: '{date_str}'")
            raise ValueError(f"Cannot parse date: {date_str}") from e

    # Start with the reference year
    year = reference_date.year

    # If the parsed month is greater than reference month, it's from last year
    # e.g., if reference is Jan 24 and we see Dec 31, it's Dec of previous year
    if parsed.month > reference_date.month:
        year -= 1
    elif parsed.month == reference_date.month and parsed.day > reference_date.day:
        # Same month but day is in the future -> last year
        year -= 1

    result = parsed.replace(year=year)
    logger.debug(f"Inferred year for '{date_str}': {result}")
    return result
