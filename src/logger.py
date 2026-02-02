"""
Logging configuration for the analyseertool.
Provides centralized logging setup.
"""

import logging
import sys
from pathlib import Path
from .config import LOG_LEVEL, LOG_FORMAT

# Log file path
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "analyseertool.log"


def setup_logger(name: str) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Name of the logger (typically __name__)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # File handler - detailed logs
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)
    
    # Console handler - INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)
    
    return logger


def log_dataframe_stats(df, logger: logging.Logger, name: str = "DataFrame"):
    """Log statistics about a DataFrame."""
    if df.empty:
        logger.warning(f"{name}: Empty DataFrame")
        return
    
    logger.info(
        f"{name}: {len(df)} rows, "
        f"{len(df.columns)} columns, "
        f"date range: {df['datetime_from'].min()} to {df['datetime_from'].max()}"
    )