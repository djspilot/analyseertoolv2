"""
Database module for SQLite operations.
Handles connection management and CRUD operations for activity data.
"""

import sqlite3
import shutil
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
import pandas as pd

from ..config import DB_PATH, BACKUP_PATH
from ..logger import setup_logger

logger = setup_logger(__name__)


@contextmanager
def get_connection():
    """Context manager for database connections."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Initialize the database with the activities table."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS activities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                activity_type TEXT NOT NULL,
                duration_hours REAL NOT NULL,
                datetime_from TEXT NOT NULL,
                datetime_to TEXT NOT NULL,
                comment TEXT,
                is_deep_work INTEGER DEFAULT 0,
                fragmentation_risk INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_datetime_from 
            ON activities(datetime_from)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_activity_type 
            ON activities(activity_type)
        """)
        conn.commit()


def insert_activities(df: pd.DataFrame) -> int:
    """
    Bulk insert activities from a DataFrame.
    
    Args:
        df: DataFrame with columns matching the activities table
        
    Returns:
        Number of rows inserted
    """
    with get_connection() as conn:
        # Clear existing data before import
        conn.execute("DELETE FROM activities")
        
        records = df.to_dict('records')
        conn.executemany("""
            INSERT INTO activities 
            (activity_type, duration_hours, datetime_from, datetime_to, comment, is_deep_work, fragmentation_risk)
            VALUES (:activity_type, :duration_hours, :datetime_from, :datetime_to, :comment, :is_deep_work, :fragmentation_risk)
        """, records)
        conn.commit()
        return len(records)


def get_all_activities() -> pd.DataFrame:
    """
    Retrieve all activities as a DataFrame.
    
    Returns:
        DataFrame with all activity records
    """
    with get_connection() as conn:
        df = pd.read_sql_query("""
            SELECT * FROM activities 
            ORDER BY datetime_from DESC
        """, conn)
        
        # Convert datetime strings to datetime objects
        if not df.empty:
            df['datetime_from'] = pd.to_datetime(df['datetime_from'])
            df['datetime_to'] = pd.to_datetime(df['datetime_to'])
        
        return df


def add_activity(
    activity: str,
    category: str,
    datetime_from: datetime,
    datetime_to: datetime,
    duration_hours: float,
    comment: str = None,
    is_deep_work: int = 0,
) -> int:
    """
    Add a single activity to the database.
    
    Args:
        activity: Activity name/type
        category: Category (e.g., "Werk", "Sport")
        datetime_from: Start time
        datetime_to: End time
        duration_hours: Duration in hours
        comment: Optional comment
        is_deep_work: Whether this is deep work (0 or 1)
        
    Returns:
        The ID of the inserted row
    """
    # Combine activity and category for activity_type field
    activity_type = f"{activity} ({category})" if category else activity
    
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO activities 
            (activity_type, duration_hours, datetime_from, datetime_to, comment, is_deep_work, fragmentation_risk)
            VALUES (?, ?, ?, ?, ?, ?, 0)
        """, (
            activity_type,
            duration_hours,
            datetime_from.isoformat() if isinstance(datetime_from, datetime) else str(datetime_from),
            datetime_to.isoformat() if isinstance(datetime_to, datetime) else str(datetime_to),
            comment,
            is_deep_work,
        ))
        conn.commit()
        return cursor.lastrowid


def update_activity(activity_id: int, **kwargs) -> bool:
    """
    Update a single activity record.
    
    Args:
        activity_id: The ID of the activity to update
        **kwargs: Fields to update (activity_type, duration_hours, etc.)
        
    Returns:
        True if update was successful
    """
    allowed_fields = {'activity_type', 'duration_hours', 'datetime_from', 
                      'datetime_to', 'comment', 'is_deep_work', 'fragmentation_risk'}
    
    # Filter to only allowed fields
    updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
    
    if not updates:
        return False
    
    # Build SET clause
    set_parts = [f"{k} = ?" for k in updates.keys()]
    set_clause = ", ".join(set_parts)
    values = list(updates.values()) + [activity_id]
    
    with get_connection() as conn:
        cursor = conn.execute(f"""
            UPDATE activities 
            SET {set_clause}, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, values)
        conn.commit()
        return cursor.rowcount > 0


def delete_all_activities() -> int:
    """Delete all activities and return count of deleted rows."""
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM activities")
        conn.commit()
        return cursor.rowcount


def get_activity_count() -> int:
    """Get the total number of activities in the database."""
    with get_connection() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM activities")
        return cursor.fetchone()[0]


def backup_database() -> str:
    """
    Create a backup of the database.
    
    Returns:
        Path to the backup file
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = BACKUP_PATH.parent / f"lifestyle_backup_{timestamp}.db"
        
        # Copy database file
        shutil.copy2(DB_PATH, backup_path)
        
        logger.info(f"Database backup created: {backup_path.name}")
        return str(backup_path)
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        raise


def restore_database(backup_path: str) -> bool:
    """
    Restore database from a backup file.
    
    Args:
        backup_path: Path to the backup file
        
    Returns:
        True if restore was successful
    """
    try:
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        # Close any existing connections and restore
        shutil.copy2(backup_path, DB_PATH)
        
        logger.info(f"Database restored from: {backup_path.name}")
        return True
    except Exception as e:
        logger.error(f"Failed to restore database: {e}")
        raise


def export_to_csv(filepath: str | Path) -> int:
    """
    Export all activities to a CSV file.
    
    Args:
        filepath: Path where CSV should be saved
        
    Returns:
        Number of records exported
    """
    try:
        df = get_all_activities()
        
        if df.empty:
            logger.warning("No data to export")
            return 0
        
        filepath = Path(filepath)
        
        # Convert datetime back to desired format
        export_df = df.copy()
        export_df['From'] = export_df['datetime_from'].dt.strftime('%d %b %H:%M')
        export_df['To'] = export_df['datetime_to'].dt.strftime('%d %b %H:%M')
        export_df['Duration'] = export_df['duration_hours'].astype(str)
        export_df['Comment'] = export_df['comment'].fillna('')
        
        # Reorder columns to match original format
        export_df = export_df[['activity_type', 'duration_hours', 'From', 'To', 'comment']]
        export_df.columns = ['Activity type', 'Duration', 'From', 'To', 'Comment']
        
        export_df.to_csv(filepath, sep=';', index=False, encoding='utf-8', decimal=',')
        
        logger.info(f"Exported {len(export_df)} records to {filepath.name}")
        return len(export_df)
    except Exception as e:
        logger.error(f"Failed to export CSV: {e}")
        raise
