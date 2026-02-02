"""
API package - Data access layer (database, cache).
"""

from .database import (
    get_connection,
    init_db,
    insert_activities,
    get_all_activities,
    update_activity,
    delete_all_activities,
    get_activity_count,
    backup_database,
    restore_database,
    export_to_csv,
)
from .cache import cache_chart, ENABLE_CACHE

__all__ = [
    'get_connection',
    'init_db',
    'insert_activities',
    'get_all_activities',
    'update_activity',
    'delete_all_activities',
    'get_activity_count',
    'backup_database',
    'restore_database',
    'export_to_csv',
    'cache_chart',
    'ENABLE_CACHE',
]