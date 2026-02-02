"""
Integrations subpackage for external time tracking services.
"""

from .toggl import (
    sync_entries as toggl_sync,
    get_current_entry as toggl_current,
    start_timer as toggl_start,
    stop_timer as toggl_stop,
    is_configured as toggl_configured,
    test_connection as toggl_test,
)

from .clockify import (
    sync_entries as clockify_sync,
    is_configured as clockify_configured,
    test_connection as clockify_test,
)

from .atimelogger import (
    sync_entries as atimelogger_sync,
    get_current_activity as atimelogger_current,
    is_configured as atimelogger_configured,
    test_connection as atimelogger_test,
    full_sync as atimelogger_full_sync,
)

from .apple_health import (
    import_health_export,
    parse_health_auto_export_payload,
)

from .background_sync import (
    BackgroundSyncService,
    SyncConfig,
    SyncStatus,
    get_sync_service,
    start_background_sync,
    stop_background_sync,
)

__all__ = [
    # Toggl
    "toggl_sync",
    "toggl_current",
    "toggl_start", 
    "toggl_stop",
    "toggl_configured",
    "toggl_test",
    # Clockify
    "clockify_sync",
    "clockify_configured",
    "clockify_test",
    # aTimeLogger
    "atimelogger_sync",
    "atimelogger_current",
    "atimelogger_configured",
    "atimelogger_test",
    "atimelogger_full_sync",
    # Apple Health
    "import_health_export",
    "parse_health_auto_export_payload",
    # Background Sync
    "BackgroundSyncService",
    "SyncConfig",
    "SyncStatus",
    "get_sync_service",
    "start_background_sync",
    "stop_background_sync",
]


async def get_configured_integrations() -> list[dict]:
    """Get list of configured integrations with status."""
    integrations = []
    
    # Toggl
    if toggl_configured():
        success, message = await toggl_test()
        integrations.append({
            "name": "Toggl Track",
            "id": "toggl",
            "configured": True,
            "connected": success,
            "status": message,
            "features": ["sync", "real-time", "start/stop"],
        })
    else:
        integrations.append({
            "name": "Toggl Track",
            "id": "toggl",
            "configured": False,
            "connected": False,
            "status": "Set TOGGL_API_TOKEN to enable",
            "features": ["sync", "real-time", "start/stop"],
        })
    
    # Clockify
    if clockify_configured():
        success, message = await clockify_test()
        integrations.append({
            "name": "Clockify",
            "id": "clockify",
            "configured": True,
            "connected": success,
            "status": message,
            "features": ["sync"],
        })
    else:
        integrations.append({
            "name": "Clockify",
            "id": "clockify",
            "configured": False,
            "connected": False,
            "status": "Set CLOCKIFY_API_KEY to enable",
            "features": ["sync"],
        })
    
    # Apple Health (always available via export)
    integrations.append({
        "name": "Apple Health",
        "id": "apple_health",
        "configured": True,
        "connected": True,
        "status": "Import via Health export.zip",
        "features": ["import"],
    })
    
    return integrations


async def sync_all() -> dict:
    """
    Sync data from all configured integrations.
    
    Returns:
        Dict with sync results per integration
    """
    results = {}
    
    if toggl_configured():
        try:
            entries = await toggl_sync(days=7)
            results["toggl"] = {"success": True, "count": len(entries), "entries": entries}
        except Exception as e:
            results["toggl"] = {"success": False, "error": str(e)}
    
    if clockify_configured():
        try:
            entries = await clockify_sync(days=7)
            results["clockify"] = {"success": True, "count": len(entries), "entries": entries}
        except Exception as e:
            results["clockify"] = {"success": False, "error": str(e)}
    
    return results
