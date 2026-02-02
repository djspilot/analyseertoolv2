"""
Background Sync Service
Automatically syncs data from integrations at regular intervals.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
import threading

from ..logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class SyncStatus:
    """Current sync status."""
    last_sync: Optional[datetime] = None
    last_success: bool = True
    last_error: Optional[str] = None
    entries_synced: int = 0
    is_syncing: bool = False
    next_sync: Optional[datetime] = None


@dataclass
class SyncConfig:
    """Sync configuration."""
    enabled: bool = True
    interval_minutes: int = 5  # Sync every 5 minutes (15 on Pi)
    sync_on_start: bool = True
    toggl_enabled: bool = True
    clockify_enabled: bool = True
    atimelogger_enabled: bool = True  # aTimeLogger support
    days_to_sync: int = 1  # Only sync last day for speed
    batch_size: int = 200  # Max entries per sync
    low_power_mode: bool = False  # Reduce CPU usage
    
    @classmethod
    def for_raspberry_pi(cls) -> 'SyncConfig':
        """Get Pi-optimized config."""
        return cls(
            interval_minutes=15,  # Less frequent
            sync_on_start=False,  # Don't block startup
            days_to_sync=1,
            batch_size=50,
            low_power_mode=True,
        )


class BackgroundSyncService:
    """
    Background service that automatically syncs data from integrations.
    
    Usage:
        sync_service = BackgroundSyncService()
        sync_service.start()
        
        # Later...
        sync_service.stop()
    """
    
    def __init__(
        self,
        config: Optional[SyncConfig] = None,
        on_sync_complete: Optional[Callable[[list[dict]], Any]] = None,
        on_status_change: Optional[Callable[[SyncStatus], Any]] = None,
    ):
        self.config = config or SyncConfig()
        self.on_sync_complete = on_sync_complete
        self.on_status_change = on_status_change
        self.status = SyncStatus()
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    def start(self):
        """Start the background sync service."""
        if self._running:
            logger.warning("Sync service already running")
            return
        
        self._running = True
        logger.info(f"Starting background sync (interval: {self.config.interval_minutes} min)")
        
        # Start in background thread to not block
        thread = threading.Thread(target=self._run_event_loop, daemon=True)
        thread.start()
    
    def _run_event_loop(self):
        """Run the async event loop in a background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._sync_loop())
        except Exception as e:
            logger.error(f"Sync loop error: {e}")
        finally:
            self._loop.close()
    
    async def _sync_loop(self):
        """Main sync loop."""
        # Initial sync on start (skip if low power mode)
        if self.config.sync_on_start and not self.config.low_power_mode:
            await self._do_sync()
        elif self.config.low_power_mode:
            # In low power mode, wait a bit before first sync to not slow down app startup
            logger.info("Low power mode: delaying initial sync by 60s")
            await asyncio.sleep(60)
            if self._running:
                await self._do_sync()
        
        while self._running:
            # Calculate next sync time
            self.status.next_sync = datetime.now() + timedelta(minutes=self.config.interval_minutes)
            self._notify_status()
            
            # Wait for interval (in low power mode, use smaller sleep chunks to reduce memory)
            if self.config.low_power_mode:
                # Sleep in 60s chunks to allow for quicker shutdown
                for _ in range(self.config.interval_minutes):
                    if not self._running:
                        break
                    await asyncio.sleep(60)
            else:
                await asyncio.sleep(self.config.interval_minutes * 60)
            
            if self._running and self.config.enabled:
                await self._do_sync()
    
    async def _do_sync(self):
        """Perform a sync."""
        if self.status.is_syncing:
            logger.debug("Sync already in progress, skipping")
            return
        
        self.status.is_syncing = True
        self.status.last_sync = datetime.now()
        self._notify_status()
        
        all_entries = []
        
        try:
            # Import here to avoid circular imports
            from . import toggl_configured, toggl_sync, clockify_configured, clockify_sync
            
            # In low power mode, add small delays between API calls
            delay = 1.0 if self.config.low_power_mode else 0
            
            # Toggl sync
            if self.config.toggl_enabled and toggl_configured():
                try:
                    entries = await toggl_sync(days=self.config.days_to_sync)
                    # Limit entries in low power mode
                    if self.config.low_power_mode and len(entries) > self.config.batch_size:
                        entries = entries[:self.config.batch_size]
                    all_entries.extend(entries)
                    logger.info(f"Toggl: synced {len(entries)} entries")
                    if delay:
                        await asyncio.sleep(delay)
                except Exception as e:
                    logger.error(f"Toggl sync failed: {e}")
            
            # Clockify sync
            if self.config.clockify_enabled and clockify_configured():
                try:
                    entries = await clockify_sync(days=self.config.days_to_sync)
                    if self.config.low_power_mode and len(entries) > self.config.batch_size:
                        entries = entries[:self.config.batch_size]
                    all_entries.extend(entries)
                    logger.info(f"Clockify: synced {len(entries)} entries")
                except Exception as e:
                    logger.error(f"Clockify sync failed: {e}")
            
            # aTimeLogger sync (your main app!)
            from . import atimelogger_configured, atimelogger_sync
            if self.config.atimelogger_enabled and atimelogger_configured():
                try:
                    entries = await atimelogger_sync(days=self.config.days_to_sync)
                    if self.config.low_power_mode and len(entries) > self.config.batch_size:
                        entries = entries[:self.config.batch_size]
                    all_entries.extend(entries)
                    logger.info(f"aTimeLogger: synced {len(entries)} entries")
                    if delay:
                        await asyncio.sleep(delay)
                except Exception as e:
                    logger.error(f"aTimeLogger sync failed: {e}")
            
            self.status.last_success = True
            self.status.last_error = None
            self.status.entries_synced = len(all_entries)
            
            # Callback with synced entries
            if all_entries and self.on_sync_complete:
                try:
                    self.on_sync_complete(all_entries)
                except Exception as e:
                    logger.error(f"Sync callback failed: {e}")
        
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            self.status.last_success = False
            self.status.last_error = str(e)
        
        finally:
            self.status.is_syncing = False
            self._notify_status()
    
    def _notify_status(self):
        """Notify status change callback."""
        if self.on_status_change:
            try:
                self.on_status_change(self.status)
            except Exception as e:
                logger.error(f"Status callback failed: {e}")
    
    def stop(self):
        """Stop the background sync service."""
        logger.info("Stopping background sync")
        self._running = False
    
    async def sync_now(self):
        """Trigger an immediate sync."""
        await self._do_sync()
    
    def get_status(self) -> SyncStatus:
        """Get current sync status."""
        return self.status
    
    def update_config(self, **kwargs):
        """Update sync configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        logger.info(f"Sync config updated: {kwargs}")


# Global singleton
_sync_service: Optional[BackgroundSyncService] = None


def get_sync_service() -> BackgroundSyncService:
    """Get or create the global sync service."""
    global _sync_service
    if _sync_service is None:
        _sync_service = BackgroundSyncService()
    return _sync_service


def start_background_sync(
    interval_minutes: int = 5,
    on_sync_complete: Optional[Callable[[list[dict]], Any]] = None,
    on_status_change: Optional[Callable[[SyncStatus], Any]] = None,
) -> BackgroundSyncService:
    """
    Start background sync with configuration.
    
    Args:
        interval_minutes: Sync interval
        on_sync_complete: Callback when sync completes with new entries
        on_status_change: Callback when status changes
        
    Returns:
        The sync service instance
    """
    global _sync_service
    
    config = SyncConfig(interval_minutes=interval_minutes)
    _sync_service = BackgroundSyncService(
        config=config,
        on_sync_complete=on_sync_complete,
        on_status_change=on_status_change,
    )
    _sync_service.start()
    
    return _sync_service


def stop_background_sync():
    """Stop the background sync service."""
    global _sync_service
    if _sync_service:
        _sync_service.stop()
        _sync_service = None
