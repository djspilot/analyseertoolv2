"""
Caching module for chart data.
Implements memoization and cache invalidation for performance optimization.
"""

import hashlib
import pickle
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional
import pandas as pd

from ..logger import setup_logger

logger = setup_logger(__name__)

# Cache configuration
CACHE_DIR = Path(__file__).parent.parent / ".cache"
CACHE_TTL_MINUTES = 30  # Cache expires after 30 minutes
ENABLE_CACHE = True


def _get_cache_key(data: Any) -> str:
    """Generate a hash-based cache key from data."""
    if isinstance(data, pd.DataFrame):
        # Hash based on shape and a sample of data
        sample = str(data.shape) + str(data.head(10).values.tobytes())
    elif isinstance(data, (list, dict, tuple)):
        sample = str(sorted(data) if isinstance(data, dict) else data)
    else:
        sample = str(data)
    
    return hashlib.md5(sample.encode()).hexdigest()


def _get_cache_path(key: str, suffix: str = "pkl") -> Path:
    """Get the cache file path for a key."""
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{key}.{suffix}"


def _is_cache_valid(cache_path: Path, ttl_minutes: int) -> bool:
    """Check if a cache file is still valid based on age."""
    if not cache_path.exists():
        return False
    
    age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
    return age <= timedelta(minutes=ttl_minutes)


def clear_cache(pattern: Optional[str] = None) -> int:
    """
    Clear cached data.
    
    Args:
        pattern: Optional pattern to match cache files (e.g., "chart_")
        
    Returns:
        Number of cache files cleared
    """
    if not CACHE_DIR.exists():
        return 0
    
    cleared = 0
    for cache_file in CACHE_DIR.iterdir():
        if pattern is None or cache_file.name.startswith(pattern):
            try:
                cache_file.unlink()
                cleared += 1
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file.name}: {e}")
    
    if cleared > 0:
        logger.info(f"Cleared {cleared} cache file(s)")
    
    return cleared


def cache_chart(ttl_minutes: int = CACHE_TTL_MINUTES, enabled: bool = ENABLE_CACHE):
    """
    Decorator to cache chart generation results.
    
    Args:
        ttl_minutes: Time-to-live for cache in minutes
        enabled: Whether caching is enabled
        
    Example:
        @cache_chart(ttl_minutes=15)
        def create_chart(df, param1, param2):
            # Expensive chart generation
            return fig
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not enabled:
                return func(*args, **kwargs)
            
            # Generate cache key from function name and arguments
            key_parts = [func.__name__]
            
            # Add DataFrame hashes
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    key_parts.append(_get_cache_key(arg))
                else:
                    key_parts.append(str(arg))
            
            # Add keyword arguments
            sorted_kwargs = sorted(kwargs.items())
            for k, v in sorted_kwargs:
                if isinstance(v, pd.DataFrame):
                    key_parts.append(f"{k}={_get_cache_key(v)}")
                else:
                    key_parts.append(f"{k}={v}")
            
            cache_key = hashlib.md5("_".join(key_parts).encode()).hexdigest()
            cache_path = _get_cache_path(cache_key)
            
            # Check cache
            if _is_cache_valid(cache_path, ttl_minutes):
                try:
                    logger.debug(f"Cache hit: {func.__name__}")
                    with open(cache_path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load cache: {e}")
            
            # Generate result
            result = func(*args, **kwargs)
            
            # Save to cache
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(result, f)
                logger.debug(f"Cached: {func.__name__} (key: {cache_key[:8]}...)")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
            
            return result
        
        return wrapper
    return decorator