"""
Production configuration for Raspberry Pi deployment.
Optimizes memory, CPU usage, and performance for low-resource environments.
"""

import os
from pathlib import Path

# ============================================================================
# ENVIRONMENT DETECTION
# ============================================================================

def is_raspberry_pi() -> bool:
    """Detect if running on Raspberry Pi."""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            return 'Raspberry' in f.read() or 'BCM' in f.read()
    except:
        return False

def is_production() -> bool:
    """Check if running in production mode."""
    return os.environ.get('ENVIRONMENT', 'development').lower() == 'production'


# ============================================================================
# RASPBERRY PI OPTIMIZATIONS
# ============================================================================

RASPBERRY_PI_MODE = is_raspberry_pi() or os.environ.get('RASPBERRY_PI_MODE', 'false').lower() == 'true'

if RASPBERRY_PI_MODE:
    # Reduce memory usage
    os.environ.setdefault('MALLOC_TRIM_THRESHOLD_', '100000')
    
    # Limit pandas memory
    import pandas as pd
    pd.options.mode.copy_on_write = True  # Reduce memory copies


# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

class ProductionConfig:
    """Production configuration optimized for Raspberry Pi."""
    
    # Database
    DB_POOL_SIZE = 2 if RASPBERRY_PI_MODE else 5
    DB_MAX_OVERFLOW = 1 if RASPBERRY_PI_MODE else 10
    
    # Caching
    CACHE_ENABLED = True
    CACHE_TTL_MINUTES = 120 if RASPBERRY_PI_MODE else 30  # Much longer cache on Pi
    CACHE_MAX_SIZE_MB = 30 if RASPBERRY_PI_MODE else 200  # Smaller cache
    
    # Charts
    MAX_DATA_POINTS = 200 if RASPBERRY_PI_MODE else 1000  # Fewer points
    CHART_ANIMATION = False if RASPBERRY_PI_MODE else True
    USE_WEBGL = True
    CHART_QUALITY = 'low' if RASPBERRY_PI_MODE else 'high'
    
    # API/LLM
    API_TIMEOUT_SECONDS = 180 if RASPBERRY_PI_MODE else 90  # Longer timeout
    API_MAX_RETRIES = 2 if RASPBERRY_PI_MODE else 3
    
    # Reflex/Frontend
    REFLEX_COMPRESS = True
    REFLEX_MINIFY = True if is_production() else False
    
    # Memory limits
    MAX_UPLOAD_SIZE_MB = 5 if RASPBERRY_PI_MODE else 50
    MAX_DATAFRAME_ROWS = 3000 if RASPBERRY_PI_MODE else 50000
    
    # Background Sync (for integrations)
    SYNC_INTERVAL_MINUTES = 15 if RASPBERRY_PI_MODE else 5  # Less frequent on Pi
    SYNC_DAYS_LOOKBACK = 1  # Only sync last day
    SYNC_BATCH_SIZE = 50 if RASPBERRY_PI_MODE else 200
    SYNC_ON_STARTUP = not RASPBERRY_PI_MODE  # Skip initial sync on Pi
    
    # Lazy Loading
    LAZY_CHART_DELAY_MS = 500 if RASPBERRY_PI_MODE else 100
    MAX_CONCURRENT_CHARTS = 2 if RASPBERRY_PI_MODE else 6
    
    @classmethod
    def apply(cls):
        """Apply production settings to environment."""
        os.environ['CACHE_ENABLED'] = str(cls.CACHE_ENABLED)
        os.environ['MAX_DATA_POINTS'] = str(cls.MAX_DATA_POINTS)
        os.environ['SYNC_INTERVAL'] = str(cls.SYNC_INTERVAL_MINUTES)


# ============================================================================
# RASPBERRY PI SPECIFIC SETUP
# ============================================================================

def setup_raspberry_pi():
    """
    Apply Raspberry Pi specific optimizations.
    Call this at application startup.
    """
    if not RASPBERRY_PI_MODE:
        return
    
    import logging
    logging.info("🍓 Raspberry Pi mode enabled - applying optimizations")
    
    # 1. Reduce Python garbage collection frequency
    import gc
    gc.set_threshold(700, 10, 10)  # Default is (700, 10, 10)
    
    # 2. Set process priority (nice level)
    try:
        os.nice(5)  # Lower priority to not hog CPU
    except:
        pass
    
    # 3. Limit thread count
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    
    # 4. Apply config
    ProductionConfig.apply()


# ============================================================================
# MEMORY MONITORING
# ============================================================================

def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def check_memory_pressure() -> bool:
    """Check if system is under memory pressure."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.percent > 85  # More than 85% used
    except ImportError:
        return False


def cleanup_if_needed():
    """Run garbage collection if memory pressure is high."""
    if check_memory_pressure():
        import gc
        gc.collect()
        
        # Clear matplotlib figure cache if exists
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except:
            pass


# ============================================================================
# STARTUP HOOK
# ============================================================================

# Auto-apply settings on import if in production
if is_production() or RASPBERRY_PI_MODE:
    setup_raspberry_pi()
