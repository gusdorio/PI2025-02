"""
Dashboard Configuration
=======================
Configuration settings and database connection for the Streamlit dashboard.
"""

import os
import sys
import streamlit as st

# Add parent directory to path to import shared models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import MongoDBConnection, DatabaseConfig


# ============================================================================
# DATABASE CONNECTION
# ============================================================================

@st.cache_resource
def get_database_connection():
    """
    Initialize and cache database connection.

    Returns:
        MongoDBConnection: Database connection instance or None if failed

    Usage in pages:
        from config import get_database_connection
        db = get_database_connection()
        collection = db.get_collection('my_collection')
    """
    try:
        db_conn = MongoDBConnection()
        return db_conn
    except Exception as e:
        st.error(f"Failed to initialize database connection: {e}")
        return None


def check_database_health(db_conn):
    """
    Check if database connection is healthy.

    Args:
        db_conn: MongoDBConnection instance

    Returns:
        tuple: (is_healthy: bool, message: str)
    """
    if db_conn is None:
        return False, "Database connection not initialized"

    try:
        is_healthy = db_conn.test_connection()
        if is_healthy:
            return True, "Database connection healthy"
        else:
            return False, "Database connection test failed"
    except Exception as e:
        return False, f"Database health check error: {str(e)}"


# ============================================================================
# DASHBOARD CONFIGURATION
# ============================================================================

class DashboardConfig:
    """Dashboard-specific configuration settings."""

    # Streamlit Configuration
    PAGE_TITLE = "ML Operations Dashboard"
    PAGE_ICON = ":bar_chart:"
    LAYOUT = "wide"

    # Refresh Intervals (in seconds)
    METRICS_REFRESH_INTERVAL = 30
    LOGS_REFRESH_INTERVAL = 5

    # Display Settings
    MAX_ROWS_DISPLAY = 100
    CHART_HEIGHT = 400
    CHART_WIDTH = 600

    # Feature Flags
    ENABLE_DATA_EXPORT = os.getenv('ENABLE_DATA_EXPORT', 'true').lower() == 'true'
    ENABLE_QUERY_BUILDER = os.getenv('ENABLE_QUERY_BUILDER', 'true').lower() == 'true'
    ENABLE_REAL_TIME_MONITORING = os.getenv('ENABLE_REAL_TIME_MONITORING', 'false').lower() == 'true'

    # Pagination
    ITEMS_PER_PAGE = 20

    @classmethod
    def is_development(cls) -> bool:
        """Check if running in development mode."""
        return os.getenv('ENVIRONMENT', 'production') == 'development'

    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production mode."""
        return os.getenv('ENVIRONMENT', 'production') == 'production'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_db_config_info():
    """Get database configuration info for display."""
    return {
        'host': DatabaseConfig.MONGO_HOST,
        'port': DatabaseConfig.MONGO_PORT,
        'database': DatabaseConfig.MONGO_DATABASE,
        'pool_size': DatabaseConfig.POOL_SIZE
    }
