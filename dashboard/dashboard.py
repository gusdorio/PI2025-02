"""
ML Operations Dashboard - Main Entry Point
==========================================
Minimal homepage displaying system and database status.
"""

import streamlit as st
import os
from datetime import datetime, timezone

from config import (
    get_database_connection,
    check_database_health,
    get_db_config_info,
    DashboardConfig
)
from theme import apply_theme


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title=DashboardConfig.PAGE_TITLE,
    page_icon=DashboardConfig.PAGE_ICON,
    layout=DashboardConfig.LAYOUT,
    initial_sidebar_state="expanded",
)


# ============================================================================
# STYLING
# ============================================================================

apply_theme()


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### ML Operations Dashboard")
    st.markdown("---")

    # Environment Info
    st.markdown("### Environment")
    env = os.getenv('ENVIRONMENT', 'unknown')
    if env == 'development':
        st.info(f"MODE: {env.upper()}")
    elif env == 'production':
        st.success(f"MODE: {env.upper()}")
    else:
        st.warning(f"MODE: {env.upper()}")

    # System Time
    st.markdown("### System Time")
    st.text(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'))


# ============================================================================
# MAIN PAGE - SYSTEM STATUS
# ============================================================================

st.markdown('<p class="main-header">ML Operations Dashboard</p>', unsafe_allow_html=True)
st.markdown("System monitoring and database status.")

st.markdown("---")

# System Status Metrics
st.subheader("System Status")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<span class="status-indicator"></span>', unsafe_allow_html=True)
    st.metric(
        label="Dashboard Status",
        value="Online",
        delta="Healthy"
    )

with col2:
    db_conn = get_database_connection()
    is_healthy, message = check_database_health(db_conn)

    if is_healthy:
        st.metric(
            label="Database Status",
            value="Connected",
            delta="Healthy"
        )
    else:
        st.metric(
            label="Database Status",
            value="Error",
            delta="Unhealthy",
            delta_color="inverse"
        )

with col3:
    st.metric(
        label="Environment",
        value=os.getenv('ENVIRONMENT', 'unknown').upper()
    )

# Database Connection Details
st.markdown("---")
st.subheader("Database Connection")

if is_healthy:
    st.markdown(f'<div class="status-box status-success">SUCCESS: {message}</div>', unsafe_allow_html=True)

    # Display connection info
    with st.expander("Connection Details"):
        db_info = get_db_config_info()
        st.code(f"""
Host: {db_info['host']}
Port: {db_info['port']}
Database: {db_info['database']}
Pool Size: {db_info['pool_size']}
        """)
else:
    st.markdown(f'<div class="status-box status-error">ERROR: {message}</div>', unsafe_allow_html=True)
    st.error("Please check your database configuration and ensure MongoDB is running.")

# Quick Actions
st.markdown("---")
st.subheader("Quick Actions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Refresh Connection", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

with col2:
    if st.button("View Collections", use_container_width=True):
        if db_conn and is_healthy:
            try:
                collections = db_conn.db.list_collection_names()
                st.success(f"Found {len(collections)} collections")
                if collections:
                    st.write(collections)
                else:
                    st.info("No collections found in database")
            except Exception as e:
                st.error(f"Error listing collections: {e}")
        else:
            st.warning("Database not connected")

with col3:
    if st.button("Test Query", use_container_width=True):
        if db_conn and is_healthy:
            with st.spinner("Testing database query..."):
                try:
                    result = db_conn.client.admin.command('ping')
                    st.success(f"Query successful: {result}")
                except Exception as e:
                    st.error(f"Query failed: {e}")
        else:
            st.warning("Database not connected")
