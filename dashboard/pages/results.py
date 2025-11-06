"""
Results Page
============
View and explore ML processing results from MongoDB.

CURRENT FUNCTIONALITY:
- Database connection check
- Collection browser
- Document viewer with pagination
- Basic metrics display

EXTENSION POINTS:
1. Results Query
   - Filter by date range, status, batch_id
   - Sort by different fields
   - Advanced search functionality

2. Metrics Display
   - Query pipeline_runs collection
   - Aggregate statistics (success rate, avg time)
   - Display trend charts

3. Visualizations
   - Plot model performance metrics
   - Feature importance charts
   - Time series analysis
   - Comparison between runs

4. Data Export
   - Export filtered results to CSV/Excel
   - Generate PDF reports
   - Download specific collections

5. Integration
   - Link to upload page for reprocessing
   - Drill-down into specific results
   - Real-time updates for active runs
"""

import streamlit as st
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_database_connection, check_database_health, DashboardConfig


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Results",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)


# ============================================================================
# MAIN PAGE
# ============================================================================

st.title("ML Processing Results")
st.markdown("View and explore machine learning analysis results.")

st.markdown("---")

# Database Status Check
db_conn = get_database_connection()
is_healthy, message = check_database_health(db_conn)

if not is_healthy:
    st.error("Database connection is not available. Cannot load results.")
    st.stop()

# Results Overview
st.subheader("Results Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Total Runs",
        value="0",
        help="Total pipeline executions"
    )

with col2:
    st.metric(
        label="Successful",
        value="0",
        help="Successfully completed runs"
    )

with col3:
    st.metric(
        label="Failed",
        value="0",
        help="Failed pipeline runs"
    )

# Query Results from Database
st.markdown("---")
st.subheader("Recent Results")

try:
    # Check if collections exist
    collections = db_conn.db.list_collection_names()

    if collections:
        st.success(f"Found {len(collections)} collection(s) in database")

        # Collection selector
        selected_collection = st.selectbox(
            "Select collection to view",
            collections,
            help="Choose a MongoDB collection to explore"
        )

        if selected_collection:
            collection = db_conn.get_collection(selected_collection)

            # Get document count
            doc_count = collection.count_documents({})
            st.info(f"Collection '{selected_collection}' contains {doc_count} document(s)")

            if doc_count > 0:
                # Pagination
                items_per_page = DashboardConfig.ITEMS_PER_PAGE
                total_pages = (doc_count + items_per_page - 1) // items_per_page

                page = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=max(1, total_pages),
                    value=1,
                    help=f"Navigate through {total_pages} page(s)"
                )

                # Fetch documents
                skip = (page - 1) * items_per_page
                documents = list(collection.find().skip(skip).limit(items_per_page))

                # Display documents
                st.markdown(f"**Showing {len(documents)} document(s) on page {page} of {total_pages}**")

                for idx, doc in enumerate(documents, start=1):
                    with st.expander(f"Document {skip + idx}"):
                        st.json(doc, expanded=False)

            else:
                st.info("No documents found in this collection")

    else:
        st.info("No collections found in database. Upload and process data to see results.")

except Exception as e:
    st.error(f"Error querying database: {e}")

# Placeholder for future visualizations
st.markdown("---")
st.subheader("Visualizations")
st.info("Charts and graphs will be implemented here to visualize ML results")
