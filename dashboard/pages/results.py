"""
Results Page
============
View ML processing results with card-based interface.

Each uploaded dataset is displayed as a card that can be clicked
to view detailed information including original data and ML results.
"""

import streamlit as st
import sys
import os
from datetime import datetime
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_database_connection, check_database_health


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Results",
    page_icon="📊",
    layout="wide"
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_dataset_card(dataset_doc, pipeline_run_doc=None):
    """
    Create a card display for a dataset.

    Parameters:
    -----------
    dataset_doc : dict
        Dataset document from MongoDB
    pipeline_run_doc : dict, optional
        Pipeline run document if available
    """
    batch_id = dataset_doc.get('_id', 'unknown')
    filename = dataset_doc.get('filename', 'unnamed')
    timestamp = dataset_doc.get('upload_timestamp', datetime.now())

    # Parse timestamp if string
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp)
        except:
            timestamp = datetime.now()

    # Get metadata
    metadata = dataset_doc.get('metadata', {})
    row_count = metadata.get('row_count', 0)
    column_count = metadata.get('column_count', 0)

    # Get status from pipeline run
    status = 'uploaded'
    if pipeline_run_doc:
        status = pipeline_run_doc.get('status', 'uploaded')

    # Card container with border
    with st.container():
        card = st.container()
        with card:
            # Card header with filename and status
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.markdown(f"### 📁 {filename}")
                st.caption(f"Batch ID: {batch_id[:20]}...")

            with col2:
                # Status indicator
                status_emoji = {
                    'completed': '✅',
                    'failed': '❌',
                    'processing': '⏳',
                    'uploaded': '📤'
                }.get(status, '❓')
                st.markdown(f"**Status:** {status_emoji} {status.title()}")

            with col3:
                # View button
                if st.button("View Details", key=f"view_{batch_id}"):
                    st.session_state['selected_dataset'] = batch_id
                    st.rerun()

            # Card body with metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Rows", f"{row_count:,}")

            with col2:
                st.metric("Columns", column_count)

            with col3:
                st.metric("Upload Time", timestamp.strftime('%Y-%m-%d %H:%M'))

            with col4:
                has_missing = metadata.get('has_missing_values', False)
                st.metric("Data Quality", "Has Missing" if has_missing else "Complete")

        st.markdown("---")


def show_dataset_details(dataset_doc, pipeline_run_doc=None):
    """
    Show detailed view of a dataset.

    Parameters:
    -----------
    dataset_doc : dict
        Dataset document from MongoDB
    pipeline_run_doc : dict, optional
        Pipeline run document if available
    """
    filename = dataset_doc.get('filename', 'unnamed')
    batch_id = dataset_doc.get('_id', 'unknown')

    # Header with back button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back to Results"):
            st.session_state.pop('selected_dataset', None)
            st.rerun()

    with col2:
        st.title(f"Dataset: {filename}")
        st.caption(f"Batch ID: {batch_id}")

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Original Data",
        "🤖 ML Results",
        "📈 Visualizations",
        "ℹ️ Metadata"
    ])

    # Tab 1: Original Data
    with tab1:
        st.subheader("Original Dataset")

        # Load data from MongoDB document
        data_list = dataset_doc.get('data', [])
        if data_list:
            df = pd.DataFrame(data_list)

            # Display controls
            col1, col2 = st.columns([3, 1])
            with col1:
                # Row display slider
                max_rows = min(1000, len(df))
                num_rows = st.slider(
                    "Number of rows to display",
                    min_value=10,
                    max_value=max_rows,
                    value=min(100, max_rows),
                    step=10
                )

            with col2:
                # Download button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{filename}.csv",
                    mime='text/csv',
                    help="Download the full dataset"
                )

            # Display dataframe
            st.dataframe(df.head(num_rows), use_container_width=True, height=400)
            st.caption(f"Showing {min(num_rows, len(df))} of {len(df):,} rows")

            # Statistics
            st.subheader("Statistical Summary")
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            else:
                st.info("No numeric columns found for statistical summary")
        else:
            st.warning("No data found for this dataset")

    # Tab 2: ML Results
    with tab2:
        st.subheader("Machine Learning Results")

        # Check if ML processing has been done
        if pipeline_run_doc:
            ml_status = pipeline_run_doc.get('summary', {}).get('ml_status', 'not_processed')

            if ml_status == 'skipped':
                st.info("🔄 ML processing was skipped (minimal mode)")
                st.markdown("""
                **To enable ML processing:**
                1. Change processing mode in `ml_model/components/server.py`
                2. Re-upload and process the dataset
                """)
            elif ml_status == 'completed':
                st.success("✅ ML processing completed")
                # Future: Display actual ML results here
                st.markdown("*ML results display will be implemented here*")
            else:
                st.warning(f"⚠️ ML Status: {ml_status}")
        else:
            st.info("📤 Dataset uploaded but not yet processed through ML pipeline")

    # Tab 3: Visualizations
    with tab3:
        st.subheader("Data Visualizations")

        if data_list:
            df = pd.DataFrame(data_list)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

            if len(numeric_cols) > 0:
                # Column selector for visualization
                col1, col2 = st.columns(2)

                with col1:
                    x_col = st.selectbox("Select X axis", numeric_cols)

                with col2:
                    y_col = st.selectbox("Select Y axis", numeric_cols[1:] + [numeric_cols[0]])

                # Create scatter plot
                if x_col and y_col:
                    st.scatter_chart(df[[x_col, y_col]].head(500))

                # Distribution plots
                st.subheader("Distribution Analysis")
                selected_col = st.selectbox("Select column for distribution", numeric_cols)
                if selected_col:
                    st.bar_chart(df[selected_col].value_counts().head(20))
            else:
                st.info("No numeric columns available for visualization")
        else:
            st.warning("No data available for visualization")

    # Tab 4: Metadata
    with tab4:
        st.subheader("Dataset Metadata")

        # Display metadata
        metadata = dataset_doc.get('metadata', {})

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### File Information")
            st.json({
                'filename': dataset_doc.get('filename'),
                'file_type': metadata.get('file_type'),
                'upload_timestamp': str(dataset_doc.get('upload_timestamp')),
                'batch_id': dataset_doc.get('_id')
            })

        with col2:
            st.markdown("### Data Properties")
            st.json({
                'row_count': metadata.get('row_count'),
                'column_count': metadata.get('column_count'),
                'has_missing_values': metadata.get('has_missing_values'),
                'column_names': metadata.get('column_names', [])
            })

        # Pipeline run information
        if pipeline_run_doc:
            st.markdown("### Processing Information")
            st.json(pipeline_run_doc.get('summary', {}))


# ============================================================================
# MAIN PAGE
# ============================================================================

st.title("📊 ML Processing Results")
st.markdown("View and explore your uploaded datasets and their ML analysis results.")

st.markdown("---")

# Database connection check
db_conn = get_database_connection()
is_healthy, message = check_database_health(db_conn)

if not is_healthy:
    st.error("❌ Database connection is not available. Cannot load results.")
    st.stop()

# Check if user selected a specific dataset
if 'selected_dataset' in st.session_state:
    # Detail view mode
    try:
        # Fetch the selected dataset
        datasets_collection = db_conn.get_collection('datasets')
        dataset_doc = datasets_collection.find_one({'_id': st.session_state['selected_dataset']})

        if dataset_doc:
            # Try to get pipeline run info
            pipeline_runs_collection = db_conn.get_collection('pipeline_runs')
            pipeline_run_doc = pipeline_runs_collection.find_one({'batch_id': st.session_state['selected_dataset']})

            # Show detailed view
            show_dataset_details(dataset_doc, pipeline_run_doc)
        else:
            st.error("Dataset not found")
            st.session_state.pop('selected_dataset', None)
            if st.button("Back to Results"):
                st.rerun()

    except Exception as e:
        st.error(f"Error loading dataset details: {e}")
        st.session_state.pop('selected_dataset', None)
        if st.button("Back to Results"):
            st.rerun()

else:
    # List view mode - show all datasets as cards

    # Filters sidebar
    with st.sidebar:
        st.markdown("### 🔍 Filters")

        # Time range filter
        time_filter = st.selectbox(
            "Time Range",
            ["All Time", "Last 24 Hours", "Last 7 Days", "Last 30 Days"]
        )

        # Status filter
        status_filter = st.multiselect(
            "Status",
            ["uploaded", "completed", "failed", "processing"],
            default=["uploaded", "completed", "failed", "processing"]
        )

        # Sort order
        sort_order = st.radio(
            "Sort By",
            ["Newest First", "Oldest First", "Name A-Z", "Name Z-A"]
        )

        st.markdown("---")
        st.caption("Click on any dataset card to view details")

    # Main content area
    try:
        # Get datasets from MongoDB
        datasets_collection = db_conn.get_collection('datasets')
        pipeline_runs_collection = db_conn.get_collection('pipeline_runs')

        # Build query filters
        query = {}

        # Apply time filter
        if time_filter != "All Time":
            from datetime import timedelta
            now = datetime.now()
            if time_filter == "Last 24 Hours":
                time_threshold = now - timedelta(days=1)
            elif time_filter == "Last 7 Days":
                time_threshold = now - timedelta(days=7)
            elif time_filter == "Last 30 Days":
                time_threshold = now - timedelta(days=30)

            query['upload_timestamp'] = {'$gte': time_threshold}

        # Apply sorting
        sort_field = 'upload_timestamp'
        sort_direction = -1  # Newest first by default

        if sort_order == "Oldest First":
            sort_direction = 1
        elif sort_order == "Name A-Z":
            sort_field = 'filename'
            sort_direction = 1
        elif sort_order == "Name Z-A":
            sort_field = 'filename'
            sort_direction = -1

        # Fetch datasets
        datasets = list(datasets_collection.find(query).sort(sort_field, sort_direction).limit(50))

        if datasets:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            # Get all pipeline runs for summary
            all_runs = list(pipeline_runs_collection.find())
            runs_by_batch = {run['batch_id']: run for run in all_runs}

            # Calculate metrics
            total = len(datasets)
            completed = sum(1 for d in datasets if runs_by_batch.get(d['_id'], {}).get('status') == 'completed')
            failed = sum(1 for d in datasets if runs_by_batch.get(d['_id'], {}).get('status') == 'failed')
            pending = total - completed - failed

            with col1:
                st.metric("📁 Total Datasets", total)

            with col2:
                st.metric("✅ Completed", completed)

            with col3:
                st.metric("❌ Failed", failed)

            with col4:
                st.metric("⏳ Pending", pending)

            st.markdown("---")

            # Display datasets as cards
            st.subheader("📋 Datasets")

            for dataset in datasets:
                # Get pipeline run for this dataset if exists
                batch_id = dataset.get('_id')
                pipeline_run = runs_by_batch.get(batch_id)

                # Apply status filter
                if pipeline_run:
                    status = pipeline_run.get('status', 'uploaded')
                else:
                    status = 'uploaded'

                if status in status_filter:
                    create_dataset_card(dataset, pipeline_run)

            # Show message if no datasets match filters
            if not any(runs_by_batch.get(d.get('_id'), {}).get('status', 'uploaded') in status_filter for d in datasets):
                st.info("No datasets match the selected filters")

        else:
            st.info("📭 No datasets found. Upload data through the Upload page to see results here.")

            # Quick link to upload page
            st.markdown("### Getting Started")
            st.markdown("""
            1. Go to the **Upload Data** page
            2. Upload a CSV or Excel file
            3. The data will be processed through the ML pipeline
            4. Return here to view and analyze results
            """)

    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        st.info("Make sure the database is running and contains data")