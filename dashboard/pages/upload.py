"""
Data Upload Page
================
Upload data files and process them through the data pipeline.

This page uses the DataUploadPipeline to orchestrate the workflow:
- File upload and validation
- Data transformation (currently skipped)
- ML service communication
- Results display

The pipeline provides a clean separation of concerns with stages
that can be independently modified and extended.
"""

import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.uploader import UploadService
from components.data_upload_pipeline import DataUploadPipeline, TransformMode


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Upload Data",
    page_icon="⬆️",
    layout="wide"
)

# ============================================================================
# SIDEBAR - ML SERVICE STATUS
# ============================================================================

with st.sidebar:
    st.markdown("### 🔧 Service Status")

    # Test ML service connection
    try:
        import requests
        response = requests.get("http://ml-model:5000", timeout=2)
        # Any response (even 404) means server is running
        st.success("✅ ML Service: Online")
    except requests.ConnectionError:
        st.error("❌ ML Service: Offline")
        st.caption("Start with: `make dev-up`")
    except requests.Timeout:
        st.warning("⚠️ ML Service: Slow Response")
    except:
        st.info("ℹ️ ML Service: Unknown")

    st.markdown("---")
    st.caption("**Logs:** `make dev-logs-ml`")
    st.caption("**Status:** `make dev-status`")


# ============================================================================
# INITIALIZE SERVICES
# ============================================================================

# Initialize upload service with 50MB file size limit
upload_service = UploadService(max_file_size_mb=50)

# Initialize upload pipeline with transformation disabled (for now)
pipeline = DataUploadPipeline(transform_mode=TransformMode.NONE)


# ============================================================================
# MAIN PAGE
# ============================================================================

st.title("📊 Data Upload")
st.markdown("Upload and process your data files through the ML pipeline.")
st.markdown("**Supported formats:** CSV, Excel (.xlsx, .xls) | **Max file size:** 50 MB")

st.markdown("---")

# Upload Section
st.subheader("1. Upload File")

uploaded_file = st.file_uploader(
    "Choose a file",
    type=['xlsx', 'xls', 'csv'],
    help="Upload your data file (CSV or Excel format, max 50MB)"
)

if uploaded_file is not None:

    # ========================================================================
    # PROCESS THROUGH PIPELINE
    # ========================================================================

    try:
        with st.spinner("Processing through data pipeline..."):
            # Execute pipeline with all stages
            result = pipeline.execute(uploaded_file, upload_service)

        # ====================================================================
        # DISPLAY PIPELINE RESULTS
        # ====================================================================

        st.markdown("---")
        st.subheader("2. Pipeline Status")

        if result.is_success:
            # Pipeline completed successfully
            st.success(f"✅ {result.message}")

            # Get the dataset from result
            dataset = result.dataset
            ml_response = result.ml_response

            # ================================================================
            # ML SERVICE RESPONSE
            # ================================================================

            if ml_response:
                st.markdown("---")
                st.subheader("3. ML Service Response")

                # Display response details
                col_ml1, col_ml2, col_ml3 = st.columns(3)

                with col_ml1:
                    st.metric("Status", ml_response.get('status', 'unknown').upper())

                with col_ml2:
                    st.metric("Batch ID", ml_response.get('batch_id', 'N/A')[:20] + "...")

                with col_ml3:
                    st.metric("Rows Stored", f"{ml_response.get('row_count', 0):,}")

                # Show processing details
                if 'processing_summary' in ml_response:
                    summary_data = ml_response['processing_summary']
                    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

                    with col_s1:
                        st.metric("ML Status", summary_data.get('ml_status', 'N/A'))

                    with col_s2:
                        st.metric("Processing Mode", summary_data.get('processing_mode', 'N/A'))

                    with col_s3:
                        st.metric("Storage Status", summary_data.get('storage_status', 'N/A'))

                    with col_s4:
                        has_missing = summary_data.get('has_missing_values', False)
                        st.metric("Data Quality", "Has Missing" if has_missing else "Complete")

                # Show full response in expander
                with st.expander("📋 View Full ML Service Response"):
                    st.json(ml_response)

            # ================================================================
            # DATASET SUMMARY
            # ================================================================

            st.markdown("---")
            st.subheader("4. Dataset Summary")

            summary = dataset.get_summary()

            # Display key metrics in columns
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("📊 Rows", f"{summary['row_count']:,}")

            with col2:
                st.metric("📋 Columns", summary['column_count'])

            with col3:
                st.metric("💾 File Size", f"{uploaded_file.size / 1024:.2f} KB")

            with col4:
                st.metric("🧠 Memory", f"{summary['memory_usage_mb']} MB")

            # Missing values indicator
            st.markdown("")  # Spacing
            if summary['has_missing_values']:
                missing_total = sum(summary['missing_values_per_column'].values())
                missing_pct = (missing_total / (summary['row_count'] * summary['column_count'])) * 100
                st.warning(f"⚠️ Dataset contains **{missing_total:,}** missing values ({missing_pct:.2f}% of total)")
            else:
                st.success("✅ No missing values detected - Data is complete!")

            # ================================================================
            # TRANSFORMATION INFO (if applicable)
            # ================================================================

            if result.metadata and 'transform_info' in result.metadata:
                transform_info = result.metadata['transform_info']
                if transform_info.get('transform_status') != 'skipped':
                    st.markdown("---")
                    st.subheader("5. Transformation Details")

                    col_t1, col_t2 = st.columns(2)
                    with col_t1:
                        st.metric("Transform Status", transform_info.get('transform_status', 'N/A'))
                    with col_t2:
                        st.metric("Transform Mode", transform_info.get('mode', 'N/A'))

                    if 'original_shape' in transform_info:
                        st.info(f"Original shape: {transform_info['original_shape']}")
                    if 'transformed_shape' in transform_info:
                        st.info(f"Transformed shape: {transform_info['transformed_shape']}")

            # ================================================================
            # TABBED INTERFACE FOR DATA VIEWING
            # ================================================================

            st.markdown("---")
            st.subheader("6. Dataset Details")

            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Data Preview",
                "📋 Column Info",
                "📈 Statistics",
                "💾 Export"
            ])

            # TAB 1: DATA PREVIEW
            with tab1:
                st.markdown("### Interactive Data Preview")

                # Configurable number of rows
                col_a, col_b = st.columns([3, 1])

                with col_a:
                    max_preview_rows = min(100, summary['row_count'])
                    num_rows = st.slider(
                        "Number of rows to display",
                        min_value=5,
                        max_value=max_preview_rows,
                        value=min(20, max_preview_rows),
                        step=5,
                        help=f"Preview up to {max_preview_rows} rows from your dataset"
                    )

                with col_b:
                    st.markdown("")  # Spacing
                    st.markdown("")  # Spacing
                    show_all = st.checkbox("Show all rows", help="Display entire dataset (may be slow for large files)")

                # Display data
                if show_all:
                    st.warning(f"⚠️ Displaying all {summary['row_count']:,} rows - this may take a moment...")
                    preview_df = dataset.raw_dataframe
                else:
                    preview_df = dataset.get_preview(n_rows=num_rows)

                st.dataframe(
                    preview_df,
                    use_container_width=True,
                    height=400  # Fixed height with scroll
                )

                st.caption(f"Showing {len(preview_df):,} of {summary['row_count']:,} rows")

            # TAB 2: COLUMN INFORMATION
            with tab2:
                st.markdown("### Column Details")

                # Create column info table
                import pandas as pd

                col_info = pd.DataFrame({
                    'Column Name': summary['column_names'],
                    'Data Type': [summary['data_types'][col] for col in summary['column_names']],
                    'Missing Values': [summary['missing_values_per_column'][col] for col in summary['column_names']],
                    'Missing %': [
                        f"{(summary['missing_values_per_column'][col] / summary['row_count'] * 100):.1f}%"
                        for col in summary['column_names']
                    ]
                })

                st.dataframe(col_info, use_container_width=True, hide_index=True, height=400)

                # Data type distribution
                st.markdown("#### Data Type Distribution")
                type_counts = pd.Series(summary['data_types'].values()).value_counts()

                col_x, col_y = st.columns(2)
                with col_x:
                    for dtype, count in type_counts.items():
                        st.metric(dtype, count)

            # TAB 3: STATISTICS
            with tab3:
                st.markdown("### Statistical Summary")

                try:
                    # Generate statistics for numeric columns only
                    numeric_df = dataset.raw_dataframe.select_dtypes(include=['number'])

                    if not numeric_df.empty:
                        stats_df = numeric_df.describe().T
                        stats_df = stats_df.round(3)

                        st.dataframe(stats_df, use_container_width=True, height=400)

                        # Additional insights
                        st.markdown("#### Quick Insights")
                        col_i, col_j = st.columns(2)

                        with col_i:
                            st.metric("Numeric Columns", len(numeric_df.columns))
                            st.metric("Total Data Points", f"{numeric_df.size:,}")

                        with col_j:
                            categorical_cols = len(dataset.raw_dataframe.columns) - len(numeric_df.columns)
                            st.metric("Categorical Columns", categorical_cols)
                            if summary['has_missing_values']:
                                missing_total = sum(summary['missing_values_per_column'].values())
                                missing_pct = (missing_total / (summary['row_count'] * summary['column_count'])) * 100
                                st.metric("Completeness", f"{100 - missing_pct:.1f}%")
                    else:
                        st.info("No numeric columns found in dataset")

                except Exception as e:
                    st.error(f"Error generating statistics: {str(e)}")

            # TAB 4: EXPORT OPTIONS
            with tab4:
                st.markdown("### Export Options")

                col_m, col_n = st.columns(2)

                with col_m:
                    st.markdown("#### 📥 Download Pipeline Result")
                    st.json(result.to_dict())

                with col_n:
                    st.markdown("#### 📊 Download Full Dataset")

                    # CSV download
                    csv_data = dataset.raw_dataframe.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download as CSV",
                        data=csv_data,
                        file_name=f"processed_{uploaded_file.name.replace('.xlsx', '.csv').replace('.xls', '.csv')}",
                        mime="text/csv",
                        help="Download the dataset as CSV"
                    )

                    # Excel download (if openpyxl available)
                    try:
                        from io import BytesIO
                        excel_buffer = BytesIO()
                        dataset.raw_dataframe.to_excel(excel_buffer, index=False, engine='openpyxl')
                        excel_data = excel_buffer.getvalue()

                        st.download_button(
                            label="Download as Excel",
                            data=excel_data,
                            file_name=f"processed_{uploaded_file.name}",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            help="Download the dataset as Excel"
                        )
                    except ImportError:
                        st.info("Excel export requires openpyxl package")

            # ================================================================
            # PIPELINE CONFIGURATION
            # ================================================================

            st.markdown("---")
            st.subheader("7. Pipeline Configuration")

            col_config1, col_config2 = st.columns([2, 1])

            with col_config1:
                st.info(f"""
                **Current Pipeline Mode:** `{pipeline.transform_mode.value}`

                The data pipeline is currently operating with transformation disabled.
                When enabled, it will apply the trasformator.py pipeline for:
                - Data cleaning and normalization
                - Outlier treatment
                - Dimensionality reduction (PCA/UMAP)
                - Categorical encoding
                """)

            with col_config2:
                st.markdown("#### 📊 Pipeline Status")
                st.metric("Pipeline", "✅ Success")
                st.metric("Transform", "⏭️ Skipped")
                st.metric("ML Comm", "✅ Sent" if ml_response else "❌ Failed")

            # Store dataset in session state for future use
            st.session_state['uploaded_dataset'] = dataset
            st.session_state['upload_timestamp'] = summary['upload_timestamp']
            st.session_state['pipeline_result'] = result

        else:
            # Pipeline failed
            st.error(f"❌ {result.message}")

            # Show errors
            if result.errors:
                st.markdown("### Error Details")
                for error in result.errors:
                    st.error(f"**Stage:** {error.get('stage', 'unknown')} - **Error:** {error.get('error', 'Unknown error')}")

            st.info("""
            **Troubleshooting:**
            - Check that the ML service is running
            - Verify the file format and content
            - Review the error messages above
            - Check service logs: `make dev-logs-ml`
            """)

    except Exception as e:
        # Global error handler for unexpected issues
        st.error("❌ An unexpected error occurred!")
        st.error(f"**Error:** {str(e)}")

        # Log error for debugging
        import traceback
        with st.expander("🔧 Technical Details"):
            st.code(traceback.format_exc())

else:
    st.info("👆 Please upload a file to begin")
    st.markdown("""
    ### How it works:

    The **Data Upload Pipeline** orchestrates your data through three stages:

    1. **Validation Stage** ✓
       - File type and size validation
       - Data structure verification
       - Column analysis

    2. **Transformation Stage** (Currently Skipped)
       - Data cleaning and normalization
       - Dimensionality reduction
       - Feature engineering

    3. **ML Communication Stage** ✓
       - Serialization and transmission
       - Response processing
       - Result storage

    ### What you'll see:
    - **Pipeline Status:** Real-time execution feedback
    - **ML Response:** Processing results and batch ID
    - **Dataset Summary:** Key metrics and quality indicators
    - **Interactive Tabs:** Preview, statistics, and export options
    """)