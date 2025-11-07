"""
Data Upload Page - Step 1 MVP
==============================
Upload data files and validate them for ML processing.

STEP 1 FUNCTIONALITY (MVP):
- File upload (Excel/CSV)
- File validation (size, type, content)
- Data preview (configurable up to 100 rows)
- Dataset summary statistics
- In-memory processing only (no DB storage yet)
- Robust error handling

FUTURE ENHANCEMENTS (Step 2+):
1. ML Processing Trigger (components/ml_client.py)
   - Send file to ML service
   - Track processing status
   - Handle async operations

2. Database Storage
   - Save file metadata to MongoDB
   - Store upload timestamp and user info
   - Track processing history

3. Advanced Features
   - Multiple file uploads
   - Batch processing
   - Processing status tracking
"""

import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.uploader import UploadService


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Upload Data",
    page_icon=":arrow_up:",
    layout="wide"
)


# ============================================================================
# INITIALIZE UPLOAD SERVICE
# ============================================================================

# Initialize upload service with 50MB file size limit
upload_service = UploadService(max_file_size_mb=50)


# ============================================================================
# MAIN PAGE
# ============================================================================

st.title("Data Upload - Step 1 MVP")
st.markdown("Upload and validate your data files for machine learning processing.")
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
    # PROCESS UPLOAD WITH GLOBAL ERROR HANDLING
    # ========================================================================

    try:
        with st.spinner("Processing file..."):
            dataset, errors = upload_service.process_upload(uploaded_file)

        # ====================================================================
        # DISPLAY VALIDATION RESULTS
        # ====================================================================

        st.markdown("---")
        st.subheader("2. Validation Status")

        if errors:
            # Show errors
            st.error("File validation failed!")
            for error in errors:
                st.error(f"❌ {error}")

            st.info("Please fix the issues above and try uploading again.")

        else:
            # Validation successful
            st.success(f"✅ File validated successfully: **{uploaded_file.name}**")

            # ================================================================
            # DATASET SUMMARY
            # ================================================================

            st.markdown("---")
            st.subheader("3. Dataset Summary")

            summary = dataset.get_summary()

            # Display key metrics in columns with color-coded cards
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("📊 Rows", f"{summary['row_count']:,}")

            with col2:
                st.metric("📋 Columns", summary['column_count'])

            with col3:
                st.metric("💾 File Size", f"{uploaded_file.size / 1024:.2f} KB")

            with col4:
                st.metric("🧠 Memory", f"{summary['memory_usage_mb']} MB")

            # Missing values indicator with better styling
            st.markdown("")  # Spacing
            if summary['has_missing_values']:
                missing_total = sum(summary['missing_values_per_column'].values())
                missing_pct = (missing_total / (summary['row_count'] * summary['column_count'])) * 100
                st.warning(f"⚠️ Dataset contains **{missing_total:,}** missing values ({missing_pct:.2f}% of total)")
            else:
                st.success("✅ No missing values detected - Data is complete!")

            # ================================================================
            # TABBED INTERFACE FOR ORGANIZED VIEWING
            # ================================================================

            st.markdown("---")
            st.subheader("4. Dataset Details")

            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Data Preview",
                "📋 Column Info",
                "📈 Statistics",
                "💾 Export"
            ])

            # TAB 1: DATA PREVIEW WITH CONFIGURABLE ROWS
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

                # Create enhanced column info table
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

            # TAB 3: BASIC STATISTICS
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
                    st.markdown("#### 📥 Download Dataset Info")
                    st.json(dataset.to_dict())

                with col_n:
                    st.markdown("#### 📊 Download Full Dataset")

                    # CSV download
                    csv_data = dataset.raw_dataframe.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download as CSV",
                        data=csv_data,
                        file_name=f"processed_{uploaded_file.name.replace('.xlsx', '.csv').replace('.xls', '.csv')}",
                        mime="text/csv",
                        help="Download the uploaded dataset as CSV"
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
                            help="Download the uploaded dataset as Excel"
                        )
                    except ImportError:
                        st.info("Excel export requires openpyxl package")

            # ================================================================
            # PROCESSING STATUS
            # ================================================================

            st.markdown("---")
            st.subheader("5. Processing Status")

            col_status1, col_status2 = st.columns([2, 1])

            with col_status1:
                st.success("✅ Dataset object created successfully!")

                st.info(f"""
                **Dataset Object:** `{dataset}`

                The uploaded data has been validated and is ready for processing.

                **What's Next (Step 2+):**
                - Apply ML transformations using `data_transformation` pipeline
                - Send to ML service for analysis
                - Save results to MongoDB
                - Display processing results
                """)

            with col_status2:
                st.markdown("#### 📊 Upload Summary")
                st.metric("Status", "✅ Ready")
                st.metric("Timestamp", summary['upload_timestamp'][:19])
                st.metric("Quality", "Good" if not summary['has_missing_values'] else "Has Missing")

            # Store dataset in session state for future use
            st.session_state['uploaded_dataset'] = dataset
            st.session_state['upload_timestamp'] = summary['upload_timestamp']

    except Exception as e:
        # Global error handler for unexpected issues
        st.error("❌ An unexpected error occurred during file processing!")
        st.error(f"**Error details:** {str(e)}")
        st.info("""
        **Troubleshooting tips:**
        - Ensure the file is not corrupted
        - Try saving the file in a different format
        - Check that the file contains valid data
        - If the issue persists, contact support
        """)

        # Log error for debugging (in production, send to logging service)
        import traceback
        with st.expander("🔧 Technical Details (for debugging)"):
            st.code(traceback.format_exc())

else:
    st.info("👆 Please upload a file to begin")
    st.markdown("""
    ### How to use:
    1. Click the **Browse files** button above
    2. Select a CSV or Excel file (max 50MB)
    3. Wait for validation and preview
    4. Review the dataset summary and preview
    5. Use the tabs to explore your data in detail
    6. Proceed to next steps (coming in Step 2)

    ### What you'll see:
    - **Validation Status:** Confirms your file passed all checks
    - **Dataset Summary:** Key metrics about your data
    - **Data Preview Tab:** View up to 100 rows with a slider
    - **Column Info Tab:** Details about each column
    - **Statistics Tab:** Statistical summary for numeric columns
    - **Export Tab:** Download options for your data
    """)
