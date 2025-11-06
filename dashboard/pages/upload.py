"""
Data Upload Page
================
Upload data files for ML processing.

CURRENT FUNCTIONALITY:
- File upload (Excel/CSV)
- Basic file info display
- DB connection check

EXTENSION POINTS:
1. File Validation (components/uploader.py)
   - Validate file format and structure
   - Check data quality
   - Return validation errors/warnings

2. ML Processing Trigger (components/ml_client.py)
   - Send file to ML service
   - Track processing status
   - Handle async operations

3. Database Storage
   - Save file metadata to MongoDB
   - Store upload timestamp and user info
   - Track processing history

4. User Feedback
   - Progress bar during upload
   - Real-time status updates
   - Error handling and retry logic
"""

import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_database_connection, check_database_health


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Upload Data",
    page_icon=":arrow_up:",
    layout="wide"
)


# ============================================================================
# MAIN PAGE
# ============================================================================

st.title("Data Upload")
st.markdown("Upload data files for machine learning processing.")

st.markdown("---")

# Database Status Check
db_conn = get_database_connection()
is_healthy, message = check_database_health(db_conn)

if not is_healthy:
    st.warning("Database connection is not available. Upload functionality requires database connection.")
    st.stop()

# Upload Section
st.subheader("Upload File")

uploaded_file = st.file_uploader(
    "Choose a file (Excel or CSV)",
    type=['xlsx', 'xls', 'csv'],
    help="Upload your data file for ML processing"
)

if uploaded_file is not None:
    st.success(f"File uploaded: {uploaded_file.name}")
    st.info("File processing will be implemented via components/uploader.py")

    # Display file info
    file_details = {
        "Filename": uploaded_file.name,
        "FileType": uploaded_file.type,
        "FileSize": f"{uploaded_file.size} bytes"
    }
    st.json(file_details)

    # Placeholder for future implementation
    st.markdown("---")
    st.subheader("Next Steps")
    st.markdown("""
    - File validation will be handled by `components/uploader.py`
    - Data processing will be triggered via `components/ml_client.py`
    - Results will be stored in MongoDB
    - Processing status will be displayed here
    """)

else:
    st.info("Please upload a file to continue")
