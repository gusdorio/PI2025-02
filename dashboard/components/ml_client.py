"""
ML Service Client

HTTP client for communicating with the ML model service.
Sends dataset data for processing and retrieves results.
"""

import requests
import streamlit as st
import json
import numpy as np
from typing import Dict, Optional


def send_dataset_to_ml(dataset) -> Optional[Dict]:
    """
    Send an UploadedDataset to the ML model service for processing.

    This function serializes the dataset and sends it to the ml-model
    microservice via HTTP POST. In Docker Container Apps environment,
    services communicate by container name.

    Parameters:
    -----------
    dataset : UploadedDataset
        The uploaded dataset object to send for processing

    Returns:
    --------
    dict : Response from ML service with processing status
    None : If request fails

    Example Response:
    {
        "status": "received",
        "message": "Dataset received successfully",
        "timestamp": "2024-01-15T10:30:00",
        "row_count": 1000,
        "column_count": 15,
        "filename": "data.xlsx"
    }
    """
    try:
        # Prepare payload with dataset metadata and full data
        print("\n" + "=" * 60)
        print("[ML CLIENT] Preparing to send dataset to ML service")
        print("=" * 60)

        metadata = dataset.get_summary()

        # Convert numpy types to native Python types for JSON serialization
        # numpy.bool_ -> bool, numpy.int64 -> int, etc.
        def convert_numpy_types(obj):
            """Recursively convert numpy types to native Python types"""
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        # Clean metadata of numpy types
        clean_metadata = convert_numpy_types(metadata)

        payload = {
            "filename": dataset.filename,
            "data": dataset.raw_dataframe.to_json(orient='records'),
            "metadata": clean_metadata
        }

        # Log request details (this should work now with clean types)
        try:
            payload_size_kb = len(json.dumps(payload)) / 1024
        except Exception as e:
            print(f"[WARNING] Could not calculate payload size: {str(e)}")
            payload_size_kb = 0
        print(f"[INFO] Dataset details:")
        print(f"  - Filename: {dataset.filename}")
        print(f"  - Rows: {dataset.row_count:,}")
        print(f"  - Columns: {dataset.column_count}")
        if payload_size_kb > 0:
            print(f"  - Payload size: {payload_size_kb:.2f} KB")
        print(f"[INFO] Target URL: http://ml-model:5000/process")
        print(f"[INFO] Timeout: 60 seconds")

        # Send POST request to ML service
        # In Container Apps, services communicate by container name
        print(f"[INFO] Sending HTTP POST request...")

        response = requests.post(
            'http://ml-model:5000/process',
            json=payload,
            timeout=60,  # Account for cold starts and processing time
            headers={'Content-Type': 'application/json'}
        )

        # Check response status
        print(f"[INFO] Response received - Status Code: {response.status_code}")
        response.raise_for_status()

        # Parse and return response
        result = response.json()

        print(f"[SUCCESS] ✅ ML service responded successfully!")
        print(f"[INFO] Response status: {result.get('status', 'unknown')}")
        print(f"[INFO] Response message: {result.get('message', 'N/A')}")
        print("=" * 60 + "\n")

        return result

    except requests.ConnectionError as e:
        error_msg = "Cannot connect to ML service. Is the ml-model container running?"
        print("\n" + "🔴" * 30)
        print("[ERROR] Connection Error")
        print("🔴" * 30)
        print(f"[ERROR] {error_msg}")
        print(f"[ERROR] Details: {str(e)}")
        print("🔴" * 30 + "\n")

        st.error(f"🔌 **Connection Error**\n\n{error_msg}")
        st.info("**Troubleshooting:**\n- Verify ml-model service is running: `make dev-status`\n- Check ML logs: `make dev-logs-ml`\n- Ensure Docker network is configured correctly")
        return None

    except requests.Timeout:
        error_msg = "Request timed out after 60 seconds"
        print("\n" + "🟡" * 30)
        print("[WARNING] Request Timeout")
        print("🟡" * 30)
        print(f"[WARNING] {error_msg}")
        print("🟡" * 30 + "\n")

        st.warning("⏱️ **Processing Timeout**\n\nThe ML service is taking longer than expected...")
        st.info("This might happen with large datasets or cold starts. Check ML service logs for details.")
        return None

    except requests.HTTPError as e:
        status_code = e.response.status_code
        try:
            error_data = e.response.json()
            error_msg = error_data.get('message', str(e))
        except:
            error_msg = str(e)

        print("\n" + "🔴" * 30)
        print(f"[ERROR] HTTP Error {status_code}")
        print("🔴" * 30)
        print(f"[ERROR] {error_msg}")
        print(f"[ERROR] Response: {e.response.text[:200]}")
        print("🔴" * 30 + "\n")

        if status_code == 400:
            st.error(f"❌ **Validation Error**\n\n{error_msg}")
            st.info("The dataset format was rejected by the ML service. Check data structure and server logs.")
        elif status_code == 500:
            st.error(f"⚠️ **ML Service Error**\n\n{error_msg}")
            st.info("An internal error occurred in the ML service. Check ml-model logs: `make dev-logs-ml`")
        else:
            st.error(f"❌ **HTTP Error {status_code}**\n\n{error_msg}")

        return None

    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON response from ML service: {str(e)}"
        print("\n" + "🔴" * 30)
        print("[ERROR] JSON Decode Error")
        print("🔴" * 30)
        print(f"[ERROR] {error_msg}")
        print("🔴" * 30 + "\n")

        st.error(f"⚠️ **Response Error**\n\n{error_msg}")
        return None

    except TypeError as e:
        if "not JSON serializable" in str(e):
            error_msg = f"JSON serialization error: {str(e)}"
            print("\n" + "🔴" * 30)
            print("[ERROR] JSON Serialization Error")
            print("🔴" * 30)
            print(f"[ERROR] {error_msg}")
            print("[INFO] This usually means numpy types weren't converted properly")
            import traceback
            traceback.print_exc()
            print("🔴" * 30 + "\n")

            st.error(f"❌ **Data Format Error**\n\n{error_msg}")
            st.info("The dataset contains data types that can't be sent as JSON. This is a bug that should be fixed.")
        else:
            raise  # Re-raise if it's a different TypeError

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print("\n" + "🔴" * 30)
        print("[ERROR] Unexpected Error")
        print("🔴" * 30)
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        print("🔴" * 30 + "\n")

        st.error(f"❌ **Unexpected Error**\n\n{error_msg}")
        st.info("Check console logs for full traceback")
        return None


def trigger_ml_processing(data_id: str) -> Optional[Dict]:
    """
    Trigger ML processing by data ID (for database-stored datasets).

    NOTE: This function is a placeholder for future implementation
    when datasets are stored in MongoDB. Currently, use send_dataset_to_ml()
    for direct dataset transmission.

    Parameters:
    -----------
    data_id : str
        Database ID of the dataset to process

    Returns:
    --------
    dict : Response from ML service
    None : If request fails
    """
    try:
        response = requests.post(
            'http://ml-model:5000/process',
            json={'data_id': data_id},
            timeout=60
        )

        response.raise_for_status()
        return response.json()

    except Exception as e:
        st.error(f"❌ ML Processing Error: {str(e)}")
        return None
