"""
ML Service Client

Simplified HTTP client for the DataUploadPipeline.
This module now only provides the essential communication function
used by the pipeline architecture.

All data serialization and numpy conversion has been moved to
data_upload_pipeline.py for proper separation of concerns.
"""

import requests
from typing import Dict, Optional


def send_dataset_to_ml_service(payload: Dict) -> Optional[Dict]:
    """
    Send prepared payload to ML service.

    This is a thin HTTP wrapper used by DataUploadPipeline.
    All data preparation and serialization is handled by the pipeline.

    Parameters:
    -----------
    payload : dict
        Pre-serialized payload from DataUploadPipeline

    Returns:
    --------
    dict : Response from ML service
    None : If request fails
    """
    try:
        response = requests.post(
            'http://ml-model:5000/process',
            json=payload,
            timeout=60,
            headers={'Content-Type': 'application/json'}
        )

        response.raise_for_status()
        return response.json()

    except requests.ConnectionError as e:
        print(f"[ML CLIENT ERROR] Connection failed: {str(e)}")
        return None

    except requests.Timeout:
        print(f"[ML CLIENT ERROR] Request timeout after 60s")
        return None

    except requests.HTTPError as e:
        print(f"[ML CLIENT ERROR] HTTP {e.response.status_code}: {str(e)}")
        return None

    except Exception as e:
        print(f"[ML CLIENT ERROR] Unexpected error: {str(e)}")
        return None