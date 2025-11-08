"""
Data Upload Pipeline
====================
Orchestrates the data upload workflow from file validation through ML service communication.

This module follows the same pipeline pattern as ml_model/components/data_processor.py,
providing a clean, stage-based approach to data handling.

Architecture:
    DataUploadPipeline → [Validation → Transformation → ML Communication]
                                        ↓
                                  (SKIPPED for now)

The transformation stage is prepared for future integration with models/trasformator.py
but currently operates in pass-through mode.
"""

import json
import numpy as np
import requests
from enum import Enum
from typing import Dict, Optional, Tuple, Any
from datetime import datetime

# Import dataset model
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.dataset import UploadedDataset


# =============================================================================
# ENUMS AND CONFIGURATION
# =============================================================================

class TransformMode(Enum):
    """Transformation modes for upload pipeline"""
    NONE = "none"           # Skip transformation (current default)
    BASIC = "basic"         # Basic cleaning only (future)
    FULL = "full"           # Complete trasformator.py pipeline (future)


class PipelineStatus(Enum):
    """Pipeline execution status"""
    STARTED = "started"
    VALIDATING = "validating"
    TRANSFORMING = "transforming"
    SENDING = "sending"
    COMPLETED = "completed"
    FAILED = "failed"
    ERROR = "error"


# =============================================================================
# PIPELINE RESULT
# =============================================================================

class UploadPipelineResult:
    """
    Standardized result object for upload pipeline execution.

    Provides consistent interface for all pipeline responses, similar to
    ml_model's PipelineResult pattern.
    """

    def __init__(self,
                 status: str,
                 message: str,
                 dataset: Optional[UploadedDataset] = None,
                 ml_response: Optional[Dict] = None,
                 errors: Optional[list] = None,
                 metadata: Optional[Dict] = None):
        """
        Initialize pipeline result.

        Parameters:
        -----------
        status : str
            Pipeline status (completed, failed, error)
        message : str
            Human-readable message
        dataset : UploadedDataset, optional
            The processed dataset
        ml_response : dict, optional
            Response from ML service
        errors : list, optional
            List of error dictionaries
        metadata : dict, optional
            Additional metadata about processing
        """
        self.status = status
        self.message = message
        self.dataset = dataset
        self.ml_response = ml_response
        self.errors = errors or []
        self.metadata = metadata or {}
        self.timestamp = datetime.now()

    @property
    def is_success(self) -> bool:
        """Check if pipeline succeeded"""
        return self.status == PipelineStatus.COMPLETED.value

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging or API responses"""
        result = {
            "status": self.status,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

        if self.ml_response:
            result["ml_response"] = self.ml_response

        if self.errors:
            result["errors"] = self.errors

        return result


# =============================================================================
# DATA TRANSFORMER
# =============================================================================

class DataTransformer:
    """
    Transformation stage for data preprocessing.

    Currently operates in pass-through mode (NONE), but is prepared for
    future integration with models/trasformator.py for data compaction
    and normalization.
    """

    def __init__(self, mode: TransformMode = TransformMode.NONE):
        """
        Initialize transformer with specified mode.

        Parameters:
        -----------
        mode : TransformMode
            Transformation mode (NONE, BASIC, or FULL)
        """
        self.mode = mode

    def transform(self, dataset: UploadedDataset) -> Tuple[UploadedDataset, Dict]:
        """
        Transform dataset using specified transformation mode.

        Currently passes through the data unchanged (NONE mode).
        Future implementation will apply trasformator.py pipeline for
        data cleaning, normalization, and dimensionality reduction.

        Parameters:
        -----------
        dataset : UploadedDataset
            Input dataset to transform

        Returns:
        --------
        tuple : (transformed_dataset, transform_metadata)
            The dataset (possibly transformed) and metadata about the transformation
        """

        if self.mode == TransformMode.NONE:
            # Pass through unchanged - no transformation
            print("[TRANSFORMER] Mode: NONE - Skipping transformation")
            return dataset, {
                "transform_status": "skipped",
                "mode": self.mode.value,
                "original_shape": (dataset.row_count, dataset.column_count)
            }

        elif self.mode == TransformMode.BASIC:
            # Future: Basic cleaning without dimensionality reduction
            print("[TRANSFORMER] Mode: BASIC - Not implemented yet")
            return dataset, {
                "transform_status": "not_implemented",
                "mode": self.mode.value,
                "message": "Basic transformation will be implemented in future"
            }

        elif self.mode == TransformMode.FULL:
            # Future: Full trasformator.py pipeline
            # from models.trasformator import data_transformation
            #
            # transformer = data_transformation(
            #     raw_data=dataset.raw_dataframe,
            #     normalization_type='z_score',
            #     compactation_method='umap',
            #     umap_n_components=10
            # )
            #
            # # Run complete pipeline
            # processed_df = transformer.run_pipeline(
            #     clean_method='drop_na',
            #     outlier_method='iqr',
            #     categorical_method='one_hot'
            # )
            #
            # # Create new dataset with transformed data
            # transformed_dataset = UploadedDataset(
            #     dataframe=processed_df,
            #     filename=f"{dataset.filename}_transformed",
            #     file_type=dataset.file_type
            # )
            #
            # return transformed_dataset, {
            #     "transform_status": "completed",
            #     "mode": self.mode.value,
            #     "original_shape": (dataset.row_count, dataset.column_count),
            #     "transformed_shape": (transformed_dataset.row_count, transformed_dataset.column_count),
            #     "normalization": "z_score",
            #     "compactation": "umap"
            # }

            print("[TRANSFORMER] Mode: FULL - Not implemented yet")
            return dataset, {
                "transform_status": "not_implemented",
                "mode": self.mode.value,
                "message": "Full transformation pipeline will integrate trasformator.py"
            }

        # Unknown mode
        return dataset, {
            "transform_status": "error",
            "mode": self.mode.value,
            "message": f"Unknown transformation mode: {self.mode}"
        }


# =============================================================================
# DATA SERIALIZER
# =============================================================================

class DataSerializer:
    """
    Handles data serialization for ML service communication.

    Includes numpy type conversion (moved from ml_client.py) and
    payload preparation with transformation metadata.
    """

    @staticmethod
    def convert_numpy_types(obj: Any) -> Any:
        """
        Recursively convert numpy types to native Python types.

        This is necessary for JSON serialization as numpy types
        (numpy.bool_, numpy.int64, etc.) are not JSON serializable.

        Parameters:
        -----------
        obj : any
            Object to convert (dict, list, numpy type, or native type)

        Returns:
        --------
        any : Object with all numpy types converted to native Python types
        """
        if isinstance(obj, dict):
            return {k: DataSerializer.convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [DataSerializer.convert_numpy_types(item) for item in obj]
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

    def serialize(self, dataset: UploadedDataset, transform_metadata: Dict) -> Dict:
        """
        Serialize dataset for ML service transmission.

        Creates a JSON-serializable payload including dataset data,
        metadata, and transformation information.

        Parameters:
        -----------
        dataset : UploadedDataset
            Dataset to serialize
        transform_metadata : dict
            Metadata about any transformations applied

        Returns:
        --------
        dict : JSON-serializable payload for ML service
        """
        # Get dataset summary and clean numpy types
        metadata = self.convert_numpy_types(dataset.get_summary())

        # Prepare payload
        payload = {
            "filename": dataset.filename,
            "data": dataset.raw_dataframe.to_json(orient='records'),
            "metadata": metadata,
            "transform_info": transform_metadata  # Include transformation metadata
        }

        return payload


# =============================================================================
# ML SERVICE CLIENT
# =============================================================================

class MLServiceClient:
    """
    Handles communication with the ML model service.

    Configurable client for sending datasets to the ML service,
    with proper error handling and response processing.
    """

    def __init__(self,
                 endpoint: str = 'http://ml-model:5000/process',
                 timeout: int = 60):
        """
        Initialize ML service client.

        Parameters:
        -----------
        endpoint : str
            ML service endpoint URL
        timeout : int
            Request timeout in seconds
        """
        self.endpoint = endpoint
        self.timeout = timeout
        self.serializer = DataSerializer()

    def send_dataset(self, dataset: UploadedDataset, transform_metadata: Dict) -> Optional[Dict]:
        """
        Send dataset to ML service for processing.

        Parameters:
        -----------
        dataset : UploadedDataset
            Dataset to send
        transform_metadata : dict
            Metadata about transformations applied

        Returns:
        --------
        dict : Response from ML service
        None : If request fails
        """
        try:
            # Serialize dataset
            payload = self.serializer.serialize(dataset, transform_metadata)

            # Calculate payload size for logging
            try:
                payload_size_kb = len(json.dumps(payload)) / 1024
                print(f"[ML CLIENT] Payload size: {payload_size_kb:.2f} KB")
            except:
                pass

            print(f"[ML CLIENT] Sending to {self.endpoint}")
            print(f"[ML CLIENT] Timeout: {self.timeout} seconds")

            # Send request
            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )

            print(f"[ML CLIENT] Response status: {response.status_code}")
            response.raise_for_status()

            # Parse and return response
            result = response.json()
            print(f"[ML CLIENT] Success - Status: {result.get('status', 'unknown')}")

            return result

        except requests.ConnectionError as e:
            print(f"[ML CLIENT ERROR] Connection failed: {str(e)}")
            return None

        except requests.Timeout:
            print(f"[ML CLIENT ERROR] Request timeout after {self.timeout}s")
            return None

        except requests.HTTPError as e:
            print(f"[ML CLIENT ERROR] HTTP {e.response.status_code}: {str(e)}")
            return None

        except json.JSONDecodeError as e:
            print(f"[ML CLIENT ERROR] Invalid JSON response: {str(e)}")
            return None

        except Exception as e:
            print(f"[ML CLIENT ERROR] Unexpected error: {str(e)}")
            return None


# =============================================================================
# DATA UPLOAD PIPELINE (MAIN ORCHESTRATOR)
# =============================================================================

class DataUploadPipeline:
    """
    Main orchestrator for the data upload workflow.

    Implements a stage-based pipeline pattern similar to ml_model's DataPipeline,
    orchestrating the flow from file validation through ML service communication.

    Pipeline stages:
    1. File Validation and Reading (via existing uploader.py)
    2. Data Transformation (currently skipped, ready for trasformator.py)
    3. ML Service Communication
    """

    def __init__(self,
                 transform_mode: TransformMode = TransformMode.NONE,
                 ml_endpoint: Optional[str] = None,
                 ml_timeout: Optional[int] = None):
        """
        Initialize upload pipeline with configuration.

        Parameters:
        -----------
        transform_mode : TransformMode
            Transformation mode for data preprocessing
        ml_endpoint : str, optional
            ML service endpoint (defaults to ml-model:5000)
        ml_timeout : int, optional
            ML request timeout (defaults to 60 seconds)
        """
        self.transform_mode = transform_mode
        self.transformer = DataTransformer(mode=transform_mode)
        self.ml_client = MLServiceClient(
            endpoint=ml_endpoint or 'http://ml-model:5000/process',
            timeout=ml_timeout or 60
        )

    def execute(self, uploaded_file, upload_service) -> UploadPipelineResult:
        """
        Execute the complete upload pipeline.

        Orchestrates the multi-stage process with proper error handling
        and status tracking at each stage.

        Parameters:
        -----------
        uploaded_file : UploadedFile
            Streamlit uploaded file object
        upload_service : UploadService
            Service for file validation and reading

        Returns:
        --------
        UploadPipelineResult : Standardized result object
        """
        print("\n" + "=" * 60)
        print("[UPLOAD PIPELINE] Starting execution")
        print(f"[UPLOAD PIPELINE] Transform mode: {self.transform_mode.value}")
        print("=" * 60)

        try:
            # =========================================================
            # STAGE 1: FILE VALIDATION AND READING
            # =========================================================
            print("\n[STAGE 1] File Validation and Reading")
            print("-" * 40)

            # Use existing upload service for validation and reading
            dataset, errors = upload_service.process_upload(uploaded_file)

            if errors:
                print(f"[VALIDATION] Failed with {len(errors)} error(s)")
                for error in errors:
                    print(f"[VALIDATION] - {error}")

                return UploadPipelineResult(
                    status=PipelineStatus.FAILED.value,
                    message="File validation failed",
                    errors=errors
                )

            print(f"[VALIDATION] Success")
            print(f"[VALIDATION] Dataset: {dataset.row_count} rows × {dataset.column_count} columns")

            # =========================================================
            # STAGE 2: DATA TRANSFORMATION
            # =========================================================
            print("\n[STAGE 2] Data Transformation")
            print("-" * 40)

            transformed_dataset, transform_metadata = self.transformer.transform(dataset)

            print(f"[TRANSFORMER] Status: {transform_metadata.get('transform_status')}")
            print(f"[TRANSFORMER] Mode: {transform_metadata.get('mode')}")

            if self.transform_mode == TransformMode.NONE:
                print("[TRANSFORMER] ⏭️ Skipped (pass-through mode)")
            else:
                print(f"[TRANSFORMER] Metadata: {transform_metadata}")

            # =========================================================
            # STAGE 3: ML SERVICE COMMUNICATION
            # =========================================================
            print("\n[STAGE 3] ML Service Communication")
            print("-" * 40)

            print("[ML CLIENT] Preparing to send dataset")
            ml_response = self.ml_client.send_dataset(transformed_dataset, transform_metadata)

            if not ml_response:
                print("[ML CLIENT] ❌ Communication failed")
                return UploadPipelineResult(
                    status=PipelineStatus.FAILED.value,
                    message="ML service communication failed",
                    dataset=transformed_dataset,
                    metadata=transform_metadata,
                    errors=[{"stage": "ml_communication", "error": "No response from ML service"}]
                )

            print(f"[ML CLIENT] ✅ Response received")
            print(f"[ML CLIENT] Status: {ml_response.get('status')}")
            print(f"[ML CLIENT] Message: {ml_response.get('message')}")

            # =========================================================
            # SUCCESS RESPONSE
            # =========================================================
            print("\n" + "=" * 60)
            print("[UPLOAD PIPELINE] ✅ EXECUTION COMPLETED SUCCESSFULLY")
            print("=" * 60 + "\n")

            return UploadPipelineResult(
                status=PipelineStatus.COMPLETED.value,
                message="Dataset processed successfully",
                dataset=transformed_dataset,
                ml_response=ml_response,
                metadata={
                    "transform_info": transform_metadata,
                    "original_filename": dataset.filename,
                    "rows_processed": dataset.row_count,
                    "columns_processed": dataset.column_count
                }
            )

        except Exception as e:
            # =========================================================
            # ERROR HANDLING
            # =========================================================
            error_msg = f"Pipeline execution error: {str(e)}"
            print("\n" + "🔴" * 30)
            print(f"[PIPELINE ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            print("🔴" * 30 + "\n")

            return UploadPipelineResult(
                status=PipelineStatus.ERROR.value,
                message=error_msg,
                errors=[{
                    "stage": "pipeline",
                    "error": str(e),
                    "type": type(e).__name__
                }]
            )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_upload_pipeline(transform_mode: str = "none") -> DataUploadPipeline:
    """
    Factory function to create upload pipeline with specified mode.

    Parameters:
    -----------
    transform_mode : str
        Transformation mode ('none', 'basic', or 'full')

    Returns:
    --------
    DataUploadPipeline : Configured pipeline instance
    """
    try:
        mode = TransformMode(transform_mode)
    except ValueError:
        print(f"[WARNING] Unknown transform mode '{transform_mode}', using NONE")
        mode = TransformMode.NONE

    return DataUploadPipeline(transform_mode=mode)