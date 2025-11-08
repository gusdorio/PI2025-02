"""
Data Processing Pipeline
========================
Orchestrates the complete data processing workflow from receipt to storage.

Architecture:
    HTTP Request → DataPipeline → [Validation → Processing → Storage → Response]

This module implements a clear pipeline pattern with stages that can be
independently extended and tested. Each stage has single responsibility.

Current Implementation:
- Validation: Complete
- ML Processing: Placeholder (to be implemented)
- Storage: MongoDB integration
- Response: Standardized format

Future Enhancements:
- Async processing
- Parallel ML model execution
- Result caching
- Stream processing for large datasets
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple, Any, List
from enum import Enum
import traceback
import hashlib
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import MongoDB models and connection
from models.database import get_database_connection, MongoDBConnection
from models.mongodb import PipelineRunSummary
from models.dataset import UploadedDataset


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class PipelineStatus(Enum):
    """Pipeline execution status"""
    STARTED = "started"
    VALIDATING = "validating"
    PROCESSING = "processing"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingMode(Enum):
    """Processing modes for different workflows"""
    MINIMAL = "minimal"      # Just store data (current)
    ANALYSIS = "analysis"     # Run basic analysis
    FULL_ML = "full_ml"      # Complete ML pipeline
    CUSTOM = "custom"        # User-defined pipeline


# =============================================================================
# PIPELINE RESULT
# =============================================================================

class PipelineResult:
    """
    Standardized result object for pipeline execution.
    Provides consistent interface for all pipeline responses.
    """

    def __init__(self,
                 status: str,
                 message: str,
                 batch_id: Optional[str] = None,
                 data: Optional[Dict] = None,
                 errors: Optional[List[Dict]] = None):
        """
        Initialize pipeline result.

        Parameters:
        -----------
        status : str
            Pipeline status (completed, failed, etc.)
        message : str
            Human-readable message
        batch_id : str, optional
            Unique identifier for this processing batch
        data : dict, optional
            Additional result data
        errors : list, optional
            List of error dictionaries if failed
        """
        self.status = status
        self.message = message
        self.batch_id = batch_id
        self.data = data or {}
        self.errors = errors or []
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict:
        """Convert to dictionary for HTTP response"""
        result = {
            "status": self.status,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "batch_id": self.batch_id
        }

        # Add data if present
        if self.data:
            result.update(self.data)

        # Add errors if present
        if self.errors:
            result["errors"] = self.errors

        return result

    @property
    def is_success(self) -> bool:
        """Check if pipeline succeeded"""
        return self.status == PipelineStatus.COMPLETED.value


# =============================================================================
# DATA VALIDATOR
# =============================================================================

class DataValidator:
    """
    Validates incoming data structure and content.
    First stage of the pipeline - ensures data quality.
    """

    def __init__(self):
        """Initialize validator with validation rules"""
        self.required_fields = ['filename', 'data', 'metadata']
        self.required_metadata = ['row_count', 'column_count', 'column_names']
        self.max_rows = 1000000  # 1M rows max
        self.max_columns = 1000   # 1K columns max

    def validate(self, data: Dict) -> Tuple[bool, Optional[str], Optional[UploadedDataset]]:
        """
        Validate incoming data and create UploadedDataset if valid.

        Parameters:
        -----------
        data : dict
            Raw data from HTTP request

        Returns:
        --------
        tuple : (is_valid, error_message, dataset_object)
        """
        try:
            # Check for transformation metadata (from dashboard pipeline)
            transform_info = data.get('transform_info', {})
            if transform_info:
                transform_status = transform_info.get('transform_status', 'unknown')
                transform_mode = transform_info.get('mode', 'unknown')
                print(f"[VALIDATION] Received data with transformation info")
                print(f"[VALIDATION] Transform status: {transform_status}")
                print(f"[VALIDATION] Transform mode: {transform_mode}")

                if transform_status == 'completed':
                    print(f"[VALIDATION] Data has been transformed")
                    if 'original_shape' in transform_info:
                        print(f"[VALIDATION] Original shape: {transform_info['original_shape']}")
                    if 'transformed_shape' in transform_info:
                        print(f"[VALIDATION] Transformed shape: {transform_info['transformed_shape']}")
                    # Future: Apply inverse transformation if needed
                    # For now, continue with validation as normal
                elif transform_status == 'skipped':
                    print(f"[VALIDATION] Transformation was skipped (pass-through mode)")

            # Check required fields
            missing_fields = [f for f in self.required_fields if f not in data]
            if missing_fields:
                return False, f"Missing required fields: {', '.join(missing_fields)}", None

            # Validate metadata structure
            metadata = data.get('metadata', {})
            if not isinstance(metadata, dict):
                return False, "Metadata must be a dictionary", None

            missing_metadata = [f for f in self.required_metadata if f not in metadata]
            if missing_metadata:
                return False, f"Missing metadata fields: {', '.join(missing_metadata)}", None

            # Validate data size limits
            row_count = metadata.get('row_count', 0)
            col_count = metadata.get('column_count', 0)

            if row_count > self.max_rows:
                return False, f"Dataset too large: {row_count} rows exceeds limit of {self.max_rows}", None

            if col_count > self.max_columns:
                return False, f"Too many columns: {col_count} exceeds limit of {self.max_columns}", None

            # Parse DataFrame from JSON
            try:
                df = pd.read_json(data['data'])
            except Exception as e:
                return False, f"Failed to parse DataFrame: {str(e)}", None

            # Verify parsed data matches metadata
            actual_rows = len(df)
            actual_cols = len(df.columns)

            if actual_rows != row_count:
                return False, f"Row count mismatch: metadata says {row_count}, actual is {actual_rows}", None

            if actual_cols != col_count:
                return False, f"Column count mismatch: metadata says {col_count}, actual is {actual_cols}", None

            # Create UploadedDataset object
            dataset = UploadedDataset(
                dataframe=df,
                filename=data['filename'],
                file_type=metadata.get('file_type', 'unknown')
            )

            return True, None, dataset

        except Exception as e:
            return False, f"Validation error: {str(e)}", None


# =============================================================================
# ML PROCESSOR (PLACEHOLDER)
# =============================================================================

class MLProcessor:
    """
    Machine Learning processing stage.
    Currently a placeholder - will integrate AutoMLSelector in future.
    """

    def __init__(self, mode: ProcessingMode = ProcessingMode.MINIMAL):
        """
        Initialize ML processor.

        Parameters:
        -----------
        mode : ProcessingMode
            Processing mode (minimal, analysis, full_ml)
        """
        self.mode = mode
        self.supported_algorithms = [
            'Linear Regression', 'Random Forest', 'Gradient Boosting',
            'Logistic Regression', 'SVM', 'Decision Tree'
        ]

    def process(self, dataset: UploadedDataset, batch_id: str) -> Dict:
        """
        Process dataset through ML pipeline.

        Currently returns placeholder results. Future implementation will:
        1. Detect problem type (regression/classification)
        2. Run AutoMLSelector
        3. Generate predictions
        4. Calculate metrics

        Parameters:
        -----------
        dataset : UploadedDataset
            Validated dataset object
        batch_id : str
            Unique batch identifier

        Returns:
        --------
        dict : Processing results (currently minimal)
        """

        # Current implementation: Just return metadata
        if self.mode == ProcessingMode.MINIMAL:
            return {
                "ml_status": "skipped",
                "mode": self.mode.value,
                "reason": "Minimal mode - ML processing disabled",
                "available_algorithms": self.supported_algorithms,
                "future_capabilities": {
                    "auto_ml": "Will automatically select best algorithm",
                    "cross_validation": "5-fold cross-validation",
                    "metrics": "R², RMSE, accuracy, F1-score",
                    "feature_importance": "For tree-based models"
                }
            }

        # Placeholder for ANALYSIS mode
        elif self.mode == ProcessingMode.ANALYSIS:
            # Future: Basic statistical analysis
            return {
                "ml_status": "analysis_only",
                "statistics": {
                    "rows": dataset.row_count,
                    "columns": dataset.column_count,
                    "missing_values": dataset.has_missing_values
                },
                "note": "Full analysis not yet implemented"
            }

        # Placeholder for FULL_ML mode
        elif self.mode == ProcessingMode.FULL_ML:
            # Future: Complete ML pipeline with AutoMLSelector
            return {
                "ml_status": "not_implemented",
                "message": "Full ML pipeline coming soon",
                "will_include": {
                    "model_selection": "Best model from 8+ algorithms",
                    "hyperparameter_tuning": "Grid search optimization",
                    "predictions": "On test set",
                    "model_persistence": "Save trained model"
                }
            }

        return {"ml_status": "unknown_mode"}


# =============================================================================
# DATA STORAGE
# =============================================================================

class DataStorage:
    """
    Handles all database operations for the pipeline.
    Manages MongoDB collections and data persistence.
    """

    def __init__(self, db_connection: Optional[MongoDBConnection] = None):
        """
        Initialize storage handler.

        Parameters:
        -----------
        db_connection : MongoDBConnection, optional
            MongoDB connection instance
        """
        self.db = db_connection or get_database_connection()
        self.datasets_collection = self.db.get_collection('datasets')
        self.pipeline_runs_collection = self.db.get_collection('pipeline_runs')
        self.ml_results_collection = self.db.get_collection('ml_results')

    def store_dataset(self, dataset: UploadedDataset, batch_id: str) -> Tuple[bool, Optional[str]]:
        """
        Store dataset in MongoDB.

        Parameters:
        -----------
        dataset : UploadedDataset
            Dataset to store
        batch_id : str
            Unique batch identifier

        Returns:
        --------
        tuple : (success, error_message)
        """
        try:
            # Prepare document for storage
            # IMPORTANT: Convert numpy types to native Python types for MongoDB BSON encoder
            document = {
                "_id": batch_id,  # Use batch_id as document ID
                "filename": dataset.filename,
                "file_type": dataset.file_type,
                "upload_timestamp": dataset.upload_timestamp,
                "metadata": convert_numpy_types(dataset.get_summary()),  # Clean numpy types
                "data": convert_numpy_types(dataset.raw_dataframe.to_dict('records')),  # Clean numpy types
                "row_count": int(dataset.row_count),  # Ensure native int
                "column_count": int(dataset.column_count),  # Ensure native int
                "column_names": list(dataset.column_names),  # Ensure native list
                "has_missing_values": bool(dataset.has_missing_values),  # Ensure native bool
                "created_at": datetime.now(),
                "processing_status": "stored"
            }

            # Insert into MongoDB
            result = self.datasets_collection.insert_one(document)

            if result.inserted_id:
                print(f"[STORAGE] Dataset stored successfully with ID: {batch_id}")
                return True, None
            else:
                return False, "Failed to insert dataset into database"

        except Exception as e:
            error_msg = f"Storage error: {str(e)}"
            print(f"[STORAGE ERROR] {error_msg}")
            return False, error_msg

    def store_pipeline_run(self, batch_id: str, status: str, summary: Dict, errors: List = None) -> bool:
        """
        Store pipeline run summary in MongoDB.

        Parameters:
        -----------
        batch_id : str
            Unique batch identifier
        status : str
            Pipeline status
        summary : dict
            Execution summary
        errors : list, optional
            List of errors if any

        Returns:
        --------
        bool : Success status
        """
        try:
            # Clean summary and errors of numpy types
            clean_summary = convert_numpy_types(summary)
            clean_errors = convert_numpy_types(errors or [])

            # Create PipelineRunSummary object
            run_summary = PipelineRunSummary(
                batch_id=batch_id,
                status=status,
                summary=clean_summary,
                errors=clean_errors,
                metadata={
                    "processor_version": "1.0.0",
                    "storage_version": "1.0.0"
                }
            )

            # Update completed timestamp if completed
            if status == PipelineStatus.COMPLETED.value:
                run_summary.completed_at = datetime.now()

            # Store in MongoDB
            result = self.pipeline_runs_collection.insert_one(run_summary.to_dict())

            return result.inserted_id is not None

        except Exception as e:
            print(f"[STORAGE ERROR] Failed to store pipeline run: {str(e)}")
            return False

    def get_dataset(self, batch_id: str) -> Optional[Dict]:
        """
        Retrieve dataset from MongoDB.

        Parameters:
        -----------
        batch_id : str
            Unique batch identifier

        Returns:
        --------
        dict : Dataset document or None
        """
        try:
            return self.datasets_collection.find_one({"_id": batch_id})
        except Exception as e:
            print(f"[STORAGE ERROR] Failed to retrieve dataset: {str(e)}")
            return None


# =============================================================================
# DATA PIPELINE (MAIN ORCHESTRATOR)
# =============================================================================

class DataPipeline:
    """
    Main pipeline orchestrator that coordinates all processing stages.
    Implements the complete data flow from receipt to storage.
    """

    def __init__(self,
                 mode: ProcessingMode = ProcessingMode.MINIMAL,
                 db_connection: Optional[MongoDBConnection] = None):
        """
        Initialize pipeline with all components.

        Parameters:
        -----------
        mode : ProcessingMode
            Processing mode for ML stage
        db_connection : MongoDBConnection, optional
            Database connection to use
        """
        self.mode = mode
        self.validator = DataValidator()
        self.processor = MLProcessor(mode=mode)
        self.storage = DataStorage(db_connection)

    def generate_batch_id(self, data: Dict) -> str:
        """
        Generate unique batch ID for this processing run.

        Parameters:
        -----------
        data : dict
            Input data

        Returns:
        --------
        str : Unique batch identifier
        """
        # Create unique ID from filename and timestamp
        timestamp = datetime.now().isoformat()
        filename = data.get('filename', 'unknown')

        # Create hash for uniqueness
        content = f"{filename}_{timestamp}".encode()
        hash_digest = hashlib.md5(content).hexdigest()[:8]

        # Format: YYYYMMDD_HHMMSS_filename_hash
        batch_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename.replace('.', '_')}_{hash_digest}"

        return batch_id

    def execute(self, data: Dict) -> PipelineResult:
        """
        Execute complete pipeline with error handling and status tracking.

        Flow:
        1. Generate batch ID
        2. Validate data
        3. Process through ML (placeholder)
        4. Store in MongoDB
        5. Return standardized result

        Parameters:
        -----------
        data : dict
            Raw data from HTTP request

        Returns:
        --------
        PipelineResult : Standardized result object
        """
        batch_id = self.generate_batch_id(data)
        errors = []

        print("\n" + "=" * 60)
        print(f"[PIPELINE] Starting execution - Batch ID: {batch_id}")
        print(f"[PIPELINE] Mode: {self.mode.value}")
        print("=" * 60)

        try:
            # =========================================================
            # STAGE 1: VALIDATION
            # =========================================================
            print("\n[STAGE 1] Data Validation")
            print("-" * 40)

            is_valid, error_msg, dataset = self.validator.validate(data)

            if not is_valid:
                print(f"[VALIDATION] ❌ Failed: {error_msg}")
                self.storage.store_pipeline_run(
                    batch_id=batch_id,
                    status=PipelineStatus.FAILED.value,
                    summary={"stage": "validation", "error": error_msg},
                    errors=[{"stage": "validation", "message": error_msg}]
                )
                return PipelineResult(
                    status=PipelineStatus.FAILED.value,
                    message=f"Validation failed: {error_msg}",
                    batch_id=batch_id,
                    errors=[{"stage": "validation", "message": error_msg}]
                )

            print(f"[VALIDATION] ✅ Passed")
            print(f"[VALIDATION] Dataset: {dataset.row_count} rows × {dataset.column_count} columns")

            # =========================================================
            # STAGE 2: ML PROCESSING (PLACEHOLDER)
            # =========================================================
            print("\n[STAGE 2] ML Processing")
            print("-" * 40)

            ml_results = self.processor.process(dataset, batch_id)
            print(f"[ML PROCESSOR] Mode: {self.mode.value}")
            print(f"[ML PROCESSOR] Status: {ml_results.get('ml_status', 'unknown')}")

            if self.mode == ProcessingMode.MINIMAL:
                print("[ML PROCESSOR] ⏭️ Skipped (minimal mode)")
            else:
                print(f"[ML PROCESSOR] Results: {ml_results}")

            # =========================================================
            # STAGE 3: DATA STORAGE
            # =========================================================
            print("\n[STAGE 3] Data Storage")
            print("-" * 40)

            storage_success, storage_error = self.storage.store_dataset(dataset, batch_id)

            if not storage_success:
                print(f"[STORAGE] ❌ Failed: {storage_error}")
                self.storage.store_pipeline_run(
                    batch_id=batch_id,
                    status=PipelineStatus.FAILED.value,
                    summary={"stage": "storage", "error": storage_error},
                    errors=[{"stage": "storage", "message": storage_error}]
                )
                return PipelineResult(
                    status=PipelineStatus.FAILED.value,
                    message=f"Storage failed: {storage_error}",
                    batch_id=batch_id,
                    errors=[{"stage": "storage", "message": storage_error}]
                )

            print(f"[STORAGE] ✅ Dataset stored in MongoDB")
            print(f"[STORAGE] Collection: datasets")
            print(f"[STORAGE] Document ID: {batch_id}")

            # =========================================================
            # STAGE 4: RECORD PIPELINE RUN
            # =========================================================
            print("\n[STAGE 4] Recording Pipeline Run")
            print("-" * 40)

            # Prepare summary (ensure all values are native Python types)
            summary = {
                "filename": dataset.filename,
                "rows_processed": int(dataset.row_count),  # Convert numpy.int64 to int
                "columns_processed": int(dataset.column_count),  # Convert numpy.int64 to int
                "has_missing_values": bool(dataset.has_missing_values),  # Convert numpy.bool_ to bool
                "ml_status": ml_results.get('ml_status', 'not_run'),
                "processing_mode": self.mode.value,
                "storage_status": "success"
            }

            # Include transformation info if present
            transform_info = data.get('transform_info', {})
            if transform_info:
                summary["transform_status"] = transform_info.get('transform_status', 'none')
                summary["transform_mode"] = transform_info.get('mode', 'none')

            # Store pipeline run
            run_stored = self.storage.store_pipeline_run(
                batch_id=batch_id,
                status=PipelineStatus.COMPLETED.value,
                summary=summary
            )

            if run_stored:
                print(f"[PIPELINE] ✅ Run recorded in pipeline_runs collection")
            else:
                print(f"[PIPELINE] ⚠️ Warning: Failed to record pipeline run")

            # =========================================================
            # SUCCESS RESPONSE
            # =========================================================
            print("\n" + "=" * 60)
            print(f"[PIPELINE] ✅ EXECUTION COMPLETED SUCCESSFULLY")
            print(f"[PIPELINE] Batch ID: {batch_id}")
            print("=" * 60 + "\n")

            # Ensure all response data is free of numpy types for JSON serialization
            return PipelineResult(
                status=PipelineStatus.COMPLETED.value,
                message="Dataset processed and stored successfully",
                batch_id=batch_id,
                data={
                    "row_count": int(dataset.row_count),
                    "column_count": int(dataset.column_count),
                    "filename": dataset.filename,
                    "storage_location": "mongodb://datasets",
                    "ml_processing": convert_numpy_types(ml_results),  # Clean ML results
                    "processing_summary": summary  # Already cleaned above
                }
            )

        except Exception as e:
            # =========================================================
            # ERROR HANDLING
            # =========================================================
            error_msg = f"Pipeline execution error: {str(e)}"
            print("\n" + "🔴" * 30)
            print(f"[PIPELINE ERROR] {error_msg}")
            print("🔴" * 30)
            print(f"[TRACE]\n{traceback.format_exc()}")
            print("🔴" * 30 + "\n")

            # Try to store error in pipeline_runs
            try:
                self.storage.store_pipeline_run(
                    batch_id=batch_id,
                    status=PipelineStatus.FAILED.value,
                    summary={"error": error_msg, "stage": "unknown"},
                    errors=[{"message": error_msg, "trace": traceback.format_exc()}]
                )
            except:
                pass  # Silent fail on error storage

            return PipelineResult(
                status=PipelineStatus.FAILED.value,
                message=error_msg,
                batch_id=batch_id,
                errors=[{"message": error_msg, "trace": traceback.format_exc()}]
            )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types.

    This is necessary because MongoDB's BSON encoder and JSON serializer
    cannot handle numpy types (numpy.bool_, numpy.int64, etc.) that come
    from pandas operations.

    Parameters:
    -----------
    obj : any
        Object to convert (dict, list, numpy type, or native type)

    Returns:
    --------
    any : Object with all numpy types converted to native Python types
    """
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


def create_pipeline(mode: str = "minimal") -> DataPipeline:
    """
    Factory function to create pipeline with specified mode.

    Parameters:
    -----------
    mode : str
        Processing mode (minimal, analysis, full_ml)

    Returns:
    --------
    DataPipeline : Configured pipeline instance
    """
    try:
        processing_mode = ProcessingMode(mode)
    except ValueError:
        print(f"[WARNING] Unknown mode '{mode}', using MINIMAL")
        processing_mode = ProcessingMode.MINIMAL

    return DataPipeline(mode=processing_mode)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Test the pipeline with sample data
    """
    print("Testing Data Processing Pipeline")
    print("=" * 60)

    # Create sample test data
    test_data = {
        "filename": "test_data.csv",
        "data": pd.DataFrame({
            "col1": [1, 2, 3, 4, 5],
            "col2": ["A", "B", "C", "D", "E"],
            "col3": [10.5, 20.3, 30.1, 40.7, 50.9]
        }).to_json(orient='records'),
        "metadata": {
            "row_count": 5,
            "column_count": 3,
            "column_names": ["col1", "col2", "col3"],
            "file_type": "csv",
            "has_missing_values": False
        }
    }

    # Create and execute pipeline
    pipeline = create_pipeline("minimal")
    result = pipeline.execute(test_data)

    # Print results
    print("\nPipeline Result:")
    print(json.dumps(result.to_dict(), indent=2, default=str))