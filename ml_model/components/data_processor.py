"""
Data Processing Pipeline
========================
Orchestrates the complete data processing workflow from receipt to storage.

Architecture:
    HTTP Request → DataPipeline → [Validation → Processing → Storage → Response]
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

# IMPORTA O COMPONENTE DE ML
from ml_model.components.analyses_ml import AutoMLSelector


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class PipelineStatus(Enum):
    STARTED = "started"
    VALIDATING = "validating"
    PROCESSING = "processing"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingMode(Enum):
    MINIMAL = "minimal"
    ANALYSIS = "analysis"
    FULL_ML = "full_ml"
    CUSTOM = "custom"


# =============================================================================
# PIPELINE RESULT (Sem alteração)
# =============================================================================
class PipelineResult:
    def __init__(self,
                 status: str,
                 message: str,
                 batch_id: Optional[str] = None,
                 data: Optional[Dict] = None,
                 errors: Optional[List[Dict]] = None):
        self.status = status
        self.message = message
        self.batch_id = batch_id
        self.data = data or {}
        self.errors = errors or []
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict:
        result = {
            "status": self.status,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "batch_id": self.batch_id
        }
        if self.data:
            result.update(self.data)
        if self.errors:
            result["errors"] = self.errors
        return result

    @property
    def is_success(self) -> bool:
        return self.status == PipelineStatus.COMPLETED.value


# =============================================================================
# DATA VALIDATOR
# =============================================================================

class DataValidator:
    def __init__(self):
        self.required_fields = ['filename', 'data', 'metadata']
        self.required_metadata = ['row_count', 'column_count', 'column_names', 'target_column'] 
        self.max_rows = 1000000
        self.max_columns = 1000

    def validate(self, data: Dict) -> Tuple[bool, Optional[str], Optional[UploadedDataset], Optional[str]]:
        """
        Validate incoming data and create UploadedDataset if valid.

        Returns:
        --------
        tuple : (is_valid, error_message, dataset_object, target_column)
        """
        try:
            # (Lógica de info de transformação permanece)
            transform_info = data.get('transform_info', {})
            if transform_info:
                print(f"[VALIDATION] Received data with transform status: {transform_info.get('transform_status')}")

            # Check required fields
            missing_fields = [f for f in self.required_fields if f not in data]
            if missing_fields:
                return False, f"Missing required fields: {', '.join(missing_fields)}", None, None

            # Validate metadata structure
            metadata = data.get('metadata', {})
            if not isinstance(metadata, dict):
                return False, "Metadata must be a dictionary", None, None

            # *** VALIDAÇÃO DA TARGET_COLUMN ***
            missing_metadata = [f for f in self.required_metadata if f not in metadata]
            if missing_metadata:
                return False, f"Missing metadata fields: {', '.join(missing_metadata)}", None, None

            target_column = metadata.get('target_column')
            if not target_column:
                 return False, "Metadata 'target_column' is missing or empty", None, None

            print(f"[VALIDATION] Target Column identified: {target_column}")

            # (Validação de limites e parse do DataFrame permanecem)
            row_count = metadata.get('row_count', 0)
            col_count = metadata.get('column_count', 0)
            if row_count > self.max_rows:
                return False, f"Dataset too large: {row_count} rows", None, None
            if col_count > self.max_columns:
                return False, f"Too many columns: {col_count}", None, None

            try:
                df = pd.read_json(data['data'])
            except Exception as e:
                return False, f"Failed to parse DataFrame: {str(e)}", None, None
            
            # (Verificação de match de metadados permanece)
            if len(df) != row_count:
                return False, f"Row count mismatch", None, None
            if len(df.columns) != col_count:
                return False, f"Column count mismatch", None, None

            # Create UploadedDataset object
            dataset = UploadedDataset(
                dataframe=df,
                filename=data['filename'],
                file_type=metadata.get('file_type', 'unknown')
            )

            # *** RETORNA A TARGET_COLUMN ***
            return True, None, dataset, target_column

        except Exception as e:
            return False, f"Validation error: {str(e)}", None, None


# =============================================================================
# ML PROCESSOR (AGORA FUNCIONAL)
# =============================================================================

class MLProcessor:
    def __init__(self, mode: ProcessingMode = ProcessingMode.MINIMAL):
        self.mode = mode

    def process(self, dataset: UploadedDataset, batch_id: str, target_column: str) -> Tuple[str, Dict, Optional[Dict]]:
        """
        Process dataset through ML pipeline.
        
        Returns:
        --------
        tuple : (ml_status, dashboard_summary, storage_results)
            - ml_status: "completed", "skipped", "failed"
            - dashboard_summary: JSON-serializable metrics for dashboard response
            - storage_results: Detailed results for MongoDB storage
        """

        if self.mode == ProcessingMode.MINIMAL:
            print("[ML PROCESSOR] Mode: MINIMAL - Skipping ML processing")
            return "skipped", {"reason": "Minimal mode - ML processing disabled"}, None

        elif self.mode == ProcessingMode.FULL_ML:
            if not target_column:
                print("[ML PROCESSOR] ❌ ERROR: Full ML mode requires a target column.")
                return "failed", {"reason": "Target column not provided for FULL_ML mode"}, None
            
            print(f"[ML PROCESSOR] Mode: FULL_ML - Starting AutoMLSelector")
            print(f"[ML PROCESSOR] Target Column: {target_column}")

            try:
                # 1. Inicializa e treina o AutoML
                selector = AutoMLSelector(target_column=target_column, cv_folds=5)
                selector.fit(dataset.raw_dataframe)
                
                if selector.best_model_name is None:
                    raise Exception("AutoML failed to find a best model.")

                print(f"[ML PROCESSOR] ✅ AutoML fit complete. Best model: {selector.best_model_name}")

                # 2. Prepara o resumo para o DASHBOARD (JSON)
                # (Apenas as métricas principais)
                results_summary_df = selector.get_results_summary()
                dashboard_summary = {
                    "best_model_name": selector.best_model_name,
                    "problem_type": selector.problem_type,
                    "summary_metrics": results_summary_df.to_dict('records')
                }

                # 3. Prepara os resultados DETALHADOS para o MONGODB
                # (Inclui tudo: matriz de confusão, ranges de erro, etc.)
                storage_results = selector.results
                
                print(f"[ML PROCESSOR] ✅ Results generated for dashboard and storage.")

                return "completed", dashboard_summary, storage_results
            
            except Exception as e:
                print(f"[ML PROCESSOR] ❌ ERROR: AutoMLSelector failed: {str(e)}")
                traceback.print_exc()
                return "failed", {"reason": f"AutoMLSelector failed: {str(e)}"}, None

        # Placeholder for ANALYSIS mode
        elif self.mode == ProcessingMode.ANALYSIS:
            print("[ML PROCESSOR] Mode: ANALYSIS - Not implemented")
            return "skipped", {"reason": "Analysis mode not yet implemented"}, None

        return "unknown_mode", {"reason": "Unknown processing mode"}, None


# =============================================================================
# DATA STORAGE
# =============================================================================

class DataStorage:
    def __init__(self, db_connection: Optional[MongoDBConnection] = None):
        self.db = db_connection or get_database_connection()
        self.datasets_collection = self.db.get_collection('datasets')
        self.pipeline_runs_collection = self.db.get_collection('pipeline_runs')
        self.ml_results_collection = self.db.get_collection('ml_results') #<-- New collection

    def store_dataset(self, dataset: UploadedDataset, batch_id: str) -> Tuple[bool, Optional[str]]:
        """Armazena o dataset bruto (sem alteração)"""
        try:
            document = {
                "_id": batch_id,
                "filename": dataset.filename,
                "file_type": dataset.file_type,
                "upload_timestamp": dataset.upload_timestamp,
                "metadata": convert_numpy_types(dataset.get_summary()),
                "data": convert_numpy_types(dataset.raw_dataframe.to_dict('records')),
                "row_count": int(dataset.row_count),
                "column_count": int(dataset.column_count),
                "column_names": list(dataset.column_names),
                "has_missing_values": bool(dataset.has_missing_values),
                "created_at": datetime.now(),
                "processing_status": "stored"
            }
            result = self.datasets_collection.insert_one(document)
            if result.inserted_id:
                return True, None
            else:
                return False, "Failed to insert dataset"
        except Exception as e:
            error_msg = f"Storage error (datasets): {str(e)}"
            return False, error_msg

    def store_pipeline_run(self, batch_id: str, status: str, summary: Dict, errors: List = None) -> bool:
        """Armazena o resumo da execução (sem alteração)"""
        try:
            clean_summary = convert_numpy_types(summary)
            clean_errors = convert_numpy_types(errors or [])

            run_summary = PipelineRunSummary(
                batch_id=batch_id,
                status=status,
                summary=clean_summary,
                errors=clean_errors,
                metadata={"processor_version": "1.0.1"} # Versão atualizada
            )
            if status == PipelineStatus.COMPLETED.value:
                run_summary.completed_at = datetime.now()

            result = self.pipeline_runs_collection.insert_one(run_summary.to_dict())
            return result.inserted_id is not None
        except Exception as e:
            print(f"[STORAGE ERROR] Failed to store pipeline run: {str(e)}")
            return False

    def store_ml_results(self, batch_id: str, results_data: Dict) -> Tuple[bool, Optional[str]]:
        """
        *** NOVO MÉTODO ***
        Armazena os resultados detalhados de ML na coleção 'ml_results'.
        """
        try:
            # Limpa tipos numpy (matrizes de confusão, etc.)
            clean_results = convert_numpy_types(results_data)
            
            document = {
                "_id": batch_id, # Usa o mesmo ID do batch
                "batch_id": batch_id,
                "results": clean_results,
                "created_at": datetime.now()
            }
            
            result = self.ml_results_collection.insert_one(document)
            
            if result.inserted_id:
                print(f"[STORAGE] ✅ Detailed ML results stored for batch: {batch_id}")
                return True, None
            else:
                return False, "Failed to insert ML results"
                
        except Exception as e:
            error_msg = f"Storage error (ml_results): {str(e)}"
            print(f"[STORAGE ERROR] {error_msg}")
            return False, error_msg

    def get_dataset(self, batch_id: str) -> Optional[Dict]:
        """Recupera dataset (sem alteração)"""
        try:
            return self.datasets_collection.find_one({"_id": batch_id})
        except Exception as e:
            print(f"[STORAGE ERROR] Failed to retrieve dataset: {str(e)}")
            return None


# =============================================================================
# DATA PIPELINE (ORQUESTRADOR PRINCIPAL)
# =============================================================================

class DataPipeline:
    def __init__(self,
                 mode: ProcessingMode = ProcessingMode.MINIMAL,
                 db_connection: Optional[MongoDBConnection] = None):
        self.mode = mode
        self.validator = DataValidator()
        self.processor = MLProcessor(mode=mode)
        self.storage = DataStorage(db_connection)

    def generate_batch_id(self, data: Dict) -> str:
        # (Sem alteração)
        timestamp = datetime.now().isoformat()
        filename = data.get('filename', 'unknown')
        content = f"{filename}_{timestamp}".encode()
        hash_digest = hashlib.md5(content).hexdigest()[:8]
        batch_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename.replace('.', '_')}_{hash_digest}"
        return batch_id

    def execute(self, data: Dict) -> PipelineResult:
        batch_id = self.generate_batch_id(data)
        print("\n" + "=" * 60)
        print(f"[PIPELINE] Starting execution - Batch ID: {batch_id}")
        print(f"[PIPELINE] Mode: {self.mode.value}")
        print("=" * 60)

        try:
            # =========================================================
            # STAGE 1: VALIDATION
            # =========================================================
            print("\n[STAGE 1] Data Validation")
            is_valid, error_msg, dataset, target_column = self.validator.validate(data)

            if not is_valid:
                print(f"[VALIDATION] ❌ Failed: {error_msg}")
                self.storage.store_pipeline_run(
                    batch_id=batch_id, status=PipelineStatus.FAILED.value,
                    summary={"stage": "validation", "error": error_msg},
                    errors=[{"stage": "validation", "message": error_msg}]
                )
                return PipelineResult(
                    status=PipelineStatus.FAILED.value, message=f"Validation failed: {error_msg}",
                    batch_id=batch_id, errors=[{"stage": "validation", "message": error_msg}]
                )

            print(f"[VALIDATION] ✅ Passed")
            print(f"[VALIDATION] Target Column: {target_column}")

            # =========================================================
            # STAGE 2: ML PROCESSING
            # =========================================================
            print("\n[STAGE 2] ML Processing")
            ml_status, ml_summary_dashboard, ml_results_storage = self.processor.process(
                dataset, batch_id, target_column
            )
            
            print(f"[ML PROCESSOR] Status: {ml_status}")

            if ml_status == 'failed':
                print(f"[ML PROCESSOR] ❌ Failed: {ml_summary_dashboard.get('reason')}")
                # Falha de ML não deve parar o armazenamento dos dados brutos
                # Mas registramos o erro
                errors = [{"stage": "ml_processing", "message": ml_summary_dashboard.get('reason')}]
            else:
                print(f"[ML PROCESSOR] ✅ Success")
                errors = []


            # =========================================================
            # STAGE 3: DATA STORAGE (EM MÚLTIPLAS COLEÇÕES)
            # =========================================================
            print("\n[STAGE 3] Data Storage")

            # 3a. Armazena o DATASET BRUTO
            storage_success, storage_error = self.storage.store_dataset(dataset, batch_id)
            if not storage_success:
                print(f"[STORAGE] ❌ Failed (datasets): {storage_error}")
                # Esta é uma falha crítica
                self.storage.store_pipeline_run(
                    batch_id=batch_id, status=PipelineStatus.FAILED.value,
                    summary={"stage": "storage_dataset", "error": storage_error},
                    errors=[{"stage": "storage", "message": storage_error}]
                )
                return PipelineResult(
                    status=PipelineStatus.FAILED.value, message=f"Storage failed: {storage_error}",
                    batch_id=batch_id, errors=[{"stage": "storage", "message": storage_error}]
                )
            print(f"[STORAGE] ✅ Dataset stored in 'datasets' collection")
            
            # 3b. Armazena os RESULTADOS DE ML (se houver)
            if ml_results_storage:
                ml_storage_success, ml_storage_error = self.storage.store_ml_results(
                    batch_id, ml_results_storage
                )
                if not ml_storage_success:
                    print(f"[STORAGE] ⚠️ Warning (ml_results): {ml_storage_error}")
                    # Adiciona ao log de erros, mas não falha a pipeline
                    errors.append({"stage": "storage_ml_results", "message": ml_storage_error})
            else:
                print("[STORAGE] ⏭️ No ML results to store.")

            # =========================================================
            # STAGE 4: RECORD PIPELINE RUN
            # =========================================================
            print("\n[STAGE 4] Recording Pipeline Run")

            # Prepara o resumo final para 'pipeline_runs'
            summary = {
                "filename": dataset.filename,
                "rows_processed": int(dataset.row_count),
                "columns_processed": int(dataset.column_count),
                "has_missing_values": bool(dataset.has_missing_values),
                "ml_status": ml_status,
                "processing_mode": self.mode.value,
                "storage_status": "success",
                "target_column": target_column,
                "transform_status": data.get('transform_info', {}).get('transform_status', 'none'),
                "ml_summary_dashboard": ml_summary_dashboard # <-- Salva o resumo de métricas aqui
            }

            run_stored = self.storage.store_pipeline_run(
                batch_id=batch_id,
                status=PipelineStatus.COMPLETED.value,
                summary=summary,
                errors=errors
            )
            print(f"[PIPELINE] ✅ Run recorded in 'pipeline_runs' collection")

            # =========================================================
            # SUCCESS RESPONSE
            # =========================================================
            print("\n" + "=" * 60)
            print(f"[PIPELINE] ✅ EXECUTION COMPLETED SUCCESSFULLY")
            print("=" * 60 + "\n")

            # Retorna o resumo de métricas para o dashboard
            return PipelineResult(
                status=PipelineStatus.COMPLETED.value,
                message="Dataset processed and stored successfully",
                batch_id=batch_id,
                data={
                    "row_count": int(dataset.row_count),
                    "column_count": int(dataset.column_count),
                    "filename": dataset.filename,
                    "storage_location": "mongodb://datasets",
                    "ml_processing": ml_summary_dashboard, # <-- Retorna o resumo de métricas
                    "processing_summary": summary # Retorna o resumo completo
                }
            )

        except Exception as e:
            # (Tratamento de erro global sem alteração)
            error_msg = f"Pipeline execution error: {str(e)}"
            print(f"[PIPELINE ERROR] {error_msg}")
            traceback.print_exc()
            try:
                self.storage.store_pipeline_run(
                    batch_id=batch_id, status=PipelineStatus.FAILED.value,
                    summary={"error": error_msg, "stage": "unknown"},
                    errors=[{"message": error_msg, "trace": traceback.format_exc()}]
                )
            except: pass
            return PipelineResult(
                status=PipelineStatus.FAILED.value, message=error_msg,
                batch_id=batch_id, errors=[{"message": error_msg, "trace": traceback.format_exc()}]
            )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def convert_numpy_types(obj):
    """Recursively convert numpy types (sem alteração)"""
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
    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    else:
        return obj


def create_pipeline(mode: str = "minimal") -> DataPipeline:
    """Factory function (sem alteração)"""
    try:
        processing_mode = ProcessingMode(mode)
    except ValueError:
        print(f"[WARNING] Unknown mode '{mode}', using MINIMAL")
        processing_mode = ProcessingMode.MINIMAL
    return DataPipeline(mode=processing_mode)


# =============================================================================
# TESTING (Sem alteração)
# =============================================================================
if __name__ == "__main__":
    print("Testing Data Processing Pipeline (com target_column)")
    print("=" * 60)
    test_data = {
        "filename": "test_data.csv",
        "data": pd.DataFrame({
            "col1": [1, 2, 3, 4, 5],
            "col2": ["A", "B", "C", "D", "E"],
            "col3": [10.5, 20.3, 30.1, 40.7, 50.9] # Target
        }).to_json(orient='records'),
        "metadata": {
            "row_count": 5,
            "column_count": 3,
            "column_names": ["col1", "col2", "col3"],
            "file_type": "csv",
            "has_missing_values": False,
            "target_column": "col3" # <-- Adicionando o target
        }
    }
    pipeline = create_pipeline("full_ml") # <-- Testando com FULL_ML
    result = pipeline.execute(test_data)
    print("\nPipeline Result:")
    print(json.dumps(result.to_dict(), indent=2, default=str))