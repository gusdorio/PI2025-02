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
    
def request_prediction(batch_id: str, input_data: Dict) -> Optional[Dict]:
    """
    Envia uma nova amostra para o serviço de ML para previsão on-demand.

    Parameters:
    -----------
    batch_id : str
        O ID do batch (modelo) a ser usado para a previsão.
    input_data : dict
        Um dicionário de {feature_name: value} para a nova amostra.

    Returns:
    --------
    dict : Resultado da previsão
    None : Se a requisição falhar
    """
    try:
        payload = {
            "batch_id": batch_id,
            "input_data": input_data
        }
        
        response = requests.post(
            'http://ml-model:5000/predict', # <-- NOVO ENDPOINT
            json=payload,
            timeout=30, # Previsões devem ser rápidas
            headers={'Content-Type': 'application/json'}
        )

        response.raise_for_status()
        return response.json()

    except requests.ConnectionError as e:
        print(f"[ML CLIENT ERROR] Prediction connection failed: {str(e)}")
        return None
    except requests.Timeout:
        print(f"[ML CLIENT ERROR] Prediction request timeout after 30s")
        return None
    except requests.HTTPError as e:
        print(f"[ML CLIENT ERROR] Prediction HTTP {e.response.status_code}: {str(e)}")
        return None
    except Exception as e:
        print(f"[ML CLIENT ERROR] Prediction unexpected error: {str(e)}")
        return None
    
def request_batch_prediction(batch_id: str, batch_data_json: str, mode: str, target_column_name: Optional[str]) -> Optional[Dict]:
    """
    Envia um DataFrame JSON para o serviço de ML para previsão em lote.

    Parameters:
    -----------
    batch_id : str
        O ID do modelo (batch) a ser usado.
    batch_data_json : str
        O DataFrame (com features e/ou labels) serializado como JSON.
    mode : str
        'testar' (se o JSON contém a coluna alvo) ou 'prever' (se não contém).
    target_column_name : str, optional
        O nome da coluna alvo (necessário se mode='testar').

    Returns:
    --------
    dict : Dicionário com 'predictions' e (opcionalmente) 'metrics'.
    None : Se a requisição falhar.
    """
    try:
        payload = {
            "batch_id": batch_id,
            "batch_data_json": batch_data_json,
            "mode": mode,
            "target_column_name": target_column_name
        }
        
        # Usar um timeout maior para previsões em lote
        response = requests.post(
            'http://ml-model:5000/batch_predict', # <-- NOVO ENDPOINT
            json=payload,
            timeout=300, # 5 minutos para lotes maiores
            headers={'Content-Type': 'application/json'}
        )

        response.raise_for_status()
        return response.json()

    except requests.ConnectionError as e:
        print(f"[ML CLIENT ERROR] Batch prediction connection failed: {str(e)}")
        return None
    except requests.Timeout:
        print(f"[ML CLIENT ERROR] Batch prediction request timeout after 300s")
        return None
    except requests.HTTPError as e:
        print(f"[ML CLIENT ERROR] Batch prediction HTTP {e.response.status_code}: {str(e)}")
        return None
    except Exception as e:
        print(f"[ML CLIENT ERROR] Batch prediction unexpected error: {str(e)}")
        return None