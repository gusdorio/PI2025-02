import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import traceback

# Metrics
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

# Imports do projeto
from models.database import get_database_connection
from ml_model.components.analyses_ml import AutoMLSelector

# Mapeia os nomes dos modelos para suas classes reais
MODEL_CLASS_MAP = {
    **AutoMLSelector.REGRESSION_MODELS,
    **AutoMLSelector.CLASSIFICATION_MODELS
}


def handle_prediction(batch_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lida com uma solicitação de previsão em tempo real.

    Esta função recria o melhor modelo treinado (re-treina) usando
    os dados e parâmetros originais e, em seguida, faz uma previsão
    sobre os novos dados de entrada.
    """
    print(f"[PREDICTOR] Nova solicitação de previsão para batch_id: {batch_id}")
    
    try:
        # 1. Conectar ao DB
        db = get_database_connection()
        
        # 2. Obter informações da execução
        print(f"[PREDICTOR] Buscando informações da execução (pipeline_runs)...")
        run_doc = db.get_collection('pipeline_runs').find_one({'batch_id': batch_id})
        if not run_doc:
            raise ValueError(f"Nenhuma execução de pipeline encontrada para o batch_id: {batch_id}")

        summary = run_doc.get('summary', {})
        ml_summary = summary.get('ml_summary_dashboard', {})
        
        target_column = summary.get('target_column')
        best_model_name = ml_summary.get('best_model_name')
        problem_type = ml_summary.get('problem_type')

        if not all([target_column, best_model_name, problem_type]):
            raise ValueError("Informações do modelo (target, best_model, type) ausentes no 'pipeline_runs'")
            
        print(f"[PREDICTOR] Modelo: {best_model_name}, Target: {target_column}, Tipo: {problem_type}")

        # 3. Obter os dados de treinamento originais
        print(f"[PREDICTOR] Buscando dados de treinamento (datasets)...")
        dataset_doc = db.get_collection('datasets').find_one({'_id': batch_id})
        if not dataset_doc:
            raise ValueError(f"Nenhum dataset encontrado para o batch_id: {batch_id}")

        df_train = pd.DataFrame(dataset_doc['data'])
        X_train = df_train.drop(columns=[target_column])
        y_train = df_train[target_column]
        
        # 4. Recriar e treinar o melhor modelo
        print(f"[PREDICTOR] Instanciando e treinando o modelo '{best_model_name}'...")
        
        # Pega a CLASSE do modelo do mapa
        model_class = MODEL_CLASS_MAP.get(best_model_name)
        if model_class is None:
            raise ValueError(f"Nome do modelo '{best_model_name}' não reconhecido.")
        
        # Cria uma NOVA INSTÂNCIA do modelo
        # (Isso é crucial para segurança entre threads/requisições)
        model = model_class()
        
        # Define parâmetros que sabemos que foram usados (para consistência)
        if 'random_state' in model.get_params():
            model.set_params(random_state=42)
        if best_model_name == 'Logistic Regression':
            model.set_params(max_iter=1000)
        if best_model_name in ['Random Forest Regressor', 'Random Forest Classifier', 'Gradient Boosting Regressor', 'Gradient Boosting Classifier']:
            model.set_params(n_estimators=100)
        if best_model_name == 'SVC':
            model.set_params(probability=True)

        model.fit(X_train, y_train)
        print(f"[PREDICTOR] Modelo treinado com sucesso.")
        
        # 5. Formatar a entrada e fazer a previsão
        # Converte a entrada (que pode ser string) para os tipos corretos
        for col in X_train.columns:
            if col in input_data:
                # Tenta converter para numérico se o tipo da coluna de treino for numérico
                if pd.api.types.is_numeric_dtype(X_train[col]):
                    input_data[col] = pd.to_numeric(input_data[col], errors='coerce')
        
        # Garante a ordem correta das colunas
        X_pred = pd.DataFrame([input_data], columns=X_train.columns)
        
        prediction = model.predict(X_pred)[0]
        
        # Converte tipos numpy para JSON serializável
        if isinstance(prediction, np.generic):
            prediction = prediction.item()
        
        result = {"prediction": prediction}
        
        # 6. Obter probabilidades para classificação
        if problem_type == 'classification' and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_pred)[0]
            classes = model.classes_
            result['probabilities'] = {str(cls): prob for cls, prob in zip(classes, probabilities)}
            
        print(f"[PREDICTOR] Previsão: {result}")
        return result

    except Exception as e:
        print(f"[PREDICTOR] ❌ ERRO: {str(e)}")
        print(traceback.format_exc())
        raise

def handle_batch_prediction(batch_id: str, batch_data_json: str, mode: str, target_column_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Lida com uma solicitação de previsão em lote a partir de um JSON de DataFrame.
    """
    print(f"[PREDICTOR-BATCH] Nova solicitação em lote para batch_id: {batch_id} | Modo: {mode}")
    
    try:
        # 1. Carregar Modelo e Dados de Treino (lógica reutilizada)
        db = get_database_connection()
        run_doc = db.get_collection('pipeline_runs').find_one({'batch_id': batch_id})
        if not run_doc:
            raise ValueError(f"Execução não encontrada para batch_id: {batch_id}")

        summary = run_doc.get('summary', {})
        ml_summary = summary.get('ml_summary_dashboard', {})
        
        train_target_col = summary.get('target_column')
        best_model_name = ml_summary.get('best_model_name')
        problem_type = ml_summary.get('problem_type')

        if not all([train_target_col, best_model_name, problem_type]):
            raise ValueError("Informações do modelo ausentes no 'pipeline_runs'")

        dataset_doc = db.get_collection('datasets').find_one({'_id': batch_id})
        if not dataset_doc:
            raise ValueError(f"Dataset de treino não encontrado para batch_id: {batch_id}")

        df_train = pd.DataFrame(dataset_doc['data'])
        X_train = df_train.drop(columns=[train_target_col])
        y_train = df_train[train_target_col]

        # 2. Recriar e treinar o melhor modelo (lógica reutilizada)
        print(f"[PREDICTOR-BATCH] Re-treinando o modelo '{best_model_name}'...")
        model_class = MODEL_CLASS_MAP.get(best_model_name)
        if model_class is None:
            raise ValueError(f"Nome do modelo '{best_model_name}' não reconhecido.")
        
        model = model_class()
        # Define parâmetros de consistência
        if 'random_state' in model.get_params(): model.set_params(random_state=42)
        if best_model_name == 'Logistic Regression': model.set_params(max_iter=1000)
        if 'n_estimators' in model.get_params(): model.set_params(n_estimators=100)
        if best_model_name == 'SVC': model.set_params(probability=True)

        model.fit(X_train, y_train)
        print(f"[PREDICTOR-BATCH] Modelo treinado com sucesso.")

        # 3. Preparar DataFrame de Previsão
        df_new = pd.read_json(batch_data_json, orient='records')
        
        y_true = None
        X_pred = None
        
        if mode == "testar":
            # Modo TESTAR: O usuário forneceu rótulos
            if not target_column_name or target_column_name not in df_new.columns:
                raise ValueError(f"Modo 'Testar' selecionado, mas a coluna alvo '{target_column_name}' não foi encontrada no arquivo enviado.")
            y_true = df_new[target_column_name]
            X_pred = df_new.drop(columns=[target_column_name])
        else:
            # Modo PREVER: Apenas features
            X_pred = df_new

        # 4. Alinhar colunas (MUITO IMPORTANTE)
        # Garante que o DataFrame de previsão tenha exatamente as mesmas colunas
        # que o DataFrame de treinamento, na mesma ordem.
        X_pred_aligned = pd.DataFrame(columns=X_train.columns)
        for col in X_train.columns:
            if col in X_pred.columns:
                X_pred_aligned[col] = X_pred[col]
            else:
                X_pred_aligned[col] = np.nan # Preenche com NaN se faltar (modelo deve lidar)
        
        # Converte tipos para garantir compatibilidade
        for col in X_pred_aligned.columns:
            if pd.api.types.is_numeric_dtype(X_train[col]):
                X_pred_aligned[col] = pd.to_numeric(X_pred_aligned[col], errors='coerce')
        
        X_pred_aligned = X_pred_aligned.fillna(X_train.mean(numeric_only=True)) # Simples imputação pela média de treino

        print(f"[PREDICTOR-BATCH] Colunas alinhadas. Fazendo {len(X_pred_aligned)} previsões...")

        # 5. Fazer Previsões em Lote
        predictions = model.predict(X_pred_aligned)
        
        # Converte tipos numpy
        predictions_list = [p.item() if isinstance(p, np.generic) else p for p in predictions]
        
        result = {"predictions": predictions_list}

        # 6. Calcular Métricas (se modo 'testar')
        if y_true is not None:
            metrics = {}
            if problem_type == 'classification':
                acc = accuracy_score(y_true, predictions)
                metrics = {"accuracy": acc, "total": len(y_true), "correct": int(acc * len(y_true))}
                print(f"[PREDICTOR-BATCH] Teste de Classificação: Accuracy = {acc:.4f}")
            else: # Regressão
                r2 = r2_score(y_true, predictions)
                rmse = np.sqrt(mean_squared_error(y_true, predictions))
                metrics = {"r2_score": r2, "rmse": rmse}
                print(f"[PREDICTOR-BATCH] Teste de Regressão: R² = {r2:.4f} | RMSE = {rmse:.4f}")
            
            result["metrics"] = metrics

        print(f"[PREDICTOR-BATCH] Previsão em lote concluída.")
        return result

    except Exception as e:
        print(f"[PREDICTOR-BATCH] ❌ ERRO: {str(e)}")
        print(traceback.format_exc())
        raise