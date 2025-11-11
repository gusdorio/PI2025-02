import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold, StratifiedKFold
from sklearn.metrics import (
    mean_squared_error, r2_score, confusion_matrix, 
    classification_report, accuracy_score
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')


class AutoMLSelector:
    """
    Classe para seleção automática do melhor algoritmo de Machine Learning.
    
    Determina automaticamente se o problema é de regressão ou classificação
    e seleciona o melhor modelo com base em métricas apropriadas.
    
    Parâmetros:
    -----------
    target_column : str
        Nome da coluna alvo a ser predita
    objective : str, opcional
        'auto' (padrão): detecta automaticamente
        'regression': força regressão
        'classification': força classificação
    cv_folds : int, opcional
        Número de folds para validação cruzada (padrão: 5)
    test_size : float, opcional
        Proporção dos dados para teste final (padrão: 0.2)
    random_state : int, opcional
        Seed para reprodutibilidade (padrão: 42(Razao da vida ao universokk)) 
    """
    
    def __init__(self, target_column, objective='auto', cv_folds=5, test_size=0.2, random_state=42):
        self.target_column = target_column
        self.objective = objective
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.random_state = random_state
        # Variaveis inicializadas
        self.problem_type = None
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X = None
        self.y = None
        
        # Modelos de regressão
        self.regression_models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=random_state),
            'Lasso Regression': Lasso(random_state=random_state),
            'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=random_state),
            'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, random_state=random_state),
            'SVR': SVR(),
            'Decision Tree Regressor': DecisionTreeRegressor(random_state=random_state),
            'KNN Regressor': KNeighborsRegressor()
        }
        
        # Modelos de classificação
        self.classification_models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
            'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'Gradient Boosting Classifier': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
            'SVC': SVC(probability=True, random_state=random_state),
            'Decision Tree Classifier': DecisionTreeClassifier(random_state=random_state),
            'KNN Classifier': KNeighborsClassifier()
        }
    
    def _detect_problem_type(self, y):
        """
        Detecta automaticamente se é problema de regressão ou classificação.
        """
        if self.objective != 'auto':
            return self.objective
        
        # Verifica se é numérico
        if not np.issubdtype(y.dtype, np.number):
            return 'classification'
        
        # Verifica número de valores únicos
        unique_values = len(np.unique(y))
        total_values = len(y)
        
        # Se menos de 10 valores únicos ou menos de 5% de valores únicos, é classificação
        if unique_values < 10 or (unique_values / total_values) < 0.05:
            return 'classification'
        
        return 'regression'
    
    def _calculate_error_range(self, y_true, y_pred):
        """
        Calcula o intervalo de probabilidade de erro (range de erro).
        Retorna percentis do erro absoluto.
        """
        errors = np.abs(y_true - y_pred)
        return {
            'min': np.min(errors),
            'max': np.max(errors),
            'mean': np.mean(errors),
            'std': np.std(errors),
            'p25': np.percentile(errors, 25),
            'p50': np.percentile(errors, 50),
            'p75': np.percentile(errors, 75),
            'p95': np.percentile(errors, 95)
        }
    
    def _evaluate_regression_model(self, model, model_name):
        """
        Avalia modelo de regressão com validação cruzada completa.
        """
        try:
            # Define estratégia de validação cruzada
            cv_strategy = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            
            # Validação cruzada com múltiplas métricas
            scoring = {
                'neg_mse': 'neg_mean_squared_error',
                'r2': 'r2',
                'neg_mae': 'neg_mean_absolute_error'
            }
            
            cv_results = cross_validate(
                model, self.X, self.y, 
                cv=cv_strategy, 
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1
            )
            
            # Calcula métricas de CV
            cv_rmse = np.sqrt(-cv_results['test_neg_mse'].mean())
            cv_r2 = cv_results['test_r2'].mean()
            cv_mae = -cv_results['test_neg_mae'].mean()
            
            cv_rmse_std = np.sqrt(-cv_results['test_neg_mse']).std()
            cv_r2_std = cv_results['test_r2'].std()
            
            # Treina no conjunto de treino para predições finais
            model.fit(self.X_train, self.y_train)
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            # Métricas no conjunto de teste
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            test_r2 = r2_score(self.y_test, y_pred_test)
            
            # Métricas no conjunto de treino (para detectar overfitting)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
            train_r2 = r2_score(self.y_train, y_pred_train)
            
            # Calcula range de erro com CV
            error_range_test = self._calculate_error_range(self.y_test, y_pred_test)
            
            return {
                # Métricas de Validação Cruzada (principal)
                'cv_rmse': cv_rmse,
                'cv_rmse_std': cv_rmse_std,
                'cv_r2': cv_r2,
                'cv_r2_std': cv_r2_std,
                'cv_mae': cv_mae,
                # Métricas de Teste
                'test_rmse': test_rmse,
                'test_r2': test_r2,
                # Métricas de Treino (detectar overfitting)
                'train_rmse': train_rmse,
                'train_r2': train_r2,
                # Range de erro
                'error_range': error_range_test,
                # Score para seleção (usamos CV R²)
                'score': cv_r2
            }
        except Exception as e:
            print(f"Erro ao avaliar {model_name}: {str(e)}")
            return None
    
    def _evaluate_classification_model(self, model, model_name):
        """
        Avalia modelo de classificação com validação cruzada completa.
        """
        try:
            # Define estratégia de validação cruzada (estratificada para classificação)
            cv_strategy = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            
            # Validação cruzada com múltiplas métricas
            scoring = {
                'accuracy': 'accuracy',
                'precision_macro': 'precision_macro',
                'recall_macro': 'recall_macro',
                'f1_macro': 'f1_macro'
            }
            
            cv_results = cross_validate(
                model, self.X, self.y,
                cv=cv_strategy,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1
            )
            
            # Calcula métricas de CV
            cv_accuracy = cv_results['test_accuracy'].mean()
            cv_accuracy_std = cv_results['test_accuracy'].std()
            cv_precision = cv_results['test_precision_macro'].mean()
            cv_recall = cv_results['test_recall_macro'].mean()
            cv_f1 = cv_results['test_f1_macro'].mean()
            
            # Treina no conjunto de treino para predições finais
            model.fit(self.X_train, self.y_train)
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            # Probabilidades (se disponível)
            y_pred_proba_test = None
            if hasattr(model, 'predict_proba'):
                y_pred_proba_test = model.predict_proba(self.X_test)
            
            # Métricas no conjunto de teste
            test_accuracy = accuracy_score(self.y_test, y_pred_test)
            test_conf_matrix = confusion_matrix(self.y_test, y_pred_test)
            
            # Métricas no conjunto de treino (para detectar overfitting)
            train_accuracy = accuracy_score(self.y_train, y_pred_train)
            
            # Calcula range de confiança
            error_range = None
            if y_pred_proba_test is not None:
                pred_confidence = np.max(y_pred_proba_test, axis=1)
                error_range = {
                    'min_confidence': np.min(pred_confidence),
                    'max_confidence': np.max(pred_confidence),
                    'mean_confidence': np.mean(pred_confidence),
                    'std_confidence': np.std(pred_confidence),
                    'p25_confidence': np.percentile(pred_confidence, 25),
                    'p50_confidence': np.percentile(pred_confidence, 50),
                    'p75_confidence': np.percentile(pred_confidence, 75),
                    'p95_confidence': np.percentile(pred_confidence, 95)
                }
            
            return {
                # Métricas de Validação Cruzada (principal)
                'cv_accuracy': cv_accuracy,
                'cv_accuracy_std': cv_accuracy_std,
                'cv_precision': cv_precision,
                'cv_recall': cv_recall,
                'cv_f1': cv_f1,
                # Métricas de Teste
                'test_accuracy': test_accuracy,
                'confusion_matrix': test_conf_matrix,
                # Métricas de Treino (detectar overfitting)
                'train_accuracy': train_accuracy,
                # Range de confiança
                'error_range': error_range,
                'classification_report': classification_report(self.y_test, y_pred_test),
                # Score para seleção (usamos CV accuracy)
                'score': cv_accuracy
            }
        except Exception as e:
            print(f"Erro ao avaliar {model_name}: {str(e)}")
            return None
    
    def fit(self, data):
        """
        Treina todos os modelos e seleciona o melhor.
        
        Parâmetros:
        -----------
        data : pandas.DataFrame
            DataFrame contendo todas as features e a coluna alvo
        """
        # Separa features e target
        if self.target_column not in data.columns:
            raise ValueError(f"Coluna '{self.target_column}' não encontrada no DataFrame")
        
        self.X = data.drop(columns=[self.target_column])
        self.y = data[self.target_column]
        
        # Detecta tipo de problema
        self.problem_type = self._detect_problem_type(self.y)
        print(f"Tipo de problema detectado: {self.problem_type.upper()}")
        print(f"Target: {self.target_column}")
        print(f"Número de features: {self.X.shape[1]}")
        print(f"Número de amostras: {self.X.shape[0]}")
        print(f"Validação Cruzada: {self.cv_folds} folds")
        print("-" * 60)
        
        # Split dos dados (reserva um conjunto de teste final)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, 
            random_state=self.random_state,
            stratify=self.y if self.problem_type == 'classification' else None
        )
        
        # Seleciona modelos apropriados
        models = self.regression_models if self.problem_type == 'regression' else self.classification_models
        evaluate_func = self._evaluate_regression_model if self.problem_type == 'regression' else self._evaluate_classification_model
        
        # Avalia todos os modelos
        print(f"\nAvaliando {len(models)} modelos...\n")
        best_score = -np.inf
        
        for model_name, model in models.items():
            print(f"Treinando: {model_name}...")
            result = evaluate_func(model, model_name)
            
            if result is not None:
                self.results[model_name] = result
                
                if result['score'] > best_score:
                    best_score = result['score']
                    self.best_model = result['model']
                    self.best_model_name = model_name
        
        print("\n" + "=" * 60)
        print(f"MELHOR MODELO: {self.best_model_name}")
        print("=" * 60)
        
        return self
    
    def get_results_summary(self):
        """
        Retorna um resumo dos resultados de todos os modelos.
        """
        if not self.results:
            print("Nenhum modelo foi treinado ainda. Execute fit() primeiro.")
            return None
        
        summary = []
        
        if self.problem_type == 'regression':
            for model_name, result in self.results.items():
                summary.append({
                    'Modelo': model_name,
                    'CV R² (média)': result['cv_r2'],
                    'CV R² (std)': result['cv_r2_std'],
                    'CV RMSE': result['cv_rmse'],
                    'Test R²': result['test_r2'],
                    'Test RMSE': result['test_rmse'],
                    'Train R²': result['train_r2'],
                    'Overfitting?': 'Sim' if (result['train_r2'] - result['test_r2']) > 0.1 else 'Não'
                })
        else:
            for model_name, result in self.results.items():
                summary.append({
                    'Modelo': model_name,
                    'CV Accuracy (média)': result['cv_accuracy'],
                    'CV Accuracy (std)': result['cv_accuracy_std'],
                    'CV F1-Score': result['cv_f1'],
                    'Test Accuracy': result['test_accuracy'],
                    'Train Accuracy': result['train_accuracy'],
                    'Overfitting?': 'Sim' if (result['train_accuracy'] - result['test_accuracy']) > 0.1 else 'Não'
                })
        
        df_summary = pd.DataFrame(summary)
        
        # Ordena pelo score apropriado
        if self.problem_type == 'regression':
            df_summary = df_summary.sort_values('CV R² (média)', ascending=False)
        else:
            df_summary = df_summary.sort_values('CV Accuracy (média)', ascending=False)
        
        return df_summary
    
    def print_best_model_details(self):
        """
        Imprime detalhes do melhor modelo.
        """
        if self.best_model is None:
            print("Nenhum modelo foi treinado ainda. Execute fit() primeiro.")
            return
        
        print(f"\n{'='*60}")
        print(f"DETALHES DO MELHOR MODELO: {self.best_model_name}")
        print(f"{'='*60}\n")
        
        result = self.results[self.best_model_name]
        
        if self.problem_type == 'regression':
            print(f"Validação Cruzada ({self.cv_folds} folds):")
            print(f"  - R² Score: {result['cv_r2']:.4f} (±{result['cv_r2_std']:.4f})")
            print(f"  - RMSE: {result['cv_rmse']:.4f} (±{result['cv_rmse_std']:.4f})")
            print(f"  - MAE: {result['cv_mae']:.4f}")
            print(f"\nDesempenho no Conjunto de Teste:")
            print(f"  - R² Score: {result['test_r2']:.4f}")
            print(f"  - RMSE: {result['test_rmse']:.4f}")
            print(f"\nDesempenho no Conjunto de Treino:")
            print(f"  - R² Score: {result['train_r2']:.4f}")
            print(f"  - RMSE: {result['train_rmse']:.4f}")
            
            overfitting_gap = result['train_r2'] - result['test_r2']
            print(f"\nDiferença Train-Test R²: {overfitting_gap:.4f}")
            if overfitting_gap > 0.1:
                print("ATENÇÃO: Possível overfitting detectado!")
            
            print(f"\nRange de Erro no Conjunto de Teste:")
            print(f"  - Erro Mínimo: {result['error_range']['min']:.4f}")
            print(f"  - Erro Médio: {result['error_range']['mean']:.4f}")
            print(f"  - Erro Máximo: {result['error_range']['max']:.4f}")
            print(f"  - Desvio Padrão: {result['error_range']['std']:.4f}")
            print(f"  - Percentil 25%: {result['error_range']['p25']:.4f}")
            print(f"  - Percentil 50% (Mediana): {result['error_range']['p50']:.4f}")
            print(f"  - Percentil 75%: {result['error_range']['p75']:.4f}")
            print(f"  - Percentil 95%: {result['error_range']['p95']:.4f}")
        else:
            print(f"Validação Cruzada ({self.cv_folds} folds):")
            print(f"  - Accuracy: {result['cv_accuracy']:.4f} (±{result['cv_accuracy_std']:.4f})")
            print(f"  - Precision (macro): {result['cv_precision']:.4f}")
            print(f"  - Recall (macro): {result['cv_recall']:.4f}")
            print(f"  - F1-Score (macro): {result['cv_f1']:.4f}")
            print(f"\nDesempenho no Conjunto de Teste:")
            print(f"  - Accuracy: {result['test_accuracy']:.4f}")
            print(f"\nDesempenho no Conjunto de Treino:")
            print(f"  - Accuracy: {result['train_accuracy']:.4f}")
            
            overfitting_gap = result['train_accuracy'] - result['test_accuracy']
            print(f"\nDiferença Train-Test Accuracy: {overfitting_gap:.4f}")
            if overfitting_gap > 0.1:
                print("ATENÇÃO: Possível overfitting detectado!")
            
            print(f"\nMatriz de Confusão (Teste):")
            print(result['confusion_matrix'])
            print(f"\nRelatório de Classificação (Teste):")
            print(result['classification_report'])
            if result['error_range']:
                print(f"\nRange de Confiança das Predições (Teste):")
                print(f"  - Confiança Mínima: {result['error_range']['min_confidence']:.4f}")
                print(f"  - Confiança Média: {result['error_range']['mean_confidence']:.4f}")
                print(f"  - Confiança Máxima: {result['error_range']['max_confidence']:.4f}")
                print(f"  - Desvio Padrão: {result['error_range']['std_confidence']:.4f}")
                print(f"  - Percentil 25%: {result['error_range']['p25_confidence']:.4f}")
                print(f"  - Percentil 50%: {result['error_range']['p50_confidence']:.4f}")
                print(f"  - Percentil 75%: {result['error_range']['p75_confidence']:.4f}")
                print(f"  - Percentil 95%: {result['error_range']['p95_confidence']:.4f}")
    
    def predict(self, X):
        """
        Faz predições usando o melhor modelo.
        
        Parâmetros:
        -----------
        X : pandas.DataFrame ou numpy.array
            Features para predição
        
        Retorna:
        --------
        numpy.array : Predições
        """
        if self.best_model is None:
            raise ValueError("Nenhum modelo foi treinado ainda. Execute fit() primeiro.")
        
        return self.best_model.predict(X)
    
    def predict_proba(self, X):
        """
        Retorna probabilidades de classe (apenas para classificação).
        
        Parâmetros:
        -----------
        X : pandas.DataFrame ou numpy.array
            Features para predição
        
        Retorna:
        --------
        numpy.array : Probabilidades de cada classe
        """
        if self.best_model is None:
            raise ValueError("Nenhum modelo foi treinado ainda. Execute fit() primeiro.")
        
        if self.problem_type != 'classification':
            raise ValueError("predict_proba só está disponível para problemas de classificação.")
        
        if not hasattr(self.best_model, 'predict_proba'):
            raise ValueError(f"O modelo {self.best_model_name} não suporta predict_proba.")
        
        return self.best_model.predict_proba(X)