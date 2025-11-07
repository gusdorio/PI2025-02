# Imports
import pandas as pd
import polars as pl
import numpy as np
from sklearn.decomposition import PCA
import umap
from typing import Literal, List, Optional

class data_transformation:
    """
    Classe para encapsular um pipeline de transformação de dados usando Polars
    para processamento interno de alta performance.

    A classe é projetada para se integrar a um ecossistema Pandas:
    - Recebe um DataFrame do Pandas.
    - Converte para um DataFrame Polars para transformações.
    - Retorna um DataFrame do Pandas processado.
    """

    def __init__(self, raw_data: pd.DataFrame,
                 normalization_type: Literal['z_score', 'min_max', 'none'] = 'z_score',
                 compactation_method: Literal['umap', 'pca', 'none'] = 'umap',
                 pca_variance_target: float = 0.95,
                 umap_n_components: int = 10):
        """
        Inicializa o transformador.

        Parâmetros:
        -----------
        raw_data : pd.DataFrame
            O DataFrame de dados brutos (Pandas) a ser processado.
        normalization_type : str, opcional
            'z_score' (StandardScaler), 'min_max' (MinMaxScaler), ou 'none' (padrão: 'z_score').
        compactation_method : str, opcional
            'umap' (robusto, não-linear), 'pca' (linear), ou 'none' (padrão: 'umap').
        pca_variance_target : float, opcional
            Se o método for 'pca', define a variância explicada desejada (padrão: 0.95).
        umap_n_components : int, opcional
            Se o método for 'umap', define o número de dimensões de destino (padrão: 10).
        """
        try:
            # Converte de Pandas para Polars para processamento interno
            self.data = pl.from_pandas(raw_data)
        except Exception as e:
            print(f"Erro ao converter DataFrame Pandas para Polars: {e}")
            raise
            
        self.normalization_type = normalization_type
        self.compactation_method = compactation_method
        self.pca_variance_target = pca_variance_target
        self.umap_n_components = umap_n_components
        
        # Listas para armazenar os tipos de colunas detectados
        self.categorical_cols: List[str] = []
        self.numerical_cols: List[str] = []

    
    def _detect_column_types(self):
        """
        Método privado para detectar automaticamente colunas numéricas e categóricas
        usando os tipos de dados do Polars.
        """
        # Tipos de dados numéricos no Polars
        numeric_dtypes = pl.NUMERIC_DTYPES
        
        # Tipos de dados categóricos/string
        categorical_dtypes = [pl.Utf8, pl.Categorical]

        self.numerical_cols = [col.name for col in self.data.select(pl.col(numeric_dtypes))]
        self.categorical_cols = [col.name for col in self.data.select(pl.col(categorical_dtypes))]
        
        print(f"[Transformer] Colunas Numéricas detectadas: {self.numerical_cols}")
        print(f"[Transformer] Colunas Categóricas detectadas: {self.categorical_cols}")
        return self

    def data_cleaning(self, method: Literal['drop_na', 'mean', 'median', 'mode']):
        """
        Trata valores ausentes (NaN/Null) usando expressões Polars.
        """
        print(f"[Transformer] Iniciando data_cleaning com método: {method}")
        
        if method == 'drop_na':
            self.data = self.data.drop_nulls()
        else:
            # Constrói uma lista de expressões de preenchimento
            fill_expressions = []
            
            # Expressões para colunas numéricas
            if method in ['mean', 'median']:
                for col in self.numerical_cols:
                    if method == 'mean':
                        fill_value = pl.col(col).mean()
                    else: # median
                        fill_value = pl.col(col).median()
                    fill_expressions.append(pl.col(col).fill_null(fill_value))
            
            # Expressões para colunas categóricas (sempre usa moda)
            for col in self.categorical_cols:
                # Preenche com a moda ou 'Desconhecido' se a moda for nula
                fill_value = pl.col(col).mode().first().fill_null("Desconhecido")
                fill_expressions.append(pl.col(col).fill_null(fill_value))
            
            if fill_expressions:
                self.data = self.data.with_columns(fill_expressions)

        print(f"[Transformer] Formato dos dados após cleaning: {self.data.shape}")
        return self

    def category_column_treatment(self, method: Literal['one_hot', 'dummy']):
        """
        Trata colunas categóricas, convertendo-as em numéricas usando Polars.
        """
        print(f"[Transformer] Iniciando category_column_treatment com método: {method}")
        if not self.categorical_cols:
            print("[Transformer] Nenhuma coluna categórica para tratar.")
            return self
        
        try:
            drop_first = True if method == 'dummy' else False
            self.data = self.data.to_dummies(columns=self.categorical_cols, drop_first=drop_first)
            
            # Atualiza as listas de colunas
            self._detect_column_types() # Re-detecta tipos após o encoding

        except Exception as e:
            print(f"Erro durante o encoding categórico: {e}")
            # Se falhar (ex: coluna já não existe), apenas re-detecta
            self._detect_column_types()

        print(f"[Transformer] Formato dos dados após encoding: {self.data.shape}")
        return self
        
    def outlier_treatment(self, method: Literal['iqr', 'z_score', 'none'], threshold=3.0):
        """
        Trata outliers em colunas numéricas usando "clipping" com expressões Polars.
        """
        print(f"[Transformer] Iniciando outlier_treatment com método: {method}")
        if method == 'none' or not self.numerical_cols:
            return self

        expressions = []
        for col_name in self.numerical_cols:
            col = pl.col(col_name)
            
            if method == 'iqr':
                q1 = col.quantile(0.25)
                q3 = col.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (1.5 * iqr)
                upper_bound = q3 + (1.5 * iqr)
                # Adiciona expressão de clip (só aplica se iqr > 0)
                expr = pl.when(iqr > 0).then(col.clip_horizontal(lower_bound, upper_bound)).otherwise(col).alias(col_name)
                expressions.append(expr)
                
            elif method == 'z_score':
                mean = col.mean()
                std = col.std()
                lower_bound = mean - (threshold * std)
                upper_bound = mean + (threshold * std)
                # Adiciona expressão de clip (só aplica se std > 0)
                expr = pl.when(std > 0).then(col.clip_horizontal(lower_bound, upper_bound)).otherwise(col).alias(col_name)
                expressions.append(expr)
        
        if expressions:
            self.data = self.data.with_columns(expressions)
        
        return self
    
    def z_score_normalization(self):
        """Método privado: Aplica Z-Score (Standardization) usando Polars"""
        if not self.numerical_cols:
            return
        
        print("[Transformer] Aplicando Z-Score (StandardScaler)")
        expressions = []
        for col_name in self.numerical_cols:
            col = pl.col(col_name)
            mean = col.mean()
            std = col.std()
            # (col - mean) / std. Só aplica se std > 0, senão retorna 0 (ou o valor, se preferir)
            expr = pl.when(std > 0).then((col - mean) / std).otherwise(0.0).alias(col_name)
            expressions.append(expr)
            
        self.data = self.data.with_columns(expressions)

    def min_max_normalization(self):
        """Método privado: Aplica Min-Max Scaling (Normalization) usando Polars"""
        if not self.numerical_cols:
            return
        
        print("[Transformer] Aplicando Min-Max (MinMaxScaler)")
        expressions = []
        for col_name in self.numerical_cols:
            col = pl.col(col_name)
            min_val = col.min()
            max_val = col.max()
            range_val = max_val - min_val
            # (col - min) / (max - min). Só aplica se range > 0, senão retorna 0
            expr = pl.when(range_val > 0).then((col - min_val) / range_val).otherwise(0.0).alias(col_name)
            expressions.append(expr)

        self.data = self.data.with_columns(expressions)

    def normalization(self):
        """
        Chama o método de normalização/scaling apropriado 
        baseado no parâmetro 'normalization_type' da inicialização.
        """
        print(f"[Transformer] Iniciando normalization com tipo: {self.normalization_type}")
        if self.normalization_type == 'z_score':
            self.z_score_normalization()
        elif self.normalization_type == 'min_max':
            self.min_max_normalization()
        elif self.normalization_type == 'none':
            print("[Transformer] Normalização pulada.")
        else:
            raise ValueError(f"Tipo de normalização desconhecido: {self.normalization_type}")
        return self

    def compactation(self):
        """
        Aplica redução de dimensionalidade (UMAP ou PCA) via scikit-learn/umap-learn.
        Converte Polars -> NumPy -> Polars.
        """
        print(f"[Transformer] Iniciando compactation com método: {self.compactation_method}")

        if self.compactation_method == 'none':
            print("[Transformer] Compactação pulada.")
            return self
        
        # Garante que todas as colunas são numéricas para a compactação
        non_numeric_cols = [c.name for c in self.data.select(pl.exclude(pl.NUMERIC_DTYPES))]
        
        if non_numeric_cols:
            print(f"AVISO: Removendo colunas não-numéricas restantes antes da compactação: {non_numeric_cols}")
            self.data = self.data.drop(non_numeric_cols)
        
        if self.data.width == 0:
            print("[Transformer] Não é possível aplicar compactação sem colunas numéricas.")
            return self

        # 1. Converte de Polars para NumPy (requerido por PCA e UMAP)
        data_numpy = self.data.to_numpy()
        
        n_components_final = 0
        final_data_numpy = None
        schema_prefix = ""

        # 2. Aplica a técnica escolhida
        if self.compactation_method == 'pca':
            print(f"[Transformer] Aplicando PCA (target variance: {self.pca_variance_target})")
            pca = PCA(n_components=self.pca_variance_target, random_state=42)
            final_data_numpy = pca.fit_transform(data_numpy)
            n_components_final = pca.n_components_
            schema_prefix = "PCA_Comp"
            
        elif self.compactation_method == 'umap':
            print(f"[Transformer] Aplicando UMAP (n_components: {self.umap_n_components})")
            reducer = umap.UMAP(n_components=self.umap_n_components, random_state=42, n_neighbors=min(15, data_numpy.shape[0]-1))
            final_data_numpy = reducer.fit_transform(data_numpy)
            n_components_final = self.umap_n_components
            schema_prefix = "UMAP_Comp"
            
        else:
            raise ValueError(f"Método de compactação desconhecido: {self.compactation_method}")

        print(f"[Transformer] {self.compactation_method.upper()}: Dimensionalidade reduzida de {self.data.width} para {n_components_final} componentes.")
        
        # 3. Converte de volta para Polars DataFrame
        self.data = pl.from_numpy(
            final_data_numpy, 
            schema=[f'{schema_prefix}_{i+1}' for i in range(n_components_final)]
        )
            
        return self
    
    def get_processed_data(self) -> pd.DataFrame:
        """
        Retorna o DataFrame processado final, convertido de volta para Pandas
        para compatibilidade com o restante do ecossistema (ex: scikit-learn).
        """
        print("[Transformer] Convertendo DataFrame Polars de volta para Pandas.")
        return self.data.to_pandas()

    @staticmethod
    def run_pipeline(raw_data: pd.DataFrame, 
                     normalization_type: Literal['z_score', 'min_max', 'none'] = 'z_score', 
                     compactation_method: Literal['umap', 'pca', 'none'] = 'umap',
                     clean_method: Literal['drop_na', 'mean', 'median', 'mode'] = 'drop_na',
                     outlier_method: Literal['iqr', 'z_score', 'none'] = 'iqr',
                     categorical_method: Literal['one_hot', 'dummy'] = 'one_hot',
                     pca_variance_target: float = 0.95,
                     umap_n_components: int = 10
                     ) -> pd.DataFrame:
        """
        Método estático para rodar o pipeline completo de forma fluente (encadeada).
        
        Exemplo de uso:
        --------------
        df = pd.read_csv("meus_dados.csv")
        processed_df = data_transformation.run_pipeline(df, compactation_method='umap')
        """
        print("--- INICIANDO PIPELINE DE TRANSFORMAÇÃO (com Polars) ---")
        
        transformer = data_transformation(
            raw_data, 
            normalization_type, 
            compactation_method,
            pca_variance_target,
            umap_n_components
        )
        
        # Executa o pipeline em ordem
        transformer._detect_column_types()
        
        processed_data_pandas = (
            transformer
            .data_cleaning(method=clean_method)
            .category_column_treatment(method=categorical_method)
            .outlier_treatment(method=outlier_method)
            .normalization()
            .compactation()
            .get_processed_data()
        )
        
        print("--- PIPELINE DE TRANSFORMAÇÃO CONCLUÍDO ---")
        return processed_data_pandas