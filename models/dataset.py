"""
Simple Dataset Handler
=======================
Lightweight data wrapper without heavy dependencies.
Separated from trasformator.py to avoid importing polars/umap in dashboard service.

This module only requires pandas, which is available in all services.
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List


class UploadedDataset:
    """
    Simple object mapper for uploaded data - MVP version

    This class provides a lightweight wrapper around uploaded data,
    storing metadata and providing basic access methods.
    Designed to be easily extensible for future processing steps.

    Dependencies: Only pandas (lightweight)
    """

    def __init__(self, dataframe: pd.DataFrame, filename: str, file_type: str):
        """
        Initialize the uploaded dataset wrapper

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The raw uploaded data
        filename : str
            Original filename
        file_type : str
            File type ('csv' or 'excel')
        """
        self.raw_dataframe = dataframe
        self.filename = filename
        self.file_type = file_type
        self.row_count = len(dataframe)
        self.column_count = len(dataframe.columns)
        self.column_names = list(dataframe.columns)
        self.data_types = {col: str(dtype) for col, dtype in dataframe.dtypes.items()}
        self.has_missing_values = dataframe.isnull().any().any()
        self.upload_timestamp = datetime.now()

    def get_summary(self) -> Dict:
        """
        Return metadata summary as dictionary

        Returns:
        --------
        dict : Summary information about the dataset
        """
        missing_per_column = self.raw_dataframe.isnull().sum().to_dict()

        return {
            'filename': self.filename,
            'file_type': self.file_type,
            'row_count': self.row_count,
            'column_count': self.column_count,
            'column_names': self.column_names,
            'data_types': self.data_types,
            'has_missing_values': self.has_missing_values,
            'missing_values_per_column': missing_per_column,
            'upload_timestamp': self.upload_timestamp.isoformat(),
            'memory_usage_mb': round(self.raw_dataframe.memory_usage(deep=True).sum() / 1024**2, 2)
        }

    def get_preview(self, n_rows: int = 10) -> pd.DataFrame:
        """
        Get first N rows for preview

        Parameters:
        -----------
        n_rows : int, optional
            Number of rows to return (default: 10)

        Returns:
        --------
        pd.DataFrame : First n_rows of the dataset
        """
        return self.raw_dataframe.head(n_rows)

    def to_dict(self) -> Dict:
        """
        Serialize dataset for future storage/transmission

        Returns:
        --------
        dict : Serialized representation (without raw data)
        """
        return {
            'metadata': self.get_summary(),
            'preview': self.get_preview(5).to_dict('records')
        }

    def __repr__(self) -> str:
        return (f"UploadedDataset(filename='{self.filename}', "
                f"rows={self.row_count}, cols={self.column_count})")
