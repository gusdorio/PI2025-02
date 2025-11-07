"""
Data Upload Component
======================
Handles file upload, validation, and processing for the dashboard.

This module provides a clean, interface-based design for easy extension:
- FileValidator: Validates file properties and content
- FileReader: Reads CSV/Excel files into DataFrames
- UploadService: Orchestrates the upload process

Design Principles:
- Single Responsibility: Each class has one job
- Easy Extension: Add new validators/readers without changing core logic
- Clear Interfaces: Methods have clear inputs/outputs
- Graceful Error Handling: User-friendly error messages
"""

import pandas as pd
import sys
import os
from typing import Tuple, List, Optional
from io import BytesIO

# Add models to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.dataset import UploadedDataset


# ============================================================================
# FILE VALIDATOR
# ============================================================================
class FileValidator:
    """
    Validates uploaded files for size, type, and basic content requirements.
    """

    def __init__(self, max_file_size_mb: int = 50):
        """
        Initialize validator with configurable limits

        Parameters:
        -----------
        max_file_size_mb : int, optional
            Maximum file size in MB (default: 50)
        """
        self.max_file_size_mb = max_file_size_mb
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.allowed_types = ['csv', 'xlsx', 'xls']
        self.errors: List[str] = []

    def validate_file_size(self, file) -> bool:
        """
        Check if file size is within limits

        Parameters:
        -----------
        file : UploadedFile
            Streamlit uploaded file object

        Returns:
        --------
        bool : True if valid, False otherwise
        """
        if file.size > self.max_file_size_bytes:
            self.errors.append(
                f"File size ({file.size / 1024**2:.2f} MB) exceeds "
                f"maximum allowed size ({self.max_file_size_mb} MB)"
            )
            return False
        return True

    def validate_file_type(self, file) -> bool:
        """
        Check if file type is supported

        Parameters:
        -----------
        file : UploadedFile
            Streamlit uploaded file object

        Returns:
        --------
        bool : True if valid, False otherwise
        """
        file_extension = file.name.split('.')[-1].lower()

        if file_extension not in self.allowed_types:
            self.errors.append(
                f"File type '.{file_extension}' is not supported. "
                f"Allowed types: {', '.join(self.allowed_types)}"
            )
            return False
        return True

    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Perform basic validation on the DataFrame

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to validate

        Returns:
        --------
        bool : True if valid, False otherwise
        """
        # Check if DataFrame is None
        if df is None:
            self.errors.append("Failed to read file - DataFrame is None")
            return False

        # Check if DataFrame is empty
        if df.empty:
            self.errors.append("The uploaded file contains no data")
            return False

        # Check if DataFrame has columns
        if len(df.columns) == 0:
            self.errors.append("The uploaded file has no columns")
            return False

        # Check for minimum row count
        if len(df) < 1:
            self.errors.append("The uploaded file must have at least 1 row of data")
            return False

        # Check for all columns being unnamed
        unnamed_count = sum(1 for col in df.columns if str(col).startswith('Unnamed:'))
        if unnamed_count == len(df.columns):
            self.errors.append("All columns are unnamed - please ensure your file has column headers")
            return False

        # Check if all values are NaN
        if df.isnull().all().all():
            self.errors.append("All values in the file are empty or null")
            return False

        # All checks passed
        return True

    def reset_errors(self):
        """Clear accumulated errors"""
        self.errors = []

    def get_errors(self) -> List[str]:
        """Get list of validation errors"""
        return self.errors


# ============================================================================
# FILE READER
# ============================================================================
class FileReader:
    """
    Reads uploaded files into pandas DataFrames with error handling.
    """

    @staticmethod
    def read_csv(file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Read CSV file into DataFrame

        Parameters:
        -----------
        file : UploadedFile
            Streamlit uploaded file object

        Returns:
        --------
        Tuple[Optional[pd.DataFrame], Optional[str]]
            (DataFrame, error_message) - DataFrame is None if error occurs
        """
        try:
            # Try common encodings
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    file.seek(0)  # Reset file pointer
                    df = pd.read_csv(file, encoding=encoding)
                    return df, None
                except UnicodeDecodeError:
                    continue

            # If all encodings fail
            return None, "Unable to decode CSV file. Please check the file encoding."

        except Exception as e:
            return None, f"Error reading CSV file: {str(e)}"

    @staticmethod
    def read_excel(file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Read Excel file into DataFrame

        Parameters:
        -----------
        file : UploadedFile
            Streamlit uploaded file object

        Returns:
        --------
        Tuple[Optional[pd.DataFrame], Optional[str]]
            (DataFrame, error_message) - DataFrame is None if error occurs
        """
        try:
            file.seek(0)  # Reset file pointer
            # Read the first sheet by default
            df = pd.read_excel(file, engine='openpyxl')
            return df, None

        except Exception as e:
            return None, f"Error reading Excel file: {str(e)}"

    @staticmethod
    def auto_read(file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Automatically detect and read file based on extension

        Parameters:
        -----------
        file : UploadedFile
            Streamlit uploaded file object

        Returns:
        --------
        Tuple[Optional[pd.DataFrame], Optional[str]]
            (DataFrame, error_message) - DataFrame is None if error occurs
        """
        file_extension = file.name.split('.')[-1].lower()

        if file_extension == 'csv':
            return FileReader.read_csv(file)
        elif file_extension in ['xlsx', 'xls']:
            return FileReader.read_excel(file)
        else:
            return None, f"Unsupported file type: .{file_extension}"


# ============================================================================
# UPLOAD SERVICE ORCHESTRATOR
# ============================================================================
class UploadService:
    """
    Main service that orchestrates the upload process.
    Coordinates validation, reading, and object creation.
    """

    def __init__(self, max_file_size_mb: int = 50):
        """
        Initialize the upload service

        Parameters:
        -----------
        max_file_size_mb : int, optional
            Maximum file size in MB (default: 50)
        """
        self.validator = FileValidator(max_file_size_mb=max_file_size_mb)
        self.reader = FileReader()

    def process_upload(self, file) -> Tuple[Optional[UploadedDataset], List[str]]:
        """
        Process uploaded file through complete pipeline

        This is the main entry point for file upload processing.
        It orchestrates validation, reading, and object creation.

        Parameters:
        -----------
        file : UploadedFile
            Streamlit uploaded file object

        Returns:
        --------
        Tuple[Optional[UploadedDataset], List[str]]
            (UploadedDataset object, list of errors)
            UploadedDataset is None if processing fails
        """
        # Reset errors from previous runs
        self.validator.reset_errors()

        # Step 1: Validate file properties
        if not self.validator.validate_file_size(file):
            return None, self.validator.get_errors()

        if not self.validator.validate_file_type(file):
            return None, self.validator.get_errors()

        # Step 2: Read file into DataFrame
        df, read_error = self.reader.auto_read(file)

        if read_error:
            self.validator.errors.append(read_error)
            return None, self.validator.get_errors()

        # Step 3: Validate DataFrame content
        if not self.validator.validate_dataframe(df):
            return None, self.validator.get_errors()

        # Step 4: Create UploadedDataset object
        file_extension = file.name.split('.')[-1].lower()
        file_type = 'csv' if file_extension == 'csv' else 'excel'

        try:
            dataset = UploadedDataset(
                dataframe=df,
                filename=file.name,
                file_type=file_type
            )
            return dataset, []

        except Exception as e:
            self.validator.errors.append(f"Error creating dataset object: {str(e)}")
            return None, self.validator.get_errors()

    def get_validation_errors(self) -> List[str]:
        """
        Get list of validation errors from last operation

        Returns:
        --------
        List[str] : List of error messages
        """
        return self.validator.get_errors()
