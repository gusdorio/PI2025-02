"""
MongoDB Document Models - Abstract Base Structures
Generic schemas to be extended with project-specific implementations - Just as examples...
These dataclasses serve as templates for defining MongoDB document schemas.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone


# ==================== BASE EMBEDDED DOCUMENTS ====================

@dataclass
class BaseMetrics:
    """
    Abstract metrics container for model performance
    Override with specific metrics as needed

    Example usage:
        metrics = BaseMetrics(timestamp=datetime.utcnow(), metadata={'accuracy': 0.95})
    """
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)  # Flexible storage for additional metrics


@dataclass
class BaseResult:
    """
    Abstract base document for analysis/model results

    MongoDB indexes recommended: ['batch_id', 'created_at']

    Example usage:
        result = BaseResult(batch_id='batch_001', metadata={'model': 'rf_v1'})
    """
    batch_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)  # Project-specific data


@dataclass
class BaseAnalysis:
    """
    Abstract base document for data analysis results

    MongoDB indexes recommended: ['batch_id', 'created_at']

    Example usage:
        analysis = BaseAnalysis(
            batch_id='batch_001',
            analysis_type='feature_importance',
            results={'features': [...], 'scores': [...]},
            parameters={'n_estimators': 100}
        )
    """
    batch_id: str
    analysis_type: str
    results: Dict[str, Any]  # Flexible results storage
    parameters: Optional[Dict[str, Any]] = None  # Analysis parameters
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PipelineRunSummary:
    """
    Summary of a complete pipeline execution

    MongoDB collection name: 'pipeline_runs'
    MongoDB indexes recommended: ['batch_id', 'status', 'started_at']

    Example usage:
        run = PipelineRunSummary(
            batch_id='batch_001',
            status='in_progress',
            summary={'records_processed': 1000}
        )
    """
    batch_id: str  # Should be unique
    status: str  # Choices: 'started', 'in_progress', 'completed', 'failed'
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    summary: Dict[str, Any] = field(default_factory=dict)  # Aggregated results
    errors: List[Dict[str, Any]] = field(default_factory=list)  # Error tracking
    metadata: Dict[str, Any] = field(default_factory=dict)  # Version info, parameters, etc.

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary for MongoDB insertion"""
        return {
            'batch_id': self.batch_id,
            'status': self.status,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'summary': self.summary,
            'errors': self.errors,
            'metadata': self.metadata
        }
