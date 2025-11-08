# Dashboard Application

Streamlit dashboard with orchestrated data upload pipeline.

## Structure

```
dashboard/
├── dashboard.py                # Main entry point (homepage with DB status)
├── config.py                  # DB connection and configuration
├── pages/                     # Auto-discovered Streamlit pages
│   ├── upload.py             # Data upload page using pipeline
│   └── results.py            # Results viewer
├── components/               # Service components
│   ├── data_upload_pipeline.py  # Upload pipeline orchestrator
│   ├── uploader.py          # File validation and reading
│   └── ml_client.py         # Minimal HTTP wrapper for pipeline
└── .streamlit/              # Streamlit configuration
```

## Data Upload Pipeline

The dashboard uses an orchestrated pipeline pattern for data processing.

### Architecture

```
DataUploadPipeline → [Validation → Transformation → ML Communication]
                                        ↓
                                   (SKIPPED)
```

### Pipeline Components

#### data_upload_pipeline.py

Main orchestrator that manages the upload workflow through stages.

```python
from components.data_upload_pipeline import DataUploadPipeline, TransformMode

# Initialize pipeline (transformation disabled)
pipeline = DataUploadPipeline(transform_mode=TransformMode.NONE)

# Execute pipeline
result = pipeline.execute(uploaded_file, upload_service)

# Result structure
UploadPipelineResult:
  - status: "completed" or "failed"
  - message: Human-readable message
  - dataset: UploadedDataset object
  - ml_response: Response from ML service
  - metadata: Processing information
```

**Processing Modes:**
- `TransformMode.NONE`: Skip transformation (current default)
- `TransformMode.BASIC`: Basic cleaning (not implemented)
- `TransformMode.FULL`: Complete trasformator.py pipeline (not implemented)

### Pipeline Stages

#### Stage 1: File Validation
Uses existing `UploadService` from `uploader.py` to validate and read files.

#### Stage 2: Data Transformation
Currently **SKIPPED** in `NONE` mode. Prepared for integration with `models/trasformator.py`.

```python
class DataTransformer:
    def transform(self, dataset: UploadedDataset) -> Tuple[UploadedDataset, Dict]:
        if self.mode == TransformMode.NONE:
            # Pass through unchanged
            return dataset, {"transform_status": "skipped"}

        # Future: Apply trasformator.py pipeline
        # from models.trasformator import data_transformation
        # transformer = data_transformation(dataset.raw_dataframe)
        # processed_df = transformer.run_pipeline()
```

#### Stage 3: ML Communication
Serializes data and sends to ML service. Includes:
- Numpy type conversion (moved from ml_client)
- Payload preparation with transformation metadata
- HTTP transmission

### Enabling Transformation

To enable data transformation, change the pipeline mode:

```python
# In upload.py
pipeline = DataUploadPipeline(transform_mode=TransformMode.FULL)
```

Then uncomment the transformation logic in `DataTransformer.transform()` method.

## Database Connection

MongoDB connection is accessed through `config.py`.

```python
from config import get_database_connection

db = get_database_connection()
collection = db.get_collection('my_collection')
```

### Query Examples

```python
# Insert document
collection.insert_one({'key': 'value'})

# Find documents
results = collection.find({'status': 'completed'})

# Count documents
count = collection.count_documents({})

# Update document
collection.update_one({'_id': doc_id}, {'$set': {'status': 'processed'}})
```

## Pages

### upload.py
Uses DataUploadPipeline to process uploads:

```python
from components.data_upload_pipeline import DataUploadPipeline, TransformMode

# Initialize pipeline
pipeline = DataUploadPipeline(transform_mode=TransformMode.NONE)

# Process upload
result = pipeline.execute(uploaded_file, upload_service)

if result.is_success:
    # Display results
    dataset = result.dataset
    ml_response = result.ml_response
```

### results.py
Queries MongoDB for pipeline runs and displays metrics:

```python
# Query results
runs = db.get_collection('pipeline_runs').find({'status': 'completed'})

# Aggregate
pipeline = [{'$group': {'_id': '$status', 'count': {'$sum': 1}}}]
stats = db.get_collection('pipeline_runs').aggregate(pipeline)
```