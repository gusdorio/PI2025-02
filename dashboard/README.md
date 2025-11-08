# Dashboard Application

Streamlit dashboard with orchestrated data upload and results visualization pipelines.

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
│   ├── data_results_pipeline.py # Results pipeline (TODO)
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

## Complete End-to-End Workflow

The following diagram illustrates the complete data flow from file upload to results visualization:

```mermaid
flowchart TD
    subgraph "User Interface"
        A[User opens Dashboard] --> B[Navigate to Upload page]
        B --> C[Select and upload file]
        L --> M[Navigate to Results page]
        M --> N[View processed results]
    end

    subgraph "Dashboard Upload Pipeline"
        C --> D[Stage 1: File Validation<br/>components/uploader.py]
        D --> D1{Valid file?}
        D1 -->|Yes| E[Create UploadedDataset]
        D1 -->|No| ERR1[Return validation errors]

        E --> F[Stage 2: Data Transformation<br/>components/data_upload_pipeline.py]
        F --> F1{Transform mode}
        F1 -->|NONE| F2[Skip transformation]
        F1 -->|BASIC/FULL| F3[Apply trasformator.py<br/>(Future)]

        F2 --> G[Stage 3: ML Communication<br/>components/data_upload_pipeline.py]
        F3 --> G
        G --> G1[Serialize dataset]
        G1 --> G2[Convert numpy types]
        G2 --> G3[Send HTTP POST]
    end

    subgraph "ML Model Service"
        G3 --> H[ML Service receives data<br/>ml_model/components/server.py]
        H --> I[Process through pipeline]
        I --> J[Store in MongoDB<br/>- datasets collection<br/>- pipeline_runs collection]
    end

    subgraph "Dashboard Results Pipeline"
        J --> K[MongoDB]
        K --> L[Data Results Pipeline<br/>components/data_results_pipeline.py<br/>(TODO)]
        L --> L1[Query processed datasets]
        L1 --> L2[Fetch ML results]
        L2 --> L3[Prepare visualizations]
    end

    style A fill:#e1f5ff
    style C fill:#e1f5ff
    style N fill:#e1f5ff
    style D fill:#fff4e1
    style F fill:#fff4e1
    style G fill:#fff4e1
    style I fill:#ffe1f5
    style J fill:#ffe1f5
    style K fill:#ffe1f5
    style L fill:#ffcccc
    style ERR1 fill:#ffcccc
```

## Data Results Pipeline

### Overview
The results pipeline handles fetching and preparing ML processing results for visualization. The abstract interface is defined in `components/data_results_pipeline.py` following the same orchestrated pipeline pattern as the upload pipeline.

### Architecture
```
DataResultsPipeline → [Query → Transform → Prepare → Visualize]
                         ↓         ↓          ↓          ↓
                    MongoDB   Domain Obj  Format    Streamlit
```

### Implementation Directives

#### 1. Concrete Stage Implementations

Create concrete implementations for each abstract stage in separate classes:

```python
# components/data_results_pipeline_impl.py

from components.data_results_pipeline import (
    QueryStage, TransformStage, PrepareStage, VisualizeStage,
    DatasetResult, ResultsCollection
)
from config import get_database_connection

class MongoQueryStage(QueryStage):
    """Concrete MongoDB query implementation"""

    def __init__(self):
        self.db = get_database_connection()

    def query_datasets(self, filters=None, limit=None):
        collection = self.db.get_collection('datasets')
        cursor = collection.find(filters or {})
        if limit:
            cursor = cursor.limit(limit)
        return list(cursor.sort('upload_timestamp', -1))

    def query_pipeline_runs(self, filters=None, limit=None):
        collection = self.db.get_collection('pipeline_runs')
        # Implementation here

    def query_ml_results(self, batch_ids):
        # Query ml_results collection when ML processing is enabled
        pass

class ResultTransformStage(TransformStage):
    """Transform MongoDB documents to domain objects"""

    def transform_to_dataset_result(self, dataset_doc, pipeline_doc=None, ml_doc=None):
        return DatasetResult(
            batch_id=dataset_doc['_id'],
            filename=dataset_doc['filename'],
            status=DatasetStatus.COMPLETED,
            upload_timestamp=dataset_doc['upload_timestamp'],
            original_data=pd.DataFrame(dataset_doc['data']),
            ml_results=ml_doc or {},
            processing_metadata=pipeline_doc.get('summary', {})
        )

class DataPrepareStage(PrepareStage):
    """Prepare data for visualization"""

    def prepare_table_data(self, result, columns=None):
        df = result.original_data
        if columns:
            df = df[columns]
        return df.head(100)  # Limit for performance

class StreamlitVisualizeStage(VisualizeStage):
    """Render visualizations in Streamlit"""

    def render_dataset_card(self, result):
        import streamlit as st

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Batch ID", result.batch_id[:12] + "...")
        with col2:
            st.metric("Status", result.status.value)
        with col3:
            st.metric("Rows", len(result.original_data))
```

#### 2. Pipeline Initialization

Instantiate the pipeline with concrete stages:

```python
# In pages/results.py

from components.data_results_pipeline import DataResultsPipeline, ResultsMode
from components.data_results_pipeline_impl import (
    MongoQueryStage, ResultTransformStage,
    DataPrepareStage, StreamlitVisualizeStage
)

def initialize_results_pipeline():
    """Factory function to create pipeline with concrete stages"""
    pipeline = DataResultsPipeline(mode=ResultsMode.DETAILED)

    # Inject concrete implementations
    pipeline.query_stage = MongoQueryStage()
    pipeline.transform_stage = ResultTransformStage()
    pipeline.prepare_stage = DataPrepareStage()
    pipeline.visualize_stage = StreamlitVisualizeStage()

    return pipeline
```

#### 3. Integration with results.py

Update the results page to use the pipeline:

```python
# pages/results.py

import streamlit as st
from components.data_results_pipeline import ResultsMode

st.set_page_config(page_title="Results", page_icon="📊", layout="wide")

# Initialize pipeline
pipeline = initialize_results_pipeline()

# Sidebar filters
with st.sidebar:
    st.subheader("Filters")

    # Time range filter
    time_range = st.selectbox(
        "Time Range",
        ["Last 24 hours", "Last 7 days", "Last 30 days", "All time"]
    )

    # Status filter
    status_filter = st.multiselect(
        "Status",
        ["completed", "processing", "failed", "queued"]
    )

    # Display mode
    display_mode = st.radio(
        "Display Mode",
        ["Simple", "Detailed", "Comparison"]
    )

# Main content
st.title("📊 Processing Results")

# Fetch results using pipeline
filters = build_filters(time_range, status_filter)
results_collection = pipeline.fetch_all_results(filters=filters, limit=50)

# Display overview metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Datasets", len(results_collection.results))
with col2:
    completed = len(results_collection.filter_by_status(DatasetStatus.COMPLETED))
    st.metric("Completed", completed)
with col3:
    processing = len(results_collection.filter_by_status(DatasetStatus.PROCESSING))
    st.metric("Processing", processing)
with col4:
    failed = len(results_collection.filter_by_status(DatasetStatus.FAILED))
    st.metric("Failed", failed)

# Render results based on mode
if display_mode == "Simple":
    for result in results_collection.get_recent(10):
        pipeline.visualize_stage.render_dataset_card(result)

elif display_mode == "Detailed":
    selected_batch = st.selectbox(
        "Select Dataset",
        [r.batch_id for r in results_collection.results]
    )
    if selected_batch:
        result = results_collection.get_by_batch_id(selected_batch)
        pipeline.visualize_stage.render_detailed_view(result)

elif display_mode == "Comparison":
    selected_batches = st.multiselect(
        "Select Datasets to Compare",
        [r.batch_id for r in results_collection.results],
        max_selections=4
    )
    if len(selected_batches) >= 2:
        results = [results_collection.get_by_batch_id(b) for b in selected_batches]
        pipeline.visualize_stage.render_comparison_view(results)
```

#### 4. Job Queue Integration (Future)

When implementing the job queue system for Azure Container Apps:

```python
# components/job_queue_impl.py

from components.data_results_pipeline import JobQueueMonitor
import asyncio

class RedisJobQueueMonitor(JobQueueMonitor):
    """Redis-based job queue monitor"""

    def __init__(self, pipeline, redis_client):
        super().__init__(pipeline)
        self.redis = redis_client

    def start_monitoring(self, job_id, batch_id):
        self.active_jobs[job_id] = batch_id
        # Subscribe to Redis pub/sub for job updates

    def get_queue_position(self, job_id):
        # Query Redis for queue position
        return self.redis.lindex('job_queue', job_id)

# Integration in results page
if pipeline.enable_realtime:
    monitor = RedisJobQueueMonitor(pipeline, redis_client)

    # Display real-time status
    placeholder = st.empty()
    while monitor.active_jobs:
        with placeholder.container():
            for job_id, batch_id in monitor.active_jobs.items():
                position = monitor.get_queue_position(job_id)
                st.info(f"Job {job_id}: Position {position} in queue")
        time.sleep(1)
```

#### 5. Caching Strategy

Implement intelligent caching to improve performance:

```python
# Add to MongoQueryStage
@st.cache_data(ttl=60)  # Cache for 60 seconds
def query_datasets_cached(self, filters_hash):
    return self.query_datasets(filters)

# Generate stable hash for filters
def get_filters_hash(filters):
    import hashlib
    import json
    return hashlib.md5(json.dumps(filters, sort_keys=True).encode()).hexdigest()
```

#### 6. Error Handling

Implement comprehensive error handling at each stage:

```python
class DataResultsPipelineImpl(DataResultsPipeline):
    def fetch_all_results(self, filters=None, limit=20):
        try:
            # Stage 1: Query
            datasets = self.query_stage.query_datasets(filters, limit)
            if not datasets:
                return ResultsCollection([])

            # Stage 2: Transform
            batch_ids = [d['_id'] for d in datasets]
            pipeline_runs = self.query_stage.query_pipeline_runs(
                {'batch_id': {'$in': batch_ids}}
            )

            # Create lookup dict
            runs_by_batch = {r['batch_id']: r for r in pipeline_runs}

            # Transform to domain objects
            results = []
            for dataset in datasets:
                pipeline_run = runs_by_batch.get(dataset['_id'])
                result = self.transform_stage.transform_to_dataset_result(
                    dataset, pipeline_run
                )
                results.append(result)

            return ResultsCollection(results)

        except Exception as e:
            st.error(f"Failed to fetch results: {str(e)}")
            return ResultsCollection([])
```

#### 7. Export Functionality

Implement the export feature:

```python
def export_results(self, batch_ids, format="csv"):
    """Export results in specified format"""

    results = [self.fetch_single_result(bid) for bid in batch_ids]
    results = [r for r in results if r]  # Filter None values

    if format == "csv":
        # Combine all dataframes
        dfs = [r.original_data for r in results]
        combined = pd.concat(dfs, keys=batch_ids)
        return combined.to_csv().encode('utf-8')

    elif format == "excel":
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            for result in results:
                sheet_name = result.batch_id[:31]  # Excel sheet name limit
                result.original_data.to_excel(writer, sheet_name=sheet_name)
        return buffer.getvalue()

    elif format == "json":
        export_data = {
            bid: {
                'metadata': r.processing_metadata,
                'ml_results': r.ml_results,
                'data': r.original_data.to_dict('records')
            }
            for bid, r in zip(batch_ids, results)
        }
        return json.dumps(export_data).encode('utf-8')
```

### Testing Strategy

1. **Unit Tests**: Test each stage independently with mock data
2. **Integration Tests**: Test complete pipeline flow with test MongoDB
3. **Performance Tests**: Ensure pipeline handles large datasets efficiently
4. **UI Tests**: Verify Streamlit components render correctly

### Performance Considerations

- Use MongoDB indexes on `upload_timestamp` and `batch_id`
- Implement pagination for large result sets
- Cache frequently accessed data with TTL
- Use async queries for job queue monitoring
- Limit DataFrame sizes in visualization stage

### Migration Path

1. Start with SIMPLE mode implementation
2. Add DETAILED mode with single dataset views
3. Implement COMPARISON mode for multi-dataset analysis
4. Add job queue integration when Azure Container Apps are configured
5. Enable REALTIME mode with WebSocket/SSE updates