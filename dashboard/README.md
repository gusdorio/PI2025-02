# Dashboard Application

Streamlit dashboard for ML operations monitoring and data management.

## Structure

```
dashboard/
├── dashboard.py          # Main entry point (homepage with DB status)
├── config.py            # DB connection and configuration
├── pages/               # Auto-discovered Streamlit pages
│   ├── upload.py       # Data upload and ML service communication
│   └── results.py      # Results viewer
├── components/          # Service components
│   ├── uploader.py     # File upload handler and validation
│   └── ml_client.py    # ML service HTTP client
└── .streamlit/          # Streamlit configuration
```

## Database Connection Usage

This section guides about how to use the MongoDB connection over the application.
All pages can access the MongoDB connection through `config.py`.

### Basic Usage

```python
from config import get_database_connection

# Get cached connection
db = get_database_connection()

# Access database
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

## ML Service Communication

The dashboard communicates with the ML model service via HTTP.

### ml_client.py

Sends datasets to the ML service for processing.

```python
from components.ml_client import send_dataset_to_ml

# Send dataset to ML service
response = send_dataset_to_ml(dataset)

# Response structure
{
    "status": "received",
    "message": "Dataset received successfully",
    "timestamp": "2024-01-15T10:30:00",
    "row_count": 1000,
    "column_count": 15,
    "filename": "data.xlsx"
}
```

**Key Features:**
- Automatic numpy type conversion for JSON serialization
- Comprehensive error handling (connection, timeout, HTTP errors)
- Detailed logging for debugging
- 60-second timeout for large datasets

## Pages Extension

This section guides about how the pages logic works and how to extend functionality.

### upload.py
Upload file → Validate → Send to ML → Display response

```python
from components.uploader import UploadService
from components.ml_client import send_dataset_to_ml

# Validate uploaded file
dataset, errors = upload_service.process_upload(uploaded_file)

# Send to ML service
ml_response = send_dataset_to_ml(dataset)
```

### results.py
Query DB → Display metrics → Visualize

```python
# Query results
runs = db.get_collection('pipeline_runs').find({'status': 'completed'})

# Aggregate
pipeline = [{'$group': {'_id': '$status', 'count': {'$sum': 1}}}]
stats = db.get_collection('pipeline_runs').aggregate(pipeline)
```