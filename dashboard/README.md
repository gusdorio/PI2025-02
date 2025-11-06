# Dashboard Application

Streamlit dashboard for ML operations monitoring and data management.

## Structure

```
dashboard/
├── dashboard.py          # Main entry point (homepage with DB status)
├── config.py            # DB connection and configuration
├── pages/               # Auto-discovered Streamlit pages
│   ├── upload.py       # Data upload interface
│   └── results.py      # Results viewer
├── components/          # Custom components (future)
│   ├── uploader.py     # File upload handler
│   └── ml_client.py    # ML service client
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

## Pages Extension

This section guides about how the pages logic works and how to extend functionality.

### upload.py
Upload file → Validate → Send to ML → Store metadata

```python
from components.uploader import validate_file
from components.ml_client import send_to_ml_service

is_valid, errors = validate_file(uploaded_file)
job_id = send_to_ml_service(file_path, metadata)
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