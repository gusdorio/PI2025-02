# ML Model Service

Machine learning model service for data processing and analysis.

## Structure

```
ml_model/
├── main.py              # Service entry point (starts HTTP server)
├── requirements.txt     # Python dependencies
└── components/          # Core components
    ├── server.py        # HTTP server for receiving requests
    ├── analyses_ml.py   # ML algorithms and model selection
    └── data_processor.py # Data transformation pipeline
```

## Overview

The ML service is a standalone HTTP server that receives datasets from the dashboard, processes them through machine learning pipelines, and returns results.

**Current Implementation:**
- [x] HTTP server with health checks and data endpoints
- [x] Dataset validation and parsing
- [x] Detailed logging for debugging
- [] Database storage (planned)
- [] ML processing integration (planned)
- [] Queue process manager for user tracking (planned)

## HTTP Server

### server.py

Minimal HTTP server using Python's built-in `http.server` module.

**Endpoints:**

```
GET /health
GET /
  Returns service health status
  Response: {"status": "healthy", "service": "ml-model", ...}

POST /process
  Accepts dataset for ML processing
  Request body: {
    "filename": str,
    "data": str (JSON-encoded DataFrame),
    "metadata": {
      "row_count": int,
      "column_count": int,
      "column_names": list,
      ...
    }
  }
  Response: {
    "status": "received",
    "message": "Dataset received successfully",
    "timestamp": ISO datetime,
    "row_count": int,
    "column_count": int,
    "filename": str
  }
```

**Key Features:**
- Comprehensive request/response logging with visual indicators
- JSON validation and error handling
- Support for large payloads (tested up to 50MB)
- No external framework dependencies

**Starting the Server:**
```python
from ml_model.components.server import start_server

# Start on default port 5000
start_server()

# Custom host/port
start_server(host='0.0.0.0', port=8080)
```

## ML Components

### analyses_ml.py

Automated ML model selection and training.

**AutoMLSelector Class:**

```python
from ml_model.components.analyses_ml import AutoMLSelector

# Initialize selector
selector = AutoMLSelector(
    target_column='target',
    cv_folds=5
)

# Fit models
selector.fit(dataframe)

# Get results
results = selector.get_results_summary()

# Make predictions
predictions = selector.predict(X_test)
```

**Features:**
- Auto-detection of problem type (regression vs classification)
- 8 regression algorithms (Linear, Ridge, Lasso, Random Forest, Gradient Boosting, etc.)
- 6 classification algorithms (Logistic, Random Forest, SVC, Decision Tree, etc.)
- K-Fold cross-validation
- Comprehensive metrics (R², MAE, accuracy, F1-score, etc.)
- Overfitting detection

### data_processor.py

Data transformation pipeline (planned).

**Planned Features:**
- Feature engineering
- Missing value handling
- Categorical encoding
- Feature scaling
- Data splitting

## Development Guide


### Integrating ML Processing

Current flow (minimal):
```
Receive data → Validate → Log → Return acknowledgment
```

Future flow (to implement):
```
Receive data → Validate → Process with AutoMLSelector → Store results → Return results
```

**Integration Steps:**
1. In `server.py`, modify `_process_dataset()` method
2. Convert JSON data back to pandas DataFrame
3. Call `AutoMLSelector` with target column
4. Store results in MongoDB
5. Return processing results

Example:
```python
def _process_dataset(self, data):
    import pandas as pd
    from ml_model.components.analyses_ml import AutoMLSelector

    # Convert JSON to DataFrame
    df = pd.read_json(data['data'])

    # Process with ML
    selector = AutoMLSelector(target_column='target')
    selector.fit(df)
    results = selector.get_results_summary()

    return {
        "status": "completed",
        "results": results,
        "timestamp": datetime.now().isoformat()
    }
```

## Logging

The service uses detailed console logging for debugging:

**Log Indicators:**
- 🟢 Server startup and health checks
- 🔵 Incoming requests and processing steps
- 🔴 Errors and exceptions
- 🟡 Warnings

## Testing

**Manual Testing:**
It's possible to do manual testing via curl, if wanted for easily workflows...

```bash
# Start service
make dev-up

# In another terminal, send test request
curl -X POST http://localhost:5000/process \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "test.csv",
    "data": "[{\"col1\": 1, \"col2\": 2}]",
    "metadata": {
      "row_count": 1,
      "column_count": 2,
      "column_names": ["col1", "col2"]
    }
  }'
```

**Expected Response:**
```json
{
  "status": "received",
  "message": "Dataset received successfully",
  "timestamp": "2024-01-15T10:30:00",
  "row_count": 1,
  "column_count": 2,
  "filename": "test.csv"
}
```
