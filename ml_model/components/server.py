"""
HTTP Server for ML Model Service

Minimal HTTP server using built-in http.server module (no external frameworks).
Receives dataset data from the dashboard service and processes ML requests.

Integration:
- Uses DataPipeline for orchestrated data processing
- Stores received data in MongoDB
- Returns standardized pipeline results
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from datetime import datetime
import traceback

# Import data processing pipeline
from ml_model.components.data_processor import DataPipeline, ProcessingMode, PipelineResult


class MLHandler(BaseHTTPRequestHandler):
    """Request handler for ML processing endpoints."""

    def do_POST(self):
        """Handle POST requests to /process endpoint."""
        if self.path == '/process':
            print("\n" + "🔵" * 30)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 📥 NEW REQUEST RECEIVED")
            print("🔵" * 30)

            try:
                # Read request body
                content_length = int(self.headers.get('Content-Length', 0))
                print(f"[INFO] Content-Length: {content_length:,} bytes ({content_length/1024:.2f} KB)")

                if content_length == 0:
                    print("[ERROR] Empty request body detected")
                    self._send_error(400, "Empty request body")
                    return

                print(f"[INFO] Reading {content_length:,} bytes from request body...")
                post_data = self.rfile.read(content_length)
                print(f"[SUCCESS] Request body read successfully")

                # Parse JSON data
                try:
                    print("[INFO] Parsing JSON data...")
                    data = json.loads(post_data.decode('utf-8'))
                    print(f"[SUCCESS] JSON parsed successfully")
                except json.JSONDecodeError as e:
                    print(f"[ERROR] JSON parsing failed: {str(e)}")
                    self._send_error(400, f"Invalid JSON: {str(e)}")
                    return

                # Log received data
                self._log_request(data)

                # Delegate to pipeline for processing (includes validation, ML, storage)
                print("[INFO] Delegating to DataPipeline...")
                result = self._process_dataset(data)

                # Send success response
                print("[INFO] Sending response to client...")
                self._send_success(result)
                print("[SUCCESS] Response sent successfully ✅")
                print("🔵" * 30 + "\n")

            except Exception as e:
                # Handle unexpected errors
                error_msg = f"Server error: {str(e)}"
                print("\n" + "🔴" * 30)
                print(f"[ERROR] UNEXPECTED ERROR OCCURRED")
                print("🔴" * 30)
                print(f"[ERROR] {error_msg}")
                print("[ERROR] Full traceback:")
                traceback.print_exc()
                print("🔴" * 30 + "\n")
                self._send_error(500, error_msg)
        else:
            # Endpoint not found
            print(f"[WARNING] Unknown POST endpoint requested: {self.path}")
            self._send_error(404, f"Endpoint not found: {self.path}")

    def do_GET(self):
        """Handle GET requests for health checks."""
        if self.path == "/" or self.path == "/health":
            # Health check endpoint - used by dashboard sidebar
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🟢 Health check request from {self.client_address[0]}")

            health_response = {
                "status": "healthy",
                "service": "ml-model",
                "timestamp": datetime.now().isoformat(),
                "endpoints": {
                    "POST /process": "Submit dataset for ML processing"
                }
            }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(health_response).encode('utf-8'))
        else:
            # Unknown GET endpoint
            print(f"[WARNING] Unknown GET endpoint requested: {self.path}")
            self._send_error(404, f"Endpoint not found: {self.path}")

    def _log_request(self, data):
        """Log received request for debugging."""
        metadata = data.get('metadata', {})
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        print("\n" + "-" * 60)
        print(f"[{timestamp}] REQUEST RECEIVED")
        print("-" * 60)
        print(f"Filename:       {data.get('filename', 'N/A')}")
        print(f"Rows:           {metadata.get('row_count', 'N/A'):,}")
        print(f"Columns:        {metadata.get('column_count', 'N/A')}")
        print(f"Column Names:   {', '.join(metadata.get('column_names', []))[:60]}...")
        print(f"File Type:      {metadata.get('file_type', 'N/A')}")
        print(f"Missing Values: {metadata.get('has_missing_values', 'N/A')}")
        print("-" * 60)

    def _process_dataset(self, data):
        """
        Process the received dataset using the DataPipeline.

        This integrates with the full data processing pipeline which:
        1. Validates the incoming data
        2. Processes through ML (currently placeholder)
        3. Stores in MongoDB
        4. Returns standardized results

        Parameters:
        -----------
        data : dict
            Raw data from HTTP request

        Returns:
        --------
        dict : Pipeline execution results
        """
        try:
            # Initialize pipeline with FULL ML mode
            pipeline = DataPipeline(mode=ProcessingMode.FULL_ML)

            # Execute pipeline and get results
            print("[SERVER] Delegating to DataPipeline (Mode: FULL_ML)...")
            result = pipeline.execute(data)

            # Convert PipelineResult to dictionary for HTTP response
            return result.to_dict()

        except Exception as e:
            # If pipeline fails catastrophically, return error
            print(f"[SERVER ERROR] Pipeline execution failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Processing failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def _send_success(self, result):
        """Send successful response."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode('utf-8'))

    def _send_error(self, status_code, message):
        """Send error response."""
        error_response = {
            "status": "error",
            "message": message,
            "timestamp": datetime.now().isoformat()
        }

        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(error_response).encode('utf-8'))

    def log_message(self, format, *args):
        """Override to customize access log format."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {format % args}")


def start_server(host='0.0.0.0', port=5000):
    """
    Start the HTTP server in standby mode.

    Args:
        host: Host to bind to (default: 0.0.0.0 for Docker)
        port: Port to listen on (default: 5000)
    """
    print("\n" + "🟢" * 30)
    print("INITIALIZING HTTP SERVER")
    print("🟢" * 30)

    server_address = (host, port)
    try:
        httpd = HTTPServer(server_address, MLHandler)
        print(f"✅ Server initialized successfully")
        print(f"✅ Bound to address: {host}:{port}")
        print(f"✅ Endpoint available: POST /process")
        print(f"✅ Status: READY - Waiting for requests...")
        print("🟢" * 30)
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Server is now in standby mode\n")

        # Run forever (blocking call)
        httpd.serve_forever()
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print(f"\n❌ ERROR: Port {port} is already in use!")
            print(f"   Another process is already using this port.")
            print(f"   Try stopping other services or use a different port.\n")
        else:
            print(f"\n❌ ERROR: Failed to start server: {str(e)}\n")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: Unexpected error during server startup: {str(e)}\n")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Allow running server directly for testing
    print("Starting ML Model HTTP Server...")
    start_server()
