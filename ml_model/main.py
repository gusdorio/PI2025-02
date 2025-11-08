"""
Main entry point for the ML model service.
Starts the HTTP server to listen for processing requests from the dashboard.
"""

import sys
from datetime import datetime
from ml_model.components.server import start_server

def main():
    """Start the ML model HTTP server in standby mode."""
    print("=" * 60)
    print("ML Model Service - HTTP Server")
    print("=" * 60)
    print(f"Starting server at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Listening on: 0.0.0.0:5000")
    print("Endpoint: POST /process")
    print("Status: Standby mode - waiting for requests...")
    print("=" * 60)

    try:
        # Start the HTTP server (blocks here)
        start_server()
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Server shutdown requested")
        print("=" * 60)
        sys.exit(0)
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"FATAL ERROR: {str(e)}")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()
