# Using built-in http.server (ultra-minimal)
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class MLHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/process':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            
            # Process ML
            result = {"status": "ok"}
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

HTTPServer(('0.0.0.0', 5000), MLHandler).serve_forever()