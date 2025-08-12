import http.server
import socketserver
import os

PORT = 8000

# We create a custom handler to override the default behavior
class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # We start by getting the path of the requested file.
        request_path = self.path
        
        # If the root path is requested, we serve index.html
        if request_path == '/':
            request_path = '/index.html'

        # We construct the full file path from the current directory and the requested path.
        full_path = os.path.join(os.getcwd(), request_path[1:])

        # Check if the path points to a directory.
        if os.path.isdir(full_path):
            # If it's a directory, we check for an index.html file inside.
            # If there isn't one, we serve the 404 page.
            if not os.path.exists(os.path.join(full_path, 'index.html')):
                self.send_response(404)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                # We open and read the 404.html file.
                try:
                    with open('404.html', 'rb') as f:
                        self.wfile.write(f.read())
                except FileNotFoundError:
                    # Fallback for when 404.html doesn't exist.
                    self.wfile.write(b"<h1>404 Not Found</h1><p>The requested directory does not have an index file.</p>")
                return
            # If an index.html exists, the original handler will handle it.

        # Check if the requested file exists at all.
        if not os.path.exists(full_path):
            # If the file doesn't exist, we serve our 404.html page.
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            try:
                with open('404.html', 'rb') as f:
                    self.wfile.write(f.read())
            except FileNotFoundError:
                self.wfile.write(b"<h1>404 Not Found</h1><p>The requested file was not found and a custom 404 page was not available.</p>")
            return

        # If the file exists and is not a directory, we call the original handler to serve the file as usual.
        # This will correctly handle all other requests for existing files (like CSS, JS, images, etc.).
        super().do_GET()

# Set up the server using our custom handler
with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
    print("serving at port", PORT)
    print("Visit http://localhost:8000")
    httpd.serve_forever()
