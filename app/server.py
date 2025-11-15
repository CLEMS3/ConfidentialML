import os
from flask import Flask, jsonify

# Create the Flask application
app = Flask(__name__)

# Get the port number from the environment variable, default to 8080
# This is good practice, though we hard-coded it in the K8s file.
port = int(os.environ.get("PORT", 8080))

@app.route("/")
def index():
    """Provides a basic 'up' message."""
    return jsonify({"status": "server_is_running"})

@app.route("/hello")
def hello_from_server():
    """A simple endpoint for clients to call."""
    print("Received a request from a client!")
    
    # In a real FL system, this is where you'd send
    # the global model or training instructions.
    return jsonify({
        "message": "Hello from the server!",
        "model_version": 1
    })

if __name__ == "__main__":
    print(f"Starting server on 0.0.0.0:{port}...")
    # 'host="0.0.0.0"' is crucial for Docker
    # It means "listen on all available network interfaces"
    app.run(host="0.0.0.0", port=port, debug=True)