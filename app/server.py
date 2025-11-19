import os
from flask import Flask, jsonify, request

app = Flask(__name__)
port = int(os.environ.get("PORT", 8080))

# --- FL Global State ---
# 1. The Global Model (initially just a list of zeros)
global_model = [0.0, 0.0, 0.0] 
global_round = 0

# 2. Buffer to hold updates from clients before aggregating
client_updates = []

# 3. How many clients must report back before we aggregate?
# We set this to 3 for testing (even if you have 5 clients running)
MIN_UPDATES_TO_AGGREGATE = 3

@app.route("/")
def index():
    return jsonify({"status": "server_ready", "round": global_round})

@app.route("/get_model", methods=["GET"])
def get_model():
    """Clients call this to download the current global model."""
    return jsonify({
        "round": global_round,
        "weights": global_model
    })

@app.route("/send_update", methods=["POST"])
def receive_update():
    """Clients call this to upload their trained local models."""
    global global_model, global_round, client_updates

    data = request.json
    client_id = data.get("client_id")
    local_weights = data.get("weights")
    
    print(f"Server received update from {client_id} for round {global_round}")

    # Store the update
    client_updates.append(local_weights)

    # Check if we have enough updates to aggregate
    if len(client_updates) >= MIN_UPDATES_TO_AGGREGATE:
        print(f"Aggregating {len(client_updates)} updates...")
        
        # --- AGGREGATION LOGIC (Simple Average) ---
        # Zip clusters the 1st weights together, 2nd weights together, etc.
        # e.g. zip([1,1], [2,2], [3,3]) -> (1,2,3), (1,2,3)
        new_weights = [
            sum(w) / len(w) 
            for w in zip(*client_updates)
        ]
        
        # Update Global State
        global_model = new_weights
        global_round += 1
        client_updates = [] # Clear the buffer for the next round
        
        print(f"Round {global_round} complete. New global model: {global_model}")
        return jsonify({"status": "aggregated", "new_round": global_round})
    
    else:
        return jsonify({"status": "accepted_waiting_for_others"})

if __name__ == "__main__":
    # We set debug=False usually for production, but True helps here.
    # host="0.0.0.0" is required for Docker.
    app.run(host="0.0.0.0", port=port, debug=True)