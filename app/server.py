import os
import random
from flask import Flask, jsonify, request

app = Flask(__name__)
port = int(os.environ.get("PORT", 8080))

# --- Global State ---
global_model = [0.0, 0.0, 0.0] 
global_round = 0 # We will start at 0, and bump to 1 once enough clients join

# Client Management
registered_clients = set()
selected_clients = []
MIN_CLIENTS_TO_START = 3  # <--- Make sure you run at least 3 clients in Docker!
MIN_UPDATES_TO_AGGREGATE = 3

# Buffer
client_updates = []

@app.route("/")
def index():
    return jsonify({"status": "server_ready", "round": global_round})

@app.route("/register", methods=["POST"])
def register_client():
    global selected_clients, global_round
    client_id = request.json.get("client_id")
    registered_clients.add(client_id)
    
    print(f" -> Client registered: {client_id}")
    print(f" -> Total clients: {len(registered_clients)} / {MIN_CLIENTS_TO_START}")
    
    # Check if we can start the very first round
    if global_round == 0 and len(registered_clients) >= MIN_CLIENTS_TO_START:
        print("=== ENOUGH CLIENTS JOINED. STARTING ROUND 1 ===")
        start_new_round()
        
    return jsonify({"status": "registered"})

def start_new_round():
    global global_round, selected_clients, client_updates
    
    # 1. Increment Round
    global_round += 1
    
    # 2. Select Clients
    available = list(registered_clients)
    # Select up to MIN_UPDATES_TO_AGGREGATE clients
    k = min(len(available), MIN_UPDATES_TO_AGGREGATE)
    selected_clients = random.sample(available, k)
    
    # 3. Clear updates from previous round
    client_updates = []
    
    print(f"\n--- STARTING ROUND {global_round} ---")
    print(f"Selected Clients: {selected_clients}")

@app.route("/get_model", methods=["GET"])
def get_model():
    return jsonify({
        "round": global_round,
        "weights": global_model,
        "selected_clients": selected_clients 
    })

@app.route("/send_update", methods=["POST"])
def receive_update():
    global global_model, global_round, client_updates

    data = request.json
    client_id = data.get("client_id")
    local_weights = data.get("weights")
    num_samples = data.get("num_samples")
    
    print(f"Received update from {client_id}")

    client_updates.append((local_weights, num_samples))

    if len(client_updates) >= MIN_UPDATES_TO_AGGREGATE:
        print(f"Aggregating {len(client_updates)} updates...")

        # --- Aggregation Logic ---
        total_samples = sum(samp for _, samp in client_updates)
        weights_only = [x[0] for x in client_updates]
        samples_only = [x[1] for x in client_updates]

        new_weights = []
        for layer_updates in zip(*weights_only):
            weighted_sum = 0
            for client_w, client_n in zip(layer_updates, samples_only):
                weighted_sum += client_w * client_n
            new_weights.append(weighted_sum / total_samples)

        global_model = new_weights
        print(f"Round {global_round} complete. Aggregated Model: {global_model}")
        
        # Start Next Round
        start_new_round()
        
        return jsonify({"status": "aggregated", "new_round": global_round})
    
    else:
        return jsonify({"status": "waiting_for_others"})

if __name__ == "__main__":
    # Threaded=True prevents the server from blocking if multiple requests hit at once
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)