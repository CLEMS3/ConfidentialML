import os
import random
from flask import Flask, jsonify, request
import requests
import phe.paillier
import json

app = Flask(__name__)
port = int(os.environ.get("PORT", 8080))

# --- Global State ---
global_model = [] # Will hold encrypted weights (SUM)
global_round = 0 
public_key = None
global_total_samples = 1 # Denominator for the global model
key_gen_triggered = False # Flag to ensure we only trigger keygen once
leader_id = None # Store the Leader ID to request key sharing later

# Client Management
registered_clients = set()
selected_clients = []
MIN_CLIENTS_TO_START = 3
MIN_UPDATES_TO_AGGREGATE = 3

# Buffer
client_updates = []

@app.route("/")
def index():
    return jsonify({"status": "server_ready", "round": global_round})

@app.route("/register", methods=["POST"])
def register_client():
    global selected_clients, global_round, key_gen_triggered, leader_id
    client_id = request.json.get("client_id")
    registered_clients.add(client_id)
    
    print(f" -> Client registered: {client_id}")
    print(f" -> Total clients: {len(registered_clients)} / {MIN_CLIENTS_TO_START}")
    
    # Check if we can start the very first round
    if global_round == 0 and len(registered_clients) >= MIN_CLIENTS_TO_START and not key_gen_triggered:
        key_gen_triggered = True
        print("=== ENOUGH CLIENTS JOINED. INITIATING KEY GENERATION ===")
        initiate_key_generation()
    
    # Handle Late Joiners: If keys already generated, ask Leader to share keys with this new client
    elif key_gen_triggered and leader_id:
        print(f"New client {client_id} joined after keygen. Requesting Leader {leader_id} to share keys.")
        try:
            requests.post(f"http://{leader_id}:5000/share_keys", json={"peers": [client_id]})
        except Exception as e:
            print(f"Failed to request key sharing from Leader {leader_id}: {e}")
        
    return jsonify({"status": "registered"})

def initiate_key_generation():
    global registered_clients, leader_id
    # Select Leader
    leader_id = random.choice(list(registered_clients))
    peers = [c for c in registered_clients if c != leader_id]
    
    print(f"Selected Leader: {leader_id}. Peers: {peers}")
    
    # Trigger Leader to generate keys and share with peers
    try:
        # Assuming client_id is the hostname/address
        requests.post(f"http://{leader_id}:5000/trigger_keygen", json={"peers": peers})
    except Exception as e:
        print(f"Failed to trigger keygen on {leader_id}: {e}")

@app.route("/public_key", methods=["POST"])
def receive_public_key():
    global public_key, global_model, global_total_samples
    data = request.json
    n = int(data["n"])
    public_key = phe.paillier.PaillierPublicKey(n)
    print("Received Public Key from Leader.")
    
    # Init with 3 weights as per original code
    # We init with Encrypted(0). The denominator will be 1.
    global_model = [public_key.encrypt(0.0) for _ in range(3)]
    global_total_samples = 1
    
    start_new_round()
    return jsonify({"status": "received"})

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
    # Serialize encrypted model for sending
    serialized_model = []
    if public_key and global_model:
        for val in global_model:
            serialized_model.append((str(val.ciphertext()), val.exponent))
            
    return jsonify({
        "round": global_round,
        "weights": serialized_model,
        "total_samples": global_total_samples,
        "selected_clients": selected_clients,
        "public_key": {"n": str(public_key.n)} if public_key else None
    })

@app.route("/send_update", methods=["POST"])
def receive_update():
    global global_model, global_round, client_updates, global_total_samples

    data = request.json
    client_id = data.get("client_id")
    local_weights_serialized = data.get("weights")
    num_samples = data.get("num_samples")
    
    print(f"Received update from {client_id}")

    # Deserialize weights
    local_weights = []
    if local_weights_serialized and public_key:
        for (ctxt, exp) in local_weights_serialized:
            local_weights.append(phe.paillier.EncryptedNumber(public_key, int(ctxt), int(exp)))

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
                if weighted_sum == 0:
                    weighted_sum = client_w * client_n
                else:
                    weighted_sum += client_w * client_n
            
            # Division by total_samples (scalar)
            # MOVED TO CLIENT: We only store the weighted sum here to avoid float precision issues in HE
            new_weights.append(weighted_sum)

        global_model = new_weights
        global_total_samples = total_samples
        print(f"Round {global_round} complete. Aggregated Encrypted Model (Sum). Total Samples: {total_samples}")
        
        # Start Next Round
        start_new_round()
        
        return jsonify({"status": "aggregated", "new_round": global_round})
    
    else:
        return jsonify({"status": "waiting_for_others"})

if __name__ == "__main__":
    # Threaded=True prevents the server from blocking if multiple requests hit at once
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)