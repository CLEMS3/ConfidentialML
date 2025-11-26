import os
import time
import random
import requests
import socket
import threading
import json
from flask import Flask, request, jsonify
import phe.paillier

# --- Configuration ---
SERVER_ADDR = os.environ.get("SERVER_ADDRESS", "http://server:8080")
CLIENT_ID = socket.gethostname()
CLIENT_PORT = 5000

# --- Global State ---
public_key = None
private_key = None
keys_received_event = threading.Event()

# --- Background P2P Server ---
app = Flask(__name__)

@app.route("/trigger_keygen", methods=["POST"])
def trigger_keygen():
    global public_key, private_key
    print("!!! I have been selected as the LEADER for Key Generation !!!")
    
    peers = request.json.get("peers", [])
    
    # 1. Generate Keys
    print("Generating Paillier Keypair...")
    public_key, private_key = phe.paillier.generate_paillier_keypair()
    keys_received_event.set()
    
    # 2. Share Private Key with Peers (P2P)
    # We send n, p, q. (n is public, p and q are private factors)
    # Reconstructing private key needs public_key (n) and p, q.
    key_data = {
        "public_key": {"n": str(public_key.n)},
        "private_key": {"p": str(private_key.p), "q": str(private_key.q)}
    }
    
    for peer in peers:
        try:
            print(f"Sending keys to peer: {peer}")
            # Assuming peer ID is hostname
            requests.post(f"http://{peer}:{CLIENT_PORT}/receive_keys", json=key_data, timeout=5)
        except Exception as e:
            print(f"Failed to send keys to {peer}: {e}")
            
    # 3. Share Public Key with Server
    print("Sending Public Key to Server...")
    requests.post(f"{SERVER_ADDR}/public_key", json={"n": str(public_key.n)})
    
    return jsonify({"status": "keys_generated_and_shared"})

@app.route("/share_keys", methods=["POST"])
def share_keys():
    global public_key, private_key
    print("Received request to share keys with new peers.")
    
    peers = request.json.get("peers", [])
    
    # Wait for keys if they are currently being generated
    if not keys_received_event.is_set():
        print("Keys not yet available. Waiting for generation to complete...")
        if not keys_received_event.wait(timeout=15):
            print("Timeout waiting for keys.")
            return jsonify({"error": "Keys not generated in time"}), 503

    if not public_key or not private_key:
        return jsonify({"error": "No keys to share"}), 400

    key_data = {
        "public_key": {"n": str(public_key.n)},
        "private_key": {"p": str(private_key.p), "q": str(private_key.q)}
    }
    
    for peer in peers:
        try:
            print(f"Sending keys to peer: {peer}")
            requests.post(f"http://{peer}:{CLIENT_PORT}/receive_keys", json=key_data, timeout=5)
        except Exception as e:
            print(f"Failed to send keys to {peer}: {e}")
            
    return jsonify({"status": "keys_shared"})

@app.route("/receive_keys", methods=["POST"])
def receive_keys():
    global public_key, private_key
    print("Received keys from Leader.")
    
    data = request.json
    n = int(data["public_key"]["n"])
    p = int(data["private_key"]["p"])
    q = int(data["private_key"]["q"])
    
    public_key = phe.paillier.PaillierPublicKey(n)
    private_key = phe.paillier.PaillierPrivateKey(public_key, p, q)
    
    keys_received_event.set()
    
    return jsonify({"status": "keys_received"})

def run_background_server():
    app.run(host="0.0.0.0", port=CLIENT_PORT, debug=False, use_reloader=False)

# --- Main Client Logic ---
print(f"Client {CLIENT_ID} starting...")

# Start Background Server
server_thread = threading.Thread(target=run_background_server, daemon=True)
server_thread.start()

# 1. Register
while True:
    try:
        # Send our ID. Server uses this to contact us.
        resp = requests.post(f"{SERVER_ADDR}/register", json={"client_id": CLIENT_ID})
        if resp.status_code == 200:
            print("Successfully registered with server.")
            break
    except:
        print("Waiting for server to come online...")
        time.sleep(3)

last_processed_round = 0

while True:
    try:
        # Poll Server
        response = requests.get(f"{SERVER_ADDR}/get_model", timeout=5)
        data = response.json()
        
        server_round = data["round"]
        selected_clients = data["selected_clients"]
        
        # Check if we have keys yet. If not, we can't really do anything useful 
        # (unless we are just waiting to be triggered as leader).
        # But we should wait for keys before processing any training round.
        if not keys_received_event.is_set():
            # If round > 0, we should have keys. If round == 0, we might be waiting for trigger.
            if server_round > 0:
                print("Waiting for keys...")
            time.sleep(2)
            continue

        if server_round > last_processed_round:
            # A new round has started!
            
            if CLIENT_ID in selected_clients:
                print(f"\n[ROUND {server_round}] I was selected! Training...")
                
                # 1. Decrypt Global Model
                encrypted_weights_serialized = data["weights"]
                total_samples = data.get("total_samples", 1)
                
                # If it's the very first round, weights might be empty or encrypted zeros
                
                plaintext_weights = []
                if encrypted_weights_serialized:
                    for (ctxt, exp) in encrypted_weights_serialized:
                        enc_num = phe.paillier.EncryptedNumber(public_key, int(ctxt), int(exp))
                        # Decrypt Sum
                        decrypted_sum = private_key.decrypt(enc_num)
                        # Divide by Total Samples to get Average
                        plaintext_weights.append(decrypted_sum / total_samples)
                else:
                    # Fallback if empty (shouldn't happen if server inits correctly)
                    plaintext_weights = [0.0, 0.0, 0.0]
                
                print(f"Decrypted Global Model: {plaintext_weights}")

                # 2. Simulate Training
                time.sleep(random.uniform(0.5, 2.0))
                num_samples = random.randint(10, 100)
                # Simple training simulation: add random noise
                local_weights = [w + random.uniform(0.1, 0.5) for w in plaintext_weights]
                
                print(f"Trained Local Model: {local_weights}")

                # 3. Encrypt Local Update
                encrypted_local_weights = []
                for w in local_weights:
                    enc_w = public_key.encrypt(w)
                    encrypted_local_weights.append((str(enc_w.ciphertext()), enc_w.exponent))
                
                # 4. Send Update
                payload = {
                    "client_id": CLIENT_ID,
                    "weights": encrypted_local_weights,
                    "num_samples": num_samples
                }
                requests.post(f"{SERVER_ADDR}/send_update", json=payload)
                print(f"Update sent. Samples: {num_samples}")
                
                last_processed_round = server_round
            else:
                print(f"\n[ROUND {server_round}] I was NOT selected. Skipping.")
                last_processed_round = server_round
        
        elif server_round == 0:
            # Waiting for Round 1
            pass
            
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(5)
        
    time.sleep(2)