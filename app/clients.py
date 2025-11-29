import os
import time
import random
import requests
import socket
import threading
import json
from flask import Flask, request, jsonify
import phe.paillier
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.datasets import make_classification

# --- Configuration ---
SERVER_ADDR = os.environ.get("SERVER_ADDRESS", "http://server:8080")
CLIENT_ID = socket.gethostname()
CLIENT_PORT = 5000

# --- Global State ---
public_key = None
private_key = None
keys_received_event = threading.Event()

# --- ML State ---
X_train = None
y_train = None
X_test = None
y_test = None
NUM_FEATURES = 30 # V1-V28 + Amount + Bias (Time is dropped)

# --- Background P2P Server ---
app = Flask(__name__)

@app.route("/trigger_keygen", methods=["POST"])
def trigger_keygen():
    global public_key, private_key
    print("!!! I have been selected as the LEADER for Key Generation !!!")
    
    peers = request.json.get("peers", [])
    
    # 1. Generate Keys
    print("Generating Paillier Keypair...")
    public_key, private_key = phe.paillier.generate_paillier_keypair(n_length=1024)
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
            requests.post(f"http://{peer}:{CLIENT_PORT}/receive_keys", json=key_data, timeout=300)
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
            requests.post(f"http://{peer}:{CLIENT_PORT}/receive_keys", json=key_data, timeout=300)
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

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_model(weights, X, y, epochs=1, learning_rate=0.1):
    m = len(y)
    weights = np.array(weights)
    
    for _ in range(epochs):
        z = np.dot(X, weights)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / m
        weights -= learning_rate * gradient
        
    return weights.tolist()

def load_data():
    global X_train, y_train, X_test, y_test, NUM_FEATURES
    
    dataset_name = os.environ.get("DATASET_NAME", "creditcard")
    print(f"Loading dataset: {dataset_name}")
    
    try:
        if dataset_name == "synthetic":
            # Generate synthetic balanced data
            print("Generating synthetic data...")
            X, y = make_classification(
                n_samples=10000, 
                n_features=30, 
                n_informative=20, 
                n_redundant=5, 
                n_classes=2, 
                random_state=42
            )
            # Scale the features
            X = StandardScaler().fit_transform(X)
            
            # Add bias term
            X = np.c_[np.ones((X.shape[0], 1)), X]
            
        elif dataset_name == "creditcard":
            print("Reading creditcard.csv...")
            df = pd.read_csv("creditcard.csv")
            
            # Preprocessing
            # 1. Drop Time
            df = df.drop(['Time'], axis=1)
            
            # 2. Scale Amount
            df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
            
            # 3. Split Features and Target
            X = df.drop(['Class'], axis=1).values
            y = df['Class'].values
            
            # 4. Add Bias Term
            X = np.c_[np.ones((X.shape[0], 1)), X]
            
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # 5. Split for this client (Simulate FL by taking a random chunk)
        # We will take a random 5000 sample of the dataset for this client.
        # Ensure we don't sample more than available
        n_samples = min(5000, len(X))
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_subset = X[indices]
        y_subset = y[indices]
        
        # Split into Train (80%) and Test (20%)
        split_idx = int(0.8 * len(X_subset))
        X_train = X_subset[:split_idx]
        y_train = y_subset[:split_idx]
        X_test = X_subset[split_idx:]
        y_test = y_subset[split_idx:]
        
        NUM_FEATURES = X_train.shape[1]
        print(f"Data loaded. Train: {X_train.shape}, Test: {X_test.shape}. Features: {NUM_FEATURES}")
        
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        # Fallback to dummy data
        X_train = np.random.randn(100, 31)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 31)
        y_test = np.random.randint(0, 2, 20)
        NUM_FEATURES = 31

# --- Main Client Logic ---
print(f"Client {CLIENT_ID} starting...")

# Start Background Server
server_thread = threading.Thread(target=run_background_server, daemon=True)
server_thread.start()

# Load Data
load_data()

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
        response = requests.get(f"{SERVER_ADDR}/get_model", timeout=300)
        data = response.json()
        
        if data.get("complete"):
            print("\n=== TRAINING COMPLETE. STARTING EVALUATION ===")
            
            # Decrypt Final Model
            encrypted_weights_serialized = data["weights"]
            total_samples = data.get("total_samples", 1)
            
            final_weights = []
            if encrypted_weights_serialized:
                print("Decrypting final global model...")
                for (ctxt, exp) in encrypted_weights_serialized:
                    enc_num = phe.paillier.EncryptedNumber(public_key, int(ctxt), int(exp))
                    final_weights.append(private_key.decrypt(enc_num) / total_samples)
            else:
                final_weights = [0.0] * NUM_FEATURES

            # Evaluate on Test Set
            print(f"Evaluating on {len(X_test)} test samples...")
            z = np.dot(X_test, final_weights)
            predictions = sigmoid(z)
            y_pred = (predictions > 0.5).astype(int)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            
            print("\n" + "="*40)
            print(f"CLIENT {CLIENT_ID} FINAL RESULTS")
            print("="*40)
            print(f"Accuracy:  {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall:    {rec:.4f}")
            print(f"F1 Score:  {f1:.4f}")
            print("-" * 20)
            print(f"Confusion Matrix:\n{cm}")
            print("="*40 + "\n")
            
            # Save Model
            model_data = {
                "weights": final_weights,
                "metrics": {
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1
                }
            }
            with open("final_model.json", "w") as f:
                json.dump(model_data, f)
            print("Final model saved to final_model.json")
            
            break
        
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
                    plaintext_weights = [0.0] * NUM_FEATURES
                
                print(f"Decrypted Global Model (First 5): {plaintext_weights[:5]}")

                # 2. Real Training
                print("Training on local data...")
                
                # Ensure weights length matches features
                if len(plaintext_weights) != NUM_FEATURES:
                    print(f"WARNING: Weight shape mismatch. Server: {len(plaintext_weights)}, Local: {NUM_FEATURES}")
                    if len(plaintext_weights) < NUM_FEATURES:
                        plaintext_weights.extend([0.0] * (NUM_FEATURES - len(plaintext_weights)))
                    else:
                        plaintext_weights = plaintext_weights[:NUM_FEATURES]

                local_weights = train_model(plaintext_weights, X_train, y_train, epochs=10, learning_rate=1.0)
                
                print(f"Trained Local Model (First 5): {local_weights[:5]}")

                # 3. Encrypt Local Update
                encrypted_local_weights = []
                for w in local_weights:
                    enc_w = public_key.encrypt(w)
                    encrypted_local_weights.append((str(enc_w.ciphertext()), enc_w.exponent))
                
                # 4. Send Update
                payload = {
                    "client_id": CLIENT_ID,
                    "weights": encrypted_local_weights,
                    "num_samples": len(X_train)
                }
                requests.post(f"{SERVER_ADDR}/send_update", json=payload, timeout=300)
                print(f"Update sent. Samples: {len(X_train)}")
                
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