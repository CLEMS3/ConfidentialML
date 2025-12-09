import os
import time
import random
import requests
import socket
import threading
import json
from flask import Flask, request, jsonify
import phe.paillier
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.datasets import make_classification

SERVER_ADDR = os.environ.get("SERVER_ADDRESS", "http://server:8080")
CLIENT_ID = socket.gethostname()
CLIENT_PORT = 5000

public_key = None
private_key = None
keys_received_event = threading.Event()

DP_ENABLED = os.environ.get("DP_ENABLED", "1") not in ("0", "false", "False")
CLIP_NORM = float(os.environ.get("CLIP_NORM", "1.0"))
NOISE_MULTIPLIER = float(os.environ.get("NOISE_MULTIPLIER", "1.0"))
DP_EPS = float(os.environ.get("DP_EPS", "5.0"))
DP_DELTA = float(os.environ.get("DP_DELTA", "1e-5"))

_rng = np.random.default_rng()

TARGET_TOTAL_EPS = float(os.environ.get("TARGET_TOTAL_EPS", "10.0"))
REPORT_DELTA = float(os.environ.get("REPORT_DELTA", DP_DELTA))
RDP_ORDERS = [1.25,1.5,2,2.5,3,5,10,20,50,100]
rounds_done = 0

X_train = None
y_train = None
X_test = None
y_test = None
NUM_FEATURES = 30

app = Flask(__name__)

@app.route("/trigger_keygen", methods=["POST"])
def trigger_keygen():
    global public_key, private_key
    print("!!! I have been selected as the LEADER for Key Generation !!!")
    
    peers = request.json.get("peers", [])
    
    # step 1: generating the keys
    print("Generating Paillier Keypair...")
    public_key, private_key = phe.paillier.generate_paillier_keypair(n_length=2048)
    keys_received_event.set()
    
    # step 2: sharing the private key with peers
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
            
    # step 3: sharing the public key with the server
    print("Sending Public Key to Server...")
    requests.post(f"{SERVER_ADDR}/public_key", json={"n": str(public_key.n)})
    
    return jsonify({"status": "keys_generated_and_shared"})

@app.route("/share_keys", methods=["POST"])
def share_keys():
    global public_key, private_key
    print("Received request to share keys with new peers.")
    
    peers = request.json.get("peers", [])

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
        # here we create a synthetic dataset that is balanced for the training and testing
        if dataset_name == "synthetic":
            print("Generating synthetic data...")
            X, y = make_classification(
                n_samples=10000, 
                n_features=30, 
                n_informative=20, 
                n_redundant=5, 
                n_classes=2, 
                random_state=42
            )
            X = StandardScaler().fit_transform(X)
            X = np.c_[np.ones((X.shape[0], 1)), X]
            
        elif dataset_name == "creditcard":
            print("Reading creditcard.csv...")
            df = pd.read_csv("creditcard.csv")
            
            df = df.drop(['Time'], axis=1)
            df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
            
            X = df.drop(['Class'], axis=1).values
            y = df['Class'].values
            
            X = np.c_[np.ones((X.shape[0], 1)), X]
            
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # step 4. Split for this client
        n_samples = min(5000, len(X))
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_subset = X[indices]
        y_subset = y[indices]
        
        # step 5. Split into Train (80%) and Test (20%)
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

# Main Client Logic
print(f"Client {CLIENT_ID} starting...")

server_thread = threading.Thread(target=run_background_server, daemon=True)
server_thread.start()

def gaussian_sigma_for_eps_delta(C: float, eps: float, delta: float) -> float:
    """
    Standard sufficient sigma for (eps, delta)-DP with Gaussian mechanism for L2 sensitivity C.
    sigma = C * sqrt(2 ln(1.25/delta)) / eps
    """
    if eps <= 0 or delta <= 0 or delta >= 1:
        raise ValueError("eps>0 and 0<delta<1 required")
    return C * math.sqrt(2 * math.log(1.25 / delta)) / eps

def l2_clip_vector(vec: np.ndarray, C: float):
    norm = np.linalg.norm(vec)
    if norm == 0.0 or norm <= C:
        return vec.copy(), norm, 1.0
    factor = C / (norm + 1e-12)
    return vec * factor, norm, factor

def add_gaussian_noise_vec(vec: np.ndarray, sigma: float, rng: np.random.Generator):
    noise = rng.normal(loc=0.0, scale=sigma, size=vec.shape)
    return vec + noise

def compute_rdp_gaussian(sigma: float, C: float, orders):
    # Mironov formula for Gaussian mechanism RDP (per-round)
    if sigma <= 0:
        return [float('inf')] * len(orders)
    return [(alpha * (C ** 2)) / (2.0 * (sigma ** 2)) for alpha in orders]

def get_eps_from_rdp(orders, rdp, delta):
    # convert RDP -> (eps, delta). returns best epsilon over orders
    if delta <= 0:
        raise ValueError("delta must be > 0")
    log1_delta = math.log(1.0 / delta)
    eps_candidates = []
    for alpha, eps_alpha in zip(orders, rdp):
        if alpha <= 1: 
            continue
        eps = eps_alpha + log1_delta / (alpha - 1.0)
        eps_candidates.append(eps)
    return min(eps_candidates) if eps_candidates else float('inf')

def compose_rdp_and_get_eps(sigma: float, C: float, orders, delta, rounds):
    per_round_rdp = compute_rdp_gaussian(sigma, C, orders)
    total_rdp = [r * rounds for r in per_round_rdp]
    return get_eps_from_rdp(orders, total_rdp, delta)


per_round_sigma = None
if DP_ENABLED:
    try:
        per_round_sigma = gaussian_sigma_for_eps_delta(CLIP_NORM, DP_EPS, DP_DELTA) * NOISE_MULTIPLIER
        print(f"Per-round noise sigma computed: {per_round_sigma:.6f}")
    except Exception as e:
        print("Error computing sigma:", e)
        per_round_sigma = None

load_data()

while True:
    try:
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
        response = requests.get(f"{SERVER_ADDR}/get_model", timeout=300)
        data = response.json()
        
        if data.get("complete"):
            print("\n=== TRAINING COMPLETE. STARTING EVALUATION ===")
            
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
        
        if not keys_received_event.is_set():
            if server_round > 0:
                print("Waiting for keys...")
            time.sleep(2)
            continue

        if server_round > last_processed_round:
            if CLIENT_ID in selected_clients:
                print(f"\n[ROUND {server_round}] I was selected! Training...")
                encrypted_weights_serialized = data["weights"]
                total_samples = data.get("total_samples", 1)
                
                plaintext_weights = []
                if encrypted_weights_serialized:
                    for (ctxt, exp) in encrypted_weights_serialized:
                        enc_num = phe.paillier.EncryptedNumber(public_key, int(ctxt), int(exp))
                        decrypted_sum = private_key.decrypt(enc_num)
                        plaintext_weights.append(decrypted_sum / total_samples)
                else:
                    plaintext_weights = [0.0] * NUM_FEATURES
                
                print(f"Decrypted Global Model (First 5): {plaintext_weights[:5]}")

                print("Training on local data...")
                
                if len(plaintext_weights) != NUM_FEATURES:
                    print(f"WARNING: Weight shape mismatch. Server: {len(plaintext_weights)}, Local: {NUM_FEATURES}")
                    if len(plaintext_weights) < NUM_FEATURES:
                        plaintext_weights.extend([0.0] * (NUM_FEATURES - len(plaintext_weights)))
                    else:
                        plaintext_weights = plaintext_weights[:NUM_FEATURES]


                local_weights = train_model(plaintext_weights, X_train, y_train, epochs=10, learning_rate=1.0)
                
                print(f"Trained Local Model (First 5): {local_weights[:5]}")

                plaintext_weights_np = np.array(plaintext_weights, dtype=np.float64)
                local_weights_np = np.array(local_weights, dtype=np.float64)

                update = local_weights_np - plaintext_weights_np

                if DP_ENABLED:
                    clipped_update, orig_norm, factor = l2_clip_vector(update, CLIP_NORM)
                    sigma = per_round_sigma if per_round_sigma is not None else gaussian_sigma_for_eps_delta(CLIP_NORM, DP_EPS, DP_DELTA) * NOISE_MULTIPLIER
                    noised_update = add_gaussian_noise_vec(clipped_update, sigma, _rng)
                    noised_local_np = plaintext_weights_np + noised_update
                    noised_local = noised_local_np.tolist()
                    print(f"DP: orig_norm={orig_norm:.4f}, clipped_factor={factor:.6f}, per-round_sigma={sigma:.6f}")
                else:
                    noised_local = local_weights



                encrypted_local_weights = []
                for w in noised_local:
                    enc_w = public_key.encrypt(w)
                    encrypted_local_weights.append((str(enc_w.ciphertext()), enc_w.exponent))

                payload = {
                    "client_id": CLIENT_ID,
                    "weights": encrypted_local_weights,
                    "num_samples": len(X_train)
                }
                requests.post(f"{SERVER_ADDR}/send_update", json=payload, timeout=300)
                print(f"Update sent. Samples: {len(X_train)}")
                
                rounds_done += 1

                try:
                    composed_eps = compose_rdp_and_get_eps(per_round_sigma, CLIP_NORM, RDP_ORDERS, REPORT_DELTA, rounds_done)
                    print(f"Rounds so far: {rounds_done}. Composed epsilon (delta={REPORT_DELTA}): eps ≈ {composed_eps:.4f}")
                except Exception as e:
                    print("RDP accounting error:", e)

                if composed_eps > TARGET_TOTAL_EPS:
                    print(f"Privacy budget exceeded: composed_eps {composed_eps:.4f} > TARGET_TOTAL_EPS {TARGET_TOTAL_EPS}")
                    import sys
                    sys.exit(0)


                last_processed_round = server_round
            else:
                print(f"\n[ROUND {server_round}] I was NOT selected. Skipping.")
                last_processed_round = server_round
        
        elif server_round == 0:
            pass
            
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(5)
        
    time.sleep(2)