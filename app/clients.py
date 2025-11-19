import os
import time
import random
import requests
import socket

SERVER_ADDR = os.environ.get("SERVER_ADDRESS", "http://server:8080")
CLIENT_ID = socket.gethostname() 

print(f"Client {CLIENT_ID} starting...")

# 1. Register
while True:
    try:
        resp = requests.post(f"{SERVER_ADDR}/register", json={"client_id": CLIENT_ID})
        if resp.status_code == 200:
            print("Successfully registered with server.")
            break
    except:
        print("Waiting for server to come online...")
        time.sleep(3)

last_processed_round = 0 # Start at 0 because Server starts at 0

while True:
    try:
        # Poll Server
        response = requests.get(f"{SERVER_ADDR}/get_model", timeout=5)
        data = response.json()
        
        server_round = data["round"]
        selected_clients = data["selected_clients"]

        # LOGIC DEBUGGING
        # Only print status if things change to avoid spamming logs, 
        # OR if we are waiting for a round to start.
        
        if server_round > last_processed_round:
            # A new round has started!
            
            if CLIENT_ID in selected_clients:
                print(f"\n[ROUND {server_round}] I was selected! Training...")
                
                # Simulate Training
                time.sleep(random.uniform(0.5, 2.0))
                num_samples = random.randint(10, 100)
                local_weights = [w + random.uniform(0.1, 0.5) for w in data["weights"]]
                
                # Send Update
                payload = {
                    "client_id": CLIENT_ID,
                    "weights": local_weights,
                    "num_samples": num_samples
                }
                requests.post(f"{SERVER_ADDR}/send_update", json=payload)
                print(f"Update sent. Samples: {num_samples}")
                
                last_processed_round = server_round
            else:
                print(f"\n[ROUND {server_round}] I was NOT selected. Skipping.")
                # Mark as processed so we don't check this round again
                last_processed_round = server_round
        
        elif server_round == 0:
            print("Waiting for Round 1 to start (Waiting for more clients)...")
            time.sleep(3)
            
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(5)
        
    time.sleep(2)