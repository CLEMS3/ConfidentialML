import os
import time
import random
import requests
import socket # To get a unique ID for the container

# Configuration
SERVER_ADDR = os.environ.get("SERVER_ADDRESS", "http://server:8080")
CLIENT_ID = socket.gethostname() # Use the container ID as the client ID

print(f"Client {CLIENT_ID} starting...")

# State to track which round we have already processed
last_processed_round = -1

def train_local_model(global_weights):
    """
    Simulate training. 
    In reality, you would run TensorFlow/PyTorch here.
    For now, we just add a random number to the weights.
    """
    print(f"Training on data... (Simulated)")
    time.sleep(1) # Pretend to work
    
    # Create new weights by adding random noise to global weights
    # This simulates "learning" something unique to this client
    modification = random.uniform(0.1, 0.5)
    local_weights = [w + modification for w in global_weights]
    
    return local_weights

while True:
    try:
        # 1. Poll Server for the Global Model
        response = requests.get(f"{SERVER_ADDR}/get_model", timeout=5)
        data = response.json()
        
        server_round = data["round"]
        global_weights = data["weights"]

        # 2. Check if there is a new round available
        if server_round > last_processed_round:
            print(f"\n--- Starting Round {server_round} ---")
            print(f"Current Global Weights: {global_weights}")

            # 3. Train (Simulated)
            updated_weights = train_local_model(global_weights)
            print(f"Local Training Complete. New Weights: {updated_weights}")

            # 4. Send Update Back to Server
            payload = {
                "client_id": CLIENT_ID,
                "weights": updated_weights
            }
            requests.post(f"{SERVER_ADDR}/send_update", json=payload)
            print("Update sent to server.")

            # Mark this round as done locally
            last_processed_round = server_round
        
        else:
            # If the round hasn't changed yet, we wait
            print(f"Waiting for next round (Current: {server_round})...")

    except requests.exceptions.ConnectionError:
        print("Server unreachable. Retrying...")
    except Exception as e:
        print(f"Error: {e}")

    # Sleep to prevent spamming the server
    time.sleep(3)