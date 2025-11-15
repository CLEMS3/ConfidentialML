import os
import time
import requests

# 1. Get the server address from the environment variable
#    This is set in your '3-client-deployment.yaml'
server_addr = os.environ.get("SERVER_ADDRESS")

if not server_addr:
    print("Error: SERVER_ADDRESS environment variable not set.")
    # Exit if the server address is not provided
    exit(1)

# Construct the full URL for the server's endpoint
server_hello_url = f"{server_addr}/hello"

print(f"Client started. Will contact server at: {server_hello_url}")

# Run in an infinite loop to periodically 'check in' with the server
while True:
    try:
        # 2. Make a request to the server
        response = requests.get(server_hello_url, timeout=5)
        response.raise_for_status()  # Raise an exception for bad status codes

        # 3. Process the server's response
        data = response.json()
        print(f"Successfully connected to server. Response: {data}")
        
        # In a real FL system, you would:
        # 1. Receive the model from 'data'
        # 2. Train the model on local data
        # 3. Send the updated weights back to the server (e.g., via a POST request)

    except requests.exceptions.ConnectionError:
        print("Server is not reachable. Retrying...")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
    
    # Wait for 5 seconds before contacting the server again
    print("Waiting 5 seconds before next check-in...\n")
    time.sleep(5)