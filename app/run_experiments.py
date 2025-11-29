import subprocess
import re
import csv
import os
import time
import argparse
import sys

# Configuration
ROUNDS_TO_TEST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]
RUNS_PER_ROUND = 5
OUTPUT_FILE = "experiment_results.csv"
DOCKER_COMPOSE_CMD = ["docker-compose", "up", "--build", "--scale", "client=5"]
DOCKER_DOWN_CMD = ["docker-compose", "down"]

def parse_logs(logs):
    """
    Parses the docker-compose logs to extract metrics for each client.
    Returns a list of dictionaries containing metrics for each client.
    """
    client_metrics = {}
    
    # Regex patterns
    # client-1  | Accuracy:  0.7140
    acc_pattern = re.compile(r"client-\d+\s+\|\s+Accuracy:\s+(\d+\.\d+)")
    prec_pattern = re.compile(r"client-\d+\s+\|\s+Precision:\s+(\d+\.\d+)")
    rec_pattern = re.compile(r"client-\d+\s+\|\s+Recall:\s+(\d+\.\d+)")
    f1_pattern = re.compile(r"client-\d+\s+\|\s+F1 Score:\s+(\d+\.\d+)")
    
    # We need to associate metrics with specific clients, but the logs might be interleaved.
    # However, the user request asks for the AVERAGE between the 5 clients.
    # So we can just collect all values found and average them.
    
    accuracies = [float(x) for x in acc_pattern.findall(logs)]
    precisions = [float(x) for x in prec_pattern.findall(logs)]
    recalls = [float(x) for x in rec_pattern.findall(logs)]
    f1_scores = [float(x) for x in f1_pattern.findall(logs)]
    
    if not accuracies:
        return None

    return {
        "Accuracy": sum(accuracies) / len(accuracies),
        "Precision": sum(precisions) / len(precisions),
        "Recall": sum(recalls) / len(recalls),
        "F1 Score": sum(f1_scores) / len(f1_scores)
    }

def run_experiment(rounds, run_id):
    print(f"--- Running Experiment: Rounds={rounds}, Run={run_id} ---")
    
    # Set environment variable for this run
    env = os.environ.copy()
    env["MAX_ROUNDS"] = str(rounds)
    
    # Ensure clean state
    subprocess.run(DOCKER_DOWN_CMD, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Run Docker Compose
    # We need to capture output. Since the process runs until completion (server exits), 
    # we can just run it and capture stdout/stderr.
    # NOTE: The server/clients need to exit for this to finish. 
    # The server code exits when training is complete? 
    # Looking at server.py: 
    # if global_round > MAX_ROUNDS: ... training_complete = True ... return
    # But it doesn't stop the Flask server. The server keeps running.
    # The clients loop: while True. They break if data.get("complete") is True.
    # So clients will exit.
    # The server does NOT exit automatically in the current code.
    # We need to detect when clients have finished and then kill the process?
    # Or rely on the fact that clients exit.
    # If clients exit, the container stops.
    # But the server container will keep running.
    # So `docker-compose up` will NOT exit because server is still running.
    
    # We need to run docker-compose in detached mode or handle the process manually.
    # Better approach:
    # 1. Start docker-compose up -d
    # 2. Follow logs until we see "FINAL RESULTS" from all 5 clients or a timeout.
    # 3. Stop docker-compose.
    
    print("Starting Docker Compose...")
    subprocess.run(DOCKER_COMPOSE_CMD + ["-d"], env=env, check=True)
    
    print("Waiting for training to complete...")
    # Follow logs
    process = subprocess.Popen(["docker-compose", "logs", "-f"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', env=env)
    
    logs = []
    clients_finished = 0
    start_time = time.time()
    timeout = 600 # 10 minutes timeout
    
    try:
        while True:
            line = process.stdout.readline()
            if not line:
                break
            logs.append(line)
            # print(line, end='') # Optional: print logs to console
            
            if "FINAL RESULTS" in line:
                clients_finished += 1
                print(f"Client finished ({clients_finished}/5)")
            
            if clients_finished >= 5:
                print("All clients finished.")
                break
            
            if time.time() - start_time > timeout:
                print("Timeout reached!")
                break
    except KeyboardInterrupt:
        print("Interrupted by user.")
        
    process.terminate()
    
    # Get full logs one last time to be sure (or just use what we captured)
    # The captured logs should be sufficient.
    
    print("Stopping Docker Compose...")
    subprocess.run(DOCKER_DOWN_CMD, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    full_log_text = "".join(logs)
    metrics = parse_logs(full_log_text)
    
    if metrics:
        print(f"Results: {metrics}")
        return metrics
    else:
        print("Failed to parse metrics!")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run a quick test (1 round, 1 run)")
    args = parser.parse_args()
    
    rounds_list = [1] if args.test else ROUNDS_TO_TEST
    runs_per_round = 1 if args.test else RUNS_PER_ROUND
    
    # Initialize CSV
    file_exists = os.path.isfile(OUTPUT_FILE)
    with open(OUTPUT_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Rounds", "Run", "Accuracy", "Precision", "Recall", "F1 Score"])
        
    for r in rounds_list:
        for run in range(1, runs_per_round + 1):
            metrics = run_experiment(r, run)
            
            with open(OUTPUT_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                if metrics:
                    writer.writerow([r, run, metrics["Accuracy"], metrics["Precision"], metrics["Recall"], metrics["F1 Score"]])
                else:
                    writer.writerow([r, run, "N/A", "N/A", "N/A", "N/A"])
            
            # Short sleep between runs
            time.sleep(5)

if __name__ == "__main__":
    main()
