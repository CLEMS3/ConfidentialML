# 🔐 ConfidentialML: Privacy-Preserving Federated Learning

A secure federated learning system that combines **Homomorphic Encryption** and **Differential Privacy** to enable collaborative model training without exposing sensitive data — even to the server.

This project uses Docker Compose to create a simple, local federated learning (FL) infrastructure. It starts one central server and `n` clients that can communicate with it.

---

## 🧠 Overview

Multiple clients collaboratively train a shared model under a strict **honest-but-curious threat model**: all parties follow the protocol, but any of them might try to infer sensitive information from the data they receive.

**The core idea:**
- Clients train locally on private data — raw data never leaves their machine.
- Model updates are encrypted using **Paillier Homomorphic Encryption** before being sent to the server.
- The server aggregates updates *without ever decrypting them*.
- **Differential Privacy** (Gaussian noise + L2 clipping) is applied client-side to further protect individual data contributions.

---

## 🎯 Problem Statement

> *How can several parties collaboratively train a model if they don't want to share their data with each other and can't fully trust the server?*

**Real-world examples:**
- A consortium of banks wants to train a fraud detection model, but data-sharing is prohibited by law.
- NATO allies want to train a joint intelligence model without exposing classified military data.

---

## 🔧 Technical Stack

| Component | Technology |
|---|---|
| Federated Learning | Cross-silo FedAvg |
| Homomorphic Encryption | Paillier Cryptosystem |
| Differential Privacy | Gaussian Mechanism + Rényi DP accounting |
| Server/Client Framework | Flask |
| Containerization | Docker + Docker Compose |

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────┐
│                  SERVER                        │
│  - Orchestrates rounds                         │
│  - Aggregates encrypted updates (never decrypts│
│  - Holds only the Paillier public key          │
└───────────┬────────────────────────────────────┘
            │  Encrypted model updates
   ┌─────────┼──────────┐
   ▼         ▼          ▼
Client 1  Client 2  Client N
  │           │          │
  ├─ Local training on private data
  ├─ L2 clipping + Gaussian noise (DP)
  └─ Paillier encryption → send to server
```

**Round lifecycle:**
1. Server registers clients and elects a **Leader** client.
2. Leader generates the Paillier keypair, shares the private key peer-to-peer with other clients, and sends only the public key to the server.
3. Server selects `k` clients per round.
4. Each selected client decrypts the global model, trains locally, applies DP, re-encrypts, and sends the update.
5. Server performs **Federated Averaging under encryption** and distributes the new global model.
6. Repeat for `MAX_ROUNDS`.

---

## 🔒 Security Model

| Threat | Mitigation |
|---|---|
| Curious server inspecting updates | Paillier HE — server sees only ciphertext |
| Curious client inferring others' data | Differential Privacy — noise masks individual contributions |
| Server learning private key | Key generated and distributed client-side (P2P); server never sees it |

---

## 📋 Prerequisites

* [Docker Desktop](https://www.docker.com/products/docker-desktop/) must be installed and running.

---

## 🚀 How to Run

First, go to the project directory:

```bash
cd app
```

This one command builds your Docker image from scratch and starts 1 server and 5 clients in the background.

```bash
docker-compose up -d --build --scale client=5
```

After that, the containers should run.

You can check what containers are running with:

```bash
docker-compose ps
```

To check that everything is running properly, or see if there is any problem, you can check the logs.

For the server:

```bash
docker-compose logs -f server
```

For the clients:

```bash
docker-compose logs -f client
```

When you are finished, this command stops and removes all containers and the network.

```bash
docker-compose down
```

### Configuration

Key environment variables can be adjusted in `docker-compose.yaml`:

| Variable | Default | Description |
|---|---|---|
| `MAX_ROUNDS` | 10 | Number of federated training rounds |
| `DP_EPS` | 2.0 | Differential privacy epsilon (ε) |
| `MIN_CLIENTS_TO_START` | 3 | Minimum clients before training begins |

---

## 📊 Results

Validated end-to-end on a credit card fraud detection dataset. Performance stabilized at the following optimal hyperparameters:

| Parameter | Optimal Value |
|---|---|
| Epsilon (ε) | 2 |
| Clipping Norm (C) | 10 |
| Noise Multiplier | 1.5 |

**Key finding:** Stronger privacy guarantees (lower ε) reduce model accuracy. This trade-off is inherent to differentially private training.

---
