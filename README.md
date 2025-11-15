# ConfidentialML: Client/Server Setup

This project uses Docker Compose to create a simple, local federated learning (FL) infrastructure. It starts one central server and 'n' clients that can communicate with it.

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

You can check what containers are running with
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

```Bash
docker-compose down
```
