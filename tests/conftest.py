import os
import sys

import pytest

# Make the app modules (server.py, clients.py) importable as top-level modules.
APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")
sys.path.insert(0, os.path.abspath(APP_DIR))


@pytest.fixture(scope="session")
def keypair():
    """A single Paillier keypair reused across the suite (keygen is expensive).

    Matches the key size the app uses in clients.trigger_keygen.
    """
    import phe.paillier

    public_key, private_key = phe.paillier.generate_paillier_keypair(n_length=2048)
    return public_key, private_key
