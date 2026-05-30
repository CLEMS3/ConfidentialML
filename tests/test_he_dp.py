"""Tests for the homomorphic-encryption + differential-privacy primitives.

These exercise the real functions in app/clients.py and app/server.py:
  - Paillier encrypt/decrypt (phe, as used inline by both modules)
  - clients.l2_clip_vector
  - clients.gaussian_sigma_for_eps_delta
  - server.perform_aggregation (encrypted FedAvg)
  - clients.compose_rdp_and_get_eps / compute_rdp_gaussian / get_eps_from_rdp
"""
import math

import numpy as np
import pytest

import clients
import server


# --------------------------------------------------------------------------
# Paillier round-trip
# --------------------------------------------------------------------------
@pytest.mark.parametrize("value", [0.0, 1.0, -3.5, 1234.5678, -0.000123, 42.0])
def test_paillier_roundtrip(keypair, value):
    public_key, private_key = keypair
    enc = public_key.encrypt(value)
    assert private_key.decrypt(enc) == pytest.approx(value, rel=1e-9, abs=1e-9)


# --------------------------------------------------------------------------
# Additive homomorphism: ciphertext+ciphertext, ciphertext+plaintext
# --------------------------------------------------------------------------
def test_additive_homomorphism_sum(keypair):
    public_key, private_key = keypair
    a, b = 3.25, -1.75

    # Encrypted + Encrypted
    enc_sum = public_key.encrypt(a) + public_key.encrypt(b)
    assert private_key.decrypt(enc_sum) == pytest.approx(a + b, rel=1e-9, abs=1e-9)

    # Encrypted + plaintext scalar
    enc_plus_scalar = public_key.encrypt(a) + b
    assert private_key.decrypt(enc_plus_scalar) == pytest.approx(a + b, rel=1e-9, abs=1e-9)


# --------------------------------------------------------------------------
# Multiplicative homomorphism by a plaintext scalar (used in FedAvg: w * n)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("scalar", [0, 1, 7, 100, -4])
def test_homomorphism_scalar_multiply(keypair, scalar):
    public_key, private_key = keypair
    a = 2.5
    enc_scaled = public_key.encrypt(a) * scalar
    assert private_key.decrypt(enc_scaled) == pytest.approx(a * scalar, rel=1e-9, abs=1e-9)


# --------------------------------------------------------------------------
# L2 clipping caps the norm at C
# --------------------------------------------------------------------------
def test_l2_clip_caps_norm_when_exceeding():
    C = 1.0
    vec = np.array([3.0, 4.0])  # norm = 5.0 > C
    clipped, orig_norm, factor = clients.l2_clip_vector(vec, C)

    assert orig_norm == pytest.approx(5.0)
    assert factor == pytest.approx(C / (5.0 + 1e-12))
    # Clipped norm must be at (or just under) C, never above it.
    assert np.linalg.norm(clipped) <= C + 1e-9
    assert np.linalg.norm(clipped) == pytest.approx(C, abs=1e-6)


def test_l2_clip_leaves_small_vectors_unchanged():
    C = 10.0
    vec = np.array([1.0, 2.0, 2.0])  # norm = 3.0 <= C
    clipped, orig_norm, factor = clients.l2_clip_vector(vec, C)

    assert factor == 1.0
    assert orig_norm == pytest.approx(3.0)
    assert np.allclose(clipped, vec)


# --------------------------------------------------------------------------
# Gaussian sigma formula
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "C,eps,delta",
    [(1.0, 5.0, 1e-5), (2.0, 1.0, 1e-6), (0.5, 10.0, 1e-4)],
)
def test_sigma_formula(C, eps, delta):
    expected = C * math.sqrt(2 * math.log(1.25 / delta)) / eps
    assert clients.gaussian_sigma_for_eps_delta(C, eps, delta) == pytest.approx(expected)


@pytest.mark.parametrize("eps,delta", [(0.0, 1e-5), (-1.0, 1e-5), (5.0, 0.0), (5.0, 1.0)])
def test_sigma_formula_rejects_bad_inputs(eps, delta):
    with pytest.raises(ValueError):
        clients.gaussian_sigma_for_eps_delta(1.0, eps, delta)


# --------------------------------------------------------------------------
# Encrypted FedAvg equals the plaintext weighted average
# --------------------------------------------------------------------------
def test_encrypted_fedavg_equals_plaintext_weighted_average(keypair):
    public_key, private_key = keypair

    # Two clients, NUM_FEATURES-length weight vectors and sample counts.
    w1 = [0.5, -1.0, 2.0, 0.0]
    w2 = [1.5, 1.0, -2.0, 4.0]
    n1, n2 = 120, 80

    enc_w1 = [public_key.encrypt(v) for v in w1]
    enc_w2 = [public_key.encrypt(v) for v in w2]

    # Drive the real server aggregation through its module globals.
    server.public_key = public_key
    server.client_updates = [(enc_w1, n1), (enc_w2, n2)]
    server.global_model = None
    server.global_round = 0
    server.global_total_samples = 1
    server.registered_clients = set()

    server.perform_aggregation()

    assert server.global_model is not None, "aggregation produced no model"
    total_samples = server.global_total_samples
    assert total_samples == n1 + n2

    # Server holds the encrypted *weighted sum*; the client divides by total
    # samples after decryption to recover the weighted average (FedAvg).
    decrypted_avg = [private_key.decrypt(layer) / total_samples for layer in server.global_model]
    expected_avg = [(a * n1 + b * n2) / (n1 + n2) for a, b in zip(w1, w2)]

    assert decrypted_avg == pytest.approx(expected_avg, rel=1e-9, abs=1e-9)


# --------------------------------------------------------------------------
# RDP composed epsilon increases with the number of rounds
# --------------------------------------------------------------------------
def test_rdp_composed_eps_increases_with_rounds():
    sigma = 1.0
    C = 1.0
    delta = 1e-5
    orders = clients.RDP_ORDERS

    rounds = [1, 2, 4, 8, 16]
    eps_values = [
        clients.compose_rdp_and_get_eps(sigma, C, orders, delta, r) for r in rounds
    ]

    # Strictly increasing as composition accumulates privacy loss.
    for earlier, later in zip(eps_values, eps_values[1:]):
        assert later > earlier

    # All finite and positive for a valid sigma.
    assert all(math.isfinite(e) and e > 0 for e in eps_values)


def test_compute_rdp_gaussian_infinite_for_zero_sigma():
    orders = clients.RDP_ORDERS
    rdp = clients.compute_rdp_gaussian(0.0, 1.0, orders)
    assert all(r == float("inf") for r in rdp)
