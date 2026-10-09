"""
Phase 2, Milestone 5: imperfect CSI.

Standard additive-error model (assumption -- see docs/ASSUMPTIONS.md):
the BS knows noisy estimates Hhat = H + E_H, ghat = g + E_g, where each
error entry is i.i.d. complex Gaussian with variance
sigma_e^2 = epsilon * (average true-channel entry power), for a
normalised error level epsilon. AO is run using the ESTIMATES only; the
resulting (theta, W) is then evaluated against the TRUE channels to get
the actual achieved rate.
"""
from __future__ import annotations

import numpy as np


def add_csi_error(H: np.ndarray, G: np.ndarray, epsilon: float,
                   rng: np.random.Generator) -> tuple:
    """Returns (Hhat, Ghat) = (H, G) + independent complex Gaussian error,
    variance epsilon * mean(|entry|^2) (computed separately for H and G,
    since they generally have different power levels)."""
    if epsilon <= 0:
        return H.copy(), G.copy()
    var_H = epsilon * np.mean(np.abs(H) ** 2)
    var_G = epsilon * np.mean(np.abs(G) ** 2)
    E_H = (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape)) * np.sqrt(var_H / 2)
    E_G = (rng.standard_normal(G.shape) + 1j * rng.standard_normal(G.shape)) * np.sqrt(var_G / 2)
    return H + E_H, G + E_G


def rzf_robust(H_eff_hat: np.ndarray, P: float, sigma2: float, epsilon: float) -> np.ndarray:
    """"Robust-ish" RZF (Phase 2 spec): inflates the usual RZF regulariser
    by the estimated channel-error power, so the precoder trusts a noisier
    channel estimate less. Reduces to the ordinary rzf() at epsilon=0.
    reg = K*(sigma2 + epsilon*mean(|H_eff_hat|^2))/P * I -- a standard
    MMSE-under-estimation-error heuristic (the error acts like extra
    effective noise on the estimated channel)."""
    K = H_eff_hat.shape[0]
    error_power = epsilon * np.mean(np.abs(H_eff_hat) ** 2)
    reg = (K * (sigma2 + error_power) / P) * np.eye(K)
    W = H_eff_hat.conj().T @ np.linalg.inv(H_eff_hat @ H_eff_hat.conj().T + reg)
    scale = np.sqrt(P) / np.linalg.norm(W, "fro")
    return W * scale
