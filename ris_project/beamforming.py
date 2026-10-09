"""
BS active beamforming given the *effective* channel (after the RIS reflection
has been folded in).

System model (see ao.py / README for the full derivation):
    y_k = h_k^H W x + n_k,         h_k^H = g_k^H Phi H  (+ direct term in benchmarks)
    SINR_k = |h_k^H w_k|^2 / (sum_{j!=k} |h_k^H w_j|^2 + sigma^2)

Every precoder below returns W with shape (Nt, K) scaled so that
||W||_F^2 = P exactly (checked in tests).
"""
from __future__ import annotations

import numpy as np


def mrt(h_eff: np.ndarray, P: float) -> np.ndarray:
    """Maximum-ratio transmission for a single user.

    h_eff is h^H as a 1D array (the row produced by effective_channel's
    convention: h_eff[n] = conj(h)_n). By Cauchy-Schwarz, |h^H w| <= ||h|| *
    ||w|| with equality iff w is proportional to h itself (NOT to h^H) --
    so w = sqrt(P) * conj(h_eff) / ||h_eff||, shape (Nt, 1). This is the
    textbook matched-filter precoder, achieving |h^H w|^2 = P*||h||^2
    exactly (verified: w ∝ h_eff without the conjugate -- an earlier,
    incorrect version of this function -- under-achieves the optimal SINR
    by ~6-7 dB on average; see docs/ASSUMPTIONS.md, "Milestone 0").
    """
    h_eff = h_eff.reshape(-1, 1)
    w = np.conj(h_eff) / np.linalg.norm(h_eff)
    return np.sqrt(P) * w


def zf(H_eff: np.ndarray, P: float) -> np.ndarray:
    """Zero-forcing precoder for K>1 users.

    H_eff has shape (K, Nt) (row k = h_k^H). W = H_eff^H (H_eff H_eff^H)^-1,
    then rescaled so that ||W||_F^2 = P. Nulls all inter-user interference
    (requires Nt >= K).
    """
    W = H_eff.conj().T @ np.linalg.inv(H_eff @ H_eff.conj().T)
    scale = np.sqrt(P) / np.linalg.norm(W, "fro")
    return W * scale


def rzf(H_eff: np.ndarray, P: float, sigma2: float) -> np.ndarray:
    """Regularized zero-forcing (MMSE-type) precoder for K>1 users.

    W = H_eff^H (H_eff H_eff^H + (K*sigma2/P) I)^-1, rescaled so that
    ||W||_F^2 = P. Reduces to ZF as sigma2/P -> 0; trades off interference
    nulling against noise enhancement.
    """
    K = H_eff.shape[0]
    reg = (K * sigma2 / P) * np.eye(K)
    W = H_eff.conj().T @ np.linalg.inv(H_eff @ H_eff.conj().T + reg)
    scale = np.sqrt(P) / np.linalg.norm(W, "fro")
    return W * scale
