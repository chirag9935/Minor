"""
Alternating Optimization (AO) driver: alternates BS active beamforming (W)
and RIS passive phase shifts (theta), each half-step fixing the other
variable, as proposed in Wu & Zhang [1] (docs/REFERENCES.md).

    repeat:
        fix theta -> update W   (mrt for K=1, zf/rzf for K>1)
        fix W     -> update theta (closed form for K=1, manifold ascent for K>1)
    until sum-rate improvement < tol or max_iter reached

Each half-step is only accepted if it does not decrease the sum rate (a
standard safeguarded-AO trick), so the returned history is non-decreasing.
"""
from __future__ import annotations

import numpy as np

from beamforming import mrt, zf, rzf
from ris_opt import phase_update_single_user, phase_update_manifold
from metrics import effective_channel, sum_rate


def run_ao(H: np.ndarray, G: np.ndarray, P: float, sigma2: float, mode: str,
           theta0: np.ndarray | None = None, max_iter: int = 20, tol: float = 1e-4,
           manifold_iters: int = 50) -> tuple:
    """mode in {"mrt" (K=1), "zf", "rzf"} (K>1). manifold_iters caps the inner
    Riemannian-ascent budget per outer AO iteration (unused for mode="mrt",
    which has a closed-form theta update); lowering it trades a little
    per-iteration accuracy for speed in large Monte-Carlo sweeps, since the
    outer AO loop revisits theta on every iteration anyway. Returns
    (theta, W, rate_history)."""
    M, Nt = H.shape
    if theta0 is None:
        theta = np.ones(M, dtype=complex)
    else:
        theta0 = np.asarray(theta0, dtype=complex)
        theta = theta0 / np.abs(theta0)

    def make_W(th):
        H_eff = effective_channel(G, H, th)
        if mode == "mrt":
            W = mrt(H_eff[0], P)
        elif mode == "zf":
            W = zf(H_eff, P)
        elif mode == "rzf":
            W = rzf(H_eff, P, sigma2)
        else:
            raise ValueError(f"unknown AO mode: {mode}")
        return W, H_eff

    W = None
    best_rate = -np.inf
    history = []

    for _ in range(max_iter):
        # ---- fix Phi, update W ----
        W_cand, H_eff_cand = make_W(theta)
        rate_cand = sum_rate(H_eff_cand, W_cand, sigma2)
        if rate_cand >= best_rate:
            W, best_rate = W_cand, rate_cand
        history.append(best_rate)

        # ---- fix W, update theta ----
        if mode == "mrt":
            theta_cand = phase_update_single_user(G[:, 0], H, W[:, 0])
        else:
            theta_cand, _ = phase_update_manifold(G, H, W, theta, sigma2, iters=manifold_iters)
        H_eff_cand = effective_channel(G, H, theta_cand)
        rate_cand = sum_rate(H_eff_cand, W, sigma2)
        if rate_cand >= best_rate:
            theta, best_rate = theta_cand, rate_cand
        history.append(best_rate)

        if len(history) >= 4 and abs(history[-1] - history[-3]) < tol:
            break

    return theta, W, history
